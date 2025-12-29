import json
import re
import os
from typing import List, Tuple
from state import GovernanceState
from governance_data import get_governance_data
from cortex_connection import cortex
from cortex_utils import cortex_chat_text, parse_json_object

# Analyze user query to determine intent and identify relevant steps.

def analyze_query_node(state: GovernanceState) -> GovernanceState:
    user_message = state["user_message"]
    previous_state = state.get("previous_state")
    governance_data = get_governance_data()
    
    # Step 1: Classify intent using LLM
    intent, intent_confidence = _classify_intent(user_message, previous_state)
    state["intent"] = intent
    
    # Step 2: Identify steps based on intent
    if intent in ["ask_about_step", "ask_next_step", "ask_previous_step", 
                  "mark_complete", "mark_in_progress", "automation_request",
                  "clarification_needed"]: 
        
        focus_step_id, candidates, method, confidence = _identify_steps(
            user_message, 
            previous_state,
            intent
        )
        
        
        state["focus_step_id"] = focus_step_id
        state["candidate_step_ids"] = candidates
        state["match_method"] = method
        state["match_confidence"] = confidence
        
        # If multiple candidates found, this is disambiguation 
        if intent != "automation_request" and candidates and len(candidates) > 1:
            print(f"  Setting needs_disambiguation = True (found {len(candidates)} candidates)")
            state["needs_disambiguation"] = True
            # Override intent to make response generation clearer
            state["intent"] = "ask_about_step"
        else:
            state["needs_disambiguation"] = False
        
        # Step 3: Gather step details and edges
        if intent == "ask_next_step" and not focus_step_id and previous_state:
            # If the user asks "what's next?" without naming a step, use prior context.
            anchor = previous_state.get("anchor_step_id") or previous_state.get("focus_step_id")
            if anchor:
                outgoing = governance_data.get_outgoing_edges(anchor)
                next_step_ids = sorted(
                    set([e["to"] for e in outgoing]),
                    key=governance_data.step_index
                )
                state["next_step_ids"] = next_step_ids
                state["anchor_step_id"] = anchor

        if focus_step_id:
            step_details = governance_data.get_step_record(focus_step_id)
            state["step_details"] = step_details
            
            # Get next steps
            outgoing = governance_data.get_outgoing_edges(focus_step_id)
            next_step_ids = sorted(
                set([e["to"] for e in outgoing]),
                key=governance_data.step_index
            )
            state["next_step_ids"] = next_step_ids
            
            # Store anchor for navigation
            state["anchor_step_id"] = focus_step_id
            
            # Track automation capability
            if step_details and step_details.get("automatable"):
                state["automatable_step_id"] = focus_step_id
                state["automation_step"] = step_details.get("automation_step")
    
    elif intent == "greeting":
        state["needs_disambiguation"] = False
    
    return state

# Classify user intent using LLM

def _classify_intent(user_message: str, previous_state: dict = None) -> Tuple[str, float]:
    # Build context from previous state
    context = ""
    if previous_state:
        last_step = previous_state.get("focus_step_id")
        if last_step:
            context = f"\nPrevious context: User was discussing step {last_step}."
    
    prompt = f"""Classify the user's intent in this governance workflow conversation.

User message: "{user_message}"{context}

Available intents:
- greeting: General greeting or hello
- ask_about_step: Asking about a specific governance step
- ask_next_step: Asking what comes next / what to do next
- ask_previous_step: Asking what came before
- mark_complete: Marking a step as complete
- mark_in_progress: Marking a step as in progress
- automation_request: Requesting to run an automation (e.g., validate name)
- clarification_needed: Unclear or needs more info
- unknown: Cannot determine

Respond with ONLY a JSON object:
{{
  "intent": "one of the above intents",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}"""
    
    messages = [{"role": "user", "content": prompt}]
    response = cortex_chat_text(cortex.get_chat_response(
        messages, 
        max_tokens=200, 
        temperature=0.0,
        thinking_enabled=False
    ))
    
    try:
        data = parse_json_object(
            response,
            {
                "intent": str,
                "confidence": (int, float),
            },
        )
        if data:
            intent = data.get("intent", "unknown")
            confidence = float(data.get("confidence", 0.5))
            allowed = {
                "greeting",
                "ask_about_step",
                "ask_next_step",
                "ask_previous_step",
                "mark_complete",
                "mark_in_progress",
                "automation_request",
                "clarification_needed",
                "unknown",
            }
            if intent in allowed and 0.0 <= confidence <= 1.0:
                return intent, confidence
    except Exception:
        pass
    
    # Fallback to simple keyword matching
    msg_lower = user_message.lower()
    
    if any(w in msg_lower for w in ["hello", "hi", "hey", "good morning", "good afternoon"]):
        return "greeting", 0.8
    
    if any(w in msg_lower for w in ["run", "execute", "validate", "check", "perform"]):
        if any(w in msg_lower for w in ["name", "validation", "odp", "fdp", "cdp"]):
            return "automation_request", 0.85
    
    if any(w in msg_lower for w in ["next", "after", "then", "what do i do"]):
        return "ask_next_step", 0.75
    
    if any(w in msg_lower for w in ["what is", "tell me about", "explain", "describe"]):
        return "ask_about_step", 0.7
    
    return "ask_about_step", 0.6

# Identify step(s) mentioned in user message

def _identify_steps(
    user_message: str,
    previous_state: dict = None,
    intent: str = "ask_about_step"
) -> Tuple[str, List[str], str, float]:
    governance_data = get_governance_data()
    
    # Try deterministic matching first (ID, alias, substring)
    focus_id, method, confidence = governance_data.deterministic_match(user_message)
    
     
    if focus_id and confidence >= 0.85 and method != "embedding_match":
        # High confidence deterministic match (not embedding-based)
        return focus_id, [focus_id], method, confidence
    
    # Always get top semantic candidates
    candidates = governance_data.semantic_candidates(user_message, top_k=5)
    
    
    if len(candidates) == 0:
        return None, [], "no_match", 0.0
    
    # Check for disambiguation using original criteria:
    # Both candidates reasonably strong and fairly close in score
    if len(candidates) >= 2:
        c1, c2 = candidates[0], candidates[1]

        if os.getenv("GOV_DEBUG_MATCHING") == "1":
            print("\nDEBUG: semantic_candidates (top 5)")
            for c in candidates[:5]:
                print(
                    f"  {c['id']}: score={c.get('score'):.3f} "
                    f"(emb={c.get('embedding_score', 0.0):.3f}, lex={c.get('lexical_score', 0.0):.3f})"
                )

        # If the top hit has strong lexical evidence (rare/high-signal tokens from purpose/description),
        # prefer it even if embedding scores are close. This helps resolve "looks ambiguous but isn't".
        lex1 = float(c1.get("lexical_score", 0.0) or 0.0)
        lex2 = float(c2.get("lexical_score", 0.0) or 0.0)
        if lex1 >= 0.50 and (lex1 - lex2) >= 0.20 and c1["score"] >= 0.30:
            return c1["id"], [c1["id"]], "lexical_tiebreak", c1["score"]
        
        # Disambiguation logic:
        if (c1["score"] >= 0.40 and 
            c2["score"] >= 0.35 and 
            (c1["score"] - c2["score"]) <= 0.15):
            # Ambiguous - return top 3 for disambiguation
            candidate_ids = [c["id"] for c in candidates[:3]]
            return c1["id"], candidate_ids, "embedding_match_ambiguous", c1["score"]
        else:
            print(f"  → No disambiguation (criteria not met)")
    
    # Single best match (original: score >= 0.30)
    if candidates[0]["score"] >= 0.30:
        return candidates[0]["id"], [candidates[0]["id"]], "embedding_match", candidates[0]["score"]
    
    # No good match
    return None, [], "no_match", 0.0
