"""
Query analyzer node: Classifies intent and identifies relevant steps
"""
import json
import re
from typing import List, Tuple
from state import GovernanceState
from governance_data import governance_data
from cortex_connection import cortex


def analyze_query_node(state: GovernanceState) -> GovernanceState:
    """
    Analyze user query to determine intent and identify relevant steps.
    
    This node:
    1. Classifies user intent (greeting, ask_about_step, automation_request, etc.)
    2. Identifies step(s) mentioned in the query
    3. Determines if disambiguation is needed
    4. Gathers step details and edges
    
    Args:
        state: Current governance state
        
    Returns:
        Updated state with intent, focus_step_id, and candidate_step_ids
    """
    user_message = state["user_message"]
    previous_state = state.get("previous_state")
    
    # Step 1: Classify intent using LLM
    intent, intent_confidence = _classify_intent(user_message, previous_state)
    state["intent"] = intent
    
    # Step 2: Identify steps based on intent
    # Note: We run step identification for clarification_needed too, because the user
    # might be asking about steps in an ambiguous way (e.g., "security bits or cloud checks?")
    if intent in ["ask_about_step", "ask_next_step", "ask_previous_step", 
                  "mark_complete", "mark_in_progress", "automation_request",
                  "clarification_needed"]:  # Added clarification_needed!
        
        focus_step_id, candidates, method, confidence = _identify_steps(
            user_message, 
            previous_state,
            intent
        )
        
        print(f"\nDEBUG query_analyzer:")
        print(f"  Intent: {intent}")
        print(f"  Focus step: {focus_step_id}")
        print(f"  Candidates: {candidates}")
        print(f"  Match method: {method}")
        print(f"  Confidence: {confidence}")
        
        state["focus_step_id"] = focus_step_id
        state["candidate_step_ids"] = candidates
        state["match_method"] = method
        state["match_confidence"] = confidence
        
        # If we found multiple candidates, this is disambiguation - regardless of intent!
        if candidates and len(candidates) > 1:
            print(f"  Setting needs_disambiguation = True (found {len(candidates)} candidates)")
            state["needs_disambiguation"] = True
            # Override intent to make response generation clearer
            state["intent"] = "ask_about_step"
        else:
            state["needs_disambiguation"] = False
        
        # Step 3: Gather step details and edges
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
    
    elif intent == "ask_next_step" and previous_state:
        # Use anchor from previous state
        anchor = previous_state.get("anchor_step_id") or previous_state.get("focus_step_id")
        if anchor:
            outgoing = governance_data.get_outgoing_edges(anchor)
            next_step_ids = sorted(
                set([e["to"] for e in outgoing]),
                key=governance_data.step_index
            )
            state["next_step_ids"] = next_step_ids
            state["anchor_step_id"] = anchor
            
            # Focus on first next step
            if next_step_ids:
                state["focus_step_id"] = next_step_ids[0]
                state["step_details"] = governance_data.get_step_record(next_step_ids[0])
    
    elif intent == "greeting":
        state["needs_disambiguation"] = False
    
    return state


def _classify_intent(user_message: str, previous_state: dict = None) -> Tuple[str, float]:
    """
    Classify user intent using LLM
    
    Args:
        user_message: User's input
        previous_state: State from previous turn (for context)
        
    Returns:
        Tuple of (intent, confidence)
    """
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
    response = cortex.get_chat_response(
        messages, 
        max_tokens=200, 
        temperature=0.0,
        thinking_enabled=False
    )
    
    try:
        # Extract JSON from response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("intent", "unknown"), data.get("confidence", 0.5)
    except:
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


def _identify_steps(
    user_message: str,
    previous_state: dict = None,
    intent: str = "ask_about_step"
) -> Tuple[str, List[str], str, float]:
    """
    Identify step(s) mentioned in user message
    
    Args:
        user_message: User's input
        previous_state: State from previous turn
        intent: Classified intent
        
    Returns:
        Tuple of (focus_step_id, candidate_step_ids, match_method, confidence)
    """
    # Try deterministic matching first
    focus_id, method, confidence = governance_data.map_text_to_step_id(user_message)
    
    if focus_id and confidence >= 0.85:
        # High confidence single match
        return focus_id, [focus_id], method, confidence
    
    # For lower confidence or no match, try embedding-based ranking
    if confidence < 0.85:
        candidates = _find_candidate_steps(user_message, top_k=5)
        
        if len(candidates) == 0:
            return None, [], "no_match", 0.0
        
        elif len(candidates) == 1:
            return candidates[0]["id"], [candidates[0]["id"]], "embedding_match", candidates[0]["score"]
        
        else:
            # Multiple candidates - check if top candidate is significantly better
            top_score = candidates[0]["score"]
            second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
            
            if top_score > 0.8 and (top_score - second_score) > 0.15:
                # Clear winner
                return candidates[0]["id"], [candidates[0]["id"]], "embedding_match", top_score
            else:
                # Return top 3 for disambiguation
                candidate_ids = [c["id"] for c in candidates[:3]]
                return candidates[0]["id"], candidate_ids, "embedding_match_ambiguous", top_score
    
    return focus_id, [focus_id] if focus_id else [], method, confidence


def _find_candidate_steps(query: str, top_k: int = 5) -> List[dict]:
    """
    Find top-k candidate steps using embedding similarity
    
    Args:
        query: User's query
        top_k: Number of candidates to return
        
    Returns:
        List of dicts with 'id', 'score', and 'text'
    """
    query_emb = cortex.get_embedding(query)
    
    scores = []
    for step_id, data in governance_data.step_embeddings.items():
        sim = governance_data.cosine_similarity(query_emb, data["embedding"])
        scores.append({
            "id": step_id,
            "score": sim,
            "text": data["text"]
        })
    
    # Sort by score descending
    scores.sort(key=lambda x: x["score"], reverse=True)
    
    return scores[:top_k]