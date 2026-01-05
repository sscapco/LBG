import os
import re
from typing import List, Tuple

from cortex_connection import cortex

from .cortex_utils import cortex_chat_text, parse_json_object
from .governance_data import get_governance_data
from .prompts import classify_intent_prompt
from .state import GovernanceState


def analyze_query_node(state: GovernanceState) -> GovernanceState:
    user_message = state["user_message"]
    previous_state = state.get("previous_state")
    governance_data = get_governance_data()

    intent, intent_confidence = _classify_intent(user_message, previous_state)
    state["intent"] = intent

    if intent in [
        "ask_about_step",
        "ask_next_step",
        "ask_previous_step",
        "mark_complete",
        "mark_in_progress",
        "automation_request",
        "clarification_needed",
    ]:
        focus_step_id, candidates, method, confidence = _identify_steps(user_message, previous_state, intent)

        state["focus_step_id"] = focus_step_id
        state["candidate_step_ids"] = candidates
        state["match_method"] = method
        state["match_confidence"] = confidence

        if intent != "automation_request" and candidates and len(candidates) > 1:
            state["needs_disambiguation"] = True
            state["intent"] = "ask_about_step"
        else:
            state["needs_disambiguation"] = False

        if intent == "ask_next_step" and not focus_step_id and previous_state:
            anchor = previous_state.get("anchor_step_id") or previous_state.get("focus_step_id")
            if anchor:
                outgoing = governance_data.get_outgoing_edges(anchor)
                next_step_ids = sorted(set([e["to"] for e in outgoing]), key=governance_data.step_index)
                state["next_step_ids"] = next_step_ids
                state["anchor_step_id"] = anchor

        if focus_step_id:
            step_details = governance_data.get_step_record(focus_step_id)
            state["step_details"] = step_details

            outgoing = governance_data.get_outgoing_edges(focus_step_id)
            next_step_ids = sorted(set([e["to"] for e in outgoing]), key=governance_data.step_index)
            state["next_step_ids"] = next_step_ids

            state["anchor_step_id"] = focus_step_id

            if step_details and step_details.get("automatable"):
                state["automatable_step_id"] = focus_step_id
                state["automation_step"] = step_details.get("automation_step")

    elif intent == "greeting":
        state["needs_disambiguation"] = False

    return state


def _classify_intent(user_message: str, previous_state: dict = None) -> Tuple[str, float]:
    previous_focus = None
    if previous_state:
        previous_focus = previous_state.get("focus_step_id")

    prompt = classify_intent_prompt(user_message, previous_focus_step_id=previous_focus)
    messages = [{"role": "user", "content": prompt}]
    response = cortex_chat_text(
        cortex.get_chat_response(
            messages,
            max_tokens=200,
            temperature=0.0,
            thinking_enabled=False,
        )
    )

    try:
        data = parse_json_object(response, {"intent": str, "confidence": (int, float)})
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


def _identify_steps(user_message: str, previous_state: dict = None, intent: str = "ask_about_step") -> Tuple[str, List[str], str, float]:
    governance_data = get_governance_data()

    focus_id, method, confidence = governance_data.deterministic_match(user_message)
    if focus_id and confidence >= 0.85 and method != "embedding_match":
        return focus_id, [focus_id], method, confidence

    candidates = governance_data.semantic_candidates(user_message, top_k=5)
    if len(candidates) == 0:
        return None, [], "no_match", 0.0

    if len(candidates) >= 2:
        c1, c2 = candidates[0], candidates[1]

        if os.getenv("GOV_DEBUG_MATCHING") == "1":
            print("\nDEBUG: semantic_candidates (top 5)")
            for c in candidates[:5]:
                print(
                    f"  {c['id']}: score={c.get('score'):.3f} "
                    f"(emb={c.get('embedding_score', 0.0):.3f}, lex={c.get('lexical_score', 0.0):.3f})"
                )

        lex1 = float(c1.get("lexical_score", 0.0) or 0.0)
        lex2 = float(c2.get("lexical_score", 0.0) or 0.0)
        if lex1 >= 0.50 and (lex1 - lex2) >= 0.20 and c1["score"] >= 0.30:
            return c1["id"], [c1["id"]], "lexical_tiebreak", c1["score"]

        if c1["score"] >= 0.40 and c2["score"] >= 0.35 and (c1["score"] - c2["score"]) <= 0.15:
            candidate_ids = [c["id"] for c in candidates[:3]]
            return c1["id"], candidate_ids, "embedding_match_ambiguous", c1["score"]

    if candidates[0]["score"] >= 0.30:
        return candidates[0]["id"], [candidates[0]["id"]], "embedding_match", candidates[0]["score"]

    return None, [], "no_match", 0.0
