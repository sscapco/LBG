import os
import re
from typing import List, Tuple, Optional, Dict, Any

from cortex_connection import cortex

from .cortex_utils import cortex_chat_text, parse_json_object
from .governance_data import get_governance_data
from .prompts import classify_intent_prompt
from .state import GovernanceState


def analyze_query_node(state: GovernanceState) -> GovernanceState:
    user_message = state["user_message"]
    previous_state = state.get("previous_state")
    governance_data = get_governance_data()

    # Track any explicit step references mentioned by the user (e.g., "S2 vs S21").
    mentioned_ids = [m.upper() for m in re.findall(r"\bS\d{1,3}\b", user_message or "", flags=re.IGNORECASE)]
    if mentioned_ids:
        # Keep order but dedupe.
        seen = set()
        state["referenced_ids"] = [sid for sid in mentioned_ids if not (sid in seen or seen.add(sid))]

    # If the user is explicitly comparing two concepts (not necessarily step IDs),
    # try to resolve each side independently to a step.
    comparison = _extract_comparison_terms(user_message or "")
    if comparison and not state.get("referenced_ids"):
        a_term, b_term = comparison
        a_id = _map_term_to_step_id(a_term, governance_data)
        b_id = _map_term_to_step_id(b_term, governance_data)
        if a_id and b_id and a_id != b_id:
            state["referenced_ids"] = [a_id, b_id]

    intent, intent_confidence, intent_meta = _classify_intent(user_message, previous_state)
    state["intent"] = intent
    if intent_meta:
        state["query_analysis"] = intent_meta

        # If the classifier extracted a comparison pair, resolve each side to a step.
        comp = intent_meta.get("comparison")
        if comp and not state.get("referenced_ids"):
            a_term = _normalize_term(str(comp.get("a", "")))
            b_term = _normalize_term(str(comp.get("b", "")))
            if a_term and b_term:
                a_id = _map_term_to_step_id(a_term, governance_data)
                b_id = _map_term_to_step_id(b_term, governance_data)
                if a_id and b_id and a_id != b_id:
                    state["referenced_ids"] = [a_id, b_id]

    # Navigation intents: if the user asks "what comes next?" and we have prior context,
    # do not run semantic step matching (it will often look "ambiguous" across the whole workflow).
    if intent == "ask_next_step":
        anchor = None

        # Explicit "after S##" style reference wins.
        referenced_ids = state.get("referenced_ids") or []
        if referenced_ids:
            anchor = referenced_ids[0]

        # Otherwise use session context (last explained step preferred).
        if not anchor and previous_state:
            explained = previous_state.get("explained_step_ids") or []
            anchor = (
                previous_state.get("anchor_step_id")
                or previous_state.get("focus_step_id")
                or (explained[-1] if explained else None)
            )

        if anchor:
            outgoing = governance_data.get_outgoing_edges(anchor)
            next_step_ids = sorted(set([e["to"] for e in outgoing]), key=governance_data.step_index)
            state["next_step_ids"] = next_step_ids
            state["anchor_step_id"] = anchor
            state["needs_disambiguation"] = False
            return state

    if intent in [
        "compare_steps",
        "ask_about_step",
        "ask_next_step",
        "ask_previous_step",
        "mark_complete",
        "mark_in_progress",
        "automation_request",
        "clarification_needed",
    ]:
        # If this is a comparison and we already resolved two steps, skip single-step matching.
        if intent == "compare_steps" and (state.get("referenced_ids") and len(state["referenced_ids"]) >= 2):
            focus_step_id, candidates, method, confidence = None, [], "comparison", 1.0
        else:
            focus_step_id, candidates, method, confidence = _identify_steps(user_message, previous_state, intent)

        state["focus_step_id"] = focus_step_id
        state["candidate_step_ids"] = candidates
        state["match_method"] = method
        state["match_confidence"] = confidence

        if intent not in {"automation_request", "ask_next_step"} and candidates and len(candidates) > 1:
            state["needs_disambiguation"] = True
            state["intent"] = "ask_about_step"
        else:
            state["needs_disambiguation"] = False

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


def _classify_intent(user_message: str, previous_state: dict = None) -> Tuple[str, float, Optional[Dict[str, Any]]]:
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
                "compare_steps",
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
                # Keep extra fields (e.g., comparison extraction) for downstream logic.
                return intent, confidence, data
    except Exception:
        pass

    msg_lower = user_message.lower()

    if any(w in msg_lower for w in ["hello", "hi", "hey", "good morning", "good afternoon"]):
        return "greeting", 0.8, None

    # Comparison intent (fallback heuristic)
    if _extract_comparison_terms(user_message or ""):
        return "compare_steps", 0.75, None

    if any(w in msg_lower for w in ["run", "execute", "validate", "check", "perform"]):
        if any(w in msg_lower for w in ["name", "validation", "odp", "fdp", "cdp"]):
            return "automation_request", 0.85, None

    if any(w in msg_lower for w in ["next", "after", "then", "what do i do"]):
        return "ask_next_step", 0.75, None

    if any(w in msg_lower for w in ["what is", "tell me about", "explain", "describe"]):
        return "ask_about_step", 0.7, None
    
    return "ask_about_step", 0.6, None


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


def _extract_comparison_terms(message: str) -> Tuple[str, str] | None:
    m = (message or "").strip()
    if not m:
        return None

    # Common comparison phrasings.
    patterns = [
        # Accept minor variations/typos of "difference" by matching the stem.
        r"(?i)\b(?:what'?s\s+the\s+)?differenc\w*\s+between\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:\?|$)",
        r"(?i)\bcompare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:\?|$)",
        r"(?i)\b(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",
    ]
    for pat in patterns:
        match = re.search(pat, m)
        if not match:
            continue
        a, b = match.group(1), match.group(2)
        a = _normalize_term(a)
        b = _normalize_term(b)
        if a and b:
            return a, b
    return None


def _normalize_term(term: str) -> str:
    t = (term or "").strip().strip('"').strip("'").strip()
    # Remove generic suffixes like "step"/"steps".
    t = re.sub(r"(?i)\bsteps?\b", "", t).strip()
    # Remove trailing punctuation.
    t = t.strip(" .,:;—-")
    # Keep short phrases only (avoid whole paragraphs).
    if len(t) > 80:
        t = t[:80].rsplit(" ", 1)[0].strip()
    return t


def _map_term_to_step_id(term: str, governance_data) -> str | None:
    # Prefer deterministic match when the term is a shorthand/acronym (e.g., "dpwg", "doi").
    sid, _, conf = governance_data.deterministic_match(term)
    if sid and conf >= 0.85:
        return sid

    # Otherwise allow semantic match for the term.
    candidates = governance_data.semantic_candidates(term, top_k=1)
    if not candidates:
        return None
    best = candidates[0]
    # Conservative threshold so we don't force a wrong comparison.
    if float(best.get("score", 0.0)) >= 0.33:
        return best["id"]
    return None
