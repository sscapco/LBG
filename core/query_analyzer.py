import os
import re
from typing import List, Tuple, Optional, Dict, Any

from cortex_connection import cortex

from .cortex_utils import cortex_chat_text, parse_json_object
from .governance_data import get_governance_data
from .prompts import classify_intent_prompt, rerank_candidates_prompt
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

    # Very strict greeting check - must be primarily a greeting, not a question
    greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if any(msg_lower.strip().startswith(w) for w in greeting_words):
        # Only classify as greeting if it's short and doesn't contain question words
        if len(msg_lower.split()) <= 5 and not any(q in msg_lower for q in ["what", "where", "when", "how", "why", "which", "?"]):
            return "greeting", 0.8, None

    # Comparison intent (fallback heuristic)
    if _extract_comparison_terms(user_message or ""):
        return "compare_steps", 0.75, None

    # Explicit automation requests with action verbs + automation keywords
    if any(w in msg_lower for w in ["run", "execute", "validate", "check", "perform"]):
        if any(w in msg_lower for w in ["name", "validation", "naming"]):
            return "automation_request", 0.85, None

    # Next step indicators - but not if describing completed work
    if any(phrase in msg_lower for phrase in ["what's next", "what do i do next", "what comes next", "where do we go from here"]):
        return "ask_next_step", 0.75, None

    # If message contains "next" or "after" but also describes work done, it's asking about current step
    has_next_keywords = any(w in msg_lower for w in ["next", "after"])
    has_work_keywords = any(w in msg_lower for w in ["raised", "created", "completed", "delivered", "finished", "got", "obtained", "submitted"])

    if has_next_keywords and not has_work_keywords:
        return "ask_next_step", 0.70, None

    # Default to asking about a step - this is the most common intent
    return "ask_about_step", 0.65, None


def _identify_steps(user_message: str, previous_state: dict = None, intent: str = "ask_about_step") -> Tuple[str, List[str], str, float]:
    governance_data = get_governance_data()

    # Try deterministic match first - ID regex and aliases only (no substring)
    focus_id, method, confidence = governance_data.deterministic_match(user_message)
    # Only accept high-confidence deterministic matches (ID/alias, not substring)
    if focus_id and method in ["id_match", "alias_match"] and confidence >= 0.90:
        return focus_id, [focus_id], method, confidence

    # Fall back to semantic matching
    candidates = governance_data.semantic_candidates(user_message, top_k=5)
    if len(candidates) == 0:
        return None, [], "no_match", 0.0

    if os.getenv("GOV_DEBUG_MATCHING") == "1":
        print("\nDEBUG: semantic_candidates (top 5)")
        for c in candidates[:5]:
            print(
                f"  {c['id']}: score={c.get('score'):.3f} "
                f"(emb={c.get('embedding_score', 0.0):.3f}, lex={c.get('lexical_score', 0.0):.3f}, "
                f"name={c.get('name_similarity', 0.0):.3f}, desc={c.get('description_similarity', 0.0):.3f})"
            )

    # CRITICAL: Apply strict thresholds to filter out weak matches
    # embedding_score >= 0.5: ensures semantic relevance
    # lexical_score >= 0.3: ensures meaningful keyword overlap
    valid_candidates = [
        c for c in candidates
        if c.get("embedding_score", 0.0) >= 0.5 and c.get("lexical_score", 0.0) >= 0.3
    ]

    if not valid_candidates:
        # No candidates meet thresholds = likely out-of-scope or no relevant match
        return None, [], "below_threshold", 0.0

    # Check if we have a clear winner (high score, good gap to next candidate)
    c1 = valid_candidates[0]
    if len(valid_candidates) >= 2:
        c2 = valid_candidates[1]
        score_gap = c1["score"] - c2["score"]

        # Clear winner: high score AND significant gap
        if c1["score"] >= 0.60 and score_gap >= 0.15:
            return c1["id"], [c1["id"]], "high_confidence_match", c1["score"]

        # Close scores = ambiguous, need to ask user
        if c1["score"] >= 0.45 and c2["score"] >= 0.40 and score_gap < 0.10:
            candidate_ids = [c["id"] for c in valid_candidates[:3]]
            return c1["id"], candidate_ids, "threshold_ambiguous", c1["score"]
    else:
        # Only one valid candidate
        if c1["score"] >= 0.50:
            return c1["id"], [c1["id"]], "single_valid_match", c1["score"]

    # Borderline cases (score 0.35-0.60 with no clear winner): use LLM validation
    if 0.35 <= c1["score"] < 0.60:
        validated_ids, validation_result, llm_confidence = _llm_validate_scope(user_message, valid_candidates[:3])

        if validation_result == "out_of_scope":
            return None, [], "out_of_scope", llm_confidence
        elif validation_result == "no_match":
            return None, [], "no_match_validated", llm_confidence
        elif validation_result == "ambiguous" and validated_ids:
            return validated_ids[0], validated_ids, "llm_validated_ambiguous", llm_confidence
        elif validation_result == "valid" and validated_ids:
            return validated_ids[0], validated_ids, "llm_validated", llm_confidence

    # Very low scores even after threshold filtering = no match
    if c1["score"] < 0.35:
        return None, [], "low_confidence", 0.0

    # Fallback: return top candidate
    return c1["id"], [c1["id"]], "threshold_match", c1["score"]


def _llm_rerank_candidates(user_message: str, candidates: List[Dict[str, Any]]) -> Tuple[Optional[str], List[str]]:
    """Use LLM to intelligently rerank candidates by understanding context."""
    governance_data = get_governance_data()

    # Prepare candidate details for LLM
    candidate_details = []
    for c in candidates:
        step_record = governance_data.get_step_record(c["id"])
        if step_record:
            candidate_details.append({
                "id": step_record["id"],
                "name": step_record["name"],
                "purpose": step_record["purpose"],
                "description": step_record["description"][:500],  # Truncate long descriptions
                "initial_score": c["score"],
            })

    if not candidate_details:
        return None, []

    prompt = rerank_candidates_prompt(user_message, candidate_details)
    messages = [{"role": "user", "content": prompt}]

    try:
        response = cortex_chat_text(
            cortex.get_chat_response(
                messages,
                max_tokens=300,
                temperature=0.0,
                thinking_enabled=False,
            )
        )

        data = parse_json_object(response, {"best_match": str, "confidence": (int, float)})
        if data:
            best_match = data.get("best_match", "").strip().upper()
            confidence = float(data.get("confidence", 0.0))
            ambiguous = data.get("ambiguous_candidates")

            # Only trust high-confidence LLM decisions
            if confidence >= 0.65:
                if best_match == "AMBIGUOUS" and isinstance(ambiguous, list) and ambiguous:
                    return "AMBIGUOUS", [s.strip().upper() for s in ambiguous if s]
                elif best_match and best_match != "AMBIGUOUS":
                    # Verify it's one of the candidates
                    if any(c["id"] == best_match for c in candidate_details):
                        return best_match, []

    except Exception:
        pass

    # Fallback: return None to use original ranking
    return None, []


def _llm_validate_scope(user_message: str, candidates: List[Dict[str, Any]]) -> Tuple[List[str], str, float]:
    """
    Use LLM to validate if query is in-scope and select best matching step(s).

    Returns:
        (step_ids, validation_result, confidence)
        - step_ids: List of step IDs (empty if out of scope or no match)
        - validation_result: "valid", "ambiguous", "no_match", or "out_of_scope"
        - confidence: 0.0 to 1.0
    """
    governance_data = get_governance_data()

    # Prepare candidate details for LLM with full context
    candidate_details = []
    for c in candidates:
        step_record = governance_data.get_step_record(c["id"])
        if step_record:
            candidate_details.append({
                "id": step_record["id"],
                "name": step_record["name"],
                "purpose": step_record["purpose"],
                "description": step_record["description"][:600],  # Include more context
                "embedding_score": c.get("embedding_score", 0.0),
                "lexical_score": c.get("lexical_score", 0.0),
                "overall_score": c.get("score", 0.0),
            })

    if not candidate_details:
        return [], "no_match", 0.0

    from core.prompts import validate_scope_prompt
    prompt = validate_scope_prompt(user_message, candidate_details)
    messages = [{"role": "user", "content": prompt}]

    try:
        response = cortex_chat_text(
            cortex.get_chat_response(
                messages,
                max_tokens=400,
                temperature=0.0,
                thinking_enabled=False,
            )
        )

        data = parse_json_object(response, {"scope": str, "validation": str, "confidence": (int, float)})
        if data:
            scope = data.get("scope", "").upper()
            validation = data.get("validation", "").upper()
            confidence = float(data.get("confidence", 0.0))
            step_ids = data.get("step_ids") or []

            # Normalize step IDs
            step_ids = [str(sid).strip().upper() for sid in step_ids if sid]

            # Map LLM response to our validation result
            if scope == "OUT_OF_SCOPE":
                return [], "out_of_scope", confidence
            elif validation == "NO_MATCH":
                return [], "no_match", confidence
            elif validation == "AMBIGUOUS" and step_ids:
                return step_ids, "ambiguous", confidence
            elif validation == "VALID" and step_ids:
                # Verify step IDs are from candidates
                valid_ids = [sid for sid in step_ids if any(c["id"] == sid for c in candidate_details)]
                if valid_ids:
                    return valid_ids, "valid", confidence

    except Exception as e:
        # Log error but don't fail the pipeline
        if os.getenv("GOV_DEBUG_MATCHING") == "1":
            print(f"DEBUG: LLM validation failed: {e}")

    # Fallback: treat as no match if LLM fails
    return [], "no_match", 0.0


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
