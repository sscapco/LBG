import json
from typing import Dict, List

from cortex_connection import cortex

from .cortex_utils import cortex_chat_text
from .governance_data import get_governance_data
from .prompts import compare_steps_prompt, disambiguation_prompt, fallback_prompt, next_step_prompt, step_response_prompt
from .state import GovernanceState


def response_generator_node(state: GovernanceState) -> GovernanceState:
    if state.get("answer"):
        return state

    referenced_ids = state.get("referenced_ids", []) or []
    if len(referenced_ids) >= 2:
        state["answer"] = _generate_comparison_response(state, referenced_ids[:2])
        state["explained_step_ids"] = referenced_ids[:2]
        return state

    intent = state.get("intent")
    needs_disambiguation = state.get("needs_disambiguation")
    focus_step_id = state.get("focus_step_id")

    if intent == "greeting":
        state["answer"] = _generate_greeting_response()
    elif intent == "automation_request":
        state["answer"] = "I can help with automation. Please specify which step and provide the required parameters."
    elif needs_disambiguation:
        state["answer"] = _generate_disambiguation_response(state)
    elif focus_step_id:
        state["answer"] = _generate_step_response(state)
        state["referenced_ids"] = [focus_step_id]
        state["explained_step_ids"] = [focus_step_id]
    elif intent == "ask_next_step":
        state["answer"] = _generate_next_step_response(state)
    else:
        state["answer"] = _generate_fallback_response(state)

    return state


def _generate_greeting_response() -> str:
    return (
        "Hello! I'm your governance workflow assistant. "
        "I can help you:\n"
        "• Understand specific governance steps\n"
        "• Navigate through the workflow\n"
        "• Run automations like name validation\n\n"
        "What would you like to know?"
    )


def _generate_disambiguation_response(state: GovernanceState) -> str:
    governance_data = get_governance_data()
    candidates = state.get("candidate_step_ids", [])
    user_message = state.get("user_message", "")

    if not candidates:
        return "I couldn't find a matching step. Could you provide more details?"

    candidate_details: List[Dict] = []
    for step_id in candidates[:3]:
        record = governance_data.get_step_record(step_id)
        if record:
            candidate_details.append(
                {
                    "id": record["id"],
                    "name": record["name"],
                    "purpose": record["purpose"],
                    "description": record["description"],
                }
            )

    prompt = disambiguation_prompt(user_message, candidate_details)
    messages = [{"role": "user", "content": prompt}]
    return cortex_chat_text(
        cortex.get_chat_response(
            messages,
            max_tokens=2000,
            temperature=0.0,
            thinking_enabled=False,
        )
    )


def _generate_step_response(state: GovernanceState) -> str:
    governance_data = get_governance_data()
    step_details = state.get("step_details")
    if not step_details:
        return "I couldn't find information about that step."

    next_steps = state.get("next_step_ids", [])
    automatable = state.get("automatable_step_id") is not None
    automation_step = state.get("automation_step")

    next_step_info = []
    for next_id in next_steps[:3]:
        next_rec = governance_data.get_step_record(next_id)
        if next_rec:
            next_step_info.append(next_rec)

    automation_info = ""
    if automatable and automation_step:
        from .automation_registry import get_automation_info

        auto_info = get_automation_info(automation_step)
        if auto_info:
            automation_info = f"""
🤖 **Automation Available**: {auto_info.get('display_name', automation_step)}
{auto_info.get('description', 'Automated helper for this step')}

To use it, simply ask me to run it. For example:
- "Run the {auto_info.get('display_name', 'automation').lower()}"
- "Can you validate this for me?"
"""

    prompt = step_response_prompt(
        state.get("user_message", ""),
        step_details,
        next_step_info,
        automation_info,
        state.get("intent"),
    )
    messages = [{"role": "user", "content": prompt}]
    return cortex_chat_text(
        cortex.get_chat_response(
            messages,
            max_tokens=1500,
            temperature=0.0,
            thinking_enabled=False,
        )
    )


def _generate_next_step_response(state: GovernanceState) -> str:
    governance_data = get_governance_data()
    next_steps = state.get("next_step_ids", [])
    anchor_id = state.get("anchor_step_id")

    if not next_steps:
        return "I don't see any defined next steps from here. You might want to review the workflow or ask about a specific step."

    next_step_details = []
    for step_id in next_steps:
        record = governance_data.get_step_record(step_id)
        if record:
            next_step_details.append(record)

    anchor_record = governance_data.get_step_record(anchor_id) if anchor_id else None

    prompt = next_step_prompt(state.get("user_message", ""), next_step_details, anchor_record)
    messages = [{"role": "user", "content": prompt}]
    return cortex_chat_text(
        cortex.get_chat_response(
            messages,
            max_tokens=1000,
            temperature=0.0,
            thinking_enabled=False,
        )
    )


def _generate_fallback_response(state: GovernanceState) -> str:
    prompt = fallback_prompt(state.get("user_message", ""))
    messages = [{"role": "user", "content": prompt}]
    return cortex_chat_text(
        cortex.get_chat_response(
            messages,
            max_tokens=300,
            temperature=0.3,
            thinking_enabled=False,
        )
    )


def _generate_comparison_response(state: GovernanceState, step_ids: List[str]) -> str:
    governance_data = get_governance_data()
    user_message = state.get("user_message", "")

    details: List[Dict] = []
    missing: List[str] = []
    for sid in step_ids:
        rec = governance_data.get_step_record(sid)
        if not rec:
            missing.append(sid)
            continue
        details.append(rec)

    if missing:
        known = ", ".join([d.get("id", "") for d in details if d.get("id")])
        missing_str = ", ".join(missing)
        if known:
            return f"I can compare {known}, but I couldn't find information for {missing_str}. Can you check the step ID(s)?"
        return f"I couldn't find information for {missing_str}. Can you check the step ID(s)?"

    prompt = compare_steps_prompt(user_message, details)
    messages = [{"role": "user", "content": prompt}]
    return cortex_chat_text(
        cortex.get_chat_response(
            messages,
            max_tokens=1200,
            temperature=0.0,
            thinking_enabled=False,
        )
    )
