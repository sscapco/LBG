import json
import re
from typing import Optional

from .state import GovernanceState

try:
    from .automation_registry import run_automation_step, format_automation_result_for_user, get_automation_info
except Exception:
    def run_automation_step(automation_step: str, user_input: Optional[str] = None):
        return {"status": "error", "message": "Automation registry not found"}

    def format_automation_result_for_user(result: dict) -> str:
        return result.get("message", "Automation result")

    def get_automation_info(automation_step: str):
        return None


def automation_handler_node(state: GovernanceState) -> GovernanceState:
    intent = state.get("intent")
    user_message = state.get("user_message", "")

    if intent != "automation_request":
        return state

    automation_step = state.get("automation_step")
    if not automation_step:
        if detect_automation_request(user_message):
            automation_step = "validate_name"
            state["automation_step"] = automation_step
        else:
            state["answer"] = "I can run automations like name validation. Tell me which automation you want, and include the required parameters."
            return state

    auto_info = get_automation_info(automation_step)
    if not auto_info:
        state["answer"] = f"Automation '{automation_step}' is not configured. Please contact support."
        return state

    params_json = _extract_automation_params(user_message, automation_step)

    if params_json is None:
        result = run_automation_step(automation_step, user_input=None)
        state["automation_result"] = result
        state["answer"] = format_automation_result_for_user(result)
        return state

    try:
        result = run_automation_step(automation_step, user_input=params_json)
        state["automation_result"] = result
        state["answer"] = format_automation_result_for_user(result)
    except Exception as e:
        state["answer"] = f"There was an error executing the automation: {str(e)}"
        state["automation_result"] = {"status": "error", "message": str(e)}

    return state


def detect_automation_request(message: str) -> bool:
    msg_lower = message.lower()

    action_words = ["run", "execute", "validate", "check", "perform"]
    has_action = any(word in msg_lower for word in action_words)

    has_name = bool(re.search(r"\b[a-z]{2}\d{5}\.[a-z0-9.]+\b", msg_lower))
    has_type = any(t in msg_lower for t in ["odp", "fdp", "cdp"])

    return has_action and has_name and has_type


def _extract_automation_params(message: str, automation_step: str) -> Optional[str]:
    if automation_step == "validate_name":
        return _extract_name_validation_params(message)
    return None


def _extract_name_validation_params(message: str) -> Optional[str]:
    name_match = re.search(r"([A-Z]{2}\d{5}\.[A-Za-z0-9.]+)", message, re.IGNORECASE)
    if not name_match:
        return None

    name = name_match.group(1).strip()

    msg_lower = message.lower()
    type_val = None
    if "odp" in msg_lower:
        type_val = "ODP"
    elif "fdp" in msg_lower:
        type_val = "FDP"
    elif "cdp" in msg_lower:
        type_val = "CDP"

    if not type_val:
        return None

    params = {"name": name, "type": type_val, "max_len": 75}
    return json.dumps(params, ensure_ascii=False)

