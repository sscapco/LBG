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

    action_words = ["run", "execute", "validate", "check", "perform", "verify"]
    has_action = any(word in msg_lower for word in action_words)

    has_naming_intent = any(k in msg_lower for k in ["naming", "name check", "namechecker", "validate name"])

    # JSON payload support: {"name":"...","type":"ODP"} etc.
    has_json = bool(re.search(r"\{.*\}", msg_lower, flags=re.DOTALL))

    # Prefer strict AppID.BusinessName; fall back to any dot-separated token string if the message
    # is clearly about naming/validation.
    has_strict_name = bool(re.search(r"\b[a-z]{2}\d{5}\.[a-z0-9.]+\b", msg_lower))
    has_loose_name = bool(re.search(r"\b[a-z0-9]+(?:\.[a-z0-9]+)+\b", msg_lower))
    has_name = has_strict_name or (has_naming_intent and has_loose_name) or has_json
    has_type = any(t in msg_lower for t in ["odp", "fdp", "cdp"])

    return (has_action or has_naming_intent) and has_name


def _extract_automation_params(message: str, automation_step: str) -> Optional[str]:
    if automation_step == "validate_name":
        return _extract_name_validation_params(message)
    return None


def _extract_name_validation_params(message: str) -> Optional[str]:
    # 1) JSON payload
    match = re.search(r"\{.*\}", message or "", flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict) and ("name" in payload or "type" in payload):
                name = payload.get("name")
                dp_type = payload.get("type") or payload.get("dp_type")
                max_len = payload.get("max_len", 75)
                params = {}
                if name:
                    params["name"] = str(name).strip()
                if dp_type:
                    params["type"] = str(dp_type).strip().upper()
                params["max_len"] = int(max_len) if str(max_len).isdigit() else 75
                return json.dumps(params, ensure_ascii=False)
        except Exception:
            pass

    # 2) Strict AppID.BusinessName pattern (best signal)
    name_match = re.search(r"([A-Z]{2}\d{5}\.[A-Za-z0-9.]+)", message or "", re.IGNORECASE)
    name = name_match.group(1).strip() if name_match else None

    # 3) Loose dot-separated token string (only when message indicates naming)
    if not name:
        msg_lower = (message or "").lower()
        if any(k in msg_lower for k in ["naming", "name", "validate", "check"]):
            loose = re.search(r"\b([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+)\b", message or "")
            if loose:
                name = loose.group(1).strip()

    if not name:
        return None

    msg_lower = message.lower()
    type_val = None
    if "odp" in msg_lower:
        type_val = "ODP"
    elif "fdp" in msg_lower:
        type_val = "FDP"
    elif "cdp" in msg_lower:
        type_val = "CDP"

    # Allow missing dp_type; the tool will ask a follow-up instead of failing.
    params = {"name": name, "max_len": 75}
    if type_val:
        params["type"] = type_val
    return json.dumps(params, ensure_ascii=False)
