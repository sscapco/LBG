import json
import re
from typing import Optional, Tuple
from state import GovernanceState


# Import automation registry (will be copied from your code)
try:
    from automation_registry import (
        run_automation_step,
        format_automation_result_for_user,
        get_automation_info
    )
except ImportError:
    # Fallback stubs for testing
    def run_automation_step(automation_step: str, user_input: Optional[str] = None):
        return {"status": "error", "message": "Automation registry not found"}
    
    def format_automation_result_for_user(result: dict) -> str:
        return result.get("message", "Automation result")
    
    def get_automation_info(automation_step: str):
        return None

# Handle automation requests by detecting parameters and executing automation.

def automation_handler_node(state: GovernanceState) -> GovernanceState:
    
    intent = state.get("intent")
    user_message = state.get("user_message", "")
    
    # Only process if automation intent
    if intent != "automation_request":
        return state
    
    # Check if automation is available
    automation_step = state.get("automation_step")
    if not automation_step:
        # If the user asked for a known automation (e.g., name validation) without referencing a step,
        # infer the tool directly from the message.
        if detect_automation_request(user_message):
            automation_step = "validate_name"
            state["automation_step"] = automation_step
        else:
            state["answer"] = "I can run automations like name validation. Tell me which automation you want, and include the required parameters."
            return state
    
    # Verify automation exists in registry
    auto_info = get_automation_info(automation_step)
    if not auto_info:
        state["answer"] = f"Automation '{automation_step}' is not configured. Please contact support."
        return state
    
    # Detect and extract automation parameters
    params_json = _extract_automation_params(user_message, automation_step)
    
    if params_json is None:
        # Request parameters from user
        result = run_automation_step(automation_step, user_input=None)
        formatted = format_automation_result_for_user(result)
        state["answer"] = formatted
        state["automation_result"] = result
        return state
    
    # Execute automation
    try:
        result = run_automation_step(automation_step, user_input=params_json)
        formatted = format_automation_result_for_user(result)
        
        state["automation_result"] = result
        state["answer"] = formatted
        
    except Exception as e:
        state["answer"] = f"There was an error executing the automation: {str(e)}"
        state["automation_result"] = {"status": "error", "message": str(e)}
    
    return state

# Detect if user is requesting automation using Python logic.

def detect_automation_request(message: str) -> bool:
    msg_lower = message.lower()
    
    # Check for action words
    action_words = ["run", "execute", "validate", "check", "perform"]
    has_action = any(word in msg_lower for word in action_words)
    
    # Check for data product name pattern (AppID.BusinessName), where AppID is 2 letters + 5 digits
    # (e.g., AL18725.CustomerData, XY12345.CustomerData).
    has_name = bool(re.search(r'\b[a-z]{2}\d{5}\.[a-z0-9.]+\b', msg_lower))
    
    # Check for type keywords
    has_type = any(t in msg_lower for t in ["odp", "fdp", "cdp"])
    
    return has_action and has_name and has_type

# Extract parameters from user message based on automation type
def _extract_automation_params(message: str, automation_step: str) -> Optional[str]:
    
    if automation_step == "validate_name":
        return _extract_name_validation_params(message)
    
    
    return None

# Extract name validation parameters (name and type) from message

def _extract_name_validation_params(message: str) -> Optional[str]:    
    # Extract name (AppID.BusinessName pattern), where AppID is 2 letters + 5 digits.
    name_match = re.search(r'([A-Z]{2}\d{5}\.[A-Za-z0-9.]+)', message, re.IGNORECASE)
    if not name_match:
        return None
    
    name = name_match.group(1)
    # Normalize the name
    name = name.strip()
    
    # Extract type (ODP, FDP, or CDP)
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
    
    
    # Return as JSON string with proper escaping
    params = {
        "name": name,
        "type": type_val,
        "max_len": 75
    }
    
    json_str = json.dumps(params, ensure_ascii=False)

    return json_str
