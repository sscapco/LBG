"""
Automation handler node: Detects and executes automation requests
"""
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


def automation_handler_node(state: GovernanceState) -> GovernanceState:
    """
    Handle automation requests by detecting parameters and executing automation.
    
    This node:
    1. Checks if automation is requested (based on intent)
    2. Detects if automation is available for current step
    3. Extracts parameters from user message
    4. Executes automation and formats result
    
    Args:
        state: Current governance state
        
    Returns:
        Updated state with automation_result
    """
    intent = state.get("intent")
    user_message = state.get("user_message", "")
    
    # Only process if automation intent
    if intent != "automation_request":
        return state
    
    # Check if automation is available
    automation_step = state.get("automation_step")
    if not automation_step:
        state["answer"] = "I don't see an automation available for this step. Please specify which step you'd like to automate."
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


def detect_automation_request(message: str) -> bool:
    """
    Detect if user is requesting automation using Python logic.
    This is a backup to LLM intent classification.
    
    Args:
        message: User's message
        
    Returns:
        True if automation request detected
    """
    msg_lower = message.lower()
    
    # Check for action words
    action_words = ["run", "execute", "validate", "check", "perform"]
    has_action = any(word in msg_lower for word in action_words)
    
    # Check for data product name pattern (AL####.Name)
    has_name = bool(re.search(r'al\d+\.\w+', msg_lower))
    
    # Check for type keywords
    has_type = any(t in msg_lower for t in ["odp", "fdp", "cdp"])
    
    return has_action and has_name and has_type


def _extract_automation_params(message: str, automation_step: str) -> Optional[str]:
    """
    Extract parameters from user message based on automation type
    
    Args:
        message: User's message
        automation_step: Type of automation (e.g., "validate_name")
        
    Returns:
        JSON string of parameters, or None if extraction failed
    """
    if automation_step == "validate_name":
        return _extract_name_validation_params(message)
    
    # Add other automation parameter extractors here
    
    return None


def _extract_name_validation_params(message: str) -> Optional[str]:
    """
    Extract name validation parameters (name and type) from message
    
    Args:
        message: User's message
        
    Returns:
        JSON string with {"name": str, "type": str} or None
    """
    print(f"DEBUG: Extracting params from message: {message}")
    
    # Extract name (AL#####.Name pattern) - be more flexible with the pattern
    name_match = re.search(r'(AL\d+\.[A-Za-z0-9.]+)', message, re.IGNORECASE)
    if not name_match:
        print("DEBUG: No name match found")
        return None
    
    name = name_match.group(1)
    # Normalize the name
    name = name.strip()
    print(f"DEBUG: Extracted name: {name}")
    
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
        print("DEBUG: No type found")
        return None
    
    print(f"DEBUG: Extracted type: {type_val}")
    
    # Return as JSON string with proper escaping
    params = {
        "name": name,
        "type": type_val,
        "max_len": 75
    }
    
    json_str = json.dumps(params, ensure_ascii=False)
    print(f"DEBUG: Generated JSON: {json_str}")
    
    return json_str