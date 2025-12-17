import re
import json
from typing import Optional, Dict, Any


######### Automation Detection #########

def detect_automation_request(message: str) -> bool:
    """
    Detect if user is requesting automation using reliable Python logic
    
    Looks for patterns like:
    - Action words (run, execute, validate, check, perform)
    - Data product name pattern (AL####.Name)
    - Type keywords (ODP, FDP, CDP)
    """
    msg_lower = message.lower()
    
    # Check for action words
    action_words = ["run", "execute", "validate", "check", "perform", "automate"]
    has_action = any(word in msg_lower for word in action_words)
    
    # Check for data product name pattern (AL####.Name)
    has_name = bool(re.search(r'al\d+\.\w+', msg_lower))
    
    # Check for type keywords
    has_type = any(t in msg_lower for t in ["odp", "fdp", "cdp"])
    
    return has_action and has_name and has_type


def extract_automation_params(message: str) -> Optional[str]:
    """
    Extract automation parameters from message using Python regex
    
    Returns:
        JSON string with {"name": "...", "type": "..."} or None
    """
    # Extract name 
    name_match = re.search(r'(AL\d+\.\w+)', message, re.IGNORECASE)
    if not name_match:
        return None
    
    name = name_match.group(1)
    
    # Extract type
    msg_lower = message.lower()
    if "odp" in msg_lower:
        type_val = "ODP"
    elif "fdp" in msg_lower:
        type_val = "FDP"
    elif "cdp" in msg_lower:
        type_val = "CDP"
    else:
        return None
    
    # Return as JSON string
    return json.dumps({"name": name, "type": type_val})


######### Automation Registry Integration #########

# Note: This expects automation_registry2.py to exist with these functions
try:
    from automation_registry2 import (
        get_automation_info,
        run_automation_step,
        format_automation_result_for_user,
        list_available_automations
    )
    AUTOMATION_AVAILABLE = True
except ImportError:
    print("⚠️ automation_registry2.py not found. Automation features will be disabled.")
    AUTOMATION_AVAILABLE = False
    
    # Provide stub functions
    def get_automation_info(automation_step: str) -> Optional[Dict]:
        return None
    
    def run_automation_step(automation_step: str, user_input: Optional[str] = None) -> Dict:
        return {"status": "error", "message": "Automation registry not available"}
    
    def format_automation_result_for_user(result: Dict) -> str:
        return result.get("message", str(result))
    
    def list_available_automations() -> Dict:
        return {}


def execute_automation(
    automation_step: str,
    user_input: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute an automation step from the registry
    
    Args:
        automation_step: The automation step key (e.g., "naming_convention_check")
        user_input: Optional JSON string with parameters
        
    Returns:
        Dict with status, formatted_message, and raw_result
    """
    if not AUTOMATION_AVAILABLE:
        return {
            "status": "error",
            "formatted_message": "❌ Automation registry not available",
            "raw_result": {"error": "automation_not_available"}
        }
    
    # Get automation info
    info = get_automation_info(automation_step)
    if not info:
        available = list(list_available_automations().keys())
        return {
            "status": "error",
            "formatted_message": f"❌ Unknown automation: {automation_step}. Available: {', '.join(available)}",
            "raw_result": {"error": "unknown_automation", "available": available}
        }
    
    try:
        # Execute automation
        result = run_automation_step(automation_step, user_input)
        
        # Format for user
        formatted = format_automation_result_for_user(result)
        
        return {
            "status": result.get("status"),
            "formatted_message": formatted,
            "raw_result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "formatted_message": f"❌ Automation execution failed: {str(e)}",
            "raw_result": {"error": str(e)}
        }


def handle_automation_in_message(
    message: str,
    automation_step: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Check if message contains automation request and execute if applicable
    
    Args:
        message: User's message
        automation_step: The automation step key from workflow analysis
        
    Returns:
        Automation result dict if executed, None otherwise
    """
    if not detect_automation_request(message):
        return None
    
    if not automation_step:
        return None
    
    # Extract parameters
    params_json = extract_automation_params(message)
    
    if not params_json:
        return {
            "status": "error",
            "formatted_message": "❌ Could not extract required parameters from your message. Please specify the data product name (e.g., AL1234.ProductName) and type (ODP/FDP/CDP).",
            "raw_result": {"error": "missing_parameters"}
        }
    
    # Execute automation
    return execute_automation(automation_step, params_json)