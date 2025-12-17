from typing import Dict, Callable, Any, Optional
import json
from automation_tools.handler import check_name_both


# ==================== HANDLER FUNCTIONS ====================

def validate_name_handler(user_input: Optional[str] = None) -> Dict[str, Any]:
    
    if user_input is None:
        # Request input from user
        return {
            "status": "needs_input",
            "prompt": (
                "I'll validate your data product name against the naming standards.\n\n"
                "Please provide:\n"
                "1. The data product name you want to validate\n"
                "2. The data product type (ODP, FDP, or CDP)\n\n"
                "Example: 'AL12345.CustomerData' for type ODP"
            ),
            "expected_format": {
                "name": "AL12345.CustomerData",
                "type": "ODP",
                "max_len": 75  # optional
            }
        }
    
    try:
        # Parse user input
        if isinstance(user_input, str):
            params = json.loads(user_input)
        else:
            params = user_input
        
        name = params.get("name")
        dp_type = params.get("type")
        max_len = params.get("max_len", 75)
        
        if not name:
            return {
                "status": "error",
                "message": "Missing required field: 'name'"
            }
        
        if not dp_type:
            return {
                "status": "error",
                "message": "Missing required field: 'type' (must be ODP, FDP, or CDP)"
            }
        
        # Run validation
        result = check_name_both(name, dp_type, max_len)
        
        # Format response for user
        overall = result.get("overall", {})
        verdict = overall.get("verdict", "unknown")
        explanation = overall.get("explanation", "")
        suggestion = overall.get("suggestion")
        
        # Build user-friendly response
        response = {
            "status": "success",
            "verdict": verdict,
            "input_name": name,
            "type": dp_type,
            "explanation": explanation
        }
        
        if suggestion:
            response["suggested_name"] = suggestion
            edits = overall.get("edits", [])
            if edits:
                response["edits"] = edits
        
        # Include key checks
        checks = overall.get("checks", [])
        blocking_issues = [c for c in checks if c.get("severity") == "block" and c.get("status") != "pass"]
        if blocking_issues:
            response["blocking_issues"] = blocking_issues
        
        # Add detailed results for reference
        response["detailed_results"] = result
        
        return response
        
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "message": f"Invalid JSON format: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Validation error: {str(e)}"
        }


# ==================== FUTURE AUTOMATION HANDLERS ====================
# Add new automation handlers here following the same pattern

def example_future_automation(user_input: Optional[str] = None) -> Dict[str, Any]:
    """
    Example template for future automation handlers.
    
    Args:
        user_input: Optional input from user (JSON string or dict)
    
    Returns:
        Dictionary with automation results or request for input
    """
    if user_input is None:
        return {
            "status": "needs_input",
            "prompt": "Describe what input you need from the user",
            "expected_format": {
                "param1": "value1",
                "param2": "value2"
            }
        }
    
    # Process input and return results
    return {
        "status": "success",
        "message": "Automation completed successfully"
    }


# ==================== AUTOMATION REGISTRY ====================

AUTOMATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "validate_name": {
        "handler": validate_name_handler,
        "description": "Validate data product names against naming standards (ODP/FDP/CDP)",
        "display_name": "Data Product Name Validation",
        "category": "naming"
    },
    # Add more automation tools here as they're developed:
    # "check_lineage": {
    #     "handler": check_lineage_handler,
    #     "description": "Verify data lineage connections",
    #     "display_name": "Data Lineage Checker",
    #     "category": "lineage"
    # },
}


# ==================== AUTOMATION EXECUTION ====================

def get_automation_info(automation_step: str) -> Optional[Dict[str, Any]]:
    """
    Get information about an automation tool.
    
    Args:
        automation_step: The automation step identifier (e.g., "validate_name")
    
    Returns:
        Dictionary with automation info or None if not found
    """
    return AUTOMATION_REGISTRY.get(automation_step)


def run_automation_step(
    automation_step: str, 
    user_input: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute an automation step from the registry.
    
    Args:
        automation_step: The automation step identifier (e.g., "validate_name")
        user_input: Optional user input (JSON string or dict)
    
    Returns:
        Dictionary with results, which may include:
        - status: "success", "error", or "needs_input"
        - For success: automation results
        - For needs_input: prompt and expected_format
        - For error: error message
    """
    automation = AUTOMATION_REGISTRY.get(automation_step)
    
    if not automation:
        return {
            "status": "error",
            "message": f"Unknown automation step: '{automation_step}'",
            "available_automations": list(AUTOMATION_REGISTRY.keys())
        }
    
    handler = automation["handler"]
    
    try:
        result = handler(user_input)
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"Automation execution failed: {str(e)}"
        }


def list_available_automations() -> Dict[str, Dict[str, str]]:
    """
    Get a list of all available automation tools.
    
    Returns:
        Dictionary mapping automation_step to info dict with:
        - description
        - display_name
        - category
    """
    return {
        step: {
            "description": info["description"],
            "display_name": info["display_name"],
            "category": info["category"]
        }
        for step, info in AUTOMATION_REGISTRY.items()
    }


# ==================== HELPER FOR FORMATTING RESULTS ====================

def format_automation_result_for_user(result: Dict[str, Any]) -> str:
    """
    Format automation results into a user-friendly string.
    
    Args:
        result: Result dictionary from run_automation_step
    
    Returns:
        Formatted string for display to user
    """
    status = result.get("status")
    
    if status == "needs_input":
        prompt = result.get("prompt", "Please provide input")
        expected = result.get("expected_format", {})
        return f"{prompt}\n\nExpected format:\n{json.dumps(expected, indent=2)}"
    
    elif status == "error":
        message = result.get("message", "Unknown error")
        return f"❌ Error: {message}"
    
    elif status == "success":
        # Check if this is a name validation result
        if "verdict" in result:
            verdict = result["verdict"]
            input_name = result.get("input_name", "")
            explanation = result.get("explanation", "")
            
            if verdict == "valid":
                return f"✅ **Valid**: '{input_name}' passes all checks.\n\n{explanation}"
            elif verdict == "needs_changes":
                suggested = result.get("suggested_name", "")
                return (
                    f"⚠️ **Needs Changes**: '{input_name}'\n\n"
                    f"Suggested: '{suggested}'\n\n"
                    f"{explanation}"
                )
            else:  # invalid
                return (
                    f"❌ **Invalid**: '{input_name}'\n\n"
                    f"{explanation}"
                )
        else:
            # Generic success message
            message = result.get("message", "Automation completed successfully")
            return f"✅ {message}"
    
    return "Unexpected result format"