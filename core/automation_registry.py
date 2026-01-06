from typing import Any, Dict, Optional
import json

from automation_tools.handler import check_name_both


def validate_name_handler(user_input: Optional[str] = None) -> Dict[str, Any]:
    if user_input is None:
        return {
            "status": "needs_input",
            "prompt": (
                "I'll validate your data product name against the naming standards.\n\n"
                "Please provide:\n"
                "1. The data product name you want to validate\n"
                "2. The data product type (ODP, FDP, or CDP)\n\n"
                "Example: 'AL12345.CustomerData' for type ODP"
            ),
            "expected_format": {"name": "AL12345.CustomerData", "type": "ODP", "max_len": 75},
        }

    try:
        if isinstance(user_input, str):
            params = json.loads(user_input)
        else:
            params = user_input

        name = (params.get("name") or "").strip() or None
        dp_type = (params.get("type") or "").strip() or None
        max_len = params.get("max_len", 75)

        if not name or not dp_type:
            missing = []
            if not name:
                missing.append("name")
            if not dp_type:
                missing.append("type")

            prompt_lines = ["To run name validation, I still need:"]
            if "name" in missing:
                prompt_lines.append("- The data product name (e.g., AL12345.CustomerData)")
            if "type" in missing:
                prompt_lines.append("- The data product type (ODP, FDP, or CDP)")

            expected = {"max_len": 75}
            if name:
                expected["name"] = name
            if dp_type:
                expected["type"] = dp_type
            else:
                expected["type"] = "ODP"

            return {
                "status": "needs_input",
                "prompt": "\n".join(prompt_lines),
                "expected_format": expected,
            }

        result = check_name_both(name, dp_type, max_len)

        overall = result.get("overall", {})
        verdict = overall.get("verdict", "unknown")
        explanation = overall.get("explanation", "")
        suggestion = overall.get("suggestion")

        response: Dict[str, Any] = {
            "status": "success",
            "verdict": verdict,
            "input_name": name,
            "type": dp_type,
            "explanation": explanation,
        }

        if suggestion:
            response["suggested_name"] = suggestion
            edits = overall.get("edits", [])
            if edits:
                response["edits"] = edits

        checks = overall.get("checks", [])
        blocking_issues = [c for c in checks if c.get("severity") == "block" and c.get("status") != "pass"]
        if blocking_issues:
            response["blocking_issues"] = blocking_issues

        response["detailed_results"] = result
        return response

    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Validation error: {str(e)}"}


AUTOMATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "validate_name": {
        "handler": validate_name_handler,
        "description": "Validate data product names against naming standards (ODP/FDP/CDP)",
        "display_name": "Data Product Name Validation",
        "category": "naming",
    },
}


def get_automation_info(automation_step: str) -> Optional[Dict[str, Any]]:
    return AUTOMATION_REGISTRY.get(automation_step)


def run_automation_step(automation_step: str, user_input: Optional[str] = None) -> Dict[str, Any]:
    automation = AUTOMATION_REGISTRY.get(automation_step)
    if not automation:
        return {
            "status": "error",
            "message": f"Unknown automation step: '{automation_step}'",
            "available_automations": list(AUTOMATION_REGISTRY.keys()),
        }

    handler = automation["handler"]
    try:
        return handler(user_input)
    except Exception as e:
        return {"status": "error", "message": f"Automation execution failed: {str(e)}"}


def list_available_automations() -> Dict[str, Dict[str, str]]:
    return {
        step: {"description": info["description"], "display_name": info["display_name"], "category": info["category"]}
        for step, info in AUTOMATION_REGISTRY.items()
    }


def format_automation_result_for_user(result: Dict[str, Any]) -> str:
    status = result.get("status")

    if status == "needs_input":
        prompt = result.get("prompt", "Please provide input")
        expected = result.get("expected_format", {})
        return f"{prompt}\n\nExpected format:\n{json.dumps(expected, indent=2)}"

    if status == "error":
        message = result.get("message", "Unknown error")
        return f"❌ Error: {message}"

    if status == "success":
        if "verdict" in result:
            verdict = result["verdict"]
            input_name = result.get("input_name", "")
            explanation = result.get("explanation", "")

            if verdict == "valid":
                return f"✅ **Valid**: '{input_name}' passes all checks.\n\n{explanation}"
            if verdict == "needs_changes":
                suggested = result.get("suggested_name", "")
                return f"⚠️ **Needs Changes**: '{input_name}'\n\nSuggested: '{suggested}'\n\n{explanation}"
            return f"❌ **Invalid**: '{input_name}'\n\n{explanation}"

        message = result.get("message", "Automation completed successfully")
        return f"✅ {message}"

    return "Unexpected result format"
