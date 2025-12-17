"""
Automation detection: Detect when user requests automation.
"""
import re
import json
from typing import Optional


class AutomationDetector:
    """Detects automation requests from user messages"""
    
    def detect(self, user_message: str) -> bool:
        """Check if message is an automation request"""
        msg_lower = user_message.lower()
        
        # Check for action words
        action_words = ["run", "execute", "validate", "check", "perform"]
        has_action = any(word in msg_lower for word in action_words)
        
        # Check for data product name pattern (AL####.Name)
        has_name = bool(re.search(r'al\d+\.\w+', msg_lower))
        
        # Check for type keywords
        has_type = any(t in msg_lower for t in ["odp", "fdp", "cdp"])
        
        return has_action and has_name and has_type
    
    def extract_params(self, user_message: str) -> Optional[str]:
        """Extract automation parameters from message"""
        # Extract name
        name_match = re.search(r'(AL\d+\.\w+)', user_message, re.IGNORECASE)
        if not name_match:
            return None
        
        name = name_match.group(1)
        
        # Extract type
        msg_lower = user_message.lower()
        if "odp" in msg_lower:
            type_val = "ODP"
        elif "fdp" in msg_lower:
            type_val = "FDP"
        elif "cdp" in msg_lower:
            type_val = "CDP"
        else:
            return None
        
        return json.dumps({"name": name, "type": type_val})
