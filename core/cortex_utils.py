import json
import re
from typing import Any, Dict, Optional, Mapping, Union


def cortex_chat_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    # OpenAI-style object: response.choices[0].message.content
    try:
        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content
            content = getattr(first, "text", None)
            if isinstance(content, str):
                return content
    except Exception:
        pass

    # Dict payload: {"choices": [{"message": {"content": "..."}}]}
    if isinstance(response, Mapping):
        try:
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, Mapping):
                    msg = first.get("message")
                    if isinstance(msg, Mapping) and isinstance(msg.get("content"), str):
                        return msg["content"]
                    if isinstance(first.get("text"), str):
                        return first["text"]
        except Exception:
            pass

    return str(response)


def parse_json_object(text: str, required: Dict[str, Union[type, tuple]]) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return None
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    for key, expected in required.items():
        if key not in payload:
            return None
        if not isinstance(payload[key], expected):
            return None

    return payload

