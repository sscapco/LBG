from typing import TypedDict, Optional, Dict, Any

class ChatState(TypedDict, total=False):
    session_id: str
    io_mode: str
    user_message: str

    # Set by the router node or inferred
    active_agent: Optional[str]
    route_score: Optional[float]
    route_sim: Optional[float]
    routing_debug: Dict[str, Any]

    # Final payload your UI already knows how to render
    envelope: Dict[str, Any]
