from typing import TypedDict, List, Optional, Literal, Annotated
from datetime import datetime
import operator


# State for the governance workflow graph.
class GovernanceState(TypedDict, total=False):
    # Current turn inputs
    user_message: str
    session_id: str

    # Intent classification
    intent: Literal[
        "greeting",
        "ask_about_step",
        "ask_next_step",
        "ask_previous_step",
        "mark_complete",
        "mark_in_progress",
        "automation_request",
        "clarification_needed",
        "unknown",
    ]

    # Step identification
    focus_step_id: Optional[str]  # Primary step being discussed
    anchor_step_id: Optional[str]  # Workflow anchor for next/prev navigation
    candidate_step_ids: List[str]  # Multiple possible matches requiring disambiguation

    # Step collections
    next_step_ids: List[str]  # Downstream steps from focus
    completed_ids: Annotated[List[str], operator.add]  # Steps marked complete (accumulates)
    in_progress_ids: Annotated[List[str], operator.add]  # Steps marked in progress (accumulates)
    referenced_ids: List[str]  # All steps mentioned this turn

    # Automation
    automatable_step_id: Optional[str]  # Step that can be automated
    automation_step: Optional[str]  # Automation key (e.g., "validate_name")
    automation_result: Optional[dict]  # Result from automation execution

    # Matching details
    match_method: str  # How step was identified (id_match, alias_match, etc.)
    match_confidence: float  # Confidence score for matching

    # Response generation
    needs_disambiguation: bool  # True if multiple candidates found
    answer: str  # Final response to user

    # Conversation history (for memory across turns)
    previous_state: Optional[dict]  # State from previous turn
    timestamp: str  # ISO timestamp of current turn

    # Intermediate processing
    query_analysis: Optional[dict]  # Raw analysis from query analyzer
    step_details: Optional[dict]  # Detailed step information
    edges_info: Optional[dict]  # Edge information (next steps, etc.)


# Persistent session state stored across conversations.
class SessionState(TypedDict):
    last_intent: Optional[str]
    last_focus_step_id: Optional[str]
    last_anchor_step_id: Optional[str]
    last_next_step_ids: List[str]
    last_completed_ids: List[str]
    last_in_progress_ids: List[str]
    last_referenced_ids: List[str]
    last_automatable_step_id: Optional[str]
    last_automation_step: Optional[str]
    timestamp: str


def create_initial_state(user_message: str, session_id: str = "default") -> GovernanceState:
    return GovernanceState(
        user_message=user_message,
        session_id=session_id,
        intent="unknown",
        focus_step_id=None,
        anchor_step_id=None,
        candidate_step_ids=[],
        next_step_ids=[],
        completed_ids=[],
        in_progress_ids=[],
        referenced_ids=[],
        automatable_step_id=None,
        automation_step=None,
        automation_result=None,
        match_method="none",
        match_confidence=0.0,
        needs_disambiguation=False,
        answer="",
        previous_state=None,
        timestamp=datetime.utcnow().isoformat() + "Z",
        query_analysis=None,
        step_details=None,
        edges_info=None,
    )


def state_to_session_state(state: GovernanceState) -> SessionState:
    return SessionState(
        last_intent=state.get("intent"),
        last_focus_step_id=state.get("focus_step_id"),
        last_anchor_step_id=state.get("anchor_step_id"),
        last_next_step_ids=state.get("next_step_ids", []),
        last_completed_ids=state.get("completed_ids", []),
        last_in_progress_ids=state.get("in_progress_ids", []),
        last_referenced_ids=state.get("referenced_ids", []),
        last_automatable_step_id=state.get("automatable_step_id"),
        last_automation_step=state.get("automation_step"),
        timestamp=state.get("timestamp", datetime.utcnow().isoformat() + "Z"),
    )


def session_state_to_previous_state(session_state: Optional[SessionState]) -> Optional[dict]:
    if not session_state:
        return None

    return {
        "intent": session_state.get("last_intent"),
        "focus_step_id": session_state.get("last_focus_step_id"),
        "anchor_step_id": session_state.get("last_anchor_step_id"),
        "next_step_ids": session_state.get("last_next_step_ids", []),
        "completed_ids": session_state.get("last_completed_ids", []),
        "in_progress_ids": session_state.get("last_in_progress_ids", []),
        "referenced_ids": session_state.get("last_referenced_ids", []),
        "automatable_step_id": session_state.get("last_automatable_step_id"),
        "automation_step": session_state.get("last_automation_step"),
        "timestamp": session_state.get("timestamp"),
    }

