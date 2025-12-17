"""
LangGraph state definition.
"""
from typing import TypedDict, Annotated, Sequence, Optional, Dict
from langchain_core.messages import BaseMessage
import operator


class GovernanceGraphState(TypedDict):
    """State that flows through the governance workflow graph"""
    
    # Messages for LangGraph
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Input
    user_message: str
    session_id: str
    previous_state: Optional[Dict]
    
    # Analysis results (populated by nodes)
    parsed_intent: Optional[Dict]
    mapped_steps: Optional[Dict]
    workflow_state: Optional[Dict]
    
    # Output
    answer: Optional[str]
    updated_state: Optional[Dict]
    
    # Automation
    should_run_automation: bool
    automation_params: Optional[str]
    automation_result: Optional[str]
