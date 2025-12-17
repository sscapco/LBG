from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class GovernanceSessionState(BaseModel):
    """State tracking for governance workflow sessions"""
    
    last_intent: Optional[str] = None
    last_focus_step_id: Optional[str] = None
    last_anchor_step_id: Optional[str] = None
    last_next_step_ids: List[str] = Field(default_factory=list)
    last_completed_ids: List[str] = Field(default_factory=list)
    last_in_progress_ids: List[str] = Field(default_factory=list)
    last_referenced_ids: List[str] = Field(default_factory=list)
    last_automatable_step_id: Optional[str] = None
    last_automation_step: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class GovernanceToolResult(BaseModel):
    """Result from governance analysis tool"""
    
    answer: str
    updated_state: GovernanceSessionState


class GraphState(BaseModel):
    """State for LangGraph workflow"""
    
    # Input
    user_message: str
    session_id: str = "default_session"
    
    # Workflow state
    intent: Optional[str] = None
    focus_step_id: Optional[str] = None
    anchor_step_id: Optional[str] = None
    next_step_ids: List[str] = Field(default_factory=list)
    completed_ids: List[str] = Field(default_factory=list)
    in_progress_ids: List[str] = Field(default_factory=list)
    referenced_ids: List[str] = Field(default_factory=list)
    
    # Context information
    matched_steps: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_steps: List[Dict[str, Any]] = Field(default_factory=list)
    needs_disambiguation: bool = False
    
    # Automation
    automation_options: List[Dict[str, Any]] = Field(default_factory=list)
    automation_requested: bool = False
    automation_params: Optional[str] = None
    automation_result: Optional[Dict[str, Any]] = None
    
    # Output
    answer: str = ""
    
    # Session state tracking
    previous_state: Optional[GovernanceSessionState] = None
    updated_state: Optional[GovernanceSessionState] = None
    
    class Config:
        arbitrary_types_allowed = True


class StepRecord(BaseModel):
    """Record for a single governance step"""
    
    id: str
    name: str
    purpose: str
    description: str
    stage: str
    stage_name: str
    automatable: bool = False
    automation_step: Optional[str] = None


class EdgeRecord(BaseModel):
    """Record for an edge between steps"""
    
    from_step: str
    to_step: str
    edge_type: str
    guard_condition: Optional[str] = None
    question_probe: Optional[str] = None