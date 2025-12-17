"""
Pydantic models for type safety and validation.
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class ParsedAction(BaseModel):
    """Action extracted from user message"""
    description: str
    status: str  # "completed" | "in_progress" | "mentioned_only"


class IntentParseResult(BaseModel):
    """Result of intent parsing"""
    intent: str  # "what_next" | "help_current" | "ask_about_step" | "what_missed" | "run_automation" | "other"
    actions: List[ParsedAction] = Field(default_factory=list)
    focus_action_index: Optional[int] = None
    is_starting_question: bool = False


class StepCandidate(BaseModel):
    """Candidate step from semantic matching"""
    step_id: str
    score: float


class MappedSteps(BaseModel):
    """Result of mapping parsed actions to steps"""
    intent: str
    is_starting_question: bool
    completed_ids: List[str] = Field(default_factory=list)
    in_progress_ids: List[str] = Field(default_factory=list)
    referenced_ids: List[str] = Field(default_factory=list)
    focus_step_candidate_id: Optional[str] = None
    semantic_candidates: List[StepCandidate] = Field(default_factory=list)
    has_direct_mapping: bool = False


class WorkflowState(BaseModel):
    """Complete workflow state after reasoning"""
    intent: str
    is_starting_question: bool
    completed_ids: List[str] = Field(default_factory=list)
    in_progress_ids: List[str] = Field(default_factory=list)
    referenced_ids: List[str] = Field(default_factory=list)
    anchor_step_id: Optional[str] = None
    focus_step_id: Optional[str] = None
    next_step_ids: List[str] = Field(default_factory=list)
    missing_prereq_ids: List[str] = Field(default_factory=list)
    step_details: Dict[str, Dict] = Field(default_factory=dict)
    prereq_edges: Dict[str, List[Dict]] = Field(default_factory=dict)
    semantic_candidates: List[StepCandidate] = Field(default_factory=list)
    needs_disambiguation: bool = False


class GovernanceSessionState(BaseModel):
    """Session state tracking conversation history"""
    last_intent: Optional[str] = None
    last_focus_step_id: Optional[str] = None
    last_anchor_step_id: Optional[str] = None
    last_next_step_ids: List[str] = Field(default_factory=list)
    last_completed_ids: List[str] = Field(default_factory=list)
    last_in_progress_ids: List[str] = Field(default_factory=list)
    last_referenced_ids: List[str] = Field(default_factory=list)
    last_automatable_step_id: Optional[str] = None
    last_automation_step: Optional[str] = None
    timestamp: Optional[str] = None


class GovernanceToolResult(BaseModel):
    """Final result with answer and updated state"""
    answer: str
    updated_state: GovernanceSessionState


class StepRecord(BaseModel):
    """Governance step record from Excel"""
    id: str
    name: str
    purpose: str
    description: str
    stage: int
    stage_name: str
    automatable: bool = False
    automation_step: Optional[str] = None
