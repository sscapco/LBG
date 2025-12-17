"""
LangGraph node implementations.
"""
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from datetime import datetime
import json
from typing import List, Dict

from graph.state import GovernanceGraphState
from core.models import (
    IntentParseResult,
    MappedSteps,
    WorkflowState,
    GovernanceSessionState,
)
from workflow import IntentParser, StepMapper, WorkflowComputer, AutomationDetector
from tools.registry import run_automation_step, format_automation_result_for_user
from core.cortex_client import get_cortex_llm
from config import WorkflowConfig


class GovernanceNodes:
    """Collection of graph node functions"""
    
    def __init__(
        self,
        intent_parser: IntentParser,
        step_mapper: StepMapper,
        workflow_computer: WorkflowComputer,
        automation_detector: AutomationDetector,
    ):
        self.intent_parser = intent_parser
        self.step_mapper = step_mapper
        self.workflow_computer = workflow_computer
        self.automation_detector = automation_detector
        self.answer_llm = get_cortex_llm(
            temperature=WorkflowConfig.TEMPERATURE,
            max_tokens=WorkflowConfig.ANSWER_GENERATION_MAX_TOKENS
        )
    
    def parse_intent_node(self, state: GovernanceGraphState) -> GovernanceGraphState:
        """Node 1: Parse user intent"""
        print("📍 Node: Parsing Intent")
        
        user_message = state["user_message"]
        previous_state_obj = (
            GovernanceSessionState(**state["previous_state"])
            if state.get("previous_state")
            else None
        )
        
        parsed = self.intent_parser.parse(user_message, previous_state_obj)
        state["parsed_intent"] = parsed.model_dump()
        
        return state
    
    def map_steps_node(self, state: GovernanceGraphState) -> GovernanceGraphState:
        """Node 2: Map to steps"""
        print("📍 Node: Mapping to Steps")
        
        user_message = state["user_message"]
        parsed = IntentParseResult(**state["parsed_intent"])
        previous_state_obj = (
            GovernanceSessionState(**state["previous_state"])
            if state.get("previous_state")
            else None
        )
        
        mapped = self.step_mapper.map(user_message, parsed, previous_state_obj)
        state["mapped_steps"] = mapped.model_dump()
        
        return state
    
    def compute_workflow_node(self, state: GovernanceGraphState) -> GovernanceGraphState:
        """Node 3: Compute workflow"""
        print("📍 Node: Computing Workflow")
        
        mapped = MappedSteps(**state["mapped_steps"])
        workflow_state = self.workflow_computer.compute(mapped)
        state["workflow_state"] = workflow_state.model_dump()
        
        return state
    
    def generate_answer_node(self, state: GovernanceGraphState) -> GovernanceGraphState:
        """Node 4: Generate answer"""
        print("📍 Node: Generating Answer")
        
        workflow_state = WorkflowState(**state["workflow_state"])
        user_message = state["user_message"]
        
        # Build context for LLM
        context = self._build_context(workflow_state, user_message)
        
        # Generate answer
        system_prompt = "You are a helpful, practical data governance assistant."
        user_prompt = self._build_answer_prompt(context)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.answer_llm.invoke(messages)
        answer = response.content
        
        # Build updated state
        automation_options = context.get("automation_options", [])
        auto_step_id = automation_options[0]["step_id"] if automation_options else None
        auto_step_key = automation_options[0]["automation_step"] if automation_options else None
        
        updated_state = {
            "last_intent": workflow_state.intent,
            "last_focus_step_id": workflow_state.focus_step_id,
            "last_anchor_step_id": workflow_state.anchor_step_id,
            "last_next_step_ids": workflow_state.next_step_ids,
            "last_completed_ids": workflow_state.completed_ids,
            "last_in_progress_ids": workflow_state.in_progress_ids,
            "last_referenced_ids": workflow_state.referenced_ids,
            "last_automatable_step_id": auto_step_id,
            "last_automation_step": auto_step_key,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        state["answer"] = answer
        state["updated_state"] = updated_state
        state["messages"] = list(state["messages"]) + [AIMessage(content=answer)]
        
        return state
    
    def check_automation_node(self, state: GovernanceGraphState) -> GovernanceGraphState:
        """Node 5: Check for automation request"""
        print("📍 Node: Checking for Automation")
        
        user_message = state["user_message"]
        should_run = self.automation_detector.detect(user_message)
        
        if should_run:
            params = self.automation_detector.extract_params(user_message)
            state["automation_params"] = params
        
        state["should_run_automation"] = should_run
        return state
    
    def run_automation_node(self, state: GovernanceGraphState) -> GovernanceGraphState:
        """Node 6: Run automation"""
        print("📍 Node: Running Automation")
        
        automation_step = state["updated_state"].get("last_automation_step")
        params_json = state["automation_params"]
        
        if automation_step and params_json:
            try:
                result = run_automation_step(automation_step, params_json)
                formatted = format_automation_result_for_user(result)
                state["answer"] = formatted
                state["automation_result"] = formatted
                print("   ✓ Automation executed")
            except Exception as e:
                error_msg = f"Automation error: {str(e)}"
                state["answer"] = error_msg
                state["automation_result"] = error_msg
                print(f"   ⚠️ {e}")
        
        return state
    
    def _build_context(self, workflow_state: WorkflowState, user_message: str) -> dict:
        """Build context dict for answer generation"""
        step_details = workflow_state.step_details
        
        def _step_label(s_id: str) -> str:
            if not s_id:
                return ""
            rec = step_details.get(s_id)
            if not rec:
                return s_id
            return f"{s_id} ({rec.get('name', s_id)})"
        
        # Determine automation options
        automation_options = []
        candidate_ids: List[str] = []
        if workflow_state.focus_step_id:
            candidate_ids.append(workflow_state.focus_step_id)
        candidate_ids.extend(workflow_state.next_step_ids)
        
        seen_auto = set()
        for sid in candidate_ids:
            if sid in seen_auto:
                continue
            seen_auto.add(sid)
            rec = step_details.get(sid)
            if rec and rec.get("automatable") and rec.get("automation_step"):
                automation_options.append({
                    "step_id": sid,
                    "label": _step_label(sid),
                    "automation_step": rec["automation_step"],
                })
        
        return {
            "user_message": user_message,
            "intent": workflow_state.intent,
            "is_starting_question": workflow_state.is_starting_question,
            "needs_disambiguation": workflow_state.needs_disambiguation,
            "completed_steps": [{"step_id": s, "label": _step_label(s)} for s in workflow_state.completed_ids],
            "in_progress_steps": [{"step_id": s, "label": _step_label(s)} for s in workflow_state.in_progress_ids],
            "referenced_steps": [{"step_id": s, "label": _step_label(s)} for s in workflow_state.referenced_ids],
            "anchor_step": {"step_id": workflow_state.anchor_step_id, "label": _step_label(workflow_state.anchor_step_id)} if workflow_state.anchor_step_id else None,
            "focus_step": {"step_id": workflow_state.focus_step_id, "label": _step_label(workflow_state.focus_step_id)} if workflow_state.focus_step_id else None,
            "next_steps": [{"step_id": s, "label": _step_label(s)} for s in workflow_state.next_step_ids],
            "missing_prerequisites": [{"step_id": s, "label": _step_label(s)} for s in workflow_state.missing_prereq_ids],
            "step_details": workflow_state.step_details,
            "prereq_edges": workflow_state.prereq_edges,
            "semantic_candidates": [{"step_id": c.step_id, "label": _step_label(c.step_id), "score": c.score} for c in workflow_state.semantic_candidates],
            "automation_options": automation_options,
        }
    
    def _build_answer_prompt(self, context: dict) -> str:
        """Build prompt for answer generation - using your original complete prompt"""
        return f"""
You are a governance workflow assistant helping users navigate a structured set of steps.

The context you must use is:
{json.dumps(context, indent=2)}

DEFINITIONS:
- intent: "what_next", "help_current", "ask_about_step", "what_missed"
- is_starting_question: true if asking where/how to start
- needs_disambiguation: true if question could map to multiple steps
- focus_step: main step to explain
- next_steps: step(s) to do next
- missing_prerequisites: mandatory steps not mentioned yet
- semantic_candidates: semantically similar steps with scores
- automation_options: available automation helpers

HARD RULES:

0) If needs_disambiguation = true:
   - Acknowledge ambiguity
   - List 2-3 candidate steps from semantic_candidates
   - Ask 1-2 clarifying questions
   - Do NOT pretend to know the single correct step

1) When intent = "ask_about_step" AND needs_disambiguation = false:
   - Focus explanation on focus_step
   - Clearly state step ID and name
   - Use purpose and description
   - May mention prerequisites and next steps as context

2) When intent = "help_current" AND needs_disambiguation = false:
   - Describe how to complete focus_step
   - Use the description as main source
   - May mention what comes next

3) When intent = "what_next" AND needs_disambiguation = false:
   - If next_steps not empty, primary "next step" MUST be first item
   - Phrase as: "Your next step is <step_id>: <step_name>"
   - Describe that step
   - NEVER describe missing_prerequisites as "next step"

4) When intent = "what_missed":
   - Do NOT assert anything is definitely missing
   - List candidates from missing_prerequisites to double-check
   - Use wording: "here are some steps to double-check..."

5) Missing prerequisites:
   - Treat as steps NOT MENTIONED, not definitely skipped
   - NEVER say "You haven't completed X"
   - Only ask gentle confirmation questions
   - Speak conditionally

6) If no progress and intent = "what_next" and is_starting_question = true:
   - Suggest first step as starting point

7) Automation:
   - NEVER run automation yourself
   - If automation_options NON-EMPTY:
     * Mention helper available
     * Explain what it does
     * Invite user to request it
   - Do NOT assume user wants automation

8) Tone:
   - Start with brief acknowledgement
   - Use bullet points if helpful
   - Keep concise and practical

Now, write your response to the user.
        """
