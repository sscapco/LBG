"""
Intent parsing: Classify user messages and extract actions.
"""
import json
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage

from core.models import IntentParseResult, ParsedAction, GovernanceSessionState
from core.cortex_client import CortexChatModel
from config import WorkflowConfig


class IntentParser:
    """Parses user intent and extracts actions"""
    
    def __init__(self, llm: CortexChatModel):
        self.llm = llm
    
    def parse(
        self,
        user_message: str,
        previous_state: Optional[GovernanceSessionState]
    ) -> IntentParseResult:
        """
        Parse user message to extract intent and actions.
        
        Returns:
            IntentParseResult with intent, actions, and metadata
        """
        previous_state_safe = json.loads(previous_state.model_dump_json()) if previous_state else {}
        
        system_prompt = (
            "You are a classifier for a data governance workflow. "
            "You do NOT guess step IDs or names; you ONLY work with natural-language actions and intents."
        )
        
        user_prompt = f"""
User message:
\"\"\"{user_message}\"\"\"

Previous workflow state (may be empty):
{json.dumps(previous_state_safe, indent=2)}

Your tasks:

1) Determine the user's intent:
   - "what_next" → they clearly ask what to do next or what comes after something
   - "help_current" → they ask for help completing something they say they are currently doing
   - "ask_about_step" → they want to understand what a specific step/process is or involves
   - "what_missed" → they ask which steps they might have missed or skipped
   - "run_automation" → they explicitly ask you to RUN an automation
   - "other" → anything else

2) Extract ACTIONS mentioned in the message as natural language phrases.
   For each action, assign a status:
     - "completed" → they say they finished/endorsed/approved it
     - "in_progress" → they say they are doing/preparing/working on it now
     - "mentioned_only" → they refer to it but don't say if it's done or in progress

3) Choose focus_action_index:
   - If the message clearly refers to one main step or action, set focus_action_index
     to the zero-based index in the actions list.
   - If there is no clear main action, set focus_action_index to null.

4) Decide if this is a STARTING QUESTION:
   - is_starting_question = true if the user is essentially asking:
       * "where do I start?"
       * "what's the first thing I should do?"
       * "how do I get started with this new data thing?"
     and they have NOT described any concrete progress or completed steps yet.
   - Otherwise, is_starting_question = false.

OUTPUT FORMAT (strict JSON only):

{{
  "intent": "what_next" | "help_current" | "ask_about_step" | "what_missed" | "run_automation" | "other",
  "actions": [
    {{
      "description": "<short phrase>",
      "status": "completed" | "in_progress" | "mentioned_only"
    }}
  ],
  "focus_action_index": <integer index or null>,
  "is_starting_question": true | false
}}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt + "\n\nRespond ONLY with valid JSON.")
        ]
        
        response = self.llm.invoke(
            messages,
            max_tokens=WorkflowConfig.INTENT_PARSING_MAX_TOKENS,
            temperature=WorkflowConfig.TEMPERATURE,
            response_format="json"
        )
        
        try:
            raw = json.loads(response.content)
        except Exception:
            raw = {}
        
        # Validate and normalize intent
        intent = raw.get("intent", "other")
        if intent not in ["what_next", "help_current", "ask_about_step", "what_missed", "run_automation", "other"]:
            intent = "other"
        
        # Parse actions
        actions_raw = raw.get("actions", [])
        actions = []
        for a in actions_raw:
            if not isinstance(a, dict):
                continue
            desc = (a.get("description") or "").strip()
            status = (a.get("status") or "mentioned_only").strip()
            if not desc:
                continue
            if status not in ["completed", "in_progress", "mentioned_only"]:
                status = "mentioned_only"
            actions.append(ParsedAction(description=desc, status=status))
        
        # Validate focus index
        focus_idx = raw.get("focus_action_index")
        if not isinstance(focus_idx, int) or not (0 <= focus_idx < len(actions)):
            focus_idx = None
        
        is_starting = bool(raw.get("is_starting_question", False))
        
        # Heuristic boost for obvious starting questions
        lower_msg = (user_message or "").lower()
        starting_phrases = [
            "where do i start", "where should i start", "how do i start",
            "how do we start", "how do i get started", "how do we get started",
            "first thing i should do", "first thing we should do", "where to begin",
        ]
        if any(p in lower_msg for p in starting_phrases):
            is_starting = True
        
        # Heuristic boost for "what next" questions
        if any(phrase in lower_msg for phrase in [
            "what should i do next", "what do i do next", "what comes after",
            "what's after", "what is after", "what comes next", "next step", "what next",
        ]):
            if intent not in ["what_missed", "run_automation"]:
                intent = "what_next"
        
        return IntentParseResult(
            intent=intent,
            actions=actions,
            focus_action_index=focus_idx,
            is_starting_question=is_starting,
        )
