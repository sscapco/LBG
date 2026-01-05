import json
from typing import Any, Dict, List, Optional


def classify_intent_prompt(user_message: str, previous_focus_step_id: Optional[str] = None) -> str:
    context = ""
    if previous_focus_step_id:
        context = f"\nPrevious context: User was discussing step {previous_focus_step_id}."

    return f"""Classify the user's intent in this governance workflow conversation.

User message: "{user_message}"{context}

Available intents:
- greeting: General greeting or hello
- ask_about_step: Asking about a specific governance step
- ask_next_step: Asking what comes next / what to do next
- ask_previous_step: Asking what came before
- mark_complete: Marking a step as complete
- mark_in_progress: Marking a step as in progress
- automation_request: Requesting to run an automation (e.g., validate name)
- clarification_needed: Unclear or needs more info
- unknown: Cannot determine

Respond with ONLY a JSON object:
{{
  "intent": "one of the above intents",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}"""


def disambiguation_prompt(user_message: str, candidate_details: List[Dict[str, Any]]) -> str:
    return f"""The user asked: "{user_message}"

This question could relate to multiple governance steps. DO NOT pretend to know the single correct step.

Here are the candidate steps with full details:

{json.dumps(candidate_details, indent=2)}

Your task:
1. Briefly acknowledge the ambiguity (e.g., "This could relate to a couple of different steps")
2. For EACH candidate step, provide:
   - The step ID and name
   - A clear 2-3 sentence explanation of what it involves (use the description)
   - How it differs from the other candidates
3. Ask 1-2 clarifying questions to help them choose which applies

Guidelines:
- Use natural language, not JSON formatting
- Use bullet points or numbered lists for clarity
- Explain the KEY DIFFERENCES between the steps
- Keep it practical and action-oriented
- End with clarifying questions

Example structure:
"Thanks for your question! This could relate to a couple of different steps:

**S16: Information Security** involves [explain what it covers using description]

**S8: Cloud Control Assurance** specifically focuses on [explain what it covers using description]

The key difference is [explain how they differ].

[Ask clarifying question to help them choose]"

Now write your response:"""


def step_response_prompt(
    user_message: str,
    step_details: Dict[str, Any],
    next_step_info: List[Dict[str, Any]],
    automation_info: str,
    intent: Optional[str],
) -> str:
    return f"""The user asked: "{user_message}"

Current Step: {step_details['name']} (ID: {step_details['id']})

**Purpose**: {step_details['purpose']}

**What you need to do** (use verbatim as bullet points):
{step_details['description']}

{f"Next Steps: " + json.dumps(next_step_info, indent=2) if next_step_info else ""}

{automation_info}

Create a helpful response that:
1. Clearly states the step ID and name
2. Explains the PURPOSE first (one line)
3. Provides the EXACT description as bullet points (do not summarize or reword)
4. If next steps exist, briefly mention them
5. If automation is available, include the automation section EXACTLY as shown above

Keep it clear and practical."""


def next_step_prompt(user_message: str, next_step_details: List[Dict[str, Any]], anchor_record: Optional[Dict[str, Any]]) -> str:
    return f"""The user asked about the next step{f' after {anchor_record["name"]}' if anchor_record else ''}.

Next step(s):
{json.dumps(next_step_details, indent=2)}

Create a helpful response that:
1. Confirms what comes next
2. Briefly explains each next step
3. If multiple next steps, help them understand the branching
4. Keep it practical and action-oriented

Use a natural, conversational tone."""


def fallback_prompt(user_message: str) -> str:
    return f"""The user asked: "{user_message}"

I'm not quite sure what they're looking for in the governance workflow.

Create a brief, helpful response that:
1. Acknowledges their question
2. Offers to help them with:
   - Learning about specific steps
   - Understanding what comes next
   - Running automations
3. Asks a clarifying question

Keep it friendly and concise (2-3 sentences)."""

