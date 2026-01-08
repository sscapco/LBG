import json
from typing import Any, Dict, List, Optional


def classify_intent_prompt(user_message: str, previous_focus_step_id: Optional[str] = None) -> str:
    context = ""
    if previous_focus_step_id:
        context = f"\nPrevious context: User was discussing step {previous_focus_step_id}."

    return f"""Classify the user's intent in this governance workflow conversation.

User message: "{user_message}"{context}

Available intents with examples:
- greeting: General greeting or hello (e.g., "Hi", "Hello", "Good morning")
- compare_steps: Asking for differences/comparison between two steps/concepts (e.g., "What's the difference between X and Y?", "How does A compare to B?")
- ask_about_step: Asking about a specific governance step or describing work done (e.g., "I've completed the DOI form", "We created a record in ServiceNow", "I raised a JIRA ticket", "What do I need to do for security?")
- ask_next_step: Asking what comes next / what to do next (e.g., "What's next?", "What should I do after this?", "Where do we go from here?")
- ask_previous_step: Asking what came before (e.g., "What step comes before this?", "What did I need to do earlier?")
- mark_complete: Explicitly marking a step as complete (e.g., "Mark S1 as complete", "I finished step 2")
- mark_in_progress: Explicitly marking a step as in progress (e.g., "Mark S3 as in progress", "I'm working on step 5")
- automation_request: Explicitly requesting to run an automation (e.g., "Run the naming validation", "Can you validate this name?", "Execute the check")
- clarification_needed: Unclear or needs more info (e.g., "What?", "Huh?", "I don't understand")
- unknown: Cannot determine

IMPORTANT:
- If the user describes completing work, reaching a milestone, or mentions specific artifacts/activities (like "raised a ticket", "completed a form", "got endorsement", "delivered an artefact"), classify as "ask_about_step"
- Only use "greeting" for actual greetings, NOT for questions about work or progress
- Questions starting with "what", "where", "when", "how", "why" about governance activities should be "ask_about_step"

Respond with ONLY a JSON object:
{{
  "intent": "one of the above intents",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation",
  "comparison": null or {{
    "a": "string (first concept/step mentioned)",
    "b": "string (second concept/step mentioned)"
  }}
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


def compare_steps_prompt(user_message: str, steps: List[Dict[str, Any]]) -> str:
    return f"""The user asked: "{user_message}"

They want a comparison of governance steps using ONLY the Nodes-sheet information.

Here are the steps (do not invent details not present):
{json.dumps(steps, indent=2)}

Output format (no markdown fences, no JSON):

For EACH step:
- "<ID>: <Step_Name>"
- "Purpose: <Purpose>" (verbatim)
- "Description:" followed by the EXACT description as bullet points (verbatim; do not summarize or reword)

Then provide:
- "Key differences:" 2–4 bullet points that compare them based ONLY on the two purposes/descriptions above.

Do not ask clarifying questions unless a step is missing."""


def rerank_candidates_prompt(user_message: str, candidates: List[Dict[str, Any]]) -> str:
    return f"""Given a user query about a governance workflow, determine which step best matches their situation.

User query: "{user_message}"

Candidate steps (in order of initial similarity):
{json.dumps(candidates, indent=2)}

Your task:
1. Carefully read the user query and identify the KEY ACTIONS, ARTIFACTS, or MILESTONES mentioned
2. For each candidate, evaluate how well it matches those key elements
3. Consider:
   - Does the step name match the activity described?
   - Does the purpose align with what the user is asking about?
   - Does the description mention the specific artifacts/milestones the user referenced?
4. Return ONLY the step ID that BEST matches, or "AMBIGUOUS" if multiple steps genuinely match

Respond with ONLY a JSON object:
{{
  "best_match": "step_id or AMBIGUOUS",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation of why this step matches best",
  "ambiguous_candidates": ["step_id1", "step_id2"] or null
}}

IMPORTANT:
- If the user mentions a specific artifact (e.g., "JIRA ticket", "DOI form", "Threat Model", "ServiceNow record"), prioritize steps that explicitly mention it
- If multiple steps could genuinely apply, return "AMBIGUOUS" and list the candidates
- Be conservative: only return a single best_match if you're confident (>0.7)"""


def validate_scope_prompt(user_message: str, candidates: List[Dict[str, Any]]) -> str:
    """Prompt for LLM to validate if query is in-scope and which candidates match."""
    return f"""You are validating whether a user query relates to a governance workflow.

User query: "{user_message}"

Top candidate steps from semantic matching:
{json.dumps(candidates, indent=2)}

Your task:
1. **Scope Check**: Determine if this query is about the governance workflow at all
   - IN_SCOPE: Query asks about governance steps, processes, requirements, or describes work done
   - OUT_OF_SCOPE: Query is unrelated (greetings, weather, general questions, off-topic)

2. **If IN_SCOPE**: Determine which step(s) best match
   - Check if the query describes activities, artifacts, or milestones mentioned in step purposes/descriptions
   - Return step ID(s) if there's a clear match
   - Return AMBIGUOUS if multiple steps apply equally
   - Return NO_MATCH if query is in-scope but doesn't match any specific step well

3. **Be CONSERVATIVE**:
   - If semantic similarity seems weak or coincidental, return OUT_OF_SCOPE or NO_MATCH
   - Only return VALID with step IDs if you're confident (>0.7) there's a real match

Respond with ONLY a JSON object:
{{
  "scope": "IN_SCOPE" | "OUT_OF_SCOPE",
  "validation": "VALID" | "NO_MATCH" | "AMBIGUOUS",
  "step_ids": ["S1"] or ["S1", "S2"] or null,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation of your decision"
}}

Examples:
- Query: "How's the weather?" → scope: OUT_OF_SCOPE, validation: null, step_ids: null
- Query: "What is Python?" → scope: OUT_OF_SCOPE, validation: null, step_ids: null
- Query: "I raised a JIRA ticket" → scope: IN_SCOPE, validation: VALID, step_ids: ["S1"]
- Query: "What governance steps exist?" → scope: IN_SCOPE, validation: NO_MATCH, step_ids: null
"""
