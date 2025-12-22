"""
Response generator node: Creates the final user-facing response
"""
import json
from typing import List, Dict
from state import GovernanceState
from governance_data import governance_data
from cortex_connection import cortex


def response_generator_node(state: GovernanceState) -> GovernanceState:
    """
    Generate final response to user based on analyzed query and gathered information.
    
    This node:
    1. Checks if answer already exists (e.g., from automation)
    2. Handles different intents appropriately
    3. Formats step information professionally
    4. Handles disambiguation when multiple candidates exist
    5. Generates natural language response using LLM
    
    Args:
        state: Current governance state
        
    Returns:
        Updated state with 'answer' field populated
    """
    # If answer already set (e.g., by automation node), skip
    if state.get("answer"):
        return state
    
    intent = state.get("intent")
    needs_disambiguation = state.get("needs_disambiguation")
    focus_step_id = state.get("focus_step_id")
    
    print(f"\nDEBUG response_generator:")
    print(f"  Intent: {intent}")
    print(f"  Needs disambiguation: {needs_disambiguation}")
    print(f"  Focus step: {focus_step_id}")
    
    # Handle different intents
    if intent == "greeting":
        print(f"  → Generating greeting")
        state["answer"] = _generate_greeting_response(state)
    
    elif intent == "automation_request":
        # Should have been handled by automation node
        print(f"  → Generating automation prompt")
        state["answer"] = "I can help with automation. Please specify which step and provide the required parameters."
    
    elif needs_disambiguation:
        print(f"  → Generating DISAMBIGUATION response")
        state["answer"] = _generate_disambiguation_response(state)
    
    elif focus_step_id:
        print(f"  → Generating step response for {focus_step_id}")
        state["answer"] = _generate_step_response(state)
    
    elif intent == "ask_next_step":
        state["answer"] = _generate_next_step_response(state)
    
    else:
        state["answer"] = _generate_fallback_response(state)
    
    return state


def _generate_greeting_response(state: GovernanceState) -> str:
    """Generate friendly greeting response"""
    return (
        "Hello! I'm your governance workflow assistant. "
        "I can help you:\n"
        "• Understand specific governance steps\n"
        "• Navigate through the workflow\n"
        "• Run automations like name validation\n\n"
        "What would you like to know?"
    )


def _generate_disambiguation_response(state: GovernanceState) -> str:
    """Generate response when multiple step candidates found"""
    candidates = state.get("candidate_step_ids", [])
    user_message = state.get("user_message", "")
    
    if not candidates:
        return "I couldn't find a matching step. Could you provide more details?"
    
    # Get FULL details for each candidate (not just purpose!)
    candidate_details = []
    for step_id in candidates[:3]:  # Top 3
        record = governance_data.get_step_record(step_id)
        if record:
            candidate_details.append({
                "id": record["id"],
                "name": record["name"],
                "purpose": record["purpose"],
                "description": record["description"]  # Include full description!
            })
    
    # Build rich prompt for LLM to explain candidates
    prompt = f"""The user asked: "{user_message}"

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
    
    messages = [{"role": "user", "content": prompt}]
    response = cortex.get_chat_response(
        messages,
        max_tokens=1200,  # Need more tokens for detailed explanations
        temperature=0.0,
        thinking_enabled=False
    )
    
    return response


def _generate_step_response(state: GovernanceState) -> str:
    """Generate detailed response about a specific step"""
    step_details = state.get("step_details")
    if not step_details:
        return "I couldn't find information about that step."
    
    focus_id = state.get("focus_step_id")
    next_steps = state.get("next_step_ids", [])
    automatable = state.get("automatable_step_id") is not None
    automation_step = state.get("automation_step")
    
    # Get next step details
    next_step_info = []
    for next_id in next_steps[:3]:  # Top 3
        next_rec = governance_data.get_step_record(next_id)
        if next_rec:
            next_step_info.append(next_rec)
    
    # Build automation explanation if available
    automation_info = ""
    if automatable and automation_step:
        # Get automation details from registry
        from automation_registry import get_automation_info
        auto_info = get_automation_info(automation_step)
        if auto_info:
            automation_info = f"""
🤖 **Automation Available**: {auto_info.get('display_name', automation_step)}
{auto_info.get('description', 'Automated helper for this step')}

To use it, simply ask me to run it. For example:
- "Run the {auto_info.get('display_name', 'automation').lower()}"
- "Can you validate this for me?"
"""
    
    # Build context for LLM
    context = {
        "user_question": state.get("user_message"),
        "step": step_details,
        "next_steps": next_step_info,
        "automation_info": automation_info,
        "intent": state.get("intent")
    }
    
    prompt = f"""The user asked: "{state.get('user_message')}"

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
    
    messages = [{"role": "user", "content": prompt}]
    response = cortex.get_chat_response(
        messages,
        max_tokens=1200,
        temperature=0.0,
        thinking_enabled=False
    )
    
    return response


def _generate_next_step_response(state: GovernanceState) -> str:
    """Generate response about what comes next"""
    next_steps = state.get("next_step_ids", [])
    anchor_id = state.get("anchor_step_id")
    
    if not next_steps:
        return "I don't see any defined next steps from here. You might want to review the workflow or ask about a specific step."
    
    # Get details for next steps
    next_step_details = []
    for step_id in next_steps:
        record = governance_data.get_step_record(step_id)
        if record:
            next_step_details.append(record)
    
    anchor_record = None
    if anchor_id:
        anchor_record = governance_data.get_step_record(anchor_id)
    
    prompt = f"""The user asked about the next step{f' after {anchor_record["name"]}' if anchor_record else ''}.

Next step(s):
{json.dumps(next_step_details, indent=2)}

Create a helpful response that:
1. Confirms what comes next
2. Briefly explains each next step
3. If multiple next steps, help them understand the branching
4. Keep it practical and action-oriented

Use a natural, conversational tone."""
    
    messages = [{"role": "user", "content": prompt}]
    response = cortex.get_chat_response(
        messages,
        max_tokens=1000,
        temperature=0.0,
        thinking_enabled=False
    )
    
    return response


def _generate_fallback_response(state: GovernanceState) -> str:
    """Generate fallback response when intent unclear"""
    user_message = state.get("user_message", "")
    
    # Try to provide something helpful
    prompt = f"""The user asked: "{user_message}"

I'm not quite sure what they're looking for in the governance workflow.

Create a brief, helpful response that:
1. Acknowledges their question
2. Offers to help them with:
   - Learning about specific steps
   - Understanding what comes next
   - Running automations
3. Asks a clarifying question

Keep it friendly and concise (2-3 sentences)."""
    
    messages = [{"role": "user", "content": prompt}]
    response = cortex.get_chat_response(
        messages,
        max_tokens=300,
        temperature=0.3,
        thinking_enabled=False
    )
    
    return response