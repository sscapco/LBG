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
    
    # Handle different intents
    if intent == "greeting":
        state["answer"] = _generate_greeting_response(state)
    
    elif intent == "automation_request":
        # Should have been handled by automation node
        state["answer"] = "I can help with automation. Please specify which step and provide the required parameters."
    
    elif state.get("needs_disambiguation"):
        state["answer"] = _generate_disambiguation_response(state)
    
    elif state.get("focus_step_id"):
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
    
    if not candidates:
        return "I couldn't find a matching step. Could you provide more details?"
    
    # Get details for each candidate
    candidate_details = []
    for step_id in candidates[:3]:  # Top 3
        record = governance_data.get_step_record(step_id)
        if record:
            candidate_details.append(record)
    
    # Build prompt for LLM to create disambiguation message
    prompt = f"""The user asked: "{state.get('user_message')}"

I found multiple possible steps they might be referring to:

{json.dumps(candidate_details, indent=2)}

Create a friendly, concise response that:
1. Acknowledges their question
2. Lists the matching steps with brief descriptions
3. Asks them to clarify which one they meant

Keep it conversational and helpful. Use bullet points for the step list."""
    
    messages = [{"role": "user", "content": prompt}]
    response = cortex.get_chat_response(
        messages,
        max_tokens=800,
        temperature=0.3,
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
Purpose: {step_details['purpose']}

Full Description (use verbatim):
{step_details['description']}

{f"Next Steps: " + json.dumps(next_step_info, indent=2) if next_step_info else ""}

{automation_info}

Create a helpful response that:
1. Clearly states the step ID and name
2. Provides the EXACT description from above as bullet points (do not summarize)
3. If next steps exist, briefly mention them
4. If automation is available, include the automation section EXACTLY as shown above

Keep it clear and practical. Use bullet points for the description."""
    
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