import json
import re
from typing import List, Dict, Optional, Tuple
from cortex_connection import cortex_client
from governance_data import (
    nodes_df, edges_df, get_step_record, map_text_to_step_id,
    step_index, ORDERED_STEP_IDS
)


######### Intent Classification #########

def classify_intent(user_message: str, previous_state: Optional[Dict] = None) -> str:
    """
    Classify user's intent from their message
    
    Returns one of:
    - "navigate_to_step"
    - "ask_next_step"
    - "ask_prev_step"
    - "list_all_steps"
    - "explain_step"
    - "mark_complete"
    - "general_question"
    """
    msg = user_message.lower()
    
    # Navigation patterns
    if any(word in msg for word in ["go to", "move to", "navigate to", "jump to"]):
        return "navigate_to_step"
    
    # Next step patterns
    if any(phrase in msg for phrase in ["what's next", "next step", "what do i do next", "where do i go"]):
        return "ask_next_step"
    
    # Previous step patterns
    if any(phrase in msg for phrase in ["what was", "previous", "go back", "before"]):
        return "ask_prev_step"
    
    # List all steps
    if any(phrase in msg for phrase in ["list all", "show all", "all steps", "complete workflow"]):
        return "list_all_steps"
    
    # Mark complete patterns
    if any(phrase in msg for phrase in ["completed", "finished", "done with"]):
        return "mark_complete"
    
    # Explanation patterns
    if any(word in msg for word in ["explain", "what is", "tell me about", "describe"]):
        return "explain_step"
    
    # Default: general question (likely navigating to a step)
    return "general_question"


######### Step Matching #########

def find_matching_steps(
    user_message: str,
    emb_threshold: float = 0.5,
    max_candidates: int = 3
) -> Tuple[List[Dict], str, float]:
    """
    Find steps matching the user's message
    
    Returns:
        (matched_steps, method, confidence)
    """
    step_id, method, confidence = map_text_to_step_id(user_message, emb_threshold)
    
    if step_id:
        rec = get_step_record(step_id)
        if rec:
            return [rec], method, confidence
    
    return [], method, confidence


######### Next Steps Logic #########

def find_next_steps(current_step_id: str, completed_ids: List[str]) -> List[str]:
    """Find next steps from current step, excluding completed ones"""
    outgoing = edges_df[edges_df["from"] == current_step_id]
    
    next_ids = []
    for _, edge in outgoing.iterrows():
        to_id = edge["to"]
        if to_id not in completed_ids:
            next_ids.append(to_id)
    
    # Sort by canonical order
    next_ids.sort(key=step_index)
    return next_ids


def find_previous_steps(current_step_id: str) -> List[str]:
    """Find previous steps that lead to current step"""
    incoming = edges_df[edges_df["to"] == current_step_id]
    prev_ids = list(incoming["from"].unique())
    prev_ids.sort(key=step_index)
    return prev_ids


######### Disambiguation Logic #########

def needs_disambiguation_check(candidates: List[Dict], confidence: float) -> bool:
    """Check if we need to ask user to disambiguate between candidates"""
    if len(candidates) > 1:
        return True
    if len(candidates) == 1 and confidence < 0.7:
        return True
    return False


######### Automation Detection #########

def find_automation_options(step_ids: List[str]) -> List[Dict]:
    """Find automation options for given steps"""
    automation_options = []
    
    for sid in step_ids:
        rec = get_step_record(sid)
        if rec and rec.get("automatable") and rec.get("automation_step"):
            automation_options.append({
                "step_id": sid,
                "step_name": rec["name"],
                "automation_step": rec["automation_step"],
            })
    
    return automation_options


######### Workflow Context Building #########

def build_workflow_context(
    intent: str,
    matched_steps: List[Dict],
    focus_step: Optional[Dict],
    anchor_step: Optional[Dict],
    next_steps: List[Dict],
    previous_steps: List[Dict],
    completed_ids: List[str],
    in_progress_ids: List[str],
    needs_disambiguation: bool,
    candidate_steps: List[Dict],
    automation_options: List[Dict],
    user_message: str
) -> str:
    """Build context string for LLM to generate response"""
    
    context_parts = []
    
    # User intent
    context_parts.append(f"User Intent: {intent}")
    context_parts.append(f"User Message: {user_message}")
    
    # For "ask_next_step" intent, ONLY include next steps
    if intent == "ask_next_step":
        if next_steps:
            context_parts.append("\nNext Step(s) to Focus On:")
            for step in next_steps:
                context_parts.append(f"\n{step['id']}: {step['name']}")
                context_parts.append(f"Purpose: {step['purpose']}")
                context_parts.append(f"Description: {step['description']}")
        else:
            context_parts.append("\nNo next steps available - workflow may be complete")
        
        # Add automation if available
        if automation_options:
            context_parts.append("\n🤖 Automation Available:")
            for opt in automation_options:
                context_parts.append(f"  - {opt['step_name']}: Can be automated via '{opt['automation_step']}'")
        
        return "\n".join(context_parts)
    
    # For disambiguation, show candidate options
    if needs_disambiguation and candidate_steps:
        context_parts.append("\n⚠️ Multiple Steps Match - User Needs to Choose:")
        for step in candidate_steps:
            context_parts.append(f"\n{step['id']}: {step['name']}")
            context_parts.append(f"Purpose: {step['purpose']}")
            context_parts.append(f"Description: {step['description']}")
        
        return "\n".join(context_parts)
    
    # For explain_step intent, focus on the specific step
    if intent == "explain_step" and focus_step:
        context_parts.append(f"\nStep Being Asked About:")
        context_parts.append(f"{focus_step['id']}: {focus_step['name']}")
        context_parts.append(f"Purpose: {focus_step['purpose']}")
        context_parts.append(f"Description: {focus_step['description']}")
        
        if focus_step.get('automatable'):
            context_parts.append(f"\n🤖 This step can be automated")
        
        return "\n".join(context_parts)
    
    # For other intents, provide minimal context
    if focus_step:
        context_parts.append(f"\nCurrent Focus: {focus_step['id']} - {focus_step['name']}")
    
    # Only include next steps if relevant to the query
    if next_steps and ("next" in user_message.lower() or "after" in user_message.lower()):
        context_parts.append("\nNext Steps Available:")
        for step in next_steps:
            context_parts.append(f"  - {step['id']}: {step['name']}")
    
    # Only include previous if relevant
    if previous_steps and ("previous" in user_message.lower() or "before" in user_message.lower()):
        context_parts.append("\nPrevious Steps:")
        for step in previous_steps:
            context_parts.append(f"  - {step['id']}: {step['name']}")
    
    # Progress tracking (minimal)
    if completed_ids:
        context_parts.append(f"\nCompleted: {len(completed_ids)} steps")
    
    # Automation options
    if automation_options:
        context_parts.append("\n🤖 Automation Available:")
        for opt in automation_options:
            context_parts.append(f"  - {opt['step_name']}")
    
    return "\n".join(context_parts)


######### LLM Response Generation #########

def generate_llm_response(
    context: str,
    user_message: str,
    needs_disambiguation: bool,
    focus_step: Optional[Dict] = None,
    next_steps: List[Dict] = None,
    candidate_steps: List[Dict] = None
) -> str:
    """Generate natural language response using LLM"""
    
    system_prompt = """You are a helpful, practical data governance assistant.

Your job is to help users navigate governance workflows by:
- Answering their specific question directly and concisely
- Providing step details (purpose + description) ONLY when relevant to the question
- When multiple steps could apply, present options with details
- Being focused and practical

CRITICAL RULES:
1. Answer the ACTUAL QUESTION - don't explain things they didn't ask about
2. If they ask "what's next", ONLY describe the next step (not current or previous steps)
3. If they need to choose between options, present EACH option clearly
4. Keep responses focused and actionable - no unnecessary background

Keep your responses:
- Direct and focused on their question
- Complete when describing a step (include purpose + description)
- Concise - don't explain steps they didn't ask about"""

    response_prompt = f"""Based on this workflow context, respond to the user's question.

{context}

Instructions:
- Answer their SPECIFIC question - don't over-explain
- If they ask "what's next", focus ONLY on the next step(s)
- If disambiguation needed, present the options clearly
- If automation available for the step they're asking about, mention it
- Be concise but complete for what they actually asked

User's Question: {user_message}

Your Response:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": response_prompt}
    ]
    
    answer = cortex_client.chat_completion(
        messages=messages,
        temperature=0,
        max_tokens=1000  # Reduced from 1500 for more concise responses
    )
    
    return answer


######### Main Analysis Function #########

def analyze_user_query(
    user_message: str,
    previous_state: Optional[Dict] = None
) -> Dict:
    """
    Main function to analyze user query and determine response
    
    Returns:
        Dict with analysis results including intent, steps, and context
    """
    # Classify intent
    intent = classify_intent(user_message, previous_state)
    
    # Initialize state
    completed_ids = previous_state.get("last_completed_ids", []) if previous_state else []
    in_progress_ids = previous_state.get("last_in_progress_ids", []) if previous_state else []
    anchor_step_id = previous_state.get("last_anchor_step_id") if previous_state else None
    
    # Find matching steps based on user message
    matched_steps, method, confidence = find_matching_steps(user_message)
    
    # Determine focus step
    focus_step_id = None
    focus_step = None
    
    if matched_steps:
        focus_step_id = matched_steps[0]["id"]
        focus_step = matched_steps[0]
    elif anchor_step_id:
        focus_step_id = anchor_step_id
        focus_step = get_step_record(anchor_step_id)
    
    # Check for disambiguation
    needs_disambiguation = needs_disambiguation_check(matched_steps, confidence)
    candidate_steps = matched_steps if needs_disambiguation else []
    
    # Find next and previous steps
    next_step_ids = []
    next_steps = []
    previous_steps = []
    
    if focus_step_id:
        next_step_ids = find_next_steps(focus_step_id, completed_ids)
        next_steps = [get_step_record(sid) for sid in next_step_ids if get_step_record(sid)]
        
        prev_step_ids = find_previous_steps(focus_step_id)
        previous_steps = [get_step_record(sid) for sid in prev_step_ids if get_step_record(sid)]
    
    # Get anchor step info
    anchor_step = get_step_record(anchor_step_id) if anchor_step_id else None
    
    # Find automation options
    automation_steps_to_check = [focus_step_id] if focus_step_id else []
    automation_steps_to_check.extend(next_step_ids[:2])  # Check next 2 steps too
    automation_options = find_automation_options(automation_steps_to_check)
    
    # Build context and generate response
    context = build_workflow_context(
        intent=intent,
        matched_steps=matched_steps,
        focus_step=focus_step,
        anchor_step=anchor_step,
        next_steps=next_steps,
        previous_steps=previous_steps,
        completed_ids=completed_ids,
        in_progress_ids=in_progress_ids,
        needs_disambiguation=needs_disambiguation,
        candidate_steps=candidate_steps,
        automation_options=automation_options,
        user_message=user_message
    )
    
    answer = generate_llm_response(
        context=context,
        user_message=user_message,
        needs_disambiguation=needs_disambiguation,
        focus_step=focus_step,
        next_steps=next_steps,
        candidate_steps=candidate_steps
    )
    
    return {
        "intent": intent,
        "focus_step_id": focus_step_id,
        "anchor_step_id": focus_step_id or anchor_step_id,  # Update anchor if we found a focus
        "next_step_ids": next_step_ids,
        "matched_steps": matched_steps,
        "candidate_steps": candidate_steps,
        "needs_disambiguation": needs_disambiguation,
        "automation_options": automation_options,
        "answer": answer,
        "completed_ids": completed_ids,
        "in_progress_ids": in_progress_ids,
        "method": method,
        "confidence": confidence
    }