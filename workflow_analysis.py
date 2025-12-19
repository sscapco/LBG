import json
import re
from typing import List, Dict, Optional, Tuple
from cortex_connection import cortex_client
from governance_data import (
    nodes_df, edges_df, get_step_record, map_text_to_step_id,
    step_index, ORDERED_STEP_IDS
)
from automation_registry2 import get_automation_info


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
    max_candidates: int = 5  # Increased from 3
) -> Tuple[List[Dict], str, float]:
    """
    Find steps matching the user's message
    
    Returns:
        (matched_steps, method, confidence)
    """
    # Try direct matching first
    step_id, method, confidence = map_text_to_step_id(user_message, emb_threshold)
    
    if step_id:
        rec = get_step_record(step_id)
        if rec:
            return [rec], method, confidence
    
    # For ambiguous queries, find multiple matching steps using embeddings
    from governance_data import STEP_EMBEDDINGS, cosine_similarity, cortex_client
    
    # Check for common ambiguous patterns
    ambiguous_patterns = [
        ("security", ["security", "e2e information security"]),
        ("cloud", ["cloud", "e2e cloud"]),
        ("architecture", ["architecture", "e2e architecture"]),
        ("test", ["testing", "e2e testing"]),
    ]
    
    # Check if this is an ambiguous query
    msg_lower = user_message.lower()
    for pattern, keywords in ambiguous_patterns:
        if pattern in msg_lower:
            # Find all steps matching these keywords
            candidates = []
            query_emb = cortex_client.get_embedding(user_message)
            
            for sid, data in STEP_EMBEDDINGS.items():
                sim = cosine_similarity(query_emb, data["embedding"])
                if sim >= (emb_threshold - 0.1):  # Lower threshold for ambiguous queries
                    rec = get_step_record(sid)
                    if rec:
                        candidates.append((rec, sim))
            
            if candidates:
                # Sort by similarity and return top candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                matched = [c[0] for c in candidates[:max_candidates]]
                avg_conf = sum(c[1] for c in candidates[:max_candidates]) / len(matched)
                return matched, "embedding_multi_match", avg_conf
    
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
    # If we found multiple candidates with good confidence, show them all
    if len(candidates) > 1 and confidence > 0.4:
        return True
    # If single candidate but low confidence, ask for clarification
    if len(candidates) == 1 and confidence < 0.6:
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
    """
    Generate natural language response
    
    For single-step queries: Return exact description from Excel
    For disambiguation: Present all options with exact descriptions
    For automation: Include what it does and how to use it
    """
    
    # For "what's next" queries - return exact next step details
    if "next" in user_message.lower() and next_steps:
        if len(next_steps) == 1:
            step = next_steps[0]
            response = f"The next step is **{step['id']}: {step['name']}**.\n\n"
            response += f"**Purpose:** {step['purpose']}\n\n"
            response += f"**Description:**\n{step['description']}"
            
            # Add automation info if available
            if step.get('automatable') and step.get('automation_step'):
                auto_info = get_automation_info(step['automation_step'])
                if auto_info:
                    response += f"\n\n🤖 **Automation Available:** {auto_info['display_name']}\n"
                    response += f"{auto_info['description']}\n"
                    response += f"To use it, ask me to run the automation with the required parameters."
            
            return response
        else:
            # Multiple next steps
            response = "There are multiple possible next steps:\n\n"
            for step in next_steps:
                response += f"**{step['id']}: {step['name']}**\n"
                response += f"{step['purpose']}\n\n"
            response += "Which would you like to know more about?"
            return response
    
    # For disambiguation - show all candidates with full details
    if needs_disambiguation and candidate_steps:
        response = "I found multiple steps that might match. Here are the options:\n\n"
        for step in candidate_steps:
            response += f"**{step['id']}: {step['name']}**\n"
            response += f"**Purpose:** {step['purpose']}\n"
            response += f"**Description:**\n{step['description']}\n\n"
        response += "Which one are you asking about?"
        return response
    
    # For specific step queries - return exact description
    if focus_step:
        response = f"**{focus_step['id']}: {focus_step['name']}**\n\n"
        response += f"**Purpose:** {focus_step['purpose']}\n\n"
        response += f"**Description:**\n{focus_step['description']}"
        
        # Add automation info if available
        if focus_step.get('automatable') and focus_step.get('automation_step'):
            auto_info = get_automation_info(focus_step['automation_step'])
            if auto_info:
                response += f"\n\n🤖 **Automation Available:** {auto_info['display_name']}\n"
                response += f"{auto_info['description']}\n"
                response += f"To use it, you can say: 'Run {auto_info['display_name'].lower()} for [your parameters]'"
        
        return response
    
    # Fallback to LLM for complex queries
    system_prompt = """You are a data governance assistant. Answer the user's question based on the context provided.
    
Be direct and helpful. If describing steps, use the exact descriptions provided in the context."""

    response_prompt = f"""Context:\n{context}\n\nUser Question: {user_message}\n\nProvide a helpful answer:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": response_prompt}
    ]
    
    answer = cortex_client.chat_completion(
        messages=messages,
        temperature=0,
        max_tokens=1000
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