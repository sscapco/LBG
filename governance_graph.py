from typing import TypedDict, Annotated, Sequence, Optional
from langgraph.graph import StateGraph, END
from data_models import GraphState, GovernanceSessionState
from workflow_analysis import analyze_user_query
from automation_handler import handle_automation_in_message
from datetime import datetime
import json


######### Graph Nodes #########

def analyze_query_node(state: GraphState) -> GraphState:
    """
    Analyze user's query to understand intent and find relevant steps
    """
    print("🔍 Analyzing query...")
    
    # Convert previous state to dict if exists
    previous_state_dict = None
    if state.previous_state:
        previous_state_dict = state.previous_state.model_dump()
    
    # Run analysis
    analysis = analyze_user_query(
        user_message=state.user_message,
        previous_state=previous_state_dict
    )
    
    # Update state with analysis results
    return GraphState(
        user_message=state.user_message,
        session_id=state.session_id,
        intent=analysis["intent"],
        focus_step_id=analysis["focus_step_id"],
        anchor_step_id=analysis["anchor_step_id"],
        next_step_ids=analysis["next_step_ids"],
        matched_steps=analysis["matched_steps"],
        candidate_steps=analysis["candidate_steps"],
        needs_disambiguation=analysis["needs_disambiguation"],
        automation_options=analysis["automation_options"],
        answer=analysis["answer"],
        completed_ids=analysis["completed_ids"],
        in_progress_ids=analysis["in_progress_ids"],
        previous_state=state.previous_state
    )


def check_automation_node(state: GraphState) -> GraphState:
    """
    Check if user is requesting automation and execute if applicable
    """
    print("🤖 Checking for automation request...")
    
    # Get automation step from options
    automation_step = None
    if state.automation_options:
        automation_step = state.automation_options[0]["automation_step"]
    
    # Check if message contains automation request
    automation_result = handle_automation_in_message(
        message=state.user_message,
        automation_step=automation_step
    )
    
    if automation_result:
        print(f"   ✓ Automation executed: {automation_result['status']}")
        # Replace answer with automation result
        return GraphState(
            **state.model_dump(),
            automation_requested=True,
            automation_result=automation_result,
            answer=automation_result["formatted_message"]
        )
    else:
        print("   ℹ️ No automation request detected")
        return state


def update_session_node(state: GraphState) -> GraphState:
    """
    Update session state with current workflow position
    """
    print("💾 Updating session state...")
    
    # Get automation info for state
    auto_step_id = state.automation_options[0]["step_id"] if state.automation_options else None
    auto_step_key = state.automation_options[0]["automation_step"] if state.automation_options else None
    
    # Create updated state
    updated_state = GovernanceSessionState(
        last_intent=state.intent,
        last_focus_step_id=state.focus_step_id,
        last_anchor_step_id=state.anchor_step_id,
        last_next_step_ids=state.next_step_ids,
        last_completed_ids=state.completed_ids,
        last_in_progress_ids=state.in_progress_ids,
        last_referenced_ids=[s["id"] for s in state.matched_steps],
        last_automatable_step_id=auto_step_id,
        last_automation_step=auto_step_key,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
    
    return GraphState(
        **state.model_dump(),
        updated_state=updated_state
    )


######### Conditional Edges #########

def should_check_automation(state: GraphState) -> str:
    """Decide if we should check for automation"""
    # Always check for automation after analysis
    return "check_automation"


def should_end(state: GraphState) -> str:
    """Decide if workflow should end"""
    # Always end after updating session
    return "end"


######### Graph Construction #########

def create_governance_graph():
    """
    Create the LangGraph workflow for governance Q&A
    
    Workflow:
    1. Analyze query -> Understand intent, find steps
    2. Check automation -> Detect and execute if requested
    3. Update session -> Save workflow state
    4. End -> Return result
    """
    
    # Create graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("check_automation", check_automation_node)
    workflow.add_node("update_session", update_session_node)
    
    # Set entry point
    workflow.set_entry_point("analyze_query")
    
    # Add edges
    workflow.add_conditional_edges(
        "analyze_query",
        should_check_automation,
        {
            "check_automation": "check_automation"
        }
    )
    
    workflow.add_edge("check_automation", "update_session")
    
    workflow.add_conditional_edges(
        "update_session",
        should_end,
        {
            "end": END
        }
    )
    
    # Compile graph
    app = workflow.compile()
    
    return app


######### Graph Execution #########

async def run_governance_graph(
    user_message: str,
    session_id: str = "default_session",
    previous_state: Optional[GovernanceSessionState] = None
) -> tuple[str, GovernanceSessionState]:
    """
    Run the governance graph for a user query
    
    Args:
        user_message: User's query
        session_id: Session identifier for state tracking
        previous_state: Previous session state if any
        
    Returns:
        (answer, updated_state)
    """
    # Create graph
    app = create_governance_graph()
    
    # Initial state
    initial_state = GraphState(
        user_message=user_message,
        session_id=session_id,
        previous_state=previous_state
    )
    
    # Run graph
    final_state = await app.ainvoke(initial_state)
    
    return final_state["answer"], final_state["updated_state"]


# Synchronous version for compatibility
def run_governance_graph_sync(
    user_message: str,
    session_id: str = "default_session",
    previous_state: Optional[GovernanceSessionState] = None
) -> tuple[str, GovernanceSessionState]:
    """
    Synchronous version of run_governance_graph
    """
    import asyncio
    
    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run async function
    return loop.run_until_complete(
        run_governance_graph(user_message, session_id, previous_state)
    )