"""
Main LangGraph workflow for governance Q&A pipeline
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from state import GovernanceState, create_initial_state, state_to_session_state, session_state_to_previous_state
from query_analyzer import analyze_query_node
from automation_handler import automation_handler_node
from response_generator import response_generator_node


# Session storage (in-memory for now, can be replaced with Redis/DB)
SESSION_STORE: Dict[str, Dict[str, Any]] = {}


def should_run_automation(state: GovernanceState) -> str:
    """
    Routing function: Determine if we should run automation
    
    Args:
        state: Current state
        
    Returns:
        "automation" if automation request, else "response"
    """
    intent = state.get("intent")
    
    if intent == "automation_request":
        return "automation"
    else:
        return "response"


def create_governance_graph() -> StateGraph:
    """
    Create the LangGraph workflow for governance Q&A
    
    Workflow:
    1. analyze_query -> Classify intent and identify steps
    2. Conditional:
       - If automation_request -> automation_handler -> response_generator
       - Else -> response_generator
    3. END
    
    Returns:
        Compiled StateGraph
    """
    # Create graph
    workflow = StateGraph(GovernanceState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("automation_handler", automation_handler_node)
    workflow.add_node("response_generator", response_generator_node)
    
    # Define edges
    workflow.set_entry_point("analyze_query")
    
    # Conditional routing after query analysis
    workflow.add_conditional_edges(
        "analyze_query",
        should_run_automation,
        {
            "automation": "automation_handler",
            "response": "response_generator"
        }
    )
    
    # Automation handler goes to response generator
    workflow.add_edge("automation_handler", "response_generator")
    
    # Response generator ends the workflow
    workflow.add_edge("response_generator", END)
    
    # Compile graph
    return workflow.compile()


# Create global graph instance
governance_graph = create_governance_graph()


async def process_query(
    user_message: str,
    session_id: str = "default"
) -> str:
    """
    Process a user query through the governance workflow
    
    Args:
        user_message: User's input message
        session_id: Session identifier for conversation memory
        
    Returns:
        Answer string for the user
    """
    print("\n" + "=" * 80)
    print(f"🎯 GOVERNANCE QUERY (session: {session_id})")
    print(f"📝 {user_message}")
    print("=" * 80)
    
    # Retrieve previous session state
    previous_session = SESSION_STORE.get(session_id)
    previous_state = session_state_to_previous_state(previous_session)
    
    # Create initial state
    initial_state = create_initial_state(user_message, session_id)
    initial_state["previous_state"] = previous_state
    
    # Run graph
    final_state = await governance_graph.ainvoke(initial_state)
    
    # Extract answer
    answer = final_state.get("answer", "I'm not sure how to respond to that.")
    
    # Update session storage
    session_state = state_to_session_state(final_state)
    SESSION_STORE[session_id] = session_state
    
    print("\n" + "=" * 80)
    print("✅ Query processed successfully")
    print("=" * 80 + "\n")
    
    return answer


def process_query_sync(
    user_message: str,
    session_id: str = "default"
) -> str:
    """
    Synchronous version of process_query
    
    Args:
        user_message: User's input message
        session_id: Session identifier for conversation memory
        
    Returns:
        Answer string for the user
    """
    print("\n" + "=" * 80)
    print(f"🎯 GOVERNANCE QUERY (session: {session_id})")
    print(f"📝 {user_message}")
    print("=" * 80)
    
    # Retrieve previous session state
    previous_session = SESSION_STORE.get(session_id)
    previous_state = session_state_to_previous_state(previous_session)
    
    # Create initial state
    initial_state = create_initial_state(user_message, session_id)
    initial_state["previous_state"] = previous_state
    
    # Run graph synchronously
    final_state = governance_graph.invoke(initial_state)
    
    # Extract answer
    answer = final_state.get("answer", "I'm not sure how to respond to that.")
    
    # Update session storage
    session_state = state_to_session_state(final_state)
    SESSION_STORE[session_id] = session_state
    
    print("\n" + "=" * 80)
    print("✅ Query processed successfully")
    print("=" * 80 + "\n")
    
    return answer


def clear_session(session_id: str = "default"):
    """
    Clear session state
    
    Args:
        session_id: Session to clear
    """
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
        print(f"✅ Cleared session: {session_id}")


def get_session_state(session_id: str = "default") -> Dict[str, Any]:
    """
    Get current session state
    
    Args:
        session_id: Session to retrieve
        
    Returns:
        Session state dictionary
    """
    return SESSION_STORE.get(session_id, {})