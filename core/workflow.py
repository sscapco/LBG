import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from .automation_handler import automation_handler_node
from .query_analyzer import analyze_query_node
from .response_generator import response_generator_node
from .state import GovernanceState, create_initial_state, session_state_to_previous_state, state_to_session_state


logging.getLogger("langgraph").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

SESSION_STORE: Dict[str, Dict[str, Any]] = {}


def create_governance_graph() -> StateGraph:
    workflow = StateGraph(GovernanceState)
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("automation_handler", automation_handler_node)
    workflow.add_node("response_generator", response_generator_node)

    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "automation_handler")
    workflow.add_edge("automation_handler", "response_generator")
    workflow.add_edge("response_generator", END)

    return workflow.compile()


governance_graph = create_governance_graph()


async def process_query(user_message: str, session_id: str = "default") -> str:
    previous_session = SESSION_STORE.get(session_id)
    previous_state = session_state_to_previous_state(previous_session)

    initial_state = create_initial_state(user_message, session_id)
    initial_state["previous_state"] = previous_state

    final_state = await governance_graph.ainvoke(initial_state)
    answer = final_state.get("answer", "I'm not sure how to respond to that.")

    SESSION_STORE[session_id] = state_to_session_state(final_state)
    return answer


def process_query_sync(user_message: str, session_id: str = "default") -> str:
    previous_session = SESSION_STORE.get(session_id)
    previous_state = session_state_to_previous_state(previous_session)

    initial_state = create_initial_state(user_message, session_id)
    initial_state["previous_state"] = previous_state

    final_state = governance_graph.invoke(initial_state)
    answer = final_state.get("answer", "I'm not sure how to respond to that.")

    SESSION_STORE[session_id] = state_to_session_state(final_state)
    return answer


def clear_session(session_id: str = "default"):
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]


def get_session_state(session_id: str = "default") -> Dict[str, Any]:
    return SESSION_STORE.get(session_id, {})

