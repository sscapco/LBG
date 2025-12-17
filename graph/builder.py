"""
LangGraph construction.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import GovernanceGraphState
from graph.nodes import GovernanceNodes


def create_governance_graph(nodes: GovernanceNodes):
    """Build the governance workflow graph"""
    
    workflow = StateGraph(GovernanceGraphState)
    
    # Add nodes
    workflow.add_node("parse_intent", nodes.parse_intent_node)
    workflow.add_node("map_steps", nodes.map_steps_node)
    workflow.add_node("compute_workflow", nodes.compute_workflow_node)
    workflow.add_node("generate_answer", nodes.generate_answer_node)
    workflow.add_node("check_automation", nodes.check_automation_node)
    workflow.add_node("run_automation", nodes.run_automation_node)
    
    # Define flow
    workflow.set_entry_point("parse_intent")
    workflow.add_edge("parse_intent", "map_steps")
    workflow.add_edge("map_steps", "compute_workflow")
    workflow.add_edge("compute_workflow", "generate_answer")
    workflow.add_edge("generate_answer", "check_automation")
    
    # Conditional: run automation or end?
    def should_run_automation(state: GovernanceGraphState) -> str:
        if state["should_run_automation"] and state.get("automation_params"):
            return "run_automation"
        return "end"
    
    workflow.add_conditional_edges(
        "check_automation",
        should_run_automation,
        {"run_automation": "run_automation", "end": END}
    )
    
    workflow.add_edge("run_automation", END)
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
