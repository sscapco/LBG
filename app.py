"""
Main Governance Application.
Production-ready governance Q&A pipeline using LangGraph + Cortex.
"""
from typing import Dict, Optional
from langchain_core.messages import HumanMessage

from config import validate_all_config, CortexConfig, WorkflowConfig
from core import get_cortex_llm, get_cortex_embeddings, EmbeddingManager
from data import GovernanceDataLoader, EmbeddingCache
from workflow import IntentParser, StepMapper, WorkflowComputer, AutomationDetector
from graph import GovernanceNodes, create_governance_graph


class GovernanceApp:
    """
    Main application class for governance Q&A.
    
    Usage:
        app = GovernanceApp()
        response = await app.process_query("What should I do next?", session_id="user123")
    """
    
    def __init__(self):
        print("\n" + "=" * 80)
        print(" INITIALIZING GOVERNANCE PIPELINE")
        print("=" * 80)
        
        # Validate configuration
        validate_all_config()
        
        # Initialize core components
        print("\n📦 Initializing components...")
        self.llm = get_cortex_llm(
            temperature=WorkflowConfig.TEMPERATURE,
            max_tokens=WorkflowConfig.INTENT_PARSING_MAX_TOKENS
        )
        self.embeddings_client = get_cortex_embeddings()
        self.embedding_manager = EmbeddingManager(self.embeddings_client)
        
        # Load data
        print("\n📊 Loading governance data...")
        self.data_loader = GovernanceDataLoader()
        
        # Initialize embedding cache
        print("\n🔢 Setting up embeddings...")
        self.embedding_cache = EmbeddingCache(
            self.embedding_manager,
            self.data_loader
        )
        
        # Initialize workflow components
        print("\n⚙️ Setting up workflow components...")
        self.intent_parser = IntentParser(self.llm)
        self.step_mapper = StepMapper(
            self.data_loader,
            self.embedding_manager,
            self.embedding_cache
        )
        self.workflow_computer = WorkflowComputer(self.data_loader)
        self.automation_detector = AutomationDetector()
        
        # Create graph
        print("\n🔀 Building LangGraph workflow...")
        self.nodes = GovernanceNodes(
            self.intent_parser,
            self.step_mapper,
            self.workflow_computer,
            self.automation_detector
        )
        self.graph = create_governance_graph(self.nodes)
        
        # Session state storage
        self.session_states: Dict[str, Dict] = {}
        
        print("\n✅ Governance pipeline initialized successfully!")
        print("=" * 80 + "\n")
    
    async def process_query(
        self,
        user_message: str,
        session_id: str = "default"
    ) -> str:
        """
        Process a governance query.
        
        Args:
            user_message: The user's question or statement
            session_id: Session identifier for conversation context
        
        Returns:
            str: The assistant's response
        """
        print(f"\n{'=' * 80}")
        print(f" QUERY (session: {session_id})")
        print(f" {user_message}")
        print(f"{'=' * 80}")
        
        # Get previous state
        previous_state = self.session_states.get(session_id, {})
        
        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "user_message": user_message,
            "session_id": session_id,
            "previous_state": previous_state if previous_state else None,
            "parsed_intent": None,
            "mapped_steps": None,
            "workflow_state": None,
            "answer": None,
            "updated_state": None,
            "should_run_automation": False,
            "automation_params": None,
            "automation_result": None,
        }
        
        # Run the graph
        config = {"configurable": {"thread_id": session_id}}
        final_state = await self.graph.ainvoke(initial_state, config)
        
        # Update session state
        if final_state["updated_state"]:
            self.session_states[session_id] = final_state["updated_state"]
        
        answer = final_state["answer"]
        
        print(f"\n{'=' * 80}")
        print(f" RESPONSE:")
        print(f" {answer}")
        print(f"{'=' * 80}\n")
        
        return answer
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.session_states:
            del self.session_states[session_id]
            print(f"✅ Cleared session: {session_id}")
    
    def get_session_state(self, session_id: str) -> Optional[Dict]:
        """Get current session state"""
        return self.session_states.get(session_id)


# Singleton instance
_app_instance: Optional[GovernanceApp] = None


def get_app() -> GovernanceApp:
    """Get or create the singleton app instance"""
    global _app_instance
    if _app_instance is None:
        _app_instance = GovernanceApp()
    return _app_instance
