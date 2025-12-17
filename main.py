import asyncio
from governance_graph import run_governance_graph
from session_manager import session_manager
from typing import Optional


async def process_governance_query(
    user_message: str,
    session_id: str = "default_session"
) -> str:
    """
    Process a governance query and return the answer
    
    Args:
        user_message: User's query
        session_id: Session identifier for tracking state across queries
        
    Returns:
        Answer string
    """
    print("\n" + "=" * 80)
    print(f" GOVERNANCE QUERY (session: {session_id})")
    print(f" {user_message}")
    print("=" * 80)
    
    # Get previous state from session manager
    previous_state = session_manager.get_session(session_id)
    
    # Run the governance graph
    answer, updated_state = await run_governance_graph(
        user_message=user_message,
        session_id=session_id,
        previous_state=previous_state
    )
    
    # Update session
    session_manager.update_session(session_id, updated_state)
    
    print("=" * 80 + "\n")
    
    return answer


def process_governance_query_sync(
    user_message: str,
    session_id: str = "default_session"
) -> str:
    """
    Synchronous version of process_governance_query for easier integration
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        process_governance_query(user_message, session_id)
    )


######### Demo / Testing #########

async def run_demo():
    """Run demo queries to test the system"""
    
    print("\n" + "=" * 80)
    print(" GOVERNANCE Q&A PIPELINE WITH LANGGRAPH + GEMINI")
    print("=" * 80)
    
    demos = [
        ("session_flow", "I'm working on the DOI form, can you remind me what to do?"),
        ("session_flow", "Yes, can you run the naming check for me?"),
        ("session_new", "I think I need to log something for this new data thing—where do I start?"),
    ]
    
    for i, (sess, query) in enumerate(demos, 1):
        print(f"\n{'=' * 80}")
        print(f"DEMO {i} (session: {sess})")
        print(f"{'=' * 80}\n")
        
        response = await process_governance_query(query, session_id=sess)
        print(f"\n ANSWER:\n{response}\n")
        
        if i < len(demos):
            print("\n Next demo...\n")
            await asyncio.sleep(1)
    
    print("\n" + "=" * 80)
    print(" DEMOS COMPLETE")
    print("=" * 80)


######### Interactive Mode #########

async def interactive_mode():
    """
    Run in interactive mode for testing
    """
    print("\n" + "=" * 80)
    print(" GOVERNANCE Q&A - INTERACTIVE MODE")
    print(" Type 'quit' to exit, 'clear' to clear session, 'sessions' to list")
    print("=" * 80 + "\n")
    
    session_id = "interactive_session"
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == "clear":
                session_manager.clear_session(session_id)
                print("✅ Session cleared!")
                continue
            
            if user_input.lower() == "sessions":
                sessions = session_manager.list_sessions()
                print("\nActive Sessions:")
                for sid, info in sessions.items():
                    print(f"  - {sid}: {info}")
                continue
            
            # Process query
            answer = await process_governance_query(user_input, session_id)
            print(f"\nAssistant: {answer}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


######### Main Entry Point #########

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            asyncio.run(run_demo())
        elif sys.argv[1] == "interactive":
            asyncio.run(interactive_mode())
        else:
            print("Usage: python main.py [demo|interactive]")
    else:
        # Default: run demo
        asyncio.run(run_demo())


if __name__ == "__main__":
    main()