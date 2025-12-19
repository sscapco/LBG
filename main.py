"""
Main application entry point for Governance Q&A Pipeline
"""
import asyncio
from workflow import process_query, process_query_sync, clear_session


async def main_async():
    """
    Async demo of the governance Q&A pipeline
    """
    print("\n" + "=" * 80)
    print("🏛️ GOVERNANCE Q&A PIPELINE - LANGGRAPH VERSION")
    print("=" * 80)
    
    # Demo conversations
    demos = [
        ("session_flow", "I'm working on the DOI form, can you remind me what to do?"),
        ("session_flow", "Can you run the naming validation for AL12345.CustomerData as ODP?"),
        ("session_new", "I think I need to log something for this new data thing—where do I start?"),
        ("session_new", "Tell me more about S1"),
    ]
    
    for i, (sess, query) in enumerate(demos, 1):
        print(f"\n{'=' * 80}")
        print(f"DEMO {i} (session: {sess})")
        print(f"{'=' * 80}\n")
        
        response = await process_query(query, session_id=sess)
        print(f"\n💬 ANSWER:\n{response}\n")
        
        if i < len(demos):
            print("\n⏸️ Next demo...\n")
            await asyncio.sleep(1)
    
    print("\n" + "=" * 80)
    print("✅ DEMOS COMPLETE")
    print("=" * 80)


def main_sync():
    """
    Synchronous demo of the governance Q&A pipeline
    """
    print("\n" + "=" * 80)
    print("🏛️ GOVERNANCE Q&A PIPELINE - LANGGRAPH VERSION")
    print("=" * 80)
    
    # Demo conversations
    demos = [
        ("session_flow", "I've finished my DPWG presentation and got endorsement from DPC. What should I do next?"),
        ("session_flow", "Can you tell me what I need to do for the DOI step?"),
        ("session_new", "Run the naming check. The name is AL12345.CustomerData and it's an ODP"),
        ("session_new", "Do I need to do the security bits or the cloud checks—or are they the same thing?"),
    ]
    
    for i, (sess, query) in enumerate(demos, 1):
        print(f"\n{'=' * 80}")
        print(f"DEMO {i} (session: {sess})")
        print(f"{'=' * 80}\n")
        
        response = process_query_sync(query, session_id=sess)
        print(f"\n💬 ANSWER:\n{response}\n")
        
        if i < len(demos):
            print("\n⏸️ Next demo...\n")
    
    print("\n" + "=" * 80)
    print("✅ DEMOS COMPLETE")
    print("=" * 80)


def interactive_mode():
    """
    Interactive mode for testing queries
    """
    print("\n" + "=" * 80)
    print("🏛️ GOVERNANCE Q&A PIPELINE - INTERACTIVE MODE")
    print("=" * 80)
    print("\nType 'quit' or 'exit' to stop")
    print("Type 'clear' to clear session\n")
    
    session_id = "interactive"
    
    while True:
        try:
            user_input = input("\n📝 Your query: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"]:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == "clear":
                clear_session(session_id)
                continue
            
            response = process_query_sync(user_input, session_id=session_id)
            print(f"\n💬 ANSWER:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    import sys
    
    # Check for mode argument
    mode = sys.argv[1] if len(sys.argv) > 1 else "sync"
    
    if mode == "async":
        asyncio.run(main_async())
    elif mode == "interactive":
        interactive_mode()
    else:
        main_sync()
