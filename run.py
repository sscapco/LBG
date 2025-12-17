"""
CLI runner for the governance pipeline.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

from app import GovernanceApp


async def run_demo():
    """Run demo queries"""
    
    print("\n" + "=" * 80)
    print(" GOVERNANCE Q&A PIPELINE - DEMO MODE")
    print("=" * 80)
    
    app = GovernanceApp()
    
    demos = [
        ("session1", "I'm working on the DOI form, can you remind me what to do?"),
        ("session1", "Yes, can you run the naming check for AL1234.MyProduct as an ODP?"),
        ("session2", "I think I need to log something for this new data thing—where do I start?"),
        ("session2", "What comes after DPWG?"),
    ]
    
    for i, (sess, query) in enumerate(demos, 1):
        print(f"\n{'=' * 80}")
        print(f" DEMO {i}/{len(demos)} (session: {sess})")
        print(f"{'=' * 80}")
        
        response = await app.process_query(query, session_id=sess)
        
        if i < len(demos):
            print("\n⏳ Next demo in 2 seconds...")
            await asyncio.sleep(2)
    
    print("\n" + "=" * 80)
    print(" ✅ ALL DEMOS COMPLETE")
    print("=" * 80 + "\n")


async def run_interactive():
    """Run interactive mode"""
    
    print("\n" + "=" * 80)
    print(" GOVERNANCE Q&A PIPELINE - INTERACTIVE MODE")
    print(" Type 'quit' or 'exit' to stop")
    print(" Type 'clear' to reset conversation")
    print("=" * 80 + "\n")
    
    app = GovernanceApp()
    session_id = "interactive"
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!\n")
                break
            
            if user_input.lower() == "clear":
                app.clear_session(session_id)
                print("✅ Conversation cleared\n")
                continue
            
            response = await app.process_query(user_input, session_id=session_id)
            print(f"\n🤖 Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()


async def run_single_query(query: str):
    """Run a single query"""
    app = GovernanceApp()
    response = await app.process_query(query)
    print(f"\n{response}\n")


async def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "demo":
            await run_demo()
        elif mode == "interactive" or mode == "i":
            await run_interactive()
        elif mode == "query":
            if len(sys.argv) > 2:
                query = " ".join(sys.argv[2:])
                await run_single_query(query)
            else:
                print("Usage: python run.py query <your question>")
        else:
            print("Usage:")
            print("  python run.py demo           - Run demo queries")
            print("  python run.py interactive    - Interactive mode")
            print("  python run.py query <text>   - Single query")
    else:
        # Default to demo
        await run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
