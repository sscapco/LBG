import asyncio
from cortex_llm import get_llm_client

async def test():
    print("Testing Cortex client...")
    
    # Test sync client
    print("\n1. Testing SYNC client:")
    sync_client = get_llm_client(async_mode=False)
    
    try:
        response = sync_client.chat.completions.create(
            model="vertex_ai/gemini-2.5-flash",
            messages=[{"role": "user", "content": "Say 'Hello World'"}],
            max_tokens=50
        )
        print(f"✅ Sync response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Sync failed: {e}")
    
    # Test async client
    print("\n2. Testing ASYNC client:")
    async_client = get_llm_client(async_mode=True)
    
    try:
        response = await async_client.chat.completions.create(
            model="vertex_ai/gemini-2.5-flash",
            messages=[{"role": "user", "content": "Say 'Hello World'"}],
            max_tokens=50
        )
        print(f"✅ Async response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Async failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())