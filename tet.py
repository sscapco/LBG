# test_cortex_simple.py
import asyncio
from cortex_llm import get_llm_client

async def test():
    print("Testing Cortex Client\n")
    
    # Test sync
    print("1. SYNC TEST:")
    sync_client = get_llm_client(async_mode=False)
    
    try:
        response = sync_client.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'Hello World'"}],
            max_tokens=50
        )
        print(f"✅ Success: {response.choices[0].message.content}\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")
    
    # Test async
    print("2. ASYNC TEST:")
    async_client = get_llm_client(async_mode=True)
    
    try:
        response = await async_client.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'Hello World'"}],
            max_tokens=50
        )
        print(f"✅ Success: {response.choices[0].message.content}\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Test embeddings
    print("3. EMBEDDINGS TEST:")
    try:
        emb_response = sync_client.embeddings.create_embedding(
            model="vertex_ai/text-embedding-004",
            input=["test text"]
        )
        print(f"✅ Embedding vector length: {len(emb_response.data[0].embedding)}\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")

if __name__ == "__main__":
    asyncio.run(test())