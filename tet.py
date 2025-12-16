import asyncio
from agents import OpenAIChatCompletionsModel
from cortex_llm import get_llm_client

async def test_cortex_with_agents_sdk():
    """Test if Cortex wrapper works with OpenAI Agents SDK"""
    
    print("Initializing Cortex async client...")
    async_client = get_llm_client(async_mode=True)
    
    print("Creating OpenAIChatCompletionsModel with Cortex client...")
    try:
        shared_model = OpenAIChatCompletionsModel(
            model="vertex_ai/gemini-2.5-flash",
            openai_client=async_client,
        )
        print("✅ Model created successfully!")
        
        # Try a simple chat completion
        print("\nTesting chat completion...")
        response = await async_client.chat.completions.create(
            model="vertex_ai/gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "Say hello in JSON format"}
            ],
            response_format={"type": "json_object"},
            max_tokens=100
        )
        
        print(f"✅ Response received: {response.choices[0].message.content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_cortex_with_agents_sdk())
    if success:
        print("\n✅ Cortex wrapper is compatible with OpenAI Agents SDK!")
    else:
        print("\n❌ Compatibility issues detected")