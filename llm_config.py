"""
LLM Configuration
Switch between Azure OpenAI and Cortex API
"""
import os
from dotenv import load_dotenv

load_dotenv()


from cortex_llm import get_llm_client

# Cortex config
DEPLOYMENT_NAME = "vertex_ai/gemini-2.5-flash"
EMBEDDING_MODEL = "vertex_ai/text-embedding-004"

def get_sync_client():
    return get_llm_client(async_mode=False)

def get_async_client():
    return get_llm_client(async_mode=True)


# Export the clients
def initialize_clients():
    """Initialize and return both sync and async clients"""
    sync_client = get_sync_client()
    async_client = get_async_client()
    
    print(f"   Model: {DEPLOYMENT_NAME}")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    
    return sync_client, async_client