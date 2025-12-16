from cortex_llm import get_llm_client
    
DEPLOYMENT_NAME = "vertex_ai/gemini-2.5-flash"
EMBEDDING_MODEL = "vertex_ai/text-embedding-004"

def get_sync_client():
    return get_llm_client(async_mode=False)

def get_async_client():
    return get_llm_client(async_mode=True)

# Initialise and return both sync and async clients
def initialize_clients():
    return get_sync_client(), get_async_client()