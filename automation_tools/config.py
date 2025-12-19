import yaml
import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv(override=True)

class Settings:
    def __init__(self, path="automation_tools/app.yaml"):
        # Try to load YAML config
        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            # Fallback to defaults if no YAML
            cfg = {}
        
        # Use Gemini as the LLM provider
        self.llm_provider = cfg.get("llm_provider", "gemini")
        self.llm_model = cfg.get("llm_model", "vertex_ai/gemini-2.5-flash")
        
        # Embedding provider (not used for naming validation)
        self.embedding_provider = cfg.get("embedding_provider", "gemini")
        self.embedding_model = cfg.get("embedding_model", "vertex_ai/text-embedding-004")
        
        # Gemini/Cortex API credentials (from constants.py)
        # These will be used by llms.py to call cortex_connection
        self.gemini_base_url = os.getenv("BASEURL", "")
        self.gemini_client_id = os.getenv("CLIENT_ID", "")
        self.gemini_client_secret = os.getenv("CLIENT_SECRET", "")
        
        # Legacy Azure fields (kept for compatibility but not used)
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
        self.azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "")
        
        # Vector store (not used for naming validation)
        self.vector_store = cfg.get("vector_store", "lancedb")
        self.store_path = cfg.get("store_path", "./gov_db")

settings = Settings()