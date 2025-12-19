import yaml
import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv(override=True)

class Settings:
    def __init__(self, path="automation_tools/app.yaml"):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # LLM provider
        self.llm_provider = cfg.get("llm_provider", "cortex")
        self.llm_model = cfg.get("llm_model", "vertex_ai/gemini-2.5-flash")
        
        # Cortex API credentials
        self.cortex_baseurl = os.getenv("CORTEX_BASEURL")
        self.cortex_client_id = os.getenv("CORTEX_CLIENT_ID")
        self.cortex_client_secret = os.getenv("CORTEX_CLIENT_SECRET")
        self.cortex_root_ca = os.getenv("CORTEX_ROOT_CA")
        
        # Vector store (if needed)
        self.vector_store = cfg.get("vector_store", "lancedb")
        self.store_path = cfg.get("store_path", "./gov_db")

settings = Settings()