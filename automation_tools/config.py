import yaml

class Settings:
    def __init__(self, path="automation_tools/app.yaml"):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # LLM provider
        self.llm_provider = cfg.get("llm_provider", "cortex")
        self.llm_model = cfg.get("llm_model", "vertex_ai/gemini-2.5-flash")
        
        # Cortex API credentials - HARDCODED
        # Import from parent constants.py
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import constants
        
        self.cortex_baseurl = constants.BASEURL
        self.cortex_client_id = constants.CLIENT_ID
        self.cortex_client_secret = constants.CLIENT_SECRET
        self.cortex_root_ca = constants.ROOT_CA
        
        # Vector store (if needed)
        self.vector_store = cfg.get("vector_store", "lancedb")
        self.store_path = cfg.get("store_path", "./gov_db")

settings = Settings()
