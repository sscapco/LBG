import yaml
import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv(override=True)

class Settings:
    def __init__(self, path="config/app.yaml"):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        self.llm_provider = cfg.get("llm_provider")
        self.llm_model = cfg.get("llm_model")
        self.embedding_provider = cfg.get("embedding_provider")
        self.embedding_model = cfg.get("embedding_model")
        # LLM Azure-specific
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        # Embeddings (Azure OpenAI)
        self.embedding_provider = cfg.get("embedding_provider", "azure_openai")
        self.azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
         # **Vector store (needed for your indexer)**
        self.vector_store = cfg.get("vector_store", "lancedb")
        self.store_path = cfg.get("store_path", "./gov_db")

settings = Settings()
