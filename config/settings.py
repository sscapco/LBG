"""
Configuration management for the governance pipeline.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)


class CortexConfig:
    """Cortex API configuration"""
    BASE_URL: str = os.getenv("CORTEX_BASE_URL", "")
    CLIENT_ID: str = os.getenv("CORTEX_CLIENT_ID", "")
    CLIENT_SECRET: str = os.getenv("CORTEX_CLIENT_SECRET", "")
    
    CHAT_MODEL: str = os.getenv("CORTEX_CHAT_MODEL", "vertex_ai/gemini-2.5-flash")
    EMBEDDING_MODEL: str = os.getenv("CORTEX_EMBEDDING_MODEL", "vertex_ai/text-embedding-004")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "256"))
    
    # Timeouts
    CHAT_TIMEOUT: int = 60
    EMBEDDING_TIMEOUT: int = 30
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration"""
        if not cls.BASE_URL:
            raise ValueError("CORTEX_BASE_URL not set in environment")
        if not cls.CLIENT_ID:
            raise ValueError("CORTEX_CLIENT_ID not set in environment")
        if not cls.CLIENT_SECRET:
            raise ValueError("CORTEX_CLIENT_SECRET not set in environment")


class DataConfig:
    """Data file configuration"""
    EXCEL_PATH: Path = Path(os.getenv("GOVERNANCE_EXCEL_PATH", "nodes_edges_governance.xlsx"))
    EMBEDDING_CACHE_PATH: Path = CACHE_DIR / os.getenv("EMBEDDING_CACHE_PATH", "step_embeddings_cache.json")
    
    @classmethod
    def validate(cls) -> None:
        """Validate data files exist"""
        if not cls.EXCEL_PATH.exists():
            raise FileNotFoundError(f"Excel file not found: {cls.EXCEL_PATH}")


class WorkflowConfig:
    """Workflow behavior configuration"""
    # Embedding similarity threshold
    EMBEDDING_THRESHOLD: float = 0.5
    
    # Ambiguity detection thresholds
    AMBIGUITY_MIN_SCORE: float = 0.40
    AMBIGUITY_SECONDARY_MIN_SCORE: float = 0.35
    AMBIGUITY_SCORE_DIFF_MAX: float = 0.15
    
    # Semantic candidates
    SEMANTIC_TOP_K: int = 3
    
    # LLM parameters
    INTENT_PARSING_MAX_TOKENS: int = 700
    ANSWER_GENERATION_MAX_TOKENS: int = 900
    TEMPERATURE: float = 0.0


class LogConfig:
    """Logging configuration"""
    LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def validate_all_config():
    """Validate all configuration"""
    CortexConfig.validate()
    DataConfig.validate()
    print("✅ Configuration validated successfully")


if __name__ == "__main__":
    # Test configuration
    try:
        validate_all_config()
        print(f"Cortex Base URL: {CortexConfig.BASE_URL}")
        print(f"Excel Path: {DataConfig.EXCEL_PATH}")
        print(f"Cache Path: {DataConfig.EMBEDDING_CACHE_PATH}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
