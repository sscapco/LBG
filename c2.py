# API Configuration Constants

# Base API configuration
BASEURL = "YOUR_BASE_URL_HERE"  # Replace with actual base URL
ROOT_CA = "path/to/root_ca.pem"  # Replace with actual certificate path
CLIENT_ID = "YOUR_CLIENT_ID_HERE"  # Replace with actual client ID
CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"  # Replace with actual client secret

# Model configuration
CHAT_MODEL = "vertex_ai/gemini-2.5-flash"
EMBEDDING_MODEL = "vertex_ai/text-embedding-004"
EMBEDDING_DIMENSIONS = 256

# Token limits
MAX_TOKENS = 1000
THINKING_BUDGET_TOKENS = 300

# File paths
EXCEL_PATH = "nodes_edges_governance.xlsx"
EMBED_CACHE_PATH = "step_embeddings_cache.json"