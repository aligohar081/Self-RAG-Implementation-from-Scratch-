import os
from dotenv import load_dotenv

load_dotenv()

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Use a currently supported model - CHANGE THIS LINE:
GROQ_MODEL = "llama-3.3-70b-versatile"  # or "llama-3.1-8b-instant" for faster responses

# Paths
DATA_PATH = "./data"
VECTOR_STORE_PATH = "./vector_store"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Configuration
TOP_K_RETRIEVAL = 5
RELEVANCE_THRESHOLD = 0.6

# Hallucination Check
MAX_REGENERATION_ATTEMPTS = 3

# Web Search
USE_WEB_SEARCH_FALLBACK = True