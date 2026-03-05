import os

# -----------------------------
# LLM (local vLLM OpenAI-compatible)
# -----------------------------
LOCAL_OPENAI_BASE_URL = os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:8000/v1")
LOCAL_OPENAI_MODEL = os.getenv("LOCAL_OPENAI_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LOCAL_OPENAI_API_KEY = os.getenv("LOCAL_OPENAI_API_KEY", "EMPTY")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# -----------------------------
# Chroma / Embeddings
# -----------------------------
PERSIST_DIR = os.getenv("PERSIST_DIR", "./LG_chromadb")
COLLECTION = os.getenv("COLLECTION", "Knowledge_base2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Tavily (web search)
# -----------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_SEARCH_CALLS_PER_RUN = int(os.getenv("TAVILY_MAX_SEARCH_CALLS_PER_RUN", "1"))
TAVILY_MAX_RESULTS_PER_CALL = int(os.getenv("TAVILY_MAX_RESULTS_PER_CALL", "1"))

# -----------------------------
# KB weak-hit threshold (Chroma distance; lower is better)
# Tune if needed
# -----------------------------
KB_WEAK_DISTANCE_THRESHOLD = float(os.getenv("KB_WEAK_DISTANCE_THRESHOLD", "0.55"))
