from dotenv import load_dotenv
import os

# Load .env from project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

# -------------------- LLM CONFIG --------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "my_model")  
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

# API keys for cloud providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Map provider to API key dynamically
LLM_API_KEYS = {
    "openai": OPENAI_API_KEY,
    "groq": GROQ_API_KEY,
    "openrouter": OPENROUTER_API_KEY,
    "huggingface": HF_TOKEN,
    "local": None
}

# -------------------- PATHS --------------------
UPLOAD_FOLDER = "data/uploads/"
PROCESSED_FOLDER = "data/processed/"
VECTOR_STORE_PATH = "data/vector_store/"
EXAMS_OUTPUT = "exams/"

for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, VECTOR_STORE_PATH, EXAMS_OUTPUT]:
    os.makedirs(folder, exist_ok=True)

# -------------------- RAG / PROCESSING --------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
MAX_QUESTIONS_PER_LEVEL = int(os.getenv("MAX_QUESTIONS_PER_LEVEL", 20))

# -------------------- HELPER FUNCTIONS --------------------
def get_active_llm_model() -> str:
    """Return the model name for the active provider."""
    return LLM_MODEL

def get_llm_api_key() -> str:
    """Return API key for active provider (None for local)."""
    return LLM_API_KEYS.get(LLM_PROVIDER)
