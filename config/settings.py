from dotenv import load_dotenv
import os

# Charger le fichier .env
dotenv_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    ".env"
)
load_dotenv(dotenv_path)

# =========================
# Fournisseur LLM
# =========================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# =========================
# Clés API (depuis .env)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# Modèles IA
# =========================
OPENAI_MODEL = "gpt-4o-mini"
GROQ_MODEL = "llama-3.3-70b-versatile"

# =========================
# Embeddings
# =========================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =========================
# Chemins de données
# =========================
UPLOAD_FOLDER = "data/uploads/"
PROCESSED_FOLDER = "data/processed/"
VECTOR_STORE_PATH = "data/vector_store/"
EXAMS_OUTPUT = "exams/"

# =========================
# Paramètres globaux
# =========================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VALIDATION_THRESHOLD = 85
MAX_QUESTIONS_PER_LEVEL = 20

# =========================
# Création automatique des dossiers
# =========================
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(EXAMS_OUTPUT, exist_ok=True)

# =========================
# DEBUG (temporaire)
# =========================
print("DEBUG LLM_PROVIDER :", LLM_PROVIDER)
print("DEBUG GROQ KEY :", "OK" if GROQ_API_KEY else "❌ MANQUANTE")
