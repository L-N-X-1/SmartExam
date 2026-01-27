# config/settings.py
from dotenv import load_dotenv
import os

load_dotenv()

# === CLÉS API ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROK_API_KEY = os.getenv("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")
GROQ_API_KEY = os.getenv("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")  # ← AJOUTE CECI

# === MODÈLES IA ===
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # ← AJOUTE CECI

# === CHEMINS DE DONNÉES ===
UPLOAD_FOLDER = "data/uploads/"
PROCESSED_FOLDER = "data/processed/"
VECTOR_STORE_PATH = "data/vector_store/"
EXAMS_OUTPUT = "exams/"

# === PARAMÈTRES GLOBAUX ===
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VALIDATION_THRESHOLD = 85
MAX_QUESTIONS_PER_LEVEL = 20

# Création automatique des dossiers
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(EXAMS_OUTPUT, exist_ok=True)