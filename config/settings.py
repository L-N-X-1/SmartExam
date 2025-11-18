# =============================================================================
# SMART EXAM – Générateur intelligent d'examens basé sur la Taxonomie de Bloom
# Fichier : config/settings.py
# À QUOI SERT CE FICHIER :
# Chargement des variables d'environnement (.env) avec python-dotenv.
# Contient toutes les clés API et configurations globales du projet :
# - Clés API (OpenAI, Anthropic, Grok)
# - Modèles LLM et d'embedding
# - Chemins de stockage
# Tâche assignée → Groupe 1 (RAG & Config)
# =============================================================================

from dotenv import load_dotenv
import os

# Chargement du fichier .env (à la racine du projet)
load_dotenv()

# === CLÉS API ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

# === MODÈLES IA ===
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # par défaut si pas défini dans .env
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# === CHEMINS DE DONNÉES ===
UPLOAD_FOLDER = "data/uploads/"
PROCESSED_FOLDER = "data/processed/"
VECTOR_STORE_PATH = "data/vector_store/"
EXAMS_OUTPUT = "exams/"

# === PARAMÈTRES GLOBAUX ===
CHUNK_SIZE = 1000          # taille des chunks pour le RAG
CHUNK_OVERLAP = 200        # overlap entre chunks
VALIDATION_THRESHOLD = 85  # score minimum pour accepter une question (sur 100)
MAX_QUESTIONS_PER_LEVEL = 20

# Création automatique des dossiers si ils n'existent pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(EXAMS_OUTPUT, exist_ok=True)