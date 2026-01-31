# =============================================================================
# SMART EXAM – LLM Interface
# Fichier : core/llm_interface.py
# À QUOI SERT CE FICHIER :
# Wrapper unique pour appeler les LLM (OpenAI, Anthropic, Grok, Llama).
# Fonctions : get_completion() et get_embedding().
# Centralise toutes les appels API.
# =============================================================================

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from groq import Groq
from sentence_transformers import SentenceTransformer


# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize embedding model
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")


def get_completion(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    model: str = LLM_MODEL
) -> str:
    """
    Appelle Groq API pour obtenir une réponse au prompt.
    
    Args:
        prompt (str): Prompt d'entrée
        temperature (float): Température (0-1)
        max_tokens (int): Nombre maximum de tokens
        model (str): Modèle à utiliser
        
    Returns:
        str: Réponse du modèle
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert educational question generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error calling Groq LLM: {e}")
        return ""


def get_embedding(text: str) -> list:
    """
    Génère un embedding pour un texte.
    
    Args:
        text (str): Texte à encoder
        
    Returns:
        list: Vecteur d'embedding
    """
    
    try:
        embedding = embedding_model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def batch_embeddings(texts: list) -> list:
    """
    Génère des embeddings pour une liste de textes.
    
    Args:
        texts (list): Liste de textes
        
    Returns:
        list: Liste de vecteurs d'embedding
    """
    
    try:
        embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
        return embeddings.tolist()
    
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        return []


def health_check() -> bool:
    """
    Vérifie que les API Groq sont accessibles.
    
    Returns:
        bool: True si OK, False sinon
    """
    
    try:
        # Test Groq connection
        if not GROQ_API_KEY:
            print("❌ GROQ_API_KEY not set")
            return False
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Say 'OK'"}],
            max_tokens=10
        )
        
        print("✓ Groq API: OK")
        return True
    
    except Exception as e:
        print(f"❌ Error in Groq health check: {e}")
        return False
