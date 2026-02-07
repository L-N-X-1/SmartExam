"""
Wrapper pour appeler les LLM (OpenAI, GROQ)
Centralise tous les appels API
"""

import requests

from config.settings import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    EMBEDDING_MODEL
)

# =========================
# OpenAI client
# =========================
from openai import OpenAI
client_openai = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# GROQ – Appel réel via REST
# =========================
def groq_completion(prompt, temperature=0.7, max_tokens=2000):
    """
    Appel réel à l'API GROQ (compatible OpenAI)
    """
    if not GROQ_API_KEY:
        raise ValueError("❌ GROQ_API_KEY est vide ou non défini")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,  # ex: "llama3-70b-8192"
        "messages": [
            {
                "role": "system",
                "content": "Tu es un expert en pédagogie et création de questions d'examen."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    # Lève une erreur si status != 200
    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]


# =========================
# Fonction centrale LLM
# =========================
def get_completion(prompt, temperature=0.7, max_tokens=2000):
    """
    Appelle le LLM configuré (GROQ ou OpenAI)
    """
    if LLM_PROVIDER.lower() == "groq":
        return groq_completion(prompt, temperature, max_tokens)

    # ---- Fallback OpenAI ----
    try:
        response = client_openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert en pédagogie et création de questions d'examen."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Erreur appel OpenAI : {e}")
        raise


# =========================
# Embeddings (RAG)
# =========================
def get_embedding(text):
    """
    Obtient l'embedding d'un texte (utilisé pour FAISS / RAG)
    """
    try:
        response = client_openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    except Exception as e:
        print(f"❌ Erreur lors de la récupération de l'embedding : {e}")
        return None
