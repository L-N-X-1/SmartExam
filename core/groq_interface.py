# core/groq_interface.py
"""
Interface pour appeler Groq API
"""

from groq import Groq
import os
from dotenv import load_dotenv

# Force le rechargement du .env
load_dotenv(override=True)

# Récupère la clé
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')

# Debug : affiche les 20 premiers caractères
if GROQ_API_KEY:
    print(f"✅ Clé Groq chargée : {GROQ_API_KEY[:20]}...")
else:
    print("❌ GROQ_API_KEY non trouvée dans .env")

# Initialise le client
if not GROQ_API_KEY or GROQ_API_KEY == "":
    print("⚠️  GROQ_API_KEY non configurée")
    client = None
else:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print(f"✅ Client Groq initialisé avec {GROQ_MODEL}")
    except Exception as e:
        print(f"❌ Erreur init Groq : {e}")
        client = None

def get_groq_completion(prompt, system_message=None, temperature=0.7, max_tokens=2000):
    """
    Appelle Groq pour obtenir une réponse
    """
    if client is None:
        raise ValueError(
            "❌ Groq n'est pas configuré.\n"
            "Vérifiez que GROQ_API_KEY est dans votre fichier .env"
        )
    
    try:
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"❌ Erreur Groq API : {e}")
        raise

def rag_query(question, context, language="français"):
    """
    Répond à une question en utilisant le contexte RAG
    """
    system_message = f"""Tu es un assistant pédagogique expert. 
Tu réponds aux questions en te basant UNIQUEMENT sur le contexte fourni.
Si l'information n'est pas dans le contexte, dis-le clairement.
Réponds en {language}, de manière claire et structurée."""

    prompt = f"""CONTEXTE DU COURS :
{context}

QUESTION DE L'ÉTUDIANT :
{question}

INSTRUCTIONS :
1. Réponds UNIQUEMENT en te basant sur le contexte fourni
2. Si l'info n'est pas dans le contexte, dis "Je ne trouve pas cette information dans le document"
3. Sois précis et cite des passages du contexte si nécessaire
4. Structure ta réponse avec des bullets points si pertinent

RÉPONSE :"""

    return get_groq_completion(prompt, system_message, temperature=0.3)