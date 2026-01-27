# core/llm_interface.py
"""
Wrapper pour appeler les LLM (OpenAI, Anthropic, Grok)
Centralise tous les appels API
"""

from config.settings import OPENAI_API_KEY, LLM_MODEL
from openai import OpenAI

# Initialise le client OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
def generate_exam_questions(course_name, config):
    questions = []
    for level, count in config['distribution'].items():
        for i in range(count):
            questions.append({
                "text": f"Question mock {i+1} pour niveau {level}",
                "type": "mcq" if level < 4 else "open",
                "bloom_level": level,
                "difficulty": config['difficulty'],
                "marks": 2,
                "options": ["A", "B", "C", "D"] if level < 4 else None,
                "correct_answer": "A" if level < 4 else None
            })
    return questions

def get_completion(prompt, temperature=0.7, max_tokens=2000):
    """
    Appelle le LLM pour obtenir une réponse
    
    INPUT:
        - prompt (str) : le prompt à envoyer
        - temperature (float) : créativité (0-1)
        - max_tokens (int) : longueur max de la réponse
    
    OUTPUT:
        - response (str) : réponse du LLM
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Tu es un assistant expert en pédagogie et en création de questions d'examen basées sur la Taxonomie de Bloom."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"❌ Erreur appel LLM : {e}")
        raise

def get_embedding(text):
    """
    Obtient l'embedding d'un texte (pour le RAG)
    Utilise OpenAI Embeddings comme backup.
    """
    from config.settings import EMBEDDING_MODEL
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Erreur lors de la récupération de l'embedding: {e}")
        return None
