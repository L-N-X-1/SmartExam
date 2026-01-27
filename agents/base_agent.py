# agents/base_agent.py
"""
Classe abstraite pour tous les agents Bloom
Tous les agents héritent de cette classe
"""

from abc import ABC, abstractmethod
from core.rag_engine import get_relevant_context
from core.llm_interface import get_completion

class BaseBloomAgent(ABC):
    """
    Classe de base pour tous les agents Bloom
    Chaque agent doit implémenter generate_questions()
    """
    
    def __init__(self, bloom_level, bloom_verbs):
        """
        INPUT:
            - bloom_level (int) : niveau de Bloom (1-6)
            - bloom_verbs (list) : verbes d'action pour ce niveau
        """
        self.bloom_level = bloom_level
        self.bloom_verbs = bloom_verbs
        self.level_name = self._get_level_name()
    
    def _get_level_name(self):
        """Retourne le nom du niveau Bloom"""
        levels = {
            1: "Remember",
            2: "Understand", 
            3: "Apply",
            4: "Analyze",
            5: "Evaluate",
            6: "Create"
        }
        return levels.get(self.bloom_level, "Unknown")
    
    @abstractmethod
    def generate_questions(self, context, num_questions=5, topic=None):
        """
        Méthode abstraite que tous les agents doivent implémenter
        
        INPUT:
            - context (str) : contexte extrait du RAG
            - num_questions (int) : nombre de questions à générer
            - topic (str) : sujet spécifique (optionnel)
        
        OUTPUT:
            - questions (list) : liste de dict avec {text, type, marks, difficulty}
        """
        pass
    
    def _create_prompt(self, context, num_questions, topic=None):
        """
        Crée le prompt pour le LLM
        """
        verbs_str = ", ".join(self.bloom_verbs[:8])  # Utilise les 8 premiers verbes
        
        topic_instruction = f"sur le sujet '{topic}'" if topic else ""
        
        prompt = f"""Tu es un expert en création de questions d'examen basées sur la Taxonomie de Bloom.

NIVEAU BLOOM : {self.bloom_level} - {self.level_name}
VERBES D'ACTION À UTILISER : {verbs_str}

CONTEXTE DU COURS :
{context}

INSTRUCTIONS :
1. Génère exactement {num_questions} questions de niveau {self.level_name} {topic_instruction}
2. Chaque question DOIT utiliser un des verbes d'action listés
3. Les questions doivent être claires, précises et évaluables
4. Varie les types de questions : QCM, questions courtes, questions ouvertes

FORMAT DE RÉPONSE (JSON) :
[
  {{
    "text": "Question complète ici",
    "type": "mcq/short/open",
    "marks": 2-10,
    "difficulty": "easy/medium/hard",
    "bloom_verb": "verbe utilisé"
  }}
]

IMPORTANT : Réponds UNIQUEMENT avec le JSON, sans texte avant ou après, sans markdown."""

        return prompt
    
    def parse_llm_response(self, response_text):
        """
        Parse la réponse du LLM et extrait les questions
        """
        import json
        import re
        
        # Nettoie la réponse (enlève markdown si présent)
        clean_response = response_text.strip()
        clean_response = re.sub(r'```json\s*', '', clean_response)
        clean_response = re.sub(r'```\s*', '', clean_response)
        
        try:
            questions = json.loads(clean_response)
            
            # Ajoute le niveau Bloom à chaque question
            for q in questions:
                q['bloom_level'] = self.bloom_level
                
            return questions
            
        except json.JSONDecodeError as e:
            print(f"❌ Erreur parsing JSON : {e}")
            print(f"Réponse reçue : {response_text[:200]}...")
            return []