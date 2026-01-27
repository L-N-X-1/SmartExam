# agents/remember_agent.py
"""
Agent niveau 1 (Remember) - Rappel de faits
Verbes : List, Define, Recall, Name, Identify, Label, State, Describe
"""

from agents.base_agent import BaseBloomAgent
from core.llm_interface import get_completion

class RememberAgent(BaseBloomAgent):
    """
    Agent spécialisé pour les questions de niveau Remember (rappel)
    """
    
    def __init__(self):
        bloom_verbs = [
            "List", "Define", "Recall", "Name", "Identify", 
            "Label", "State", "Describe", "Match", "Select",
            "Cite", "Enumerate", "Tell", "Show", "Recognize"
        ]
        super().__init__(bloom_level=1, bloom_verbs=bloom_verbs)
    
    def generate_questions(self, context, num_questions=5, topic=None):
        """
        Génère des questions de niveau Remember
        
        INPUT:
            - context (str) : extrait du cours
            - num_questions (int) : nombre de questions
            - topic (str) : sujet optionnel
        
        OUTPUT:
            - questions (list) : liste de questions
        """
        print(f"   🧠 Agent Remember : génération de {num_questions} questions...")
        
        # Crée le prompt spécifique
        prompt = self._create_prompt(context, num_questions, topic)
        
        # Appelle le LLM
        try:
            response = get_completion(prompt, temperature=0.7)
            questions = self.parse_llm_response(response)
            
            print(f"   ✅ {len(questions)} questions Remember générées")
            return questions
            
        except Exception as e:
            print(f"   ❌ Erreur génération : {e}")
            return []