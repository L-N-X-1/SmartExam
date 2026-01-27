#Agent niveau 6 (Create) → Design, Invent, Propose a new, Create, Develop a plan...# agents/create_agent.py
"""
Agent niveau 6 (Create) - Création
Verbes : Design, Invent, Propose, Create, Develop, Formulate, Construct
"""

from agents.base_agent import BaseBloomAgent
from core.llm_interface import get_completion

class CreateAgent(BaseBloomAgent):
    """
    Agent spécialisé pour les questions de niveau Create (création)
    """
    
    def __init__(self):
        bloom_verbs = [
            "Design", "Invent", "Propose", "Create", "Develop",
            "Formulate", "Construct", "Plan", "Generate", "Compose",
            "Devise", "Build", "Produce", "Assemble", "Imagine"
        ]
        super().__init__(bloom_level=6, bloom_verbs=bloom_verbs)
    
    def generate_questions(self, context, num_questions=5, topic=None):
        """
        Génère des questions de niveau Create
        """
        print(f"   🎨 Agent Create : génération de {num_questions} questions...")
        
        prompt = self._create_prompt(context, num_questions, topic)
        
        try:
            response = get_completion(prompt, temperature=0.8)  # Plus créatif
            questions = self.parse_llm_response(response)
            
            print(f"   ✅ {len(questions)} questions Create générées")
            return questions
            
        except Exception as e:
            print(f"   ❌ Erreur génération : {e}")
            return []