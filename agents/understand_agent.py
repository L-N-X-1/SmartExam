# agents/understand_agent.py
"""
Agent niveau 2 (Understand) - Compréhension
Verbes : Explain, Summarize, Describe, Interpret, Give examples, Paraphrase
"""

from agents.base_agent import BaseBloomAgent
from core.llm_interface import get_completion

class UnderstandAgent(BaseBloomAgent):
    """
    Agent spécialisé pour les questions de niveau Understand (compréhension)
    """
    
    def __init__(self):
        bloom_verbs = [
            "Explain", "Summarize", "Describe", "Interpret", 
            "Give examples", "Paraphrase", "Classify", "Compare",
            "Illustrate", "Infer", "Discuss", "Predict", "Restate"
        ]
        super().__init__(bloom_level=2, bloom_verbs=bloom_verbs)
    
    def generate_questions(self, context, num_questions=5, topic=None):
        """
        Génère des questions de niveau Understand
        """
        print(f"   💡 Agent Understand : génération de {num_questions} questions...")
        
        prompt = self._create_prompt(context, num_questions, topic)
        
        try:
            response = get_completion(prompt, temperature=0.7)
            questions = self.parse_llm_response(response)
            
            print(f"   ✅ {len(questions)} questions Understand générées")
            return questions
            
        except Exception as e:
            print(f"   ❌ Erreur génération : {e}")
            return []