#Agent niveau 5 (Evaluate) → Justify, Critique, Assess, Defend, Recommend...# agents/evaluate_agent.py
"""
Agent niveau 5 (Evaluate) - Évaluation
Verbes : Justify, Critique, Assess, Defend, Recommend, Judge, Argue
"""

from agents.base_agent import BaseBloomAgent
from core.llm_interface import get_completion

class EvaluateAgent(BaseBloomAgent):
    """
    Agent spécialisé pour les questions de niveau Evaluate (évaluation)
    """
    
    def __init__(self):
        bloom_verbs = [
            "Justify", "Critique", "Assess", "Defend", "Recommend",
            "Judge", "Argue", "Evaluate", "Support", "Conclude",
            "Appraise", "Criticize", "Prioritize", "Rate", "Validate"
        ]
        super().__init__(bloom_level=5, bloom_verbs=bloom_verbs)
    
    def generate_questions(self, context, num_questions=5, topic=None):
        """
        Génère des questions de niveau Evaluate
        """
        print(f"   ⚖️ Agent Evaluate : génération de {num_questions} questions...")
        
        prompt = self._create_prompt(context, num_questions, topic)
        
        try:
            response = get_completion(prompt, temperature=0.7)
            questions = self.parse_llm_response(response)
            
            print(f"   ✅ {len(questions)} questions Evaluate générées")
            return questions
            
        except Exception as e:
            print(f"   ❌ Erreur génération : {e}")
            return []