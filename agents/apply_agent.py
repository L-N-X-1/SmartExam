# agents/apply_agent.py
"""
Agent niveau 3 (Apply) - Application
Verbes : Solve, Use, Implement, Calculate, Demonstrate, Execute, Apply
"""

from agents.base_agent import BaseBloomAgent
from core.llm_interface import get_completion

class ApplyAgent(BaseBloomAgent):
    """
    Agent spécialisé pour les questions de niveau Apply (application)
    """
    
    def __init__(self):
        bloom_verbs = [
            "Solve", "Use", "Implement", "Calculate", "Demonstrate",
            "Execute", "Apply", "Construct", "Operate", "Practice",
            "Compute", "Modify", "Prepare", "Produce", "Show"
        ]
        super().__init__(bloom_level=3, bloom_verbs=bloom_verbs)
    
    def generate_questions(self, context, num_questions=5, topic=None):
        """
        Génère des questions de niveau Apply
        """
        print(f"   🔧 Agent Apply : génération de {num_questions} questions...")
        
        prompt = self._create_prompt(context, num_questions, topic)
        
        try:
            response = get_completion(prompt, temperature=0.7)
            questions = self.parse_llm_response(response)
            
            print(f"   ✅ {len(questions)} questions Apply générées")
            return questions
            
        except Exception as e:
            print(f"   ❌ Erreur génération : {e}")
            return []