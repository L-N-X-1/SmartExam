#Agent niveau 4 (Analyze) → Compare, Contrast, Differentiate, Classify, Organize...# agents/analyze_agent.py
"""
Agent niveau 4 (Analyze) - Analyse
Verbes : Compare, Contrast, Differentiate, Classify, Organize, Examine
"""

from agents.base_agent import BaseBloomAgent
from core.llm_interface import get_completion

class AnalyzeAgent(BaseBloomAgent):
    """
    Agent spécialisé pour les questions de niveau Analyze (analyse)
    """
    
    def __init__(self):
        bloom_verbs = [
            "Compare", "Contrast", "Differentiate", "Classify",
            "Organize", "Examine", "Analyze", "Categorize",
            "Distinguish", "Investigate", "Relate", "Separate",
            "Order", "Connect", "Divide"
        ]
        super().__init__(bloom_level=4, bloom_verbs=bloom_verbs)
    
    def generate_questions(self, context, num_questions=5, topic=None):
        """
        Génère des questions de niveau Analyze
        """
        print(f"   🔍 Agent Analyze : génération de {num_questions} questions...")
        
        prompt = self._create_prompt(context, num_questions, topic)
        
        try:
            response = get_completion(prompt, temperature=0.7)
            questions = self.parse_llm_response(response)
            
            print(f"   ✅ {len(questions)} questions Analyze générées")
            return questions
            
        except Exception as e:
            print(f"   ❌ Erreur génération : {e}")
            return []