# agents/evaluate_agent.py

"""
Agent niveau 5 (Evaluate) - Évaluation
Verbes : Justify, Critique, Assess, Defend, Recommend, Judge, Argue
"""

from agents.base_agent import BaseBloomAgent


class EvaluateAgent(BaseBloomAgent):
    """
    Agent spécialisé pour les questions de niveau Evaluate (niveau 5)
    """

    def __init__(self, llm):
        if llm is None:
            raise ValueError("LLM client must be provided to EvaluateAgent")

        bloom_verbs = [
            "Justify", "Critique", "Assess", "Defend", "Recommend",
            "Judge", "Argue", "Evaluate", "Support", "Conclude",
            "Appraise", "Criticize", "Prioritize", "Rate", "Validate"
        ]

        super().__init__(
            bloom_level=5,
            bloom_verbs=bloom_verbs,
            temperature=0.7,   
            max_context_chars=600,
            llm=llm,
            debug=False
        )

    def generate_questions(self, context: str, num_questions: int = 5, topic: str = None):
        """
        Génère des questions de niveau Evaluate en utilisant BaseBloomAgent
        """
        print(f"   🧠 Agent Evaluate : génération de {num_questions} questions...")

        prompt = self._create_prompt(context, num_questions, topic)

        try:
            raw_response = self.call_llm(prompt, max_tokens=600)
            questions = self.parse_llm_response(raw_response)
            questions = self._clean_questions(questions)
            print(f"   ✅ {len(questions)} questions Evaluate générées")
            return questions

        except Exception as e:
            print(f"   ❌ Erreur génération : {e}")
            return []
