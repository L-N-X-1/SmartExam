# agents/understand_agent.py

"""
Agent niveau 2 (Understand) - Compréhension
Verbes : Explain, Summarize, Describe, Interpret, Give examples, Paraphrase
"""

from agents.base_agent import BaseBloomAgent


class UnderstandAgent(BaseBloomAgent):
    """
    Agent spécialisé pour les questions de niveau Understand (niveau 2)
    """

    def __init__(self, llm):
        if llm is None:
            raise ValueError("LLM client must be provided to UnderstandAgent")

        bloom_verbs = [
            "Explain", "Summarize", "Describe", "Interpret",
            "Give examples", "Paraphrase", "Classify", "Compare",
            "Illustrate", "Infer", "Discuss", "Predict", "Restate"
        ]

        super().__init__(
            bloom_level=2,
            bloom_verbs=bloom_verbs,
            temperature=0.5,   
            max_context_chars=450,
            llm=llm,
            debug=False
        )

    def generate_questions(self, context: str, num_questions: int = 5, topic: str = None):
        """
        Génère des questions de niveau Understand en utilisant BaseBloomAgent
        """
        print(f"   📘 Agent Understand : génération de {num_questions} questions...")

        prompt = self._create_prompt(context, num_questions, topic)

        try:
            raw_response = self.call_llm(prompt, max_tokens=400)
            questions = self.parse_llm_response(raw_response)
            questions = self._clean_questions(questions)
            print(f"   ✅ {len(questions)} questions Understand générées")
            return questions

        except Exception as e:
            print(f"   ❌ Erreur génération : {e}")
            return []
