# agents/analyze_agent.py
"""
Agent niveau 4 (Analyze) - Analyse
Verbes : Compare, Contrast, Differentiate, Classify, Organize, Examine
"""

from agents.base_agent import BaseBloomAgent


class AnalyzeAgent(BaseBloomAgent):
    """
    Agent spécialisé pour les questions de niveau Analyze (niveau 4)
    """

    def __init__(self, llm=None):
        bloom_verbs = [
            "Compare", "Contrast", "Differentiate", "Classify",
            "Organize", "Examine", "Analyze", "Categorize",
            "Distinguish", "Investigate", "Relate", "Separate",
            "Order", "Connect", "Divide"
        ]
        super().__init__(
            bloom_level=4,
            bloom_verbs=bloom_verbs,
            temperature=0.7,
            max_context_chars=600,   # allow slightly more context
            llm=llm
        )

    def generate_questions(self, context: str, num_questions: int = 5, topic: str = None):
        """
        Génère des questions de niveau Analyze en utilisant BaseBloomAgent logic
        """
        print(f"   🔍 Agent Analyze : génération de {num_questions} questions...")

        # Build prompt using BaseBloomAgent
        prompt = self._create_prompt(context, num_questions, topic)

        try:
            # Use unified BaseBloomAgent LLM call (includes retries & token optimization)
            raw_response = self.call_llm(prompt, max_tokens=500)
            questions = self.parse_llm_response(raw_response)

            # clean questions (remove duplicates / invalid verbs)
            questions = self._clean_questions(questions)

            print(f"   ✅ {len(questions)} questions Analyze générées")
            return questions

        except Exception as e:
            print(f"   ❌ Erreur génération : {e}")
            return []
