# agents/apply_agent.py
"""
Agent niveau 3 (Apply) - Application
Verbes : Solve, Use, Implement, Calculate, Demonstrate, Execute, Apply
"""

from agents.base_agent import BaseBloomAgent


class ApplyAgent(BaseBloomAgent):
    """
    Agent spécialisé pour les questions de niveau Apply (niveau 3)
    """

    def __init__(self, llm=None):
        bloom_verbs = [
            "Solve", "Use", "Implement", "Calculate", "Demonstrate",
            "Execute", "Apply", "Construct", "Operate", "Practice",
            "Compute", "Modify", "Prepare", "Produce", "Show"
        ]
        super().__init__(
            bloom_level=3,
            bloom_verbs=bloom_verbs,
            temperature=0.6,
            max_context_chars=550,
            llm=llm
        )

    def generate_questions(self, context: str, num_questions: int = 5, topic: str = None):
        """
        Génère des questions de niveau Apply en utilisant BaseBloomAgent logic
        """
        print(f"   🔧 Agent Apply : génération de {num_questions} questions...")

        # Build prompt using BaseBloomAgent
        prompt = self._create_prompt(context, num_questions, topic)

        try:
            # Use unified BaseBloomAgent LLM call (includes retries & token optimization)
            raw_response = self.call_llm(prompt, max_tokens=500)
            questions = self.parse_llm_response(raw_response)

            # clean questions (remove duplicates / invalid verbs)
            questions = self._clean_questions(questions)

            print(f"   ✅ {len(questions)} questions Apply générées")
            return questions

        except Exception as e:
            print(f"   ❌ Erreur génération : {e}")
            return []
