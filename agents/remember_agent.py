# agents/remember_agent.py

"""
Agent niveau 1 (Remember) - Rappel de faits
Verbes : List, Define, Recall, Name, Identify, Label, State, Describe
"""

from agents.base_agent import BaseBloomAgent


class RememberAgent(BaseBloomAgent):
    """
    Agent spécialisé pour les questions de niveau Remember (niveau 1)
    """

    def __init__(self, llm):
        if llm is None:
            raise ValueError("LLM client must be provided to RememberAgent")

        bloom_verbs = [
            "List", "Define", "Recall", "Name", "Identify",
            "Label", "State", "Describe", "Match", "Select",
            "Cite", "Enumerate", "Tell", "Show", "Recognize"
        ]

        super().__init__(
            bloom_level=1,
            bloom_verbs=bloom_verbs,
            temperature=0.3,   
            max_context_chars=400,
            llm=llm,
            debug=False
        )

    def generate_questions(self, context: str, num_questions: int = 5, topic: str = None):
        """
        Génère des questions de niveau Remember en utilisant BaseBloomAgent
        """
        print(f"   📝 Agent Remember : génération de {num_questions} questions...")

        prompt = self._create_prompt(context, num_questions, topic)

        try:
            raw_response = self.call_llm(prompt, max_tokens=400)
            questions = self.parse_llm_response(raw_response)
            questions = self._clean_questions(questions)
            print(f"   ✅ {len(questions)} questions Remember générées")
            return questions

        except Exception as e:
            print(f"   ❌ Erreur génération : {e}")
            return []
