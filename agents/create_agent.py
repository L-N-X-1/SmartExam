# agents/create_agent.py

"""
Agent niveau 6 (Create) - Création
Verbes : Design, Invent, Propose, Create, Develop, Formulate, Construct
"""

from agents.base_agent import BaseBloomAgent


class CreateAgent(BaseBloomAgent):
    """
    Agent spécialisé pour les questions de niveau Create (niveau 6)
    """

    def __init__(self, llm=None):
        bloom_verbs = [
            "Design", "Invent", "Propose", "Create", "Develop",
            "Formulate", "Construct", "Plan", "Generate", "Compose",
            "Devise", "Build", "Produce", "Assemble", "Imagine"
        ]

        super().__init__(
            bloom_level=6,
            bloom_verbs=bloom_verbs,
            temperature=0.8,  
            max_context_chars=600,
            llm=llm
        )

    def generate_questions(self, context: str, num_questions: int = 5, topic: str = None):
        """
        Génère des questions de niveau Create en utilisant BaseBloomAgent
        """
        print(f"   ✨ Agent Create : génération de {num_questions} questions...")

        prompt = self._create_prompt(context, num_questions, topic)

        try:
            raw_response = self.call_llm(prompt, max_tokens=600)
            questions = self.parse_llm_response(raw_response)
            questions = self._clean_questions(questions)
            print(f"   ✅ {len(questions)} questions Create générées")
            return questions

        except Exception as e:
            print(f"   ❌ Erreur génération : {e}")
            return []
