from typing import List
from agents.base_agent import BaseAgent


class UnderstandAgent(BaseAgent):
    """
    Level 2: Understand - Comprehension and meaning.
    Forces verbs like Explain, Summarize, Describe, Interpret, Compare.
    Focuses on explaining concepts in own words.
    """
    
    def __init__(self, llm, **kwargs):
        allowed_verbs = [
            "Explain", "Summarize", "Describe", "Interpret", "Compare",
            "Contrast", "Discuss", "Illustrate", "Paraphrase", "Classify",
            "Outline", "Relate", "Distinguish", "Predict", "Translate",
            "Demonstrate understanding", "Clarify", "Express", "Indicate"
        ]
        
        super().__init__(
            llm=llm,
            name="UnderstandAgent",
            bloom_level="Level 2: Understand",
            allowed_verbs=allowed_verbs,
            temperature=0.3,
            **kwargs
        )
    
    def generate_questions(self, rag_chunks: List[dict], num: int = 10) -> dict:
        """
        Generate understand-level questions requiring comprehension.
        """
        return super().generate_questions(rag_chunks, num)