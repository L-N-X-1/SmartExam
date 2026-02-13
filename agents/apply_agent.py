from typing import List
from agents.base_agent import BaseAgent


class ApplyAgent(BaseAgent):
    """
    Level 3: Apply - Using knowledge in new situations.
    Forces verbs like Solve, Use, Implement, Demonstrate, Execute.
    Includes practical exercises and calculations.
    """
    
    def __init__(self, llm, **kwargs):
        allowed_verbs = [
            "Solve", "Use", "Implement", "Demonstrate", "Execute",
            "Apply", "Calculate", "Perform", "Practice", "Operate",
            "Show how", "Carry out", "Utilize", "Employ", "Run",
            "Complete", "Illustrate how", "Put into practice", "Work through"
        ]
        
        super().__init__(
            llm=llm,
            name="ApplyAgent",
            bloom_level="Level 3: Apply",
            allowed_verbs=allowed_verbs,
            temperature=0.4,
            **kwargs
        )
    
    def generate_questions(self, rag_chunks: List[dict], num: int = 8) -> dict:
        """
        Generate apply-level questions with practical exercises.
        """
        return super().generate_questions(rag_chunks, num)