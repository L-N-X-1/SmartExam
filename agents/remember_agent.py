from typing import List
from agents.base_agent import BaseAgent


class RememberAgent(BaseAgent):
    """
    Level 1: Remember - Basic recall and recognition.
    Forces verbs like List, Define, Recall, Name, Identify, State.
    Generates 10-15 questions focused on factual recall.
    """
    
    def __init__(self, llm, **kwargs):
        allowed_verbs = [
            "List", "Define", "Recall", "Name", "Identify", "State",
            "Recognize", "Label", "Match", "Select", "Locate", "Find",
            "Repeat", "Write", "Quote", "Enumerate", "Specify"
        ]
        
        super().__init__(
            llm=llm,
            name="RememberAgent",
            bloom_level="Level 1: Remember",
            allowed_verbs=allowed_verbs,
            temperature=0.1,  # Lower temperature for factual recall
            **kwargs
        )
    
    def generate_questions(self, rag_chunks: List[dict], num: int = 12) -> dict:
        """
        Generate 10-15 remember-level questions.
        Default to 12 questions if not specified.
        """
        # Ensure we generate between 10-15 questions
        num = max(10, min(15, num))
        return super().generate_questions(rag_chunks, num)