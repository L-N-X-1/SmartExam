from typing import List
from agents.base_agent import BaseAgent


class EvaluateAgent(BaseAgent):
    """
    Level 5: Evaluate - Making judgments and decisions.
    Forces verbs like Justify, Critique, Assess, Defend, Judge.
    Requires evaluation based on criteria or standards.
    """
    
    def __init__(self, llm, **kwargs):
        allowed_verbs = [
            "Justify", "Critique", "Assess", "Defend", "Judge",
            "Evaluate", "Argue", "Validate", "Rate", "Rank",
            "Appraise", "Critically evaluate", "Determine value", "Recommend",
            "Support", "Refute", "Debate", "Prioritize", "Measure",
            "Verify", "Endorse", "Conclude", "Resolve"
        ]
        
        super().__init__(
            llm=llm,
            name="EvaluateAgent",
            bloom_level="Level 5: Evaluate",
            allowed_verbs=allowed_verbs,
            temperature=0.6,
            **kwargs
        )
    
    def generate_questions(self, rag_chunks: List[dict], num: int = 6) -> dict:
        """
        Generate evaluate-level questions requiring judgment.
        """
        return super().generate_questions(rag_chunks, num)