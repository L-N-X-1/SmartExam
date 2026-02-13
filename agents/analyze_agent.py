from typing import List
from agents.base_agent import BaseAgent


class AnalyzeAgent(BaseAgent):
    """
    Level 4: Analyze - Breaking down information into parts.
    Forces verbs like Compare, Contrast, Differentiate, Classify, Examine.
    Focuses on identifying relationships and patterns.
    """
    
    def __init__(self, llm, **kwargs):
        allowed_verbs = [
            "Compare", "Contrast", "Differentiate", "Classify", "Examine",
            "Analyze", "Break down", "Separate", "Distinguish", "Investigate",
            "Categorize", "Deconstruct", "Dissect", "Inspect", "Scrutinize",
            "Organize", "Attribute", "Question", "Probe", "Test"
        ]
        
        super().__init__(
            llm=llm,
            name="AnalyzeAgent",
            bloom_level="Level 4: Analyze",
            allowed_verbs=allowed_verbs,
            temperature=0.5,
            **kwargs
        )
    
    def generate_questions(self, rag_chunks: List[dict], num: int = 8) -> dict:
        """
        Generate analyze-level questions requiring breakdown of information.
        """
        return super().generate_questions(rag_chunks, num)