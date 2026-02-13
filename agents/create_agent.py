from typing import List
from agents.base_agent import BaseAgent


class CreateAgent(BaseAgent):
    """
    Level 6: Create - Producing new or original work.
    Forces verbs like Design, Invent, Propose, Create, Construct.
    Most open-ended and creative questions requiring synthesis.
    """
    
    def __init__(self, llm, **kwargs):
        allowed_verbs = [
            "Design", "Invent", "Propose", "Create", "Construct",
            "Develop", "Formulate", "Plan", "Build", "Generate",
            "Compose", "Devise", "Originate", "Conceive", "Produce",
            "Make", "Synthesize", "Combine", "Integrate", "Imagine",
            "Hypothesize", "Theorize", "Model", "Scheme"
        ]
        
        super().__init__(
            llm=llm,
            name="CreateAgent",
            bloom_level="Level 6: Create",
            allowed_verbs=allowed_verbs,
            temperature=0.7,  # Higher temperature for creativity
            **kwargs
        )
    
    def generate_questions(self, rag_chunks: List[dict], num: int = 5) -> dict:
        """
        Generate create-level questions requiring original work.
        Fewer questions due to complexity and creativity required.
        """
        return super().generate_questions(rag_chunks, num)