# =============================================================================
# SMART EXAM – Understand Agent (Bloom Level 2) with LangGraph
# =============================================================================

from typing import List, Dict
from .base_agent import BaseAgent


class UnderstandAgent(BaseAgent):
    """Understand level agent (Bloom Level 2)"""
    
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        super().__init__("Understand", model)
    
    def _get_bloom_instructions(self) -> str:
        return """Generate comprehension questions that:
- Ask students to EXPLAIN, SUMMARIZE, DESCRIBE, or INTERPRET concepts
- Require understanding but not application
- Test ability to rephrase or reorganize information"""
    
    def generate_questions(self, context: str, num_questions: int = 5, topic: str = None) -> List[Dict]:
        return self.invoke_workflow(context, num_questions, topic)
