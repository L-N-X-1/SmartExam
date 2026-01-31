# =============================================================================
# SMART EXAM – Apply Agent (Bloom Level 3) with LangGraph
# =============================================================================

from typing import List, Dict
from .base_agent import BaseAgent


class ApplyAgent(BaseAgent):
    """Apply level agent (Bloom Level 3)"""
    
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        super().__init__("Apply", model)
    
    def _get_bloom_instructions(self) -> str:
        return """Generate application questions that:
- Ask students to USE, DEMONSTRATE, SOLVE, or PRACTICE concepts
- Require applying knowledge to new situations
- Include practical examples or scenarios"""
    
    def generate_questions(self, context: str, num_questions: int = 5, topic: str = None) -> List[Dict]:
        return self.invoke_workflow(context, num_questions, topic)
