# =============================================================================
# SMART EXAM – Analyze Agent (Bloom Level 4) with LangGraph
# =============================================================================

from typing import List, Dict
from .base_agent import BaseAgent


class AnalyzeAgent(BaseAgent):
    """Analyze level agent (Bloom Level 4)"""
    
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        super().__init__("Analyze", model)
    
    def _get_bloom_instructions(self) -> str:
        return """Generate analysis questions that:
- Ask students to BREAK DOWN, EXAMINE, CATEGORIZE, or DIFFERENTIATE concepts
- Require identifying patterns, relationships, or structure
- Compare and contrast different ideas"""
    
    def generate_questions(self, context: str, num_questions: int = 5, topic: str = None) -> List[Dict]:
        return self.invoke_workflow(context, num_questions, topic)
