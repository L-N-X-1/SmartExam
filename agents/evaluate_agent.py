# =============================================================================
# SMART EXAM – Evaluate Agent (Bloom Level 5) with LangGraph
# =============================================================================

from typing import List, Dict
from .base_agent import BaseAgent


class EvaluateAgent(BaseAgent):
    """Evaluate level agent (Bloom Level 5)"""
    
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        super().__init__("Evaluate", model)
    
    def _get_bloom_instructions(self) -> str:
        return """Generate evaluation questions that:
- Ask students to JUDGE, CRITIQUE, DEFEND, or JUSTIFY positions
- Require critical thinking and evidence-based reasoning
- Compare using defined criteria"""
    
    def generate_questions(self, context: str, num_questions: int = 5, topic: str = None) -> List[Dict]:
        return self.invoke_workflow(context, num_questions, topic)
