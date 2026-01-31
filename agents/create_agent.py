# =============================================================================
# SMART EXAM – Create Agent (Bloom Level 6) with LangGraph
# =============================================================================

from typing import List, Dict
from .base_agent import BaseAgent


class CreateAgent(BaseAgent):
    """Create level agent (Bloom Level 6)"""
    
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        super().__init__("Create", model)
    
    def _get_bloom_instructions(self) -> str:
        return """Generate synthesis questions that:
- Ask students to DESIGN, CONSTRUCT, COMBINE, or INVENT new concepts
- Require combining elements to form new wholes
- Test ability to generate new ideas or solutions"""
    
    def generate_questions(self, context: str, num_questions: int = 5, topic: str = None) -> List[Dict]:
        return self.invoke_workflow(context, num_questions, topic)
