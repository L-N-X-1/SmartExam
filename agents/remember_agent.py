# =============================================================================
# SMART EXAM – Remember Agent (Bloom Level 1) with LangGraph
# =============================================================================

from typing import List, Dict
from .base_agent import BaseAgent


class RememberAgent(BaseAgent):
    """Remember level agent (Bloom Level 1)"""
    
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        super().__init__("Remember", model)
    
    def _get_bloom_instructions(self) -> str:
        return """Generate ONLY factual recall questions that:
- Ask to LIST, DEFINE, NAME, IDENTIFY, or RECALL specific facts
- Have clear, unambiguous correct answers
- Focus on key terms, definitions, facts from the material"""
    
    def generate_questions(self, context: str, num_questions: int = 5, topic: str = None) -> List[Dict]:
        return self.invoke_workflow(context, num_questions, topic)
