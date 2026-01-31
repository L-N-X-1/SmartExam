# =============================================================================
# SMART EXAM – Validator Agent with LangGraph
# =============================================================================

from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import GROQ_API_KEY, VALIDATION_THRESHOLD
from config.bloom_taxonomy import get_bloom_level


class ValidationState(TypedDict):
    """State for validation workflow"""
    question: Dict
    bloom_level: str
    context: str
    validation_result: Dict
    error: str


class ValidatorAgent:
    """Validator using LangGraph"""
    
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.model = model
        self.client = Groq(api_key=GROQ_API_KEY)
        self.threshold = VALIDATION_THRESHOLD
        self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(ValidationState)
        
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("parse", self._parse_node)
        workflow.add_node("error", self._error_node)
        
        workflow.set_entry_point("validate")
        workflow.add_conditional_edges(
            "validate",
            lambda s: "error" if s.get('error') else "success",
            {"error": "error", "success": "parse"}
        )
        workflow.add_edge("parse", END)
        workflow.add_edge("error", END)
        
        self.graph = workflow.compile()
    
    def _validate_node(self, state: ValidationState) -> ValidationState:
        """Validate a question"""
        try:
            q = state['question']
            bloom_info = get_bloom_level(state['bloom_level']) or {}
            verbs = bloom_info.get("verbs", [])[:5]
            
            prompt = f"""Evaluate this {state['bloom_level']} question.
Question: {q.get('question', '')}
Answer: {q.get('answer', '')}
Explanation: {q.get('explanation', '')}

Verbs for {state['bloom_level']}: {', '.join(verbs)}

Score these (0-20 each): clarity, bloom_alignment, correctness, completeness, appropriateness.

Return JSON:
{{"clarity_score": <0-20>, "bloom_score": <0-20>, "correctness_score": <0-20>, "completeness_score": <0-20>, "appropriateness_score": <0-20>, "total_score": <0-100>, "feedback": "<text>"}}"""
            
            response_text = ""
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert exam question validator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=1000,
                top_p=1,
                stream=True,
                stop=None
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            state['validation_result'] = response_text
        except Exception as e:
            state['error'] = str(e)
        return state
    
    def _parse_node(self, state: ValidationState) -> ValidationState:
        """Parse validation result"""
        try:
            text = state['validation_result']
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                scores = json.loads(text[json_start:json_end])
            else:
                scores = {"total_score": 0}
            
            total = scores.get("total_score", 0)
            state['validation_result'] = {
                "question": state['question'].get("question", ""),
                "score": total,
                "valid": total >= self.threshold,
                "feedback": scores.get("feedback", "")
            }
        except Exception as e:
            state['error'] = f"Parse error: {str(e)}"
        return state
    
    def _error_node(self, state: ValidationState) -> ValidationState:
        """Handle error"""
        state['validation_result'] = {"score": 0, "valid": False}
        return state
    
    def validate_question(self, question: Dict, bloom_level: str, context: str = None) -> Dict:
        """Validate a single question"""
        initial = ValidationState(
            question=question,
            bloom_level=bloom_level,
            context=context or "",
            validation_result={},
            error=""
        )
        result = self.graph.invoke(initial)
        return result.get('validation_result', {})
    
    def validate_exam(self, exam: Dict, context: str = "") -> Dict:
        """Validate entire exam"""
        try:
            questions = exam.get('questions', []) if isinstance(exam, dict) else []
            if not questions:
                return {"valid": False, "message": "No questions in exam"}
            
            validation_results = []
            valid_count = 0
            
            for q in questions:
                # Extract bloom level from question if available
                bloom_level = q.get('bloom_level', 'Remember')
                val_result = self.validate_question(q, bloom_level, context)
                validation_results.append(val_result)
                if val_result.get('valid', False):
                    valid_count += 1
            
            total_score = sum(r.get('score', 0) for r in validation_results) / len(validation_results) if validation_results else 0
            
            return {
                "valid": valid_count >= len(validation_results) * 0.7,  # At least 70% pass
                "total_questions": len(validation_results),
                "valid_questions": valid_count,
                "average_score": round(total_score, 2),
                "details": validation_results
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}
