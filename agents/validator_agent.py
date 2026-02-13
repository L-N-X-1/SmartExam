import json
import re
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda


class ValidatorAgent(ABC):
    """
    Core validation agent that implements heuristics plus zero-shot LLM scoring.
    Acceptance threshold ≥ 85%.
    
    Validates:
    - Clarity and comprehensibility
    - Bloom level accuracy
    - Context alignment
    - Grammar and structure
    - Cognitive complexity matching
    """
    
    def __init__(
        self,
        llm: Any,
        acceptance_threshold: float = 0.85,
        temperature: float = 0.3,
        debug: bool = False,
        max_retries: int = 2
    ):
        self.llm = llm
        self.acceptance_threshold = acceptance_threshold
        self.temperature = temperature
        self.debug = debug
        self.max_retries = max_retries
        self._last_raw = None
        
        # Create LangGraph node
        self.node = RunnableLambda(self._process_state)
    
    # -------------------------
    # Heuristic validation methods
    # -------------------------
    
    def _check_basic_structure(self, question: str) -> Tuple[bool, Dict]:
        """Basic structural validation."""
        issues = []
        score = 1.0
        
        # Length check
        if len(question.strip()) < 10:
            issues.append("Question too short")
            score -= 0.3
        elif len(question.strip()) > 200:
            issues.append("Question too long")
            score -= 0.2
        
        # Grammar checks (basic)
        if not question[0].isupper():
            issues.append("Missing capitalization")
            score -= 0.1
        
        if not question.endswith('?'):
            issues.append("Missing question mark")
            score -= 0.1
        
        # Check for incomplete sentences
        if question.count('.') > 0 and not question.endswith('.'):
            issues.append("Possible incomplete sentence")
            score -= 0.1
        
        return score >= 0.7, {
            "heuristic_score": score,
            "issues": issues,
            "passed": score >= 0.7
        }
    
    def _check_bloom_verb_alignment(self, question: str, expected_level: str, allowed_verbs: List[str]) -> Tuple[bool, Dict]:
        """Check if question uses appropriate Bloom verbs."""
        question_lower = question.lower().strip()
        
        # Check if starts with allowed verb
        verb_found = None
        for verb in allowed_verbs:
            if question_lower.startswith(verb.lower() + " "):
                verb_found = verb
                break
        
        if not verb_found:
            return False, {
                "heuristic_score": 0.0,
                "verb_found": None,
                "expected_verbs": allowed_verbs,
                "issue": "No allowed Bloom verb found at start"
            }
        
        # Score based on verb appropriateness
        score = 1.0
        issues = []
        
        # Check for multiple cognitive tasks
        cognitive_indicators = ["and", "then", "also", "as well as", "in addition"]
        for indicator in cognitive_indicators:
            if f" {indicator} " in question_lower:
                issues.append(f"Multiple tasks detected: {indicator}")
                score -= 0.3
        
        return score >= 0.7, {
            "heuristic_score": score,
            "verb_found": verb_found,
            "issues": issues,
            "passed": score >= 0.7
        }
    
    def _check_context_alignment(self, question: str, context: str) -> Tuple[bool, Dict]:
        """Check if question can be answered from context."""
        if not context or not context.strip():
            return True, {
                "heuristic_score": 0.5,
                "note": "No context provided for validation"
            }
        
        # Extract key terms from question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'which', 'who'}
        question_terms = question_words - common_words
        
        # Check if question terms appear in context
        context_lower = context.lower()
        matches = 0
        for term in question_terms:
            if term in context_lower:
                matches += 1
        
        # Score based on term overlap
        if len(question_terms) == 0:
            score = 0.5
        else:
            score = matches / len(question_terms)
        
        issues = []
        if score < 0.3:
            issues.append("Low term overlap with context")
        elif score < 0.6:
            issues.append("Moderate term overlap with context")
        
        return score >= 0.4, {
            "heuristic_score": score,
            "term_matches": matches,
            "total_terms": len(question_terms),
            "issues": issues,
            "passed": score >= 0.4
        }
    
    # -------------------------
    # LLM-based validation
    # -------------------------
    
    def _build_validation_prompt(self, question: str, context: str, bloom_level: str) -> str:
        """Build prompt for LLM-based validation."""
        
        return f"""
You are an expert exam question validator. Evaluate the following question based on multiple criteria.

QUESTION: {question}

BLOOM LEVEL: {bloom_level}

CONTEXT:
{context[:1500] if len(context) > 1500 else context}

Evaluate on these criteria (score 0-100 for each):
1. CLARITY: Is the question clear, unambiguous, and well-formed?
2. BLOOM_ALIGNMENT: Does the question match the specified Bloom level?
3. CONTEXT_DEPENDENCY: Can this question be answered ONLY from the provided context?
4. COGNITIVE_COMPLEXITY: Is the cognitive complexity appropriate for the Bloom level?
5. GRAMMAR_STRUCTURE: Is the grammar correct and structure sound?

Return ONLY a JSON object with this format:
{{
  "clarity_score": <0-100>,
  "bloom_alignment_score": <0-100>,
  "context_dependency_score": <0-100>,
  "cognitive_complexity_score": <0-100>,
  "grammar_structure_score": <0-100>,
  "overall_score": <0-100>,
  "strengths": ["list", "of", "strengths"],
  "weaknesses": ["list", "of", "weaknesses"],
  "recommendation": "accept/reject/revise"
}}

No markdown. No explanations. Just the JSON.
"""
    
    def _call_llm(self, prompt: str) -> str:
        """Direct LLM call for validation."""
        try:
            if hasattr(self.llm, 'invoke'):
                messages = [
                    SystemMessage(content="You are an expert exam question validator."),
                    HumanMessage(content=prompt)
                ]
                response = self.llm.invoke(messages)
                
                if hasattr(response, 'content'):
                    return response.content
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
            
            elif callable(self.llm):
                return self.llm(prompt, temperature=self.temperature)
            
            elif hasattr(self.llm, 'generate'):
                result = self.llm.generate(prompt=prompt, temperature=self.temperature)
                if hasattr(result, 'text'):
                    return result.text
                return str(result)
            
            else:
                raise RuntimeError(f"LLM object {type(self.llm)} is not callable")
                
        except Exception as e:
            raise RuntimeError(f"LLM validation call failed: {str(e)}")
    
    def _parse_llm_response(self, raw: str) -> Dict:
        """Parse LLM validation response."""
        self._last_raw = raw
        
        if not raw:
            return {"error": "Empty response"}
        
        # Try to extract JSON
        raw = raw.strip()
        
        # Remove markdown fences
        if "```" in raw:
            fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            if fence:
                raw = fence.group(1).strip()
        
        # Find JSON object
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            raw = json_match.group(0)
        
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {str(e)}", "raw": raw[:200]}
    
    # -------------------------
    # Combined validation logic
    # -------------------------
    
    def _validate_single_question(
        self, 
        question: str, 
        context: str, 
        bloom_level: str,
        allowed_verbs: List[str]
    ) -> Tuple[bool, Dict]:
        """Validate a single question using both heuristics and LLM."""
        
        # 1. Heuristic validation
        basic_score, basic_meta = self._check_basic_structure(question)
        verb_score, verb_meta = self._check_bloom_verb_alignment(question, bloom_level, allowed_verbs)
        context_score, context_meta = self._check_context_alignment(question, context)
        
        heuristic_avg = (basic_meta["heuristic_score"] + verb_meta["heuristic_score"] + context_meta["heuristic_score"]) / 3
        
        # 2. LLM validation
        llm_prompt = self._build_validation_prompt(question, context, bloom_level)
        
        for attempt in range(self.max_retries + 1):
            try:
                llm_raw = self._call_llm(llm_prompt)
                llm_result = self._parse_llm_response(llm_raw)
                
                if "error" not in llm_result and "overall_score" in llm_result:
                    break
            except Exception as e:
                if attempt == self.max_retries:
                    llm_result = {"error": f"LLM validation failed: {str(e)}"}
                else:
                    continue
        else:
            llm_result = {"error": "Max retries exceeded"}
        
        # 3. Combine scores
        if "error" in llm_result:
            # Fallback to heuristic-only
            final_score = heuristic_avg
            validation_meta = {
                "final_score": final_score,
                "heuristic_score": heuristic_avg,
                "llm_score": 0,
                "heuristic_details": {
                    "basic": basic_meta,
                    "verb": verb_meta,
                    "context": context_meta
                },
                "llm_error": llm_result.get("error"),
                "accepted": final_score >= self.acceptance_threshold
            }
        else:
            # Weight LLM more heavily
            llm_score = llm_result["overall_score"] / 100
            final_score = (heuristic_avg * 0.3) + (llm_score * 0.7)
            
            validation_meta = {
                "final_score": final_score,
                "heuristic_score": heuristic_avg,
                "llm_score": llm_score,
                "heuristic_details": {
                    "basic": basic_meta,
                    "verb": verb_meta,
                    "context": context_meta
                },
                "llm_details": llm_result,
                "accepted": final_score >= self.acceptance_threshold
            }
        
        return validation_meta["accepted"], validation_meta
    
    # -------------------------
    # Batch validation
    # -------------------------
    
    def validate(
        self, 
        question: str, 
        context: str, 
        bloom_level: str,
        allowed_verbs: Optional[List[str]] = None
    ) -> Tuple[bool, Dict]:
        """
        Legacy validation method for single question.
        Returns (accepted, metadata)
        """
        if allowed_verbs is None:
            # Default verbs based on bloom level
            verb_map = {
                "remember": ["List", "Define", "Recall", "Name", "Identify"],
                "understand": ["Explain", "Summarize", "Describe", "Interpret"],
                "apply": ["Solve", "Use", "Implement", "Demonstrate"],
                "analyze": ["Compare", "Contrast", "Differentiate", "Classify"],
                "evaluate": ["Justify", "Critique", "Assess", "Defend"],
                "create": ["Design", "Invent", "Propose", "Create"]
            }
            allowed_verbs = verb_map.get(bloom_level.lower(), [])
        
        return self._validate_single_question(question, context, bloom_level, allowed_verbs)
    
    def validate_batch(
        self, 
        questions: List[Dict], 
        context: str, 
        bloom_level: str,
        allowed_verbs: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Validate multiple questions.
        Returns list of validated questions with metadata.
        """
        validated = []
        
        for q in questions:
            if not isinstance(q, dict) or "question" not in q:
                continue
            
            accepted, meta = self.validate(
                q["question"], 
                context, 
                bloom_level,
                allowed_verbs
            )
            
            if accepted:
                validated_q = q.copy()
                validated_q["validation"] = meta
                validated.append(validated_q)
        
        return validated
    
    # -------------------------
    # LangGraph Integration
    # -------------------------
    
    def _process_state(self, state: Dict) -> Dict:
        """
        Process LangGraph state for validation.
        
        Expected input state:
        {
            "questions": List[Dict],
            "context": str,
            "bloom_level": str,
            "allowed_verbs": List[str] (optional)
        }
        
        Output state adds:
        {
            "validated_questions": List[Dict],
            "validation_stats": Dict
        }
        """
        questions = state.get("questions", [])
        context = state.get("context", "")
        bloom_level = state.get("bloom_level", "remember")
        allowed_verbs = state.get("allowed_verbs")
        
        validated = self.validate_batch(questions, context, bloom_level, allowed_verbs)
        
        # Calculate stats
        total = len(questions)
        accepted = len(validated)
        rejected = total - accepted
        
        validation_stats = {
            "total_questions": total,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": accepted / total if total > 0 else 0,
            "bloom_level": bloom_level,
            "threshold": self.acceptance_threshold
        }
        
        return {
            **state,
            "validated_questions": validated,
            "validation_stats": validation_stats
        }
    
    def __call__(self, state: Dict) -> Dict:
        """
        Make validator LangGraph-native callable.
        """
        return self.node.invoke(state)
    
    def create_graph(self) -> StateGraph:
        """
        Create a LangGraph for validation workflow.
        """
        workflow = StateGraph(dict)
        
        workflow.add_node("validate", self._process_state)
        workflow.add_edge(START, "validate")
        workflow.add_edge("validate", END)
        
        return workflow.compile()