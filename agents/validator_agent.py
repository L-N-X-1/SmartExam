# agents/validator_agent.py

"""
ValidatorAgent

Validates generated questions using:
- Heuristic scoring (clarity, specificity, cognitive level alignment)
- Optional LLM-based judgment

Designed for LangGraph + dependency injection architecture.
"""

import json
from typing import Dict, Tuple, Any, Optional
from langchain_core.runnables import RunnableLambda
from core.llm_interface import LLMClient


class ValidatorAgent:
    """
    Validates a generated question and returns:
        - validation_passed: bool
        - validation_score: int
        - validation_details: dict
    """

    def __init__(
        self,
        llm: Any = None,
        threshold: int = 70,
        use_llm_judge: bool = True,
        verbose: bool = False,
    ):
        self.llm = llm or LLMClient()
        self.threshold = threshold
        self.use_llm_judge = use_llm_judge
        self.verbose = verbose

        # LangGraph node wrapper
        self.node = RunnableLambda(self._process_state)

    # -------------------------
    # Public validation API
    # -------------------------
    def validate(
        self,
        question: str,
        context: str,
        bloom_level: str,
        threshold: Optional[int] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        effective_threshold = threshold if threshold is not None else self.threshold

        # 1️⃣ Heuristic scoring
        heuristic_score, heuristic_meta = self._heuristic_score(question, context, bloom_level)
        final_score = heuristic_score
        llm_meta = {}

        # 2️⃣ Optional LLM scoring
        if self.use_llm_judge and self.llm:
            try:
                llm_score, llm_meta = self._llm_judge(question, context, bloom_level)
                final_score = int((heuristic_score + llm_score) / 2)
            except Exception as e:
                if self.verbose:
                    print(f"[Validator] LLM judge failed: {e}")
                llm_meta = {"score": 50, "reasoning": "LLM judge failed"}

        meta = {
            "heuristic_score": heuristic_score,
            "llm_score": llm_meta.get("score"),
            "final_score": final_score,
            "details": {**heuristic_meta, **llm_meta},
        }

        return final_score >= effective_threshold, meta

    # -------------------------
    # Heuristic scoring
    # -------------------------
    def _heuristic_score(self, question: str, context: str, bloom_level: str) -> Tuple[int, Dict[str, Any]]:
        words = question.split()
        score = 0
        details = {}

        # Clarity
        if 8 <= len(words) <= 40:
            score += 30
            details["clarity"] = 30
        else:
            details["clarity"] = 10

        # Context relevance
        context_words = set(context.lower().split())
        overlap = len(set(word.lower() for word in words) & context_words)
        if overlap > 3:
            score += 30
            details["relevance"] = 30
        else:
            details["relevance"] = 15

        # Bloom alignment
        bloom_keywords = {
            "remember": ["define", "list", "identify"],
            "understand": ["explain", "describe", "summarize"],
            "apply": ["use", "apply", "demonstrate"],
            "analyze": ["analyze", "compare", "contrast"],
            "evaluate": ["evaluate", "justify", "argue"],
            "create": ["design", "create", "propose"],
        }
        keywords = bloom_keywords.get(bloom_level.lower(), [])
        if any(word in question.lower() for word in keywords):
            score += 40
            details["bloom_alignment"] = 40
        else:
            details["bloom_alignment"] = 20

        return score, details

    # -------------------------
    # Optional LLM scoring
    # -------------------------
    def _llm_judge(self, question: str, context: str, bloom_level: str) -> Tuple[int, Dict[str, Any]]:
        if not self.llm:
            return 50, {"score": 50, "reasoning": "No LLM configured"}

        prompt = f"""
Evaluate this question as an expert educator.

Bloom Level: {bloom_level}

Context:
{context}

Question:
{question}

Return ONLY valid JSON: {{"score": integer 0-100, "reasoning": "short explanation"}}
"""
        try:
            # Use unified LLMClient generate interface
            response_text = self.llm.generate([{"role": "user", "content": prompt}], max_tokens=100, temperature=0.1)
            parsed = json.loads(response_text)
            score = int(parsed.get("score", 50))
            return score, parsed
        except Exception:
            return 50, {"score": 50, "reasoning": "LLM response parsing failed"}

    # -------------------------
    # LangGraph state processor
    # -------------------------
    def _process_state(self, state: Dict) -> Dict:
        question_text = state.get("generated_question")
        context = state.get("retrieved_context", "")
        bloom_level = state.get("bloom_level", "understand")
        threshold = state.get("validation_threshold", self.threshold)

        if not question_text:
            return {
                **state,
                "validation_passed": False,
                "validation_score": 0,
                "validation_details": {"error": "No question provided"},
            }

        is_valid, meta = self.validate(question_text, context, bloom_level, threshold)
        return {
            **state,
            "validation_passed": is_valid,
            "validation_score": meta["final_score"],
            "validation_details": meta,
        }

    # -------------------------
    # Callable (LangGraph)
    # -------------------------
    def __call__(self, state: Dict) -> Dict:
        return self.node.invoke(state)
