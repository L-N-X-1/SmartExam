import re
import json
from typing import Tuple, Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda


class ValidatorAgent:
    """
    Validates generated questions using:
    - heuristic checks
    - Bloom verb alignment
    - ambiguity detection
    - LLM scoring judge
    
    Now uses LangGraph directly without core/llm_interface
    """

    def __init__(
        self,
        llm: Any,  # LangChain LLM or LangGraph compatible model
        threshold: int = 85,
        use_llm_judge: bool = True
    ):
        self.llm = llm
        self.threshold = threshold
        self.use_llm_judge = use_llm_judge

        self.bloom_verbs = {
            "remember": ["define","list","name","identify","recall","state"],
            "understand": ["explain","describe","summarize","interpret","outline"],
            "apply": ["solve","use","compute","implement","demonstrate"],
            "analyze": ["compare","contrast","differentiate","classify","analyze"],
            "evaluate": ["justify","critique","defend","assess","evaluate"],
            "create": ["design","propose","create","develop","construct"]
        }
        
        # Create LangGraph node
        self.node = RunnableLambda(self._process_state)

    # ---------------------------------
    # Heuristic checks (unchanged)
    # ---------------------------------

    def _clarity_score(self, q: str) -> int:
        score = 100

        if len(q) < 12:
            score -= 40

        if "??" in q:
            score -= 20

        if q.count("?") != 1:
            score -= 20

        if len(q.split()) < 5:
            score -= 20

        return max(score, 0)

    def _single_task_score(self, q: str) -> int:
        multi_markers = [
            " and ", " or ", ", and ", ", then ",
            "also", "plus"
        ]

        count = sum(q.lower().count(m) for m in multi_markers)

        if count >= 2:
            return 40
        if count == 1:
            return 70
        return 100

    def _bloom_verb_score(self, q: str, bloom_level: str) -> int:
        verbs = self.bloom_verbs.get(bloom_level.lower(), [])
        ql = q.lower()

        for v in verbs:
            if ql.startswith(v + " "):
                return 100
            if f" {v} " in ql:
                return 85

        return 40

    def _context_overlap_score(self, q: str, context: str) -> int:
        """
        crude grounding check via keyword overlap
        """
        q_words = set(re.findall(r"\w+", q.lower()))
        ctx_words = set(re.findall(r"\w+", context.lower()))

        if not ctx_words:
            return 0

        overlap = len(q_words & ctx_words) / max(len(q_words), 1)

        if overlap > 0.6:
            return 100
        if overlap > 0.4:
            return 80
        if overlap > 0.25:
            return 60
        return 30

    # ---------------------------------
    # LLM Judge with direct LangChain integration
    # ---------------------------------

    def _call_llm(self, prompt: str) -> str:
        """
        Direct LLM call using LangChain invoke pattern.
        """
        try:
            # Try LangChain invoke pattern first
            if hasattr(self.llm, 'invoke'):
                messages = [
                    SystemMessage(content="You are an exam quality judge. Return only valid JSON."),
                    HumanMessage(content=prompt)
                ]
                response = self.llm.invoke(messages)
                
                # Handle different response formats
                if hasattr(response, 'content'):
                    return response.content
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
            
            # Fallback to simple callable
            elif callable(self.llm):
                return self.llm(prompt, temperature=0)
            
            # Fallback to generate method (legacy)
            elif hasattr(self.llm, 'generate'):
                result = self.llm.generate(prompt=prompt, temperature=0)
                if hasattr(result, 'text'):
                    return result.text
                return str(result)
            
            else:
                raise RuntimeError(f"LLM object {type(self.llm)} is not callable and has no invoke/generate method")
                
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {str(e)}")

    def _llm_score(self, question: str, context: str, bloom_level: str) -> Dict[str, int]:

        prompt = f"""
You are an exam quality judge.

Bloom target: {bloom_level}

Context:
{context}

Question:
{question}

Score 0-100:

- clarity
- bloom_alignment
- context_grounded
- single_task

Return ONLY JSON:
{{
 "clarity": int,
 "bloom_alignment": int,
 "context_grounded": int,
 "single_task": int
}}
"""

        raw = self._call_llm(prompt)

        try:
            return json.loads(raw)
        except Exception:
            return {
                "clarity": 50,
                "bloom_alignment": 50,
                "context_grounded": 50,
                "single_task": 50
            }

    # ---------------------------------
    # Final public API (unchanged)
    # ---------------------------------

    def validate(
        self,
        question: str,
        context: str,
        bloom_level: str
    ) -> Tuple[bool, Dict[str, Any]]:

        # heuristic layer
        h_clarity = self._clarity_score(question)
        h_task = self._single_task_score(question)
        h_bloom = self._bloom_verb_score(question, bloom_level)
        h_ground = self._context_overlap_score(question, context)

        heuristic_avg = int(
            (h_clarity + h_task + h_bloom + h_ground) / 4
        )

        # optional LLM judge
        if self.use_llm_judge:
            llm_scores = self._llm_score(
                question,
                context[:4000],
                bloom_level
            )

            llm_avg = int(sum(llm_scores.values()) / 4)

            final_score = int(
                heuristic_avg * 0.4 +
                llm_avg * 0.6
            )
        else:
            llm_scores = None
            final_score = heuristic_avg

        meta = {
            "heuristic": {
                "clarity": h_clarity,
                "single_task": h_task,
                "bloom_alignment": h_bloom,
                "context_grounded": h_ground
            },
            "llm": llm_scores,
            "final_score": final_score
        }

        return final_score >= self.threshold, meta

    # ---------------------------------
    # LangGraph Integration
    # ---------------------------------

    def _process_state(self, state: Dict) -> Dict:
        """
        Process LangGraph state for validation.
        Expected input state:
        {
            "questions": [{"question": "...", "bloom_level": "..."}],
            "context": "...",
            "validation_threshold": int (optional)
        }
        
        Output state adds:
        {
            "validated_questions": [...],
            "validation_results": [...],
            "validation_stats": {...}
        }
        """
        questions = state.get("questions", [])
        context = state.get("context", "")
        threshold = state.get("validation_threshold", self.threshold)
        
        validated_questions = []
        validation_results = []
        stats = {"total": len(questions), "passed": 0, "failed": 0}
        
        for q in questions:
            if isinstance(q, dict) and "question" in q:
                question_text = q["question"]
                bloom_level = q.get("bloom_level", "remember")
                
                is_valid, meta = self.validate(question_text, context, bloom_level)
                
                if is_valid:
                    validated_questions.append({
                        **q,
                        "validation": meta
                    })
                    stats["passed"] += 1
                else:
                    stats["failed"] += 1
                
                validation_results.append({
                    "question": question_text,
                    "is_valid": is_valid,
                    "meta": meta
                })
        
        return {
            **state,
            "validated_questions": validated_questions,
            "validation_results": validation_results,
            "validation_stats": stats
        }

    def __call__(self, state: Dict) -> Dict:
        """
        Make validator LangGraph-native callable.
        """
        return self.node.invoke(state)

    def validate_batch(
        self,
        questions: List[Dict],
        context: str,
        bloom_level: str
    ) -> List[Dict]:
        """
        Validate a batch of questions (legacy compatibility).
        """
        state = {
            "questions": questions,
            "context": context
        }
        
        result = self._process_state(state)
        return result["validated_questions"]