# core/base_agent.py

from abc import ABC
import json
import re
import time
from typing import Any, Dict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from utils.schema import Question
from config import settings


class BaseBloomAgent(ABC):
    """
    Optimized Bloom Agent (exam question generator)
    - Token-efficient
    - Fast retrieval
    - Strong JSON enforcement
    - Configurable LLM provider via settings.py
    """

    def __init__(
        self,
        bloom_level: int,
        bloom_verbs: List[str],
        temperature: float = 0.1,
        max_context_chars: int = 500,
        retries: int = 3,
        debug: bool = False,
        llm: Any = None,
    ):
        # ------------------ Dynamic LLM ------------------
        if llm is None:
            # Local import to avoid circular dependency
            from core.llm_interface import LLMClient
            llm = LLMClient()
        self.llm = llm

        self.bloom_level = bloom_level
        self.allowed_verbs = bloom_verbs
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        self.retries = retries
        self.debug = debug
        self._last_raw = None
        self._cache: Dict[str, List[Dict]] = {}  # cache for repeated contexts

        self.node = RunnableLambda(self._process_state)

    # -------------------------------------------------
    # Context Builder (token optimized & deduped)
    # -------------------------------------------------
    def build_context(self, rag_chunks: List[Dict]) -> str:
        texts = []
        total = 0
        seen = set()

        # Top 2 chunks only (most relevant)
        for chunk in rag_chunks[:2]:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            key = text.lower()
            if key in seen:
                continue
            seen.add(key)

            if total + len(text) > self.max_context_chars:
                remaining = self.max_context_chars - total
                if remaining > 50:
                    cut = text[:remaining]
                    if "." in cut:
                        cut = cut.rsplit(".", 1)[0] + "."
                    texts.append(cut)
                break

            texts.append(text)
            total += len(text)

        return "\n".join(texts)

    # -------------------------------------------------
    # Optimized LLM Call with caching & retry
    # -------------------------------------------------
    def call_llm(self, prompt: str, max_tokens: int = 120) -> str:
        key = prompt.strip()
        if key in self._cache:
            return json.dumps(self._cache[key])  # return cached JSON

        for attempt in range(1, self.retries + 1):
            try:
                raw = self.llm.generate(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                parsed = self.parse_llm_response(raw)
                self._cache[key] = parsed
                return raw
            except RuntimeError as e:
                msg = str(e)
                if "Too many requests" in msg or "rate limit" in msg:
                    if self.debug:
                        print(f"[Retry {attempt}] Rate limit hit, waiting {2*attempt}s...")
                    time.sleep(2 * attempt)
                    continue
                if "context length" in msg.lower():
                    raise RuntimeError("Context too large for model")
                raise e

        raise RuntimeError("[BaseBloomAgent] LLM retries exhausted")

    # -------------------------------------------------
    # Ultra-compact prompt builder
    # -------------------------------------------------
    def _create_prompt(self, context: str, num_questions: int, topic: str = None) -> str:
        verbs = ", ".join(self.allowed_verbs[:3])
        topic_line = f"Topic: {topic}\n" if topic else ""

        return (
            f"Generate {num_questions} exam questions.\n"
            f"Bloom level: {self.bloom_level}\n"
            f"Allowed verbs: {verbs}\n"
            f"Rules: use ONLY context, one verb per question, no external knowledge, "
            f"no multi-part questions, no explanations, no markdown.\n"
            f"{topic_line}"
            f"Context:\n{context}\n"
            f"Return ONLY JSON list of objects: [{{'question': '...'}}]"
        )

    # -------------------------------------------------
    # Strict JSON Parsing
    # -------------------------------------------------
    def parse_llm_response(self, raw: str) -> List[Dict]:
        self._last_raw = raw
        if not raw:
            return []

        raw = raw.strip()
        # Remove ```json fences
        fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if fence:
            raw = fence.group(1).strip()

        # Extract JSON array
        bracket = re.search(r"\[[\s\S]*\]", raw)
        if bracket:
            raw = bracket.group(0)

        raw = raw.replace(",]", "]").replace(",}", "}")
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except Exception:
            if self.debug:
                print("[parse_llm_response] JSON parsing failed:", raw)
        return []

    # -------------------------------------------------
    # Validation & Cleanup
    # -------------------------------------------------
    def _uses_allowed_verb(self, question: str) -> bool:
        q = question.lower().strip()
        return any(q.startswith(v.lower() + " ") for v in self.allowed_verbs)

    def _clean_questions(self, questions: List[Dict]) -> List[Dict]:
        cleaned = []
        seen = set()

        for q in questions:
            if not isinstance(q, dict) or "question" not in q:
                continue

            text = q["question"].strip()
            if len(text) < 10 or not self._uses_allowed_verb(text):
                continue

            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({"question": text})

        return cleaned

    # -------------------------------------------------
    # LangGraph Execution
    # -------------------------------------------------
    def _process_state(self, state: Dict) -> Dict:
        rag_chunks = state.get("rag_chunks", [])
        total_questions = state.get("num_questions", 5)
        topic = state.get("topic")

        if not rag_chunks:
            return {**state, "questions": []}

        context = self.build_context(rag_chunks)
        prompt = self._create_prompt(context, total_questions, topic)
        raw = self.call_llm(prompt, max_tokens=250)
        parsed = self.parse_llm_response(raw)
        cleaned = self._clean_questions(parsed)

        return {**state, "questions": cleaned}

    def __call__(self, state: Dict) -> Dict:
        return self.node.invoke(state)

    def create_graph(self) -> StateGraph:
        workflow = StateGraph(dict)
        workflow.add_node("generate_questions", self._process_state)
        workflow.add_edge(START, "generate_questions")
        workflow.add_edge("generate_questions", END)
        return workflow.compile()
