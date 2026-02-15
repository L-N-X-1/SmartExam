# agents/coordinator_agent.py

import math
from collections import defaultdict
from typing import Dict, List, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda


class CoordinatorAgent:
    """
    Orchestrates exam generation pipeline:
    - RAG retrieval
    - Bloom agents calls (all agents use BaseBloomAgent + LLMClient)
    - Distribution control
    - Validation filtering
    - Retry on low yield
    - Fully compatible with LangGraph nodes
    - Optimized for speed, deduplication, and multi-provider LLMs
    """

    def __init__(
        self,
        rag_engine,
        agents: Dict[str, Any],
        validator: Any = None,
        rag_k: int = 8,
        max_retries: int = 2,
    ):
        self.rag = rag_engine
        self.agents = agents
        self.validator = validator
        self.rag_k = rag_k
        self.max_retries = max_retries
        self.node = RunnableLambda(self._process_state)

    # -------------------------
    # Bloom distribution
    # -------------------------
    def _compute_distribution(self, total_questions: int, weights: Dict[str, float]) -> Dict[str, int]:
        dist = {}
        running = 0
        for level, w in weights.items():
            n = int(math.floor(total_questions * w))
            dist[level] = n
            running += n

        remainder = total_questions - running
        if remainder > 0:
            # Assign remainder to highest-weight levels first
            for level, _ in sorted(weights.items(), key=lambda x: -x[1]):
                dist[level] += 1
                remainder -= 1
                if remainder == 0:
                    break

        return dist

    # -------------------------
    # RAG retrieval
    # -------------------------
    def _retrieve_context(self, topic: str) -> List[Dict]:
        chunks = self.rag.retrieve(topic, k=self.rag_k)
        # Filter out too short or invalid chunks
        return [c for c in chunks if isinstance(c, dict) and c.get("text") and len(c["text"]) > 50]

    # -------------------------
    # Validation
    # -------------------------
    def _validate_batch(self, questions: List[Dict], context: str, bloom_level: str) -> List[Dict]:
        if not self.validator:
            return questions

        # LangGraph-compatible validator
        if hasattr(self.validator, "_process_state"):
            state = {"questions": questions, "context": context}
            result = self.validator._process_state(state)
            return result.get("validated_questions", [])

        # Legacy fallback
        valid = []
        for q in questions:
            ok, meta = self.validator.validate(q["question"], context, bloom_level)
            if ok:
                q["validation"] = meta
                valid.append(q)
        return valid

    # -------------------------
    # Main pipeline
    # -------------------------
    def generate_exam(
        self,
        topic: str,
        total_questions: int,
        bloom_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        state = {
            "topic": topic,
            "total_questions": total_questions,
            "bloom_weights": bloom_weights,
        }
        return self._process_state(state)

    def _process_state(self, state: Dict) -> Dict:
        topic = state.get("topic")
        total_questions = state.get("total_questions", 10)
        bloom_weights = state.get("bloom_weights", {})
        rag_chunks = self._retrieve_context(topic)

        if not rag_chunks:
            raise ValueError(f"[CoordinatorAgent] RAG returned no usable context for topic '{topic}'")

        context_text = "\n\n".join(c["text"] for c in rag_chunks)
        distribution = self._compute_distribution(total_questions, bloom_weights)

        results = []
        stats = defaultdict(int)
        generation_stats = {
            "agents_called": [],
            "total_attempts": 0,
            "successful_generations": 0,
            "failed_generations": 0,
        }

        for level, num_needed in distribution.items():
            if num_needed <= 0 or level not in self.agents:
                continue

            agent = self.agents[level]
            generation_stats["agents_called"].append(level)
            collected = []
            tries = 0

            while len(collected) < num_needed and tries < self.max_retries:
                agent_state = {
                    "rag_chunks": rag_chunks,
                    "num_questions": num_needed,
                    "trace_id": f"{topic}_{level}_{tries}",
                }
                batch_result = agent._process_state(agent_state)
                batch = batch_result.get("questions", [])

                generation_stats["total_attempts"] += 1

                # Validate batch
                batch = self._validate_batch(batch, context_text, level)
                collected.extend(batch)
                tries += 1

            if collected:
                generation_stats["successful_generations"] += 1
            else:
                generation_stats["failed_generations"] += 1

            # Trim to exact needed count
            collected = collected[:num_needed]
            for q in collected:
                q["bloom_level"] = level

            results.extend(collected)
            stats[level] = len(collected)

        # -------------------------
        # Global deduplication
        # -------------------------
        seen = set()
        deduped = []
        for q in results:
            key = q["question"].strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(q)

        report = {
            "topic": topic,
            "requested": total_questions,
            "generated": len(deduped),
            "distribution_requested": distribution,
            "distribution_actual": dict(stats),
        }

        return {
            **state,
            "questions": deduped,
            "report": report,
            "rag_chunks_used": len(rag_chunks),
            "generation_stats": generation_stats,
            "context": context_text,
            "rag_chunks": rag_chunks,
        }

    # -------------------------
    # LangGraph integration
    # -------------------------
    def __call__(self, state: Dict) -> Dict:
        return self.node.invoke(state)

    def create_graph(self) -> StateGraph:
        workflow = StateGraph(dict)
        workflow.add_node("generate_exam", self._process_state)
        workflow.add_edge(START, "generate_exam")
        workflow.add_edge("generate_exam", END)
        return workflow.compile()
