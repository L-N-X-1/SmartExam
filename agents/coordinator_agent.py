import math
from collections import defaultdict
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda


class CoordinatorAgent:
    """
    Orchestrates:
    - RAG retrieval
    - Bloom agents calls
    - distribution control
    - validation filtering
    - retry on low yield
    
    Now uses LangGraph directly without core/llm_interface
    """

    def __init__(
        self,
        rag_engine,
        agents: Dict[str, Any],
        validator: Optional[Any] = None,
        rag_k: int = 8,
        max_retries: int = 2
    ):
        """
        agents: dict like
        {
            "remember": RememberAgent,
            "understand": UnderstandAgent,
            ...
        }
        """
        self.rag = rag_engine
        self.agents = agents
        self.validator = validator
        self.rag_k = rag_k
        self.max_retries = max_retries
        
        # Create LangGraph node
        self.node = RunnableLambda(self._process_state)

    # -------------------------
    # Bloom distribution (unchanged)
    # -------------------------

    def _compute_distribution(self, total_questions: int, weights: Dict[str, float]) -> Dict[str, int]:
        """
        weights example:
        {
            "remember": 0.1,
            "understand": 0.2,
            ...
        }
        """
        dist = {}
        running = 0

        for level, w in weights.items():
            n = int(math.floor(total_questions * w))
            dist[level] = n
            running += n

        # fix rounding remainder
        remainder = total_questions - running
        if remainder > 0:
            # assign remainder to largest weights first
            ordered = sorted(weights.items(), key=lambda x: -x[1])
            for level, _ in ordered:
                dist[level] += 1
                remainder -= 1
                if remainder == 0:
                    break

        return dist

    # -------------------------
    # RAG retrieval (unchanged)
    # -------------------------

    def _retrieve_context(self, topic: str) -> List[Dict]:
        chunks = self.rag.retrieve(topic, k=self.rag_k)

        # filter bad retrievals
        good = [
            c for c in chunks
            if isinstance(c, dict)
            and c.get("text")
            and len(c["text"]) > 50
        ]

        return good

    # -------------------------
    # Validation wrapper (updated for LangGraph)
    # -------------------------

    def _validate_batch(self, questions: List[Dict], context: str, bloom_level: str) -> List[Dict]:
        if not self.validator:
            return questions

        # Use validator's LangGraph state processing if available
        if hasattr(self.validator, '_process_state'):
            state = {
                "questions": questions,
                "context": context
            }
            result = self.validator._process_state(state)
            return result["validated_questions"]
        
        # Fallback to legacy validate method
        valid = []
        for q in questions:
            ok, meta = self.validator.validate(
                q["question"],
                context,
                bloom_level
            )

            if ok:
                q["validation"] = meta
                valid.append(q)

        return valid

    # -------------------------
    # Main pipeline (updated for LangGraph)
    # -------------------------

    def generate_exam(
        self,
        topic: str,
        total_questions: int,
        bloom_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Main orchestration entrypoint (legacy compatibility)
        """
        state = {
            "topic": topic,
            "total_questions": total_questions,
            "bloom_weights": bloom_weights
        }
        
        return self._process_state(state)

    def _process_state(self, state: Dict) -> Dict:
        """
        Process LangGraph state for exam generation.
        
        Expected input state:
        {
            "topic": str,
            "total_questions": int,
            "bloom_weights": Dict[str, float],
            "rag_k": int (optional),
            "max_retries": int (optional)
        }
        
        Output state adds:
        {
            "questions": List[Dict],
            "report": Dict,
            "rag_chunks_used": int,
            "generation_stats": Dict
        }
        """
        topic = state.get("topic")
        total_questions = state.get("total_questions", 10)
        bloom_weights = state.get("bloom_weights", {})
        rag_k = state.get("rag_k", self.rag_k)
        max_retries = state.get("max_retries", self.max_retries)

        # 1️⃣ retrieve shared context once
        rag_chunks = self._retrieve_context(topic)

        if not rag_chunks:
            raise ValueError("RAG returned no usable context")

        # build shared context string for validator
        context_text = "\n\n".join([c["text"] for c in rag_chunks])

        # 2️⃣ compute distribution
        distribution = self._compute_distribution(
            total_questions,
            bloom_weights
        )

        results = []
        stats = defaultdict(int)
        generation_stats = {
            "agents_called": [],
            "total_attempts": 0,
            "successful_generations": 0,
            "failed_generations": 0
        }

        # 3️⃣ call each Bloom agent
        for level, num_needed in distribution.items():

            if num_needed <= 0:
                continue

            if level not in self.agents:
                continue

            agent = self.agents[level]
            generation_stats["agents_called"].append(level)

            collected = []
            tries = 0

            while len(collected) < num_needed and tries <= max_retries:

                # Use agent's LangGraph state processing if available
                if hasattr(agent, '_process_state'):
                    agent_state = {
                        "rag_chunks": rag_chunks,
                        "num_questions": num_needed,
                        "trace_id": f"{topic}_{level}_{tries}"
                    }
                    batch_result = agent._process_state(agent_state)
                    batch = batch_result.get("questions", [])
                else:
                    # Fallback to legacy generate_questions
                    batch = agent.generate_questions(
                        rag_chunks,
                        num=num_needed
                    )
                    batch = batch.get("questions", []) if isinstance(batch, dict) else batch

                generation_stats["total_attempts"] += 1

                # structure guard
                batch = [
                    q for q in batch
                    if isinstance(q, dict)
                    and "question" in q
                ]

                # validate if validator present
                batch = self._validate_batch(
                    batch,
                    context_text,
                    level
                )

                collected.extend(batch)
                tries += 1

            if collected:
                generation_stats["successful_generations"] += 1
            else:
                generation_stats["failed_generations"] += 1

            # trim to exact needed count
            collected = collected[:num_needed]

            for q in collected:
                q["bloom_level"] = level

            results.extend(collected)
            stats[level] = len(collected)

        # 4️⃣ global dedup (string-level)
        seen = set()
        deduped = []

        for q in results:
            key = q["question"].strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(q)

        # 5️⃣ final report
        report = {
            "topic": topic,
            "requested": total_questions,
            "generated": len(deduped),
            "distribution_requested": distribution,
            "distribution_actual": dict(stats)
        }

        return {
            **state,
            "questions": deduped,
            "report": report,
            "rag_chunks_used": len(rag_chunks),
            "generation_stats": generation_stats,
            "context": context_text,
            "rag_chunks": rag_chunks
        }

    # -------------------------
    # LangGraph Integration
    # -------------------------

    def __call__(self, state: Dict) -> Dict:
        """
        Make coordinator LangGraph-native callable.
        """
        return self.node.invoke(state)

    def create_graph(self) -> StateGraph:
        """
        Create a LangGraph for the complete exam generation workflow.
        """
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("generate_exam", self._process_state)
        
        # Add edges
        workflow.add_edge(START, "generate_exam")
        workflow.add_edge("generate_exam", END)
        
        return workflow.compile()

    def create_advanced_graph(self) -> StateGraph:
        """
        Create an advanced LangGraph with conditional validation.
        """
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("generate_questions", self._process_state)
        
        if self.validator:
            workflow.add_node("validate_questions", self._validate_questions_node)
            workflow.add_edge("generate_questions", "validate_questions")
            workflow.add_edge("validate_questions", END)
        else:
            workflow.add_edge("generate_questions", END)
        
        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_questions")
        
        return workflow.compile()

    def _retrieve_context_node(self, state: Dict) -> Dict:
        """Separate node for context retrieval in advanced graph."""
        topic = state.get("topic")
        rag_k = state.get("rag_k", self.rag_k)
        
        rag_chunks = self._retrieve_context(topic)
        context_text = "\n\n".join([c["text"] for c in rag_chunks])
        
        return {
            **state,
            "rag_chunks": rag_chunks,
            "context": context_text,
            "rag_chunks_used": len(rag_chunks)
        }

    def _validate_questions_node(self, state: Dict) -> Dict:
        """Separate node for validation in advanced graph."""
        if not self.validator:
            return state
            
        questions = state.get("questions", [])
        context = state.get("context", "")
        
        # Group questions by bloom level for validation
        questions_by_level = defaultdict(list)
        for q in questions:
            level = q.get("bloom_level", "remember")
            questions_by_level[level].append(q)
        
        validated_questions = []
        validation_stats = defaultdict(int)
        
        for level, level_questions in questions_by_level.items():
            validated = self._validate_batch(level_questions, context, level)
            validated_questions.extend(validated)
            validation_stats[level] = len(validated)
        
        return {
            **state,
            "questions": validated_questions,
            "validation_stats": dict(validation_stats)
        }