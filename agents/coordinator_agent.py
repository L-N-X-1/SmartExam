import math
from collections import defaultdict


class CoordinatorAgent:
    """
    Orchestrates:
    - RAG retrieval
    - Bloom agents calls
    - distribution control
    - validation filtering
    - retry on low yield
    """

    def __init__(
        self,
        rag_engine,
        agents: dict,
        validator=None,
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

    # -------------------------
    # Bloom distribution
    # -------------------------

    def _compute_distribution(self, total_questions, weights):
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
    # RAG retrieval
    # -------------------------

    def _retrieve_context(self, topic):
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
    # Validation wrapper
    # -------------------------

    def _validate_batch(self, questions, context, bloom_level):
        if not self.validator:
            return questions

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
    # Main pipeline
    # -------------------------

    def generate_exam(
        self,
        topic: str,
        total_questions: int,
        bloom_weights: dict
    ):
        """
        Main orchestration entrypoint
        """

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

        # 3️⃣ call each Bloom agent
        for level, num_needed in distribution.items():

            if num_needed <= 0:
                continue

            if level not in self.agents:
                continue

            agent = self.agents[level]

            collected = []
            tries = 0

            while len(collected) < num_needed and tries <= self.max_retries:

                batch = agent.generate_questions(
                    rag_chunks,
                    num=num_needed
                )

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
            "questions": deduped,
            "report": report,
            "rag_chunks_used": len(rag_chunks)
        }
