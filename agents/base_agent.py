from abc import ABC
import json
import re


class BaseAgent(ABC):
    """
    Base class for all Bloom agents.

    Provides:
    - context building from RAG chunks
    - prompt safety rules
    - JSON output contract
    - parsing + retry
    - bloom verb enforcement
    - deduplication + schema validation
    """

    def __init__(
        self,
        llm,
        name: str,
        bloom_level: str,
        allowed_verbs: list[str],
        max_context_chars: int = 6000,
        temperature: float = 0.2,
        retries: int = 2,
    ):
        self.llm = llm
        self.name = name
        self.bloom_level = bloom_level
        self.allowed_verbs = allowed_verbs
        self.max_context_chars = max_context_chars
        self.temperature = temperature
        self.retries = retries

    # -------------------------
    # Context handling
    # -------------------------

    def build_context(self, rag_chunks: list[dict]) -> str:
        """
        Build safe context string from RAG retrieve() output.
        Preserves sentence boundaries when truncating.
        """
        texts = []
        total = 0

        for c in rag_chunks:
            t = c.get("text", "")
            if not t:
                continue

            if total + len(t) > self.max_context_chars:
                remaining = self.max_context_chars - total
                if remaining > 50:
                    cut = t[:remaining]
                    # trim to last sentence end if possible
                    if "." in cut:
                        cut = cut.rsplit(".", 1)[0] + "."
                    texts.append(cut)
                break

            texts.append(t)
            total += len(t)

        return "\n\n".join(texts)

    # -------------------------
    # Prompt builder
    # -------------------------

    def build_prompt(self, context: str, num: int) -> str:
        verbs = ", ".join(self.allowed_verbs)

        return f"""
You generate exam questions.

Bloom level target: {self.bloom_level}
Allowed verbs: {verbs}

STRICT RULES:
- Use exactly one allowed verb per question
- Questions must be answerable ONLY from context
- No outside knowledge
- No hallucinated facts
- No multi-part questions
- No meta language
- No duplicates
- One cognitive task only
- If context is insufficient → return empty list

Context:
{context}

Generate {num} questions.

Return ONLY a JSON list.
No markdown.
No explanations.
No prose.
No code fences.

Format:
[
  {{"question": "..."}}
]
"""

    # -------------------------
    # LLM call wrapper
    # -------------------------

    def call_llm(self, prompt: str) -> str:
        """
        Adapter layer — keeps your LLM interchangeable.
        """
        return self.llm.generate(
            prompt=prompt,
            temperature=self.temperature,
        )

    # -------------------------
    # Output parsing
    # -------------------------

    def _extract_json_block(self, raw: str) -> str:
        """
        Extract JSON from markdown fences if present.
        """
        raw = raw.strip()

        match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()

        return raw

    def _try_json_repair(self, raw: str) -> str:
        """
        Fix common trailing comma errors.
        """
        return raw.replace(",]", "]").replace(",}", "}")

    def parse_output(self, raw: str):
        if not raw:
            return []

        raw = self._extract_json_block(raw)

        for candidate in (raw, self._try_json_repair(raw)):
            try:
                data = json.loads(candidate)

                if isinstance(data, list):
                    return data

                if isinstance(data, dict) and isinstance(
                    data.get("questions"), list
                ):
                    return data["questions"]

            except json.JSONDecodeError:
                continue

        return []

    # -------------------------
    # Validation helpers
    # -------------------------

    def _uses_allowed_verb(self, question: str) -> bool:
        q = question.lower().strip()
        return any(q.startswith(v.lower()) for v in self.allowed_verbs)

    def _valid_schema(self, item: dict) -> bool:
        return (
            isinstance(item, dict)
            and set(item.keys()) == {"question"}
            and isinstance(item["question"], str)
            and len(item["question"].strip()) > 10
        )

    def _clean_questions(self, questions: list[dict]) -> list[dict]:
        cleaned = []
        seen = set()

        for q in questions:
            if not self._valid_schema(q):
                continue

            text = q["question"].strip()

            if not self._uses_allowed_verb(text):
                continue

            key = text.lower()
            if key in seen:
                continue

            seen.add(key)
            cleaned.append({"question": text})

        return cleaned

    # -------------------------
    # Main API used by coordinator
    # -------------------------

    def generate_questions(self, rag_chunks, num=5):
        context = self.build_context(rag_chunks)

        if not context.strip():
            return []

        for _ in range(self.retries + 1):
            prompt = self.build_prompt(context, num)
            raw = self.call_llm(prompt)

            parsed = self.parse_output(raw)
            cleaned = self._clean_questions(parsed)

            if cleaned:
                return cleaned[:num]

        return []
