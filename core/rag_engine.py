"""
Clean RAG Engine
- FAISS + SentenceTransformers
- Definition-aware chunking & retrieval
- Safe persistence
"""

import os
import re
import json
import uuid
import faiss
import pickle
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Utilities
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"^[^\w]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_definition(text: str) -> bool:
    return bool(re.search(
        r"\b(is|are)\s*(a|an)\b"
        r"|\brefers\s*to\b"
        r"|\bdesigned\s*to\b"
        r"|\bcan\s*be\s*defined\s*as\b",
        normalize_text(text)
    ))


def extract_concept(query: str) -> str | None:
    q = normalize_text(query)
    for p in ("what is", "what's", "define", "explain", "describe"):
        if q.startswith(p):
            return q.replace(p, "").strip().split()[0]
    return None


# ============================================================
# RAG Engine
# ============================================================

class RAGEngine:

    def __init__(
        self,
        vector_store_path="data/vector_store",
        model_name="BAAI/bge-large-en-v1.5"
    ):
        self.path = vector_store_path
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.index = None
        self.embeddings = None
        self.chunks = []

        os.makedirs(self.path, exist_ok=True)

    # ------------------------
    # Embeddings
    # ------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=32, show_progress_bar=True)
        emb = np.array(emb).astype("float32")
        faiss.normalize_L2(emb)
        return emb

    # ------------------------
    # Chunking
    # ------------------------

    def process_pdf_to_chunks(
        self,
        text: str,
        chunk_size: int = 240,
        chunk_overlap: int = 20,
        source=None,
        page=None,
        title=None
    ) -> list[dict]:

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        chunks = []
        used = set()

        # ---- Pass 1: atomic definitions
        for i, line in enumerate(lines):
            if is_definition(line) and len(line.split()) >= 6:
                chunks.append(self._wrap_chunk(
                    line, "definition", source, page, title
                ))
                used.add(i)

        # ---- Pass 2: semantic blocks with overlap
        buf, tokens = [], 0
        overlap_buf = []  # Store words for overlap
        
        for i, line in enumerate(lines):
            if i in used:
                continue

            buf.append(line)
            tokens += len(line.split())

            if tokens >= chunk_size:
                block = " ".join(buf)
                ctype = "definition" if is_definition(block) else "concept"
                chunks.append(self._wrap_chunk(
                    block, ctype, source, page, title
                ))
                
                # Keep overlap tokens for next chunk
                if chunk_overlap > 0:
                    all_words = block.split()
                    overlap_words = all_words[-chunk_overlap:] if len(all_words) > chunk_overlap else all_words
                    buf = [" ".join(overlap_words)]
                    tokens = len(overlap_words)
                else:
                    buf, tokens = [], 0

        if buf:
            chunks.append(self._wrap_chunk(
                " ".join(buf), "concept", source, page, title
            ))

        return chunks

    def _wrap_chunk(self, text, ctype, source, page, title):
        return {
            "id": str(uuid.uuid4()),
            "text": text,
            "type": ctype,
            "source": source,
            "page": page,
            "title": title
        }

    # ------------------------
    # Indexing
    # ------------------------

    def create_index(self, chunks: list[dict]) -> None:
        texts = [c["text"] for c in chunks]
        self.embeddings = self.embed(texts)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        self.chunks = chunks

        self._persist(texts)

    def append_to_index(self, new_chunks: list[dict]) -> int:
        """
        Append new chunks to the existing index without replacing.
        Returns the number of new chunks added.
        """
        # Load existing index if not loaded
        if self.index is None:
            loaded = self.load_index()
            if not loaded:
                # No existing index, create new one
                self.create_index(new_chunks)
                return len(new_chunks)
        
        # If embeddings weren't loaded (they're not persisted separately),
        # we need to reconstruct them from existing chunks
        if self.embeddings is None or len(self.embeddings) == 0 or \
           (hasattr(self.embeddings, 'shape') and self.embeddings.shape[0] != len(self.chunks)):
            # Re-embed existing chunks
            existing_texts = [c["text"] for c in self.chunks]
            self.embeddings = self.embed(existing_texts)
        
        # Embed new chunks
        new_texts = [c["text"] for c in new_chunks]
        new_embeddings = self.embed(new_texts)
        
        # Combine with existing
        self.chunks.extend(new_chunks)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Rebuild index with all embeddings
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        
        # Persist updated index
        all_texts = [c["text"] for c in self.chunks]
        self._persist(all_texts)
        
        return len(new_chunks)

    def _persist(self, texts):
        # Ensure directory exists before writing
        os.makedirs(self.path, exist_ok=True)
        faiss.write_index(self.index, f"{self.path}/faiss.index")

        with open(f"{self.path}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        meta = {
            "model": self.model_name,
            "num_chunks": len(self.chunks),
            "corpus_hash": hashlib.sha256("\n".join(texts).encode()).hexdigest()
        }

        with open(f"{self.path}/metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load_index(self) -> bool:
        try:
            with open(f"{self.path}/metadata.json") as f:
                meta = json.load(f)
            if meta["model"] != self.model_name:
                raise ValueError("Embedding model mismatch")

            self.index = faiss.read_index(f"{self.path}/faiss.index")
            with open(f"{self.path}/chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)

            return True
        except Exception:
            return False

    # ------------------------
    # Retrieval
    # ------------------------

    def retrieve(self, query: str, k: int = 5) -> list[dict]:

        if self.index is None and not self.load_index():
            return []

        q_vec = self.embed([query])
        scores, indices = self.index.search(q_vec, len(self.chunks))

        concept = extract_concept(query)
        results = []

        for idx, score in zip(indices[0], scores[0]):
            chunk = self.chunks[idx]
            raw_score = float(max(0.0, min(1.0, score)))  # Clamp to 0-1
            boost_factor = self._compute_boost(concept, chunk)
            
            # Apply boost multiplicatively: pushes score toward 1.0 without exceeding it
            # Formula: score + boost_factor * (1 - score)
            # This means a 0.7 score with 0.5 boost becomes: 0.7 + 0.5 * 0.3 = 0.85
            final_score = raw_score + boost_factor * (1.0 - raw_score)
            final_score = min(1.0, max(0.0, final_score))  # Ensure 0-1 range
            
            results.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "type": chunk["type"],
                "score": round(final_score, 4)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def _compute_boost(self, concept: str | None, chunk: dict) -> float:
        """
        Compute a boost factor (0.0 to ~0.3) based on concept matching.
        Higher boost = score gets pushed closer to 1.0
        
        Returns:
            0.0  - no boost
            0.08 - concept found anywhere in text
            0.15 - concept found in first 50 chars
            0.25 - concept at the start of text
            +0.05 bonus for acronym expansion match
        """
        if not concept:
            return 0.0

        text = normalize_text(chunk["text"])
        boost = 0.0

        # Check if it's a definition-type chunk (slightly higher potential boost)
        is_def = chunk.get("type") == "definition"
        multiplier = 1.0 if is_def else 0.7

        # Graduated position-based boost (subtle values)
        if text.startswith(concept):
            boost = 0.25 * multiplier
        elif concept in text[:50]:
            boost = 0.15 * multiplier
        elif concept in text:
            boost = 0.08 * multiplier

        # Acronym expansion boost
        acronym_boost = self._acronym_match_boost(concept, text)
        boost = min(0.35, boost + acronym_boost * 0.05 * multiplier)

        return boost

    def _acronym_match_boost(self, concept: str, text: str) -> float:
        """
        Check if concept is an acronym that expands to words in the text.
        e.g., 'cpu' -> 'central processing unit'
        """
        if len(concept) < 2 or len(concept) > 10:
            return 0.0

        # Build regex pattern: c.*p.*u for 'cpu'
        pattern = r'\b' + r'.*\b'.join(concept) + r'.*'
        
        # Look for the expansion pattern in parentheses or after "is/are"
        expansion_patterns = [
            rf'\({concept}\)',  # (CPU)
            rf'{concept}\s*[-–:]\s*',  # CPU - or CPU:
            rf'\b{concept}s?\b.*?\bis\s+a\b',  # CPU is a
        ]
        
        for pat in expansion_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return 1.0

        # Check if words starting with each letter of the acronym appear in sequence
        words = text.split()
        acronym_idx = 0
        for word in words:
            if acronym_idx < len(concept) and word.startswith(concept[acronym_idx]):
                acronym_idx += 1
                if acronym_idx == len(concept):
                    return 0.75  # Found all letters in sequence
        
        return 0.0


# ============================================================
# Singleton helpers
# ============================================================

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


def process_pdf_to_chunks(*args, **kwargs):
    return get_engine().process_pdf_to_chunks(*args, **kwargs)

def create_index(chunks):
    return get_engine().create_index(chunks)

def append_to_index(chunks):
    return get_engine().append_to_index(chunks)

def retrieve(query, k=5):
    return get_engine().retrieve(query, k=k)
