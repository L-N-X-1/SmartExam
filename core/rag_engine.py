"""
Clean RAG Engine
- FAISS + SentenceTransformers
- Definition-aware chunking & retrieval
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
from config import settings
from typing import Optional, List, Dict

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

def extract_concept(query: str) -> Optional[str]:
    q = normalize_text(query)
    for p in ("what is", "what's", "define", "explain", "describe"):
        if q.startswith(p):
            words = q.replace(p, "").strip().split()
            return words[0] if words else None
    return None

def safe_path(name: str) -> str:
    """Sanitize folder name for vector store / FAISS files."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)

# ============================================================
# RAG Engine
# ============================================================

class RAGEngine:

    def __init__(self, vector_store_path: Optional[str] = None, embedding_model: Optional[str] = None):
        # ------------------------
        # Embedding model for vector store
        # ------------------------
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL
        safe_model_name = safe_path(self.embedding_model_name)
        self.path = vector_store_path or os.path.join(settings.VECTOR_STORE_PATH, safe_model_name)
        os.makedirs(self.path, exist_ok=True)

        self.model = SentenceTransformer(self.embedding_model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[Dict] = []

    # ------------------------
    # Embeddings
    # ------------------------
    def embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=32, show_progress_bar=True)
        emb = np.array(emb).astype("float32")
        faiss.normalize_L2(emb)
        return emb

    # ------------------------
    # Chunking
    # ------------------------
    def process_pdf_to_chunks(
        self, text: str, chunk_size: int = 240, chunk_overlap: int = 20, source=None, page=None, title=None
    ) -> List[Dict]:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        chunks = []
        used = set()

        # Pass 1: atomic definitions
        for i, line in enumerate(lines):
            if is_definition(line) and len(line.split()) >= 6:
                chunks.append(self._wrap_chunk(line, "definition", source, page, title))
                used.add(i)

        # Pass 2: semantic blocks with overlap
        buf, tokens = [], 0
        for i, line in enumerate(lines):
            if i in used:
                continue
            buf.append(line)
            tokens += len(line.split())
            if tokens >= chunk_size:
                block = " ".join(buf)
                ctype = "definition" if is_definition(block) else "concept"
                chunks.append(self._wrap_chunk(block, ctype, source, page, title))
                overlap_words = block.split()[-chunk_overlap:] if chunk_overlap > 0 else []
                buf = [" ".join(overlap_words)] if overlap_words else []
                tokens = len(overlap_words)

        if buf:
            chunks.append(self._wrap_chunk(" ".join(buf), "concept", source, page, title))

        return chunks

    def _wrap_chunk(self, text: str, ctype: str, source=None, page=None, title=None) -> Dict:
        return {"id": str(uuid.uuid4()), "text": text, "type": ctype, "source": source, "page": page, "title": title}

    # ------------------------
    # Indexing
    # ------------------------
    def create_index(self, chunks: List[Dict]) -> None:
        texts = [c["text"] for c in chunks]
        self.embeddings = self.embed(texts)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        self.chunks = chunks
        self._persist(texts)

    def _persist(self, texts: List[str]) -> None:
        os.makedirs(self.path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.path, "faiss.index"))
        with open(os.path.join(self.path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        meta = {
            "model": self.embedding_model_name,
            "num_chunks": len(self.chunks),
            "corpus_hash": hashlib.sha256("\n".join(texts).encode()).hexdigest()
        }
        with open(os.path.join(self.path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def save_engine(self, path: str = "data/index/rag_engine.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_engine(path: str = "data/index/rag_engine.pkl") -> Optional["RAGEngine"]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def load_index(self) -> bool:
        try:
            with open(os.path.join(self.path, "metadata.json")) as f:
                meta = json.load(f)
            if meta["model"] != self.embedding_model_name:
                raise ValueError("Embedding model mismatch")
            self.index = faiss.read_index(os.path.join(self.path, "faiss.index"))
            with open(os.path.join(self.path, "chunks.pkl"), "rb") as f:
                self.chunks = pickle.load(f)
            return True
        except Exception:
            return False

    # ------------------------
    # Retrieval
    # ------------------------
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if self.index is None and not self.load_index():
            return []
        q_vec = self.embed([query])
        scores, indices = self.index.search(q_vec, len(self.chunks))
        concept = extract_concept(query)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            chunk = self.chunks[idx]
            raw_score = float(max(0.0, min(1.0, score)))
            boost_factor = self._compute_boost(concept, chunk)
            final_score = raw_score + boost_factor * (1.0 - raw_score)
            final_score = min(1.0, max(0.0, final_score))
            results.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "type": chunk["type"],
                "score": round(final_score, 4)
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def _compute_boost(self, concept: Optional[str], chunk: Dict) -> float:
        if not concept:
            return 0.0
        text = normalize_text(chunk["text"])
        boost = 0.0
        multiplier = 1.0 if chunk.get("type") == "definition" else 0.7
        if text.startswith(concept):
            boost = 0.25 * multiplier
        elif concept in text[:50]:
            boost = 0.15 * multiplier
        elif concept in text:
            boost = 0.08 * multiplier
        acronym_boost = self._acronym_match_boost(concept, text)
        boost = min(0.35, boost + acronym_boost * 0.05 * multiplier)
        return boost

    def _acronym_match_boost(self, concept: str, text: str) -> float:
        if len(concept) < 2 or len(concept) > 10:
            return 0.0
        pattern = r'\b' + r'.*\b'.join(concept) + r'.*'
        expansion_patterns = [
            rf'\({concept}\)',
            rf'{concept}\s*[-–:]\s*',
            rf'\b{concept}s?\b.*?\bis\s+a\b',
        ]
        for pat in expansion_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return 1.0
        words = text.split()
        acronym_idx = 0
        for word in words:
            if acronym_idx < len(concept) and word.startswith(concept[acronym_idx]):
                acronym_idx += 1
                if acronym_idx == len(concept):
                    return 0.75
        return 0.0

# ============================================================
# Singleton helpers
# ============================================================

_engine: Optional[RAGEngine] = None

def get_engine(embedding_model: Optional[str] = None) -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine(embedding_model=embedding_model)
    return _engine

def process_pdf_to_chunks(*args, **kwargs):
    return get_engine().process_pdf_to_chunks(*args, **kwargs)

def create_index(chunks: List[Dict]) -> RAGEngine:
    engine = get_engine()
    engine.create_index(chunks)
    engine.save_engine()
    return engine

def retrieve(query: str, k: int = 5):
    return get_engine().retrieve(query, k=k)
