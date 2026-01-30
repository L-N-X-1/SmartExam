"""
Module RAG Engine pour SMART EXAM
Gère l'indexation FAISS et la retrieval de chunks pertinents
"""

import numpy as np
import faiss
import os
import pickle
import json
import uuid
import hashlib
from sentence_transformers import SentenceTransformer


class RAGEngine:
    """
    Moteur RAG pour l'indexation et la recherche de chunks textuels
    using FAISS and sentence embeddings
    """
    
    def __init__(self, vector_store_path="data/vector_store/", model_name="BAAI/bge-large-en-v1.5"):
        """
        Initialise le RAG Engine
        Args:
            vector_store_path (str): Chemin vers le dossier de stockage vectoriel
            model_name (str): Nom du modèle d'embedding à utiliser
        """
        self.vector_store_path = vector_store_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks_store = []  # list of dicts: {"id", "text", "source", "page", ...}
        self.embeddings = None  # np.array of embeddings aligned with chunks_store
        self.model_name = model_name
    
    def get_embedding(self, text):
        """
        Génère l'embedding d'un texte
        Args:
            text (str): Texte à encoder
        Returns:
            np.array: Vecteur d'embedding
        """
        return self.model.encode(text, show_progress_bar=False)
    
    def create_index(self, chunks, append=False):
        """
        Crée un index FAISS à partir de chunks de texte
        Args:
            chunks (list): Liste de strings (chunks de texte)
        Returns:
            faiss.Index: Index FAISS créé
        """
        # Accept either list of strings or list of dicts with 'text' and metadata
        incoming_texts = []
        incoming_chunks = []
        for i, c in enumerate(chunks):
            if isinstance(c, dict):
                text = c.get("text")
                meta = {k: v for k, v in c.items() if k != "text"}
            else:
                text = str(c)
                meta = {}

            chunk_id = meta.get("id", str(uuid.uuid4()))
            entry = {"id": chunk_id, "text": text}
            entry.update(meta)
            incoming_texts.append(text)
            incoming_chunks.append(entry)

        os.makedirs(self.vector_store_path, exist_ok=True)

        # Embeddings caching paths
        emb_path = os.path.join(self.vector_store_path, "embeddings.npy")
        meta_path = os.path.join(self.vector_store_path, "metadata.json")

        # If append requested and we already have an index/embeddings, append new docs
        if append and self.index is not None and self.embeddings is not None and len(incoming_texts) > 0:
            # compute embeddings for incoming texts
            new_emb = self.model.encode(incoming_texts, batch_size=32, show_progress_bar=True)
            new_emb = np.array(new_emb).astype("float32")
            # ensure normalization of new embeddings
            faiss.normalize_L2(new_emb)

            # append to existing embeddings and index
            self.embeddings = np.vstack([self.embeddings, new_emb])
            try:
                self.index.add(new_emb)
            except Exception:
                # If index incompatible, rebuild full index below
                pass

            # extend chunk store
            start_idx = len(self.chunks_store)
            for offset, ch in enumerate(incoming_chunks):
                ch["chunk_index"] = start_idx + offset
                self.chunks_store.append(ch)

            # update metadata and save
            corpus_hash = hashlib.sha256("\n".join([c.get("text", "") for c in self.chunks_store]).encode("utf-8")).hexdigest()
            np.save(emb_path, self.embeddings)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"model_name": self.model_name, "num_chunks": len(self.chunks_store), "corpus_hash": corpus_hash, "normalized": True}, f)

            with open(os.path.join(self.vector_store_path, "chunks.pkl"), "wb") as f:
                pickle.dump(self.chunks_store, f)

            return self.index

        # Not appending: build new index from incoming texts
        texts = incoming_texts
        new_chunks_store = incoming_chunks

        # compute corpus hash for full texts
        corpus_join = "\n".join(texts)
        corpus_hash = hashlib.sha256(corpus_join.encode("utf-8")).hexdigest()

        # If cached embeddings exist and were created with same model, same corpus size and identical texts, load them
        can_load_cache = False
        saved_meta = None
        if os.path.exists(emb_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    saved_meta = json.load(f)
                if saved_meta.get("model_name") == self.model_name and saved_meta.get("num_chunks") == len(texts) and saved_meta.get("corpus_hash") == corpus_hash:
                    can_load_cache = True
            except Exception:
                can_load_cache = False

        if can_load_cache:
            self.embeddings = np.load(emb_path)
            # if metadata says embeddings are not normalized, normalize now and update meta
            if saved_meta and not saved_meta.get("normalized", False):
                faiss.normalize_L2(self.embeddings)
                saved_meta["normalized"] = True
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(saved_meta, f)
        else:
            # Compute embeddings and save cache
            embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
            embeddings = np.array(embeddings).astype("float32")
            # normalize embeddings for cosine-sim via inner product
            faiss.normalize_L2(embeddings)
            np.save(emb_path, embeddings)
            self.embeddings = embeddings
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"model_name": self.model_name, "num_chunks": len(texts), "corpus_hash": corpus_hash, "normalized": True}, f)

        self.chunks_store = new_chunks_store

        # Création de l'index FAISS (Inner Product for cosine similarity)
        dimension = int(self.embeddings.shape[1])
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

        # Persist chunks metadata
        with open(os.path.join(self.vector_store_path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks_store, f)

        return self.index
    
    def save_index(self):
        """
        Sauvegarde l'index FAISS et les chunks
        """
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.vector_store_path, "faiss.index"))
            with open(os.path.join(self.vector_store_path, "chunks.pkl"), "wb") as f:
                pickle.dump(self.chunks_store, f)
            # save embeddings if present
            try:
                if self.embeddings is not None:
                    np.save(os.path.join(self.vector_store_path, "embeddings.npy"), self.embeddings)
                # compute corpus hash
                try:
                    corpus_hash = hashlib.sha256("\n".join([c.get("text", "") for c in self.chunks_store]).encode("utf-8")).hexdigest()
                except Exception:
                    corpus_hash = None
                # save metadata including model name and normalization flag
                meta = {"model_name": self.model_name, "num_chunks": len(self.chunks_store), "normalized": True}
                if corpus_hash:
                    meta["corpus_hash"] = corpus_hash
                with open(os.path.join(self.vector_store_path, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f)
            except Exception:
                pass
            return True
        return False
    
    def load_index(self):
        """
        Charge l'index FAISS et les chunks depuis le disque
        """
        index_path = os.path.join(self.vector_store_path, "faiss.index")
        chunks_path = os.path.join(self.vector_store_path, "chunks.pkl")
        
        meta_path = os.path.join(self.vector_store_path, "metadata.json")
        emb_path = os.path.join(self.vector_store_path, "embeddings.npy")

        if os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(meta_path):
            # verify model compatibility
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    saved_meta = json.load(f)
                saved_model = saved_meta.get("model_name")
                if saved_model != self.model_name:
                    # incompatible model used to build index
                    return False
            except Exception:
                return False

            self.index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                self.chunks_store = pickle.load(f)
            # load embeddings if present
            try:
                if os.path.exists(emb_path):
                    self.embeddings = np.load(emb_path)
            except Exception:
                self.embeddings = None

            # verify corpus hash matches the loaded chunks (guard against replaced files)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    saved_meta = json.load(f)
                saved_hash = saved_meta.get("corpus_hash")
                if saved_hash is not None:
                    corpus_join = "\n".join([c.get("text", "") for c in self.chunks_store])
                    computed = hashlib.sha256(corpus_join.encode("utf-8")).hexdigest()
                    if computed != saved_hash:
                        return False
            except Exception:
                pass

            return True
        return False
    
    def retrieve(self, query, k=5, similarity_threshold: float = 0.0):
        """
        Récupère les k chunks les plus pertinents pour une requête
        Args:
            query (str): Requête de recherche
            k (int): Nombre de chunks à retourner
        Returns:
            list: Liste de dictionnaires avec 'text' et 'similarity'
        """
        # Charger l'index si pas déjà chargé
        if self.index is None:
            if not self.load_index():
                return [{"text": "No index found. Please upload and process documents first.", "similarity": 0.0}]
        
        if len(self.chunks_store) == 0:
            return [{"text": "No documents indexed yet.", "similarity": 0.0}]
        
        # Génération de l'embedding de la requête
        query_embedding = self.get_embedding(query)
        query_embedding = np.array([query_embedding]).astype("float32")

        # 🔹 NORMALISATION (cosine)
        faiss.normalize_L2(query_embedding)

        # First-pass retrieval: get more candidates than final k for reranking
        num_chunks = len(self.chunks_store)
        if num_chunks == 0:
            return [{"text": "No documents indexed yet.", "similarity": 0.0}]

        k = min(k, num_chunks)
        k_retrieve = min(max(20, k * 4), num_chunks)
        distances, indices = self.index.search(query_embedding, k_retrieve)

        # If embeddings exist, perform exact re-ranking using cosine (dot) over stored embeddings
        results = []
        if self.embeddings is not None:
            # ensure embeddings are normalized
            emb_ret_indices = [int(idx) for idx in indices[0] if idx >= 0]
            unique_indices = []
            seen = set()
            for idx in emb_ret_indices:
                if idx not in seen and idx < num_chunks:
                    unique_indices.append(idx)
                    seen.add(idx)

            # compute dot product similarities
            q = query_embedding[0]
            candidate_embs = self.embeddings[unique_indices]
            sims = np.dot(candidate_embs, q)
            # get top-k by sims
            topk_idx = np.argsort(-sims)[:k]
            for pos in topk_idx:
                idx = unique_indices[int(pos)]
                chunk = self.chunks_store[idx]
                sim_value = float(sims[pos])
                if sim_value < similarity_threshold:
                    continue
                results.append({
                    "id": chunk.get("id", idx),
                    "text": chunk.get("text"),
                    "metadata": {k: v for k, v in chunk.items() if k not in ("id", "text")},
                    "chunk_index": idx,
                    "similarity": sim_value
                })
        else:
            # fallback: return FAISS distances directly
            for i, idx in enumerate(indices[0][:k]):
                if idx < len(self.chunks_store):
                    sim_value = float(distances[0][i])
                    if sim_value < similarity_threshold:
                        continue
                    chunk = self.chunks_store[idx]
                    results.append({
                        "id": chunk.get("id", idx) if isinstance(chunk, dict) else idx,
                        "text": chunk.get("text", chunk) if isinstance(chunk, dict) else chunk,
                        "metadata": {k: v for k, v in chunk.items() if k not in ("id", "text")} if isinstance(chunk, dict) else {},
                        "chunk_index": int(idx),
                        "similarity": sim_value
                    })

        return results
    
    def process_pdf_to_chunks(self, text, chunk_size=800, chunk_overlap=150, source=None, page=None, title=None):
        """
        Create semantic chunks from a long text using paragraph/sentence sliding window.
        chunk_size and chunk_overlap are interpreted as approximate token counts (whitespace tokens).
        Returns a list of dicts with metadata for each chunk.
        """
        # split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # improved sentence splitter: protect common abbreviations and initials
        import re
        placeholder = "<DOT>"
        abbrs = ["e.g.", "i.e.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Inc.", "Ltd.", "Jr.", "Sr.", "etc.", "vs.", "U.S.A.", "U.S.", "U.K."]

        sentences = []
        sentence_end_re = re.compile(r"(?<=[.!?])\s+")
        for p in paragraphs:
            tmp = p
            for a in abbrs:
                tmp = tmp.replace(a, a.replace('.', placeholder))
            # protect single-letter initials like 'A.'
            tmp = re.sub(r"\b([A-Z])\.", r"\1" + placeholder, tmp)

            sents = sentence_end_re.split(tmp)
            for s in sents:
                s = s.replace(placeholder, '.').strip()
                if s:
                    sentences.append(s)

        # accumulate sentences into chunks by token counts
        def token_count(s):
            return len(s.split())

        chunks = []
        current = []
        current_tokens = 0
        i = 0
        while i < len(sentences):
            s = sentences[i]
            tcount = token_count(s)
            if current_tokens + tcount <= chunk_size or not current:
                current.append(s)
                current_tokens += tcount
                i += 1
            else:
                text_chunk = " ".join(current)
                chunks.append(text_chunk)
                # apply overlap in tokens: keep last chunk_overlap tokens worth of sentences
                if chunk_overlap > 0:
                    overlap_tokens = chunk_overlap
                    # walk back through current to build overlap
                    overlap = []
                    ot = 0
                    for sent in reversed(current):
                        st = token_count(sent)
                        if ot + st <= overlap_tokens:
                            overlap.insert(0, sent)
                            ot += st
                        else:
                            break
                    current = overlap
                    current_tokens = ot
                else:
                    current = []
                    current_tokens = 0

        # flush
        if current:
            chunks.append(" ".join(current))

        # wrap chunks with metadata dicts
        wrapped = []
        for idx, c in enumerate(chunks):
            wrapped.append({
                "id": str(uuid.uuid4()),
                "text": c,
                "source": source,
                "page": page,
                "title": title,
                "chunk_index": idx
            })

        return wrapped