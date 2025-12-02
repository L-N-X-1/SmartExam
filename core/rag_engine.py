"""
Module RAG Engine pour SMART EXAM
Gère l'indexation FAISS et la retrieval de chunks pertinents
"""

import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

# Initialisation du modèle d'embedding
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Variables globales pour FAISS
index = None
chunks_store = []
VECTOR_STORE_PATH = "data/vector_store/"

def get_embedding(text):
    """
    Génère l'embedding d'un texte
    Args:
        text (str): Texte à encoder
    Returns:
        np.array: Vecteur d'embedding
    """
    return model.encode(text, show_progress_bar=False)

def create_index(chunks):
    """
    Crée un index FAISS à partir de chunks de texte
    Args:
        chunks (list): Liste de strings (chunks de texte)
    Returns:
        faiss.Index: Index FAISS créé
    """
    global index, chunks_store
    
    # Génération des embeddings
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Création de l'index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # ou IndexFlatIP pour cosine similarity
    index.add(embeddings)
    
    # Stockage des chunks
    chunks_store = chunks
    
    return index

def save_index(path=VECTOR_STORE_PATH):
    """
    Sauvegarde l'index FAISS et les chunks
    """
    global index, chunks_store
    
    os.makedirs(path, exist_ok=True)
    
    if index is not None:
        faiss.write_index(index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(chunks_store, f)
        return True
    return False

def load_index(path=VECTOR_STORE_PATH):
    """
    Charge l'index FAISS et les chunks depuis le disque
    """
    global index, chunks_store
    
    index_path = os.path.join(path, "faiss.index")
    chunks_path = os.path.join(path, "chunks.pkl")
    
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks_store = pickle.load(f)
        return True
    return False

def retrieve(query, k=5):
    """
    Récupère les k chunks les plus pertinents pour une requête
    Args:
        query (str): Requête de recherche
        k (int): Nombre de chunks à retourner
    Returns:
        list: Liste de dictionnaires avec 'text' et 'score'
    """
    global index, chunks_store
    
    # Charger l'index si pas déjà chargé
    if index is None:
        if not load_index():
            return [{"text": "No index found. Please upload and process documents first.", "score": 0.0}]
    
    if len(chunks_store) == 0:
        return [{"text": "No documents indexed yet.", "score": 0.0}]
    
    # Génération de l'embedding de la requête
    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding]).astype('float32')
    
    # Recherche dans FAISS
    k = min(k, len(chunks_store))  # Ne pas demander plus que ce qu'on a
    distances, indices = index.search(query_embedding, k)
    
    # Formatage des résultats
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks_store):
            results.append({
                "text": chunks_store[idx],
                "score": float(distances[0][i])
            })
    
    return results

def process_pdf_to_chunks(text, chunk_size=800, chunk_overlap=150):
    """
    Découpe un texte en chunks avec overlap
    Args:
        text (str): Texte complet à découper
        chunk_size (int): Taille de chaque chunk
        chunk_overlap (int): Overlap entre chunks
    Returns:
        list: Liste de chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    
    return chunks