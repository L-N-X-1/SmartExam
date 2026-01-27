# core/embeddings.py
"""
Gestion des embeddings et du vector store FAISS avec métadonnées
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL, VECTOR_STORE_PATH
import pickle
import os
import json

# Charge le modèle d'embedding
print(f"🔄 Chargement du modèle d'embedding: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✅ Modèle chargé")

def create_embeddings(chunks_text):
    """
    Transforme les chunks en vecteurs
    INPUT: chunks_text (list) - liste de textes
    OUTPUT: embeddings (numpy array)
    """
    if not chunks_text:
        raise ValueError("❌ Liste de chunks vide")
    
    print(f"   🔄 Création des embeddings pour {len(chunks_text)} chunks...")
    
    embeddings = embedding_model.encode(chunks_text, show_progress_bar=True)
    embeddings_array = np.array(embeddings).astype('float32')
    
    if len(embeddings_array.shape) == 1:
        embeddings_array = embeddings_array.reshape(1, -1)
    
    print(f"   ✅ Embeddings créés : shape {embeddings_array.shape}")
    return embeddings_array

def create_index_with_metadata(chunks_data, course_name="default"):
    """
    Crée un index FAISS avec métadonnées
    
    INPUT: 
        - chunks_data (list) : liste de dict avec text + metadata
        - course_name (str) : nom du cours
    
    OUTPUT: 
        - index (faiss.Index)
        - chunks_data (list) : données complètes sauvegardées
    """
    if not chunks_data:
        raise ValueError("❌ Liste de chunks vide")
    
    print(f"\n🔧 Création de l'index FAISS avec métadonnées...")
    
    # Extrait juste le texte pour les embeddings
    chunks_text = [chunk["text"] for chunk in chunks_data]
    
    # Crée les embeddings
    embeddings = create_embeddings(chunks_text)
    
    # Ajoute les embeddings aux métadonnées
    for i, chunk in enumerate(chunks_data):
        chunk["embedding"] = embeddings[i].tolist()  # Convertit en list pour JSON
    
    # Crée l'index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"   ✅ Index créé : {index.ntotal} vecteurs, dimension {dimension}")
    
    # Sauvegarde
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    
    index_path = os.path.join(VECTOR_STORE_PATH, f"{course_name}_index.faiss")
    metadata_path = os.path.join(VECTOR_STORE_PATH, f"{course_name}_metadata.json")
    
    faiss.write_index(index, index_path)
    
    # Sauvegarde les métadonnées en JSON
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    print(f"   💾 Index sauvegardé : {os.path.basename(index_path)}")
    print(f"   💾 Métadonnées sauvegardées : {os.path.basename(metadata_path)}")
    
    return index, chunks_data

def create_index(chunks, course_name="default"):
    """
    Version rétrocompatible (sans métadonnées structurées)
    """
    # Convertit en format avec métadonnées basiques
    chunks_data = []
    for i, text in enumerate(chunks):
        chunks_data.append({
            "id": f"chunk_{i}",
            "text": text,
            "metadata": {
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        })
    
    return create_index_with_metadata(chunks_data, course_name)

def retrieve_with_metadata(query, course_name="default", k=5):
    """
    Recherche avec métadonnées complètes
    
    OUTPUT: liste de dict avec text + metadata + score
    """
    index_path = os.path.join(VECTOR_STORE_PATH, f"{course_name}_index.faiss")
    metadata_path = os.path.join(VECTOR_STORE_PATH, f"{course_name}_metadata.json")
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"❌ Index introuvable: {index_path}")
    
    # Charge l'index et les métadonnées
    index = faiss.read_index(index_path)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    # Encode la query
    query_embedding = embedding_model.encode([query]).astype('float32')
    
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Recherche
    k = min(k, len(chunks_data))
    distances, indices = index.search(query_embedding, k)
    
    # Prépare les résultats avec métadonnées
    results = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks_data[idx].copy()
        chunk['similarity_score'] = float(1 / (1 + distances[0][i]))  # Convertit distance en score
        results.append(chunk)
    
    print(f"   🔍 {len(results)} chunks récupérés avec métadonnées")
    
    return results

def retrieve(query, course_name="default", k=5):
    """
    Version rétrocompatible (retourne juste le texte)
    """
    results = retrieve_with_metadata(query, course_name, k)
    return [r["text"] for r in results]