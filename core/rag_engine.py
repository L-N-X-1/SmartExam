# core/rag_engine.py
"""
Pipeline RAG avec métadonnées structurées
"""

def process_course_document(file_path, course_name="default"):
    """
    Pipeline complet RAG avec métadonnées
    """
    print(f"🔄 Traitement du document : {file_path}")
    
    try:
        from services.document_processor import process_document
        from utils.text_processing import clean_text, chunk_text_with_metadata
        from core.embeddings import create_index_with_metadata
    except ImportError as e:
        raise ImportError(f"Erreur d'import : {e}")
    
    # Extraction
    print(f"📖 Extraction du texte...")
    raw_text = process_document(file_path)
    print(f"✅ {len(raw_text)} caractères extraits")
    
    # Nettoyage
    print(f"🧹 Nettoyage...")
    clean = clean_text(raw_text)
    
    # Chunking avec métadonnées
    print(f"✂️ Découpage avec métadonnées...")
    chunks_data = chunk_text_with_metadata(
        clean,
        document_id=course_name,
        source=file_path
    )
    print(f"✅ {len(chunks_data)} chunks créés avec métadonnées")
    
    # Indexation
    print(f"🔧 Indexation FAISS...")
    index, _ = create_index_with_metadata(chunks_data, course_name)
    
    print(f"✅ Pipeline RAG terminé !")
    
    return {
        "success": True,
        "num_chunks": len(chunks_data),
        "course_name": course_name
    }

def get_relevant_context(query, course_name="default", k=5):
    """
    Version simple : retourne juste le texte
    """
    from core.embeddings import retrieve
    chunks = retrieve(query, course_name, k)
    return "\n\n".join(chunks)

def get_relevant_context_with_metadata(query, course_name="default", k=5):
    """
    Version avancée : retourne texte + métadonnées
    """
    from core.embeddings import retrieve_with_metadata
    return retrieve_with_metadata(query, course_name, k)