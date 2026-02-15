# utils/text_processing.py
"""
Nettoyage et découpage de texte pour le RAG avec métadonnées
"""

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
import re
from datetime import datetime

def clean_text(text):
    """
    Nettoie le texte (espaces multiples, sauts de ligne, etc.)
    """
    if not text:
        return ""
    
    # Remplace les retours à la ligne multiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Enlève les espaces en début/fin de ligne
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Enlève les espaces multiples
    text = re.sub(r' {2,}', ' ', text)
    
    # Enlève caractères spéciaux problématiques
    text = text.replace('\x00', '')
    
    return text.strip()

def detect_section_title(text):
    """
    Détecte si un chunk commence par un titre de section
    """
    # Patterns courants de titres
    patterns = [
        r'^CHAPITRE\s+\d+',
        r'^\d+\.\s+[A-Z]',
        r'^[IVX]+\.\s+[A-Z]',
        r'^#{1,3}\s+',  # Markdown
    ]
    
    first_line = text.split('\n')[0].strip()
    
    for pattern in patterns:
        if re.match(pattern, first_line, re.IGNORECASE):
            return first_line
    
    return None

def detect_language(text):
    """
    Détecte la langue du texte (simple heuristique)
    """
    french_words = ['le', 'la', 'les', 'un', 'une', 'est', 'sont', 'dans']
    english_words = ['the', 'is', 'are', 'in', 'of', 'and', 'to']
    
    text_lower = text.lower()
    
    french_count = sum(1 for word in french_words if f' {word} ' in text_lower)
    english_count = sum(1 for word in english_words if f' {word} ' in text_lower)
    
    if french_count > english_count:
        return "fr"
    elif english_count > french_count:
        return "en"
    else:
        return "unknown"

def chunk_text_with_metadata(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, 
                             document_id=None, source=None):
    """
    Découpe le texte en chunks avec métadonnées structurées
    
    INPUT: 
        - text (str) : texte complet
        - chunk_size (int) : taille max d'un chunk
        - overlap (int) : chevauchement entre chunks
        - document_id (str) : ID du document source
        - source (str) : URL ou chemin du fichier source
    
    OUTPUT: 
        - chunks_with_metadata (list) : liste de dict avec text + metadata
    
    EXEMPLE OUTPUT:
    [
        {
            "id": "chunk_abc123",
            "text": "Ce module permet de...",
            "embedding": [...],
            "metadata": {
                "document_id": "doc_xyz789",
                "section_title": "Introduction",
                "lang": "fr",
                "source": "https://...",
                "created_at": "2025-06-13T10:45:00Z",
                "chunk_index": 0,
                "total_chunks": 10
            }
        }
    ]
    """
    if not text:
        return []
    
    chunks_data = []
    chunks_text = []
    start = 0
    text_len = len(text)
    chunk_index = 0
    
    # Phase 1 : Découpe le texte
    while start < text_len:
        end = start + chunk_size
        
        if end >= text_len:
            chunk = text[start:].strip()
            if chunk:
                chunks_text.append(chunk)
            break
        
        chunk_text = text[start:end]
        last_space = chunk_text.rfind(' ')
        
        if last_space > chunk_size * 0.8:
            end = start + last_space
        
        chunk = text[start:end].strip()
        
        if chunk:
            chunks_text.append(chunk)
        
        start = end - overlap
        chunk_index += 1
    
    total_chunks = len(chunks_text)
    
    # Phase 2 : Ajoute les métadonnées à chaque chunk
    for idx, chunk_text in enumerate(chunks_text):
        
        # Génère un ID unique pour le chunk
        chunk_id = f"chunk_{document_id}_{idx}" if document_id else f"chunk_{idx}"
        
        # Détecte le titre de section
        section_title = detect_section_title(chunk_text)
        
        # Détecte la langue
        lang = detect_language(chunk_text)
        
        # Crée la structure complète
        chunk_data = {
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "document_id": document_id or "unknown",
                "section_title": section_title,
                "lang": lang,
                "source": source or "local",
                "created_at": datetime.now().isoformat(),
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split())
            }
        }
        
        chunks_data.append(chunk_data)
    
    return chunks_data

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Version simple pour rétrocompatibilité
    Retourne seulement le texte des chunks
    """
    chunks_with_metadata = chunk_text_with_metadata(text, chunk_size, overlap)
    return [chunk["text"] for chunk in chunks_with_metadata]