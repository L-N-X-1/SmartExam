#Pipeline RAG complet : chargement documents → chunking → embedding → sauvegarde dans data/vector_store/. Lance le traitement après upload.
# core/rag_engine.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.embeddings import get_embeddings, save_index, load_index
import os

def process_and_index(file_paths: list[str]):
    all_text = ""
    for path in file_paths:
        all_text += extract_text(path) + "\n\n"

    # Chunking intelligent
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(all_text)

    # Embedding + sauvegarde
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local("data/vector_store/")

    return len(chunks)
