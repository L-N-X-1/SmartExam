#Gestion du vector store (FAISS ou Chroma). Fonctions : create_index(), add_documents(), retrieve(query, k=5).
# core/embeddings.py
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables to get OPENAI_API_KEY
load_dotenv()

def create_index(chunks):
    """
    Creates a new FAISS index from the initial list of text chunks.
    
    Args:
        chunks (list[str]): List of text strings to index.
        
    Returns:
        FAISS: The initialized vector store object.
    """
    # 1. Initialize the embedding model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2. Create the FAISS index from the texts
    # This converts text -> vectors and stores them
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings_model)
    
    return vector_store

def add_documents(vector_store, new_chunks):
    """
    Adds new text chunks to an existing vector store.
    
    Args:
        vector_store (FAISS): The existing vector store object.
        new_chunks (list[str]): New text strings to add.
    """
    if vector_store:
        vector_store.add_texts(new_chunks)
    return vector_store

def retrieve(vector_store, query, k=5):
    """
    Searches the vector store for the most relevant chunks.
    
    Args:
        vector_store (FAISS): The vector store to search.
        query (str): The user's question or topic.
        k (int): Number of chunks to return (default 5).
        
    Returns:
        list[str]: A list of the most relevant text chunks.
    """
    if not vector_store:
        return []
        
    # Perform similarity search
    # This finds the vectors mathematically closest to the query vector
    results = vector_store.similarity_search(query, k=k)
    
    # Return just the text content of the results
    return [doc.page_content for doc in results]