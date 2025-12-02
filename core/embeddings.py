#Gestion du vector store (FAISS ou Chroma). Fonctions : create_index(), add_documents(), retrieve(query, k=5).# core/embeddings.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import EMBEDDING_MODEL
import os

def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def save_index(vectorstore):
    vectorstore.save_local("data/vector_store/")

def load_index():
    if not os.path.exists("data/vector_store/index.faiss"):
        raise FileNotFoundError("Index non trouvé. Traite d'abord des documents !")
    embeddings = get_embeddings()
    return FAISS.load_local("data/vector_store/", embeddings, allow_dangerous_deserialization=True)

def retrieve(query: str, k: int = 8):
    vectorstore = load_index()
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]