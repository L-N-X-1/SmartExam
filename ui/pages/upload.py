# ui/pages/upload.py
import streamlit as st
from services.document_processor import extract_text
from core.rag_engine import process_and_index
import os, shutil

st.title("Upload des supports de cours")

uploaded_files = st.file_uploader("Glissez vos PDF/DOCX ici", accept_multiple_files=True, type=["pdf", "docx"])

if uploaded_files:
    save_folder = "data/uploads"
    os.makedirs(save_folder, exist_ok=True)
    
    paths = []
    for uploaded_file in uploaded_files:
        path = os.path.join(save_folder, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        paths.append(path)
    
    if st.button("Traiter les documents et indexer"):
        with st.spinner("Extraction + chunking + embedding en cours..."):
            n_chunks = process_and_index(paths)
        st.success(f"Traitement terminé ! {n_chunks} chunks indexés et prêts pour le RAG !")
        st.balloons()