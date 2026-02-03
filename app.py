# =============================================================================
# SMART EXAM – Générateur intelligent d'examens basé sur la Taxonomie de Bloom
# Auteur : [Ton nom ou "Équipe SMART EXAM"]
# Date   : Novembre 2025
# 
# À QUOI SERT CE FICHIER :
# Point d'entrée de l'application Streamlit. Gère le menu multi-page (Upload → Configurer → Générer → Review → Analytics). Utilise st.set_page_config() et un sidebar ou option_menu.
# 
# Tâche assignée à → [Nom(s) de l'étudiant ou du groupe]
# =============================================================================
# app.py
import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI  # Pour Grok (compatible API)
import numpy as np  # Pour FAISS/retrieval
# Assume tes imports pour RAG (ajoute si besoin)
from core.rag_engine import retrieve, create_index, process_pdf_to_chunks
from PyPDF2 import PdfReader  # ✅ Majuscules

st.set_page_config(page_title="SMART EXAM", layout="wide", page_icon="🧠")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "SMART EXAM",
        ["Upload Documents", "Configurer Examen", "Générer Questions", "Review & Export", "Analytics"],
        icons=['cloud-upload', 'sliders', 'robot', 'eye', 'bar-chart'],
        menu_icon="brain", default_index=0
    )

    # Provider select (ajoute pour choisir Grok)
    provider = st.selectbox("Provider LLM", ['grok', 'groq', 'local'])
    st.session_state.provider = provider

    # Input Grok key (comme recommandé)
    if st.session_state.provider == 'grok':
        st.text_input("Grok API Key", type="password", key="grok_api_key")

# Fonction real_llm_generation (comme recommandé, fixé pour Grok)
def real_llm_generation(query, context_chunks, api_key, model="grok-2"):
    provider = st.session_state.provider
    context = "\n\n".join([c['text'] for c in context_chunks]) if context_chunks else ""
    prompt = f"Use ONLY this context: {context}\n{query}"
    if provider == 'grok':
        if not st.session_state.grok_api_key:
            return "Grok API Key not provided."
        client = OpenAI(
            api_key=st.session_state.grok_api_key,
            base_url="https://api.x.ai/v1"  # xAI base URL
        )
        try:
            response = client.chat.completions.create(
                model=model or 'grok-2',
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=350  # Fixé pour éviter erreur
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erreur Grok: {str(e)}"
    # (Ajoute autres providers si besoin, e.g., groq ou local)
    return "LLM not available."

if selected == "Upload Documents":
    st.title("📚 Upload des supports de cours")
    st.write("Glissez-déposez vos PDF, slides, notes...")
    
    uploaded_file = st.file_uploader("Upload PDF for Test", type="pdf")
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", min_value=50, max_value=2000, value=220, step=10)
    with col2:
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=20, step=10)

    if st.button("Process and Index to FAISS"):
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                
                # Chunk le texte correctement
                chunks = process_pdf_to_chunks(text, chunk_size,chunk_overlap)
                st.write(f"✅ Created {len(chunks)} chunks")
                
                # Créer et sauvegarder l'index FAISS
                create_index(chunks)
                st.success(f"✅ Successfully indexed {len(chunks)} chunks to FAISS!")

    # Test RAG Retrieval
    st.header("🔍 Test RAG Retrieval")
    test_query = st.text_input("Enter your search query:")
    if st.button("Test Retrieve"):
        if test_query:
            with st.spinner("Searching..."):
                chunks = retrieve(test_query, k=5)
                
                if not chunks:
                    st.warning("⚠️ No documents indexed yet. Please upload and process a PDF first.")
                else:
                    st.write("**Retrieved Chunks:**")
                    for i, chunk in enumerate(chunks):
                        with st.expander(f"Chunk {i+1} (Relevance: {chunk['score']*100:.1f}%)"):
                            st.write(chunk['text'])
        else:
            st.warning("Please enter a query first.")

elif selected == "Configurer Examen":
    st.title("⚙️ Configuration de l'examen")
    # Sliders Bloom, durée, etc.

else:
    st.title(f"🚧 {selected} – En cours de développement")
    st.write("Cette page sera codée par ton groupe !")

    # Test button amélioré
    if st.button("Test RAG : trouve-moi du contenu sur CNN"):
        chunks = retrieve("réseaux de neurones convolutifs", k=5)
        
        if not chunks:
            st.warning("⚠️ Aucun document indexé. Allez dans 'Upload Documents' pour charger un PDF.")
        else:
            st.write("**Contexte trouvé:**")
            for i, chunk in enumerate(chunks):
                st.write(f"**Chunk {i+1}:**")
                st.write(chunk['text'][:300] + "...")
                st.write("---")