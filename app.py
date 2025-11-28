# =============================================================================
# SMART EXAM – Générateur intelligent d'examens basé sur la Taxonomie de Bloom
# Auteur : [Ranim"]
# Date   : Novembre 2025
# 
# À QUOI SERT CE FICHIER :
# [TU COPIES LE TEXTE SPÉCIFIQUE CI-DESSOUS SELON LE FICHIER]
# 
# Tâche assignée à → [Ranim]
# =============================================================================
# app.py
import streamlit as st
from streamlit_option_menu import option_menu
from services.document_processor import process_documents
from core.rag_engine import chunk_text
from core.embeddings import retrieve

st.set_page_config(page_title="SMART EXAM", layout="wide", page_icon="🧠")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "SMART EXAM",
        ["Upload Documents", "Configurer Examen", "Générer Questions", "Review & Export", "Analytics"],
        icons=['cloud-upload', 'sliders', 'robot', 'eye', 'bar-chart'],
        menu_icon="brain", default_index=0
    )

if selected == "Upload Documents":
    st.title("📚 Upload des supports de cours")
    st.write("Glissez-déposez vos PDF, slides, notes...")
    uploaded_files = st.file_uploader(
        "Sélectionnez vos fichiers de cours (PDF, DOCX)",
        type=['pdf', 'docx'],  # Restrict to file types you can process
        accept_multiple_files=True
    )
    if uploaded_files:
        # Placeholder for processing logic (Task 2)
        st.success(f"Successfully uploaded {len(uploaded_files)} files.")
        
        # We will call the extraction function here in the next step
        # extracted_text = process_documents(uploaded_files) 
        # CALL THE PROCESSOR FUNCTION
        combined_text = process_documents(uploaded_files) 
        
        # Display the result to confirm extraction works (for debugging)
        if combined_text:
            # CALL THE CHUNKER
            chunks = chunk_text(combined_text)
            st.success(f"Document processed and split into {len(chunks)} chunks!")
        else:
             st.error("Text extraction failed.")
    # ---------------------------------------------------------
    # 🧪 ZONE DE TEST RAG (Task 6)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("🧪 Test Rapide du RAG (Retrieval)")
    
    if "vector_store" in st.session_state:
        test_query = st.text_input("Posez une question sur votre cours pour tester la recherche :")
        
        if test_query:
            # Call the retrieve function you built earlier
            results = retrieve(st.session_state["vector_store"], test_query, k=3)
            
            st.write(f"**Résultats trouvés ({len(results)}) :**")
            for i, context in enumerate(results):
                with st.expander(f"Résultat #{i+1}"):
                    st.info(context)
    else:
        st.warning("Veuillez d'abord uploader un document pour tester.")
    

elif selected == "Configurer Examen":
    st.title("⚙️ Configuration de l'examen")
    # Sliders Bloom, durée, etc.

else:
    st.title(f"🚧 {selected} – En cours de développement")
    st.write("Cette page sera codée par ton groupe !")