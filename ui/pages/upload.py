# ui/pages/upload.py

import os
import streamlit as st
from services.document_processor import extract_text
from core.rag_engine import process_pdf_to_chunks, create_index
from config import settings


def show_upload_page():
    st.title("📚 Upload des supports de cours")

    st.markdown(
        "Déposez vos fichiers PDF ou DOCX. L'application traitera automatiquement le texte et préparera le contenu pour générer des questions/examens."
    )

    # --- File uploader ---
    uploaded_files = st.file_uploader(
        "Glissez vos fichiers ici",
        accept_multiple_files=True,
        type=["pdf", "docx"],
        help="Vous pouvez uploader plusieurs fichiers à la fois."
    )

    all_text = ""

    # --- Load previously saved RAG engine if exists ---
    if "rag_engine" not in st.session_state:
        index_path = "data/index/rag_engine.pkl"
        if os.path.exists(index_path):
            import pickle
            try:
                with open(index_path, "rb") as f:
                    st.session_state["rag_engine"] = pickle.load(f)
                    st.session_state["course_ready"] = True
                st.info("🔄 Cours précédent chargé automatiquement depuis le disque.")
            except Exception as e:
                st.error(f"❌ Échec du chargement du cours précédent : {e}")

    if uploaded_files:
        save_folder = settings.UPLOAD_FOLDER
        os.makedirs(save_folder, exist_ok=True)

        st.markdown("**📄 Fichiers uploadés :**")
        for uploaded_file in uploaded_files:
            path = os.path.join(save_folder, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text safely
            try:
                file_text = extract_text(path)
                if not file_text.strip():
                    st.warning(f"⚠️ Aucun texte détecté dans {uploaded_file.name}.")
                all_text += file_text + "\n\n"
            except Exception as e:
                st.error(f"❌ Impossible d'extraire {uploaded_file.name}: {e}")
                continue

            size_kb = uploaded_file.size / 1024
            size_str = f"{size_kb:.2f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"
            st.write(f"- {uploaded_file.name} ({size_str})")

        # --- Prepare course button ---
        if st.button("✅ Préparer le cours"):
            if not all_text.strip():
                st.error("❌ Aucun texte détecté dans les fichiers. Vérifiez vos PDF/DOCX.")
                return

            with st.spinner("Extraction et préparation en cours..."):
                try:
                    # Chunking with config parameters
                    chunks = process_pdf_to_chunks(
                        all_text,
                        chunk_size=settings.CHUNK_SIZE,
                        chunk_overlap=settings.CHUNK_OVERLAP
                    )

                    # Create and persist RAG engine
                    rag_engine_obj = create_index(chunks)
                    st.session_state["rag_engine"] = rag_engine_obj
                    st.session_state["course_ready"] = True

                    st.success(f"🎉 Traitement terminé ! {len(chunks)} sections prêtes pour l'examen.")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Erreur lors de la préparation du cours : {e}")
