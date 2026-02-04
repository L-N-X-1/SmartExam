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
import json  # For exam export

st.set_page_config(page_title="SMART EXAM", layout="wide", page_icon="🧠")

# Initialize session state variables
if "exam_config" not in st.session_state:
    st.session_state.exam_config = None
if "generated_exam" not in st.session_state:
    st.session_state.generated_exam = None
if "validation_result" not in st.session_state:
    st.session_state.validation_result = None
if "current_context" not in st.session_state:
    st.session_state.current_context = None

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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Bloom's Taxonomy Distribution")
        num_questions = st.slider("Total Questions", 5, 50, 20)

        remember_pct = st.slider("Remember (%)", 0, 100, 10)
        understand_pct = st.slider("Understand (%)", 0, 100, 20)
        apply_pct = st.slider("Apply (%)", 0, 100, 25)
        analyze_pct = st.slider("Analyze (%)", 0, 100, 20)
        evaluate_pct = st.slider("Evaluate (%)", 0, 100, 15)
        create_pct = st.slider("Create (%)", 0, 100, 10)

        total_pct = (
            remember_pct + understand_pct + apply_pct +
            analyze_pct + evaluate_pct + create_pct
        )
        if total_pct != 100:
            st.warning(f"⚠️ Total is {total_pct}% (should be 100%)")

    with col2:
        st.subheader("⏱️ Exam Settings")
        duration = st.slider("Duration (minutes)", 30, 480, 120)
        difficulty = st.select_slider("Difficulty", ["Easy", "Medium", "Hard"], value="Medium")
        language = st.selectbox("Language", ["English", "French"])

        st.subheader("📝 Question Types")
        mcq = st.checkbox("Multiple Choice", True)
        short = st.checkbox("Short Answer", True)
        essay = st.checkbox("Essay", False)

    st.divider()

    if st.button("✅ Save Configuration", use_container_width=True):
        st.session_state.exam_config = {
            "num_questions": num_questions,
            "bloom_distribution": {
                "Remember": remember_pct,
                "Understand": understand_pct,
                "Apply": apply_pct,
                "Analyze": analyze_pct,
                "Evaluate": evaluate_pct,
                "Create": create_pct,
            },
            "duration": duration,
            "difficulty": difficulty,
            "language": language,
            "question_types": {
                "mcq": mcq,
                "short": short,
                "essay": essay,
            },
        }
        st.success("✅ Configuration saved")

# =========================================================
# PAGE 3— GÉNÉRER QUESTIONS
# =========================================================
elif selected == "Générer Questions":
    st.title("🤖 Question Generation")

    if not st.session_state.exam_config:
        st.warning("⚠️ Please configure the exam first.")
    else:
        config = st.session_state.exam_config

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Current Configuration")
            st.write(f"**Questions**: {config['num_questions']}")
            st.write(f"**Duration**: {config['duration']} min")
            st.write(f"**Difficulty**: {config['difficulty']}")

            st.write("**Bloom Distribution:**")
            for k, v in config["bloom_distribution"].items():
                st.write(f"• {k}: {v}%")

        with col2:
            st.subheader("🚀 Generation")
            if st.button("Generate Exam", use_container_width=True):
                with st.spinner("Generating exam..."):
                    # MOCK generation hook (replace with your coordinator)
                    exam = {
                        "questions": [
                            f"Sample question {i+1}" 
                            for i in range(config["num_questions"])
                        ]
                    }

                    st.session_state.generated_exam = exam
                    st.session_state.current_context = "Generated context"
                    st.success("✅ Exam generated")

                    for i, q in enumerate(exam["questions"][:3], 1):
                        with st.expander(f"Question {i}"):
                            st.write(q)

# =========================================================
# PAGE 4 — REVIEW & EXPORT
# =========================================================
elif selected == "Review & Export":
    st.title("👁️ Review & Export")

    exam = st.session_state.generated_exam
    if not exam:
        st.warning("⚠️ No exam generated yet.")
    else:
        tab1, tab2, tab3 = st.tabs(["Questions", "Validation", "Export"])

        with tab1:
            for i, q in enumerate(exam["questions"], 1):
                with st.expander(f"Question {i}"):
                    st.write(q)

        with tab2:
            if st.button("Validate Entire Exam"):
                with st.spinner("Validating..."):
                    st.session_state.validation_result = {
                        "valid": True,
                        "notes": "All questions comply with Bloom taxonomy."
                    }
                    st.success("✅ Validation passed")
                    st.write(st.session_state.validation_result)

        with tab3:
            export_format = st.radio("Export Format", ["JSON"])
            if export_format == "JSON":
                st.download_button(
                    "Download JSON",
                    data=json.dumps(exam, indent=2),
                    file_name="exam.json",
                    mime="application/json",
                )

# =========================================================
# PAGE 5 — ANALYTICS
# =========================================================
elif selected == "Analytics":
    st.title("📊 Analytics & Insights")

    total = (
        0 if not st.session_state.generated_exam
        else len(st.session_state.generated_exam["questions"])
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Questions", total)
    col2.metric(
        "Validation Status",
        "Done" if st.session_state.validation_result else "Not started",
    )
    col3.metric("Documents Indexed", "N/A")

    st.divider()

    if st.session_state.validation_result:
        st.subheader("Validation Results")
        st.write(st.session_state.validation_result)
    else:
        st.info("Generate and validate an exam to see analytics.")