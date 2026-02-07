# =============================================================================
# SMART EXAM – Générateur intelligent d'examens basé sur la Taxonomie de Bloom
# Auteur : Équipe SMART EXAM
# Date   : Novembre 2025
#
# À QUOI SERT CE FICHIER :
# Point d'entrée de l'application Streamlit. Gère le menu multi-page
# (Upload → Configurer → Générer → Review → Analytics).
# =============================================================================

import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI  # Pour Grok
import numpy as np  # Pour FAISS/retrieval
from PyPDF2 import PdfReader

# Imports RAG
from core.rag_engine import (
    retrieve, 
    get_embedding, 
    create_index, 
    save_index, 
    process_pdf_to_chunks
)

# Imports LangChain
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# ✅ UNIQUEMENT UnderstandAgent
from agents.understand_agent import create_understand_agent

# =============================================================================
# Configuration Streamlit
# =============================================================================

st.set_page_config(
    page_title="SMART EXAM", 
    layout="wide", 
    page_icon="🧠"
)

# =============================================================================
# Sidebar Menu + Provider Configuration
# =============================================================================

with st.sidebar:
    # Menu principal
    selected = option_menu(
        "SMART EXAM",
        [
            "Upload Documents", 
            "Configurer Examen", 
            "Générer Questions", 
            "Review & Export", 
            "Analytics"
        ],
        icons=['cloud-upload', 'sliders', 'robot', 'eye', 'bar-chart'],
        menu_icon="brain", 
        default_index=0
    )

    st.markdown("---")
    
    # Provider LLM
    providers = ['grok', 'local']
    
    # Vérifier si Groq est disponible
    try:
        from groq import Groq
        providers.insert(1, 'groq')
        HAS_GROQ = True
    except ModuleNotFoundError:
        HAS_GROQ = False
    
    provider = st.selectbox("🤖 Provider LLM", providers, index=0)
    st.session_state.provider = provider

    # Configuration selon le provider
    if provider == 'grok':
        api_key = st.text_input(
            "🔑 Grok API Key", 
            type="password", 
            key="grok_api_key",
            help="Obtenez votre clé sur https://x.ai"
        )
        if not api_key:
            st.warning("⚠️ Veuillez entrer votre clé Grok API")
    
    elif provider == 'groq':
        api_key = st.text_input(
            "🔑 Groq API Key", 
            type="password", 
            key="groq_api_key",
            help="Obtenez votre clé sur https://console.groq.com"
        )
        if not api_key:
            st.warning("⚠️ Veuillez entrer votre clé Groq API")
        
        # Sélection du modèle Groq
        groq_model = st.selectbox(
            "📦 Modèle Groq",
            [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "deepseek-r1-distill-llama-70b"
            ],
            help="Choisissez le modèle Groq à utiliser"
        )
        st.session_state.groq_model = groq_model
    
    elif provider == 'local':
        st.info("ℹ️ Mode local - Pas d'API requise")

# =============================================================================
# Fonction : Créer LLM LangChain
# =============================================================================

def get_langchain_llm():
    """
    Retourne une instance LLM LangChain selon le provider sélectionné
    
    Returns:
        LLM LangChain ou None si erreur
    """
    provider = st.session_state.get("provider", "grok")
    
    if provider == "grok":
        api_key = st.session_state.get("grok_api_key")
        if not api_key:
            st.error("❌ Grok API key manquante")
            return None
        
        try:
            return ChatOpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1",
                model="grok-2",
                temperature=0.7
            )
        except Exception as e:
            st.error(f"❌ Erreur création LLM Grok: {e}")
            return None
    
    elif provider == "groq":
        api_key = st.session_state.get("groq_api_key")
        if not api_key:
            st.error("❌ Groq API key manquante")
            return None
        
        model = st.session_state.get("groq_model", "llama-3.1-8b-instant")
        
        try:
            return ChatGroq(
                api_key=api_key,
                model=model,
                temperature=0.7
            )
        except Exception as e:
            st.error(f"❌ Erreur création LLM Groq: {e}")
            return None
    
    elif provider == "local":
        st.warning("⚠️ Provider local non encore implémenté")
        return None
    
    else:
        st.error(f"❌ Provider '{provider}' non supporté")
        return None

# =============================================================================
# Initialisation de l'agent UnderstandAgent UNIQUEMENT
# =============================================================================

def initialize_agent():
    """Initialise l'agent UnderstandAgent avec le LLM approprié"""
    if "understand_agent" not in st.session_state:
        llm = get_langchain_llm()
        if llm:
            st.session_state.understand_agent = create_understand_agent(llm, debug=True)
            return True
        return False
    return True

# =============================================================================
# PAGE 1 : Upload Documents
# =============================================================================

if selected == "Upload Documents":
    st.title("📚 Upload des supports de cours")
    st.write("Glissez-déposez vos PDF, slides, notes de cours...")
    
    # Upload PDF
    uploaded_file = st.file_uploader(
        "Choisissez un fichier PDF", 
        type="pdf",
        help="Les documents seront indexés dans FAISS pour la recherche RAG"
    )
    
    # Bouton d'indexation
    col1, col2 = st.columns([1, 3])
    with col1:
        process_button = st.button("🚀 Traiter et Indexer", type="primary")
    
    if process_button:
        if uploaded_file:
            with st.spinner("📖 Traitement du PDF en cours..."):
                try:
                    # Lire le PDF
                    reader = PdfReader(uploaded_file)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    
                    if not text.strip():
                        st.error("❌ Le PDF semble vide ou non lisible")
                    else:
                        # Découper en chunks
                        chunks = process_pdf_to_chunks(
                            text, 
                            chunk_size=800, 
                            chunk_overlap=150
                        )
                        
                        st.info(f"📄 {len(chunks)} chunks créés")
                        
                        # Créer l'index FAISS
                        create_index(chunks)
                        
                        # Sauvegarder l'index
                        if save_index():
                            st.success(f"✅ {len(chunks)} chunks indexés avec succès dans FAISS!")
                            st.balloons()
                        else:
                            st.error("❌ Erreur lors de la sauvegarde de l'index")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement: {str(e)}")
        else:
            st.warning("⚠️ Veuillez d'abord uploader un fichier PDF")
    
    # Section de test RAG
test_query = st.text_input("Enter your search query:")
if st.button("Test Retrieve"):
    if test_query:
        with st.spinner("Searching..."):
            # ⚡ Forcer le chargement de l'index FAISS
            from core import rag_engine
            engine = rag_engine._get_engine()  # récupère ou crée l'engine
            engine.load_index()  # charge l'index existant depuis le disque

            chunks = rag_engine.retrieve(test_query, k=5)
            
            if not chunks or chunks[0]['text'].startswith("No index found"):
                st.warning("⚠️ Aucun document indexé. Veuillez d'abord uploader et traiter un PDF.")
            else:
                st.write("**Retrieved Chunks:**")
                for i, chunk in enumerate(chunks):
                    score = chunk.get('similarity', None)
                    title = f"Chunk {i+1}" + (f" (Distance: {score:.4f})" if score is not None else "")
                    with st.expander(title):
                        st.write(chunk.get('text', ''))
    else:
        st.warning("⚠️ Veuillez entrer une requête")

# =============================================================================
# PAGE 2 : Configurer Examen
# =============================================================================

elif selected == "Configurer Examen":
    st.title("⚙️ Configuration de l'examen")
    st.write("Définissez les paramètres de votre examen")
    
    # Paramètres généraux
    st.header("📋 Paramètres généraux")
    
    col1, col2 = st.columns(2)
    
    with col1:
        exam_title = st.text_input(
            "Titre de l'examen",
            value=st.session_state.get("exam_title", "Examen de test"),
            help="Donnez un titre à votre examen"
        )
        
        duration = st.slider(
            "Durée (minutes)",
            min_value=10,
            max_value=180,
            value=st.session_state.get("exam_duration", 60),
            step=5,
            help="Durée totale de l'examen"
        )
    
    with col2:
        num_questions = st.slider(
            "Nombre total de questions",
            min_value=5,
            max_value=50,
            value=st.session_state.get("exam_num_questions", 20),
            help="Nombre de questions à générer"
        )
        
        difficulty = st.select_slider(
            "Niveau de difficulté",
            options=["Facile", "Moyen", "Difficile"],
            value=st.session_state.get("exam_difficulty", "Moyen")
        )
    
    # ✅ SIMPLIFICATION: Uniquement niveau Understand
    st.markdown("---")
    st.header("🎓 Niveau Bloom")
    st.info("ℹ️ Cette version utilise uniquement le niveau **Understand** (Compréhension)")
    
    # Niveau Understand toujours actif
    understand = True
    
    # Options avancées
    st.markdown("---")
    st.header("🔧 Options avancées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        randomize_order = st.checkbox(
            "Randomiser l'ordre des questions",
            value=st.session_state.get("randomize_order", True),
            help="Mélange aléatoire des questions"
        )
        
        include_solutions = st.checkbox(
            "Inclure les solutions",
            value=st.session_state.get("include_solutions", False),
            help="Génère aussi les réponses"
        )
    
    with col2:
        time_per_question = st.number_input(
            "Temps par question (minutes)",
            min_value=1,
            max_value=10,
            value=st.session_state.get("time_per_question", 3),
            help="Temps estimé par question"
        )
    
    # Bouton de sauvegarde
    st.markdown("---")
    if st.button("💾 Sauvegarder la configuration", type="primary"):
        # Construire le dict de config
        exam_config = {
            "title": exam_title,
            "duration": duration,
            "num_questions": num_questions,
            "difficulty": difficulty,
            "bloom_levels": {
                "Understand": True  # ✅ Uniquement Understand
            },
            "randomize_order": randomize_order,
            "include_solutions": include_solutions,
            "time_per_question": time_per_question
        }
        
        # Sauvegarder dans session_state
        st.session_state.exam_config = exam_config
        st.session_state.exam_title = exam_title
        st.session_state.exam_duration = duration
        st.session_state.exam_num_questions = num_questions
        st.session_state.exam_difficulty = difficulty
        st.session_state.randomize_order = randomize_order
        st.session_state.include_solutions = include_solutions
        st.session_state.time_per_question = time_per_question
        
        st.success("✅ Configuration enregistrée avec succès!")
        st.balloons()
        
        # Afficher récapitulatif
        with st.expander("📋 Récapitulatif de la configuration", expanded=True):
            st.json(exam_config)

# =============================================================================
# PAGE 3 : Générer Questions (UNIQUEMENT Understand)
# =============================================================================

elif selected == "Générer Questions":
    st.title("🤖 Génération intelligente des questions")
    st.write("Utilise RAG + Bloom (niveau Understand) + LangChain pour générer des questions automatiquement")
    
    # Vérifier que des documents sont indexés
    test_chunks = retrieve("test", k=1)
    if test_chunks and test_chunks[0]['text'] == "No index found. Please upload and process documents first.":
        st.warning("⚠️ Aucun document indexé. Allez dans 'Upload Documents' pour charger un PDF.")
        st.stop()
    
    # Récupérer la config
    exam_config = st.session_state.get("exam_config", {
        "num_questions": 5,
        "duration": 60,
        "randomize_order": True,
        "bloom_levels": {"Understand": True}
    })
    
    # Interface de génération
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input(
            "📌 Sujet ou requête :",
            placeholder="Ex: optimisation en machine learning",
            help="Décrivez le sujet des questions"
        )
        
        topic = st.text_input(
            "🎯 Thème (optionnel) :",
            placeholder="Ex: Apprentissage profond",
            help="Contexte plus large du sujet"
        )
    
    with col2:
        num_q = st.number_input(
            "Nombre de questions",
            min_value=1,
            max_value=20,
            value=min(exam_config.get("num_questions", 5), 10),
            help="Nombre de questions à générer"
        )
        
        k = st.number_input(
            "Chunks RAG",
            min_value=1,
            max_value=15,
            value=5,
            help="Nombre de chunks de contexte"
        )
    
    # ✅ Niveau Bloom fixé à Understand
    st.info("🎓 Niveau Bloom: **Understand** (Compréhension)")
    bloom_level = "Understand"
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        use_duration = st.checkbox(
            "Utiliser la durée pour calculer le nombre",
            value=False,
            help="Calcule automatiquement le nombre de questions selon la durée"
        )
    
    with col2:
        randomize = st.checkbox(
            "Mélanger l'ordre",
            value=exam_config.get("randomize_order", True)
        )
    
    duration = exam_config.get("duration", 60) if use_duration else None
    
    # Bouton de génération
    st.markdown("---")
    
    if st.button("🚀 Générer les questions", type="primary"):
        if not query:
            st.warning("⚠️ Veuillez entrer un sujet")
        else:
            # Initialiser l'agent si nécessaire
            if not initialize_agent():
                st.error("❌ Impossible d'initialiser l'agent. Vérifiez votre configuration LLM.")
                st.stop()
            
            agent = st.session_state.understand_agent
            
            # Génération
            with st.spinner(f"🤖 Génération de {num_q} questions en cours..."):
                try:
                    questions = agent.generate_questions(
                        query=query,
                        topic=topic if topic else None,
                        num_questions=num_q,
                        k=k,
                        duration=duration,
                        randomize_order=randomize
                    )
                    
                    if not questions:
                        st.error("❌ Aucune question générée. Vérifiez votre contexte RAG.")
                    else:
                        # Sauvegarder les questions
                        st.session_state.generated_questions = questions
                        
                        # Affichage
                        st.success(f"✅ {len(questions)} questions générées avec succès!")
                        st.balloons()
                        
                        st.markdown("---")
                        st.header("📝 Questions générées")
                        
                        for i, q in enumerate(questions, 1):
                            with st.container():
                                st.markdown(f"### Q{i}. {q}")
                                
                                # Boutons d'action par question
                                col1, col2, col3 = st.columns([1, 1, 3])
                                with col1:
                                    if st.button("✏️ Éditer", key=f"edit_{i}"):
                                        st.info("Fonctionnalité d'édition à venir")
                                with col2:
                                    if st.button("🗑️ Supprimer", key=f"del_{i}"):
                                        st.info("Fonctionnalité de suppression à venir")
                                
                                st.markdown("---")
                        
                        # Bouton d'export
                        st.markdown("### 💾 Export")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            txt_content = "\n\n".join([f"Q{i}. {q}" for i, q in enumerate(questions, 1)])
                            st.download_button(
                                "📄 Télécharger TXT",
                                txt_content,
                                file_name="questions.txt",
                                mime="text/plain"
                            )
                        
                        with col2:
                            import json
                            json_content = json.dumps({
                                "questions": questions,
                                "config": {
                                    "query": query,
                                    "topic": topic,
                                    "bloom_level": bloom_level,
                                    "num_questions": len(questions)
                                }
                            }, indent=2, ensure_ascii=False)
                            st.download_button(
                                "📋 Télécharger JSON",
                                json_content,
                                file_name="questions.json",
                                mime="application/json"
                            )
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération: {str(e)}")
                    import traceback
                    with st.expander("🔍 Détails de l'erreur"):
                        st.code(traceback.format_exc())

# =============================================================================
# PAGE 4 : Review & Export
# =============================================================================

elif selected == "Review & Export":
    st.title("👁️ Review & Export")
    st.write("Relisez et exportez vos questions")
    
    # Vérifier s'il y a des questions générées
    if "generated_questions" not in st.session_state or not st.session_state.generated_questions:
        st.info("ℹ️ Aucune question à afficher. Allez dans 'Générer Questions' pour en créer.")
        st.stop()
    
    questions = st.session_state.generated_questions
    
    st.success(f"📊 {len(questions)} questions disponibles")
    
    # Affichage des questions
    for i, q in enumerate(questions, 1):
        with st.expander(f"Question {i}", expanded=(i == 1)):
            st.write(q)
            
            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✏️ Modifier", key=f"mod_{i}"):
                    st.info("Fonctionnalité à venir")
            with col2:
                if st.button("❌ Retirer", key=f"rem_{i}"):
                    st.info("Fonctionnalité à venir")
    
    # Export
    st.markdown("---")
    st.header("💾 Options d'export")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        txt = "\n\n".join([f"Q{i}. {q}" for i, q in enumerate(questions, 1)])
        st.download_button("📄 Export TXT", txt, "exam.txt", "text/plain")
    
    with col2:
        import json
        json_data = json.dumps({"questions": questions}, indent=2, ensure_ascii=False)
        st.download_button("📋 Export JSON", json_data, "exam.json", "application/json")
    
    with col3:
        st.button("📊 Export PDF", disabled=True, help="À venir")
    
    with col4:
        st.button("📝 Export DOCX", disabled=True, help="À venir")

# =============================================================================
# PAGE 5 : Analytics
# =============================================================================

elif selected == "Analytics":
    st.title("📊 Analytics")
    st.write("Statistiques et analyses de vos examens")
    
    st.info("🚧 Cette page est en cours de développement")
    
    # Placeholder pour analytics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📚 Documents indexés", "1")
    
    with col2:
        st.metric("❓ Questions générées", 
                 st.session_state.get("generated_questions", []) and 
                 len(st.session_state.generated_questions) or 0)
    
    with col3:
        st.metric("🎓 Niveau Bloom", "Understand")
    
    st.markdown("---")
    st.write("Fonctionnalités à venir:")
    st.write("- Distribution des types de questions")
    st.write("- Statistiques de génération")
    st.write("- Historique des examens")
    st.write("- Analyse de qualité des questions")

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>SMART EXAM - Niveau Understand | Powered by LangChain & RAG</p>
    </div>
    """,
    unsafe_allow_html=True
)