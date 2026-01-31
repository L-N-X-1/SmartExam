# =============================================================================
# SMART EXAM – Générateur intelligent d'examens basé sur la Taxonomie de Bloom
# Auteur : [Ton nom ou "Équipe SMART EXAM"]
# Date   : Novembre 2025
# 
# À QUOI SERT CE FICHIER :
# Point d'entrée de l'application Streamlit. Intègre tous les agents Bloom.
# =============================================================================
import streamlit as st
from streamlit_option_menu import option_menu
from groq import Groq
import numpy as np
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports RAG & LangGraph Agents
from core.rag_engine import retrieve, get_embedding, create_index, save_index, process_pdf_to_chunks
from agents import (
    RememberAgent, UnderstandAgent, ApplyAgent,
    AnalyzeAgent, EvaluateAgent, CreateAgent,
    CoordinatorAgent, ValidatorAgent
)
from PyPDF2 import PdfReader
import pypdf
from config.settings import GROQ_API_KEY

st.set_page_config(page_title="SMART EXAM", layout="wide", page_icon="🧠")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize LangGraph agents in session state
if "coordinator" not in st.session_state:
    st.session_state.coordinator = CoordinatorAgent(model="openai/gpt-oss-20b")
if "validator" not in st.session_state:
    st.session_state.validator = ValidatorAgent(model="openai/gpt-oss-20b")
if "individual_agents" not in st.session_state:
    st.session_state.individual_agents = {
        "Remember": RememberAgent(model="openai/gpt-oss-20b"),
        "Understand": UnderstandAgent(model="openai/gpt-oss-20b"),
        "Apply": ApplyAgent(model="openai/gpt-oss-20b"),
        "Analyze": AnalyzeAgent(model="openai/gpt-oss-20b"),
        "Evaluate": EvaluateAgent(model="openai/gpt-oss-20b"),
        "Create": CreateAgent(model="openai/gpt-oss-20b")
    }
if "generated_exam" not in st.session_state:
    st.session_state.generated_exam = None
if "current_context" not in st.session_state:
    st.session_state.current_context = ""

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "SMART EXAM",
        ["Upload Documents", "Configurer Examen", "Générer Questions", "Review & Export", "Analytics"],
        icons=['cloud-upload', 'sliders', 'robot', 'eye', 'bar-chart'],
        menu_icon="brain", default_index=0
    )

# Fonction real_llm_generation (Groq)
def real_llm_generation(query, context_chunks, model="openai/gpt-oss-20b"):
    context = "\n\n".join([c['text'] for c in context_chunks]) if context_chunks else ""
    prompt = f"Use ONLY this context: {context}\n{query}"
    
    try:
        response_text = ""
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            stream=True,
            stop=None
        )
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        return response_text
    except Exception as e:
        return f"Error: {str(e)}"

if selected == "Upload Documents":
    st.title("📚 Upload des supports de cours")
    st.write("Glissez-déposez vos PDF, slides, notes...")
    
    uploaded_file = st.file_uploader("Upload PDF for Test", type="pdf")
    if st.button("Process and Index to FAISS"):
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                text = ""
                
                # Try PyPDF2 first
                try:
                    reader = PdfReader(uploaded_file)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                except Exception as e:
                    st.warning(f"⚠️ PyPDF2 extraction failed: {str(e)}, trying pypdf...")
                
                # If PyPDF2 didn't work, try pypdf
                if not text or len(text.strip()) == 0:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        reader = pypdf.PdfReader(uploaded_file)
                        for page in reader.pages:
                            text += page.extract_text() or ""
                    except Exception as e:
                        st.warning(f"⚠️ pypdf extraction failed: {str(e)}")
                
                # Debug: Show extracted text length
                st.write(f"📄 Extracted text length: {len(text)} characters")
                
                if not text or len(text.strip()) == 0:
                    st.error("❌ No text could be extracted from the PDF. The file might be empty or image-based. Try using OCR tools to convert scanned PDFs first.")
                else:
                    # Chunk le texte correctement
                    chunks = process_pdf_to_chunks(text, chunk_size=800, chunk_overlap=150)
                    st.write(f"✅ Created {len(chunks)} chunks")
                    
                    # Créer et sauvegarder l'index FAISS
                    try:
                        create_index(chunks)
                        if save_index():
                            st.success(f"✅ Successfully indexed {len(chunks)} chunks to FAISS!")
                        else:
                            st.error("❌ Error saving index")
                    except ValueError as e:
                        st.error(f"❌ Error creating index: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")

    # Test RAG Retrieval
    st.header("🔍 Test RAG Retrieval")
    test_query = st.text_input("Enter your search query:")
    if st.button("Test Retrieve"):
        if test_query:
            with st.spinner("Searching..."):
                chunks = retrieve(test_query, k=5)
                
                if chunks[0]['text'] == "No index found. Please upload and process documents first.":
                    st.warning("⚠️ No documents indexed yet. Please upload and process a PDF first.")
                else:
                    st.write("**Retrieved Chunks:**")
                    for i, chunk in enumerate(chunks):
                        with st.expander(f"Chunk {i+1} (Distance: {chunk['score']:.4f})"):
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
        
        total_pct = remember_pct + understand_pct + apply_pct + analyze_pct + evaluate_pct + create_pct
        if total_pct != 100:
            st.warning(f"⚠️ Total is {total_pct}% - should be 100%")
    
    with col2:
        st.subheader("⏱️ Exam Settings")
        duration_minutes = st.slider("Duration (minutes)", 30, 480, 120)
        difficulty_level = st.select_slider("Difficulty", ["Easy", "Medium", "Hard"], value="Medium")
        language = st.selectbox("Language", ["English", "French"])
        
        st.subheader("📝 Question Settings")
        include_multiple_choice = st.checkbox("Include Multiple Choice", value=True)
        include_short_answer = st.checkbox("Include Short Answer", value=True)
        include_essay = st.checkbox("Include Essay", value=False)
    
    st.divider()
    
    # Store configuration in session state
    if st.button("✅ Save Configuration", use_container_width=True):
        st.session_state.exam_config = {
            "num_questions": num_questions,
            "bloom_distribution": {
                "Remember": remember_pct,
                "Understand": understand_pct,
                "Apply": apply_pct,
                "Analyze": analyze_pct,
                "Evaluate": evaluate_pct,
                "Create": create_pct
            },
            "duration": duration_minutes,
            "difficulty": difficulty_level,
            "language": language,
            "question_types": {
                "multiple_choice": include_multiple_choice,
                "short_answer": include_short_answer,
                "essay": include_essay
            }
        }
        st.success("✅ Configuration saved successfully!")

elif selected == "Générer Questions":
    st.title("🤖 Question Generation")
    
    if "exam_config" not in st.session_state:
        st.warning("⚠️ Please configure the exam first in 'Configuration' page")
    else:
        config = st.session_state.exam_config
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Current Configuration")
            st.write(f"**Questions**: {config['num_questions']}")
            st.write(f"**Duration**: {config['duration']} minutes")
            st.write(f"**Difficulty**: {config['difficulty']}")
            
            bloom_dist = config['bloom_distribution']
            st.write("**Bloom Distribution**:")
            for level, pct in bloom_dist.items():
                st.write(f"  • {level}: {pct}%")
        
        with col2:
            st.subheader("🚀 Generation")
            if st.button("Generate Exam", use_container_width=True, key="gen_btn"):
                with st.spinner("🔄 Generating questions using AI..."):
                    try:
                        # Retrieve relevant context
                        chunks = retrieve("exam questions educational content", k=10)
                        
                        if chunks[0]['text'] == "No index found. Please upload and process documents first.":
                            st.error("❌ No indexed documents. Please upload PDFs first!")
                        else:
                            # Use coordinator agent to generate exam
                            coordinator = st.session_state.coordinator
                            
                            # Create context for generation
                            context = "\n\n".join([c['text'] for c in chunks])
                            
                            # Generate questions using LangGraph workflow
                            exam_questions = coordinator.generate_exam(
                                num_questions=config['num_questions'],
                                context=context,
                                difficulty=config['difficulty'],
                                language=config['language']
                            )
                            
                            st.session_state.generated_exam = exam_questions
                            st.session_state.current_context = context
                            st.success("✅ Exam generated successfully!")
                            
                            # Display generated questions preview
                            st.subheader("📝 Generated Questions Preview")
                            if isinstance(exam_questions, dict) and 'questions' in exam_questions:
                                for i, q in enumerate(exam_questions['questions'][:3], 1):
                                    with st.expander(f"Question {i}"):
                                        st.write(q)
                                st.info(f"... and {len(exam_questions['questions']) - 3} more questions")
                            else:
                                st.write(exam_questions)
                    
                    except Exception as e:
                        st.error(f"❌ Error generating exam: {str(e)}")

elif selected == "Review & Export":
    st.title("👁️ Review & Export")
    
    if st.session_state.generated_exam is None:
        st.warning("⚠️ No exam generated yet. Go to 'Generate Questions' page first.")
    else:
        exam = st.session_state.generated_exam
        
        st.subheader("📋 Generated Exam")
        
        # Display exam content
        if isinstance(exam, dict) and 'questions' in exam:
            total_questions = len(exam['questions'])
            st.write(f"**Total Questions**: {total_questions}")
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["Questions", "Validation", "Export"])
            
            with tab1:
                for i, question in enumerate(exam['questions'], 1):
                    with st.expander(f"Question {i}"):
                        st.write(question)
            
            with tab2:
                st.subheader("🔍 Validate Questions")
                if st.button("Validate Entire Exam"):
                    with st.spinner("Validating questions..."):
                        try:
                            validator = st.session_state.validator
                            validation_result = validator.validate_exam(exam, st.session_state.current_context)
                            st.session_state.validation_result = validation_result
                            
                            if validation_result.get('valid', False):
                                st.success("✅ Exam passed validation!")
                            else:
                                st.warning("⚠️ Some questions need refinement")
                            
                            st.write(validation_result)
                        except Exception as e:
                            st.error(f"❌ Error validating exam: {str(e)}")
            
            with tab3:
                st.subheader("💾 Export Options")
                
                export_format = st.radio("Select Format", ["PDF", "JSON", "Word", "CSV"])
                
                if st.button("Export Exam"):
                    try:
                        if export_format == "JSON":
                            json_str = json.dumps(exam, ensure_ascii=False, indent=2)
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name="exam.json",
                                mime="application/json"
                            )
                        else:
                            st.info(f"Export to {export_format} format will be implemented soon")
                    except Exception as e:
                        st.error(f"❌ Error exporting: {str(e)}")
        else:
            st.write(exam)

elif selected == "Analytics":
    st.title("📊 Analytics & Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Questions Generated", 0 if st.session_state.generated_exam is None else len(st.session_state.generated_exam.get('questions', [])))
    
    with col2:
        st.metric("Validation Status", "Not started")
    
    with col3:
        st.metric("Documents Indexed", "0")
    
    st.divider()
    
    st.subheader("📈 Generation Statistics")
    
    if st.session_state.generated_exam:
        exam = st.session_state.generated_exam
        if isinstance(exam, dict) and 'questions' in exam:
            st.write(f"**Questions by Type**:")
            st.write(f"  • Total: {len(exam['questions'])}")
            
            if "validation_result" in st.session_state:
                val_result = st.session_state.validation_result
                st.write(f"**Validation Results**:")
                st.write(val_result)
    else:
        st.info("📝 Generate an exam to see analytics")
    
    st.divider()
    
    st.subheader("🧠 Bloom's Taxonomy Distribution")
    st.write("Coming soon: Visualization of questions distribution across Bloom levels")