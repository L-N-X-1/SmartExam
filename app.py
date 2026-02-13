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
from groq import Groq  # For Groq API
import numpy as np  # Pour FAISS/retrieval
# Assume tes imports pour RAG (ajoute si besoin)
from core.rag_engine import retrieve, create_index, process_pdf_to_chunks, get_engine
from PyPDF2 import PdfReader  # ✅ Majuscules
import json  # For exam export

# Import the agents
from agents import (
    RememberAgent,
    UnderstandAgent, 
    ApplyAgent,
    AnalyzeAgent,
    EvaluateAgent,
    CreateAgent,
    ValidatorAgent,
    CoordinatorAgent
)

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
if "coordinator" not in st.session_state:
    st.session_state.coordinator = None

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "SMART EXAM",
        ["Upload Documents", "Configurer Examen", "Générer Questions", "Review & Export", "Analytics"],
        icons=['cloud-upload', 'sliders', 'robot', 'eye', 'bar-chart'],
        menu_icon="brain", default_index=0
    )

    # Provider select (ajoute pour choisir Grok)
    provider = st.selectbox("Provider LLM", ['groq', 'grok', 'local'], index=0)
    
    # Auto-reset coordinator if provider changes
    if "last_provider" not in st.session_state:
        st.session_state.last_provider = provider
    if st.session_state.last_provider != provider:
        st.session_state.coordinator = None  # Reset coordinator
        st.session_state.last_provider = provider
        st.info(f"🔄 Switched to {provider} - agents will be recreated")
    
    st.session_state.provider = provider

    # Input API key based on provider
    if st.session_state.provider == 'grok':
        st.text_input("Grok API Key", type="password", key="grok_api_key")
    elif st.session_state.provider == 'groq':
        st.text_input("Groq API Key", type="password", key="groq_api_key")
        st.caption("Get free key: [console.groq.com](https://console.groq.com)")
    elif st.session_state.provider == 'local':
        st.selectbox(
            "Ollama Model",
            ["llama3.2", "llama3.1", "mistral", "qwen2.5", "gemma2", "phi3"],
            key="ollama_model",
            help="Make sure Ollama is running and the model is pulled"
        )
        st.caption("📦 Install: [ollama.com](https://ollama.com) | Run: `ollama run llama3.2`")
    
    # Reset coordinator if key changes (so new agents get created with new key)
    if st.button("🔄 Reset Agents", help="Click after changing API key"):
        st.session_state.coordinator = None
        st.success("Agents will be recreated on next generation")

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


# =============================================================================
# LLM Wrapper for Agents (LangChain-compatible)
# =============================================================================
class GrokLLM:
    """
    Wrapper to make Grok API compatible with LangChain/LangGraph agents.
    Implements the 'invoke' pattern expected by BaseAgent.
    """
    def __init__(self, api_key: str, model: str = "grok-2"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
    
    def invoke(self, messages):
        """LangChain-style invoke method."""
        # Import LangChain message types for proper detection
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        
        # Convert LangChain messages to OpenAI format
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                # Detect message type by class name (more reliable than .type attribute)
                role = "user"
                if isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                elif isinstance(msg, HumanMessage):
                    role = "user"
                formatted_messages.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                formatted_messages.append(msg)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=0.2,
            max_tokens=1500
        )
        
        # Return object with .content attribute for compatibility
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(response.choices[0].message.content)


class GroqLLM:
    """
    Wrapper to make Groq API compatible with LangChain/LangGraph agents.
    Implements the 'invoke' pattern expected by BaseAgent.
    Free tier available at console.groq.com
    """
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=api_key)
    
    def invoke(self, messages):
        """LangChain-style invoke method."""
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        
        # Convert LangChain messages to Groq format
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                role = "user"
                if isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                elif isinstance(msg, HumanMessage):
                    role = "user"
                formatted_messages.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                formatted_messages.append(msg)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=0.2,
            max_tokens=1500
        )
        
        # Return object with .content attribute for compatibility
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(response.choices[0].message.content)


class OllamaLLM:
    """
    Wrapper to make Ollama compatible with LangChain/LangGraph agents.
    Uses OpenAI-compatible API at localhost:11434.
    Install Ollama from https://ollama.com
    """
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434/v1"):
        self.model = model
        self.base_url = base_url
        self.client = OpenAI(
            api_key="ollama",  # Ollama doesn't need a real key
            base_url=base_url
        )
    
    def invoke(self, messages):
        """LangChain-style invoke method."""
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        
        # Convert LangChain messages to OpenAI format
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                role = "user"
                if isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                elif isinstance(msg, HumanMessage):
                    role = "user"
                formatted_messages.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                formatted_messages.append(msg)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=0.2
        )
        
        # Return object with .content attribute for compatibility
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(response.choices[0].message.content)


def get_coordinator():
    """
    Create and cache the CoordinatorAgent with all Bloom-level agents.
    Supports Grok, Groq, and local (Ollama) providers.
    """
    # Check if already cached in session state
    if "coordinator" in st.session_state and st.session_state.coordinator is not None:
        return st.session_state.coordinator
    
    provider = st.session_state.get("provider", "groq")
    
    # Create LLM based on provider
    if provider == "grok":
        if not st.session_state.get("grok_api_key"):
            return None
        llm = GrokLLM(
            api_key=st.session_state.grok_api_key,
            model="grok-2"
        )
    elif provider == "groq":
        if not st.session_state.get("groq_api_key"):
            return None
        llm = GroqLLM(
            api_key=st.session_state.groq_api_key,
            model="llama-3.3-70b-versatile"  # Fast and capable
        )
    elif provider == "local":
        # Use Ollama with the selected model
        model_name = st.session_state.get("ollama_model", "llama3.2")
        llm = OllamaLLM(model=model_name)
    else:
        return None
    
    # Create all Bloom agents (debug=True to see parsing issues)
    agents = {
        "remember": RememberAgent(llm, debug=True),
        "understand": UnderstandAgent(llm, debug=True),
        "apply": ApplyAgent(llm, debug=True),
        "analyze": AnalyzeAgent(llm, debug=True),
        "evaluate": EvaluateAgent(llm, debug=True),
        "create": CreateAgent(llm, debug=True),
    }
    
    # Create validator with lower threshold for better yield
    validator = ValidatorAgent(llm, acceptance_threshold=0.70, debug=True)
    
    # Create coordinator with RAG engine
    rag_engine = get_engine()
    
    coordinator = CoordinatorAgent(
        rag_engine=rag_engine,
        agents=agents,
        validator=validator,
        rag_k=8,
        max_retries=3  # More retries for local LLM
    )
    
    # Cache in session state
    st.session_state.coordinator = coordinator
    
    return coordinator



if selected == "Upload Documents":
    st.title("📚 Upload des supports de cours")
    st.write("Glissez-déposez vos PDF, slides, notes...")
    
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", 
        type="pdf", 
        accept_multiple_files=True,
        help="You can select multiple PDF files at once"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", min_value=50, max_value=2000, value=220, step=10)
    with col2:
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=20, step=10)
    
    # Show current index stats
    from core.rag_engine import get_engine, append_to_index
    engine = get_engine()
    if engine.load_index():
        st.info(f"📊 Current index: **{len(engine.chunks)}** chunks from previous uploads")
    else:
        st.info("📊 No documents indexed yet")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        append_mode = st.checkbox("📎 Append to existing", value=True, help="Add to existing index instead of replacing")
    
    with col_b:
        if st.button("🗑️ Clear Index"):
            import shutil
            try:
                shutil.rmtree("data/vector_store", ignore_errors=True)
                st.success("✅ Index cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing index: {e}")
    
    with col_c:
        process_btn = st.button("⚡ Process and Index", use_container_width=True)

    if process_btn:
        if uploaded_files:
            total_chunks = 0
            with st.spinner("Processing PDFs..."):
                all_chunks = []
                
                for uploaded_file in uploaded_files:
                    st.write(f"📄 Processing: **{uploaded_file.name}**")
                    reader = PdfReader(uploaded_file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    
                    # Chunk with source info
                    chunks = process_pdf_to_chunks(text, chunk_size, chunk_overlap)
                    # Add source filename to chunks
                    for c in chunks:
                        c["source"] = uploaded_file.name
                    
                    all_chunks.extend(chunks)
                    st.write(f"  → Created {len(chunks)} chunks")
                
                # Index all chunks
                if append_mode:
                    added = append_to_index(all_chunks)
                    st.success(f"✅ Added {added} chunks to existing index!")
                else:
                    create_index(all_chunks)
                    st.success(f"✅ Created new index with {len(all_chunks)} chunks!")
                    
                total_chunks = len(all_chunks)
        else:
            st.warning("⚠️ Please upload at least one PDF file")

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
            
            # Topic input for RAG retrieval
            topic = st.text_input("Topic/Subject for questions", value="", placeholder="e.g. VLAN, TCP/IP, Subnetting...")
            
            if st.button("Generate Exam", use_container_width=True):
                # Validate prerequisites based on provider
                provider = st.session_state.get("provider", "groq")
                can_proceed = True
                
                if provider == "grok" and not st.session_state.get("grok_api_key"):
                    st.error("❌ Please enter your Grok API Key in the sidebar first.")
                    can_proceed = False
                elif provider == "groq" and not st.session_state.get("groq_api_key"):
                    st.error("❌ Please enter your Groq API Key in the sidebar first. Get a free key at console.groq.com")
                    can_proceed = False
                
                if not topic.strip():
                    st.error("❌ Please enter a topic for question generation.")
                    can_proceed = False
                
                if can_proceed:
                    with st.spinner("🧠 Generating exam with Bloom Agents..."):
                        try:
                            # Get or create the coordinator
                            coordinator = get_coordinator()
                            
                            if coordinator is None:
                                st.error("❌ Failed to initialize the coordinator. Check your API key.")
                            else:
                                # Convert Bloom distribution percentages to weights (0-1)
                                bloom_weights = {
                                    "remember": config["bloom_distribution"]["Remember"] / 100,
                                    "understand": config["bloom_distribution"]["Understand"] / 100,
                                    "apply": config["bloom_distribution"]["Apply"] / 100,
                                    "analyze": config["bloom_distribution"]["Analyze"] / 100,
                                    "evaluate": config["bloom_distribution"]["Evaluate"] / 100,
                                    "create": config["bloom_distribution"]["Create"] / 100,
                                }
                                
                                # Generate exam using the coordinator
                                result = coordinator.generate_exam(
                                    topic=topic,
                                    total_questions=config["num_questions"],
                                    bloom_weights=bloom_weights
                                )
                                
                                # Extract results
                                questions = result.get("questions", [])
                                report = result.get("report", {})
                                generation_stats = result.get("generation_stats", {})
                                context = result.get("context", "")
                                
                                # Store in session state
                                st.session_state.generated_exam = {
                                    "questions": questions,
                                    "report": report,
                                    "generation_stats": generation_stats,
                                    "topic": topic
                                }
                                st.session_state.current_context = context
                                
                                # Show success message with stats
                                st.success(f"✅ Generated {len(questions)}/{config['num_questions']} questions!")
                                
                                # Show generation stats
                                st.info(f"""
                                📊 **Generation Report:**
                                - Agents Called: {', '.join(generation_stats.get('agents_called', []))}
                                - Total Attempts: {generation_stats.get('total_attempts', 0)}
                                - Successful: {generation_stats.get('successful_generations', 0)}
                                - RAG Chunks Used: {result.get('rag_chunks_used', 0)}
                                """)
                                
                                # Preview first 3 questions
                                st.subheader("📝 Preview")
                                for i, q in enumerate(questions[:3], 1):
                                    bloom_level = q.get("bloom_level", "unknown")
                                    with st.expander(f"Q{i} [{bloom_level.upper()}]"):
                                        st.write(q.get("question", "No question text"))
                                        if q.get("validation"):
                                            val_score = q["validation"].get("final_score", 0)
                                            st.caption(f"Validation Score: {val_score*100:.1f}%")
                                
                        except ValueError as e:
                            st.error(f"❌ {str(e)}")
                        except Exception as e:
                            st.error(f"❌ Generation failed: {str(e)}")
                            if st.checkbox("Show error details"):
                                st.exception(e)

# =========================================================
# PAGE 4 — REVIEW & EXPORT
# =========================================================
elif selected == "Review & Export":
    st.title("👁️ Review & Export")

    exam = st.session_state.generated_exam
    if not exam:
        st.warning("⚠️ No exam generated yet.")
    else:
        # Handle both old format (list of strings) and new format (dict with questions)
        questions = exam.get("questions", exam) if isinstance(exam, dict) else exam
        
        tab1, tab2, tab3 = st.tabs(["Questions", "Validation", "Export"])

        with tab1:
            # Show topic if available
            if isinstance(exam, dict) and exam.get("topic"):
                st.info(f"📚 Topic: **{exam['topic']}**")
            
            for i, q in enumerate(questions, 1):
                # Handle both dict format (new) and string format (old)
                if isinstance(q, dict):
                    bloom_level = q.get("bloom_level", "unknown")
                    question_text = q.get("question", "No question text")
                    validation = q.get("validation", {})
                    
                    with st.expander(f"Question {i} [{bloom_level.upper()}]"):
                        st.write(question_text)
                        if validation:
                            score = validation.get("final_score", 0)
                            st.caption(f"✅ Validation Score: {score*100:.1f}%")
                else:
                    # Old format - simple string
                    with st.expander(f"Question {i}"):
                        st.write(q)

        with tab2:
            st.subheader("📊 Validation Summary")
            
            # Show report if available
            if isinstance(exam, dict) and exam.get("report"):
                report = exam["report"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Requested", report.get("requested", "N/A"))
                    st.metric("Generated", report.get("generated", "N/A"))
                with col2:
                    st.write("**Distribution (Actual):**")
                    for level, count in report.get("distribution_actual", {}).items():
                        st.write(f"• {level.capitalize()}: {count}")
            
            # Show generation stats if available
            if isinstance(exam, dict) and exam.get("generation_stats"):
                stats = exam["generation_stats"]
                st.divider()
                st.write("**Generation Stats:**")
                st.write(f"• Agents Used: {', '.join(stats.get('agents_called', []))}")
                st.write(f"• Total Attempts: {stats.get('total_attempts', 0)}")
                st.write(f"• Successful Generations: {stats.get('successful_generations', 0)}")
                st.write(f"• Failed Generations: {stats.get('failed_generations', 0)}")
            
            if st.button("Re-Validate All Questions"):
                with st.spinner("Validating..."):
                    # Count validated questions
                    validated_count = sum(1 for q in questions if isinstance(q, dict) and q.get("validation", {}).get("accepted", False))
                    st.session_state.validation_result = {
                        "valid": validated_count == len(questions),
                        "validated_count": validated_count,
                        "total_count": len(questions),
                        "notes": f"{validated_count}/{len(questions)} questions passed validation."
                    }
                    if validated_count == len(questions):
                        st.success("✅ All questions passed validation!")
                    else:
                        st.warning(f"⚠️ {len(questions) - validated_count} questions may need review.")
                    st.write(st.session_state.validation_result)

        with tab3:
            export_format = st.radio("Export Format", ["JSON", "Text"])
            if export_format == "JSON":
                st.download_button(
                    "Download JSON",
                    data=json.dumps(exam, indent=2),
                    file_name="exam.json",
                    mime="application/json",
                )
            elif export_format == "Text":
                # Create text version
                text_output = []
                if isinstance(exam, dict) and exam.get("topic"):
                    text_output.append(f"EXAM: {exam['topic']}\n{'='*50}\n")
                for i, q in enumerate(questions, 1):
                    if isinstance(q, dict):
                        bloom = q.get("bloom_level", "").upper()
                        text_output.append(f"{i}. [{bloom}] {q.get('question', '')}\n")
                    else:
                        text_output.append(f"{i}. {q}\n")
                st.download_button(
                    "Download Text",
                    data="\n".join(text_output),
                    file_name="exam.txt",
                    mime="text/plain",
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