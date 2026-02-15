import time
import json
import streamlit as st
from agents.coordinator_agent import CoordinatorAgent
from agents.remember_agent import RememberAgent
from agents.apply_agent import ApplyAgent
from agents.evaluate_agent import EvaluateAgent
from core.rag_engine import RAGEngine
from core.llm_interface import LLMClient
from config import settings

def show_generate_page():
    st.title("🧠 Génération de l'examen")

    # -------------------------------
    # Validate prerequisites
    # -------------------------------
    if not st.session_state.get("course_ready"):
        st.warning("⚠️ Aucun RAG engine trouvé. Indexez un document d'abord dans la page Upload.")
        return

    if "exam_config" not in st.session_state:
        st.warning("⚠️ Configurez l'examen avant de générer.")
        return

    config = st.session_state["exam_config"]
    course_name = config.get("title", "Examen")
    total_q = config.get("total_questions", 5)
    duration = config.get("duration", 60)
    diff = config.get("difficulty", {})

    st.markdown("### 📋 Configuration actuelle")
    st.write(f"**Cours:** {course_name}")
    st.write(f"**Questions:** {total_q}")
    st.write(f"**Durée:** {duration} min")

    # -------------------------------
    # Load RAG Engine
    # -------------------------------
    try:
        rag_engine = st.session_state.get("rag_engine") or RAGEngine.load_engine()
        st.session_state["rag_engine"] = rag_engine

        if not rag_engine or not hasattr(rag_engine, "chunks") or len(rag_engine.chunks) == 0:
            st.error("❌ RAG engine vide. Réindexez vos documents.")
            return

    except Exception as e:
        st.error("❌ Impossible de charger le RAG engine.")
        st.error(str(e))
        return

    # -------------------------------
    # LLM Client
    # -------------------------------
    try:
        llm_client = LLMClient()
    except Exception as e:
        st.error("❌ Impossible d'initialiser le LLM client.")
        st.error(str(e))
        return

    # -------------------------------
    # Normalize difficulty config
    # -------------------------------
    diff_normalized = {k.lower(): v for k, v in diff.items()}  # lowercase keys
    available_levels = [lvl for lvl in ["easy", "medium", "hard"] if diff_normalized.get(lvl, 0) > 0]

    if not available_levels:
        st.error("❌ Aucun niveau Bloom sélectionné dans la configuration.")
        return

    # Map levels to agents
    level_to_agent = {
        "easy": RememberAgent(llm_client),
        "medium": ApplyAgent(llm_client),
        "hard": EvaluateAgent(llm_client)
    }

    # -------------------------------
    # Compute normalized weights
    # -------------------------------
    total_percent = sum(diff_normalized.values())
    bloom_weights = {
        lvl.capitalize(): diff_normalized[lvl] / total_percent
        for lvl in available_levels
    }

    # -------------------------------
    # Coordinator
    # -------------------------------
    coordinator = CoordinatorAgent(
        rag_engine=rag_engine,
        agents={lvl.capitalize(): level_to_agent[lvl] for lvl in available_levels}
    )

    # -------------------------------
    # Allocate questions helper
    # -------------------------------
    def allocate_questions(total_q, weights):
        levels = list(weights.keys())
        raw_counts = {lvl: total_q * w for lvl, w in weights.items()}
        floored = {lvl: int(c) for lvl, c in raw_counts.items()}
        allocated = sum(floored.values())
        remainder = total_q - allocated
        fractions = {lvl: raw_counts[lvl] - floored[lvl] for lvl in levels}
        for lvl in sorted(fractions, key=fractions.get, reverse=True):
            if remainder <= 0:
                break
            floored[lvl] += 1
            remainder -= 1
        return floored

    # -------------------------------
    # Generate questions safe
    # -------------------------------
    def generate_questions_safe(total_questions):
        questions = []
        per_level_counts = allocate_questions(total_questions, bloom_weights)

        st.write("✅ Allocation par niveau:", per_level_counts)  # debug

        for level, num_q in per_level_counts.items():
            if num_q <= 0:
                continue
            try:
                result = coordinator.generate_exam(
                    topic=course_name,
                    total_questions=num_q,
                    bloom_weights={level: 1.0}  # full weight for this level
                )

                # -------------------------
                # UNIVERSAL RESPONSE PARSER
                # -------------------------
                if not result:
                    st.warning(f"⚠️ Aucun résultat retourné pour {level}")
                    continue

                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except:
                        questions.append({
                            "question": result.strip(),
                            "bloom_level": level,
                            "difficulty": level,
                            "type": "Question ouverte",
                            "marks": 1
                        })
                        continue

                if isinstance(result, dict) and "questions" in result:
                    raw_questions = result["questions"]
                elif isinstance(result, list):
                    raw_questions = result
                elif isinstance(result, dict):
                    raw_questions = [result]
                else:
                    raw_questions = [{"question": str(result)}]

                for q in raw_questions:
                    if isinstance(q, str):
                        q = {"question": q}
                    if not isinstance(q, dict):
                        q = {"question": str(q)}

                    q.setdefault("bloom_level", level)
                    q.setdefault("difficulty", level)
                    q.setdefault("type", "Question ouverte")
                    q.setdefault("marks", 1)

                    questions.append(q)

            except Exception as e:
                st.warning(f"⚠️ Échec génération pour {level}: {str(e)}")

        return questions

    # -------------------------------
    # Generate Button
    # -------------------------------
    if st.button("🚀 Générer l'examen"):
        st.info(f"Mode {settings.LLM_PROVIDER.capitalize()} activé ⚡ Génération en cours...")
        questions = generate_questions_safe(total_q)

        if questions:
            st.session_state["generated_questions"] = questions
            st.success(f"✅ {len(questions)} questions générées !")
        else:
            st.error("❌ Impossible de générer des questions avec le modèle actuel.")

    # -------------------------------
    # Preview
    # -------------------------------
    questions = st.session_state.get("generated_questions", [])
    if questions:
        st.markdown("### 👀 Aperçu (3 premières questions)")
        for i, q in enumerate(questions[:3]):
            with st.expander(f"Question {i+1} (Bloom {q.get('bloom_level', '?')})"):
                st.write(q.get("question") or q.get("text") or "❌ Question vide")
