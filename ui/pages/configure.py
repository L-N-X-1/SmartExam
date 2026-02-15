# ui/pages/configure.py
import streamlit as st
from typing import Optional

def clean_input(s: Optional[str]) -> Optional[str]:
    """Strip string and return None if empty or None."""
    return s.strip() if s and s.strip() else None

def show_configure_page():
    st.title("⚙️ Configuration de l'Examen")
    
    # --- Check if course is uploaded ---
    if not st.session_state.get('course_ready', False):
        st.warning("⚠️ Veuillez d'abord uploader un document dans la page **Upload**")
        return
    
    st.success("📚 Cours prêt à être utilisé !")

    # --- Load previous config safely ---
    prev_config = st.session_state.get('exam_config', {})
    if prev_config is None:
        prev_config = {}

    # --- Use form to prevent multiple re-renders ---
    with st.form("exam_config_form"):
        # --- Basic exam info ---
        st.subheader("📝 Informations de base")
        exam_title = st.text_input(
            "Titre de l'examen", 
            value=prev_config.get('title', "Examen Final")
        )
        exam_duration = st.number_input(
            "Durée (minutes)", 
            min_value=30, max_value=180, 
            value=prev_config.get('duration', 90), step=5
        )

        # --- Total questions ---
        st.subheader("📄 Nombre total de questions")
        total_questions = st.number_input(
            "Nombre de questions", 
            min_value=1, max_value=50, 
            value=prev_config.get('total_questions', 20), step=1
        )

        # --- Difficulty selection ---
        st.subheader("⚡ Répartition par difficulté")
        st.write("La somme doit être 100%.")
        difficulty_prev = prev_config.get('difficulty', {})
        easy_pct = st.slider(
            "Facile (%)", min_value=0, max_value=100, 
            value=difficulty_prev.get('easy', 40), step=5
        )
        medium_pct = st.slider(
            "Moyen (%)", min_value=0, max_value=100, 
            value=difficulty_prev.get('medium', 40), step=5
        )
        hard_pct = st.slider(
            "Difficile (%)", min_value=0, max_value=100, 
            value=difficulty_prev.get('hard', 20), step=5
        )

        total_pct = easy_pct + medium_pct + hard_pct
        if total_pct != 100:
            st.warning(f"⚠️ La somme des pourcentages doit être 100%. Actuellement: {total_pct}%")

        # --- Optional topic ---
        st.subheader("🏷️ Sujet spécifique (optionnel)")
        topic = st.text_input(
            "Sujet", 
            value=prev_config.get('topic', ""),
            placeholder="Ex: Les dérivées"
        )

        # --- Submit button ---
        submitted = st.form_submit_button("🚀 Préparer l'examen")
        if submitted:
            if total_pct != 100:
                st.error("❌ La somme des pourcentages doit être exactement 100% pour sauvegarder la configuration.")
            else:
                # Save config safely
                st.session_state['exam_config'] = {
                    'title': clean_input(exam_title) or "Examen Final",
                    'duration': exam_duration,
                    'total_questions': total_questions,
                    'difficulty': {
                        'easy': easy_pct,
                        'medium': medium_pct,
                        'hard': hard_pct
                    },
                    'topic': clean_input(topic)
                }
                st.success("✅ Configuration sauvegardée !")
                st.info("👉 Passez à la page **Générer Questions** pour créer l'examen")
