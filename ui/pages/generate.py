"""
Page de génération des questions
"""

import streamlit as st
from agents.coordinator_agent import CoordinatorAgent


def show_generate_page():
    st.title("🤖 Génération des Questions")

    # Vérifie les prérequis
    if not st.session_state.get('course_ready', False):
        st.warning("⚠️ Uploadez d'abord un document")
        return

    if not st.session_state.get('exam_config'):
        st.warning("⚠️ Configurez d'abord l'examen")
        return

    config = st.session_state['exam_config']
    course_name = st.session_state['course_name']

    # Affiche la config
    st.subheader("📋 Configuration Actuelle")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cours", course_name)
    with col2:
        st.metric("Questions", sum(config['distribution'].values()))
    with col3:
        st.metric("Durée", f"{config['duration']} min")

    # Bouton de génération
    if st.button("✨ Générer Maintenant", type="primary"):

        with st.spinner("🤖 Les agents Bloom travaillent..."):
            try:
                coordinator = CoordinatorAgent()

                questions = coordinator.generate_exam_questions(
                    course_name=course_name,
                    distribution=config['distribution'],
                    topic=config.get('topic')
                )

                st.session_state['generated_questions'] = questions
                st.session_state['generation_complete'] = True

                st.success(f"✅ {len(questions)} questions générées !")

                # Aperçu
                st.subheader("📝 Aperçu des Questions")

                for i, q in enumerate(questions[:5], 1):
                    # 🔒 SAFE ACCESS
                    level = (
                        q.get("level_name")
                        or q.get("bloom_level")
                        or q.get("difficulty", "N/A")
                    )

                    with st.expander(f"Question {i} - Niveau {level}"):
                        st.write(f"**{q.get('text', 'Question sans texte')}**")

                        st.write(
                            f"Type: {q.get('type', 'N/A')} | "
                            f"Points: {q.get('marks', 'N/A')} | "
                            f"Difficulté: {q.get('difficulty', 'N/A')}"
                        )

                if len(questions) > 5:
                    st.info(f"... et {len(questions) - 5} autres questions")

                st.info("👉 Passez à **Review & Export** pour voir toutes les questions")

            except Exception as e:
                st.error(f"❌ Erreur : {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    show_generate_page()
