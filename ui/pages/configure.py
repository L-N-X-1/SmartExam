# ui/pages/configure.py
"""
Page de configuration de l'examen
"""

import streamlit as st

def show_configure_page():
    st.title("⚙️ Configuration de l'Examen")
    
    # Vérifie qu'un cours est chargé
    if not st.session_state.get('course_ready', False):
        st.warning("⚠️ Veuillez d'abord uploader un document dans la page **Upload**")
        return
    
    st.success(f"📚 Cours chargé : **{st.session_state['course_name']}**")
    
    # Configuration de base
    st.subheader("📝 Informations Générales")
    
    col1, col2 = st.columns(2)
    with col1:
        exam_title = st.text_input("Titre de l'examen", value="Examen Final")
    with col2:
        exam_duration = st.number_input("Durée (minutes)", min_value=30, max_value=240, value=120)
    
    # Distribution Bloom
    st.subheader("🎯 Distribution des Niveaux Bloom")
    st.write("Définissez combien de questions pour chaque niveau :")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        level1 = st.number_input("1️⃣ Remember", min_value=0, max_value=50, value=10)
        level2 = st.number_input("2️⃣ Understand", min_value=0, max_value=50, value=15)
    
    with col2:
        level3 = st.number_input("3️⃣ Apply", min_value=0, max_value=50, value=10)
        level4 = st.number_input("4️⃣ Analyze", min_value=0, max_value=50, value=5)
    
    with col3:
        level5 = st.number_input("5️⃣ Evaluate", min_value=0, max_value=50, value=5)
        level6 = st.number_input("6️⃣ Create", min_value=0, max_value=50, value=5)
    
    total_questions = level1 + level2 + level3 + level4 + level5 + level6
    st.metric("📊 Total de questions", total_questions)
    
    # Sujet optionnel
    topic = st.text_input("Sujet spécifique (optionnel)", placeholder="Ex: Les dérivées")
    
    # Bouton de génération
    if st.button("🚀 Générer les Questions", type="primary"):
        if total_questions == 0:
            st.error("❌ Configurez au moins 1 question !")
            return
        
        # Sauvegarde la config
        st.session_state['exam_config'] = {
            'title': exam_title,
            'duration': exam_duration,
            'distribution': {
                1: level1, 2: level2, 3: level3,
                4: level4, 5: level5, 6: level6
            },
            'topic': topic if topic else None
        }
        
        st.success("✅ Configuration sauvegardée !")
        st.info("👉 Passez à la page **Générer Questions**")

if __name__ == "__main__":
    show_configure_page()