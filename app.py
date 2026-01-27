# app.py
import streamlit as st
from streamlit_option_menu import option_menu

# Configuration de la page
st.set_page_config(
    page_title="SMART EXAM", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Initialise session_state
if 'course_ready' not in st.session_state:
    st.session_state['course_ready'] = False
if 'exam_config' not in st.session_state:
    st.session_state['exam_config'] = None
if 'generated_questions' not in st.session_state:
    st.session_state['generated_questions'] = []

# Sidebar menu
with st.sidebar:
    st.title("🧠 SMART EXAM")
    st.write("Générateur intelligent d'examens")
    
    selected = option_menu(
        "Navigation",
        ["Upload", "Configurer", "Générer", "Review", "Analytics"],
        icons=['cloud-upload', 'sliders', 'robot', 'eye', 'bar-chart'],
        menu_icon="brain",
        default_index=0
    )
    
    st.divider()
    
    # Status
    if st.session_state.get('course_ready'):
        st.success("✅ Cours chargé")
    else:
        st.info("📚 Aucun cours chargé")

# Router vers les pages
if selected == "Upload":
    from ui.pages.upload import show_upload_page
    show_upload_page()

elif selected == "Configurer":
    from ui.pages.configure import show_configure_page
    show_configure_page()

elif selected == "Générer":
    from ui.pages.generate import show_generate_page
    show_generate_page()

elif selected == "Review":
    st.title("👁️ Review & Export")
    if st.session_state.get('generated_questions'):
        st.write(f"**{len(st.session_state['generated_questions'])} questions générées**")
        for i, q in enumerate(st.session_state['generated_questions'], 1):
            with st.expander(f"Q{i} - {q['level_name']} - {q['marks']} pts"):
                st.write(q['text'])
    else:
        st.info("Aucune question générée pour le moment")

elif selected == "Analytics":
    st.title("📊 Analytics")
    st.info("Page en construction - Groupe 4")