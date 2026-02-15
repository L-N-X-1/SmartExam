# =============================================================================
# SMART EXAM 
# =============================================================================

import streamlit as st
from streamlit_option_menu import option_menu

from ui.pages import upload, configure, generate, review, analytics
from core.llm_interface import LLMClient
from config import settings

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="SMART EXAM", layout="wide", page_icon="🧠")

# ----------------------------
# Sidebar menu
# ----------------------------
with st.sidebar:
    selected = option_menu(
        "SMART EXAM",
        ["Upload Documents", "Configurer Examen", "Générer Questions", "Review & Export", "Analytics"],
        icons=['cloud-upload', 'sliders', 'robot', 'eye', 'bar-chart'],
        menu_icon="brain",
        default_index=0
    )

# ----------------------------
# Initialize session state defaults
# ----------------------------
session_defaults = {
    "course_ready": False,
    "course_name": None,
    "exam_config": None,
    "generated_questions": [],
    "generation_complete": False,
    "rag_engine": None,
    "bloom_agents": {},        # safe default for agents
    "llm_client": None,        # will initialize below
}

for key, val in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ----------------------------
# Initialize LLM client (shared across pages)
# ----------------------------
if st.session_state["llm_client"] is None:
    st.session_state["llm_client"] = LLMClient()
llm_client = st.session_state["llm_client"]

# ----------------------------
# Default course name after upload
# ----------------------------
if st.session_state["course_ready"] and not st.session_state["course_name"]:
    st.session_state["course_name"] = "Cours Uploadé"

# ----------------------------
# Page routing dictionary
# ----------------------------
pages = {
    "Upload Documents": upload.show_upload_page,
    "Configurer Examen": configure.show_configure_page,
    "Générer Questions": generate.show_generate_page,
    "Review & Export": review.show_review_page,
    "Analytics": analytics.show_analytics_page,
}

# ----------------------------
# Execute selected page
# ----------------------------
if selected in pages:
    pages[selected]()
else:
    st.title(f"🚧 {selected} – En cours de développement")
    st.write("Cette page sera codée par l'équipe !")
