# =============================================================================
# SMART EXAM – Générateur intelligent d'examens basé sur la Taxonomie de Bloom
# Auteur : [Ton nom ou "Équipe SMART EXAM"]
# Date   : Novembre 2025
# 
# À QUOI SERT CE FICHIER :
# [TU COPIES LE TEXTE SPÉCIFIQUE CI-DESSOUS SELON LE FICHIER]
# 
# Tâche assignée à → [Nom(s) de l'étudiant ou du groupe]
# =============================================================================
# app.py
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="SMART EXAM", layout="wide", page_icon="🧠")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "SMART EXAM",
        ["Upload Documents", "Configurer Examen", "Générer Questions", "Review & Export", "Analytics"],
        icons=['cloud-upload', 'sliders', 'robot', 'eye', 'bar-chart'],
        menu_icon="brain", default_index=0
    )

if selected == "Upload Documents":
    st.title("📚 Upload des supports de cours")
    st.write("Glissez-déposez vos PDF, slides, notes...")
    # uploaded_files = st.file_uploader(...)

elif selected == "Configurer Examen":
    st.title("⚙️ Configuration de l'examen")
    # Sliders Bloom, durée, etc.

else:
    st.title(f"🚧 {selected} – En cours de développement")
    st.write("Cette page sera codée par ton groupe !")


    // Importer la fonction retrieve pour tester le RAG
    # Dans app.py, ajoute temporairement pour tester
if st.button("Test RAG : trouve-moi du contenu sur CNN"):
    context = "\n\n".join(retrieve("réseaux de neurones convolutifs", k=5))
    st.write(context)