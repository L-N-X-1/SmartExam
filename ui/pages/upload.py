# ui/pages/upload.py
"""
Page Streamlit pour uploader et traiter les documents + Chat RAG avec métadonnées
"""

import streamlit as st
import os
from core.rag_engine import process_course_document, get_relevant_context, get_relevant_context_with_metadata
from config.settings import UPLOAD_FOLDER

def show_upload_page():
    st.title("📚 Upload des Documents de Cours")
    st.markdown("---")
    
    st.write("""
    Uploadez vos documents de cours (PDF, DOCX, TXT) pour créer la base de connaissances.
    Le système RAG va extraire, indexer et vous permettre de dialoguer avec votre document.
    """)
    
    # Upload de fichier
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choisissez un fichier de cours",
            type=['pdf', 'docx', 'txt'],
            help="Formats supportés : PDF, DOCX, TXT (max 10MB)"
        )
    
    with col2:
        if uploaded_file:
            st.success("✅ Fichier sélectionné")
            st.metric("Nom", uploaded_file.name)
            st.metric("Taille", f"{uploaded_file.size / 1024:.1f} KB")
    
    if uploaded_file:
        st.divider()
        
        # Nom du cours
        course_name = st.text_input(
            "📝 Identifiant du cours",
            value="cours_2025",
            help="Nom unique pour identifier ce cours",
            placeholder="Ex: maths_derivees, philo_ethique"
        )
        
        # Bouton de traitement
        if st.button("🚀 Traiter le Document avec RAG", type="primary", use_container_width=True):
            
            if not course_name.strip():
                st.error("❌ Veuillez donner un nom au cours")
            else:
                # Sauvegarde temporaire
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                st.info(f"📁 Fichier sauvegardé : {file_path}")
                
                # Pipeline RAG avec progression
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📖 Extraction du texte...")
                    progress_bar.progress(20)
                    
                    status_text.text("✂️ Découpage en chunks...")
                    progress_bar.progress(40)
                    
                    status_text.text("🧮 Création des embeddings...")
                    progress_bar.progress(60)
                    
                    status_text.text("💾 Indexation vectorielle...")
                    progress_bar.progress(80)
                    
                    # Traitement complet
                    result = process_course_document(file_path, course_name)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Traitement terminé !")
                    
                    # Mise à jour session_state
                    st.session_state['course_name'] = course_name
                    st.session_state['course_ready'] = True
                    st.session_state['num_chunks'] = result['num_chunks']
                    
                    # Résultats
                    st.success("🎉 Document traité avec succès !")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📦 Chunks", result['num_chunks'])
                    with col2:
                        st.metric("✅ Status", "Prêt")
                    with col3:
                        st.metric("📚 Cours", course_name)
                    
                    st.balloons()
                    
                    st.info("👉 **Vous pouvez maintenant dialoguer avec le document ci-dessous**")
                    
                except Exception as e:
                    progress_bar.progress(0)
                    status_text.text("")
                    st.error(f"❌ Erreur : {e}")
                    
                    with st.expander("🔍 Détails de l'erreur"):
                        import traceback
                        st.code(traceback.format_exc())
    
    # ========================================================================
    # SECTION CHAT RAG
    # ========================================================================
    
    if st.session_state.get('course_ready', False):
        st.divider()
        st.subheader("💬 Dialoguez avec votre Document (RAG)")
        
        # Info cours
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Cours :** {st.session_state['course_name']}")
        with col2:
            st.info(f"**Chunks :** {st.session_state['num_chunks']}")
        with col3:
            st.info("**État :** ✅ Indexé")
        
        st.write("---")
        
        # Init historique
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        # Instructions
        st.markdown("### 🤖 Assistant RAG avec Groq")
        st.write("Posez vos questions sur le document. L'IA répondra en utilisant **uniquement** le contenu du document.")
        
        # Exemples
        with st.expander("💡 Exemples de questions"):
            st.markdown("""
            - "Résume les points principaux du document"
            - "Qu'est-ce qu'une dérivée ?"
            - "Explique-moi les règles de dérivation"
            - "Donne-moi des exemples"
            """)
        
        # Affichage historique
        for i, chat in enumerate(st.session_state['chat_history']):
            with st.chat_message("user", avatar="👤"):
                st.write(chat['question'])
            
            with st.chat_message("assistant", avatar="🤖"):
                st.write(chat['answer'])
                
                # MÉTADONNÉES - Code bien placé ici
                if 'metadata_results' in chat:
                    with st.expander("🔍 Voir le contexte et métadonnées"):
                        for j, result in enumerate(chat['metadata_results'], 1):
                            st.markdown(f"**Chunk {j}** - Score: {result.get('similarity_score', 0):.3f}")
                            
                            # Métadonnées
                            meta = result.get('metadata', {})
                            cols = st.columns(4)
                            with cols[0]:
                                st.caption(f"📄 Section: {meta.get('section_title', 'N/A')}")
                            with cols[1]:
                                st.caption(f"🌍 Langue: {meta.get('lang', 'N/A')}")
                            with cols[2]:
                                st.caption(f"📊 Index: {meta.get('chunk_index', '?')}/{meta.get('total_chunks', '?')}")
                            with cols[3]:
                                st.caption(f"📝 Mots: {meta.get('word_count', 'N/A')}")
                            
                            # Texte
                            st.text_area(
                                f"Texte du chunk {j}", 
                                result['text'], 
                                height=120, 
                                key=f"meta_{i}_{j}",
                                disabled=True
                            )
                            
                            if j < len(chat['metadata_results']):
                                st.divider()
        
        # Input question
        user_question = st.chat_input("💬 Tapez votre question ici...")
        
        if user_question:
            # Affiche la question
            with st.chat_message("user", avatar="👤"):
                st.write(user_question)
            
            # Génère la réponse
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔍 Recherche dans le document..."):
                    try:
                        from core.groq_interface import rag_query
                        
                        course_name = st.session_state['course_name']
                        
                        # Récupère contexte simple pour la réponse
                        context = get_relevant_context(user_question, course_name, k=5)
                        
                        # Récupère avec métadonnées pour l'affichage
                        try:
                            metadata_results = get_relevant_context_with_metadata(user_question, course_name, k=3)
                        except:
                            metadata_results = None
                        
                        # Génère la réponse
                        answer = rag_query(user_question, context, language="français")
                        
                        # Affiche
                        st.write(answer)
                        
                        # Sauvegarde dans l'historique
                        chat_entry = {
                            'question': user_question,
                            'answer': answer,
                            'context': context
                        }
                        
                        if metadata_results:
                            chat_entry['metadata_results'] = metadata_results
                        
                        st.session_state['chat_history'].append(chat_entry)
                        
                        # Rerun pour afficher
                        st.rerun()
                        
                    except ValueError as e:
                        st.error("❌ Configuration Groq manquante")
                        st.warning(str(e))
                        
                        st.info("### 📝 Configuration requise :")
                        st.markdown("""
                        1. Allez sur **https://console.groq.com**
                        2. Créez une API Key
                        3. Ajoutez dans `.env` :
                        """)
                        st.code("GROQ_API_KEY=gsk_votre_cle", language="bash")
                        
                    except Exception as e:
                        st.error(f"❌ Erreur : {e}")
                        with st.expander("🔍 Détails"):
                            import traceback
                            st.code(traceback.format_exc())
        
        # Boutons actions
        st.write("")
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col2:
            if st.button("🗑️ Effacer", use_container_width=True):
                st.session_state['chat_history'] = []
                st.rerun()
        
        with col3:
            if st.button("📥 Export", use_container_width=True):
                chat_text = ""
                for chat in st.session_state['chat_history']:
                    chat_text += f"Q: {chat['question']}\n\n"
                    chat_text += f"R: {chat['answer']}\n\n"
                    chat_text += "-" * 50 + "\n\n"
                
                st.download_button(
                    label="💾 Télécharger",
                    data=chat_text,
                    file_name=f"chat_{st.session_state['course_name']}.txt",
                    mime="text/plain"
                )

# Pour l'intégration
if __name__ == "__main__":
    show_upload_page()