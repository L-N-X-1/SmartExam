# smart_exam 
Projet de generation intelligente d'examens basse sur la taxonomie de Bloom et RAG 
 
# 🎓 SMART EXAM – Répartition des Tâches pour les Étudiants

Projet de fin d'année : **Générateur intelligent d'examens basé sur la Taxonomie de Bloom + Multi-Agent + RAG**

> Objectif : Créer une application Streamlit complète qui permet à un enseignant d'uploader ses cours (PDF, etc.) → génère automatiquement un examen équilibré sur les 6 niveaux de Bloom → export PDF + analytics.

### Répartition conseillée (6-8 étudiants) :
- Groupe 1 → Upload + RAG (2 personnes)  
- Groupe 2 → Agents Bloom (3 personnes)  
- Groupe 3 → Validation + Exam Assembler (1-2 personnes)  
- Groupe 4 → Interface Streamlit + UI Components (2 personnes)

---

### Structure du projet & Tâches par fichier

| Dossier / Fichier                        | Responsable(s)          | Tâche précise à réaliser (À FAIRE OBLIGATOIREMENT)                                                                                                       |
|------------------------------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **app.py**                               | Toute l'équipe          | Point d’entrée Streamlit. Créer un `st.set_page_config()` + `st.sidebar` + menu multi-page (Upload → Configurer → Générer → Voir l’examen). Utiliser `streamlit_option_menu` ou `st.page_link`. |
| **config/settings.py**                   | Groupe 1                 | Charger les variables d’environnement (.env) avec `python-dotenv`. Exposer : `OPENAI_API_KEY`, `LLM_MODEL = "gpt-4o"`, `EMBEDDING_MODEL`, etc.            |
| **config/bloom_taxonomy.py**             | Groupe 2                 | Définir un dictionnaire ou une classe avec les 6 niveaux + liste complète de verbes d’action par niveau (au moins 10 verbes par niveau).               |
| **core/llm_interface.py**                | Groupe 1                 | Wrapper unique pour appeler OpenAI / Anthropic / Grok. Fonctions : `get_completion(prompt, temperature=0.7)` et `get_embedding(text)`.               |
| **core/embeddings.py**                   | Groupe 1                 | Gestion du vector store (FAISS ou Chroma). Fonctions : `create_index(documents)`, `retrieve(query, k=5)`.                                              |
| **core/rag_engine.py**                   | Groupe 1                 | Pipeline complet : charger PDF/DOCX → chunking (500-1000 tokens) → embed → sauvegarde dans `data/vector_store/`.                                      |
| **agents/base_agent.py**                 | Groupe 2                 | Classe abstraite avec méthode `generate_questions(context, num=5)` que tous les agents héritent.                                                        |
| **agents/remember_agent.py**             | Étudiant 1 (Groupe 2)   | Spécialisé niveau 1. Prompt qui force des verbes comme List, Define, Recall, Name... Générer 10-15 questions par appel.                               |
| **agents/understand_agent.py**           | Étudiant 1 (Groupe 2)   | Niveau 2 : Explain, Summarize, Describe, Interpret...                                                                                                   |
| **agents/apply_agent.py**                | Étudiant 2 (Groupe 2)   | Niveau 3 : Solve, Use, Implement, Demonstrate... Inclure des petits exercices pratiques ou calculs.                                                  |
| **agents/analyze_agent.py**              | Étudiant 2 (Groupe 2)   | Niveau 4 : Compare, Contrast, Differentiate, Classify...                                                                                               |
| **agents/evaluate_agent.py**             | Étudiant 3 (Groupe 2)   | Niveau 5 : Justify, Critique, Assess, Defend...                                                                                                         |
| **agents/create_agent.py**               | Étudiant 3 (Groupe 2)   | Niveau 6 : Design, Invent, Propose, Create a new... Les questions les plus ouvertes et créatives.                                                      |
| **agents/coordinator_agent.py**          | Groupe 2 (leader)       | Orchestre les 6 agents. Doit respecter la distribution Bloom demandée par l’enseignant (ex: 10% Remember, 20% Apply...).                              |
| **agents/validator_agent.py**            | Groupe 3                 | **Cœur du projet** : Implémenter les heuristiques + appel LLM zero-shot pour scorer chaque question (clarté, bloom accuracy, etc.). Seuil ≥ 85%.      |
| **models/question.py**                   | Groupe 3                 | Pydantic model `Question` avec champs : text, bloom_level (1-6), difficulty, type ("mcq"|"short"|"open"), marks, topic, validation_score, etc.        |
| **models/exam.py**                       | Groupe 3                 | Pydantic model `Exam` contenant titre, durée, liste de questions, bloom_distribution réelle, etc. + méthode `export_to_pdf()` (avec WeasyPrint ou ReportLab). |
| **services/document_processor.py**       | Groupe 1                 | Fonctions : `extract_text_from_pdf(path)`, `extract_text_from_docx(path)`. Utiliser PyPDF2 + python-docx.                                              |
| **services/question_generator.py**       | Groupe 2                 | Facade qui appelle le Coordinator → récupère toutes les questions brutes → passe au Validator → retourne seulement les validées.                     |
| **services/exam_assembler.py**           | Groupe 3                 | Prend les questions validées + config prof → sélectionne les meilleures pour respecter les % Bloom, difficulté, types → retourne un objet `Exam`.     |
| **services/validator_service.py**        | Groupe 3                 | Contient la logique de scoring (heuristique verbes + prompt LLM) utilisée par `validator_agent.py`.                                                    |
| **utils/bloom_heuristics.py**            | Groupe 3                 | Fonctions utilitaires : `detect_bloom_level_from_verb(verb)` → retourne 1-6, `check_consistency(question_text, claimed_level)`.                       |
| **ui/pages/upload.py**                   | Groupe 4                 | Page Streamlit : drag & drop fichiers + bouton "Traiter les documents" → lance RAG → st.success + aperçu des topics extraits.                         |
| **ui/pages/configure.py**                | Groupe 4                 | Formulaire : titre examen, durée, % par niveau Bloom (sliders), types de questions, difficulté → sauvegarde dans `st.session_state.exam_config`.       |
| **ui/pages/review.py**                   | Groupe 4                 | Affiche toutes les questions générées avec carte (component question_card.py), possibilité de supprimer/modifier manuellement.                        |
| **ui/pages/analytics.py**                | Groupe 4                 | Graphique camembert Bloom (Plotly) + radar chart difficulté + stats par topic.                                                                           |
| **ui/components/question_card.py**       | Groupe 4                 | Component réutilisable : affiche une question avec son niveau Bloom (badge coloré), score validation, bouton supprimer.                                |
| **ui/components/bloom_chart.py**         | Groupe 4                 | Component Plotly qui prend un dict {1: 10, 2: 15...} → camembert ou barres horizontales stylé.                                                         |

### Bonus (points supplémentaires)
- Ajouter un bouton **"Regénérer les questions manquantes"** si l’assembler détecte un déficit sur un niveau.
- Export PDF beau avec logo université + numérotation + bareme (WeasyPrint recommandé).
- Sauvegarder les examens dans `exams/` avec date.

### Deadline & Livrables
1. Chaque groupe push son code avec commits clairs (`git commit -m "feat: remember agent terminé"`)
2. Le `main.py` doit tourner chez tout le monde → `streamlit run app.py`
3. Présentation finale : démo complète (upload → config → génération → PDF)

**Que le meilleur examen gagne !** 🚀
**Architecture choisie** : Streamlit full-Python  
**Raison** : Vitesse de développement ×10, équipe Python-only, deadline courte, maintenance simple.  
On privilégie l’intelligence de l’IA plutôt que l’architecture micro-services.

— Jouini Chahd, novembre 2025
