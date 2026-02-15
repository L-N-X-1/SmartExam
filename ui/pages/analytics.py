import streamlit as st
import pandas as pd
import altair as alt

# Mapping Bloom levels (numeric -> name)
BLOOM_LEVEL_NAMES = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create"
}

BLOOM_ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
DIFFICULTY_ORDER = ["Easy", "Medium", "Hard"]

def show_analytics_page():
    st.title("📊 Analytics des Questions")

    questions = st.session_state.get('generated_questions', [])
    if not questions:
        st.warning("⚠️ Aucune question générée. Passez d'abord par la page 'Générer Questions'.")
        return

    total_questions = len(questions)

    # -------------------------
    # Prepare cleaned data
    # -------------------------
    cleaned_data = []
    for q in questions:
        level = q.get("level_name") or BLOOM_LEVEL_NAMES.get(q.get("bloom_level"))
        level = level if level in BLOOM_ORDER else "Remember"  # default to easiest if missing
        difficulty = str(q.get("difficulty") or "Easy").capitalize()
        difficulty = difficulty if difficulty in DIFFICULTY_ORDER else "Easy"
        cleaned_data.append({
            "level": level,
            "difficulty": difficulty
        })

    df = pd.DataFrame(cleaned_data)

    # -------------------------
    # Répartition par niveau Bloom
    # -------------------------
    st.subheader("🎯 Répartition par niveau Bloom")
    df_level = df.groupby("level").size().reindex(BLOOM_ORDER, fill_value=0).reset_index(name="Nombre de questions")
    df_level["Pourcentage"] = (df_level["Nombre de questions"] / total_questions * 100).round(1)

    chart_level = alt.Chart(df_level).mark_bar(color="#4CAF50").encode(
        x=alt.X('level', sort=BLOOM_ORDER, title="Niveau Bloom"),
        y=alt.Y('Nombre de questions', title="Nombre de questions"),
        tooltip=['level', 'Nombre de questions', 'Pourcentage']
    ).properties(width=600, height=400)

    st.altair_chart(chart_level, use_container_width=True)
    st.table(df_level.rename(columns={"level": "Niveau"}))

    # -------------------------
    # Répartition par difficulté
    # -------------------------
    st.subheader("📊 Répartition par difficulté")
    df_diff = df.groupby("difficulty").size().reindex(DIFFICULTY_ORDER, fill_value=0).reset_index(name="Nombre de questions")
    df_diff["Pourcentage"] = (df_diff["Nombre de questions"] / total_questions * 100).round(1)

    chart_diff = alt.Chart(df_diff).mark_bar(color="#2196F3").encode(
        x=alt.X('difficulty', sort=DIFFICULTY_ORDER, title="Difficulté"),
        y=alt.Y('Nombre de questions', title="Nombre de questions"),
        tooltip=['difficulty', 'Nombre de questions', 'Pourcentage']
    ).properties(width=600, height=300)

    st.altair_chart(chart_diff, use_container_width=True)
    st.table(df_diff.rename(columns={"difficulty": "Difficulté"}))

    st.info(f"📌 Total de questions analysées: {total_questions}")
