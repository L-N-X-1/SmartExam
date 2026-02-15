import streamlit as st
from fpdf import FPDF
import io
import json


# ----------------------------------
# Bloom Level Mapping
# ----------------------------------
BLOOM_LEVEL_NAMES = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create"
}


# ----------------------------------
# Safe text cleaner for PDF
# ----------------------------------
def clean_text(text):
    """
    Ensures text is safe for classic FPDF (latin-1 compatible).
    Prevents encoding and width errors.
    """
    if not isinstance(text, str):
        text = str(text)

    # Replace problematic unicode characters
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    return text.encode("latin-1", "replace").decode("latin-1")


# ----------------------------------
# Main Page
# ----------------------------------
def show_review_page():
    st.title("📝 Review & Export")

    # -------------------------------
    # Get generated questions
    # -------------------------------
    questions = st.session_state.get("generated_questions", [])

    if not questions:
        st.warning("⚠️ Aucune question générée. Veuillez passer par la page 'Générer Questions'.")
        return

    # -------------------------------
    # Preview Section
    # -------------------------------
    st.subheader(f"✅ {len(questions)} questions générées (aperçu)")

    for i, q in enumerate(questions[:5], start=1):

        level = (
            q.get("level_name")
            or BLOOM_LEVEL_NAMES.get(q.get("bloom_level"), q.get("bloom_level", "N/A"))
        )

        question_text = q.get("question") or q.get("text") or "Question sans texte"
        q_type = q.get("type", "Question ouverte")
        marks = q.get("marks", 1)
        difficulty = q.get("difficulty", level)

        with st.expander(f"Question {i} - Niveau {level}"):
            st.write(f"**Texte :** {question_text}")
            st.write(f"Type: {q_type} | Points: {marks} | Difficulté: {difficulty}")

    if len(questions) > 5:
        st.info(f"... et {len(questions) - 5} autres questions")

    # -------------------------------
    # Export Section
    # -------------------------------
    st.subheader("📦 Export des questions")

    # -------- JSON Export --------
    try:
        json_data = json.dumps(questions, ensure_ascii=False, indent=2)

        st.download_button(
            "Télécharger JSON",
            data=json_data,
            file_name="questions.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"Erreur lors de la génération du JSON: {str(e)}")

    # -------- PDF Export --------
    if st.button("Exporter en PDF"):

        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Header
            pdf.set_font("Arial", "B", 16)
            exam_title = (
                st.session_state.get("exam_config", {}).get("title")
                or "Examen"
            ).strip()

            pdf.cell(0, 10, clean_text(exam_title), ln=True, align="C")

            pdf.set_font("Arial", "I", 12)
            pdf.cell(
                0,
                10,
                clean_text(f"Total: {len(questions)} questions"),
                ln=True,
                align="C"
            )

            pdf.ln(10)

            # Page width
            page_width = pdf.w - 2 * pdf.l_margin

            # Questions
            pdf.set_font("Arial", "", 12)

            for i, q in enumerate(questions, start=1):

                level = (
                    q.get("level_name")
                    or BLOOM_LEVEL_NAMES.get(q.get("bloom_level"), q.get("bloom_level", "N/A"))
                )

                question_text = q.get("question") or q.get("text") or ""
                q_type = q.get("type", "Question ouverte")
                marks = q.get("marks", 1)
                difficulty = q.get("difficulty", level)

                question_line = f"{i}. [{level}] {question_text}"
                meta_line = f"Type: {q_type} | Points: {marks} | Difficulté: {difficulty}"

                pdf.multi_cell(page_width, 8, clean_text(question_line))
                pdf.multi_cell(page_width, 8, clean_text(meta_line))
                pdf.ln(3)

            # ----------------------------------
            # Safe PDF Output (fpdf + fpdf2)
            # ----------------------------------
            pdf_output = pdf.output(dest="S")

            if isinstance(pdf_output, (bytes, bytearray)):
                pdf_buffer = io.BytesIO(pdf_output)
            else:
                pdf_buffer = io.BytesIO(
                    pdf_output.encode("latin-1", "replace")
                )

            st.download_button(
                "Télécharger PDF",
                data=pdf_buffer,
                file_name="questions.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Erreur lors de la génération du PDF: {str(e)}")
