#Extraction texte depuis PDF (PyPDF2) et DOCX (python-docx). Fonctions : extract_text_from_pdf(), extract_text_from_docx().
# services/document_processor.py
from pypdf import PdfReader
from docx import Document
import os

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith((".docx", ".doc")):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Format non supporté")