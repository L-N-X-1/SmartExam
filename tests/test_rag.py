from dotenv import load_dotenv
load_dotenv()

import os
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# -------------------------
# HF LOGIN
# -------------------------

token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    raise ValueError("HUGGINGFACE_TOKEN missing in .env")

login(token=token)
print("HF login OK")

model_name = os.getenv("EMBEDDING_MODEL")
if not model_name:
    raise ValueError("EMBEDDING_MODEL missing in .env")

SentenceTransformer(model_name)
print("Embedding model OK:", model_name)

# -------------------------
# RAG ENGINE
# -------------------------

from core.rag_engine import RAGEngine

VECTOR_STORE = "data/vector_store/test_rag"
UPLOAD_DIR = "uploads"   # put your PDFs here

engine = RAGEngine(
    vector_store_path=VECTOR_STORE,
    model_name=model_name
)

# -------------------------
# PDF INGESTION
# -------------------------

def extract_pdf_chunks(pdf_path, engine):
    reader = PdfReader(pdf_path)
    all_chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        chunks = engine.process_pdf_to_chunks(
            text,
            chunk_size=180,
            chunk_overlap=40,
            source=os.path.basename(pdf_path),
            page=page_num,
            title=os.path.basename(pdf_path)
        )
        all_chunks.extend(chunks)

    return all_chunks


print("\nScanning PDFs in:", UPLOAD_DIR)

all_chunks = []

for fname in os.listdir(UPLOAD_DIR):
    if fname.lower().endswith(".pdf"):
        path = os.path.join(UPLOAD_DIR, fname)
        print("Reading:", fname)
        pdf_chunks = extract_pdf_chunks(path, engine)
        print("  chunks:", len(pdf_chunks))
        all_chunks.extend(pdf_chunks)

if not all_chunks:
    raise RuntimeError("No PDF chunks extracted — check uploads folder")

print("\nTotal chunks:", len(all_chunks))

# -------------------------
# INDEX BUILD
# -------------------------

engine.create_index(all_chunks, append=False)
engine.save_index()
print("Index built + saved")

# -------------------------
# RELOAD TEST
# -------------------------

engine2 = RAGEngine(
    vector_store_path=VECTOR_STORE,
    model_name=model_name
)

assert engine2.load_index()
print("Index reload OK")

# -------------------------
# TEST QUERIES
# -------------------------

TEST_QUERIES = [
    "define the main concept",
    "key formulas",
    "important definitions",
    "examples of applications",
    "summary of the topic"
]

for q in TEST_QUERIES:
    print("\n" + "="*60)
    print("QUERY:", q)

    results = engine2.retrieve(
        q,
        k=5,
        similarity_threshold=0.30
    )

    if not results:
        print("No results above threshold")
        continue

    for i, r in enumerate(results, start=1):
        print(f"\nResult #{i}")
        print("Similarity:", round(r["similarity"], 3))
        print("Source:", r["metadata"].get("source"))
        print("Page:", r["metadata"].get("page"))
        print("Text preview:")
        print(r["text"][:400], "...")
