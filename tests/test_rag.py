import os
from huggingface_hub import login

try:
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN non défini dans .env !")
    login(token=token)
    print("Login HF réussi ! Token valide.")
except Exception as e:
    print(f"Erreur login HF : {e}")
    print("Vérifie ton token et relance.")

# Test download modèle (ajoute ça pour vérifier)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))
print("Modèle embeddings chargé !")