# tests/test_rag.py
import sys
import os

# Ajoute le chemin du projet
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=" * 70)
print("🧪 TEST RAG - SMART EXAM")
print("=" * 70)

# Test imports
print("\n📦 Vérification des imports...")
try:
    from services.document_processor import process_document
    from utils.text_processing import clean_text, chunk_text
    from core.embeddings import create_index, retrieve
    from core.rag_engine import process_course_document, get_relevant_context
    print("✅ Tous les imports OK")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    sys.exit(1)

# Crée les dossiers
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/vector_store", exist_ok=True)

# Crée un document de test
print("\n📝 Création d'un document de test...")

test_content = """
CHAPITRE 1 : LES DÉRIVÉES EN MATHÉMATIQUES

1. Introduction
Une dérivée mesure le taux de variation d'une fonction. C'est un concept fondamental 
du calcul différentiel introduit par Isaac Newton et Gottfried Leibniz au 17ème siècle.

2. Définition Formelle
La dérivée d'une fonction f au point x est définie par la limite suivante :
f'(x) = lim(h→0) [f(x+h) - f(x)] / h

Cette limite représente la pente de la tangente à la courbe au point x.

3. Règles de Dérivation Essentielles

3.1 Dérivée d'une constante
Si f(x) = c où c est une constante, alors f'(x) = 0

3.2 Dérivée de la puissance
Si f(x) = x^n, alors f'(x) = n * x^(n-1)

3.3 Règle de la somme
Si f(x) = g(x) + h(x), alors f'(x) = g'(x) + h'(x)

3.4 Règle du produit
Si f(x) = g(x) * h(x), alors f'(x) = g'(x)*h(x) + g(x)*h'(x)

3.5 Règle de la chaîne
Si f(x) = g(h(x)), alors f'(x) = g'(h(x)) * h'(x)

4. Applications Pratiques

4.1 En Physique
- Vitesse : dérivée de la position par rapport au temps
- Accélération : dérivée de la vitesse par rapport au temps

4.2 En Économie
- Coût marginal : dérivée de la fonction de coût
- Revenu marginal : dérivée de la fonction de revenu

5. Exemples Concrets

Exemple 1 : Calculer la dérivée de f(x) = 3x^2 + 2x - 5
Solution : f'(x) = 6x + 2

Exemple 2 : Calculer la dérivée de f(x) = sin(x^2)
Solution : En utilisant la règle de la chaîne, f'(x) = cos(x^2) * 2x

6. Exercices
1. Calculez la dérivée de f(x) = 4x^3 - 2x + 7
2. Trouvez la pente de la tangente à f(x) = x^2 au point x = 3
3. Si la position d'un objet est s(t) = 5t^2 + 2t, quelle est sa vitesse à t = 2?
"""

test_file = os.path.join(project_root, "data/uploads/test_cours_maths.txt")
with open(test_file, 'w', encoding='utf-8') as f:
    f.write(test_content)

print(f"✅ Fichier créé : {os.path.basename(test_file)}")
print(f"   📏 Taille : {len(test_content)} caractères")

# Test complet du RAG
print("\n" + "=" * 70)
print("🚀 TEST DU PIPELINE RAG COMPLET")
print("=" * 70)

try:
    # Test 1 : Extraction
    print("\n1️⃣ Extraction du texte...")
    raw_text = process_document(test_file)
    print(f"   ✅ {len(raw_text)} caractères extraits")
    
    # Test 2 : Nettoyage
    print("\n2️⃣ Nettoyage du texte...")
    clean = clean_text(raw_text)
    print(f"   ✅ Texte nettoyé : {len(clean)} caractères")
    
    # Test 3 : Chunking
    print("\n3️⃣ Découpage en chunks...")
    chunks = chunk_text(clean, chunk_size=300, overlap=50)
    print(f"   ✅ {len(chunks)} chunks créés")
    print(f"\n   📄 Aperçu du chunk 1 :")
    print(f"   {chunks[0][:150]}...\n")
    
    # Test 4 : Création d'index
    print("4️⃣ Création de l'index FAISS...")
    index, saved_chunks = create_index(chunks, course_name="test_maths")
    print(f"   ✅ Index créé avec succès")
    
    # Test 5 : Recherche
    print("\n5️⃣ Test de recherche...")
    
    queries = [
        "Qu'est-ce qu'une dérivée?",
        "Règles de dérivation",
        "Applications en physique"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n   🔍 Requête {i} : {query}")
        results = retrieve(query, "test_maths", k=2)
        print(f"   📖 Résultat 1 :")
        print(f"   {results[0][:200]}...")
    
    # Test 6 : Fonction complète
    print("\n" + "=" * 70)
    print("6️⃣ Test de la fonction complète get_relevant_context...")
    context = get_relevant_context("Donne-moi des exemples de dérivées", "test_maths", k=3)
    print(f"   ✅ Contexte récupéré ({len(context)} caractères)")
    print(f"\n   📖 Contexte :")
    print(f"   {context[:300]}...\n")
    
    print("=" * 70)
    print("✅ TOUS LES TESTS RÉUSSIS !")
    print("=" * 70)
    print("\n🎉 Le système RAG fonctionne parfaitement !")
    print("\n💡 Prochaines étapes :")
    print("   1. Teste avec un vrai PDF de cours")
    print("   2. Intègre avec les agents Bloom")
    print("   3. Génère des questions basées sur le contexte RAG")
    
except Exception as e:
    print(f"\n❌ ERREUR : {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)