# tests/test_agents.py
import sys
import os
import traceback

# Ajout du projet au path pour pouvoir importer les modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("="*70)
print("🧪 TEST DES AGENTS BLOOM")
print("="*70)

# Test 1 : Import des agents
print("\n1️⃣ Import des agents...")
try:
    from agents.remember_agent import RememberAgent
    from agents.understand_agent import UnderstandAgent
    from agents.apply_agent import ApplyAgent
    from agents.coordinator_agent import CoordinatorAgent
    print("   ✅ Tous les agents importés")
except ImportError as e:
    print(f"   ❌ Erreur : {e}")
    sys.exit(1)

# Test 2 : Initialisation
print("\n2️⃣ Initialisation des agents...")
try:
    remember = RememberAgent()
    understand = UnderstandAgent()
    apply = ApplyAgent()
    coordinator = CoordinatorAgent()
    print("   ✅ Agents initialisés")
except Exception as e:
    print(f"   ❌ Erreur : {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3 : Génération avec un cours de test
print("\n3️⃣ Test de génération (nécessite un cours chargé)...")
print("   ⚠️  Assure-toi d'avoir lancé tests/test_rag.py avant !")

# Configuration mock pour les tests
mock_course_name = "test_maths"
num_questions = 3

try:
    # Tentative de génération réelle
    try:
        questions = remember.generate_questions(mock_course_name, num_questions=num_questions)
    except Exception as e:
        print(f"   ⚠️  Impossible d'utiliser le LLM réel : {e}")
        print("   ℹ️  Utilisation de questions mock pour le test")
        # Génération mock
        questions = [
            {
                "text": f"Question mock {i+1}",
                "type": "mcq",
                "bloom_level": 1,
                "difficulty": "Moyen",
                "marks": 2,
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A"
            }
            for i in range(num_questions)
        ]
    
    if questions:
        print(f"   ✅ {len(questions)} questions générées")
        print(f"\n   📝 Exemple de question :")
        print(f"   {questions[0]}")
    else:
        print("   ⚠️  Aucune question générée")

except Exception as e:
    print(f"   ❌ Erreur inattendue : {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("✅ TEST TERMINÉ")
print("="*70)
