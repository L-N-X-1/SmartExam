# agents/coordinator_agent.py
"""
Coordonne tous les agents Bloom selon la distribution demandée
"""

from agents.remember_agent import RememberAgent
from agents.understand_agent import UnderstandAgent
from agents.apply_agent import ApplyAgent

class CoordinatorAgent:
    """
    Orchestre la génération de questions sur les 6 niveaux Bloom
    """
    
    def __init__(self):
        self.agents = {
            1: RememberAgent(),
            2: UnderstandAgent(),
            3: ApplyAgent(),
            # TODO: Ajouter agents 4, 5, 6
        }
    
    def generate_exam_questions(self, course_name, distribution, topic=None):
        """
        Génère toutes les questions selon la distribution Bloom
        
        INPUT:
            - course_name (str) : nom du cours
            - distribution (dict) : {1: 10, 2: 15, 3: 20, ...} nombre de questions par niveau
            - topic (str) : sujet optionnel
        
        OUTPUT:
            - all_questions (list) : toutes les questions générées
        """
        print("\n" + "="*70)
        print("🤖 COORDINATOR : Génération des questions")
        print("="*70)
        
        all_questions = []
        
        for level, num_questions in distribution.items():
            if num_questions == 0:
                continue
            
            if level not in self.agents:
                print(f"   ⚠️  Agent niveau {level} pas encore implémenté")
                continue
            
            print(f"\n📊 Niveau {level} : {num_questions} questions demandées")
            
            agent = self.agents[level]
            questions = agent.generate_questions(course_name, num_questions, topic)
            
            all_questions.extend(questions)
        
        print("\n" + "="*70)
        print(f"✅ TOTAL : {len(all_questions)} questions générées")
        print("="*70)
        
        return all_questions