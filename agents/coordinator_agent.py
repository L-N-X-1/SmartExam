from agents.evaluate_agent import EvaluateAgent
from agents.remember_agent import RememberAgent
from agents.understand_agent import UnderstandAgent
from agents.apply_agent import ApplyAgent
from agents.analyze_agent import AnalyzeAgent
from agents.create_agent import CreateAgent

class CoordinatorAgent:
    """
    Orchestre la génération de questions sur les 6 niveaux Bloom
    """
    
    def __init__(self):
        self.agents = {
            1: RememberAgent(),
            2: UnderstandAgent(),
            3: ApplyAgent(),
            4: AnalyzeAgent(),     # niveau 4
            5: EvaluateAgent(),    # niveau 5
            6: CreateAgent()       # niveau 6
        }
    
    def generate_exam_questions(self, course_name, distribution, topic=None):
        """
        Génère toutes les questions selon la distribution Bloom
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
