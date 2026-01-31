# =============================================================================
# SMART EXAM – Coordinator Agent with LangGraph
# =============================================================================

from typing import List, Dict
from langgraph.graph import StateGraph, END
from typing import TypedDict
from .remember_agent import RememberAgent
from .understand_agent import UnderstandAgent
from .apply_agent import ApplyAgent
from .analyze_agent import AnalyzeAgent
from .evaluate_agent import EvaluateAgent
from .create_agent import CreateAgent


class CoordinatorState(TypedDict):
    """State for exam generation workflow"""
    context: str
    total_questions: int
    distribution: Dict[str, int]
    topic: str
    exam_results: Dict[str, List[Dict]]
    error: str


class CoordinatorAgent:
    """Coordinator that orchestrates all Bloom level agents using LangGraph"""
    
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.model = model
        self.agents = {
            "Remember": RememberAgent(model),
            "Understand": UnderstandAgent(model),
            "Apply": ApplyAgent(model),
            "Analyze": AnalyzeAgent(model),
            "Evaluate": EvaluateAgent(model),
            "Create": CreateAgent(model)
        }
        self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph workflow for exam generation"""
        workflow = StateGraph(CoordinatorState)
        
        # Add nodes for each level
        workflow.add_node("remember", self._generate_remember)
        workflow.add_node("understand", self._generate_understand)
        workflow.add_node("apply", self._generate_apply)
        workflow.add_node("analyze", self._generate_analyze)
        workflow.add_node("evaluate", self._generate_evaluate)
        workflow.add_node("create", self._generate_create)
        workflow.add_node("finalize", self._finalize)
        
        # Set entry and edges
        workflow.set_entry_point("remember")
        workflow.add_edge("remember", "understand")
        workflow.add_edge("understand", "apply")
        workflow.add_edge("apply", "analyze")
        workflow.add_edge("analyze", "evaluate")
        workflow.add_edge("evaluate", "create")
        workflow.add_edge("create", "finalize")
        workflow.add_edge("finalize", END)
        
        self.graph = workflow.compile()
    
    def _generate_for_level(self, state: CoordinatorState, level: str) -> CoordinatorState:
        """Generate questions for a level"""
        try:
            num_q = state['distribution'].get(level, 0)
            if num_q > 0:
                agent = self.agents[level]
                questions = agent.generate_questions(state['context'], num_q, state['topic'])
                state['exam_results'][level] = questions
                print(f"✓ Generated {len(questions)} {level} questions")
        except Exception as e:
            state['error'] = f"Error for {level}: {str(e)}"
        return state
    
    def _generate_remember(self, state: CoordinatorState) -> CoordinatorState:
        return self._generate_for_level(state, "Remember")
    
    def _generate_understand(self, state: CoordinatorState) -> CoordinatorState:
        return self._generate_for_level(state, "Understand")
    
    def _generate_apply(self, state: CoordinatorState) -> CoordinatorState:
        return self._generate_for_level(state, "Apply")
    
    def _generate_analyze(self, state: CoordinatorState) -> CoordinatorState:
        return self._generate_for_level(state, "Analyze")
    
    def _generate_evaluate(self, state: CoordinatorState) -> CoordinatorState:
        return self._generate_for_level(state, "Evaluate")
    
    def _generate_create(self, state: CoordinatorState) -> CoordinatorState:
        return self._generate_for_level(state, "Create")
    
    def _finalize(self, state: CoordinatorState) -> CoordinatorState:
        """Finalize exam generation"""
        total = sum(len(qs) for qs in state['exam_results'].values())
        print(f"\n✅ Exam complete: {total} questions")
        return state
    
    def generate_exam(
        self,
        context: str,
        num_questions: int = 30,
        distribution: Dict[str, int] = None,
        difficulty: str = "Medium",
        language: str = "English",
        topic: str = None
    ) -> Dict:
        """Generate complete exam"""
        if distribution is None:
            q_per_level = num_questions // 6
            distribution = {level: q_per_level for level in self.agents.keys()}
        
        initial_state = CoordinatorState(
            context=context,
            total_questions=num_questions,
            distribution=distribution,
            topic=topic or "General",
            exam_results={},
            error=""
        )
        
        result = self.graph.invoke(initial_state)
        exam_results = result.get('exam_results', {})
        
        # Flatten results into single questions array with bloom level metadata
        all_questions = []
        for level, questions in exam_results.items():
            if isinstance(questions, list):
                for q in questions:
                    if isinstance(q, dict):
                        q['bloom_level'] = level
                        q['difficulty'] = difficulty
                        q['language'] = language
                    all_questions.append(q)
        
        return {
            "questions": all_questions,
            "metadata": {
                "total": len(all_questions),
                "difficulty": difficulty,
                "language": language,
                "bloom_distribution": distribution
            }
        }
    
    def get_agent(self, bloom_level: str):
        """Get a specific agent"""
        return self.agents.get(bloom_level)
