from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from agents.evaluate_agent import EvaluateAgent

# État du graphe
class EvaluateState(TypedDict):
    context: str
    topic: Optional[str]
    num_questions: int
    questions: List[dict]

# Nœud LangGraph
def evaluate_node(state: EvaluateState) -> EvaluateState:
    agent = EvaluateAgent()
    questions = agent.generate_questions(
        context=state["context"],
        num_questions=state["num_questions"],
        topic=state.get("topic")
    )

    return {**state, "questions": questions}

# Construction du graphe
def generate_evaluate_questions(context, num_questions=5, topic=None):
    graph = StateGraph(EvaluateState)
    graph.add_node("evaluate", evaluate_node)
    graph.set_entry_point("evaluate")
    graph.add_edge("evaluate", END)

    app = graph.compile()

    result = app.invoke({
        "context": context,
        "topic": topic,
        "num_questions": num_questions,
        "questions": []
    })

    return result["questions"]
