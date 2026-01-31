# =============================================================================
# SMART EXAM – Base Agent Class with LangGraph
# Fichier : agents/base_agent.py
# À QUOI SERT CE FICHIER :
# Base agent using LangGraph for question generation at each Bloom level
# =============================================================================

from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_engine import retrieve
from config.settings import GROQ_API_KEY

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State for question generation workflow"""
    context: str
    topic: str
    num_questions: int
    bloom_level: str
    retrieved_context: List[Dict]
    generated_questions: str
    parsed_questions: List[Dict]
    error: str


class BaseAgent:
    """
    Base agent using LangGraph for question generation.
    Implements workflow with retrieve → generate → parse nodes.
    """
    
    def __init__(self, bloom_level: str, model: str = "openai/gpt-oss-20b"):
        """
        Args:
            bloom_level (str): Bloom's Taxonomy level
            model (str): Groq model to use
        """
        self.bloom_level = bloom_level
        self.model = model
        self.client = Groq(api_key=GROQ_API_KEY)
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("parse", self._parse_node)
        workflow.add_node("error", self._error_node)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Add edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_conditional_edges(
            "generate",
            self._check_error,
            {
                "error": "error",
                "success": "parse"
            }
        )
        workflow.add_edge("parse", END)
        workflow.add_edge("error", END)
        
        self.graph = workflow.compile()
    
    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieve context using RAG"""
        try:
            query = f"Key information about {state['topic']}" if state['topic'] else state['context'][:200]
            retrieved = retrieve(query, k=5)
            state['retrieved_context'] = retrieved
        except Exception as e:
            state['error'] = f"Retrieval error: {str(e)}"
        return state
    
    def _generate_node(self, state: AgentState) -> AgentState:
        """Generate questions using LLM"""
        try:
            context_text = "\n\n".join([c.get('text', '') for c in state['retrieved_context']])
            prompt = self._create_prompt(context_text, state['num_questions'], state['topic'])
            
            # Use direct Groq API with streaming
            response_text = ""
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educational question generator specializing in Bloom's Taxonomy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_completion_tokens=8192,
                top_p=1,
                stream=True,
                stop=None
            )
            
            # Collect streamed chunks
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            state['generated_questions'] = response_text
        except Exception as e:
            state['error'] = f"Generation error: {str(e)}"
        return state
    
    def _parse_node(self, state: AgentState) -> AgentState:
        """Parse LLM response"""
        try:
            questions = self._parse_response(
                state['generated_questions'],
                state['num_questions']
            )
            state['parsed_questions'] = questions
        except Exception as e:
            state['error'] = f"Parsing error: {str(e)}"
        return state
    
    def _error_node(self, state: AgentState) -> AgentState:
        """Handle errors"""
        print(f"Error in {self.bloom_level} agent: {state.get('error')}")
        state['parsed_questions'] = []
        return state
    
    def _check_error(self, state: AgentState) -> str:
        """Check if error occurred"""
        return "error" if state.get('error') else "success"
    
    def _create_prompt(self, context: str, num_questions: int, topic: str = None) -> str:
        """Create prompt for generation"""
        instructions = self._get_bloom_instructions()
        
        return f"""You are an expert educational question generator specializing in Bloom's Taxonomy level: {self.bloom_level}.

{f'Topic: {topic}' if topic else ''}

Based on the following course material:
<context>
{context}
</context>

{instructions}

Generate exactly {num_questions} high-quality questions at the {self.bloom_level} level.

For each question, provide in this format:
Question: [text]
Answer: [text]
Explanation: [text]

---"""
    
    def _get_bloom_instructions(self) -> str:
        """Get Bloom level specific instructions - override in subclasses"""
        return ""
    
    def _parse_response(self, response: str, num_questions: int) -> List[Dict]:
        """Parse LLM response into questions"""
        questions = []
        lines = response.split('\n')
        current = {}
        
        for line in lines:
            if 'Question:' in line:
                if current and 'question' in current:
                    questions.append(current)
                current = {'question': line.split('Question:', 1)[-1].strip()}
            elif 'Answer:' in line:
                current['answer'] = line.split('Answer:', 1)[-1].strip()
            elif 'Explanation:' in line:
                current['explanation'] = line.split('Explanation:', 1)[-1].strip()
        
        if current and 'question' in current:
            questions.append(current)
        
        return questions[:num_questions]
    
    def invoke_workflow(
        self,
        context: str,
        num_questions: int = 5,
        topic: str = None
    ) -> List[Dict]:
        """Invoke the LangGraph workflow"""
        initial_state = AgentState(
            context=context,
            topic=topic or "General",
            num_questions=num_questions,
            bloom_level=self.bloom_level,
            retrieved_context=[],
            generated_questions="",
            parsed_questions=[],
            error=""
        )
        
        result = self.graph.invoke(initial_state)
        return result.get('parsed_questions', [])
    
    def generate_questions(
        self,
        context: str,
        num_questions: int = 5,
        topic: str = None
    ) -> List[Dict[str, str]]:
        """
        Generate questions for this Bloom level.
        
        Args:
            context (str): Course content
            num_questions (int): Number of questions to generate
            topic (str): Optional topic
            
        Returns:
            List[Dict]: List of questions with question, answer, explanation
        """
        return self.invoke_workflow(context, num_questions, topic)
