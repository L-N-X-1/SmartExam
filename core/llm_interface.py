#Wrapper unique pour appeler les LLM (OpenAI, Anthropic, Grok, Llama). Fonctions : get_completion() et get_embedding(). Centralise toutes les appels API.
# core/llm_interface.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# Use the correct import for modern LangChain
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# =============================================================================
# WRAPPER UNIQUE : GENERATION (GROQ/LLAMA 3.1) + EMBEDDING (HUGGING FACE)
# =============================================================================

def get_completion(prompt: str, system_role: str = "You are a helpful assistant.", model: str = "llama-3.1-8b-instant", temperature: float = 0.5) -> str:
    """
    Generates text using Groq (hosting Llama 3.1).
    Updated to use the active model 'llama-3.1-8b-instant'.
    """
    try:
        # Initialize Groq Chat Model
        chat = ChatGroq(
            temperature=temperature,
            model_name=model,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Create message structure
        messages = [
            SystemMessage(content=system_role),
            HumanMessage(content=prompt)
        ]
        
        # Get response
        response = chat.invoke(messages)
        return response.content

    except Exception as e:
        print(f"❌ Error in get_completion: {e}")
        return "Error calling LLM."

def get_embedding(text: str, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> list[float]:
    """
    Generates embeddings using Hugging Face (MiniLM).
    """
    try:
        # Initialize Embedding Model
        embeddings_model = HuggingFaceEmbeddings(model_name=model)
        
        # Generate Vector
        vector = embeddings_model.embed_query(text)
        return vector

    except Exception as e:
        print(f"❌ Error in get_embedding: {e}")
        return []

# =============================================================================
# TEST AREA
# =============================================================================
if __name__ == "__main__":
    print("--- Testing Hybrid Interface (Groq + HF) ---")
    
    print("1. Testing Generation (Groq Llama 3.1)...")
    response = get_completion("What is the capital of France?")
    print(f"🤖 AI Answer: {response}")
    
    print("\n2. Testing Embedding (Hugging Face)...")
    vector = get_embedding("Smart Exam Test")
    if vector:
        print(f"🔢 Vector generated! Length: {len(vector)} dimensions.")
    else:
        print("❌ Embedding failed.")