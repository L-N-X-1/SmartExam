# =============================================================================
# core/llm_interface.py
# =============================================================================
"""
Interface LLM unifiée pour SMART EXAM

Fournit une fonction unique `get_completion()` qui supporte:
- Objets avec méthode `generate(prompt, temperature=...)`
- LangChain/LangGraph runnables avec `invoke()`
- LangGraph graph id / client
- Callables simples
- Intégration Grok, Groq, et LLMs locaux

Centralise l'intégration LangGraph tout en maintenant la compatibilité arrière.
"""

import importlib
import json
import streamlit as st
from typing import Any, Optional


# =============================================================================
# Fonction de normalisation des réponses
# =============================================================================

def _normalize(res: Any) -> str:
    """
    Normalise différents formats de réponses LLM en string.
    
    Gère:
    - Strings simples
    - Dicts avec 'output', 'text', 'content', ou JSON complet
    - Objects avec attribut 'content'
    - Tout autre type converti en string
    
    Args:
        res: Réponse du LLM (format variable)
    
    Returns:
        String normalisée
    """
    # String directe
    if isinstance(res, str):
        return res
    
    # Object avec attribut content (réponse LangChain)
    if hasattr(res, 'content'):
        return str(res.content)
    
    # Dict avec clés communes
    if isinstance(res, dict):
        # Essayer les clés courantes
        for key in ('output', 'text', 'content', 'response', 'answer'):
            if key in res:
                value = res[key]
                return value if isinstance(value, str) else str(value)
        
        # Fallback: JSON dump
        return json.dumps(res, ensure_ascii=False)
    
    # Fallback: conversion en string
    return str(res)


# =============================================================================
# Fonction principale d'obtention de completion
# =============================================================================

def get_completion(
    llm: Any = None,
    prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 400,
    timeout: Optional[int] = None
) -> str:
    """
    Retourne une completion en utilisant diverses interfaces LLM.
    
    Cette fonction unifie l'accès à différents LLMs:
    1. Providers directs (Grok, Groq) via session_state
    2. Objects avec méthode generate()
    3. LangChain/LangGraph runnables avec invoke()
    4. LangGraph graph id / client
    5. Callables simples
    
    Args:
        llm: Object LLM, graph id, client, dict, ou callable (optionnel si provider en session_state)
        prompt: Le prompt texte à envoyer
        temperature: Température d'échantillonnage (0-1)
        max_tokens: Nombre maximum de tokens à générer
        timeout: Timeout optionnel en secondes
    
    Returns:
        String de completion
    
    Raises:
        RuntimeError: Si aucune interface supportée n'est trouvée
    
    Note:
        Le paramètre temperature peut être ignoré par certains graphs LangGraph
        compilés. Testez toujours le comportement avec votre graph spécifique.
    """
    last_error = None
    
    # =========================================================================
    # 0. Provider direct depuis session_state (Grok/Groq)
    # =========================================================================
    
    if llm is None:
        provider = st.session_state.get("provider")
        
        # --- GROK ---
        if provider == 'grok':
            api_key = st.session_state.get("grok_api_key")
            if not api_key:
                return "❌ Grok API key not provided."
            
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.x.ai/v1"
                )
                response = client.chat.completions.create(
                    model='grok-2',
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"❌ Grok LLM Error: {str(e)}"
        
        # --- GROQ ---
        elif provider == 'groq':
            try:
                from groq import Groq
            except ImportError:
                return "❌ Groq module not installed. Install with: pip install groq"
            
            api_key = st.session_state.get("groq_api_key")
            if not api_key:
                return "❌ Groq API key not provided."
            
            try:
                client = Groq(api_key=api_key)
                model = st.session_state.get("groq_model", "llama-3.1-8b-instant")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"❌ Groq LLM Error: {str(e)}"
        
        # --- LOCAL ---
        elif provider == 'local':
            return "⚠️ Local provider not yet implemented. Please select 'grok' or 'groq'."
        
        else:
            return "❌ No provider configured. Please select a provider in the sidebar."
    
    # =========================================================================
    # 1. Méthode generate() (intégrations existantes)
    # =========================================================================
    
    gen = getattr(llm, "generate", None)
    if callable(gen):
        try:
            result = gen(prompt=prompt, temperature=temperature)
            return _normalize(result)
        except TypeError:
            # Essayer sans temperature
            try:
                result = gen(prompt=prompt)
                return _normalize(result)
            except Exception as e:
                last_error = e
        except Exception as e:
            last_error = e
    
    # =========================================================================
    # 2. Méthode invoke() (LangChain LCEL, LangGraph compilé)
    # =========================================================================
    
    invoke = getattr(llm, "invoke", None)
    if callable(invoke):
        # Pattern 1: Dict avec prompt + temperature
        try:
            res = invoke({"prompt": prompt, "temperature": temperature})
            return _normalize(res)
        except Exception as e1:
            last_error = e1
            
            # Pattern 2: Dict avec "input" key
            try:
                res = invoke({"input": prompt})
                return _normalize(res)
            except Exception as e2:
                last_error = e2
                
                # Pattern 3: Dict avec "messages" (LangChain ChatModel)
                try:
                    from langchain_core.messages import HumanMessage
                    res = invoke([HumanMessage(content=prompt)])
                    return _normalize(res)
                except Exception as e3:
                    last_error = e3
                    
                    # Pattern 4: String directe (RunnableLambda)
                    try:
                        res = invoke(prompt)
                        return _normalize(res)
                    except Exception as e4:
                        last_error = e4
    
    # =========================================================================
    # 3. Patterns LangGraph Client
    # =========================================================================
    
    try:
        lg = importlib.import_module("langgraph")
    except ImportError:
        lg = None
    
    if lg is not None:
        # Déterminer le graph id
        graph_id = None
        if isinstance(llm, dict):
            graph_id = llm.get("graph_id") or llm.get("graph")
        elif isinstance(llm, str):
            graph_id = llm
        else:
            graph_id = getattr(llm, "graph_id", None) or getattr(llm, "graph", None)
        
        # Essayer de localiser un client
        Client = getattr(lg, "Client", None) or getattr(lg, "LangGraphClient", None)
        client = None
        
        if Client:
            try:
                client = Client()
            except Exception:
                pass
        
        # Essayer les méthodes du client
        if client and graph_id:
            for method_name in ("run_graph", "run", "execute", "call", "invoke"):
                fn = getattr(client, method_name, None)
                if callable(fn):
                    try:
                        res = fn(
                            graph_id, 
                            inputs={"prompt": prompt, "temperature": temperature}
                        )
                        return _normalize(res)
                    except Exception as e:
                        last_error = e
                        continue
        
        # Essayer les helpers au niveau module
        for module_fn_name in ("run_graph", "run", "execute"):
            fn = getattr(lg, module_fn_name, None)
            if callable(fn) and graph_id:
                try:
                    res = fn(
                        graph_id, 
                        inputs={"prompt": prompt, "temperature": temperature}
                    )
                    return _normalize(res)
                except Exception as e:
                    last_error = e
                    continue
    
    # =========================================================================
    # 4. Callable fallback (fonction simple)
    # =========================================================================
    
    if callable(llm):
        # Pattern 1: Keyword args
        try:
            result = llm(prompt=prompt, temperature=temperature)
            return _normalize(result)
        except TypeError:
            # Pattern 2: Positional args
            try:
                result = llm(prompt, temperature)
                return _normalize(result)
            except TypeError:
                # Pattern 3: Prompt seulement
                try:
                    result = llm(prompt)
                    return _normalize(result)
                except Exception as e:
                    last_error = e
            except Exception as e:
                last_error = e
        except Exception as e:
            last_error = e
    
    # =========================================================================
    # Échec: aucune interface trouvée
    # =========================================================================
    
    error_msg = (
        "❌ No suitable LLM interface available.\n\n"
        "Supported interfaces:\n"
        "- Provider in session_state ('grok', 'groq')\n"
        "- Object with generate() method\n"
        "- LangChain Runnable with invoke() method\n"
        "- LangGraph graph id or client\n"
        "- Callable function\n"
    )
    
    if last_error:
        error_msg += f"\n🔴 Last error: {type(last_error).__name__}: {str(last_error)}"
    
    raise RuntimeError(error_msg)


# =============================================================================
# Fonctions helper pour compatibilité
# =============================================================================

def get_llm_response(prompt: str, temperature: float = 0.7, max_tokens: int = 400) -> str:
    """
    Alias pour get_completion() pour compatibilité avec ancien code.
    Utilise automatiquement le provider de session_state.
    
    Args:
        prompt: Le prompt à envoyer
        temperature: Température d'échantillonnage
        max_tokens: Nombre maximum de tokens
    
    Returns:
        Réponse du LLM
    """
    return get_completion(
        llm=None,  # Utilisera session_state
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )


def create_llm_from_provider(provider: str = None, api_key: str = None, model: str = None):
    """
    Crée un objet LLM LangChain depuis un provider.
    
    Args:
        provider: 'grok', 'groq', ou None (utilise session_state)
        api_key: Clé API (ou None pour utiliser session_state)
        model: Nom du modèle (ou None pour défaut)
    
    Returns:
        Instance LangChain ChatModel
    
    Raises:
        ValueError: Si provider invalide ou clé manquante
    """
    # Utiliser session_state si paramètres non fournis
    if provider is None:
        provider = st.session_state.get("provider", "grok")
    
    if api_key is None:
        if provider == "grok":
            api_key = st.session_state.get("grok_api_key")
        elif provider == "groq":
            api_key = st.session_state.get("groq_api_key")
    
    if not api_key:
        raise ValueError(f"API key required for provider '{provider}'")
    
    # Créer l'instance LangChain appropriée
    if provider == "grok":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            model=model or "grok-2",
            temperature=0.7
        )
    
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=api_key,
            model=model or st.session_state.get("groq_model", "llama-3.1-8b-instant"),
            temperature=0.7
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def test_llm_connection(provider: str = None) -> dict:
    """
    Teste la connexion avec le LLM configuré.
    
    Args:
        provider: Provider à tester (ou None pour session_state)
    
    Returns:
        Dict avec status, message, et réponse optionnelle
    """
    if provider is None:
        provider = st.session_state.get("provider", "grok")
    
    test_prompt = "Reply with only the word 'OK' if you can read this."
    
    try:
        response = get_completion(
            llm=None,
            prompt=test_prompt,
            temperature=0.0,
            max_tokens=10
        )
        
        return {
            "status": "success",
            "message": f"✅ Connection to {provider} successful",
            "response": response
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"❌ Connection to {provider} failed",
            "error": str(e)
        }


# =============================================================================
# Tests unitaires (optionnel, pour développement)
# =============================================================================

if __name__ == "__main__":
    """
    Tests basiques de la fonction get_completion()
    Usage: python -m core.llm_interface
    """
    
    print("🧪 Testing llm_interface.py\n")
    
    # Test 1: Callable simple
    def mock_llm(prompt, temperature=0.7):
        return f"Mock response to: {prompt[:50]}..."
    
    try:
        result = get_completion(mock_llm, "Test prompt")
        print("✅ Test 1 (callable): PASSED")
        print(f"   Result: {result}\n")
    except Exception as e:
        print(f"❌ Test 1 (callable): FAILED - {e}\n")
    
    # Test 2: Object avec generate()
    class MockLLM:
        def generate(self, prompt, temperature=0.7):
            return {"output": f"Generated: {prompt[:30]}..."}
    
    try:
        result = get_completion(MockLLM(), "Test prompt")
        print("✅ Test 2 (generate): PASSED")
        print(f"   Result: {result}\n")
    except Exception as e:
        print(f"❌ Test 2 (generate): FAILED - {e}\n")
    
    # Test 3: Object avec invoke()
    class MockRunnable:
        def invoke(self, input_dict):
            return {"output": f"Invoked: {input_dict.get('prompt', '')[:30]}..."}
    
    try:
        result = get_completion(MockRunnable(), "Test prompt")
        print("✅ Test 3 (invoke): PASSED")
        print(f"   Result: {result}\n")
    except Exception as e:
        print(f"❌ Test 3 (invoke): FAILED - {e}\n")
    
    # Test 4: Normalisation
    test_cases = [
        ("simple string", "simple string"),
        ({"output": "dict output"}, "dict output"),
        ({"text": "dict text"}, "dict text"),
        ({"content": "dict content"}, "dict content"),
    ]
    
    all_passed = True
    for input_val, expected in test_cases:
        result = _normalize(input_val)
        if result == expected:
            print(f"✅ Normalize test: {input_val} → {result}")
        else:
            print(f"❌ Normalize test: {input_val} → {result} (expected {expected})")
            all_passed = False
    
    if all_passed:
        print("\n✅ All normalization tests PASSED\n")
    else:
        print("\n❌ Some normalization tests FAILED\n")
    
    print("🎉 Testing complete!")