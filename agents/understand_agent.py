#Agent niveau 2 (Understand) → Explain, Summarize, Describe, Interpret, Give examples...import sys
import os
import random
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from core.rag_engine import retrieve


# ============================================================================
# Pydantic Models pour structured output
# ============================================================================

class Question(BaseModel):
    """Modèle d'une question Bloom niveau Understand"""
    question: str = Field(description="La question de compréhension")


class QuestionList(BaseModel):
    """Liste de questions"""
    questions: List[Question] = Field(description="Liste des questions générées")


# ============================================================================
# UnderstandAgent avec LangChain
# ============================================================================

class UnderstandAgent(BaseAgent):
    """
    Agent spécialisé pour les questions de niveau Understand (compréhension)
    Migré vers LangChain/LangGraph architecture
    """
    
    def __init__(self, llm: Any, debug: bool = False):
        """
        Args:
            llm: LangChain LLM (ChatOpenAI, ChatGroq, etc.)
            debug: Active les logs de debug
        """
        bloom_verbs = [
            "Explain", "Summarize", "Describe", "Interpret", 
            "Give examples", "Paraphrase", "Classify", "Compare", 
            "Illustrate", "Infer", "Discuss", "Predict", "Restate",
            "Expliquez", "Résumez", "Décrivez", "Interprétez",
            "Donnez des exemples", "Paraphrasez", "Classez", "Comparez",
            "Illustrez", "Déduisez", "Discutez", "Prédisez"
        ]
        
        # Initialiser la classe de base
        super().__init__(
            llm=llm,
            name="UnderstandAgent",
            bloom_level="Understand",
            allowed_verbs=bloom_verbs,
            max_context_chars=6000,
            temperature=0.7,
            retries=2,
            debug=debug
        )
        
        # Parser JSON pour structured output
        self.parser = JsonOutputParser(pydantic_object=QuestionList)
        
        # Créer la chaîne LangChain
        self.chain = self._create_chain()
    
    # =========================================================================
    # LangChain Chain Creation
    # =========================================================================
    
    def _create_chain(self):
        """
        Crée la chaîne LangChain pour la génération de questions
        """
        # Template de prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert pédagogique spécialisé dans la création de questions 
de niveau UNDERSTAND (Bloom niveau 2).

RÈGLES STRICTES:
1. Génère EXACTEMENT {num_questions} questions
2. Utilise UNIQUEMENT ces verbes: {allowed_verbs}
3. Les questions DOIVENT porter sur le contexte fourni
4. Format JSON strict
5. PAS de préambule ou texte supplémentaire
6. Chaque question évalue la COMPRÉHENSION

{format_instructions}"""),
            ("human", """Contexte:
{context}

Sujet: {topic}
Requête: {query}

Génère EXACTEMENT {num_questions} questions de compréhension:""")
        ])
        
        # Chaîne: Prompt → LLM → Parser
        chain = (
            prompt_template 
            | self.llm 
            | RunnableLambda(lambda x: self._extract_content(x))
            | self.parser
        )
        
        return chain
    
    def _extract_content(self, response):
        """Extrait le contenu de la réponse LLM"""
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        return str(response)
    
    # =========================================================================
    # Question Generation (interface publique)
    # =========================================================================
    
    def generate_questions(
        self, 
        query: str, 
        num_questions: Optional[int] = None, 
        topic: Optional[str] = None, 
        k: int = 5,
        duration: Optional[int] = None, 
        randomize_order: bool = False
    ) -> List[str]:
        """
        Génère des questions Understand en utilisant RAG + LangChain
        
        Args:
            query: Requête de recherche
            num_questions: Nombre de questions à générer
            topic: Sujet optionnel
            k: Nombre de chunks RAG à récupérer
            duration: Durée du quiz en minutes
            randomize_order: Si True, mélange l'ordre des questions
        
        Returns:
            Liste de questions (str)
        """
        # 🔢 VALIDATION: Calculer num_questions
        if num_questions is None:
            if duration and duration > 0:
                num_questions = max(1, duration // 2)
                print(f" ⏱️ Durée: {duration}min → {num_questions} questions calculées")
            else:
                num_questions = 5
                print(f" ⚠️ Utilisation valeur par défaut: {num_questions}")
        
        num_questions = max(1, int(num_questions))
        print(f" 💡 Agent Understand (LangChain) : génération de {num_questions} questions...")
        
        # 1️⃣ Enrichir la requête
        search_query = f"{topic}: {query}" if topic else query
        
        # 2️⃣ RAG Retrieval
        rag_results = retrieve(search_query, k=k)
        
        # 3️⃣ Filtrer les chunks pertinents
        filtered_results = self._filter_relevant_chunks(rag_results, query, topic)
        
        if not filtered_results:
            print(f" ⚠️ Aucun chunk pertinent, utilisation résultats bruts")
            filtered_results = rag_results
        
        if not filtered_results:
            print(" ❌ Aucun chunk disponible")
            return []
        
        # 4️⃣ Construire le contexte
        context = self.build_context(filtered_results)
        
        if not context.strip():
            print(" ⚠️ Contexte vide")
            return []
        
        # 5️⃣ Préparer les inputs pour la chaîne
        chain_input = {
            "context": context[:4000],
            "num_questions": num_questions,
            "topic": topic or "général",
            "query": query,
            "allowed_verbs": ", ".join(self.allowed_verbs[:10]),
            "format_instructions": self.parser.get_format_instructions()
        }
        
        try:
            # 6️⃣ Invoquer la chaîne LangChain avec retry
            questions = self._invoke_with_retry(chain_input, num_questions)
            
            if not questions:
                print(" ❌ Aucune question générée")
                return []
            
            # 7️⃣ Validation et nettoyage
            validated = self._validate_questions_soft(
                questions, query, topic, num_questions
            )
            
            # 8️⃣ Garantir le nombre exact
            if len(validated) < num_questions:
                if len(questions) >= num_questions:
                    validated = questions[:num_questions]
                elif len(validated) < num_questions // 2:
                    # Régénérer avec plus de chunks
                    print(f" 🔄 Régénération avec plus de chunks...")
                    return self.generate_questions(
                        query, num_questions, topic, k=k*2, 
                        duration=duration, randomize_order=randomize_order
                    )
            
            # Limiter au nombre demandé
            validated = validated[:num_questions]
            
            # 🔀 Mélanger si demandé
            if randomize_order:
                random.shuffle(validated)
                print(f" 🔀 Ordre mélangé aléatoirement")
            
            # ✅ Vérification finale
            final_count = len(validated)
            if final_count != num_questions:
                print(f" ⚠️ {final_count} générées au lieu de {num_questions}")
            else:
                print(f" ✅ {final_count} questions générées - EXACT !")
            
            return validated
            
        except Exception as e:
            print(f" ❌ Erreur génération : {e}")
            import traceback
            traceback.print_exc()
            return []
    
    # =========================================================================
    # LangChain invoke avec retry
    # =========================================================================
    
    def _invoke_with_retry(self, chain_input: Dict, num_questions: int) -> List[str]:
        """
        Invoque la chaîne LangChain avec retry logic
        """
        for attempt in range(self.retries + 1):
            try:
                # Invoquer la chaîne
                result = self.chain.invoke(chain_input)
                
                # Extraire les questions du résultat parsé
                if isinstance(result, dict) and 'questions' in result:
                    questions = [q['question'] for q in result['questions']]
                elif isinstance(result, list):
                    questions = [q['question'] if isinstance(q, dict) else str(q) 
                                for q in result]
                else:
                    questions = []
                
                if questions:
                    print(f" 📝 {len(questions)} questions parsées (tentative {attempt + 1})")
                    return questions
                
            except Exception as e:
                print(f" ⚠️ Tentative {attempt + 1} échouée: {e}")
                if attempt == self.retries:
                    raise
                continue
        
        return []
    
    # =========================================================================
    # Méthodes helper (identiques à l'original)
    # =========================================================================
    
    def _filter_relevant_chunks(
        self, 
        rag_results: List[Dict], 
        query: str, 
        topic: Optional[str] = None, 
        min_relevance: float = 0.05
    ) -> List[Dict]:
        """Filtre les chunks RAG pour garder les plus pertinents"""
        if not rag_results:
            return []
        
        # Construire les termes de recherche
        search_terms = set()
        if query:
            search_terms.update(query.lower().split())
        if topic:
            search_terms.update(topic.lower().split())
        
        # Filtrer stop words
        stop_words = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'à', 'au'}
        search_terms = {t for t in search_terms if len(t) > 2 and t not in stop_words}
        
        if not search_terms:
            return rag_results
        
        filtered = []
        for result in rag_results:
            text = result["text"].lower()
            matches = sum(1 for term in search_terms if term in text)
            relevance_score = matches / len(search_terms) if search_terms else 0
            
            if relevance_score >= min_relevance:
                filtered.append({
                    **result,
                    'relevance_score': relevance_score
                })
        
        filtered.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return filtered if filtered else rag_results
    
    def _validate_questions_soft(
        self, 
        questions: List[str], 
        query: str, 
        topic: Optional[str], 
        num_requested: int,
        min_overlap: float = 0.2
    ) -> List[str]:
        """Validation souple des questions"""
        if not questions:
            return []
        
        validated = []
        
        # Construire keywords
        keywords = set()
        if query:
            keywords.update(query.lower().split())
        if topic:
            keywords.update(topic.lower().split())
        
        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'à', 'au',
            'the', 'a', 'an', 'of', 'to', 'in', 'on', 'for', 'with'
        }
        keywords = {k for k in keywords if len(k) > 2 and k not in stop_words}
        
        for q in questions:
            # Convertir en string si dict
            if isinstance(q, dict):
                q = q.get('question', '')
            
            q_clean = q.lower()
            
            # Test de forme
            form_valid = (
                len(q) > 10 and
                ('?' in q or q.strip().endswith('.') or q.strip().endswith(':'))
            )
            
            if not form_valid:
                continue
            
            # Test contenu
            if keywords:
                matches = sum(1 for k in keywords if k in q_clean)
                overlap = matches / len(keywords)
                
                if overlap < min_overlap:
                    continue
            
            # Éviter questions génériques
            generic_patterns = [
                "objectif du projet", "but du projet", "main objective",
                "importance of", "general idea", "in the given code"
            ]
            
            if any(p in q_clean for p in generic_patterns):
                continue
            
            validated.append(q)
        
        return validated


# ============================================================================
# Factory function pour créer l'agent
# ============================================================================

def create_understand_agent(llm: Any, debug: bool = False) -> UnderstandAgent:
    """
    Factory pour créer un UnderstandAgent
    
    Args:
        llm: Instance LangChain LLM (ChatOpenAI, ChatGroq, etc.)
        debug: Active les logs de debug
    
    Returns:
        UnderstandAgent configuré
    """
    return UnderstandAgent(llm=llm, debug=debug)