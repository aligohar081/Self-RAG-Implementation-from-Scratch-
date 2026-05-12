from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
import os
from config import *

# ==========================================
# State Definition
# ==========================================
class AgentState(TypedDict):
    """State for the Self-RAG Agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    should_retrieve: bool
    retrieved_docs: List[Dict[str, Any]]
    relevant_docs: List[Dict[str, Any]]
    web_search_results: Optional[str]
    current_response: str
    hallucination_checked: bool
    regeneration_attempts: int
    final_response: str
    execution_trace: List[Dict[str, Any]]

# ==========================================
# Initialize LLM
# ==========================================
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.1
)

# ==========================================
# Self-RAG Graph Implementation
# ==========================================
class SelfRAGGraph:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.graph = self._build_graph()
        
    def _add_to_trace(self, state: AgentState, step: str, data: Dict[str, Any]) -> AgentState:
        """Add execution trace entry"""
        if "execution_trace" not in state:
            state["execution_trace"] = []
        state["execution_trace"].append({
            "step": step,
            **data
        })
        return state
    
    # --- Nodes ---

    def _adaptive_retrieval_decision(self, state: AgentState) -> AgentState:
        """Node 1: Decide if retrieval is needed"""
        query = state["query"]
        
        # Patterns that don't need retrieval
        no_retrieval_patterns = [
            "hi", "hello", "hey", "greetings", "good morning", 
            "what is gpa", "what does gpa stand for", "how are you", 
            "thanks", "thank you", "goodbye", "bye", "how's it going"
        ]
        
        query_lower = query.lower()
        should_retrieve = True
        
        # Check if query matches no-retrieval patterns
        for pattern in no_retrieval_patterns:
            if pattern in query_lower:
                should_retrieve = False
                break
        
        # Use LLM for complex decisions
        if should_retrieve:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a retrieval decision agent. Return ONLY 'YES' or 'NO'.
                
Answer YES if the query asks about SPECIFIC university information:
- Courses, prerequisites, credit hours
- Department details (CS, EE, BBA)
- University policies, grading, fees
- Faculty information

Answer NO if query is:
- A greeting or conversation
- General knowledge questions

Query: {query}
Decision:"""),
            ])
            
            chain = prompt | llm
            response = chain.invoke({"query": query})
            decision = response.content.strip().upper()
            should_retrieve = decision == "YES"
        
        state["should_retrieve"] = should_retrieve
        state = self._add_to_trace(state, "adaptive_retrieval", {
            "should_retrieve": should_retrieve
        })
        return state
    
    def _retrieve_documents(self, state: AgentState) -> AgentState:
        """Node 2: Retrieve documents"""
        if not state["should_retrieve"]:
            return state
        
        query = state["query"]
        
        try:
            docs = self.vector_store.similarity_search_with_score(query, k=TOP_K_RETRIEVAL)
            
            retrieved_docs = []
            for doc, score in docs:
                retrieved_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score),
                    "relevant": None
                })
            
            state["retrieved_docs"] = retrieved_docs
            state = self._add_to_trace(state, "retrieve_documents", {
                "num_docs": len(retrieved_docs)
            })
        except Exception as e:
            state["retrieved_docs"] = []
            state = self._add_to_trace(state, "retrieve_error", {"error": str(e)})
        
        return state
    
    def _grade_relevance(self, state: AgentState) -> AgentState:
        """Node 3: Grade relevance"""
        if not state.get("retrieved_docs"):
            state["relevant_docs"] = []
            return state
        
        query = state["query"]
        relevant_docs = []
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a relevance grader. Return ONLY 'RELEVANT' or 'IRRELEVANT'.

Query: {query}

Document: {document}

Judgment:"""),
        ])
        
        chain = prompt | llm
        
        for doc in state["retrieved_docs"]:
            try:
                response = chain.invoke({
                    "query": query,
                    "document": doc["content"][:1000]
                })
                is_relevant = response.content.strip().upper() == "RELEVANT"
                doc["relevant"] = is_relevant
                if is_relevant:
                    relevant_docs.append(doc)
            except Exception as e:
                doc["relevant"] = False
        
        state["relevant_docs"] = relevant_docs
        state = self._add_to_trace(state, "grade_relevance", {
            "relevant": len(relevant_docs)
        })
        return state
    
    def _web_search_fallback(self, state: AgentState) -> AgentState:
        """Node 4: Web search fallback"""
        query = state["query"]
        
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=2))
                if results:
                    search_results = []
                    for r in results:
                        search_results.append(f"Source: {r.get('title', 'Unknown')}\n{r.get('body', 'No content')}")
                    state["web_search_results"] = "\n\n".join(search_results)
                else:
                    state["web_search_results"] = "No web results found"
            
            state = self._add_to_trace(state, "web_search", {"status": "success"})
        except Exception as e:
            state["web_search_results"] = f"Web search error: {str(e)}"
            state = self._add_to_trace(state, "web_search_error", {"error": str(e)})
        
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Node 5: Generate response"""
        query = state["query"]
        context = ""
        
        if state.get("relevant_docs"):
            context = "\n\n".join([doc["content"][:500] for doc in state["relevant_docs"][:3]])
            context_type = "catalog"
        elif state.get("web_search_results"):
            context = state["web_search_results"]
            context_type = "web"
        else:
            context_type = "direct"
        
        print(f"\n[DEBUG] Generating response with context_type: {context_type}")
        
        if context_type == "direct":
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful University Course Advisory Agent. Answer conversationally."),
                ("human", "{query}")
            ])
            chain = prompt | llm
            response = chain.invoke({"query": query})
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a University Course Advisory Agent. Answer ONLY using this context:

{context}

If the context doesn't have the answer, say 'I don't have that information in my knowledge base.' Don't invent anything."""),
                ("human", "{query}")
            ])
            chain = prompt | llm
            response = chain.invoke({"query": query})
        
        print(f"[DEBUG] Raw response content: {response.content}")
        print(f"[DEBUG] Response length: {len(response.content)}")
        
        state["current_response"] = response.content
        state = self._add_to_trace(state, "generate", {"type": context_type})
        return state
    
    def _check_hallucination(self, state: AgentState) -> AgentState:
        """Node 6: Check hallucinations - IMPROVED VERSION"""
        response = state["current_response"]
        context = ""
        
        if state.get("relevant_docs"):
            context = "\n".join([doc["content"][:800] for doc in state["relevant_docs"]])
        elif state.get("web_search_results"):
            context = state["web_search_results"][:2000]
        
        if not context:
            state["hallucination_checked"] = True
            state["final_response"] = response
            return state
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a hallucination checker for a university advisory system. Be CAREFUL - many responses that seem like hallucinations are actually valid.

A response is ONLY a HALLUCINATION if it:
1. States specific facts that are CONTRADICTED by the context
2. Claims information exists that is NOT PRESENT in the context AND cannot be reasonably inferred

A response is GROUNDED if it:
1. Directly quotes or paraphrases information from the context
2. Makes reasonable inferences based on the context
3. States "I don't have that information" when the context lacks it
4. Provides general information that is common knowledge

Return ONLY 'GROUNDED' or 'HALLUCINATION'.

Context:
{context}

Response:
{response}

Judgment:"""),
        ])
        
        chain = prompt | llm
        try:
            result = chain.invoke({"context": context[:2000], "response": response})
            is_hallucination = result.content.strip().upper() == "HALLUCINATION"
            print(f"[DEBUG] Hallucination check result: {result.content.strip()}")
        except Exception as e:
            print(f"[DEBUG] Hallucination check error: {e}")
            is_hallucination = False
        
        state["hallucination_checked"] = not is_hallucination
        
        if is_hallucination:
            state["regeneration_attempts"] = state.get("regeneration_attempts", 0) + 1
            print(f"[DEBUG] Hallucination detected! Attempt {state['regeneration_attempts']}/{MAX_REGENERATION_ATTEMPTS}")
        else:
            state["final_response"] = response
        
        state = self._add_to_trace(state, "hallucination_check", {
            "is_hallucination": is_hallucination,
            "attempt": state.get("regeneration_attempts", 0)
        })
        return state
    
    # --- Conditional Edge Logic ---

    def _should_retrieve(self, state: AgentState) -> Literal["retrieve", "generate_direct"]:
        return "retrieve" if state["should_retrieve"] else "generate_direct"
    
    def _should_use_web_search(self, state: AgentState) -> Literal["web_search", "generate"]:
        if len(state.get("relevant_docs", [])) == 0 and state["should_retrieve"]:
            return "web_search"
        return "generate"
    
    def _should_regenerate_or_finalize(self, state: AgentState) -> Literal["regenerate", "finalize"]:
        """Decide whether to regenerate or finalize"""
        if state.get("final_response"):
            return "finalize"
        
        if state.get("hallucination_checked", False):
            state["final_response"] = state["current_response"]
            return "finalize"
        
        if state.get("regeneration_attempts", 0) < MAX_REGENERATION_ATTEMPTS:
            return "regenerate"
        else:
            state["final_response"] = state["current_response"] + "\n\n[Note: Please verify this information]"
            return "finalize"
    
    # --- Graph Construction ---

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("adaptive_decision", self._adaptive_retrieval_decision)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("grade_relevance", self._grade_relevance)
        workflow.add_node("web_search", self._web_search_fallback)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("generate_direct", self._generate_response)
        workflow.add_node("check_hallucination", self._check_hallucination)
        workflow.add_node("regenerate", self._generate_response)
        
        # Set entry point
        workflow.set_entry_point("adaptive_decision")
        
        # Add edges
        workflow.add_conditional_edges(
            "adaptive_decision", 
            self._should_retrieve,
            {
                "retrieve": "retrieve",
                "generate_direct": "generate_direct"
            }
        )
        
        workflow.add_edge("retrieve", "grade_relevance")
        
        workflow.add_conditional_edges(
            "grade_relevance", 
            self._should_use_web_search,
            {
                "web_search": "web_search",
                "generate": "generate"
            }
        )
        
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", "check_hallucination")
        workflow.add_edge("generate_direct", "check_hallucination")
        
        workflow.add_conditional_edges(
            "check_hallucination", 
            self._should_regenerate_or_finalize,
            {
                "regenerate": "regenerate",
                "finalize": END
            }
        )
        
        workflow.add_edge("regenerate", "check_hallucination")
        
        return workflow.compile()
    
    def run(self, query: str) -> Dict[str, Any]:
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "should_retrieve": False,
            "retrieved_docs": [],
            "relevant_docs": [],
            "web_search_results": None,
            "current_response": "",
            "hallucination_checked": False,
            "regeneration_attempts": 0,
            "final_response": "",
            "execution_trace": []
        }
        
        result = self.graph.invoke(initial_state)
        return result