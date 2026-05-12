from typing import List, Dict, Any, Optional
from langchain.tools import tool
from pydantic import BaseModel, Field
import json

class WebSearchInput(BaseModel):
    """Input schema for web search"""
    query: str = Field(description="The search query to look up on the web")

class RelevanceGradeInput(BaseModel):
    """Input schema for relevance grading"""
    query: str = Field(description="The user's original question")
    document: str = Field(description="The document content to evaluate")

class HallucinationCheckInput(BaseModel):
    """Input schema for hallucination check"""
    response: str = Field(description="The generated response to check")
    context: str = Field(description="The source context used for generation")

@tool(args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """
    Perform a web search using DuckDuckGo when the knowledge base doesn't have relevant information.
    """
    try:
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                search_results = []
                for r in results:
                    search_results.append(f"Source: {r.get('title', 'Unknown')}\n{r.get('body', 'No content')}")
                return "\n\n".join(search_results)
            else:
                return "No web search results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"

@tool(args_schema=RelevanceGradeInput)
def grade_relevance(query: str, document: str) -> Dict[str, Any]:
    """
    Grade whether a retrieved document is relevant to the user's query.
    """
    return {
        "query": query,
        "document_preview": document[:200],
        "needs_llm_evaluation": True
    }

@tool(args_schema=HallucinationCheckInput)
def check_hallucination(response: str, context: str) -> Dict[str, Any]:
    """
    Check if the generated response contains claims not supported by the source context.
    """
    return {
        "response_preview": response[:200],
        "context_preview": context[:200],
        "needs_llm_evaluation": True
    }

@tool
def get_retrieval_decision(query: str) -> Dict[str, bool]:
    """
    Decide whether retrieval is needed for the given query.
    """
    return {"should_retrieve": None, "needs_llm_evaluation": True}

ALL_TOOLS = [web_search, grade_relevance, check_hallucination, get_retrieval_decision]