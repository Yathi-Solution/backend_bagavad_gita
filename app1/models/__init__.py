from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5
    include_metadata: Optional[bool] = True

class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int

class ChatMessage(BaseModel):
    message: str
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    message: str
    relevant_passages: List[SearchResult]
    total_passages: int

class LLMChatMessage(BaseModel):
    query: str
    top_k: Optional[int] = 5
    session_id: Optional[str] = None

class LLMChatResponse(BaseModel):
    query: str
    answer: str
    relevant_passages: List[SearchResult]
    total_passages: int

__all__ = [
    "SearchQuery",
    "SearchResult", 
    "SearchResponse",
    "ChatMessage",
    "ChatResponse",
    "LLMChatMessage",
    "LLMChatResponse"
]
