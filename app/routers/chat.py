from fastapi import APIRouter, Query, HTTPException
from app.services import pinecone_client, embeddings, llm
from pydantic import BaseModel
from typing import Optional
import hashlib
import json

router = APIRouter()

# Simple in-memory cache for responses
response_cache = {}

class ChatRequest(BaseModel):
    query: str

def get_cache_key(query: str) -> str:
    """Generate a cache key for the query"""
    normalized = normalize_query(query)
    return hashlib.md5(normalized.encode()).hexdigest()

def get_cached_response(query: str) -> Optional[dict]:
    """Get cached response if available"""
    cache_key = get_cache_key(query)
    return response_cache.get(cache_key)

def cache_response(query: str, response: dict):
    """Cache the response for future use"""
    cache_key = get_cache_key(query)
    response_cache[cache_key] = response
    # Limit cache size to prevent memory issues
    if len(response_cache) > 100:
        # Remove oldest entries (simple FIFO)
        oldest_key = next(iter(response_cache))
        del response_cache[oldest_key]

@router.post("/")
def chat_post(request: ChatRequest):
    """Chat endpoint that accepts POST requests with JSON body"""
    return process_chat_query(request.query)

@router.get("/")
def chat_get(query: str = Query(..., description="Your question about the Bhagavad Gita")):
    """Chat endpoint that accepts GET requests with query parameter"""
    return process_chat_query(query)

def normalize_query(query: str) -> str:
    """Normalize and expand the query for better consistency"""
    # Convert to lowercase and strip
    normalized = query.strip().lower()
    
    # Common variations and synonyms for better matching
    query_variations = {
        "what is": ["define", "explain", "meaning of", "definition of"],
        "vishada yoga": ["arjuna vishada yoga", "first chapter", "chapter 1", "arjuna's sorrow"],
        "dharma": ["duty", "righteousness", "moral duty"],
        "karma": ["action", "deeds", "work"],
        "moksha": ["liberation", "freedom", "enlightenment"]
    }
    
    # Add variations to improve matching
    expanded_query = normalized
    for key, variations in query_variations.items():
        if key in normalized:
            expanded_query += " " + " ".join(variations)
    
    return expanded_query

def rerank_chunks(query: str, chunks: list) -> list:
    """Rerank chunks based on keyword relevance and context quality"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    def calculate_relevance_score(chunk):
        text_lower = chunk["text"].lower()
        text_words = set(text_lower.split())
        
        # Keyword overlap score
        keyword_overlap = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
        
        # Length bonus for more comprehensive chunks
        length_bonus = min(len(chunk["text"]) / 1000, 0.2)  # Cap at 0.2
        
        # Original similarity score
        similarity_score = chunk["score"]
        
        # Combined score: 60% similarity, 30% keyword overlap, 10% length
        combined_score = (similarity_score * 0.6) + (keyword_overlap * 0.3) + (length_bonus * 0.1)
        
        return {
            **chunk,
            "rerank_score": combined_score
        }
    
    # Calculate rerank scores
    reranked = [calculate_relevance_score(chunk) for chunk in chunks]
    
    # Sort by rerank score
    return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)

def process_chat_query(query: str):
    """Process the chat query and return response"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check cache first
        cached_response = get_cached_response(query)
        if cached_response:
            return cached_response
        
        # Normalize and expand the query
        normalized_query = normalize_query(query)
        
        # Generate embedding for both original and normalized query
        original_vector = embeddings.embed_text(query)
        normalized_vector = embeddings.embed_text(normalized_query)
        
        # Search for similar chunks in Pinecone with both vectors
        results_original = pinecone_client.index.query(
            vector=original_vector,
            top_k=10,  # Get more candidates
            include_metadata=True
        )
        
        results_normalized = pinecone_client.index.query(
            vector=normalized_vector,
            top_k=10,
            include_metadata=True
        )
        
        # Combine and deduplicate results
        all_matches = {}
        for match in results_original.matches + results_normalized.matches:
            chunk_id = match.metadata.get("chunk_id", "unknown")
            if chunk_id not in all_matches or match.score > all_matches[chunk_id]["score"]:
                all_matches[chunk_id] = {
                    "text": match.metadata["text"],
                    "chunk_id": chunk_id,
                    "score": match.score,
                    "metadata": match.metadata
                }
        
        # Sort by score and apply reranking
        sorted_matches = sorted(all_matches.values(), key=lambda x: x["score"], reverse=True)
        
        # Rerank based on keyword relevance and context quality
        reranked_chunks = rerank_chunks(query, sorted_matches[:15])
        
        if not reranked_chunks or reranked_chunks[0]["score"] < 0.4:  # Slightly lower threshold
            return {
                "answer": "Sorry, I couldn't find relevant information about this topic. Please try rephrasing your question or ask about a different topic.",
                "confidence": reranked_chunks[0]["score"] if reranked_chunks else 0,
                "sources": []
            }

        # Prepare context from top reranked matches
        context_chunks = reranked_chunks[:5]  # Use top 5 after reranking
        
        # Create context string with better organization
        context = "\n\n".join([f"Source {i+1}: {chunk['text']}" for i, chunk in enumerate(context_chunks)])
        
        # Generate answer using LLM
        answer = llm.generate_answer(query, context)
        
        response = {
            "answer": answer,
            "confidence": reranked_chunks[0]["rerank_score"] if reranked_chunks else 0,
            "sources": context_chunks[:3]  # Return top 3 sources
        }
        
        # Cache the response
        cache_response(query, response)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Chat service is running"}
