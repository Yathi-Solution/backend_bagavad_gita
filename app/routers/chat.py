from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from app.services import pinecone_client, embeddings, llm
from pydantic import BaseModel
from typing import Optional
import hashlib
import json
import asyncio

router = APIRouter()

# OPTIMIZATION: Enhanced in-memory cache with TTL
import time
response_cache = {}
CACHE_TTL = 300  # 5 minutes TTL

class ChatRequest(BaseModel):
    query: str

def get_cache_key(query: str) -> str:
    """Generate a cache key for the query"""
    normalized = normalize_query(query)
    return hashlib.md5(normalized.encode()).hexdigest()

def get_cached_response(query: str) -> Optional[dict]:
    """Get cached response if available and not expired"""
    cache_key = get_cache_key(query)
    if cache_key in response_cache:
        cached_data = response_cache[cache_key]
        # OPTIMIZATION: Check TTL
        if time.time() - cached_data["timestamp"] < CACHE_TTL:
            return cached_data["response"]
        else:
            # Remove expired cache
            del response_cache[cache_key]
    return None

def cache_response(query: str, response: dict):
    """Cache the response for future use with timestamp"""
    cache_key = get_cache_key(query)
    response_cache[cache_key] = {
        "response": response,
        "timestamp": time.time()
    }
    # OPTIMIZATION: Limit cache size to prevent memory issues
    if len(response_cache) > 200:  # Increased cache size
        # Remove oldest entries (simple FIFO)
        oldest_key = min(response_cache.keys(), key=lambda k: response_cache[k]["timestamp"])
        del response_cache[oldest_key]

@router.post("/")
async def chat_post(request: ChatRequest):
    """Chat endpoint that accepts POST requests with JSON body - ASYNC"""
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, process_chat_query, request.query)

@router.get("/")
async def chat_get(query: str = Query(..., description="Your question about the Bhagavad Gita")):
    """Chat endpoint that accepts GET requests with query parameter - ASYNC"""
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, process_chat_query, query)

@router.post("/stream")
async def chat_post_stream(request: ChatRequest):
    """Streaming chat endpoint that returns real-time response"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check cache first
        cached_response = get_cached_response(request.query)
        if cached_response:
            # Return cached response as streaming
            def cached_stream():
                yield f"data: {json.dumps(cached_response)}\n\n"
            return StreamingResponse(cached_stream(), media_type="text/plain")
        
        # Process query to get context
        def process_and_stream():
            try:
                # Normalize and expand the query
                normalized_query = normalize_query(request.query)
                combined_query = f"{request.query} {normalized_query}" if normalized_query != request.query else request.query
                query_vector = embeddings.embed_text(combined_query)
                
                # Search for similar chunks in Pinecone
                results = pinecone_client.index.query(
                    vector=query_vector,
                    top_k=8,
                    include_metadata=True
                )
                
                if not results.matches or results.matches[0].score < 0.3:
                    error_response = {
                        "answer": "Sorry, I couldn't find relevant information about this topic. Please try rephrasing your question or ask about a different topic.",
                        "confidence": results.matches[0].score if results.matches else 0,
                        "sources": []
                    }
                    yield f"data: {json.dumps(error_response)}\n\n"
                    return
                
                # Prepare context
                all_matches = []
                for match in results.matches:
                    if match.score > 0.3:
                        all_matches.append({
                            "text": match.metadata["text"],
                            "chunk_id": match.metadata.get("chunk_id", "unknown"),
                            "score": match.score,
                            "metadata": match.metadata
                        })
                
                reranked_chunks = rerank_chunks_fast(request.query, all_matches)
                if not reranked_chunks:
                    error_response = {
                        "answer": "Sorry, I couldn't find relevant information about this topic.",
                        "confidence": 0,
                        "sources": []
                    }
                    yield f"data: {json.dumps(error_response)}\n\n"
                    return
                
                context_chunks = reranked_chunks[:3]
                context = " ".join([chunk['text'] for chunk in context_chunks])
                
                # Start streaming the answer
                full_answer = ""
                for chunk in llm.generate_answer_stream(request.query, context):
                    full_answer += chunk
                    # Send each chunk as it arrives
                    chunk_response = {
                        "chunk": chunk,
                        "partial_answer": full_answer,
                        "is_streaming": True
                    }
                    yield f"data: {json.dumps(chunk_response)}\n\n"
                
                # Send final complete response
                final_response = {
                    "answer": full_answer,
                    "confidence": reranked_chunks[0]["rerank_score"] if reranked_chunks else 0,
                    "sources": context_chunks[:2],
                    "is_streaming": False
                }
                
                # Cache the final response
                cache_response(request.query, final_response)
                
                yield f"data: {json.dumps(final_response)}\n\n"
                
            except Exception as e:
                error_response = {
                    "error": f"Error processing query: {str(e)}",
                    "is_streaming": False
                }
                yield f"data: {json.dumps(error_response)}\n\n"
        
        return StreamingResponse(process_and_stream(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/stream")
async def chat_get_stream(query: str = Query(..., description="Your question about the Bhagavad Gita")):
    """Streaming chat endpoint for GET requests"""
    request = ChatRequest(query=query)
    return await chat_post_stream(request)

def normalize_query(query: str) -> str:
    """OPTIMIZED: Fast query normalization"""
    # Convert to lowercase and strip
    normalized = query.strip().lower()
    
    # OPTIMIZATION: Simplified variations for speed
    if "what is" in normalized:
        normalized += " define explain meaning"
    elif "dharma" in normalized:
        normalized += " duty righteousness"
    elif "karma" in normalized:
        normalized += " action deeds"
    
    return normalized

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

def rerank_chunks_fast(query: str, chunks: list) -> list:
    """OPTIMIZED: Fast reranking with simplified algorithm"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    def calculate_relevance_score_fast(chunk):
        text_lower = chunk["text"].lower()
        text_words = set(text_lower.split())
        
        # OPTIMIZATION: Simplified scoring - only similarity + keyword overlap
        keyword_overlap = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
        similarity_score = chunk["score"]
        
        # OPTIMIZATION: Simplified combined score: 70% similarity, 30% keyword overlap
        combined_score = (similarity_score * 0.7) + (keyword_overlap * 0.3)
        
        return {
            **chunk,
            "rerank_score": combined_score
        }
    
    # Calculate rerank scores
    reranked = [calculate_relevance_score_fast(chunk) for chunk in chunks]
    
    # Sort by rerank score
    return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)

def process_chat_query(query: str):
    """Process the chat query and return response - OPTIMIZED"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check cache first
        cached_response = get_cached_response(query)
        if cached_response:
            return cached_response
        
        # Normalize and expand the query
        normalized_query = normalize_query(query)
        
        # OPTIMIZATION: Use only one optimized query instead of dual search
        # Combine original and normalized query for better results
        combined_query = f"{query} {normalized_query}" if normalized_query != query else query
        query_vector = embeddings.embed_text(combined_query)
        
        # OPTIMIZATION: Single Pinecone search with fewer results for speed
        results = pinecone_client.index.query(
            vector=query_vector,
            top_k=8,  # Reduced for faster processing
            include_metadata=True
        )

        if not results.matches or results.matches[0].score < 0.3:
            return {
                "answer": "Sorry, I couldn't find relevant information about this topic. Please try rephrasing your question or ask about a different topic.",
                "confidence": results.matches[0].score if results.matches else 0,
                "sources": []
            }

        # OPTIMIZATION: Simplified processing - convert matches directly
        all_matches = []
        for match in results.matches:
            if match.score > 0.3:  # Lower threshold for faster processing
                all_matches.append({
                    "text": match.metadata["text"],
                    "chunk_id": match.metadata.get("chunk_id", "unknown"),
                    "score": match.score,
                    "metadata": match.metadata
                })
        
        # OPTIMIZATION: Fast reranking with simplified algorithm
        reranked_chunks = rerank_chunks_fast(query, all_matches)
        
        if not reranked_chunks:
            return {
                "answer": "Sorry, I couldn't find relevant information about this topic. Please try rephrasing your question or ask about a different topic.",
                "confidence": 0,
                "sources": []
            }

        # OPTIMIZATION: Use fewer chunks for faster processing
        context_chunks = reranked_chunks[:3]  # Use only top 3 for speed
        
        # OPTIMIZATION: Simplified context string
        context = " ".join([chunk['text'] for chunk in context_chunks])
        
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

@router.get("/cache/stats")
def get_cache_stats():
    """Get cache statistics for monitoring"""
    return {
        "cache_size": len(response_cache),
        "cache_ttl": CACHE_TTL,
        "cached_queries": list(response_cache.keys())[:10]  # First 10 for debugging
    }

@router.delete("/cache/clear")
def clear_cache():
    """Clear the response cache"""
    global response_cache
    response_cache.clear()
    return {"message": "Cache cleared successfully"}
