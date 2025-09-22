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
                # Analyze query intent
                intent = analyze_query_intent(request.query)
                
                # Validate query against available content
                validation_result = validate_query_against_available_content(request.query)
                if not validation_result["is_valid"]:
                    error_response = {
                        "answer": validation_result["message"],
                        "confidence": 0,
                        "sources": [],
                        "available_chapters": validation_result.get("available_chapters", [])
                    }
                    yield f"data: {json.dumps(error_response)}\n\n"
                    return
                
                # Normalize and expand the query
                normalized_query = normalize_query(request.query)
                
                # For structured content requests, expand the query further
                if intent["needs_comprehensive_context"]:
                    expanded_query = expand_query_for_structured_content(request.query, intent)
                    combined_query = f"{request.query} {normalized_query} {expanded_query}"
                else:
                    combined_query = f"{request.query} {normalized_query}" if normalized_query != request.query else request.query
                
                query_vector = embeddings.embed_text(combined_query)
                
                # Search for similar chunks in Pinecone
                results = pinecone_client.index.query(
                    vector=query_vector,
                    top_k=8,
                    include_metadata=True
                )
                
                # Adjust threshold based on query type
                min_score = 0.3 if intent["needs_comprehensive_context"] else 0.5
                
                if not results.matches or results.matches[0].score < min_score:
                    error_response = {
                        "answer": "I don't have information about this topic, Please try rephrasing your question or ask about a different topic.",
                        "confidence": results.matches[0].score if results.matches else 0,
                        "sources": []
                    }
                    yield f"data: {json.dumps(error_response)}\n\n"
                    return
                
                # Use matches based on query type and score
                all_matches = []
                for match in results.matches:
                    if match.score > min_score:  # Use adjusted threshold
                        all_matches.append({
                            "text": match.metadata["text"],
                            "chunk_id": match.metadata.get("chunk_id", "unknown"),
                            "score": match.score,
                            "metadata": match.metadata
                        })
                
                reranked_chunks = rerank_chunks_fast(request.query, all_matches)
                if not reranked_chunks:
                    error_response = {
                        "answer": "I don't have information about this topic, Please try rephrasing your question or ask about a different topic.",
                        "confidence": 0,
                        "sources": []
                    }
                    yield f"data: {json.dumps(error_response)}\n\n"
                    return
                
                # Use more chunks for structured content requests
                max_chunks = 8 if intent["needs_comprehensive_context"] else 5
                context_chunks = reranked_chunks[:max_chunks]
                
                # Create well-structured context with chunk information
                context_parts = []
                for i, chunk in enumerate(context_chunks, 1):
                    context_parts.append(f"Source {i} (Score: {chunk['score']:.2f}): {chunk['text']}")
                
                context = "\n\n".join(context_parts)
                
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
                    "sources": context_chunks[:3],
                    "total_sources_found": len(reranked_chunks),
                    "context_quality": "high" if reranked_chunks[0]["rerank_score"] > 0.7 else "medium" if reranked_chunks[0]["rerank_score"] > 0.5 else "low",
                    "response_type": "comprehensive",
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
    """Enhanced query normalization for better context matching"""
    # Convert to lowercase and strip
    normalized = query.strip().lower()
    
    # Enhanced variations for better matching
    if "what is" in normalized:
        normalized += " define explain meaning concept"
    elif "dharma" in normalized:
        normalized += " duty righteousness moral obligation"
    elif "karma" in normalized:
        normalized += " action deeds consequences"
    elif "moksha" in normalized:
        normalized += " liberation freedom enlightenment"
    elif "yoga" in normalized:
        normalized += " discipline practice union"
    elif "bhakti" in normalized:
        normalized += " devotion love worship"
    elif "jnana" in normalized:
        normalized += " knowledge wisdom understanding"
    elif "karma yoga" in normalized:
        normalized += " action without attachment selfless service"
    elif "bhakti yoga" in normalized:
        normalized += " devotional service love for god"
    elif "jnana yoga" in normalized:
        normalized += " path of knowledge wisdom"
    
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
        
        # Analyze query intent
        intent = analyze_query_intent(query)
        
        # Validate query against available content
        validation_result = validate_query_against_available_content(query)
        if not validation_result["is_valid"]:
            return {
                "answer": validation_result["message"],
                "confidence": 0,
                "sources": [],
                "available_chapters": validation_result.get("available_chapters", [])
            }
        
        # Normalize and expand the query
        normalized_query = normalize_query(query)
        
        # For structured content requests, expand the query further
        if intent["needs_comprehensive_context"]:
            expanded_query = expand_query_for_structured_content(query, intent)
            combined_query = f"{query} {normalized_query} {expanded_query}"
        else:
            combined_query = f"{query} {normalized_query}" if normalized_query != query else query
        
        query_vector = embeddings.embed_text(combined_query)
        
        # OPTIMIZATION: Single Pinecone search with fewer results for speed
        results = pinecone_client.index.query(
            vector=query_vector,
            top_k=8,  # Reduced for faster processing
            include_metadata=True
        )

        # Adjust threshold based on query type
        min_score = 0.3 if intent["needs_comprehensive_context"] else 0.5
        
        if not results.matches or results.matches[0].score < min_score:
            return {
                "answer": "I don't have information about this topic, Please try rephrasing your question or ask about a different topic.",
                "confidence": results.matches[0].score if results.matches else 0,
                "sources": []
            }

        # Use matches based on query type and score
        all_matches = []
        for match in results.matches:
            if match.score > min_score:  # Use adjusted threshold
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
                "answer": "I don't have information about this topic, Please try rephrasing your question or ask about a different topic.",
                "confidence": 0,
                "sources": []
            }

        # Use more chunks for structured content requests
        max_chunks = 8 if intent["needs_comprehensive_context"] else 5
        context_chunks = reranked_chunks[:max_chunks]
        
        # Create well-structured context with chunk information
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"Source {i} (Score: {chunk['score']:.2f}): {chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using LLM
        answer = llm.generate_answer(query, context)
        
        response = {
            "answer": answer,
            "confidence": reranked_chunks[0]["rerank_score"] if reranked_chunks else 0,
            "sources": context_chunks[:3],  # Return top 3 sources
            "total_sources_found": len(reranked_chunks),
            "context_quality": "high" if reranked_chunks[0]["rerank_score"] > 0.7 else "medium" if reranked_chunks[0]["rerank_score"] > 0.5 else "low",
            "response_type": "comprehensive"
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

@router.get("/available-chapters")
def get_available_chapters():
    """Get information about what chapters are available in Pinecone"""
    try:
        # Get a sample of vectors to see what chapters are available
        sample_results = pinecone_client.index.query(
            vector=[0.0] * 1536,  # Dummy vector
            top_k=100,
            include_metadata=True
        )
        
        chapters = set()
        chunk_samples = []
        
        for match in sample_results.matches:
            chunk_id = match.metadata.get("chunk_id", "")
            chunk_samples.append({
                "chunk_id": chunk_id,
                "score": match.score,
                "text_preview": str(match.metadata.get("text", ""))[:100] + "..." if len(str(match.metadata.get("text", ""))) > 100 else str(match.metadata.get("text", ""))
            })
            
            if "chapter" in chunk_id.lower():
                # Extract chapter number from chunk_id
                import re
                chapter_match = re.search(r'chapter(\d+)', chunk_id.lower())
                if chapter_match:
                    chapters.add(f"Chapter {chapter_match.group(1)}")
        
        return {
            "available_chapters": sorted(list(chapters)),
            "total_vectors": len(sample_results.matches),
            "chunk_samples": chunk_samples[:10],  # First 10 for debugging
            "message": "These are the chapters available in the uploaded content"
        }
    except Exception as e:
        return {"error": f"Failed to get chapter information: {str(e)}"}

@router.get("/debug-pinecone")
def debug_pinecone_content():
    """Debug endpoint to see what's actually in Pinecone"""
    try:
        # Get sample results
        sample_results = pinecone_client.index.query(
            vector=[0.0] * 1536,
            top_k=20,
            include_metadata=True
        )
        
        debug_info = {
            "total_matches": len(sample_results.matches),
            "samples": []
        }
        
        for i, match in enumerate(sample_results.matches):
            debug_info["samples"].append({
                "index": i,
                "score": match.score,
                "chunk_id": match.metadata.get("chunk_id", "N/A"),
                "text_preview": str(match.metadata.get("text", ""))[:200] + "..." if len(str(match.metadata.get("text", ""))) > 200 else str(match.metadata.get("text", "")),
                "all_metadata_keys": list(match.metadata.keys()) if match.metadata else []
            })
        
        return debug_info
    except Exception as e:
        return {"error": f"Failed to debug Pinecone content: {str(e)}"}

def validate_query_against_available_content(query: str) -> dict:
    """Validate if the query is asking about content available in Pinecone"""
    import re
    
    # Extract chapter numbers from query
    chapter_matches = re.findall(r'chapter\s*(\d+)', query.lower())
    
    if chapter_matches:
        # Get available chapters with better detection
        try:
            sample_results = pinecone_client.index.query(
                vector=[0.0] * 1536,
                top_k=100,  # Increased to get more samples
                include_metadata=True
            )
            
            available_chapters = set()
            for match in sample_results.matches:
                chunk_id = match.metadata.get("chunk_id", "").lower()
                metadata_text = str(match.metadata.get("text", "")).lower()
                
                # Try multiple patterns to find chapter numbers
                patterns = [
                    r'chapter\s*(\d+)',  # "chapter 2", "chapter2"
                    r'ch\s*(\d+)',       # "ch 2", "ch2"
                    r'chapter\s*(\d+)\s*', # "chapter 2 " with space
                    r'(\d+)\s*chapter',  # "2 chapter"
                ]
                
                for pattern in patterns:
                    chapter_match = re.search(pattern, chunk_id)
                    if chapter_match:
                        available_chapters.add(int(chapter_match.group(1)))
                        break
                
                # Also check in metadata text
                for pattern in patterns:
                    chapter_match = re.search(pattern, metadata_text)
                    if chapter_match:
                        available_chapters.add(int(chapter_match.group(1)))
                        break
            
            # If no chapters found with patterns, try a broader search
            if not available_chapters:
                # Search for any content that might be chapter-related
                broader_results = pinecone_client.index.query(
                    vector=embeddings.embed_text("chapter content"),
                    top_k=20,
                    include_metadata=True
                )
                
                for match in broader_results.matches:
                    if match.score > 0.3:  # Lower threshold for broader search
                        chunk_id = match.metadata.get("chunk_id", "").lower()
                        metadata_text = str(match.metadata.get("text", "")).lower()
                        
                        for pattern in patterns:
                            chapter_match = re.search(pattern, chunk_id)
                            if chapter_match:
                                available_chapters.add(int(chapter_match.group(1)))
                                break
                            
                            chapter_match = re.search(pattern, metadata_text)
                            if chapter_match:
                                available_chapters.add(int(chapter_match.group(1)))
                                break
            
            # Check if requested chapters are available
            requested_chapters = [int(ch) for ch in chapter_matches]
            unavailable_chapters = [ch for ch in requested_chapters if ch not in available_chapters]
            
            if unavailable_chapters and available_chapters:
                return {
                    "is_valid": False,
                    "available_chapters": sorted(list(available_chapters)),
                    "unavailable_chapters": unavailable_chapters,
                    "message": f"Chapters {unavailable_chapters} are not available in the uploaded content. Available chapters: {sorted(list(available_chapters))}"
                }
            elif unavailable_chapters and not available_chapters:
                # If no chapters found at all, allow the query to proceed
                # The actual search will determine if content exists
                return {"is_valid": True, "note": "Could not detect chapter structure, proceeding with search"}
        except Exception as e:
            return {"is_valid": True, "error": f"Could not validate: {str(e)}"}
    
    return {"is_valid": True}

def analyze_query_intent(query: str) -> dict:
    """Analyze query intent and determine if it's asking for structured content"""
    import re
    query_lower = query.lower()
    
    # Keywords that indicate structured content requests
    summary_keywords = ['summary', 'summarize', 'overview', 'brief', 'main points', 'key points']
    analysis_keywords = ['analyze', 'analysis', 'explain', 'discuss', 'describe']
    learning_keywords = ['learn', 'teachings', 'lessons', 'insights', 'wisdom']
    structure_keywords = ['outline', 'structure', 'organization', 'sections']
    
    # Check for chapter-specific requests
    chapter_pattern = r'chapter\s*(\d+)'
    chapter_matches = re.findall(chapter_pattern, query_lower)
    
    # Determine query type
    query_type = "general"
    if any(keyword in query_lower for keyword in summary_keywords):
        query_type = "summary"
    elif any(keyword in query_lower for keyword in analysis_keywords):
        query_type = "analysis"
    elif any(keyword in query_lower for keyword in learning_keywords):
        query_type = "learning"
    elif any(keyword in query_lower for keyword in structure_keywords):
        query_type = "structure"
    
    return {
        "query_type": query_type,
        "is_structured_request": query_type != "general",
        "chapter_numbers": [int(ch) for ch in chapter_matches] if chapter_matches else [],
        "needs_comprehensive_context": query_type in ["summary", "analysis", "learning", "structure"]
    }

def expand_query_for_structured_content(query: str, intent: dict) -> str:
    """Expand query to better match structured content in Pinecone"""
    query_lower = query.lower()
    expanded_query = query
    
    # Add relevant keywords based on query type
    if intent["query_type"] == "summary":
        expanded_query += " main themes concepts teachings key ideas important points overview"
    elif intent["query_type"] == "analysis":
        expanded_query += " detailed explanation concepts meaning significance interpretation"
    elif intent["query_type"] == "learning":
        expanded_query += " teachings lessons wisdom insights guidance principles"
    elif intent["query_type"] == "structure":
        expanded_query += " organization sections parts flow sequence"
    
    # Add chapter-specific context if applicable
    if intent["chapter_numbers"]:
        for chapter_num in intent["chapter_numbers"]:
            expanded_query += f" chapter {chapter_num} content text verses"
            # Add common Bhagavad Gita terms that might appear in any chapter
            expanded_query += " dharma karma yoga bhakti jnana moksha"
    
    return expanded_query

@router.delete("/cache/clear")
def clear_cache():
    """Clear the response cache"""
    global response_cache
    response_cache.clear()
    return {"message": "Cache cleared successfully"}
