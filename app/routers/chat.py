from fastapi import APIRouter, Query, HTTPException
from app.services import pinecone_client, embeddings, llm
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

@router.post("/")
def chat_post(request: ChatRequest):
    """Chat endpoint that accepts POST requests with JSON body"""
    return process_chat_query(request.query)

@router.get("/")
def chat_get(query: str = Query(..., description="Your question about the Bhagavad Gita")):
    """Chat endpoint that accepts GET requests with query parameter"""
    return process_chat_query(query)

def process_chat_query(query: str):
    """Process the chat query and return response"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Generate embedding for the query
        query_vector = embeddings.embed_text(query)
        
        # Search for similar chunks in Pinecone
        results = pinecone_client.index.query(
            vector=query_vector,
            top_k=5,  # Increased to get more context
            include_metadata=True
        )

        if not results.matches or results.matches[0].score < 0.5:  # Lowered threshold slightly
            return {
                "answer": "Sorry, I couldn't find relevant information about this topic in the Bhagavad Gita transcripts. Please try rephrasing your question or ask about a different topic.",
                "confidence": results.matches[0].score if results.matches else 0,
                "sources": []
            }

        # Prepare context from top matches
        context_chunks = []
        for match in results.matches:
            if match.score > 0.5:  # Use same threshold as above
                context_chunks.append({
                    "text": match.metadata["text"],
                    "chunk_id": match.metadata.get("chunk_id", "unknown"),
                    "score": match.score
                })
        
        # Create context string
        context = " ".join([chunk["text"] for chunk in context_chunks])
        
        # Generate answer using LLM
        answer = llm.generate_answer(query, context)
        
        return {
            "answer": answer,
            "confidence": results.matches[0].score,
            "sources": context_chunks[:3]  # Return top 3 sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Chat service is running"}
