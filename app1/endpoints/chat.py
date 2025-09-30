import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import sys

# Add the app1 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Add the services directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

# Import models directly
from models import SearchQuery, SearchResponse, ChatMessage, ChatResponse, SearchResult, LLMChatMessage, LLMChatResponse

from services.pinecone_services import PineconeService
from services.embeddings import embed_text
from services.llm_service import llm_service
from services.json_processor import JSONDataProcessor

load_dotenv()

app = FastAPI(title="Bhagavad Gita Search API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize services
pinecone_service = PineconeService()
json_processor = JSONDataProcessor()

# Mount static files
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# @app.get("/")
# async def root():
#     return {"message": "Bhagavad Gita Search API", "status": "running"}

@app.get("/")
async def serve_frontend():
    """Serve the frontend application"""
    frontend_file = os.path.join(os.path.dirname(__file__), '..', 'index.html')
    return FileResponse(frontend_file)

@app.post("/search", response_model=SearchResponse)
async def search_bhagavad_gita(query_data: SearchQuery):
    """
    Search through Bhagavad Gita Chapter 1 embeddings using semantic similarity.
    
    Args:
        query_data: Search query with optional parameters
        
    Returns:
        SearchResponse with relevant passages and metadata
    """
    try:
        # Get the Pinecone index
        index = pinecone_service.get_index()
        
        # Generate embedding for the query
        query_embedding = embed_text(query_data.query)
        
        # Search in Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=query_data.top_k,
            include_metadata=query_data.include_metadata
        )
        
        # Format results
        results = []
        for match in search_results.matches:
            result = SearchResult(
                id=match.id,
                score=match.score,
                text=match.metadata.get('text', '') if match.metadata else '',
                metadata=match.metadata or {}
            )
            results.append(result)
        
        return SearchResponse(
            query=query_data.query,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/pinecone/search", response_model=ChatResponse)
async def chat_with_bhagavad_gita(chat_data: ChatMessage):
    """
    Chat with Bhagavad Gita - finds relevant passages based on user message.
    
    Args:
        chat_data: User message with optional parameters
        
    Returns:
        ChatResponse with relevant passages from Bhagavad Gita
    """
    try:
        # Get the Pinecone index
        index = pinecone_service.get_index()
        
        # Generate embedding for the user message
        query_embedding = embed_text(chat_data.message)
        
        # Search for relevant passages
        search_results = index.query(
            vector=query_embedding,
            top_k=chat_data.top_k,
            include_metadata=True
        )
        
        # Format results
        relevant_passages = []
        for match in search_results.matches:
            passage = SearchResult(
                id=match.id,
                score=match.score,
                text=match.metadata.get('text', '') if match.metadata else '',
                metadata=match.metadata or {}
            )
            relevant_passages.append(passage)
        
        # Create a response message
        response_message = f"I found {len(relevant_passages)} relevant passages from Bhagavad Gita Chapter 1 related to your query: '{chat_data.message}'"
        
        return ChatResponse(
            message=response_message,
            relevant_passages=relevant_passages,
            total_passages=len(relevant_passages)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/process-json")
async def process_json_data():
    """
    Process JSON data and store embeddings in Pinecone.
    
    Returns:
        Processing results and statistics
    """
    try:
        result = json_processor.process_json_file()
        
        return {
            "status": "success",
            "message": "JSON data processed and stored in Pinecone",
            "results": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON processing failed: {str(e)}")

@app.get("/episodes")
async def get_episodes():
    """
    Get information about available episodes in the index.
    """
    try:
        index = pinecone_service.get_index()
        
        # Get stats about the index
        stats_info = pinecone_service.get_index_stats()
        
        return {
            "index_name": pinecone_service.index_name,
            "total_vectors": stats_info['stats'].total_vector_count,
            "dimension": stats_info['stats'].dimension,
            "source_breakdown": stats_info['source_breakdown'],
            "message": "Bhagavad Gita Chapter 1 episodes are available for search"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get episode info: {str(e)}")

@app.post("/chat", response_model=LLMChatResponse)
async def chat_with_llm(llm_chat_data: LLMChatMessage):
    """
    Chat with Bhagavad Gita using LLM - finds relevant passages and generates intelligent responses.
    
    Args:
        llm_chat_data: User message with optional parameters
        
    Returns:
        LLMChatResponse with AI-generated answer based on relevant passages
    """
    try:
        # Get the Pinecone index
        index = pinecone_service.get_index()
        
        # Generate embedding for the user query
        query_embedding = embed_text(llm_chat_data.query)
        
        # Search for relevant passages
        search_results = index.query(
            vector=query_embedding,
            top_k=llm_chat_data.top_k,
            include_metadata=True
        )
        
        # Format results for LLM
        relevant_passages = []
        for match in search_results.matches:
            passage = SearchResult(
                id=match.id,
                score=match.score,
                text=match.metadata.get('text', '') if match.metadata else '',
                metadata=match.metadata or {}
            )
            relevant_passages.append(passage)
        
        # Generate LLM response using RAG
        llm_answer = llm_service.generate_rag_response(
            query=llm_chat_data.query,
            retrieved_passages=relevant_passages
        )
        
        return LLMChatResponse(
            query=llm_chat_data.query,
            answer=llm_answer,
            relevant_passages=relevant_passages,
            total_passages=len(relevant_passages)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM chat failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Pinecone connection
        index = pinecone_service.get_index()
        stats = index.describe_index_stats()
        
        return {
            "status": "healthy",
            "pinecone_connected": True,
            "index_name": pinecone_service.index_name,
            "total_vectors": stats.total_vector_count
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "pinecone_connected": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

