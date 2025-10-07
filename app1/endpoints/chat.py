import os
import json
import uuid
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add the app1 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Add the services directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

# Import models directly
from models import SearchQuery, SearchResponse, ChatMessage, ChatResponse, SearchResult, LLMChatMessage, LLMChatResponse

from services.pinecone_services import PineconeService
from services.embeddings import embed_text, embed_text_async
from services.llm_service import llm_service
from services.json_processor import JSONDataProcessor
from services.supabase_context_service import supabase_context_service

# Define BASE_DIR
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

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
        response_message = f"I found {len(relevant_passages)} relevant passages from Bhagavad Gita related to your query: '{chat_data.message}'"
        
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
    Chat with Bhagavad Gita using LLM with context awareness - finds relevant passages and generates intelligent responses.
    
    Args:
        llm_chat_data: User message with optional parameters including session_id
        
    Returns:
        LLMChatResponse with AI-generated answer based on relevant passages and conversation history
    """
    try:
        # 1. Classify the user's intent first
        intent = await llm_service.classify_query_intent(llm_chat_data.query)

        # 2. If the user is just greeting the bot, handle it directly
        if intent == "GREETING":
            session_id = llm_chat_data.session_id or f"session_{int(datetime.now().timestamp())}"
            
            # Check if this is the first message in the session
            conversation_history = await asyncio.to_thread(
                supabase_context_service.get_conversation_context,
                session_id,
                limit=1
            )
            is_first_turn = not conversation_history
            
            # Dynamic Greeting Response Logic (Greeting only on first turn)
            if is_first_turn:
                greeting_answer = "Jai Srimannarayana! Welcome! I'm excited to help you explore the profound teachings of Bhagavad Gita. What would you like to learn about today?"
            else:
                greeting_answer = "Jai Srimannarayana! Great to see you again! How can I help you continue your journey through the Gita today?"
                
            # Store the assistant message to maintain history
            # Note: User message is stored by frontend to prevent duplication
            await asyncio.to_thread(
                supabase_context_service.store_message,
                session_id,
                greeting_answer,
                "assistant"
            )

            return LLMChatResponse(
                query=llm_chat_data.query,
                answer=greeting_answer,
                relevant_passages=[],
                total_passages=0,
                session_id=session_id
            )

        # NEW GUARDRAIL: Handle common sensical ChitChat questions
        elif intent == "CHITCHAT":
            session_id = llm_chat_data.session_id or f"session_{int(datetime.now().timestamp())}"

            # Provide a common-sensical, role-appropriate answer without RAG
            chit_chat_answer = (
                "As a specialized assistant for the Bhagavad Gita, my purpose is to "
                "provide philosophical guidance and textual context. I'm here to help you explore "
                "the profound wisdom of the Gita! What subject would you like to learn about today?"
            )
            
            # Store the assistant message to maintain history
            # Note: User message is stored by frontend to prevent duplication
            await asyncio.to_thread(
                supabase_context_service.store_message,
                session_id,
                chit_chat_answer,
                "assistant"
            )

            return LLMChatResponse(
                query=llm_chat_data.query,
                answer=chit_chat_answer,
                relevant_passages=[],
                total_passages=0,
                session_id=session_id
            )
        
        # 3. If not a greeting, proceed with the full RAG process
        # Generate session_id if not provided
        session_id = llm_chat_data.session_id or f"session_{int(datetime.now().timestamp())}"
        
        # Note: User message is stored by frontend to prevent duplication
        
        # --- START CONCURRENT RAG PIPELINE FIX ---
        
        # 1. Prepare blocking services and concurrent tasks
        index = pinecone_service.get_index()
        
        # Define tasks to run in parallel using asyncio.gather
        tasks = [
            # Task A: User Setup (Supabase)
            asyncio.to_thread(
                supabase_context_service.get_or_create_user,
                llm_chat_data.user_id,
                llm_chat_data.user_name
            ),
            # Task B: Context History fetch (Supabase)
            asyncio.to_thread(
                supabase_context_service.get_conversation_context,
                session_id,
                limit=5
            ),
            # Task C: Async Embedding Generation (OpenAI Async Client with Cache)
            embed_text_async(llm_chat_data.query)
        ]
        
        # Execute A, B, and C concurrently
        user_id, conversation_history, query_embedding = await asyncio.gather(*tasks)
        
        # Create conversation (depends on user_id from above)
        await asyncio.to_thread(
            supabase_context_service.get_or_create_conversation,
            session_id,
            user_id,
            f"Bhagavad Gita Chat - {llm_chat_data.query[:50]}..."
        )
        
        # 2. Vector Search (Dependent on embedding, but still blocking I/O)
        search_results = await asyncio.to_thread(
            index.query,
            vector=query_embedding,
            top_k=llm_chat_data.top_k,
            include_metadata=True
        )
        
        # 3. Process Context
        relevant_passages = []
        for match in search_results.matches:
            passage = SearchResult(
                id=match.id,
                score=match.score,
                text=match.metadata.get('text', '') if match.metadata else '',
                metadata=match.metadata or {}
            )
            relevant_passages.append(passage)
        
        # 4. Final LLM Generation (This method is already async, so we await it directly)
        llm_answer = await llm_service.generate_contextual_rag_response(
            query=llm_chat_data.query,
            retrieved_passages=relevant_passages,
            session_id=session_id,
            conversation_history=conversation_history
        )
        
        # 5. Store Bot Response (Blocking I/O, needs a thread wrap)
        message_id = await asyncio.to_thread(
            supabase_context_service.store_message,
            session_id,
            llm_answer,
            "assistant"
        )
        
        # 6. Extract Topics (CPU-bound, but fast - keep as sync call)
        topics_identified = supabase_context_service.extract_topics_from_conversation(
            conversation_history + [{"role": "user", "content": llm_chat_data.query}, {"role": "assistant", "content": llm_answer}]
        )
        
        # --- END CONCURRENT RAG PIPELINE FIX ---
        
        return LLMChatResponse(
            query=llm_chat_data.query,
            answer=llm_answer,
            relevant_passages=relevant_passages,
            total_passages=len(relevant_passages),
            session_id=session_id,
            message_id=message_id,
            topics_identified=topics_identified
        )
        
    except Exception as e:
        print(f"Error in contextual chat: {e}")
        raise HTTPException(status_code=500, detail=f"Contextual LLM chat failed: {str(e)}")

@app.post("/chat/stream")
async def chat_with_llm_stream(llm_chat_data: LLMChatMessage):
    """
    Streaming version of chat endpoint - streams response chunks as they're generated.
    
    Args:
        llm_chat_data: User message with optional parameters including session_id
        
    Returns:
        Server-Sent Events stream with AI-generated answer chunks
    """
    async def generate_stream():
        try:
            # 1. Classify the user's intent first
            intent = await llm_service.classify_query_intent(llm_chat_data.query)

            # 2. If the user is just greeting the bot, handle it directly
            if intent == "GREETING":
                session_id = llm_chat_data.session_id or f"session_{int(datetime.now().timestamp())}"
                
                # Check if this is the first message in the session
                conversation_history = await asyncio.to_thread(
                    supabase_context_service.get_conversation_context,
                    session_id,
                    limit=1
                )
                is_first_turn = not conversation_history
                
                # Dynamic Greeting Response Logic
                if is_first_turn:
                    greeting_answer = "Jai Srimannarayana! Welcome! I'm excited to help you explore the profound teachings of Bhagavad Gita. What would you like to learn about today?"
                else:
                    greeting_answer = "Jai Srimannarayana! Great to see you again! How can I help you continue your journey through the Gita today?"
                    
                # Store the assistant message
                await asyncio.to_thread(
                    supabase_context_service.store_message,
                    session_id,
                    greeting_answer,
                    "assistant"
                )

                # Send the greeting as SSE
                yield f"data: {json.dumps({'type': 'content', 'content': greeting_answer})}\n\n"
                yield f"data: {json.dumps({'type': 'metadata', 'session_id': session_id, 'total_passages': 0})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Handle CHITCHAT
            elif intent == "CHITCHAT":
                session_id = llm_chat_data.session_id or f"session_{int(datetime.now().timestamp())}"

                chit_chat_answer = (
                    "As a specialized assistant for the Bhagavad Gita, my purpose is to "
                    "provide philosophical guidance and textual context. I'm here to help you explore "
                    "the profound wisdom of the Gita! What subject would you like to learn about today?"
                )
                
                await asyncio.to_thread(
                    supabase_context_service.store_message,
                    session_id,
                    chit_chat_answer,
                    "assistant"
                )

                yield f"data: {json.dumps({'type': 'content', 'content': chit_chat_answer})}\n\n"
                yield f"data: {json.dumps({'type': 'metadata', 'session_id': session_id, 'total_passages': 0})}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # 3. If not a greeting, proceed with the full RAG process
            session_id = llm_chat_data.session_id or f"session_{int(datetime.now().timestamp())}"
            
            # Prepare blocking services and concurrent tasks
            index = pinecone_service.get_index()
            
            tasks = [
                asyncio.to_thread(
                    supabase_context_service.get_or_create_user,
                    llm_chat_data.user_id,
                    llm_chat_data.user_name
                ),
                asyncio.to_thread(
                    supabase_context_service.get_conversation_context,
                    session_id,
                    limit=5
                ),
                embed_text_async(llm_chat_data.query)
            ]
            
            user_id, conversation_history, query_embedding = await asyncio.gather(*tasks)
            
            await asyncio.to_thread(
                supabase_context_service.get_or_create_conversation,
                session_id,
                user_id,
                f"Bhagavad Gita Chat - {llm_chat_data.query[:50]}..."
            )
            
            search_results = await asyncio.to_thread(
                index.query,
                vector=query_embedding,
                top_k=llm_chat_data.top_k,
                include_metadata=True
            )
            
            relevant_passages = []
            for match in search_results.matches:
                passage = SearchResult(
                    id=match.id,
                    score=match.score,
                    text=match.metadata.get('text', '') if match.metadata else '',
                    metadata=match.metadata or {}
                )
                relevant_passages.append(passage)
            
            # Stream the LLM response
            full_answer = ""
            async for chunk in llm_service.generate_contextual_rag_response_stream(
                query=llm_chat_data.query,
                retrieved_passages=relevant_passages,
                session_id=session_id,
                conversation_history=conversation_history
            ):
                full_answer += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            
            # Store the complete message
            message_id = await asyncio.to_thread(
                supabase_context_service.store_message,
                session_id,
                full_answer,
                "assistant"
            )
            
            # Extract topics
            topics_identified = supabase_context_service.extract_topics_from_conversation(
                conversation_history + [{"role": "user", "content": llm_chat_data.query}, {"role": "assistant", "content": full_answer}]
            )
            
            # Send metadata
            metadata = {
                'type': 'metadata',
                'session_id': session_id,
                'message_id': message_id,
                'total_passages': len(relevant_passages),
                'topics_identified': topics_identified,
                'relevant_passages': [
                    {
                        'id': p.id,
                        'score': p.score,
                        'text': p.text[:100] + '...' if len(p.text) > 100 else p.text,
                        'metadata': p.metadata
                    } for p in relevant_passages
                ]
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"Error in streaming chat: {e}")
            error_data = {
                'type': 'error',
                'error': str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

@app.post("/llm")
async def chat_with_llm_simple(llm_chat_data: LLMChatMessage):
    """
    Simplified LLM chat endpoint using Supabase for session history.
    Handles RAG process with conversation context from database.
    
    Args:
        llm_chat_data: User query with optional session_id
        
    Returns:
        LLMChatResponse with AI-generated answer based on context and history
    """
    try:
        query = llm_chat_data.query
        session_id = llm_chat_data.session_id if llm_chat_data.session_id else str(uuid.uuid4())
        
        # 1. Classify the user's intent first
        intent = await llm_service.classify_query_intent(query)

        # 2. If the user is just greeting the bot, handle it directly
        if intent == "GREETING":
            # Check if this is the first message in the session
            conversation_history = await asyncio.to_thread(
                supabase_context_service.get_conversation_context,
                session_id,
                limit=1
            )
            is_first_turn = not conversation_history
            
            # Dynamic Greeting Response Logic (Greeting only on first turn)
            if is_first_turn:
                greeting_answer = "Jai Srimannarayana! Welcome! I'm excited to help you explore the profound teachings of Bhagavad Gita. What would you like to learn about today?"
            else:
                greeting_answer = "Jai Srimannarayana! Great to see you again! How can I help you continue your journey through the Gita today?"
                
            # Store the assistant message to maintain history
            # Note: User message is stored by frontend to prevent duplication
            await asyncio.to_thread(
                supabase_context_service.store_message,
                session_id,
                greeting_answer,
                "assistant"
            )

            return LLMChatResponse(
                query=query,
                answer=greeting_answer,
                relevant_passages=[],
                total_passages=0,
                session_id=session_id
            )

        # NEW GUARDRAIL: Handle common sensical ChitChat questions
        elif intent == "CHITCHAT":
            # Provide a common-sensical, role-appropriate answer without RAG
            chit_chat_answer = (
                "As a specialized assistant for the Bhagavad Gita, my purpose is to "
                "provide philosophical guidance and textual context. I'm here to help you explore "
                "the profound wisdom of the Gita! What subject would you like to learn about today?"
            )
            
            # Store the assistant message to maintain history
            # Note: User message is stored by frontend to prevent duplication
            await asyncio.to_thread(
                supabase_context_service.store_message,
                session_id,
                chit_chat_answer,
                "assistant"
            )

            return LLMChatResponse(
                query=query,
                answer=chit_chat_answer,
                relevant_passages=[],
                total_passages=0,
                session_id=session_id
            )
        
        # 3. If not a greeting, proceed with the full RAG process
        # --- START CONCURRENT RAG PIPELINE FIX FOR /llm ENDPOINT ---
        
        # 1. Prepare blocking services and concurrent tasks
        index = pinecone_service.get_index()
        user_id = llm_chat_data.user_id if llm_chat_data.user_id else "anonymous"
        
        # Define tasks to run in parallel using asyncio.gather
        tasks = [
            # Task A: User Setup (Supabase)
            asyncio.to_thread(
                supabase_context_service.get_or_create_user,
                user_id,
                llm_chat_data.user_name
            ),
            # Task B: Context History fetch (Supabase)
            asyncio.to_thread(
                supabase_context_service.get_conversation_context,
                session_id,
                limit=5
            ),
            # Task C: Async Embedding Generation (OpenAI Async Client with Cache)
            embed_text_async(query)
        ]
        
        # Execute A, B, and C concurrently
        user_id, conversation_history, query_embedding = await asyncio.gather(*tasks)
        
        # Create conversation (depends on user_id from above)
        await asyncio.to_thread(
            supabase_context_service.get_or_create_conversation,
            session_id,
            user_id,
            f"Bhagavad Gita Chat - {query[:50]}..."
        )
        
        # 2. Vector Search (Dependent on embedding, but still blocking I/O)
        search_results = await asyncio.to_thread(
            index.query,
            vector=query_embedding,
            top_k=llm_chat_data.top_k,
            include_metadata=True
        )
        
        # 3. Process Context
        context_chunks = []
        relevant_passages = []
        
        for match in search_results.matches:
            passage = SearchResult(
                id=match.id,
                score=match.score,
                text=match.metadata.get('text', '') if match.metadata else '',
                metadata=match.metadata or {}
            )
            relevant_passages.append(passage)
            context_chunks.append(passage.text)
        
        # Prepare context string
        context = "\n\n".join(context_chunks) if context_chunks else ""
        
        if not context_chunks:
            # Fallback response
            answer = "Sorry, I couldn't find relevant verses for that. Please try another question."
            raw_response = answer
        else:
            # 4. Final LLM Generation (This method is already async, so we await it directly)
            raw_response = await llm_service.generate_contextual_rag_response(
                query=query,
                retrieved_passages=relevant_passages,
                session_id=session_id,
                conversation_history=conversation_history
            )
            answer = raw_response
        
        # 5. Store Bot Response (Blocking I/O, needs a thread wrap)
        # Note: User message is stored by frontend to prevent duplication
        message_id = await asyncio.to_thread(
            supabase_context_service.store_message,
            session_id,
            answer,
            "assistant"
        )
        
        # --- END CONCURRENT RAG PIPELINE FIX FOR /llm ENDPOINT ---
        
        return LLMChatResponse(
            query=query,
            answer=answer,
            relevant_passages=relevant_passages,
            total_passages=len(relevant_passages),
            session_id=session_id,
            message_id=message_id,
            topics_identified=[]
        )
        
    except Exception as e:
        print(f"Error in LLM chat: {e}")
        raise HTTPException(status_code=500, detail=f"LLM chat failed: {str(e)}")

@app.get("/history")
async def get_chat_history(session_id: str = Query(..., description="Session ID to retrieve history for")):
    """
    Get conversation history for a session from Supabase.
    
    Args:
        session_id: The session ID to retrieve history for
        
    Returns:
        Dict with conversation history
    """
    try:
        # 5. 🟢 USE SUPABASE TO FETCH HISTORY
        history = await asyncio.to_thread(
            supabase_context_service.get_conversation_context,
            session_id
        )
        
        # The history returned is already the list of turns, ready for the frontend
        return {
            "session_id": session_id,
            "history": history
        }
        
    except Exception as e:
        print(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.post("/feedback")
async def submit_feedback(feedback_data: dict):
    """
    Submit user feedback for conversation quality.
    
    Args:
        feedback_data: Dictionary containing feedback information
        
    Returns:
        Success response with feedback ID
    """
    try:
        feedback_id = await asyncio.to_thread(
            supabase_context_service.store_feedback,
            session_id=feedback_data.get('session_id'),
            query_text=feedback_data.get('query_text', ''),
            answer_text=feedback_data.get('answer_text', ''),
            rating=feedback_data.get('rating'),
            feedback=feedback_data.get('feedback', ''),
            user_name=feedback_data.get('user_name', 'Anonymous')
        )
        
        return {
            "status": "success",
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

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

