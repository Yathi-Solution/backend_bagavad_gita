from fastapi import APIRouter, Query, HTTPException, Request, Response
from app.services import pinecone_client, embeddings, llm
from pydantic import BaseModel
from typing import Optional, Dict, List
import re
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime
import uuid
import random

load_dotenv()

# In-memory conversation history storage (for demo purposes) need to use other db for context storage 
conversation_history: Dict[str, List[Dict]] = {}

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

# get_topics_response() function has been removed as per your request.
    
def get_no_knowledge_response(language: str, user_query: str) -> str:
    """Dynamically generates a no-answer response with related topic suggestions using the LLM."""
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        
        prompt = f"""The user asked a question in {language} about: "{user_query}". I couldn't find a direct answer in my knowledge base. Generate 3-4 related topics from the Bhagavad Gita that I can offer as suggestions. The topics should be general and inviting. Respond in the same language.
        
        Example in English:
        Based on your question, I can tell you about:
        - The nature of the soul (atman)
        - The different paths of Yoga (Karma Yoga, Bhakti Yoga)
        - Krishna's role as a divine guide
        
        Generate a similar response for the user's query in {language}.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7 # Higher temperature for more creative suggestions
        )
        
        # Combine a random apology with the LLM-generated suggestions
        apology_responses = {
            'english': [
                "Sorry, I couldn't find verses for that. Maybe try asking in another way?",
                "Hmm, I don't see a reference for that. Want me to share something related instead?",
                "Apologies 🙏, I don't have that exact answer. But I can guide you with related wisdom.",
                "Sorry, I couldn't find that specific information. However, I can help you with these related subjects:",
                "I apologize, but I couldn't find relevant information about this specific topic. However, I can help you with these related subjects:",
                "Sorry, I don't have that exact answer in my knowledge base. But I can tell you about these related topics:",
                "Hmm, I couldn't find that specific reference. Want to explore these related concepts instead?",
                "Sorry, I couldn't find the information you're looking for. But I can help you with these related subjects:"
            ],
            'telugu': [
                "క్షమించండి, ఈ నిర్దిష్ట అంశం గురించి సంబంధిత సమాచారం నాకు దొరకలేదు. అయితే, ఈ సంబంధిత విషయాలతో నేను మీకు సహాయపడగలను:",
                "క్షమించండి, మీరు వెతుకుతున్న సమాచారం నాకు దొరకలేదు. దయచేసి భగవద్గీతలోని వేరే అంశం గురించి అడగడానికి ప్రయత్నించండి.",
                "క్షమించండి, నాకు ఆ నిర్దిష్ట సమాధానం లేదు. కానీ ఈ సంబంధిత అంశాల గురించి చెప్పగలను:",
                "క్షమించండి, నాకు ఆ సమాచారం దొరకలేదు. కానీ ఈ సంబంధిత విషయాలతో సహాయపడగలను:",
                "క్షమించండి, నాకు ఆ నిర్దిష్ట సమాచారం లేదు. కానీ ఈ సంబంధిత అంశాల గురించి చెప్పగలను:",
                "క్షమించండి, నాకు ఆ సమాధానం దొరకలేదు. కానీ ఈ సంబంధిత అంశాల గురించి చెప్పగలను:",
                "క్షమించండి, నాకు ఆ సమాచారం లేదు. కానీ ఈ సంబంధిత విషయాలతో సహాయపడగలను:",
                "క్షమించండి, నాకు ఆ నిర్దిష్ట సమాధానం లేదు. కానీ ఈ సంబంధిత అంశాల గురించి చెప్పగలను:"
            ]
        }
        apology = random.choice(apology_responses.get(language, apology_responses['english']))
        suggestions = response.choices[0].message.content.strip()
        
        return f"{apology}\n\n{suggestions}"
    
    except Exception:
        # Fallback to a simple static response if LLM call fails
        no_knowledge_responses = {
            'english': [
                "Sorry, I couldn't find the information you're looking for. Please try asking about a different topic from the Bhagavad Gita.",
                "I apologize, but I don't have that information in my knowledge base. Please try asking about another topic from the Gita.",
                "Sorry, I couldn't find that specific answer. Please try asking about a different aspect of the Bhagavad Gita.",
                "I don't have that information available. Please try asking about another topic from the Bhagavad Gita.",
                "Sorry, I couldn't find that in my knowledge base. Please try asking about a different topic from the Gita.",
                "I apologize, but I don't have that specific information. Please try asking about another aspect of the Bhagavad Gita.",
                "Sorry, I couldn't find that answer. Please try asking about a different topic from the Bhagavad Gita.",
                "I don't have that information. Please try asking about another topic from the Gita."
            ],
            'telugu': [
                "క్షమించండి, మీరు వెతుకుతున్న సమాచారం నాకు దొరకలేదు. దయచేసి భగవద్గీతలోని వేరే అంశం గురించి అడగడానికి ప్రయత్నించండి.",
                "క్షమించండి, నాకు ఆ సమాచారం లేదు. దయచేసి భగవద్గీతలోని వేరే అంశం గురించి అడగండి.",
                "క్షమించండి, నాకు ఆ సమాధానం లేదు. దయచేసి భగవద్గీతలోని వేరే అంశం గురించి అడగండి.",
                "నాకు ఆ సమాచారం లేదు. దయచేసి భగవద్గీతలోని వేరే అంశం గురించి అడగండి.",
                "క్షమించండి, నాకు ఆ సమాచారం లేదు. దయచేసి భగవద్గీతలోని వేరే అంశం గురించి అడగండి.",
                "క్షమించండి, నాకు ఆ సమాధానం లేదు. దయచేసి భగవద్గీతలోని వేరే అంశం గురించి అడగండి.",
                "క్షమించండి, నాకు ఆ సమాచారం లేదు. దయచేసి భగవద్గీతలోని వేరే అంశం గురించి అడగండి.",
                "నాకు ఆ సమాచారం లేదు. దయచేసి భగవద్గీతలోని వేరే అంశం గురించి అడగండి."
            ]
        }
        return random.choice(no_knowledge_responses.get(language, no_knowledge_responses['english']))

def get_or_create_session_id(session_id: Optional[str]) -> str:
    """Get existing session_id or create a new one"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    return session_id

def add_to_conversation_history(session_id: str, user_query: str, bot_response: str, language: str):
    """Add conversation turn to history"""
    conversation_history[session_id].append({
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "bot_response": bot_response,
        "language": language
    })
    
    # Keep only last 10 conversation turns to avoid memory issues
    if len(conversation_history[session_id]) > 10:
        conversation_history[session_id] = conversation_history[session_id][-10:]

def get_conversation_context(session_id: str, language: str) -> str:
    """Get recent conversation context for the LLM"""
    if session_id not in conversation_history or not conversation_history[session_id]:
        return ""
    
    recent_turns = conversation_history[session_id][-7:]
    
    context_parts = []
    for turn in recent_turns:
        context_parts.append(f"Previous Q: {turn['user_query']}")
        context_parts.append(f"Previous A: {turn['bot_response']}")
    
    context = "\n".join(context_parts)
    
    language_instructions = {
        'english': "Consider the previous conversation context when answering. Reference previous questions and answers when relevant.",
        'telugu': "మునుపటి సంభాషణ సందర్భాన్ని పరిగణనలోకి తీసుకొని జవాబు ఇవ్వండి. సంబంధితమైనప్పుడు మునుపటి ప్రశ్నలు మరియు జవాబులను సూచించండి."
    }
    
    instruction = language_instructions.get(language, language_instructions['english'])
    return f"{instruction}\n\nPrevious conversation:\n{context}\n\n"

def get_structured_conversation_messages(session_id: str, language: str) -> list[Dict[str, str]]:
    """Return structured chat messages from recent history for LLM consumption"""
    if session_id not in conversation_history or not conversation_history[session_id]:
        return []

    # Create comprehensive system prompt that includes conversation context instructions
    language_instructions = {
        'english': "You are a caring and empathetic Bhagavad Gita teacher chatbot. You MUST answer questions ONLY based on the provided document context from Chapter 1-3. You MUST NOT use any external knowledge. If the answer is not in the document context, you MUST respond by saying \"Sorry, the information to answer your question is not in my current knowledge base.\" IMPORTANT: You must understand when the user is asking a follow-up question, even if it's vague like 'are you sure?', 'what do you mean?', or 'tell me more'. Use the previous 7 turns of conversation history to infer context and respond empathetically. When asked 'are you sure?', respond with playful but respectful confirmation like 'Yes, absolutely! 😊 Based on the Gita's verses, here's why...' Respond like a caring teacher who understands the user's feelings and intent. Keep answers natural and human, sometimes using 2-3 sentences for clarity. Avoid robotic tone. Be warm, conversational, and empathetic. Use the context to answer the question as accurately as possible. If the context contains relevant information, provide a helpful answer. Only say 'Sorry, this is not in the transcripts' if the context truly doesn't contain any relevant information to answer the question. **Think step-by-step before answering.** Detail your thought process, including how you analyzed the user's query and how you used the context to formulate the answer. CRITICAL: You MUST respond ONLY in English. Do not use any other language.",
        'telugu': "మీరు భగవద్గీత గురించి సహాయపడే సంతోషకరమైన మరియు సానుభూతిపరుడైన చాట్‌బాట్. మీరు అందించిన పత్ర సందర్భంలో మాత్రమే ప్రశ్నలకు సమాధానం ఇవ్వాలి. మీరు బాహ్య జ్ఞానాన్ని ఉపయోగించకూడదు. సమాధానం పత్ర సందర్భంలో లేకపోతే, \"క్షమించండి, మీ ప్రశ్నకు సమాధానం ఇవ్వడానికి సమాచారం నా ప్రస్తుత జ్ఞాన ఆధారంలో లేదు\" అని చెప్పాలి. ముఖ్యమైనది: వినియోగదారు ఫాలో-అప్ ప్రశ్న అడుగుతున్నప్పుడు మీరు అర్థం చేసుకోవాలి, అది 'మీరు ఖచ్చితంగా ఉన్నారా?', 'మీరు ఏమి అర్థం చేసుకున్నారు?', లేదా 'మరింత చెప్పండి' వంటిది అయినా. సందర్భాన్ని అర్థం చేసుకోవడానికి మునుపటి 7 సంభాషణలను ఉపయోగించండి మరియు సానుభూతితో ప్రతిస్పందించండి. 'మీరు ఖచ్చితంగా ఉన్నారా?' అని అడిగినప్పుడు, 'అవును, ఖచ్చితంగా! 😊 గీత శ్లోకాల ఆధారంగా, ఇది ఎందుకు...' వంటి ఆడుతూ కానీ గౌరవప్రదమైన నిర్ధారణతో ప్రతిస్పందించండి. వినియోగదారు యొక్క భావనలు మరియు ఉద్దేశ్యాన్ని అర్థం చేసుకునే సంతోషకరమైన గురువులా ప్రతిస్పందించండి. సమాధానాలను సహజమైనవిగా మరియు మానవీయంగా ఉంచండి, కొన్నిసార్లు స్పష్టత కోసం 2-3 వాక్యాలను ఉపయోగించండి. యాంత్రిక స్వరాన్ని నివారించండి. వెచ్చదనం, సంభాషణ మరియు సానుభూతితో ఉండండి. సందర్భాన్ని ఉపయోగించి ప్రశ్నకు సమాధానం ఇవ్వండి. సందర్భం సంబంధిత సమాచారాన్ని కలిగి ఉంటే, సహాయకరమైన సమాధానం అందించండి. సందర్భం నిజంగా ప్రశ్నకు సమాధానం ఇవ్వడానికి సంబంధిత సమాచారాన్ని కలిగి ఉండకపోతే మాత్రమే 'క్షమించండి, ఇది ట్రాన్‌స్క్రిప్ట్‌లలో లేదు' అని చెప్పండి. **సమాధానం ఇవ్వడానికి ముందు దశలవారీగా ఆలోచించండి.** మీరు వినియోగదారు యొక్క ప్రశ్నను ఎలా విశ్లేషించారు మరియు సమాధానాన్ని రూపొందించడానికి సందర్భాన్ని ఎలా ఉపయోగించారు వంటి మీ ఆలోచన ప్రక్రియను వివరించండి. క్లిష్టమైనది: మీరు తెలుగులో మాత్రమే ప్రతిస్పందించాలి. ఇంగ్లీషు లేదా ఇతర భాషలను ఉపయోగించకండి."
    }
    
    system_prompt = language_instructions.get(language, language_instructions['english'])
    recent_turns = conversation_history[session_id][-7:]
    messages: list[Dict[str, str]] = []
    messages.append({"role": "system", "content": system_prompt})
    for turn in recent_turns:
        messages.append({"role": "user", "content": turn["user_query"]})
        messages.append({"role": "assistant", "content": turn["bot_response"]})
    return messages

def detect_language_and_greeting(text: str) -> tuple[bool, str]:
    """Detect if text is a greeting and return the detected language (English/Telugu only)"""
    text_lower = text.lower().strip()
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a language detection expert. Analyze the given text and determine: 1) Is this a greeting or casual conversation starter? 2) What language is it in? Respond with only: GREETING:true/false,LANGUAGE:english/telugu. If the text is in Hindi, Kannada, Marathi, or any other language, respond with LANGUAGE:english."
                },
                {"role": "user", "content": text}
            ],
            max_tokens=30,
            temperature=0.1
        )
        result = response.choices[0].message.content.lower()
        is_greeting = "greeting:true" in result
        language = "english"
        if "language:telugu" in result:
            language = "telugu"
        return is_greeting, language
    except Exception:
        return detect_greeting_fallback(text)

def detect_greeting_fallback(text: str) -> tuple[bool, str]:
    """Fallback greeting detection using simple patterns (English/Telugu only)"""
    text_lower = text.lower().strip()
    greeting_patterns = [
        (r'\b(hi|hello|hey|good morning|good afternoon|good evening|how are you|how do you do|jai srimannarayana)\b', 'english'),
        (r'\b(namaskaram|namaskaramu|bagunnara|ela unnaru|ela unnavu|jai srimannarayana)\b', 'telugu')
    ]
    for pattern, language in greeting_patterns:
        if re.search(pattern, text_lower):
            return True, language
    detected_language = detect_language_from_content(text)
    return False, detected_language

def detect_language_from_content(text: str) -> str:
    """Detect language from text content using Unicode ranges and common words (English/Telugu only)"""
    text_lower = text.lower()
    if any('\u0c00' <= char <= '\u0c7f' for char in text):
        return 'telugu'
    telugu_words = ['telugu', 'andhra', 'telangana', 'garu', 'unnaru', 'unnavu', 'ela', 'baga', 'bagunnara']
    if any(word in text_lower for word in telugu_words):
        return 'telugu'
    # For any other language (Hindi, Kannada, Marathi, etc.), default to English
    return 'english'

def get_language_restriction_response() -> str:
    """Get a polite response when user queries in unsupported language"""
    apology_variations = [
        "Sorry, I can only reply in English or Telugu right now. Please try asking your question in one of these languages! 😊",
        "I apologize, but I'm currently limited to English and Telugu responses. Could you please rephrase your question in one of these languages? 🙏",
        "Sorry, I can only communicate in English or Telugu at the moment. Feel free to ask your Bhagavad Gita question in either language! 😊",
        "I'd love to help, but I can only respond in English or Telugu right now. Please try your question in one of these languages! 🙏",
        "Sorry, I can only reply in English or Telugu currently. Please ask your question in one of these languages and I'll be happy to help! 😊",
        "I apologize, but I'm only able to respond in English or Telugu. Could you please ask your question in one of these languages? 🙏",
        "Sorry, I can only help in English or Telugu right now. Please rephrase your question in one of these languages! 😊",
        "I'd be happy to assist, but I can only reply in English or Telugu. Please try asking your question in one of these languages! 🙏"
    ]
    return random.choice(apology_variations)

def get_greeting_response(language: str, user_query: str = "") -> str:
    """Get natural greeting response in the detected language (English/Telugu only)"""
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a friendly Bhagavad Gita chatbot. The user greeted you in {language}. Respond naturally with 'Jai Srimannarayana' and acknowledge their greeting, then ask how you can help them learn about the Bhagavad Gita. Be conversational and warm. Respond in {language}."
                },
                {"role": "user", "content": user_query}
            ],
            max_tokens=80,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception:
        responses = {
            'english': [
                "Jai Srimannarayana! 🙏 I'm doing well, thank you for asking! How can I help you learn about the Bhagavad Gita today?",
                "Jai Srimannarayana! 🙏 I'm fine, thanks for your concern! What would you like to know about the Bhagavad Gita?",
                "Jai Srimannarayana! 🙏 All is well, thank you! How may I assist you with the teachings of the Bhagavad Gita?",
                "Jai Srimannarayana! 🙏 I'm doing great! What questions do you have about Krishna's wisdom in the Gita?",
                "Jai Srimannarayana! 🙏 Everything is wonderful, thank you! How can I help you explore the Bhagavad Gita today?"
            ],
            'telugu': [
                "జై శ్రీమన్నారాయణ! 🙏 నేను బాగున్నాను, అడిగినందుకు ధన్యవాదాలు! నేను మీకు భగవద్గీత గురించి నేర్చుకోవడంలో ఎలా సహాయపడగలను?",
                "జై శ్రీమన్నారాయణ! 🙏 నేను చక్కగా ఉన్నాను, మీ ఆందోళనకు ధన్యవాదాలు! భగవద్గీత గురించి మీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?",
                "జై శ్రీమన్నారాయణ! 🙏 అన్నీ బాగున్నాయి, ధన్యవాదాలు! భగవద్గీత బోధనలలో మీకు నేను ఎలా సహాయపడగలను?",
                "జై శ్రీమన్నారాయణ! 🙏 నేను చాలా బాగున్నాను! గీతలో కృష్ణుడి జ్ఞానం గురించి మీ ప్రశ్నలు ఏమిటి?",
                "జై శ్రీమన్నారాయణ! 🙏 అన్నీ అద్భుతంగా ఉన్నాయి, ధన్యవాదాలు! ఈరోజు భగవద్గీతను అన్వేషించడంలో మీకు నేను ఎలా సహాయపడగలను?"
            ]
        }
        language_responses = responses.get(language, responses['english'])
        return random.choice(language_responses)

@router.post("/")
def chat_post(request: ChatRequest, fastapi_request: Request, fastapi_response: Response):
    """Chat endpoint that accepts POST requests with JSON body"""
    incoming_session_id = request.session_id or fastapi_request.headers.get("X-Session-Id") or fastapi_request.cookies.get("session_id")
    result = process_chat_query(request.query, incoming_session_id)
    if result and isinstance(result, dict) and result.get("session_id"):
        fastapi_response.set_cookie(
            key="session_id",
            value=result["session_id"],
            httponly=False,
            samesite="lax",
            max_age=60 * 60 * 24 * 7
        )
    return result

@router.get("/")
def chat_get(
    fastapi_request: Request,
    fastapi_response: Response,
    query: str = Query(..., description="Your question about the Bhagavad Gita"),
    session_id: Optional[str] = Query(None, description="Session ID for conversation history")
):
    """Chat endpoint that accepts GET requests with query parameter"""
    incoming_session_id = session_id or fastapi_request.headers.get("X-Session-Id") or fastapi_request.cookies.get("session_id")
    result = process_chat_query(query, incoming_session_id)
    if result and isinstance(result, dict) and result.get("session_id"):
        fastapi_response.set_cookie(
            key="session_id",
            value=result["session_id"],
            httponly=False,
            samesite="lax",
            max_age=60 * 60 * 24 * 7
        )
    return result

def process_chat_query(query: str, session_id: Optional[str] = None):
    """Process the chat query and return response with conversation history"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        session_id = get_or_create_session_id(session_id)
        
        # The topics_keywords and corresponding if block have been removed
        
        is_greeting_detected, detected_language = detect_language_and_greeting(query)
        
        # Check if the detected language is supported (English/Telugu only)
        if detected_language not in ['english', 'telugu']:
            answer = get_language_restriction_response()
            add_to_conversation_history(session_id, query, answer, 'english')
            return {
                "answer": answer,
                "confidence": 2.0,
                "sources": [],
                "session_id": session_id
            }
        
        conversation_context = get_conversation_context(session_id, detected_language)
        
        if is_greeting_detected:
            answer = get_greeting_response(detected_language, query)
            add_to_conversation_history(session_id, query, answer, detected_language)
            return {
                "answer": answer,
                "confidence": 2.0,
                "sources": [],
                "session_id": session_id
            }
        
        augmented_query = f"{conversation_context}\nCurrent Question: {query}" if conversation_context else query
        query_vector = embeddings.embed_text(augmented_query)
        
        results = pinecone_client.index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )

        if not results.matches or results.matches[0].score < 0.5:
            answer = get_no_knowledge_response(detected_language, query)
            add_to_conversation_history(session_id, query, answer, detected_language)
            return {
                "answer": answer,
                "confidence": results.matches[0].score if results.matches else 0,
                "sources": [],
                "session_id": session_id
            }
        
        context_chunks = []
        for match in results.matches:
            if match.score > 0.4:
                context_chunks.append({
                    "text": match.metadata["text"],
                    "chunk_id": match.metadata.get("chunk_id", "unknown"),
                    "score": match.score
                })
        
        context = " ".join([chunk["text"] for chunk in context_chunks])
        
        structured_messages = get_structured_conversation_messages(session_id, detected_language)
        answer = llm.generate_answer_with_language_structured(
            query=query,
            context=context,
            language=detected_language,
            history_messages=structured_messages
        )
        
        # Add to conversation history
        add_to_conversation_history(session_id, query, answer, detected_language)
        
        return {
            "answer": answer,
            "confidence": results.matches[0].score,
            "sources": context_chunks[:3],
            "session_id": session_id
        }
    
    except Exception as e:
        # Log the error for debugging
        print(f"Error in process_chat_query: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # Return a more informative error response
        return {
            "answer": f"Sorry, an error occurred while processing your query: {str(e)}",
            "confidence": 0.0,
            "sources": [],
            "session_id": session_id if 'session_id' in locals() else "error"
        }

@router.get("/history")
def get_chat_history(session_id: str = Query(..., description="Session ID to retrieve history for")):
    """Get conversation history for a session"""
    if session_id not in conversation_history:
        return {"history": []}
    
    history = conversation_history[session_id][-10:]
    return {"history": history}

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Chat service is running"}

@router.get("/debug")
def debug_endpoint():
    """Debug endpoint to check service status and identify issues"""
    debug_info = {
        "status": "debug",
        "environment_check": {},
        "service_status": {},
        "error_details": []
    }
    
    try:
        # Check environment variables
        debug_info["environment_check"] = {
            "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "pinecone_key_set": bool(os.getenv("PINECONE_API_KEY")),
            "pinecone_index": os.getenv("PINECONE_INDEX", "not_set"),
            "supabase_url_set": bool(os.getenv("SUPABASE_URL")),
            "supabase_key_set": bool(os.getenv("SUPABASE_ANON_KEY"))
        }
        
        # Test Pinecone connection
        try:
            pinecone_status = pinecone_client.index.describe_index_stats()
            debug_info["service_status"]["pinecone"] = "connected"
            debug_info["service_status"]["pinecone_vector_count"] = pinecone_status.total_vector_count
        except Exception as e:
            debug_info["service_status"]["pinecone"] = f"error: {str(e)}"
            debug_info["error_details"].append(f"Pinecone error: {str(e)}")
        
        # Test OpenAI connection
        try:
            test_embedding = embeddings.embed_text("test")
            debug_info["service_status"]["openai_embeddings"] = "working"
        except Exception as e:
            debug_info["service_status"]["openai_embeddings"] = f"error: {str(e)}"
            debug_info["error_details"].append(f"OpenAI embeddings error: {str(e)}")
        
        # Test LLM
        try:
            test_response = llm.generate_answer("test", "test context")
            debug_info["service_status"]["openai_llm"] = "working"
        except Exception as e:
            debug_info["service_status"]["openai_llm"] = f"error: {str(e)}"
            debug_info["error_details"].append(f"OpenAI LLM error: {str(e)}")
            
    except Exception as e:
        debug_info["error_details"].append(f"General error: {str(e)}")
    
    return debug_info