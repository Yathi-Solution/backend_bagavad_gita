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
        
        # Combine a generic apology with the LLM-generated suggestions
        apology_responses = {
            'english': "I apologize, but I couldn't find relevant information about this specific topic. However, I can help you with these related subjects:",
            'hindi': "माफ़ करें, मुझे इस विशिष्ट विषय के बारे में प्रासंगिक जानकारी नहीं मिली। हालांकि, मैं आपको इन संबंधित विषयों में मदद कर सकता हूं:",
            'telugu': "క్షమించండి, ఈ నిర్దిష్ట అంశం గురించి సంబంధిత సమాచారం నాకు దొరకలేదు. అయితే, ఈ సంబంధిత విషయాలతో నేను మీకు సహాయపడగలను:",
            'kannada': "ಕ್ಷಮಿಸಿ, ಈ ನಿರ್ದಿಷ್ಟ ವಿಷಯದ ಬಗ್ಗೆ ನನಗೆ ಸಂಬಂಧಿತ ಮಾಹಿತಿ ಸಿಗಲಿಲ್ಲ. అయితే, ಈ ಸಂಬಂಧಿತ ವಿಷಯಗಳೊಂದಿಗೆ ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಬಹುದು:",
            'marathi': "माफ करा, मला या विशिष्ट विषयाबद्दल संबंधित माहिती सापडली नाही. तरीही, मी तुम्हाला या संबंधित विषयांमध्ये मदत करू शकेन:"
        }
        apology = apology_responses.get(language, apology_responses['english'])
        suggestions = response.choices[0].message.content.strip()
        
        return f"{apology}\n\n{suggestions}"
    
    except Exception:
        # Fallback to a simple static response if LLM call fails
        no_knowledge_responses = {
            'english': "Sorry, I couldn't find the information you're looking for. Please try asking about a different topic from the Bhagavad Gita.",
            'hindi': "क्षमा करें, मुझे वह जानकारी नहीं मिली जो आप ढूंढ रहे हैं। कृपया भगवद गीता के किसी अन्य विषय के बारे में पूछने का प्रयास करें।",
            'telugu': "క్షమించండి, మీరు వెతుకుతున్న సమాచారం నాకు దొరకలేదు. దయచేసి భగవద్గీతలోని వేరే అంశం గురించి అడగడానికి ప్రయత్నించండి.",
            'kannada': "ಕ್ಷಮಿಸಿ, ನೀವು ಹುಡುಕುತ್ತಿರುವ ಮಾಹಿತಿ ನನಗೆ ಸಿಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಭಗವದ್ಗೀತೆಯ ಬೇರೆ ವಿಷಯದ ಬಗ್ಗೆ ಕೇಳಲು ಪ್ರಯತ್ನಿಸಿ.",
            'marathi': "माफ करा, मला तुम्ही शोधत असलेली माहिती सापडली नाही. कृपया भगवद्गीतेच्या दुसऱ्या विषयाबद्दल विचारण्याचा प्रयत्न करा."
        }
        return no_knowledge_responses.get(language, no_knowledge_responses['english'])

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
    
    recent_turns = conversation_history[session_id][-4:]
    
    context_parts = []
    for turn in recent_turns:
        context_parts.append(f"Previous Q: {turn['user_query']}")
        context_parts.append(f"Previous A: {turn['bot_response']}")
    
    context = "\n".join(context_parts)
    
    language_instructions = {
        'english': "Consider the previous conversation context when answering. Reference previous questions and answers when relevant.",
        'hindi': "पिछली बातचीत के संदर्भ को ध्यान में रखते हुए उत्तर दें। जब प्रासंगिक हो तो पिछले प्रश्नों और उत्तरों का संदर्भ दें।",
        'telugu': "మునుపటి సంభాషణ సందర్భాన్ని పరిగణనలోకి తీసుకొని జవాబు ఇవ్వండి. సంబంధితమైనప్పుడు మునుపటి ప్రశ్నలు మరియు జవాబులను సూచించండి.",
        'kannada': "ಹಿಂದಿನ ಸಂಭಾಷಣೆಯ ಸಂದರ್ಭವನ್ನು ಪರಿಗಣಿಸಿ ಉತ್ತರಿಸಿ. ಸಂಬಂಧಿತವಾದಾಗ ಹಿಂದಿನ ಪ್ರಶ್ನೆಗಳು ಮತ್ತು ಉತ್ತರಗಳನ್ನು ಉಲ್ಲೇಖಿಸಿ.",
        'marathi': "मागील संभाषणाचा संदर्भ लक्षात घेऊन उत्तर द्या. प्रासंगिक असल्यास मागील प्रश्न आणि उत्तरांचा संदर्भ द्या."
    }
    
    instruction = language_instructions.get(language, language_instructions['english'])
    return f"{instruction}\n\nPrevious conversation:\n{context}\n\n"

def get_structured_conversation_messages(session_id: str, language: str) -> list[Dict[str, str]]:
    """Return structured chat messages from recent history for LLM consumption"""
    if session_id not in conversation_history or not conversation_history[session_id]:
        return []

    language_instructions = {
        'english': "Consider the prior conversation turns when answering.",
        'hindi': "उत्तर देते समय पिछली बातचीत को ध्यान में रखें।",
        'telugu': "మునుపటి సంభాషణను పరిగణనలోకి తీసుకోండి.",
        'kannada': "ಹಿಂದಿನ ಸಂಭಾಷಣೆಯನ್ನು ಪರಿಗಣಿಸಿ.",
        'marathi': "उत्तर देताना मागील संभाषणाचा विचार करा."
    }
    system_note = language_instructions.get(language, language_instructions['english'])
    recent_turns = conversation_history[session_id][-4:]
    messages: list[Dict[str, str]] = []
    messages.append({"role": "system", "content": system_note})
    for turn in recent_turns:
        messages.append({"role": "user", "content": turn["user_query"]})
        messages.append({"role": "assistant", "content": turn["bot_response"]})
    return messages

def detect_language_and_greeting(text: str) -> tuple[bool, str]:
    """Detect if text is a greeting and return the detected language"""
    text_lower = text.lower().strip()
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a language detection expert. Analyze the given text and determine: 1) Is this a greeting or casual conversation starter? 2) What language is it in? Respond with only: GREETING:true/false,LANGUAGE:english/hindi/telugu/kannada/marathi"
                },
                {"role": "user", "content": text}
            ],
            max_tokens=30,
            temperature=0.1
        )
        result = response.choices[0].message.content.lower()
        is_greeting = "greeting:true" in result
        language = "english"
        if "language:hindi" in result:
            language = "hindi"
        elif "language:telugu" in result:
            language = "telugu"
        elif "language:kannada" in result:
            language = "kannada"
        elif "language:marathi" in result:
            language = "marathi"
        return is_greeting, language
    except Exception:
        return detect_greeting_fallback(text)

def detect_greeting_fallback(text: str) -> tuple[bool, str]:
    """Fallback greeting detection using simple patterns"""
    text_lower = text.lower().strip()
    greeting_patterns = [
        (r'\b(hi|hello|hey|good morning|good afternoon|good evening|how are you|how do you do|jai srimannarayana)\b', 'english'),
        (r'\b(namaste|namaskar|pranam|kaise ho|kaise hain|jai srimannarayana)\b', 'hindi'),
        (r'\b(namaskaram|namaskaramu|bagunnara|ela unnaru|ela unnavu|jai srimannarayana)\b', 'telugu'),
        (r'\b(namaskara|howdu|chennagideya|jai srimannarayana)\b', 'kannada'),
        (r'\b(namaskar|kasa ahes|kasa aahes|jai srimannarayana)\b', 'marathi')
    ]
    for pattern, language in greeting_patterns:
        if re.search(pattern, text_lower):
            return True, language
    detected_language = detect_language_from_content(text)
    return False, detected_language

def detect_language_from_content(text: str) -> str:
    """Detect language from text content using Unicode ranges and common words"""
    text_lower = text.lower()
    if any('\u0c00' <= char <= '\u0c7f' for char in text):
        return 'telugu'
    if any('\u0900' <= char <= '\u097f' for char in text):
        marathi_words = ['ahes', 'aahes', 'kasa', 'kase', 'kasa ahes', 'marathi', 'maharashtra']
        if any(word in text_lower for word in marathi_words):
            return 'marathi'
        return 'hindi'
    if any('\u0c80' <= char <= '\u0cff' for char in text):
        return 'kannada'
    telugu_words = ['telugu', 'andhra', 'telangana', 'garu', 'unnaru', 'unnavu', 'ela', 'baga', 'bagunnara']
    if any(word in text_lower for word in telugu_words):
        return 'telugu'
    kannada_words = ['kannada', 'karnataka', 'guru', 'deya', 'howdu', 'chennagi', 'chennagideya']
    if any(word in text_lower for word in kannada_words):
        return 'kannada'
    hindi_words = ['hindi', 'hain', 'ho', 'aap', 'tum', 'kaise', 'kya', 'hai', 'kya hai']
    if any(word in text_lower for word in hindi_words):
        return 'hindi'
    marathi_words = ['marathi', 'maharashtra', 'ahes', 'aahes', 'kasa', 'kase', 'kasa ahes']
    if any(word in text_lower for word in marathi_words):
        return 'marathi'
    return 'english'

def get_greeting_response(language: str, user_query: str = "") -> str:
    """Get natural greeting response in the detected language"""
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
            'hindi': [
                "जय श्रीमन्नारायण! 🙏 मैं ठीक हूं, पूछने के लिए धन्यवाद! मैं आपको भगवद गीता के बारे में सीखने में कैसे मदद कर सकता हूं?",
                "जय श्रीमन्नारायण! 🙏 मैं अच्छा हूं, आपकी चिंता के लिए धन्यवाद! भगवद गीता के बारे में आप क्या जानना चाहते हैं?",
                "जय श्रीमन्नारायण! 🙏 सब कुछ ठीक है, धन्यवाद! मैं आपकी भगवद गीता की शिक्षाओं में कैसे सहायता कर सकता हूं?",
                "जय श्रीमन्नारायण! 🙏 मैं बहुत अच्छा हूं! गीता में कृष्ण की बुद्धिमत्ता के बारे में आपके क्या प्रश्न हैं?",
                "जय श्रीमन्नारायण! 🙏 सब कुछ शानदार है, धन्यवाद! आज भगवद गीता का अन्वेषण करने में मैं आपकी कैसे मदद कर सकता हूं?"
            ],
            'telugu': [
                "జై శ్రీమన్నారాయణ! 🙏 నేను బాగున్నాను, అడిగినందుకు ధన్యవాదాలు! నేను మీకు భగవద్గీత గురించి నేర్చుకోవడంలో ఎలా సహాయపడగలను?",
                "జై శ్రీమన్నారాయణ! 🙏 నేను చక్కగా ఉన్నాను, మీ ఆందోళనకు ధన్యవాదాలు! భగవద్గీత గురించి మీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?",
                "జై శ్రీమన్నారాయణ! 🙏 అన్నీ బాగున్నాయి, ధన్యవాదాలు! భగవద్గీత బోధనలలో మీకు నేను ఎలా సహాయపడగలను?",
                "జై శ్రీమన్నారాయణ! 🙏 నేను చాలా బాగున్నాను! గీతలో కృష్ణుడి జ్ఞానం గురించి మీ ప్రశ్నలు ఏమిటి?",
                "జై శ్రీమన్నారాయణ! 🙏 అన్నీ అద్భుతంగా ఉన్నాయి, ధన్యవాదాలు! ఈరోజు భగవద్గీతను అన్వేషించడంలో మీకు నేను ఎలా సహాయపడగలను?"
            ],
            'kannada': [
                "ಜೈ ಶ್ರೀಮನ್ನಾರಾಯಣ! 🙏 ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ, ಕೇಳಿದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು! ನಾನು ನಿಮಗೆ ಭಗವದ್ಗೀತೆಯ ಬಗ್ಗೆ ಕಲಿಯಲು ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
                "ಜೈ ಶ್ರೀಮನ್ನಾರಾಯಣ! 🙏 ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ, ನಿಮ್ಮ ಕಾಳಜಿಗೆ ಧನ್ಯವಾದಗಳು! ಭಗವದ್ಗೀತೆಯ ಬಗ್ಗೆ ನೀವು ಏನು ತಿಳಿದುಕೊಳ್ಳಲು ಬಯಸುತ್ತೀರಿ?",
                "ಜೈ ಶ್ರೀಮನ್ನಾರಾಯಣ! 🙏 ಎಲ್ಲವೂ ಚೆನ್ನಾಗಿದೆ, ಧನ್ಯವಾದಗಳು! ಭಗವದ್ಗೀತೆಯ ಬೋಧನೆಗಳಲ್ಲಿ ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
                "ಜೈ ಶ್ರೀಮನ್ನಾರಾಯಣ! 🙏 ನಾನು ತುಂಬಾ ಚೆನ್ನಾಗಿದ್ದೇನೆ! ಗೀತೆಯಲ್ಲಿ ಕೃಷ್ಣನ ಬುದ್ಧಿವಂತಿಕೆಯ ಬಗ್ಗೆ ನಿಮ್ಮ ಪ್ರಶ್ನೆಗಳು ಯಾವುವು?",
                "ಜೈ ಶ್ರೀಮನ್ನಾರಾಯಣ! 🙏 ಎಲ್ಲವೂ ಅದ್ಭುತವಾಗಿದೆ, ಧನ್ಯವಾದಗಳು! ಇಂದು ಭಗವದ್ಗೀತೆಯನ್ನು ಅನ್ವೇಷಿಸಲು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?"
            ],
            'marathi': [
                "जय श्रीमन्नारायण! 🙏 मी ठीक आहे, विचारल्याबद्दल धन्यवाद! मी तुम्हाला भगवद्गीता बद्दल शिकण्यात कशी मदत करू शकतो?",
                "जय श्रीमन्नारायण! 🙏 मी चांगले आहे, तुमच्या काळजीबद्दल धन्यवाद! भगवद्गीता बद्दल तुम्हाला काय जाणून घ्यायचे आहे?",
                "जय श्रीमन्नारायण! 🙏 सर्व काही ठीक आहे, धन्यवाद! भगवद्गीतेच्या शिकवणींमध्ये मी तुम्हाला कशी मदत करू शकतो?",
                "जय श्रीमन्नारायण! 🙏 मी खूप चांगले आहे! गीतेमध्ये कृष्णाच्या बुद्धिमत्तेबद्दल तुमचे प्रश्न काय आहेत?",
                "जय श्रीमन्नारायण! 🙏 सर्व काही अद्भुत आहे, धन्यवाद! आज भगवद्गीतेचा शोध घेण्यात मी तुम्हाला कशी मदत करू शकतो?"
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
        raw_response = llm.generate_answer_with_language_structured(
            query=query,
            context=context,
            language=detected_language,
            history_messages=structured_messages
        )
        
        try:
            llm_response_data = json.loads(raw_response)
            thought = llm_response_data.get("thought", "No thought process available.")
            answer = llm_response_data.get("answer", "No answer could be generated.")
            
            add_to_conversation_history(session_id, query, answer, detected_language)
            
            return {
                "thought": thought,
                "answer": answer,
                "confidence": results.matches[0].score,
                "sources": context_chunks[:3],
                "session_id": session_id
            }
        
        except json.JSONDecodeError:
            add_to_conversation_history(session_id, query, raw_response, detected_language)
            return {
                "answer": "Sorry, an internal error occurred. Please try again.",
                "confidence": 0.0,
                "sources": [],
                "session_id": session_id
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

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