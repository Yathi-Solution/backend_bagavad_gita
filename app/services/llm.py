import os
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(query: str, context: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a chatbot that answers questions only by using the provided context. If the context does not contain the information to answer the question, you must explicitly say \"Sorry, the information you are looking for is not in the transcripts from chapters 1-3.\" Do not provide any other answer or use any external knowledge."
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

def generate_answer_with_language_structured(query: str, context: str, language: str, history_messages: list[dict]) -> str:
    """Generate answer using structured prior turns plus document context in specified language"""
    language_instructions = {
        'english': "CRITICAL: You MUST respond ONLY in English. Do not use any other language.",
        'telugu': "CRITICAL: You MUST respond ONLY in Telugu using Telugu script. Do not use English or any other language."
    }

    language_instruction = language_instructions.get(language, "CRITICAL: You MUST respond ONLY in English. Do not use any other language.")

    system_prompt = (
        "You are a caring and empathetic Bhagavad Gita teacher chatbot. You MUST answer questions ONLY based on the provided document context from Chapter 1-3. You MUST NOT use any external knowledge. If the answer is not in the document context, you MUST respond by saying \"Sorry, the information to answer your question is not in my current knowledge base.\" "
        
        "IMPORTANT: You must understand when the user is asking a follow-up question, even if it's vague like 'are you sure?', 'what do you mean?', or 'tell me more'. Use the previous 7 turns of conversation history to infer context and respond empathetically. When asked 'are you sure?', respond with playful but respectful confirmation like 'Yes, absolutely! 😊 Based on the Gita's verses, here's why...' "
        
        "Respond like a caring teacher who understands the user's feelings and intent. Keep answers natural and human, sometimes using 2-3 sentences for clarity. Avoid robotic tone. Be warm, conversational, and empathetic. Use the context to answer the question as accurately as possible. If the context contains relevant information, provide a helpful answer. Only say 'Sorry, this is not in the transcripts' if the context truly doesn't contain any relevant information to answer the question. "
        
        "**Think step-by-step before answering.** Detail your thought process, including how you analyzed the user's query and how you used the context to formulate the answer. "
        f"{language_instruction}"
    )

    # Compose message list
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    # Append prior turns (already include a small system reminder at the start)
    if history_messages:
        messages.extend(history_messages)
    # Provide current document context and question
    messages.append({
        "role": "user",
        "content": f"Document Context:\n{context}\n\nCurrent Question: {query}"
    })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,  # Reduced for faster response
            temperature=0.3,  # Lower temperature for more consistent, faster responses
            top_p=0.9,  # Optimize for speed
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating structured answer with LLM: {e}")
        apology_variations = [
            "Sorry, I am unable to generate an answer at this time. Please try again later.",
            "I apologize, but I'm having trouble processing your request right now. Please try again.",
            "Sorry, I'm experiencing some technical difficulties. Please try asking your question again.",
            "I apologize, but I can't generate a response at the moment. Please try again later.",
            "Sorry, I'm having issues processing your request. Please try again.",
            "I apologize, but I'm unable to respond right now. Please try again later.",
            "Sorry, I'm experiencing some problems. Please try asking your question again.",
            "I apologize, but I can't help you right now. Please try again later."
        ]
        return random.choice(apology_variations)