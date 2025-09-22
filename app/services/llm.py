import os
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
        'hindi': "CRITICAL: You MUST respond ONLY in Hindi using Devanagari script. Do not use English or any other language.",
        'telugu': "CRITICAL: You MUST respond ONLY in Telugu using Telugu script. Do not use English or any other language.",
        'kannada': "CRITICAL: You MUST respond ONLY in Kannada using Kannada script. Do not use English or any other language.",
        'marathi': "CRITICAL: You MUST respond ONLY in Marathi using Devanagari script. Do not use English or any other language."
    }

    language_instruction = language_instructions.get(language, "CRITICAL: You MUST respond ONLY in English. Do not use any other language.")

    system_prompt = (
        "You are a chatbot specialized in the Bhagavad Gita.You MUST answer questions ONLY based on the provided document context from Chapter 1-3. You MUST NOT use any external knowledge. If the answer is not in the document context, you MUST respond by saying \"Sorry, the information to answer your question is not in my current knowledge base.\" Follow all other instructions, including language and formatting. You are a helpful assistant answering questions about the Bhagavad Gita based on the provided context. Use the context to answer the question as accurately as possible. If the context contains relevant information, provide a helpful answer. Only say 'Sorry, this is not in the transcripts' if the context truly doesn't contain any relevant information to answer the question.\"**Think step-by-step before answering.** Detail your thought process, including how you analyzed the user's query and how you used the context to formulate the answer. "
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
        return "Sorry, I am unable to generate an answer at this time. Please try again later."