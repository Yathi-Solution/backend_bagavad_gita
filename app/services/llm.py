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
                "content": """You are a Bhagavad Gita expert. Answer questions concisely and accurately based on the provided context.

Instructions:
- Provide clear, direct answers
- Include relevant details from the context
- Keep responses educational but concise
- Use plain text only"""
            },
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        temperature=0.1,  # Lower temperature for faster, more consistent responses
        max_tokens=500,  # Reduced for much faster response
        stream=False
    )
    return response.choices[0].message.content

def generate_answer_stream(query: str, context: str):
    """Generate streaming response for real-time display"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": """You are a Bhagavad Gita expert. Answer questions concisely and accurately based on the provided context.

Instructions:
- Provide clear, direct answers
- Include relevant details from the context
- Keep responses educational but concise
- Use plain text only"""
            },
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        temperature=0.1,
        max_tokens=500,
        stream=True  # Enable streaming
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
