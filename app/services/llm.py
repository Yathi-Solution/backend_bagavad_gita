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

IMPORTANT INSTRUCTIONS:
1. Always provide a complete, well-structured answer that covers all aspects of the question
2. Include specific details from the context (chapter names, verses, concepts)
3. Maintain consistency - similar questions should receive similar comprehensive answers
4. Structure your response with clear explanations and context
5. If the context contains relevant information, provide a detailed answer even if it's partial
6. Only say 'Sorry, I don't have information regarding this topic' if the context truly doesn't contain ANY relevant information

RESPONSE FORMAT:
- Start with a clear definition or explanation
- Provide specific details from the Bhagavad Gita context
- Include relevant background information
- Explain the significance or meaning
- Keep responses comprehensive and educational"""
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
