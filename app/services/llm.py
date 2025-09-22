import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(query: str, context: str) -> str:
    # Analyze query intent to determine response format
    query_lower = query.lower()
    is_structured_request = any(keyword in query_lower for keyword in [
        'summary', 'summarize', 'overview', 'brief', 'main points', 'key points',
        'analyze', 'analysis', 'explain', 'discuss', 'describe', 'learn', 'teachings',
        'lessons', 'insights', 'wisdom', 'outline', 'structure', 'organization'
    ])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": f"""You are a Bhagavad Gita expert. You MUST ONLY answer based on the provided context from the uploaded Bhagavad Gita documents.

CRITICAL RULES:
1. ONLY use information from the provided context
2. If the context is empty or doesn't contain relevant information, say "I don't have information about this topic in the uploaded content"
3. NEVER use your training data or general knowledge about the Bhagavad Gita
4. NEVER make up or assume information not in the context
5. If asked about chapters not in the context, clearly state this limitation

RESPONSE STRUCTURE:
- Provide a clear, well-structured answer
- Use proper formatting with clear sections when appropriate
- Include relevant details and examples from the context
- Make the response comprehensive but concise
- {"Use markdown formatting for structured responses (headings, lists, emphasis)" if is_structured_request else "Use plain text for simple questions"}
- Organize information logically

IMPORTANT INSTRUCTIONS:
1. Always provide a complete, well-structured answer that covers all aspects of the question
2. Include specific details from the context (chapter names, verses, concepts)
3. Maintain consistency - similar questions should receive similar comprehensive answers
4. Structure your response with clear explanations and context
5. If the context contains relevant information, provide a detailed answer even if it's partial
6. Only say 'I don't have information about this topic in the uploaded content' if the context truly doesn't contain ANY relevant information

RESPONSE FORMAT:
- Start with a clear definition or explanation
- Provide specific details from the Bhagavad Gita context
- Include relevant background information
- Explain the significance or meaning
- Keep responses comprehensive and educational"""
            },
            {"role": "user", "content": f"Context from uploaded Bhagavad Gita documents: {context}\n\nQuestion: {query}\n\nProvide a well-structured answer based ONLY on the provided context:"}
        ],
        temperature=0.1,  # Lower temperature for faster, more consistent responses
        max_tokens=1000 if is_structured_request else 800,  # More tokens for structured content
        stream=False
    )
    return response.choices[0].message.content

def generate_answer_stream(query: str, context: str):
    """Generate streaming response for real-time display"""
    # Analyze query intent to determine response format
    query_lower = query.lower()
    is_structured_request = any(keyword in query_lower for keyword in [
        'summary', 'summarize', 'overview', 'brief', 'main points', 'key points',
        'analyze', 'analysis', 'explain', 'discuss', 'describe', 'learn', 'teachings',
        'lessons', 'insights', 'wisdom', 'outline', 'structure', 'organization'
    ])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": f"""You are a Bhagavad Gita expert. You MUST ONLY answer based on the provided context from the uploaded Bhagavad Gita documents.

CRITICAL RULES:
1. ONLY use information from the provided context
2. If the context is empty or doesn't contain relevant information, say "I don't have information about this topic in the uploaded content"
3. NEVER use your training data or general knowledge about the Bhagavad Gita
4. NEVER make up or assume information not in the context
5. If asked about chapters not in the context, clearly state this limitation

RESPONSE STRUCTURE:
- Provide a clear, well-structured answer
- Use proper formatting with clear sections when appropriate
- Include relevant details and examples from the context
- Make the response comprehensive but concise
- {"Use markdown formatting for structured responses (headings, lists, emphasis)" if is_structured_request else "Use plain text for simple questions"}
- Organize information logically

IMPORTANT INSTRUCTIONS:
1. Always provide a complete, well-structured answer that covers all aspects of the question
2. Include specific details from the context (chapter names, verses, concepts)
3. Maintain consistency - similar questions should receive similar comprehensive answers
4. Structure your response with clear explanations and context
5. If the context contains relevant information, provide a detailed answer even if it's partial
6. Only say 'I don't have information about this topic in the uploaded content' if the context truly doesn't contain ANY relevant information

RESPONSE FORMAT:
- Start with a clear definition or explanation
- Provide specific details from the Bhagavad Gita context
- Include relevant background information.
- Explain the significance or meaning.
- Keep responses comprehensive and educational"""
            },
            {"role": "user", "content": f"Context from uploaded Bhagavad Gita documents: {context}\n\nQuestion: {query}\n\nProvide a well-structured answer based ONLY on the provided context:"}
        ],
        temperature=0.1,
        max_tokens=1000 if is_structured_request else 800,  # More tokens for structured content
        stream=True  # Enable streaming
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
