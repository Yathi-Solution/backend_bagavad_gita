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
                "content": "You are a helpful assistant answering questions about the Bhagavad Gita based on the provided context. Use the context to answer the question as accurately as possible. If the context contains relevant information, provide a helpful answer. Only say 'Sorry, this is not in the transcripts' if the context truly doesn't contain any relevant information to answer the question."
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content
