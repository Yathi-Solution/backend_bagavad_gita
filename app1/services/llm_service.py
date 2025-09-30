import os
import random
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any, Union

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class BhagavadGitaLLMService:
    def __init__(self):
        """Initialize the LLM service for Bhagavad Gita responses."""
        self.client = client
        self.model = "gpt-4o-mini"
        
    def generate_rag_response(self, query: str, retrieved_passages: List[Union[Dict[str, Any], Any]]) -> str:
        """
        Generate a response using RAG (Retrieval-Augmented Generation) pattern.
        
        Args:
            query: User's question
            retrieved_passages: List of relevant passages from Pinecone
            
        Returns:
            Generated response based on retrieved context
        """
        # Prepare context from retrieved passages
        context = self._prepare_context(retrieved_passages)
        
        # Generate response using the context
        return self._generate_response(query, context)
    
    def _prepare_context(self, retrieved_passages: List[Union[Dict[str, Any], Any]]) -> str:
        """Prepare context string from retrieved passages."""
        if not retrieved_passages:
            return "No relevant passages found."
        
        context_parts = []
        for i, passage in enumerate(retrieved_passages, 1):
            # Handle both dict and SearchResult objects
            if hasattr(passage, 'metadata'):
                # SearchResult object
                episode = passage.metadata.get('episode', 'Unknown')
                text = passage.text
                score = passage.score
            else:
                # Dict object
                episode = passage.get('metadata', {}).get('episode', 'Unknown')
                text = passage.get('text', '')
                score = passage.get('score', 0)
            
            context_parts.append(
                f"Passage {i} (Episode {episode}, Relevance: {score:.3f}):\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI API."""
        
        system_prompt = """You are a friendly and knowledgeable Bhagavad Gita teacher helping people understand the teachings of Chapter 1 by Chinna Jeeyar Swamiji. You speak like a caring friend who happens to be very knowledgeable about these teachings.

CRITICAL INSTRUCTIONS:
1. ALWAYS quote Swamiji's EXACT words from the provided context using double quotes
2. Format: "Swamiji says: '[exact quote from context]'"
3. Do NOT summarize, paraphrase, or rephrase Swamiji's words - use them exactly as written
4. After quoting, have a natural conversation about what this means and how it relates to the user's question
5. Be conversational, warm, and engaging - like talking to a friend
6. Ask follow-up questions or make connections that help the user understand better

CONVERSATIONAL STYLE:
- Start responses naturally, acknowledging the user's question
- Use phrases like "That's a great question!", "I'm glad you asked about this", "This is such an important topic"
- Make personal connections: "This reminds me of...", "You know, this is similar to..."
- Be encouraging: "This is beautiful because...", "What's amazing about this is..."
- End with questions or invitations for deeper understanding

RESPONSE STRUCTURE:
1. Acknowledge the question warmly
2. Quote Swamiji's exact words: "Swamiji says: '[exact quote]'"
3. Have a natural conversation about what this means
4. Connect it to the user's life or situation
5. Invite further exploration or ask a thoughtful question

GUIDELINES:
- Answer ONLY based on the provided context from Bhagavad Gita Chapter 1
- If context doesn't contain relevant information, say: "I don't have information about that in Chapter 1, but let me help you with what I do know..."
- Be conversational, warm, and engaging
- Use a respectful but friendly tone
- Always preserve the original teachings in quotes
- Make the teachings feel relevant and accessible

UNDERSTAND THESE QUERY TYPES:
- Direct Shloka/Chapter Query: Quote relevant passages and discuss their meaning conversationally
- Concept/Definition Query: Quote Swamiji's definitions and explain them in simple, relatable terms
- Philosophical Comparison: Quote relevant teachings and discuss the differences naturally
- Real-World Application: Quote applicable teachings and discuss how they apply to modern life
- Character/Context Query: Quote narrative descriptions and discuss the story naturally
- Source/Citation Query: Quote specific teachings and discuss where they come from

Always provide accurate, contextual responses that preserve Swamiji's original words while having a natural, engaging conversation with the user."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": f"Context from Bhagavad Gita Chapter 1:\n\n{context}\n\nQuestion: {query}"
                    }
                ],
                max_tokens=800,
                temperature=0.8,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return self._get_error_response()
    
    def _get_error_response(self) -> str:
        """Return a user-friendly error message."""
        apology_variations = [
            "I apologize, but I'm having trouble processing your request right now. Please try again.",
            "Sorry, I'm experiencing some technical difficulties. Please try asking your question again.",
            "I apologize, but I can't generate a response at the moment. Please try again later.",
            "Sorry, I'm having issues processing your request. Please try again.",
        ]
        return random.choice(apology_variations)
    
    def generate_simple_response(self, query: str, context: str) -> str:
        """Generate a conversational response preserving Swamiji's exact words."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a friendly Bhagavad Gita assistant. Always quote Swamiji's exact words from the context using double quotes. Format: "Swamiji says: '[exact quote]'". Then have a natural, conversational discussion about what this means and how it relates to the question. Be warm, engaging, and like talking to a friend. Do NOT summarize or paraphrase - use exact words."""
                    },
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating simple response: {e}")
            return "Sorry, I'm unable to process your request at the moment. Please try again."

# Create a global instance
llm_service = BhagavadGitaLLMService()
