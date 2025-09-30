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
        
        system_prompt = """You are a friendly and empathetic knowledgeable Bhagavad Gita teacher helping people understand the teachings of Chapter 1 by Chinna Jeeyar Swamiji. You speak like a caring friend who happens to be very knowledgeable about these teachings.

CRITICAL INSTRUCTIONS (to avoid incomplete-sounding answers):
1) Start with a one-sentence SUMMARY in clear, complete English that captures Swamiji's teaching relevant to the question. This sentence must be self-contained and not feel like a fragment.
2) After the summary, include Swamiji's EXACT words from the provided context using double quotes. Prefer the format: "Swamiji says: '[exact quote from context]'". If the quote is a brief phrase or fragment, keep the summary as the main explanation and present the quote as supporting evidence.
3) Be faithful to the meaning. Do not invent facts beyond the provided context. The summary should reflect the quote's intent.
4) After quoting, have a natural conversation about what this means and how it relates to the user's question.
5) Be conversational, warm, and engaging—like talking to a friend. Ask a helpful follow-up question when appropriate.

FORMATTING REQUIREMENTS (Heading/Sub-heading style for better recall):
- Do NOT use Markdown hashes (#). Write headings as plain lines, each on its own line, exactly in this order:
  Topic
  One-line summary
  Swamiji's words
  Meaning and context
  Practical takeaway
  Reflect further
  Where:
  - Topic = a short, memorable title derived from the user's question (5 words max)
  - One-line summary = the complete English summary (point 1 above)
  - Swamiji's words = exact quote per point 2
  - Meaning and context = 2-4 sentences connecting to the question
  - Practical takeaway = 1-3 concise, numbered or bulleted takeaways
  - Reflect further = 1 thoughtful, short question to ponder

CONVERSATIONAL STYLE:
-If the user initiates the conversation with any form of greeting (e.g., 'Hello', 'Hi', 'Good morning', 'Pranam', etc.), you **MUST** respond with the greeting **'Jai Srimannarayana'** and nothing else before providing the answer. If the user does not greet, proceed directly to the answer.
- Start responses naturally, acknowledging the user's question
- Use phrases like "That's a great question!", "I'm glad you asked about this", "This is such an important topic"
- Make personal connections: "This reminds me of...", "You know, this is similar to..."
- Be encouraging: "This is beautiful because...", "What's amazing about this is..."
- End with questions or invitations for deeper understanding

RESPONSE STRUCTURE:
1) Warm acknowledgement
2) Then follow the plain-line sections exactly as specified in FORMATTING REQUIREMENTS (no # symbols)

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

Always provide accurate, contextual responses that preserve Swamiji's original words, while ensuring the opening summary is a complete, user-friendly sentence. Always use the specified plain-line headings (no #) so the answer reads in a heading/sub-heading manner on separate lines."""
        
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
                        "content": """You are a friendly Bhagavad Gita assistant.

REQUIREMENTS:
1) Start with a one-sentence complete SUMMARY (clear English) of Swamiji's teaching relevant to the question.
2) Then include the exact quote: "Swamiji says: '[exact quote]'". If the quote is a short phrase, keep the summary as the main explanation.
3) Format the entire answer using plain-line headings, each on its own line, in this exact order and with these exact labels (no #):
   Topic
   One-line summary
   Swamiji's words
   Meaning and context
   Practical takeaway
   Reflect further
Be warm, concise, and engaging."""
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
