import os
import random
import uuid
from openai import OpenAI
from typing import List, Dict, Any, Union

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class BhagavadGitaLLMService:
    def __init__(self):
        """Initialize the LLM service for Bhagavad Gita responses."""
        self.client = client
        self.model = "gpt-4o-mini"
        # Import Supabase service here to avoid circular imports
        self.supabase_service = None
    
    async def classify_query_intent(self, query: str) -> str:
        """
        Classifies a user's query as either a 'GREETING', 'CHITCHAT', or a 'QUESTION'
        to determine the appropriate response path.
        """
        classification_prompt = (
            "You are an expert query classifier. Your only job is to categorize a user's "
            "message as one of three types: 'GREETING', 'CHITCHAT', or 'QUESTION'. "
            "\n- GREETING: A simple hello with no substance (e.g., 'Hello', 'Namaste', 'Good morning')."
            "\n- CHITCHAT: A simple, non-Gita related conversational question about the bot itself or its status (e.g., 'How are you?', 'What are you doing?', 'What is your name?')."
            "\n- QUESTION: A query asking for information, guidance, or philosophical content related to the Bhagavad Gita, Chapter 1."
            "\nRespond with ONLY one of the three words: 'GREETING', 'CHITCHAT', or 'QUESTION'. Do not add any other text."
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": classification_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=10,
            )
            return response.choices[0].message.content.strip().upper()
            
        except Exception as e:
            print(f"Error classifying query intent: {e}")
            return "QUESTION"
        
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
    
    async def generate_contextual_rag_response(self, query: str, retrieved_passages: List[Union[Dict[str, Any], Any]], 
                                            session_id: str = None, conversation_history: List[Dict[str, Any]] = None) -> str:
        """
        Generate a context-aware response using RAG pattern with conversation history.
        
        Args:
            query: User's question
            retrieved_passages: List of relevant passages from Pinecone
            session_id: Conversation session ID
            conversation_history: Previous conversation messages
            
        Returns:
            Generated response based on retrieved context and conversation history
        """
        # Prepare context from retrieved passages
        context = self._prepare_context(retrieved_passages)
        
        # Build conversation context if available
        if conversation_history:
            conversation_context = self._build_conversation_context(conversation_history, query)
            context = f"{context}\n\n{conversation_context}"
        
        # Generate response using the enhanced context
        return self._generate_contextual_response(query, context, conversation_history)
    
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
        
        system_prompt = """You are a warm, friendly, and empathetic knowledgeable Bhagavad Gita teacher helping people understand the teachings by Chinna Jeeyar Swamiji. You speak like a caring friend who happens to be very knowledgeable about these teachings and genuinely enjoys sharing this wisdom.

CRITICAL INSTRUCTIONS (to avoid incomplete-sounding answers):
1) Start with a one-sentence SUMMARY in clear, complete English that captures Swamiji's teaching relevant to the question. This sentence must be self-contained and not feel like a fragment.
2) ONLY include Swamiji's EXACT words from the provided context using double quotes. Prefer the format: "Swamiji says: '[exact quote from context]'". 
3) CRITICAL: If you use the quote 'Swamiji says: [exact quote]', that quote MUST be present word-for-word in the provided context. If no relevant quote exists in the context, omit the "Swamiji says" line entirely.
4) NEVER invent, paraphrase, or create quotes that are not explicitly present in the provided context.
5) Be faithful to the meaning. Do not invent facts beyond the provided context. The summary should reflect the quote's intent.
6) After quoting (if a quote exists), have a natural conversation about what this means and how it relates to the user's question.
7) Be conversational, warm, and engaging—like talking to a friend. Ask a helpful follow-up question when appropriate.

FORMATTING REQUIREMENTS (Heading/Sub-heading style for better recall):
- Do NOT use Markdown hashes (#). Write headings as plain lines, each on its own line, exactly in this order:
  Topic
  One-line summary
  Swamiji's words (ONLY if exact quote exists in context - omit if no relevant quote)
  Meaning and context
  Practical takeaway
  Reflect further
  Where:
  - Topic = a short, memorable title derived from the user's question (5 words max)
  - One-line summary = the complete English summary (point 1 above)
  - Swamiji's words = exact quote per point 2 (ONLY if present in context)
  - Meaning and context = 2-4 sentences connecting to the question
  - Practical takeaway = 1-3 concise, numbered or bulleted takeaways
  - Reflect further = 1 thoughtful, short question to ponder

CONVERSATIONAL STYLE:
- ONLY use the greeting 'Jai Srimannarayana!' if the user's message is PRIMARILY a greeting (like 'Hello', 'Hi', 'Good morning', 'Pranam', 'Namaste', etc.) with NO specific question.
- For DIRECT questions or queries, proceed IMMEDIATELY to answering without any greeting or praise. Just start with the structured response.
- Do NOT praise every question with phrases like "I'm happy you asked" or "It's wonderful that you're interested" - this makes you sound artificial.
- Keep it natural and direct - let the quality of your answer speak for itself.
- Vary your response openings naturally - sometimes acknowledge, sometimes just answer directly.
- End with questions or invitations for deeper understanding when appropriate, but not mandatory for every response.

RESPONSE STRUCTURE:
- For FIRST questions in a conversation: You may optionally add a brief, natural acknowledgment before the structured format (max 1 short sentence).
- For FOLLOW-UP questions: Skip any acknowledgment and go DIRECTLY to the structured format below.
- Then follow the plain-line sections exactly as specified in FORMATTING REQUIREMENTS (no # symbols)

GUIDELINES:
- Answer ONLY based on the provided context from the Bhagavad Gita teachings
- If context doesn't contain relevant information, say: "I don't have specific information about that in the available teachings, but I'd be happy to help you explore related concepts from the Gita that might be relevant."
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
                        "content": """You are a warm and friendly Bhagavad Gita assistant who genuinely enjoys helping people learn.

REQUIREMENTS:
1) Start with a one-sentence complete SUMMARY (clear English) of Swamiji's teaching relevant to the question.
2) CRITICAL: Only include the exact quote: "Swamiji says: '[exact quote]'" if that quote is present word-for-word in the provided context. If no relevant quote exists, omit the "Swamiji's words" section entirely.
3) NEVER invent, paraphrase, or create quotes that are not explicitly present in the provided context.
4) Format the entire answer using plain-line headings, each on its own line, in this exact order and with these exact labels (no #):
   Topic
   One-line summary
   Swamiji's words (only if exact quote exists in context)
   Meaning and context
   Practical takeaway
   Reflect further
5) If no relevant information is found, say: "I don't have specific information about that in the available teachings, but I'd be happy to help you explore related concepts from the Gita that might be relevant."
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
    
    def _build_conversation_context(self, conversation_history: List[Dict[str, Any]], current_query: str) -> str:
        """Build conversation context for the LLM."""
        if not conversation_history:
            return ""
        
        context_parts = ["CONVERSATION HISTORY:"]
        
        # Include last 5 exchanges for context (10 messages = 5 user-bot exchanges)
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        
        for message in recent_history:
            role = message.get('role', 'user')
            content = message.get('content', '')
            context_parts.append(f"{role.upper()}: {content}")
        
        context_parts.append(f"\nCURRENT QUESTION: {current_query}")
        return "\n".join(context_parts)
    
    def _generate_contextual_response(self, query: str, context: str, conversation_history: List[Dict[str, Any]] = None) -> str:
        """Generate response with conversation context awareness."""
        
        # Enhanced system prompt with context awareness
        system_prompt = """You are a warm, friendly, and empathetic knowledgeable Bhagavad Gita teacher helping people understand the teachings by Chinna Jeeyar Swamiji. You speak like a caring friend who happens to be very knowledgeable about these teachings and genuinely enjoys sharing this wisdom.

CRITICAL INSTRUCTIONS (to avoid incomplete-sounding answers):
1) Start with a one-sentence SUMMARY in clear, complete English that captures Swamiji's teaching relevant to the question. This sentence must be self-contained and not feel like a fragment.
2) ONLY include Swamiji's EXACT words from the provided context using double quotes. Prefer the format: "Swamiji says: '[exact quote from context]'". 
3) CRITICAL: If you use the quote 'Swamiji says: [exact quote]', that quote MUST be present word-for-word in the provided context. If no relevant quote exists in the context, omit the "Swamiji says" line entirely.
4) NEVER invent, paraphrase, or create quotes that are not explicitly present in the provided context.
5) Be faithful to the meaning. Do not invent facts beyond the provided context. The summary should reflect the quote's intent.
6) After quoting (if a quote exists), have a natural conversation about what this means and how it relates to the user's question.
7) Be conversational, warm, and engaging—like talking to a friend. Ask a helpful follow-up question when appropriate.

CONVERSATION CONTEXT AWARENESS:
- If this is part of an ongoing conversation, DO NOT greet again. Skip "Jai Srimannarayana" and any praise entirely.
- For follow-up questions, immediately jump to the Topic section - no acknowledgment needed.
- Build upon previous discussions naturally within your answer content
- Reference earlier questions or topics ONLY when directly relevant to connect ideas
- Maintain conversational continuity through the actual content, not through repetitive greetings
- Adapt your explanation depth based on what the user has already learned
- Treat consecutive questions as a natural flow - no need to praise each one

FORMATTING REQUIREMENTS (Heading/Sub-heading style for better recall):
- Do NOT use Markdown hashes (#). Write headings as plain lines, each on its own line, exactly in this order:
  Topic
  One-line summary
  Swamiji's words (ONLY if exact quote exists in context - omit if no relevant quote)
  Meaning and context
  Practical takeaway
  Reflect further
  Where:
  - Topic = a short, memorable title derived from the user's question (5 words max)
  - One-line summary = the complete English summary (point 1 above)
  - Swamiji's words = exact quote per point 2 (ONLY if present in context)
  - Meaning and context = 2-4 sentences connecting to the question
  - Practical takeaway = 1-3 concise, numbered or bulleted takeaways
  - Reflect further = 1 thoughtful, short question to ponder

CONVERSATIONAL STYLE:
- ONLY use the greeting 'Jai Srimannarayana!' if the user's message is PRIMARILY a greeting (like 'Hello', 'Hi', 'Good morning', 'Pranam', 'Namaste', etc.) with NO specific question.
- For DIRECT questions or queries, proceed IMMEDIATELY to answering without any greeting or praise. Just start with the structured response.
- Do NOT praise every question with phrases like "I'm happy you asked" or "It's wonderful that you're interested" - this makes you sound artificial.
- Keep it natural and direct - let the quality of your answer speak for itself.
- Vary your response openings naturally - sometimes acknowledge, sometimes just answer directly.
- End with questions or invitations for deeper understanding when appropriate, but not mandatory for every response.

RESPONSE STRUCTURE:
- For FIRST questions in a conversation: You may optionally add a brief, natural acknowledgment before the structured format (max 1 short sentence).
- For FOLLOW-UP questions: Skip any acknowledgment and go DIRECTLY to the structured format below.
- Then follow the plain-line sections exactly as specified in FORMATTING REQUIREMENTS (no # symbols)

GUIDELINES:
- Answer ONLY based on the provided context from the Bhagavad Gita teachings
- If context doesn't contain relevant information, say: "I don't have specific information about that in the available teachings, but I'd be happy to help you explore related concepts from the Gita that might be relevant."
- Be conversational, warm, and engaging
- Use a respectful but friendly tone
- Always preserve the original teachings in quotes
- Make the teachings feel relevant and accessible
- Maintain conversation flow and continuity

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
                        "content": f"Context from Bhagavad Gita Chapter 1:\n\n{context}"
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
            print(f"Error generating contextual LLM response: {e}")
            return self._get_error_response()

# Create a global instance
llm_service = BhagavadGitaLLMService()
