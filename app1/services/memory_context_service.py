"""
In-memory context service for when Supabase is not available.
Provides basic conversation history management using Python dictionaries.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

class MemoryContextService:
    def __init__(self):
        """Initialize in-memory context storage."""
        self.conversations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.users: Dict[str, Dict[str, Any]] = {}
        print("Memory context service initialized (in-memory conversation history)")
    
    async def get_or_create_user(self, user_id: str = None, user_name: str = "Anonymous") -> str:
        """Get or create a user in memory."""
        if not user_id:
            user_id = str(uuid.uuid4())
        
        if user_id not in self.users:
            self.users[user_id] = {
                'id': user_id,
                'name': user_name,
                'createdAt': datetime.now().isoformat(),
                'updatedAt': datetime.now().isoformat()
            }
        
        return user_id
    
    async def get_or_create_conversation(self, session_id: str, user_id: str, title: str = None) -> str:
        """Get or create a conversation in memory."""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        return session_id
    
    async def get_conversation_context(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation messages for context."""
        if session_id not in self.conversations:
            return []
        
        # Return last 'limit' messages
        return self.conversations[session_id][-limit:] if self.conversations[session_id] else []
    
    async def store_message(self, session_id: str, content: str, role: str) -> str:
        """Store a message in the conversation."""
        message_id = str(uuid.uuid4())
        message_data = {
            'id': message_id,
            'conversationId': session_id,
            'content': content,
            'role': role,
            'createdAt': datetime.now().isoformat()
        }
        
        self.conversations[session_id].append(message_data)
        return message_id
    
    async def store_feedback(self, session_id: str, query_text: str, answer_text: str = None, 
                           rating: int = None, feedback: str = None, user_name: str = None) -> str:
        """Store user feedback (placeholder - not persisted in memory)."""
        return str(uuid.uuid4())
    
    def build_context_prompt(self, conversation_history: List[Dict[str, Any]], current_query: str) -> str:
        """Build context-aware prompt from conversation history."""
        if not conversation_history:
            return f"Question: {current_query}"
        
        context_parts = ["CONVERSATION HISTORY:"]
        
        # Include last 5 exchanges for context
        recent_history = conversation_history[-10:]  # Last 10 messages (5 exchanges)
        
        for message in recent_history:
            role = message.get('role', 'user')
            content = message.get('content', '')
            context_parts.append(f"{role.upper()}: {content}")
        
        context_parts.append(f"\nCURRENT QUESTION: {current_query}")
        
        return "\n".join(context_parts)
    
    def extract_topics_from_conversation(self, conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Extract topics from conversation history for better context."""
        topics = []
        
        # Simple topic extraction based on keywords
        topic_keywords = {
            'dharma': ['dharma', 'duty', 'righteousness', 'moral'],
            'karma': ['karma', 'action', 'deed', 'work'],
            'bhakti': ['bhakti', 'devotion', 'worship', 'prayer'],
            'jnana': ['jnana', 'knowledge', 'wisdom', 'understanding'],
            'arjuna': ['arjuna', 'warrior', 'pandava'],
            'krishna': ['krishna', 'lord', 'god', 'deity'],
            'gita': ['gita', 'scripture', 'teaching', 'verse'],
            'samsara': ['samsara', 'cycle', 'birth', 'death', 'reincarnation'],
            'moksha': ['moksha', 'liberation', 'freedom', 'enlightenment']
        }
        
        all_content = " ".join([msg.get('content', '') for msg in conversation_history])
        all_content_lower = all_content.lower()
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in all_content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics

# Create a global instance
memory_context_service = MemoryContextService()
