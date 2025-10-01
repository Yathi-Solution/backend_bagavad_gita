import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from supabase import create_client, Client
from services.memory_context_service import memory_context_service

class SupabaseContextService:
    def __init__(self):
        """Initialize Supabase client for context management."""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase: Client = None
        self.enabled = False
        
        if not self.supabase_url or not self.supabase_key:
            print("WARNING: Supabase environment variables not found!")
            print("   SUPABASE_URL and SUPABASE_ANON_KEY are required for context awareness.")
            print("   Bot will work with in-memory context only.")
            print("   To enable full context awareness, create a .env file with:")
            print("   SUPABASE_URL=your_supabase_url")
            print("   SUPABASE_ANON_KEY=your_supabase_anon_key")
            return
        
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            self.enabled = True
            print("Supabase context service initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize Supabase: {e}")
            print("   Bot will work with in-memory context only.")
        
    async def get_or_create_user(self, user_id: str = None, user_name: str = "Anonymous") -> str:
        """Get or create a user in the database."""
        if not self.enabled:
            return await memory_context_service.get_or_create_user(user_id, user_name)
            
        try:
            if user_id:
                # Check if user exists by ID
                result = self.supabase.table('users').select('id').eq('id', user_id).execute()
                if result.data:
                    return user_id
            
            # Check if user exists by name first
            existing_user = self.supabase.table('users').select('id').eq('name', user_name).execute()
            if existing_user.data:
                return existing_user.data[0]['id']
            
            # Create new user with unique name if "Anonymous" already exists
            if user_name == "Anonymous":
                # Generate unique name for anonymous users
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                user_name = f"Anonymous_{timestamp}"
            
            new_user_id = user_id or str(uuid.uuid4())
            user_data = {
                'id': new_user_id,
                'name': user_name,
                'createdAt': datetime.now(timezone.utc).isoformat(),
                'updatedAt': datetime.now(timezone.utc).isoformat()
            }
            
            self.supabase.table('users').insert(user_data).execute()
            return new_user_id
            
        except Exception as e:
            print(f"Error managing user: {e}")
            return await memory_context_service.get_or_create_user(user_id, user_name)
    
    async def get_or_create_conversation(self, session_id: str, user_id: str, title: str = None) -> str:
        """Get or create a conversation for the session."""
        if not self.enabled:
            return await memory_context_service.get_or_create_conversation(session_id, user_id, title)
            
        try:
            # Check if conversation exists
            result = self.supabase.table('conversations').select('id').eq('id', session_id).execute()
            if result.data:
                # Update last activity
                self.supabase.table('conversations').update({
                    'updatedAt': datetime.now(timezone.utc).isoformat()
                }).eq('id', session_id).execute()
                return session_id
            
            # Create new conversation
            conversation_data = {
                'id': session_id,
                'title': title or f"Bhagavad Gita Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                'createdAt': datetime.now(timezone.utc).isoformat(),
                'updatedAt': datetime.now(timezone.utc).isoformat(),
                'userId': user_id
            }
            
            self.supabase.table('conversations').insert(conversation_data).execute()
            return session_id
            
        except Exception as e:
            print(f"Error managing conversation: {e}")
            return await memory_context_service.get_or_create_conversation(session_id, user_id, title)
    
    async def get_conversation_context(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation messages for context."""
        if not self.enabled:
            return await memory_context_service.get_conversation_context(session_id, limit)
            
        try:
            result = self.supabase.table('messages')\
                .select('id, content, role, createdAt')\
                .eq('conversationId', session_id)\
                .order('createdAt', desc=True)\
                .limit(limit)\
                .execute()
            
            # Reverse to get chronological order (oldest first)
            history_data = list(reversed(result.data)) if result.data else []
            
            # 🚨 NEW CODE: CONVERT ROLE BACK TO LOWERCASE FOR LLM API 🚨
            llm_formatted_history = []
            for turn in history_data:
                # Convert the DB's role (e.g., 'User') back to the LLM's role (e.g., 'user')
                llm_formatted_history.append({
                    "role": turn.get('role', 'user').lower(),
                    "content": turn.get('content', '')
                })
            
            return llm_formatted_history
            
        except Exception as e:
            print(f"Error fetching conversation context: {e}")
            return await memory_context_service.get_conversation_context(session_id, limit)
    
    async def add_conversation_turn(self, session_id: str, role: str, content: str):
        """Adds a single message (turn) to the conversation history in Supabase."""
        if not self.enabled:
            print("Supabase disabled. Falling back to memory service for save.")
            return await memory_context_service.add_conversation_turn(session_id, role, content)

        try:
            # Ensure role is lowercase to match database ENUM ('user', 'assistant')
            db_role = role.lower()

            message_id = str(uuid.uuid4())
            data = {
                'id': message_id,
                'conversationId': session_id,
                'role': db_role,
                'content': content,
                'createdAt': datetime.now(timezone.utc).isoformat(),
            }

            response = self.supabase.table('messages').insert(data).execute()
            
            # Check for immediate errors from the Supabase client
            if response.data and len(response.data) > 0:
                return response.data[0].get('id')
            else:
                # Handle cases where insert succeeded but returned no data (if configured that way)
                return message_id

        except Exception as e:
            print(f"Error storing message: {e}")
            # Fallback save to memory service if Supabase fails (optional but safe)
            return await memory_context_service.add_conversation_turn(session_id, role, content)
    
    async def store_message(self, session_id: str, content: str, role: str) -> str:
        """Store a message in the conversation."""
        if not self.enabled:
            return await memory_context_service.store_message(session_id, content, role)
            
        try:
            # Ensure role is lowercase to match database ENUM ('user', 'assistant')
            db_role = role.lower()
            
            message_id = str(uuid.uuid4())
            message_data = {
                'id': message_id,
                'conversationId': session_id,
                'content': content,
                'role': db_role,
                'createdAt': datetime.now(timezone.utc).isoformat()
            }
            
            self.supabase.table('messages').insert(message_data).execute()
            return message_id
            
        except Exception as e:
            print(f"Error storing message: {e}")
            return await memory_context_service.store_message(session_id, content, role)
    
    async def store_feedback(self, session_id: str, query_text: str, answer_text: str = None, 
                           rating: int = None, feedback: str = None, user_name: str = None) -> str:
        """Store user feedback for the conversation."""
        if not self.enabled:
            return await memory_context_service.store_feedback(session_id, query_text, answer_text, rating, feedback, user_name)
            
        try:
            feedback_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'type': 'conversation',
                'rating': rating,
                'feedback': feedback,
                'session_id': session_id,
                'query_text': query_text,
                'answer_text': answer_text,
                'user_name': user_name,
                'metadata': {'timestamp': datetime.now(timezone.utc).isoformat()}
            }
            
            result = self.supabase.table('feedback').insert(feedback_data).execute()
            return result.data[0]['id'] if result.data else str(uuid.uuid4())
            
        except Exception as e:
            print(f"Error storing feedback: {e}")
            return await memory_context_service.store_feedback(session_id, query_text, answer_text, rating, feedback, user_name)
    
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
            # Handle different role value formats
            if role.lower() == 'human':
                role = 'user'
            elif role.lower() == 'ai':
                role = 'assistant'
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
supabase_context_service = SupabaseContextService()
