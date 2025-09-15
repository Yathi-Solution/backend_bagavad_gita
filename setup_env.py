#!/usr/bin/env python3
"""
Setup script to help configure environment variables for the chatbot.
Run this script to create a .env file with the required API keys.
"""

import os

def create_env_file():
    """Create .env file with required environment variables"""
    
    print("🤖 Chatbot Environment Setup")
    print("=" * 40)
    print()
    
    # Check if .env already exists
    if os.path.exists('.env'):
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    print("Please provide the following API keys:")
    print()
    
    # Get OpenAI API key
    openai_key = input("🔑 Enter your OpenAI API key: ").strip()
    if not openai_key:
        print("❌ OpenAI API key is required!")
        return
    
    # Get Pinecone API key
    pinecone_key = input("🔑 Enter your Pinecone API key: ").strip()
    if not pinecone_key:
        print("❌ Pinecone API key is required!")
        return
    
    # Get Pinecone index name (optional)
    index_name = input("📊 Enter Pinecone index name (default: chatbot-index): ").strip()
    if not index_name:
        index_name = "chatbot-index"
    
    # Create .env file
    env_content = f"""# OpenAI API Key
OPENAI_API_KEY={openai_key}

# Pinecone API Key
PINECONE_API_KEY={pinecone_key}

# Pinecone Index Name
PINECONE_INDEX={index_name}
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print()
        print("✅ .env file created successfully!")
        print()
        print("Next steps:")
        print("1. Make sure your FastAPI server is running")
        print("2. Visit http://localhost:8000/docs")
        print("3. Use the /ingest/bulk endpoint to ingest your chunks")
        print("4. Test the /chat endpoint with your questions")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

if __name__ == "__main__":
    create_env_file()
