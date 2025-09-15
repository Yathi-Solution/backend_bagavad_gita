import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time

load_dotenv()

def initialize_pinecone():
    """Initialize Pinecone client and create index if it doesn't exist."""
    
    # Check for required environment variables
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)
    
    # Get index name from environment with default
    index_name = os.getenv("PINECONE_INDEX", "chatbot-index")
    
    try:
        # Check if index exists, create if it doesn't
        existing_indexes = pc.list_indexes().names()
        
        if index_name not in existing_indexes:
            print(f"Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI text-embedding-3-small dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index {index_name} created successfully!")
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            print("Index is ready!")
            
        else:
            print(f"Index {index_name} already exists")
        
        # Get the index
        return pc.Index(index_name)
        
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        raise

# Initialize the index
index = initialize_pinecone()
