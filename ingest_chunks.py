import json
import requests
import time
from tqdm import tqdm

def ingest_chunks_to_pinecone():
    """Ingest all chunks from JSON file into Pinecone via API."""
    
    # Load chunks from JSON file
    with open('all_chapter1_chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Found {len(chunks)} chunks to ingest...")
    
    # API endpoint
    url = "http://localhost:8000/ingest/"
    
    successful_ingests = 0
    failed_ingests = 0
    
    # Process chunks in batches
    for i, chunk in enumerate(tqdm(chunks, desc="Ingesting chunks")):
        try:
            response = requests.post(url, json={"transcript": chunk["text"]})
            
            if response.status_code == 200:
                successful_ingests += 1
            else:
                print(f"Failed to ingest chunk {i}: {response.text}")
                failed_ingests += 1
                
        except Exception as e:
            print(f"Error ingesting chunk {i}: {e}")
            failed_ingests += 1
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    print(f"\n✅ Ingestion complete!")
    print(f"✅ Successful: {successful_ingests}")
    print(f"❌ Failed: {failed_ingests}")

if __name__ == "__main__":
    ingest_chunks_to_pinecone()

