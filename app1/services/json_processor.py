import json
import os
import re
from typing import List, Dict, Any

# Use absolute imports for deployment compatibility
from services.embeddings import embed_text
from services.pinecone_services import PineconeService

class JSONDataProcessor:
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 200):
        """
        Initialize the JSON data processor.
        
        Args:
            chunk_size: Maximum size of each chunk
            overlap_size: Number of characters to overlap between chunks
        """
        self.pinecone_service = PineconeService()
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
    def load_json_data(self, json_file_path: str) -> List[Dict[str, Any]]:
        """Load JSON data from file."""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} episodes from JSON file")
            return data
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return []
    
    def split_content_into_chunks(self, content: str, max_length: int = 1000, overlap_size: int = 200) -> List[str]:
        """
        Split content into smaller chunks with overlap for better accuracy.
        
        Args:
            content: Text content to split
            max_length: Maximum length of each chunk
            overlap_size: Number of characters to overlap between chunks
        """
        # First, split by sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed max_length, start a new chunk
            if len(current_chunk) + len(sentence) > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap for next chunk
                if overlap_size > 0 and len(current_chunk) > overlap_size:
                    # Take the last 'overlap_size' characters as overlap
                    overlap_text = current_chunk[-overlap_size:]
                    # Try to break at sentence boundary for better overlap
                    overlap_sentences = re.split(r'(?<=[.!?])\s+', overlap_text)
                    if len(overlap_sentences) > 1:
                        # Use the last complete sentence as overlap
                        overlap_text = overlap_sentences[-1]
                    
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_json_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process JSON episodes into chunks for embedding."""
        print(f"Processing {len(episodes)} episodes from JSON data...")
        print(f"Chunk size: {self.chunk_size} characters, Overlap: {self.overlap_size} characters")
        
        chunks = []
        chunk_id = 0
        
        for episode in episodes:
            chapter = episode.get('chapter', 1)
            episode_num = episode.get('episode', 1)
            content = episode.get('content', '')
            
            if not content.strip():
                continue
            
            # Split content into smaller chunks with overlap
            content_chunks = self.split_content_into_chunks(content, max_length=self.chunk_size, overlap_size=self.overlap_size)
            
            for i, chunk_text in enumerate(content_chunks):
                if chunk_text.strip():
                    chunks.append({
                        'id': f"json_chunk_{chunk_id}",
                        'text': chunk_text.strip(),
                        'metadata': {
                            'chapter': chapter,
                            'episode': episode_num,
                            'chunk_index': i,
                            'total_chunks': len(content_chunks),
                            'source': 'bhagavad-gita-json',
                            'data_type': 'structured_json',
                            'chunk_size': self.chunk_size,
                            'overlap_size': self.overlap_size,
                            'has_overlap': i > 0  # First chunk has no overlap
                        }
                    })
                    chunk_id += 1
        
        print(f"Created {len(chunks)} chunks from JSON episodes")
        return chunks
    
    def create_embeddings_from_json(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert JSON chunks to embeddings."""
        print(f"Creating embeddings for {len(chunks)} JSON chunks...")
        
        embeddings_data = []
        
        for chunk in chunks:
            try:
                embedding = embed_text(chunk['text'])
                embeddings_data.append({
                    'id': chunk['id'],
                    'values': embedding,
                    'metadata': {
                        **chunk['metadata'],
                        'text': chunk['text']  # Include text in metadata
                    }
                })
            except Exception as e:
                print(f"Error creating embedding for chunk {chunk['id']}: {e}")
                continue
        
        print(f"Successfully created {len(embeddings_data)} embeddings from JSON")
        return embeddings_data
    
    def upsert_json_embeddings(self, embeddings_data: List[Dict[str, Any]], batch_size: int = 100):
        """Upsert JSON embeddings to Pinecone index."""
        if not self.pinecone_service.index:
            self.pinecone_service.get_index()
        
        print(f"Upserting {len(embeddings_data)} JSON embeddings to Pinecone...")
        
        total_upserted = 0
        
        for i in range(0, len(embeddings_data), batch_size):
            batch = embeddings_data[i:i + batch_size]
            
            try:
                self.pinecone_service.index.upsert(vectors=batch)
                total_upserted += len(batch)
                print(f"Upserted batch {i//batch_size + 1}/{(len(embeddings_data)-1)//batch_size + 1}")
            except Exception as e:
                print(f"Error upserting batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"Successfully upserted {total_upserted} JSON embeddings to Pinecone")
        return total_upserted
    
    def process_json_file(self, json_file_path: str = "app1/data/chapter_1_episodes.json"):
        """Complete pipeline to process JSON file and store in Pinecone."""
        print("Starting JSON data processing pipeline...")
        print("=" * 60)
        
        # Step 1: Load JSON data
        episodes = self.load_json_data(json_file_path)
        if not episodes:
            print("No episodes found in JSON file!")
            return
        
        # Step 2: Process episodes into chunks
        chunks = self.process_json_episodes(episodes)
        
        # Step 3: Create embeddings
        embeddings_data = self.create_embeddings_from_json(chunks)
        
        # Step 4: Get/create Pinecone index
        self.pinecone_service.get_index()
        
        # Step 5: Upsert to Pinecone
        upserted_count = self.upsert_json_embeddings(embeddings_data)
        
        print("=" * 60)
        print("JSON PROCESSING COMPLETE!")
        print(f"Total episodes processed: {len(episodes)}")
        print(f"Total chunks created: {len(chunks)}")
        print(f"Total embeddings created: {len(embeddings_data)}")
        print(f"Total embeddings upserted: {upserted_count}")
        print(f"Pinecone index: {self.pinecone_service.index_name}")
        
        return {
            'episodes_processed': len(episodes),
            'chunks_created': len(chunks),
            'embeddings_created': len(embeddings_data),
            'embeddings_upserted': upserted_count,
            'index_name': self.pinecone_service.index_name
        }

def main():
    """Main function to process JSON data."""
    try:
        # Initialize processor
        processor = JSONDataProcessor()
        
        # Process JSON file
        result = processor.process_json_file()
        
        print("\n" + "=" * 60)
        print("SUCCESS! JSON data is now stored in Pinecone.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
