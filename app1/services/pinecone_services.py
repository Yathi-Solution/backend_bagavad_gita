import os
import re
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import time
from tqdm import tqdm
from app1.services.embeddings import embed_text

load_dotenv()

class PineconeService:
    def __init__(self):
        """Initialize Pinecone service with API key and index configuration."""
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = os.getenv("PINECONE_INDEX", "bhagavad-gita-chapter1")
        self.index = None
        
    def create_index(self, dimension: int = 3072):
        """Create a new Pinecone index for Bhagavad Gita embeddings."""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name in existing_indexes:
                print(f"Index '{self.index_name}' already exists. Deleting it first...")
                self.pc.delete_index(self.index_name)
                time.sleep(5)  # Wait for deletion to complete
            
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,  # text-embedding-3-large dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            print(f"Index '{self.index_name}' created successfully!")
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            print("Index is ready!")
            
            # Get the index
            self.index = self.pc.Index(self.index_name)
            return self.index
            
        except Exception as e:
            print(f"Error creating index: {e}")
            raise
    
    def get_index(self):
        """Get existing index or create new one."""
        try:
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                return self.create_index()
            else:
                print(f"Using existing index: {self.index_name}")
                self.index = self.pc.Index(self.index_name)
                return self.index
                
        except Exception as e:
            print(f"Error getting index: {e}")
            raise
    
    def parse_extracted_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse the extracted text file and split into chunks."""
        print(f"Parsing extracted text file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by file separators
        file_sections = re.split(r'={80}\nFILE: (.+?)\.docx\n={80}', content)
        
        chunks = []
        chunk_id = 0
        
        for i in range(1, len(file_sections), 2):
            if i + 1 < len(file_sections):
                file_name = file_sections[i]
                file_content = file_sections[i + 1].strip()
                
                if not file_content:
                    continue
                
                # Split content into smaller chunks with overlap (max 1000 characters, 200 overlap)
                text_chunks = self.split_text_into_chunks(file_content, max_length=1000, overlap_size=200)
                
                for j, chunk_text in enumerate(text_chunks):
                    if chunk_text.strip():
                        # Determine chapter from file name, e.g., c2-ep10 -> chapter 2
                        chapter_match = re.search(r'c(\d+)-', file_name)
                        chapter_num = int(chapter_match.group(1)) if chapter_match else 1
                        chunks.append({
                            'id': f"chunk_{chunk_id}",
                            'text': chunk_text.strip(),
                            'metadata': {
                                'file_name': file_name,
                                'episode': self.extract_episode_number(file_name),
                                'chunk_index': j,
                                'total_chunks': len(text_chunks),
                                'source': f"bhagavad-gita-chapter{chapter_num}"
                            }
                        })
                        chunk_id += 1
        
        print(f"Created {len(chunks)} text chunks from {len(file_sections)//2} episodes")
        return chunks
    
    def split_text_into_chunks(self, text: str, max_length: int = 1000, overlap_size: int = 200) -> List[str]:
        """
        Split text into smaller chunks with overlap while preserving sentence boundaries.
        
        Args:
            text: Text content to split
            max_length: Maximum length of each chunk
            overlap_size: Number of characters to overlap between chunks
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
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
    
    def extract_episode_number(self, file_name: str) -> int:
        """Extract episode number from file name like 'c2-ep10' (chapter-agnostic)."""
        match = re.search(r'c\d+-ep(\d+)', file_name)
        return int(match.group(1)) if match else 0
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert text chunks to embeddings."""
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        embeddings_data = []
        
        for chunk in tqdm(chunks, desc="Creating embeddings"):
            try:
                embedding = embed_text(chunk['text'])
                embeddings_data.append({
                    'id': chunk['id'],
                    'values': embedding,
                    'metadata': {
                        **chunk['metadata'],
                        'text': chunk['text']
                    }
                })
            except Exception as e:
                print(f"Error creating embedding for chunk {chunk['id']}: {e}")
                continue
        
        print(f"Successfully created {len(embeddings_data)} embeddings")
        return embeddings_data
    
    def upsert_embeddings(self, embeddings_data: List[Dict[str, Any]], batch_size: int = 100):
        """Upsert embeddings to Pinecone index in batches."""
        if not self.index:
            self.get_index()
        
        print(f"Upserting {len(embeddings_data)} embeddings to Pinecone...")
        
        total_upserted = 0
        
        for i in tqdm(range(0, len(embeddings_data), batch_size), desc="Upserting to Pinecone"):
            batch = embeddings_data[i:i + batch_size]
            
            try:
                self.index.upsert(vectors=batch)
                total_upserted += len(batch)
            except Exception as e:
                print(f"Error upserting batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"Successfully upserted {total_upserted} embeddings to Pinecone")
        return total_upserted
    
    def search(self, query: str, top_k: int = 10, include_metadata: bool = True):
        """Search for similar text chunks in the Pinecone index."""
        if not self.index:
            self.get_index()
        
        # Generate embedding for the query
        query_embedding = embed_text(query)
        
        # Search in Pinecone
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=include_metadata
        )
        
        return search_results
    
    def get_index_stats(self):
        """Get statistics about the Pinecone index."""
        if not self.index:
            self.get_index()
        
        stats = self.index.describe_index_stats()
        
        # Get breakdown by source
        source_breakdown = {}
        if hasattr(stats, 'namespaces') and stats.namespaces:
            for namespace, data in stats.namespaces.items():
                if hasattr(data, 'metadata') and data.metadata:
                    for metadata_key, count in data.metadata.items():
                        if metadata_key == 'source':
                            source_breakdown[count] = source_breakdown.get(count, 0) + 1
        
        return {
            'stats': stats,
            'source_breakdown': source_breakdown
        }
    
    def process_chapter_text(self, text_file_path: str, chapter_number: int):
        """Complete pipeline to process Chapter text and store in Pinecone."""
        print(f"Starting Chapter {chapter_number} text processing pipeline...")
        print("=" * 60)
        
        # Step 1: Parse text file
        chunks = self.parse_extracted_text(text_file_path)
        
        # Step 2: Create embeddings
        embeddings_data = self.create_embeddings(chunks)
        
        # Step 3: Get/create Pinecone index
        self.get_index()
        
        # Step 4: Upsert to Pinecone
        upserted_count = self.upsert_embeddings(embeddings_data)
        
        print("=" * 60)
        print("PROCESSING COMPLETE!")
        print(f"Total chunks processed: {len(chunks)}")
        print(f"Total embeddings created: {len(embeddings_data)}")
        print(f"Total embeddings upserted: {upserted_count}")
        print(f"Pinecone index: {self.index_name}")
        
        return {
            'chunks_processed': len(chunks),
            'embeddings_created': len(embeddings_data),
            'embeddings_upserted': upserted_count,
            'index_name': self.index_name
        }

    def parse_episode_json(self, json_file_path: str, chapter_number: int) -> List[Dict[str, Any]]:
        """Parse episode JSON file and create chunks for ingestion."""
        print(f"Parsing episode JSON file: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            episodes = json.load(f)
        
        chunks = []
        chunk_id = 0
        
        for episode_data in episodes:
            episode_num = int(episode_data.get("episode", 0))
            content = episode_data.get("content", "")
            
            if not content.strip():
                continue
            
            # Split content into smaller chunks with overlap
            text_chunks = self.split_text_into_chunks(content, max_length=1000, overlap_size=200)
            
            for j, chunk_text in enumerate(text_chunks):
                if chunk_text.strip():
                    chunks.append({
                        'id': f"c{chapter_number}-ep{episode_num}-chunk-{j}",
                        'text': chunk_text.strip(),
                        'metadata': {
                            'file_name': f"c{chapter_number}-ep{episode_num}",
                            'episode': episode_num,
                            'chunk_index': j,
                            'total_chunks': len(text_chunks),
                            'source': f"bhagavad-gita-chapter{chapter_number}"
                        }
                    })
                    chunk_id += 1
        
        print(f"Created {len(chunks)} text chunks from {len(episodes)} episodes")
        return chunks

    def process_chapter_json(self, json_file_path: str, chapter_number: int):
        """Complete pipeline to process Chapter JSON and store in Pinecone."""
        print(f"Starting Chapter {chapter_number} JSON processing pipeline...")
        print("=" * 60)
        
        # Step 1: Parse JSON file
        chunks = self.parse_episode_json(json_file_path, chapter_number)
        
        # Step 2: Create embeddings
        embeddings_data = self.create_embeddings(chunks)
        
        # Step 3: Get/create Pinecone index
        self.get_index()
        
        # Step 4: Upsert to Pinecone
        upserted_count = self.upsert_embeddings(embeddings_data)
        
        print("=" * 60)
        print("PROCESSING COMPLETE!")
        print(f"Total chunks processed: {len(chunks)}")
        print(f"Total embeddings created: {len(embeddings_data)}")
        print(f"Total embeddings upserted: {upserted_count}")
        print(f"Pinecone index: {self.index_name}")
        
        return {
            'chunks_processed': len(chunks),
            'embeddings_created': len(embeddings_data),
            'embeddings_upserted': upserted_count,
            'index_name': self.index_name
        }

    def process_chapter1_text(self, text_file_path: str = "app1/data/chapter1_extracted_text.txt"):
        """Backward-compatible wrapper for Chapter 1 processing."""
        return self.process_chapter_text(text_file_path, 1)

def main():
    """Main function to run the Pinecone service."""
    try:
        # Initialize service
        pinecone_service = PineconeService()
        
        # Process Chapter 1 text
        result = pinecone_service.process_chapter1_text()
        
        print("\n" + "=" * 60)
        print("SUCCESS! Chapter 1 embeddings are now stored in Pinecone.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
