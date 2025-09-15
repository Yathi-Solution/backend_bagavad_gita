from fastapi import APIRouter
from app.services import pinecone_client, embeddings
import hashlib
import json
import os
from typing import List, Dict, Any

router = APIRouter()

@router.post("/")
def ingest(transcript: str):
    vector = embeddings.embed_text(transcript)
    doc_id = hashlib.md5(transcript.encode()).hexdigest()
    pinecone_client.index.upsert([(doc_id, vector, {"text": transcript})])
    return {"status": "ingested", "id": doc_id}

@router.post("/bulk")
def ingest_all_chunks():
    """Ingest all chunks from all_chapter1_chunks.json into Pinecone"""
    try:
        # Path to the chunks file
        chunks_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "all_chapter1_chunks.json")
        
        # Check if file exists
        if not os.path.exists(chunks_file):
            return {"error": "all_chapter1_chunks.json file not found"}
        
        # Load chunks
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not chunks:
            return {"error": "No chunks found in the file"}
        
        # Process chunks in batches
        batch_size = 100
        total_chunks = len(chunks)
        ingested_count = 0
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            vectors_to_upsert = []
            
            for chunk in batch:
                try:
                    # Generate embedding for the chunk text
                    vector = embeddings.embed_text(chunk["text"])
                    
                    # Use the chunk ID as the vector ID
                    vector_id = chunk["id"]
                    
                    # Prepare metadata
                    metadata = {
                        "text": chunk["text"],
                        "chunk_id": chunk["id"]
                    }
                    
                    vectors_to_upsert.append((vector_id, vector, metadata))
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk.get('id', 'unknown')}: {e}")
                    continue
            
            # Upsert batch to Pinecone
            if vectors_to_upsert:
                pinecone_client.index.upsert(vectors_to_upsert)
                ingested_count += len(vectors_to_upsert)
                print(f"Ingested batch {i//batch_size + 1}: {len(vectors_to_upsert)} chunks")
        
        return {
            "status": "success",
            "total_chunks": total_chunks,
            "ingested_chunks": ingested_count,
            "message": f"Successfully ingested {ingested_count} out of {total_chunks} chunks"
        }
        
    except Exception as e:
        return {"error": f"Failed to ingest chunks: {str(e)}"}

@router.post("/bulk-multi-chapter")
def ingest_multiple_chapters():
    """Ingest chunks from multiple chapter files into Pinecone"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # Look for all chapter chunk files
        chunk_files = []
        for filename in os.listdir(base_dir):
            if filename.startswith("all_chapter") and filename.endswith("_chunks.json"):
                chunk_files.append(os.path.join(base_dir, filename))
        
        if not chunk_files:
            return {"error": "No chapter chunk files found. Expected files like: all_chapter1_chunks.json, all_chapter2_chunks.json, etc."}
        
        all_chunks = []
        processed_files = []
        
        # Load chunks from all files
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    file_chunks = json.load(f)
                    all_chunks.extend(file_chunks)
                    processed_files.append(os.path.basename(chunk_file))
                    print(f"Loaded {len(file_chunks)} chunks from {os.path.basename(chunk_file)}")
            except Exception as e:
                print(f"Error loading {chunk_file}: {e}")
                continue
        
        if not all_chunks:
            return {"error": "No chunks found in any of the files"}
        
        # Process chunks in batches
        batch_size = 100
        total_chunks = len(all_chunks)
        ingested_count = 0
        
        for i in range(0, total_chunks, batch_size):
            batch = all_chunks[i:i + batch_size]
            vectors_to_upsert = []
            
            for chunk in batch:
                try:
                    # Generate embedding for the chunk text
                    vector = embeddings.embed_text(chunk["text"])
                    
                    # Use the chunk ID as the vector ID
                    vector_id = chunk["id"]
                    
                    # Prepare metadata
                    metadata = {
                        "text": chunk["text"],
                        "chunk_id": chunk["id"]
                    }
                    
                    vectors_to_upsert.append((vector_id, vector, metadata))
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk.get('id', 'unknown')}: {e}")
                    continue
            
            # Upsert batch to Pinecone
            if vectors_to_upsert:
                pinecone_client.index.upsert(vectors_to_upsert)
                ingested_count += len(vectors_to_upsert)
                print(f"Ingested batch {i//batch_size + 1}: {len(vectors_to_upsert)} chunks")
        
        return {
            "status": "success",
            "processed_files": processed_files,
            "total_chunks": total_chunks,
            "ingested_chunks": ingested_count,
            "message": f"Successfully ingested {ingested_count} out of {total_chunks} chunks from {len(processed_files)} files"
        }
        
    except Exception as e:
        return {"error": f"Failed to ingest multiple chapters: {str(e)}"}

@router.get("/status")
def get_ingestion_status():
    """Get the current status of the Pinecone index"""
    try:
        stats = pinecone_client.index.describe_index_stats()
        return {
            "status": "success",
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness
        }
    except Exception as e:
        return {"error": f"Failed to get index status: {str(e)}"}
