import os
import hashlib
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from functools import lru_cache
from typing import List

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple in-memory cache for embeddings (max 500 entries)
_embedding_cache = {}
_cache_max_size = 500

def _get_cache_key(text: str) -> str:
    """Generate a cache key from text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def embed_text(text: str) -> list:
    """Synchronous embedding with cache."""
    cache_key = _get_cache_key(text)
    
    # Check cache first
    if cache_key in _embedding_cache:
        print(f"✓ Cache hit for embedding (saved ~500ms)")
        return _embedding_cache[cache_key]
    
    # Generate embedding if not cached
    # NOTE: Using text-embedding-3-large to match existing Pinecone index (3072 dimensions)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    embedding = response.data[0].embedding
    
    # Store in cache (with size limit)
    if len(_embedding_cache) >= _cache_max_size:
        # Remove oldest entry (FIFO)
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]
    
    _embedding_cache[cache_key] = embedding
    return embedding

async def embed_text_async(text: str) -> List[float]:
    """Async embedding with cache for better performance."""
    cache_key = _get_cache_key(text)
    
    # Check cache first
    if cache_key in _embedding_cache:
        print(f"✓ Cache hit for embedding (saved ~500ms)")
        return _embedding_cache[cache_key]
    
    # Generate embedding if not cached
    # NOTE: Using text-embedding-3-large to match existing Pinecone index (3072 dimensions)
    response = await async_client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    embedding = response.data[0].embedding
    
    # Store in cache (with size limit)
    if len(_embedding_cache) >= _cache_max_size:
        # Remove oldest entry (FIFO)
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]
    
    _embedding_cache[cache_key] = embedding
    return embedding

def clear_embedding_cache():
    """Clear the embedding cache."""
    global _embedding_cache
    _embedding_cache = {}
    print("Embedding cache cleared")

def get_cache_stats():
    """Get cache statistics."""
    return {
        "cached_entries": len(_embedding_cache),
        "max_size": _cache_max_size,
        "cache_hit_potential": f"{len(_embedding_cache)} queries"
    }
