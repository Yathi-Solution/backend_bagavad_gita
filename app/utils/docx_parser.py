import os
import json
from docx import Document
import re

def extract_text_from_docx(file_path):
    """Extract text from a .docx file."""
    try:
        doc = Document(file_path)
        text = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text.strip())
        
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def chunk_text(text, max_words=200):
    """Split text into chunks of approximately max_words."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    
    return chunks

def save_chunks_to_json(chunks, output_file):
    """Save chunks to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chunks)} chunks to {output_file}")
    except Exception as e:
        print(f"Error saving chunks: {e}")

