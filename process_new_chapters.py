#!/usr/bin/env python3
"""
Script to process Chapter 2 and Chapter 3 documents and create chunk files.
This script will process all .docx files in the data folder and create
separate chunk files for each chapter.
"""

import os
import json
from app.utils.docx_parser import extract_text_from_docx
import hashlib

def process_chapter_documents():
    """Process all documents and create chapter-specific chunk files"""
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        return
    
    # Get all .docx files
    docx_files = [f for f in os.listdir(data_dir) if f.endswith('.docx')]
    
    if not docx_files:
        print(f"❌ No .docx files found in '{data_dir}' directory!")
        return
    
    print(f"📁 Found {len(docx_files)} .docx files")
    
    # Group files by chapter
    chapters = {}
    for filename in docx_files:
        if filename.startswith('c1-'):
            chapter = 'chapter1'
        elif filename.startswith('c2-'):
            chapter = 'chapter2'
        elif filename.startswith('c3-'):
            chapter = 'chapter3'
        else:
            print(f"⚠️  Skipping file with unknown format: {filename}")
            continue
        
        if chapter not in chapters:
            chapters[chapter] = []
        chapters[chapter].append(filename)
    
    print(f"📚 Found chapters: {list(chapters.keys())}")
    
    # Process each chapter
    for chapter, files in chapters.items():
        print(f"\n🔄 Processing {chapter} with {len(files)} files...")
        
        all_chunks = []
        
        for filename in sorted(files):
            file_path = os.path.join(data_dir, filename)
            print(f"  📄 Processing {filename}...")
            
            try:
                # Parse the document
                content = extract_text_from_docx(file_path)
                
                if not content:
                    print(f"    ⚠️  No content found in {filename}")
                    continue
                
                # Split into chunks (you can adjust chunk size)
                chunk_size = 1000  # characters per chunk
                chunks = []
                
                for i in range(0, len(content), chunk_size):
                    chunk_text = content[i:i + chunk_size]
                    if chunk_text.strip():  # Only add non-empty chunks
                        chunk_id = f"{filename.replace('.docx', '')}-chunk-{i//chunk_size}"
                        chunks.append({
                            "id": chunk_id,
                            "text": chunk_text.strip()
                        })
                
                all_chunks.extend(chunks)
                print(f"    ✅ Created {len(chunks)} chunks from {filename}")
                
            except Exception as e:
                print(f"    ❌ Error processing {filename}: {e}")
                continue
        
        # Save chunks for this chapter
        if all_chunks:
            output_file = f"all_{chapter}_chunks.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)
            
            print(f"  💾 Saved {len(all_chunks)} chunks to {output_file}")
        else:
            print(f"  ⚠️  No chunks created for {chapter}")
    
    print(f"\n✅ Processing complete!")
    print(f"\nNext steps:")
    print(f"1. Check the generated chunk files")
    print(f"2. Use /ingest/bulk-multi-chapter endpoint to ingest all chapters")
    print(f"3. Test chat functionality with questions from all chapters")

if __name__ == "__main__":
    process_chapter_documents()
