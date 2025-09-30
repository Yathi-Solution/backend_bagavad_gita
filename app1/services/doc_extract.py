import os
from docx import Document

def read_docx(file_path):
    """Extract text from a Word document"""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_all_chapter_files(chapter_folder):
    """Extract text from all .docx files in a chapter folder"""
    folder_path = f"app1/data/{chapter_folder}"
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist!")
        return
    
    # Get all .docx files in the folder
    docx_files = [f for f in os.listdir(folder_path) if f.endswith('.docx')]
    docx_files.sort()  # Sort files for consistent order
    
    print(f"Found {len(docx_files)} files in {chapter_folder}")
    print("=" * 80)
    
    all_extracted_data = {}
    
    for file_name in docx_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"\nProcessing: {file_name}")
        print("-" * 50)
        
        try:
            text_data = read_docx(file_path)
            all_extracted_data[file_name] = text_data
            print(f"[SUCCESS] Successfully extracted text from {file_name}")
            print(f"  Text length: {len(text_data)} characters")
            
            # Show first 200 characters as preview
            preview = text_data[:200].replace('\n', ' ')
            print(f"  Preview: {preview}...")
            
        except Exception as e:
            print(f"[ERROR] Error processing {file_name}: {str(e)}")
    
    return all_extracted_data

# Extract all files from chapter1
if __name__ == "__main__":
    extracted_data = extract_all_chapter_files("chapter1")
    
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(extracted_data)}")
    
    # Optionally save all extracted data to a single file
    save_to_file = input("\nDo you want to save all extracted text to a file? (y/n): ").lower().strip()
    if save_to_file == 'y':
        output_file = "app1/data/chapter1_extracted_text.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for file_name, text in extracted_data.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"FILE: {file_name}\n")
                f.write(f"{'='*80}\n")
                f.write(text)
                f.write(f"\n\n")
        print(f"[SUCCESS] All extracted text saved to: {output_file}")
