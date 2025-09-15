import glob
import os
from app.utils.docx_parser import extract_text_from_docx, chunk_text, save_chunks_to_json

OUTPUT_FILE = "all_chapter1_chunks.json"
DOCS_PATH = "data/*.docx"

all_chunks = []
for file in glob.glob(DOCS_PATH):
    filename = os.path.basename(file).replace(".docx", "")
    text = extract_text_from_docx(file)
    chunks = chunk_text(text, max_words=200)

    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "id": f"{filename}-chunk-{i}",
            "text": chunk
        })

save_chunks_to_json(all_chunks, OUTPUT_FILE)
print(f"✅ Processed {len(all_chunks)} chunks from {len(glob.glob(DOCS_PATH))} documents.")
