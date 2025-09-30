#!/usr/bin/env python3
"""
Script to process Chapter 2 and Chapter 3 documents and create chunk files.
This script will process all .docx files in the data folder and create
separate chunk files for each chapter.
"""

import os
import json
import re
from typing import List, Dict

def _parse_extracted_text_to_episodes(raw_text: str, chapter_number: int) -> List[Dict[str, object]]:
    """Parse the extracted text file into a list of episode dicts matching Chapter 1 JSON format.

    Expected separators (as seen in chapter2_extracted_text.txt):
      - Lines of '=' of length 80 surrounding: FILE: cX-epY.docx
      - Within each file section, a line starting with '3. Main Content:' followed by the content.
    """
    # Split into sections per file using the same strategy as elsewhere in the codebase
    parts = re.split(r"={80}\nFILE: (.+?)\.docx\n={80}\n", raw_text)
    # parts pattern: [prefix_text, file_name_1, file_content_1, file_name_2, file_content_2, ...]
    episodes: List[Dict[str, object]] = []
    for i in range(1, len(parts), 2):
        file_name = parts[i]
        file_content = parts[i + 1] if i + 1 < len(parts) else ""

        # Extract episode number from file name pattern like c2-ep10
        m = re.search(r"ep(\d+)", file_name)
        if not m:
            continue
        episode_num = int(m.group(1))

        # Attempt to isolate '3. Main Content:' onwards up to the next line of '*' (section divider)
        main_content = file_content
        mc_match = re.search(r"3\.\s*Main Content:\s*(.*)", file_content, flags=re.DOTALL)
        if mc_match:
            main_content = mc_match.group(1)
            # Trim trailing asterisks divider if present
            star_split = re.split(r"\n\*{80}\n", main_content)
            if star_split:
                main_content = star_split[0]

        # Clean leading/trailing whitespace
        main_content = main_content.strip()
        if not main_content:
            continue

        episodes.append({
            "chapter": chapter_number,
            "episode": episode_num,
            "content": main_content
        })

    # Sort by episode number to keep a consistent order
    episodes.sort(key=lambda e: int(e.get("episode", 0)))
    return episodes


def process_chapter_documents():
    """Create chapter2 and chapter3 episode JSON files from extracted text files, stored under app1/data."""
    base_dir = os.path.join("app1", "data")
    if not os.path.exists(base_dir):
        print(f"❌ Data directory '{base_dir}' not found!")
        return

    targets = [
        {"txt": "chapter2_extracted_text.txt", "json": "chapter2_episodes.json", "chapter": 2},
        {"txt": "chapter3_extracted_text.txt", "json": "chapter3_episodes.json", "chapter": 3},
    ]

    for target in targets:
        txt_path = os.path.join(base_dir, target["txt"])
        out_path = os.path.join(base_dir, target["json"])
        chapter_num = target["chapter"]

        if not os.path.exists(txt_path):
            print(f"⚠️  Skipping: '{txt_path}' not found")
            continue

        print(f"🔎 Reading extracted text: {txt_path}")
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        print(f"🧩 Parsing episodes for chapter {chapter_num}...")
        episodes = _parse_extracted_text_to_episodes(raw_text, chapter_num)

        if not episodes:
            print(f"⚠️  No episodes parsed for chapter {chapter_num} from {target['txt']}")
            continue

        print(f"💾 Writing {len(episodes)} episodes -> {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(episodes, f, indent=2, ensure_ascii=False)

    print("\n✅ Processing complete! Generated JSON files are stored in 'app1/data/'.")

if __name__ == "__main__":
    process_chapter_documents()
