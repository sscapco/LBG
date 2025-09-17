from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF

def parse_pdf(path: Path) -> List[Dict]:
    """
    Parse PDF into layout blocks with type=paragraph, page number, and bbox.
    Later we can add table detection.
    """
    doc = fitz.open(path)
    blocks = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            text = " ".join(text.split())
            if not text:
                continue
            blocks.append({
                "type": "paragraph",
                "text": text,
                "page": page_num,
                "bbox": [x0, y0, x1, y1],
                "header_path": None  # will fill in later if we detect headings
            })
    return blocks

def clean_blocks(blocks: List[Dict]) -> List[Dict]:
    # Naive: drop any block text that appears on >80% of pages
    from collections import Counter
    page_count = len(set(b["page"] for b in blocks))
    text_counts = Counter(b["text"] for b in blocks)
    return [
        b for b in blocks
        if text_counts[b["text"]] / page_count <= 0.8
    ]

def split_blocks(blocks: List[Dict], chunk_size=700, overlap=75) -> List[Dict]:
    chunks = []
    chunk_id = 0
    for b in blocks:
        text = b["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append({
                "id": f"{chunk_id:05}",
                "chunk_type": b["type"],
                "text": chunk_text,
                "page": b["page"],
                "bbox": b["bbox"],
                "header_path": b["header_path"],
            })
            chunk_id += 1
            start += chunk_size - overlap
    return chunks

