import hashlib, datetime
from typing import List, Dict
from pathlib import Path

def build_doc_record(source_path: Path, blocks: List[Dict]) -> Dict:
    sha = hashlib.sha256("\n".join(b["text"] for b in blocks).encode()).hexdigest()
    return {
        "doc_id": sha[:16],
        "source_url": f"file://{source_path.resolve()}",
        "source_type": "pdf_local",
        "title": source_path.stem,
        "version": f"mtime:{datetime.datetime.fromtimestamp(source_path.stat().st_mtime).isoformat()}",
        "sha256": sha,
        "language": "en",
        "tags": [],
        "pages": len(set(b["page"] for b in blocks)),
        "table_count": 0,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "status": "active"
    }

def build_chunk_records(doc_id: str, chunks: List[Dict]) -> List[Dict]:
    return [
        {
            "id": f"{doc_id}#{c['id']}",
            "doc_id": doc_id,
            "chunk_type": c["chunk_type"],
            "text": c["text"],
            "table_md": None,
            "table_rows": None,
            "vector": None,  # fill when embedding
            "section_title": None,
            "header_path": c["header_path"],
            "page": c["page"],
            "bbox": c["bbox"],
            "score_hint": None,
            "meta": {},
            "created_at": datetime.datetime.now().isoformat()
        }
        for c in chunks
    ]
