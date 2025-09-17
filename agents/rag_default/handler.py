# agents/rag_default/handler.py
import lancedb
from pathlib import Path
from typing import Dict, Any, List

from src.utils.config import Settings
from src.adapters.llms import get_llm
from src.adapters.embeddings import get_embeddings

def _dedup(rows, keylen=200):
    seen, out = set(), []
    for r in rows:
        txt = r.get("text") or r.get("chunk") or r.get("content") or ""
        key = ((r.get("doc_id") or "") + "|" + (r.get("header_path") or "") + "|" + txt[:keylen]).strip()
        if key not in seen:
            seen.add(key); out.append(r)
    return out

def _format_context(snips: List[Dict[str,Any]]):
    blocks = []
    for i, s in enumerate(snips, 1):
        loc = []
        if s.get("page") not in (None, ""): loc.append(f"p.{s['page']}")
        if s.get("header_path"): loc.append(s["header_path"])
        where = f" ({' | '.join(loc)})" if loc else ""
        text = s.get("text") or s.get("chunk") or s.get("content") or ""
        blocks.append(f"[{i}] {s.get('doc_id','')}{where}\n\n{text}")
    return "\n\n---\n\n".join(blocks)

def handle_chat(session, message, services=None, config=None):
    s = Settings()
    llm = (services or {}).get("llm") or get_llm(s)
    embedder = (services or {}).get("embedder") or get_embeddings(s)

    db = lancedb.connect(s.store_path)
    if "chunks" not in db.table_names():
        raise RuntimeError(f"No 'chunks' table in {s.store_path}. Ingest first.")
    tbl = db.open_table("chunks")

    q = (message or {}).get("text","")
    qvec = embedder.embed(q)
    rows = tbl.search(qvec).metric("cosine").limit(6).to_pandas().to_dict("records")
    rows = _dedup(rows)
    ctx = _format_context(rows)

    prompt = Path(__file__).with_name("rag_default.txt").read_text(encoding="utf-8")
    full = f"{prompt}\n\nContext:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:\n"
    answer = llm.generate(full, temperature=0.2, max_tokens=700)

    snippets = [{
        "id": i+1, "rank": i+1, "doc_id": r.get("doc_id"),
        "title": r.get("doc_title") or "", "page": r.get("page"),
        "header_path": r.get("header_path"), "source_url": r.get("source_url"),
        "text": r.get("text") or r.get("chunk") or r.get("content") or ""
    } for i, r in enumerate(rows)]

    return {
        "version": "0.2",
        "display_text": answer.strip(),
        "snippets": snippets
    }
