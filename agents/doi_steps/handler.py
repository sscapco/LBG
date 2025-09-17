# agents/doi_steps/handler.py
# DOI Steps Agent — chat-only. Universal envelope out (display_text + optional structured/snippets/citations).

import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import lancedb

from src.utils.config import Settings
from src.adapters.llms import get_llm
from src.adapters.embeddings import get_embeddings

# ---------- regex helpers ----------
JSON_FENCE_RE = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.I)
JSON_TAIL_RE  = re.compile(r"\n*\s*(\{[\s\S]*\})\s*$")


def split_answer_and_json(answer: str, mode: str = "json_fenced"):
    """Return (clean_markdown, json_blob or "")."""
    if mode == "json_fenced":
        m = JSON_FENCE_RE.search(answer or "")
        if m:
            js = m.group(1)
            cleaned = JSON_FENCE_RE.sub("", answer).strip()
            return cleaned, js
        m2 = JSON_TAIL_RE.search(answer or "")
        if m2:
            js = m2.group(1)
            cleaned = (answer or "")[:m2.start()].rstrip()
            return cleaned, js
    return (answer or "").strip(), ""


def parse_citation_numbers(markdown: str) -> List[int]:
    """Extract [1], [2], ... as ints (unique, sorted)."""
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", markdown or "") if n.isdigit()})


def dedup_keep_order(rows: List[Dict[str, Any]], keylen: int = 200):
    """Deduplicate retrieved rows by doc_id|header_path|text prefix, preserving order."""
    seen, out = set(), []
    for r in rows:
        txt = r.get("text") or r.get("chunk") or r.get("content") or ""
        key = ((r.get("doc_id") or "") + "|" + (r.get("header_path") or "") + "|" + txt[:keylen]).strip()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def format_numbered_context(snips: List[Dict[str, Any]]) -> str:
    """Render retrieved rows as numbered blocks the prompt can cite: [1]..[k]."""
    blocks = []
    for i, s in enumerate(snips, 1):
        loc = []
        pg = s.get("page")
        if pg not in (None, ""):
            loc.append(f"p.{pg}")
        if s.get("header_path"):
            loc.append(s["header_path"])
        where = f" ({' | '.join(loc)})" if loc else ""
        doc_id = s.get("doc_id") or ""
        text = s.get("text") or s.get("chunk") or s.get("content") or ""
        blocks.append(f"[{i}] {doc_id}{where}\n\n{text}")
    return "\n\n---\n\n".join(blocks)


def _open_chunks_table(store_path: str):
    """Open LanceDB 'chunks' table; raise if missing."""
    db = lancedb.connect(store_path)
    if "chunks" not in db.table_names():
        raise RuntimeError(f"No 'chunks' table in {store_path}. Ingest first.")
    return db.open_table("chunks")


def _run_core(user_text: str, k: int = 6) -> Dict[str, Any]:
    """
    Core pipeline shared by this agent:
      - retrieval
      - prompt build
      - LLM generate
      - parse to (text, json, snippets, cited_ids)
    """
    settings = Settings()
    llm = get_llm(settings)
    embedder = get_embeddings(settings)
    tbl = _open_chunks_table(settings.store_path)

    qvec = embedder.embed(user_text)
    rows = tbl.search(qvec).metric("cosine").limit(k).to_pandas().to_dict("records")
    rows = dedup_keep_order(rows)
    ctx = format_numbered_context(rows)

    prompt_path = Path(__file__).with_name("doi_checklist.txt")
    base_prompt = prompt_path.read_text(encoding="utf-8")

    full_prompt = (
        f"{base_prompt}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question:\n{user_text}\n\n"
        f"Answer:\n"
    )

    answer = llm.generate(full_prompt, temperature=0.2, max_tokens=900)
    text_clean, json_blob = split_answer_and_json(answer, mode="json_fenced")
    cited_nums = parse_citation_numbers(text_clean)

    snippets = [{
        "id": i + 1,
        "rank": i + 1,
        "doc_id": r.get("doc_id"),
        "title": r.get("doc_title") or "",
        "page": r.get("page"),
        "header_path": r.get("header_path"),
        "source_url": r.get("source_url"),
        "text": r.get("text") or r.get("chunk") or r.get("content") or "",
    } for i, r in enumerate(rows)]

    return {
        "text_clean": text_clean,
        "json_blob": json_blob,
        "snippets": snippets,
        "cited_nums": cited_nums,
    }


def _present_chat(core: Dict[str, Any]) -> Dict[str, Any]:
    """Render universal envelope for chat."""
    env = {
        "version": "0.2",
        "display_text": core["text_clean"],
        "snippets": core["snippets"],
        "citations": [{"snippet_id": n} for n in core["cited_nums"]],
    }
    if core["json_blob"]:
        env["structured"] = {"mime": "application/json", "content": core["json_blob"]}
    return env


def handle(
    session: Dict[str, Any],
    message: Dict[str, Any],
    services: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Public entrypoint (chat-only). Returns a universal envelope:
      {
        "version": "0.2",
        "display_text": "...",
        "snippets": [...],
        "structured": {"mime":"application/json","content":"..."},
        "citations": [{"snippet_id": n}],
        "alerts": [...]
      }
    """
    user_text = (message or {}).get("text", "")
    io_mode = (config or {}).get("io_mode") or kwargs.get("io_mode") or "chat"

    # Enforce chat-only for this agent
    force_chat = io_mode != "chat"

    core = _run_core(user_text)
    env = _present_chat(core)

    if force_chat:
        env.setdefault("alerts", []).append({
            "level": "info",
            "text": "This agent supports chat only; rendered in chat mode."
        })
    return env
