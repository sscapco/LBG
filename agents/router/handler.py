import importlib, math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any
import yaml

from src.utils.config import Settings
from src.adapters.embeddings import get_embeddings

def _import_entrypoint(ep: str):
    mod, fn = ep.split(":", 1)
    return getattr(importlib.import_module(mod), fn)

@lru_cache(maxsize=1)
def _load_manifest() -> Dict[str, Any]:
    p = Path(__file__).with_name("manifest.yaml")
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def _count_hits(text: str, terms):
    if not terms: return 0
    t = (text or "").lower()
    return sum(1 for w in terms if w.lower() in t)

def _cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    return dot / (na * nb)

@lru_cache(maxsize=1)
def _descriptor_vec(desc: str):
    s = Settings()
    emb = get_embeddings(s)
    return emb.embed(desc)

def handle(session, message, services=None, config=None):
    man = _load_manifest()
    routing = man.get("routing", {})

    q = (message or {}).get("text", "") or ""
    s = Settings()
    emb = (services or {}).get("embedder") or get_embeddings(s)
    qv = emb.embed(q)

    desc = routing.get("descriptor_text", "")
    sim = _cosine(qv, _descriptor_vec(desc))

    kw = routing.get("keywords", {}) or {}
    kpts = 0.2 * _count_hits(q, kw.get("strong")) + 0.1 * _count_hits(q, kw.get("weak")) - 0.2 * _count_hits(q, kw.get("negatives"))

    th = routing.get("thresholds", {}) or {}
    sim_tau = float(th.get("similarity", 0.40))
    final_tau = float(th.get("final_score", 0.55))

    score = 0.6 * sim + kpts

    doi_ep = routing.get("doi_agent")
    rag_ep = routing.get("rag_agent")

    target_ep = doi_ep if (sim >= sim_tau and score >= final_tau) else rag_ep
    alert_txt = (
        f"Routed to DOI agent (score={score:.2f}, sim={sim:.2f})."
        if target_ep == doi_ep else
        f"Routed to generic RAG (score={score:.2f}, sim={sim:.2f})."
    )

    handler = _import_entrypoint(target_ep)
    env = handler(session=session, message=message, services=services, config=None)

    env.setdefault("alerts", []).append({"level": "info", "text": alert_txt})
    return env
