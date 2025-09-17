# src/orchestration/registry.py
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

from src.utils.config import Settings
from src.adapters.embeddings import get_embeddings

AGENTS_ROOT = Path("agents")

@dataclass
class AgentRouting:
    descriptor_text: str = ""
    keywords_strong: List[str] = field(default_factory=list)
    keywords_weak: List[str] = field(default_factory=list)
    keywords_neg: List[str] = field(default_factory=list)
    thresholds: Dict[str, Any] = field(default_factory=dict)  # supports {route,margin} or {similarity,final_score}
    affinity: Dict[str, Any] = field(default_factory=dict)    # optional: {titles:[], corpora:[]}

@dataclass
class AgentMeta:
    name: str
    entrypoint: str
    io_modes: List[str]
    labels: Dict[str, str]
    tools_allow: List[str]
    routing: AgentRouting
    manifest_path: Path

def _load_yaml(p: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to read YAML at {p}: {e}")

def _discover_manifests() -> List[Path]:
    return [m for m in AGENTS_ROOT.glob("*/manifest.yaml") if m.is_file()]

def _to_meta(man: Dict[str, Any], manifest_path: Path) -> Optional[AgentMeta]:
    ident = man.get("identity", {}) or {}
    entrypoint = man.get("entrypoint")
    io_block = man.get("io", {}) or {}
    modes = io_block.get("modes") or ([io_block.get("mode")] if io_block.get("mode") else [])

    # routing block
    r = man.get("routing", {}) or {}
    kw = r.get("keywords", {}) or {}
    routing = AgentRouting(
        descriptor_text=r.get("descriptor_text", "") or "",
        keywords_strong=list(kw.get("strong", []) or []),
        keywords_weak=list(kw.get("weak", []) or []),
        keywords_neg=list(kw.get("negatives", []) or []),
        thresholds=r.get("thresholds", {}) or {},
        affinity=r.get("affinity", {}) or {},
    )

    # minimal validation
    if not ident.get("name") or not entrypoint or not modes:
        return None

    tools_allow = list((man.get("tools") or {}).get("allow", []) or [])
    labels = man.get("labels", {}) or {}

    return AgentMeta(
        name=ident["name"],
        entrypoint=entrypoint,
        io_modes=modes,
        labels=labels,
        tools_allow=tools_allow,
        routing=routing,
        manifest_path=manifest_path
    )

def load_registry() -> List[AgentMeta]:
    metas: List[AgentMeta] = []
    for mpath in _discover_manifests():
        man = _load_yaml(mpath)
        meta = _to_meta(man, mpath)
        if meta:
            metas.append(meta)
    return metas

# ---------- simple scoring helpers (for CLI probe) ----------

def _count_hits(text: str, terms: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for w in terms if w.lower() in t)

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(x*x for x in b)) or 1.0
    return dot / (na * nb)

def _descriptor_vec(desc: str, emb) -> List[float]:
    if not desc.strip():
        return []
    return emb.embed(desc)

def probe_query(query: str, registry: List[AgentMeta]) -> List[Dict[str, Any]]:
    """Return a scored list for quick manual validation (no retrieval affinity here)."""
    s = Settings()
    emb = get_embeddings(s)
    qv = emb.embed(query)

    rows = []
    for meta in registry:
        desc = meta.routing.descriptor_text or meta.name
        dv = _descriptor_vec(desc, emb)
        sim = _cosine(qv, dv) if dv else 0.0
        kpts = (
            0.2 * _count_hits(query, meta.routing.keywords_strong) +
            0.1 * _count_hits(query, meta.routing.keywords_weak) -
            0.2 * _count_hits(query, meta.routing.keywords_neg)
        )
        score = 0.6 * sim + kpts

        # support both styles of thresholds
        thr = meta.routing.thresholds or {}
        route_thr = thr.get("route")
        if route_thr is None:
            route_thr = thr.get("final_score", 0.5)

        rows.append({
            "agent": meta.name,
            "entrypoint": meta.entrypoint,
            "io_modes": meta.io_modes,
            "score": round(score, 4),
            "similarity": round(sim, 4),
            "kpoints": round(kpts, 4),
            "route_threshold": route_thr
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows

# ------------------------- CLI for quick checks -------------------------

def _cmd_list(args):
    reg = load_registry()
    print(f"Found {len(reg)} agents\n")
    for a in reg:
        print(f"- {a.name:18s}  io={a.io_modes}  entry={a.entrypoint}")
    return 0

def _cmd_probe(args):
    reg = load_registry()
    if args.io:
        reg = [a for a in reg if args.io in a.io_modes]
    rows = probe_query(args.query, reg)
    top = rows[: args.top]
    print(json.dumps(top, indent=2))
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Agent Registry utility")
    sub = ap.add_subparsers()

    ap_list = sub.add_parser("list", help="List discovered agents")
    ap_list.set_defaults(func=_cmd_list)

    ap_probe = sub.add_parser("probe", help="Score a query against agents")
    ap_probe.add_argument("query", type=str)
    ap_probe.add_argument("--io", type=str, default=None, help="Filter by io mode (e.g., chat)")
    ap_probe.add_argument("--top", type=int, default=5)
    ap_probe.set_defaults(func=_cmd_probe)

    args = ap.parse_args()
    if hasattr(args, "func"):
        raise SystemExit(args.func(args))
    ap.print_help()
