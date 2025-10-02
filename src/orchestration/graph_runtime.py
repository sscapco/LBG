"""
LangGraph orchestration v3.1 (dynamic, N agents; router excluded from scoring):
Start → Route (score all chat-capable agents from registry) → Agent → End
"""

from typing import Dict, Any, List, Optional
from functools import lru_cache
import importlib
import math
import time

from langgraph.graph import StateGraph, END

from .state import ChatState
from src.utils.config import Settings
from src.adapters.embeddings import get_embeddings
from src.orchestration.registry import load_registry  # discovers agents/*/manifest.yaml


# ----------------------------- helpers & caches -----------------------------

def _import_entrypoint(ep: str):
    if not ep or ":" not in ep:
        raise RuntimeError(f"Invalid entrypoint: {ep}")
    mod, fn = ep.split(":", 1)
    return getattr(importlib.import_module(mod), fn)

def _count_hits(text: str, terms: Optional[List[str]]) -> int:
    if not terms:
        return 0
    t = (text or "").lower()
    return sum(1 for w in terms if w and w.lower() in t)

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    return dot / (na * nb)

@lru_cache(maxsize=256)
def _descriptor_vec_cache(agent_name: str, descriptor_text: str, embedder_key: str) -> List[float]:
    s = Settings()
    emb = get_embeddings(s)
    return emb.embed(descriptor_text or agent_name)


# --------------------------------- nodes -------------------------------------

def _route_node(state: ChatState) -> ChatState:
    t0 = time.perf_counter()

    # 1) Discover candidates (filter by io_mode AND exclude the router itself)
    io_mode = (state.get("io_mode") or "chat").lower()
    all_agents = load_registry()

    def _is_candidate(a) -> bool:
        if io_mode not in (m.lower() for m in a.io_modes):
            return False
        # exclude the router agent by name or label
        if a.name == "governance_router":
            return False
        if (a.labels or {}).get("skill", "").lower() == "router":
            return False
        return True

    candidates = [a for a in all_agents if _is_candidate(a)]
    if not candidates:
        raise RuntimeError(f"No non-router agents support io_mode={io_mode}. Add agents or adjust manifests.")

    # 2) Embed the query once
    q = (state.get("user_message") or "").strip()
    s = Settings()
    emb = get_embeddings(s)
    qv = emb.embed(q)
    embedder_key = f"{s.embedding_provider}|{getattr(s, 'embedding_model', '')}"

    # 3) Score each candidate
    scored: List[Dict[str, Any]] = []
    for a in candidates:
        desc = (a.routing.descriptor_text or a.name).strip()
        dv = _descriptor_vec_cache(a.name, desc, embedder_key)
        sim = _cosine(qv, dv) if dv else 0.0

        kpts = (
            0.2 * _count_hits(q, a.routing.keywords_strong) +
            0.1 * _count_hits(q, a.routing.keywords_weak) -
            0.2 * _count_hits(q, a.routing.keywords_neg)
        )
        score = 0.75 * sim + kpts

        thr = a.routing.thresholds or {}
        route_thr = thr.get("route", thr.get("final_score", 0.50))
        margin_thr = thr.get("margin", 0.05)

        scored.append({
            "name": a.name,
            "entrypoint": a.entrypoint,
            "score": float(score),
            "similarity": float(sim),
            "kpoints": float(kpts),
            "route_threshold": float(route_thr),
            "margin_threshold": float(margin_thr),
        })

    scored.sort(key=lambda r: r["score"], reverse=True)
    top1 = scored[0]
    top2 = scored[1] if len(scored) > 1 else None
    margin = top1["score"] - (top2["score"] if top2 else 0.0)

    # 4) Choose winner or fallback
    chosen = top1
    routed_name = top1["name"]
    alert_txt = f"Routed to {routed_name} (score={top1['score']:.2f}, sim={top1['similarity']:.2f}, margin={margin:.2f})."

    # Try explicit fallback by convention 'rag_default' if low confidence
    if not (top1["score"] >= top1["route_threshold"] and margin >= top1["margin_threshold"]):
        fb = next((r for r in scored if r["name"] == "rag_default"), None)
        if fb:
            chosen = fb
            routed_name = fb["name"]
            alert_txt = f"Routed to {routed_name} (fallback; top1 score={top1['score']:.2f}, margin={margin:.2f})."
        else:
            alert_txt = f"Routed to {routed_name} (low confidence; score={top1['score']:.2f}, margin={margin:.2f})."

    route_ms = int((time.perf_counter() - t0) * 1000)

    return {
        **state,
        "route_to": routed_name,
        "active_agent_ep": chosen["entrypoint"],
        "route_score": chosen["score"],
        "route_sim": chosen["similarity"],
        "routing_debug": {
            "alert": alert_txt,
            "topk": scored[:5],
            "route_ms": route_ms,
        },
    }


def _agent_node(state: ChatState) -> ChatState:
    t0 = time.perf_counter()

    active_ep = state.get("active_agent_ep")
    if not active_ep:
        # Last-resort fallback: try 'rag_default' from registry
        reg = load_registry()
        rag = next((a for a in reg if a.name == "rag_default"), None)
        if not rag:
            raise RuntimeError("No active agent entrypoint and no 'rag_default' agent found.")
        active_ep = rag.entrypoint

    handler = _import_entrypoint(active_ep)
    session = {"session_id": state.get("session_id", "default"), "chat_history": []}
    message = {"text": state.get("user_message", "")}

    env = handler(session=session, message=message, services=None, config=None)

    # Append routing alert & telemetry
    alert_txt = (state.get("routing_debug") or {}).get("alert")
    if alert_txt:
        env.setdefault("alerts", []).append({"level": "info", "text": alert_txt})

    env.setdefault("telemetry", {}).update({
        "active_agent_ep": active_ep,
        "route_to": state.get("route_to"),
        "route_score": state.get("route_score"),
        "route_sim": state.get("route_sim"),
        "agent_ms": int((time.perf_counter() - t0) * 1000),
        "route_ms": (state.get("routing_debug") or {}).get("route_ms"),
    })

    return {**state, "envelope": env}


# ------------------------------- graph compile -------------------------------

_graph = StateGraph(ChatState)
_graph.add_node("route", _route_node)
_graph.add_node("agent", _agent_node)

_graph.set_entry_point("route")
# No conditional branches needed now; always proceed to 'agent'
_graph.add_edge("route", "agent")
_graph.add_edge("agent", END)

_APP = _graph.compile()


# ------------------------------- public gateway ------------------------------

def run_conversation(session_id: str, user_message: str, io_mode: str = "chat") -> Dict[str, Any]:
    init: ChatState = {
        "session_id": session_id,
        "io_mode": io_mode,
        "user_message": user_message,
    }
    final_state: ChatState = _APP.invoke(init)
    return final_state.get("envelope", {
        "version": "0.2",
        "display_text": "No response.",
        "alerts": [{"level": "error", "text": "No envelope returned from orchestration."}],
    })
