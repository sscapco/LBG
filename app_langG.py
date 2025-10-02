# app.py
# Streamlit chat UI for Governance Buddy (API-first with direct fallback)

# --- Env & paths (keep this tiny + robust) ---
import os, sys, json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
if os.getcwd() != str(ROOT):
    os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

USE_API = os.getenv("USE_API", "true").lower() == "true"
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))

import requests
import streamlit as st
import pandas as pd  # for optional envelope tables

# ---------------- Helpers ----------------
def call_backend(session_id: str, text: str, io_mode: str = "chat") -> dict:
    """Single entry: call API if USE_API, else call Python gateway directly."""
    if USE_API:
        try:
            r = requests.post(
                f"{API_BASE}/v1/act",
                json={"session_id": session_id, "io_mode": io_mode, "message": text},
                timeout=API_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {
                "version": "0.2",
                "display_text": "Sorry—backend API call failed.",
                "alerts": [{"level": "error", "text": f"API error: {type(e).__name__}: {e}"}],
            }
    else:
        try:
            from src.orchestration.graph_runtime import run_conversation
            return run_conversation(session_id=session_id, user_message=text, io_mode=io_mode)
        except Exception as e:
            return {
                "version": "0.2",
                "display_text": "Sorry—local backend call failed.",
                "alerts": [{"level": "error", "text": f"Local error: {type(e).__name__}: {e}"}],
            }

def list_agents() -> list[str]:
    """Show discovered agents (debug)."""
    if USE_API:
        try:
            r = requests.get(f"{API_BASE}/v1/agents", timeout=10)
            if r.ok:
                return r.json().get("agents", [])
        except Exception:
            return []
        return []
    else:
        try:
            from src.orchestration.registry import load_registry
            return [a.name for a in load_registry()]
        except Exception:
            return []

def render_envelope(envelope: dict, show_debug: bool = False):
    """Generic renderer for the universal envelope returned by agents."""
    env = envelope or {}

    # 1) Main text
    text = env.get("display_text", "")
    if text:
        st.markdown(text)

    # 2) Structured block (JSON, CSV, Markdown, etc.)
    structured = env.get("structured")
    if structured and isinstance(structured, dict):
        with st.expander("Structured output", expanded=False):
            content = structured.get("content", "")
            if isinstance(content, (dict, list)):
                st.json(content)
            else:
                try:
                    st.json(json.loads(content))
                except Exception:
                    st.code(str(content), language="json")

    # 3) Tables (optional)
    tables = env.get("tables", [])
    if tables:
        with st.expander("Tables", expanded=False):
            for t in tables:
                title = t.get("title", "Table")
                cols = t.get("columns", [])
                rows = t.get("rows", [])
                st.markdown(f"**{title}**")
                if rows:
                    st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True)
                else:
                    st.write("(empty)")

    # 4) Retrieved snippets
    snips = env.get("snippets", [])
    if snips:
        with st.expander(f"Retrieved snippets ({len(snips)})", expanded=False):
            for i, s in enumerate(snips, 1):
                loc = []
                if s.get("page") not in (None, ""):
                    loc.append(f"p.{s['page']}")
                if s.get("header_path"):
                    loc.append(s["header_path"])
                where = f" ({' | '.join(loc)})" if loc else ""
                st.markdown(f"**[{i}] {s.get('doc_id','')}{where}**")
                st.write(s.get("text", ""))

    # 5) Alerts & telemetry
    alerts = env.get("alerts", [])
    telem = env.get("telemetry", {})
    if alerts or (show_debug and telem):
        with st.expander("Alerts & Routing", expanded=False):
            for a in alerts:
                lvl = (a.get("level") or "info").lower()
                txt = a.get("text") or ""
                if lvl == "error":
                    st.error(txt)
                elif lvl == "warning":
                    st.warning(txt)
                else:
                    st.info(txt)
            if show_debug and telem:
                st.code(json.dumps(telem, indent=2), language="json")

# ------------- UI -------------
st.set_page_config(page_title="Governance Buddy (Agents)", page_icon="🤖", layout="wide")
st.title("🤖 Governance Buddy — Agent Chat")
st.caption("Generic chat UI. Agents decide what panels to show via the universal envelope.")
st.sidebar.image("assets/client_logo.png", use_container_width=True)
st.markdown("""
<style>
:root { --accent: #dff5e1; }
div[data-testid="stHeader"] { background: var(--accent); }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Governance Buddy")
    # st.image("assets/client_logo.png", use_column_width=True)  # optional
    st.caption("Routed via orchestrator to the best-fit agent.")

    show_debug = st.checkbox("Show routing debug", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = []

    # Debug info
    st.markdown("**Mode:** " + ("API" if USE_API else "Direct"))
    if USE_API:
        st.caption(f"API_BASE: {API_BASE}")
    try:
        st.write("Agents discovered:", list_agents())
    except Exception as e:
        st.error(f"Agent list error: {e}")

# Session
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"|"assistant","text":str,"envelope":dict|None}]

# Render history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["text"])
    else:
        with st.chat_message("assistant"):
            render_envelope(msg.get("envelope") or {"display_text": msg.get("text","")}, show_debug)

# Input
user_text = st.chat_input("Ask about DOI steps, evidence, naming, or anything in your docs…")
if user_text:
    st.session_state.messages.append({"role": "user", "text": user_text, "envelope": None})
    with st.chat_message("user"):
        st.markdown(user_text)

    envelope = call_backend(session_id="streamlit", text=user_text, io_mode="chat")

    st.session_state.messages.append({"role": "assistant", "text": "", "envelope": envelope})
    with st.chat_message("assistant"):
        render_envelope(envelope, show_debug)
