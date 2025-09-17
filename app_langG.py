# app.py
# Streamlit chat UI for Governance Buddy (LangGraph orchestration)

# --- Ensure Streamlit runs like CLI (cwd + sys.path + .env) ---
import os, sys
from pathlib import Path
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # optional

ROOT = Path(__file__).resolve().parent
if os.getcwd() != str(ROOT):
    os.chdir(ROOT)               # make CWD the project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if load_dotenv is not None:
    load_dotenv(override=False)  # load .env like your CLI tests
# ----------------------------------------------------------------

import json
import streamlit as st
import pandas as pd  # used for optional envelope tables

# Orchestrator (LangGraph)
from src.orchestration.graph_runtime import run_conversation

# Optional debug: show discovered agents + settings
from src.orchestration.registry import load_registry
from src.utils.config import Settings

# -------- Page setup --------
st.set_page_config(page_title="Governance Buddy", page_icon="✅", layout="wide")
st.markdown("""
<style>
:root { --accent: #dff5e1; }
div[data-testid="stHeader"] { background: var(--accent); }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# -------- Sidebar --------
with st.sidebar:
    st.title("Governance Buddy")
    # If you have a PNG logo, uncomment and point to it:
    # st.image("assets/client_logo.png", use_column_width=True)
    st.caption("Questions are routed to the best-fit agent via LangGraph.")

    show_debug = st.checkbox("Show routing debug", value=False)

    # Quick visibility into what Streamlit sees
    try:
        agents_seen = [a.name for a in load_registry()]
        st.write("Agents discovered:", agents_seen)
    except Exception as e:
        st.error(f"Registry error: {e}")

    s = Settings()
    st.code({
        "llm_provider": s.llm_provider,
        "embedding_provider": s.embedding_provider,
        "embedding_model": getattr(s, "embedding_model", None),
        "store_path": getattr(s, "store_path", None),
    })

# -------- Session state --------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "text": str, "envelope": dict|None}

# -------- Helpers --------
def render_envelope(envelope: dict, show_debug: bool = False):
    """Generic renderer for the universal envelope returned by agents."""
    # 1) Main text
    text = (envelope or {}).get("display_text", "")
    if text:
        st.markdown(text)

    # 2) Structured payload (e.g., fenced JSON)
    structured = (envelope or {}).get("structured")
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

    # 3) Optional tables
    tables = (envelope or {}).get("tables", [])
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

    # 4) Retrieved snippets (persist across history)
    snips = (envelope or {}).get("snippets", [])
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
                st.write(s.get("text",""))

    # 5) Alerts & telemetry
    alerts = (envelope or {}).get("alerts", [])
    telem  = (envelope or {}).get("telemetry", {})
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

# -------- Render chat history --------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["text"])
    else:
        with st.chat_message("assistant"):
            render_envelope(msg.get("envelope") or {"display_text": msg.get("text","")}, show_debug)

# -------- Input box --------
user_text = st.chat_input("Ask about DOI steps, evidence, or anything in your docs…")
if user_text:
    # show user message
    st.session_state.messages.append({"role": "user", "text": user_text, "envelope": None})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Orchestration call
    try:
        envelope = run_conversation(session_id="streamlit", user_message=user_text, io_mode="chat")
    except Exception as e:
        envelope = {
            "display_text": "Sorry—something went wrong running the orchestrator.",
            "alerts": [{"level": "error", "text": f"{type(e).__name__}: {e}"}],
        }

    # render assistant message
    st.session_state.messages.append({"role": "assistant", "text": "", "envelope": envelope})
    with st.chat_message("assistant"):
        render_envelope(envelope, show_debug)
