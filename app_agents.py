import os
import importlib
import glob
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import yaml

# ----------------------------------------
# Config: pick agent from backend (no UI)
# ----------------------------------------
# Default to 'doi_step_navigator' but allow override via env var
TARGET_AGENT_NAME = os.getenv("AGENT_NAME", "doi_steps").lower()
AGENTS_ROOT = os.getenv("AGENTS_ROOT", "agents")
LOGO_PATH = os.getenv("DEMO_LOGO", "assets/client_logo.png")

# -------- Registry / Loading --------
def load_registry(agents_root: str = "agents") -> List[Dict[str, Any]]:
    items = []
    for path in glob.glob(str(Path(agents_root) / "*" / "manifest.yaml")):
        p = Path(path)
        man = yaml.safe_load(p.read_text(encoding="utf-8"))
        name = (man.get("identity") or {}).get("name") or p.parent.name
        entrypoint = man.get("entrypoint")  # e.g., agents.doi_steps.handler:handle_chat
        items.append({
            "name": name,
            "manifest": man,
            "manifest_dir": str(p.parent),
            "entrypoint": entrypoint,
            "path": str(p),
        })
    items.sort(key=lambda x: x["name"].lower())
    return items


def import_handler(entrypoint: str):
    # "package.module:func" -> callable
    if not entrypoint or ":" not in entrypoint:
        raise RuntimeError("Invalid or missing entrypoint in manifest.")
    mod_name, func_name = entrypoint.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    return fn


def resolve_agent(registry: List[Dict[str, Any]], target_name: str):
    tx = (target_name or "").lower()
    for it in registry:
        names = {it["name"].lower()}
        # Also allow matching on folder name and identity.name (if differs)
        names.add(Path(it["manifest_dir"]).name.lower())
        ident_name = ((it.get("manifest") or {}).get("identity") or {}).get("name")
        if ident_name:
            names.add(ident_name.lower())
        if tx in names:
            return it
    return None

# -------- UI Setup --------
st.set_page_config(page_title="Governance Assistant", page_icon="🤖", layout="wide")
st.title("Governance Assistance")

# Sidebar: logo only (no agent picker)
with st.sidebar:
    try:
        st.image(LOGO_PATH)
    except Exception:
        st.markdown("**Governance**")

# Load registry and select agent from backend
registry = load_registry(AGENTS_ROOT)
if not registry:
    st.error("No agents found. Place manifests at agents/*/manifest.yaml")
    st.stop()

agent = resolve_agent(registry, TARGET_AGENT_NAME)
if not agent:
    st.error(f"Agent '{TARGET_AGENT_NAME}' not found.")
    st.caption("Available agents: " + ", ".join([x["name"] for x in registry]))
    st.stop()

# Chat state
if "turns" not in st.session_state:
    st.session_state.turns = []  # each: {user:str, envelope:dict}

# -------- Render history --------
for t in st.session_state.turns:
    with st.chat_message("user"):
        st.write(t["user"])

    env = t.get("envelope") or {}
    with st.chat_message("assistant"):
        # 1) main text
        st.markdown(env.get("display_text", ""))

        # 2) structured panel (optional)
        if "structured" in env:
            with st.expander("Structured output", expanded=False):
                mime = (env["structured"] or {}).get("mime", "text/plain")
                content = (env["structured"] or {}).get("content", "")
                if mime == "application/json":
                    st.code(content, language="json")
                elif mime in ("text/markdown", "text/x-markdown"):
                    st.markdown(content)
                else:
                    st.text(content)

        # 3) retrieved snippets panel (optional)
        if env.get("snippets"):
            with st.expander("Retrieved snippets", expanded=False):
                for sn in env["snippets"]:
                    where_bits = []
                    if sn.get("page") not in (None, ""):
                        where_bits.append(f"p.{sn['page']}")
                    if sn.get("header_path"):
                        where_bits.append(sn["header_path"])
                    where = " | ".join(where_bits)
                    st.markdown(f"**[{sn.get('id', '?')}]** {sn.get('doc_id','')}  _{where}_")
                    st.write(sn.get("text", ""))

        # 4) alerts (optional)
        for alert in env.get("alerts", []):
            lvl = (alert.get("level") or "info").lower()
            msg = alert.get("text", "")
            if lvl == "error":
                st.error(msg)
            elif lvl in ("warn", "warning"):
                st.warning(msg)
            else:
                st.info(msg)

        # 5) follow-ups (optional)
        if env.get("followups"):
            st.markdown("**Follow-ups:**")
            cols = st.columns(min(3, len(env["followups"])))
            for i, f in enumerate(env["followups"]):
                with cols[i % len(cols)]:
                    if st.button(f"➡️ {f}", key=f"fu-{i}-{t['user'][:8]}"):
                        st.session_state.next_prefill = f

# -------- Input --------
prefill = st.session_state.pop("next_prefill", None) if "next_prefill" in st.session_state else None
q = st.chat_input("Type your message…")

if q:
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                handler = import_handler(agent["entrypoint"])
                # Minimal session object; pass history if you want
                session = {"session_id": "streamlit", "chat_history": []}
                message = {"text": q}
                envelope = handler(session=session, message=message)
            except Exception as e:
                st.error(str(e))
                st.stop()

        # Render the current assistant turn (same renderer as history)
        st.markdown(envelope.get("display_text", ""))

        if "structured" in envelope:
            with st.expander("Structured output", expanded=False):
                mime = (envelope["structured"] or {}).get("mime", "text/plain")
                content = (envelope["structured"] or {}).get("content", "")
                if mime == "application/json":
                    st.code(content, language="json")
                elif mime in ("text/markdown", "text/x-markdown"):
                    st.markdown(content)
                else:
                    st.text(content)

        if envelope.get("snippets"):
            with st.expander("Retrieved snippets", expanded=False):
                for sn in envelope["snippets"]:
                    where_bits = []
                    if sn.get("page") not in (None, ""):
                        where_bits.append(f"p.{sn['page']}")
                    if sn.get("header_path"):
                        where_bits.append(sn["header_path"])
                    where = " | ".join(where_bits)
                    st.markdown(f"**[{sn.get('id','?')}]** {sn.get('doc_id','')}  _{where}_")
                    st.write(sn.get("text", ""))

        for alert in envelope.get("alerts", []):
            lvl = (alert.get("level") or "info").lower()
            msg = alert.get("text", "")
            if lvl == "error":
                st.error(msg)
            elif lvl in ("warn", "warning"):
                st.warning(msg)
            else:
                st.info(msg)

        if envelope.get("followups"):
            st.markdown("**Follow-ups:**")
            cols = st.columns(min(3, len(envelope["followups"])))
            for i, f in enumerate(envelope["followups"]):
                with cols[i % len(envelope["followups"])]:
                    if st.button(f"➡️ {f}", key=f"fu-live-{i}"):
                        st.session_state.next_prefill = f

    # Persist turn
    st.session_state.turns.append({
        "user": q,
        "envelope": envelope,
    })
