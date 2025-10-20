# pages/02_Name_Checker_Demo.py
import re
from typing import Dict, Any, List
import streamlit as st

# --- direct agent import (adjust path if needed) ---
from agents.name_checker.handler import check_name_both

# ---------------- Setup ----------------
st.set_page_config(page_title="Name Checker Demo", page_icon="🧭", layout="wide")
st.title("Governance Assistance — Name Checker")
st.caption("Enter a Data Product name and type. We’ll run core checks, type-specific rules, LLM wording review, and show connection warnings.")

# ---------------- Inputs ----------------
with st.form("name_checker_form", clear_on_submit=False):
    c1, c2 = st.columns([3, 1])
    with c1:
        name = st.text_input("Data Product Name", placeholder="e.g., Party.Customer.VulnerableCustomers.AL17072")
    with c2:
        options = ["— Select type —", "ODP", "FDP", "CDP"]
        choice = st.selectbox("DP Type (required)", options, index=0)
        dp_type = None if choice == options[0] else choice
    submitted = st.form_submit_button("Run checks", type="primary")

# ---------------- Helpers ----------------
_ID_RE      = re.compile(r'^[A-Z]{2}\d{5}$')
_CAMEL_RE   = re.compile(r'^[A-Z][A-Za-z0-9]*$')
_ALLOWED_RE = re.compile(r'^[A-Za-z0-9.]+$')

def verdict_style(verdict: str) -> Dict[str, str]:
    v = (verdict or "").lower()
    if v == "valid": return {"emoji": "✅", "color": "success"}
    if v == "needs_changes": return {"emoji": "🟡", "color": "warning"}
    return {"emoji": "❌", "color": "error"}

def token_diff(original: str, suggested: str) -> str:
    o = [t for t in (original or "").split(".") if t]
    s = [t for t in (suggested or "").split(".") if t]
    if not s or len(o) != len(s):
        return f"`{original}`"
    parts = []
    for oi, si in zip(o, s):
        parts.append(f"**{si}**" if oi != si else si)
    return " · ".join(parts)

def outcome_cell(kind: str) -> str:
    k = (kind or "").lower()
    return {
        "pass": "✅ Pass",
        "fail": "❌ Fail",
        "warn": "⚠️ Warning",
        "info": "🛈 Info",
    }.get(k, "•")

# ---- Build Core (Universal) checks table deterministically (no LLM) ----
def core_checks(name: str, max_len: int = 75) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    s = name or ""
    tokens = [t for t in s.split(".") if t]

    checks = [
        ("Not empty", lambda: len(s) > 0, "Name must not be empty."),
        (f"Length ≤ {max_len}", lambda: len(s) <= max_len, f"Name exceeds max length of {max_len} (got {len(s)})."),
        ("Allowed characters (A–Z, a–z, 0–9, '.')", lambda: bool(_ALLOWED_RE.fullmatch(s)), "Only letters, digits, and '.' are allowed."),
        ("No leading/trailing '.'", lambda: not (s.startswith(".") or s.endswith(".")), "No leading or trailing '.' separators."),
        ("No consecutive '..'", lambda: ".." not in s, "No consecutive '.' separators (e.g., '..')."),
    ]
    for label, fn, reason in checks:
        ok = fn()
        rows.append({"Check": label, "Outcome": outcome_cell("pass" if ok else "fail"), "Reason": "" if ok else reason})

    # Token formatting (UpperCamelCase unless token is an ID)
    bad_tok = None
    for i, tok in enumerate(tokens, start=1):
        if _ID_RE.fullmatch(tok):
            continue
        if not _CAMEL_RE.fullmatch(tok):
            bad_tok = f"Token {i} ('{tok}') must be UpperCamelCase or match the ID pattern AA99999."
            break
    ok = bad_tok is None
    rows.append({
        "Check": "Tokens are UpperCamelCase (non-ID) or valid IDs",
        "Outcome": outcome_cell("pass" if ok else "fail"),
        "Reason": "" if ok else bad_tok
    })

    return rows

# ---- DP-specific table from agent's deterministic checks ----
def dp_specific_checks(resp: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    type_nonllm = resp.get("type_nonllm") or {}
    for c in type_nonllm.get("checks", []) or []:
        status = (c.get("status") or "info").lower()
        # normalize to pass/fail/info
        status = "pass" if status == "pass" else "fail" if status == "fail" else "info"
        label  = c.get("rule") or c.get("target") or "Check"
        reason = c.get("detail") or ""
        rows.append({"Check": label, "Outcome": outcome_cell(status), "Reason": reason})
    return rows

# ---- Warnings & connections table ----
def warnings_and_connections(resp: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    # LLM issues → warnings
    llm = resp.get("llm_review") or {}
    for i in llm.get("issues", []) or []:
        note = i.get("note") or i.get("type") or ""
        tok = i.get("token")
        if tok:
            note = f"{note} (token: `{tok}`)"
        rows.append({"Check": f"LLM: {i.get('type')}", "Outcome": outcome_cell("warn"), "Reason": note})

    # Connections → info
    conn = resp.get("connections") or {}
    for c in conn.get("connections_checks", []) or []:
        rows.append({
            "Check": f"{c.get('system')}: {c.get('check')}",
            "Outcome": outcome_cell("info"),
            "Reason": f"{c.get('status')}" + (f" — {c.get('action')}" if c.get("action") else "")
        })

    return rows

# ---------------- Run & Render ----------------
if submitted:
    if not name or not dp_type:
        st.error("Please enter a name and select a DP type.")
        st.stop()

    try:
        with st.spinner("Checking…"):
            resp = check_name_both(name=name, dp_type=dp_type)  # direct agent call
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

    # Verdict header
    overall = resp.get("overall", {}) or {}
    verdict = overall.get("verdict")
    style = verdict_style(verdict)

    st.subheader("Result")
    header = f"{style['emoji']} **Verdict:** `{verdict or 'unknown'}`"
    if overall.get("scientific_name"):
        header += f" · **Suggested:** `{overall['scientific_name']}`"
    getattr(st, style["color"])(header)

    # Name diff & edits
    with st.expander("Name diff & edits", expanded=True):
        input_name = resp.get("input_name") or name
        suggested = overall.get("scientific_name") or input_name
        st.markdown(f"**Input:** `{input_name}`")
        st.markdown(f"**Rendered:** {token_diff(input_name, suggested)}")

        edits = overall.get("edits") or []
        if edits:
            st.markdown("**Edits proposed:**")
            for e in edits:
                idx, frm, to, why = e.get("index"), e.get("from"), e.get("to"), e.get("reason")
                st.markdown(f"- Token #{idx}: `{frm}` → **`{to}`** _(reason: {why})_")
        else:
            st.caption("No edits proposed.")

    # Core checks table
    st.markdown("### Core checks")
    core_rows = core_checks(resp.get("input_name") or name)
    st.table(core_rows)

    # DP-specific checks table
    st.markdown("### DP-specific checks")
    dp_rows = dp_specific_checks(resp)
    if dp_rows:
        st.table(dp_rows)
    else:
        st.caption("No type-specific checks reported.")

    # Warnings & connections table
    st.markdown("### Warnings & connections")
    wc_rows = warnings_and_connections(resp)
    if wc_rows:
        st.table(wc_rows)
    else:
        st.caption("No warnings or connection notes reported.")

    # Friendly explanation (end)
    explanation = (
        overall.get("explanation")
        or (resp.get("llm_review") or {}).get("llm_explnation")
        or (resp.get("llm_review_generic") or {}).get("llm_explnation")
        or ""
    )
    if explanation:
        st.markdown("### Explanation")
        st.write(explanation)

    # Raw response (debug)
    with st.expander("Raw response"):
        st.json(resp)
