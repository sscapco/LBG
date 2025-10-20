import re
import os
import json
from src.utils.config import Settings
from src.adapters.llms import get_llm
from typing import Any, Dict, Optional, List

def validate_dp_name(name: str, max_len: int = 75) -> dict:
    """
    Validate a Data Product name using ONLY the universal non-LLM rules:
      1) Max length <= max_len (default 75)
      2) Allowed characters: A-Z, a-z, 0-9, and '.' only
      3) Separator hygiene: no leading/trailing dots, no consecutive '..'
      4) Case style: each non-ID token must be UpperCamelCase
      5) ID shape (when a token is an ID): must match ^[A-Z]{2}\\d{5}$

    Returns a dict:
      {
        "valid": bool,
        "errors": [str, ...],
        "tokens": [str, ...]
      }

    Notes:
      - This function is category-agnostic (ODP/FDP/CDP checks are NOT applied here).
      - No acronym exceptions are made (LLM can handle acronym advice separately).
    """
    # Precompiled regexes (universal rules)
    _ALLOWED_RE = re.compile(r'^[A-Za-z0-9.]+$')       # only letters, digits, and dots
    _ID_RE      = re.compile(r'^[A-Z]{2}\d{5}$')       # ServiceNow App ID: e.g., AL18725
    _CAMEL_RE   = re.compile(r'^[A-Z][A-Za-z0-9]*$')   # simple UpperCamelCase: starts capital, then alnum
    if name is None:
        return {"valid": False, "errors": ["Name is required."], "tokens": []}

    s = name.strip()
    errors = []

    # 1) Max length
    if len(s) == 0:
        errors.append("Name must not be empty.")
    elif len(s) > max_len:
        errors.append(f"Name exceeds max length of {max_len} characters (got {len(s)}).")

    # 2) Allowed characters
    if not _ALLOWED_RE.fullmatch(s):
        errors.append("Only letters, digits, and '.' are allowed.")

    # 3) Separator hygiene
    if s.startswith(".") or s.endswith("."):
        errors.append("No leading or trailing '.' separators.")
    if ".." in s:
        errors.append("No consecutive '.' separators (e.g., '..').")

    tokens = [t for t in s.split(".") if t != ""]

    # 4) & 5) Per-token checks
    for i, tok in enumerate(tokens, start=1):
        if _ID_RE.fullmatch(tok):
            # Token is an ID; that's fine (no CamelCase check).
            continue
        # Non-ID tokens must be UpperCamelCase
        if not _CAMEL_RE.fullmatch(tok):
            errors.append(f"Token {i} ('{tok}') must be UpperCamelCase or match the ID pattern AA99999.")

    return {"valid": len(errors) == 0, "errors": errors, "tokens": tokens}


# agents- LLM review universal checks
def _first_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else None

def _camel_split(s: str) -> list[str]:
    return [p for p in re.findall(r'[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+', s) if p]

def llm_review_name(
    name: str,
    max_len: int = 75              # optional: fed into the prompt
) -> dict:
    """
    Linguistic review via LLM (acronyms, ambiguity, plurality, tense, readability).
    - Mandatory input: name (str)
    - If llm is None and you pass settings, you can build your adapter inside.
    - Returns a structured dict ready to combine with the deterministic result.
    """
    # Derive tokens & token types internally (keep interface minimal)
    _ID_RE = re.compile(r'^[A-Z]{2}\d{5}$')  # ServiceNow App ID (e.g., AL18725)
    tokens = [t for t in name.split(".") if t]
    token_types = ["id" if _ID_RE.fullmatch(t) else "name" for t in tokens]
    sub_tokens = [_camel_split(t) if tt=="name" else [t] for t, tt in zip(tokens, token_types)]

    payload = {
    "name": name,
    "tokens": tokens,
    "token_types": token_types,
    "sub_tokens": sub_tokens,      # ← new
    "issues_from_rules": []
    }

    # Load prompt template
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "name_checker.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    prompt = (
        template
        .replace("{payload_json_here}", json.dumps(payload, ensure_ascii=False))
        .replace("{max_len}", str(max_len))
    )
    settings = Settings()
    # Build LLM if only settings provided (optional)
     # or: from adapters.llm import get_llm
    llm = get_llm(settings)
    # Run LLM
    raw = llm.generate(prompt, temperature=0.1, max_tokens=400)
    data = _first_json(raw) or {}

    suggested = data.get("suggested_name", name)
    # Guardrails (lightweight): keep token order & IDs unchanged
    s_toks = [t for t in suggested.split(".") if t]
    if len(s_toks) == len(tokens):
        for i, ttype in enumerate(token_types):
            if ttype == "id" and s_toks[i] != tokens[i]:
                suggested = name  # revert if an ID was altered
                break
    else:
        suggested = name

    verdict = "suggest_changes" if suggested != name else "no_changes"

    return {
        "input_name": name,
        "tokens": tokens,
        "token_types": token_types,
        "suggested_name": suggested,
        "edits": data.get("edits", []),
        "issues": data.get("issues", []),
        "notes": data.get("notes", ""),
        "confidence": data.get("confidence", None),
        "verdict": verdict,
        "suggestion": suggested if verdict == "suggest_changes" else None,
        "llm_explnation": (data.get("llm_explnation") or "").strip()
    }



_ID_RE = re.compile(r'^[A-Z]{2}\d{5}$')  # e.g., AL18725

def _tok(name: str) -> List[str]:
    return [t for t in (name or "").split(".") if t]

def _add(checks: List[Dict[str, Any]], rule: str, status: str, severity: str, target: str, detail: str):
    checks.append({"rule": rule, "status": status, "severity": severity, "target": target, "detail": detail})

# -------------------------
# ODP (Origin Data Product)
# Grammar: AppID[.ChildAppID].BusinessName  (2–3 tokens)
# Deterministic checks only: positions, ID shapes, counts.
# -------------------------
def validate_odp_nonllm(name: str) -> Dict[str, Any]:
    tokens = _tok(name)
    checks: List[Dict[str, Any]] = []
    components: Dict[str, Optional[str]] = {
        "application_id": None,
        "child_application_id": None,
        "business_name": None
    }

    # Token count
    if not (2 <= len(tokens) <= 3):
        _add(checks, "ODP-GRAMMAR", "fail", "block", "grammar",
             "ODP requires 2–3 tokens: AppID[.ChildAppID].BusinessName")
    else:
        _add(checks, "ODP-GRAMMAR", "pass", "block", "grammar", "Token count within 2–3")

    # First token must be ID
    if len(tokens) >= 1:
        components["application_id"] = tokens[0]
        if _ID_RE.fullmatch(tokens[0]):
            _add(checks, "ODP-ID", "pass", "block", "application_id", "Valid ApplicationID")
        else:
            _add(checks, "ODP-ID", "fail", "block", "application_id", "First token must be a valid ServiceNow App ID")

    # Optional child ID (only when 3 tokens)
    if len(tokens) == 3:
        components["child_application_id"] = tokens[1]
        if _ID_RE.fullmatch(tokens[1]):
            _add(checks, "ODP-ID", "pass", "block", "child_application_id", "Valid Child ApplicationID")
        else:
            _add(checks, "ODP-ID", "fail", "block", "child_application_id", "Child ApplicationID must match App ID pattern")

    # Last token must NOT be an ID (it’s the business name)
    if len(tokens) >= 2:
        components["business_name"] = tokens[-1]
        if _ID_RE.fullmatch(tokens[-1]):
            _add(checks, "ODP-LAST-NOT-ID", "fail", "block", "business_name", "BusinessName must not be an ID")
        else:
            _add(checks, "ODP-LAST-NOT-ID", "pass", "block", "business_name", "BusinessName is not an ID")

    valid = not any(c["severity"] == "block" and c["status"] == "fail" for c in checks)
    return {"valid": valid, "components": components, "checks": checks, "notes": []}

# -------------------------
# FDP (Foundation Data Product)
# Grammar: SubjectArea.Concept[.Sub-Concept][.Specialisation][.DataCollection]  (2–5 tokens)
# No vocab or conditional rules here (deferred to LLM/connections).
# -------------------------
def validate_fdp_nonllm(name: str) -> Dict[str, Any]:
    tokens = _tok(name)
    checks: List[Dict[str, Any]] = []
    components: Dict[str, Optional[str]] = {
        "subject_area": None,
        "concept": None,
        "sub_concept": None,
        "specialisation": None,
        "data_collection": None
    }

    # Token count only
    if not (2 <= len(tokens) <= 5):
        _add(checks, "FDP-GRAMMAR", "fail", "block", "grammar",
             "FDP needs 2–5 tokens: SA.Concept[.Sub][.Spec][.Data]")
    else:
        _add(checks, "FDP-GRAMMAR", "pass", "block", "grammar", "Token count within 2–5")

    # Positional mapping (no taxonomy validation)
    if len(tokens) >= 1:
        components["subject_area"] = tokens[0]
    if len(tokens) >= 2:
        components["concept"] = tokens[1]
    if len(tokens) >= 3:
        components["sub_concept"] = tokens[2]
    if len(tokens) >= 4:
        components["specialisation"] = tokens[3]
    if len(tokens) >= 5:
        components["data_collection"] = tokens[4]

    # Note that taxonomy & conditional requirements are deferred
    _add(checks, "FDP-TAXONOMY", "info", "warn", "taxonomy",
         "Subject/Concept/Sub-Concept membership not validated (no taxonomy configured).")

    valid = not any(c["severity"] == "block" and c["status"] == "fail" for c in checks)
    return {"valid": valid, "components": components, "checks": checks, "notes": []}

# -------------------------
# CDP (Consumption Data Product)
# Grammar: SA[.Concept][.Sub-Concept].UseCaseBusinessName[."360"].UseCaseApplicationID  (4–6 tokens)
# Deterministic checks: ID at end, '360' placement and uniqueness, token count.
# -------------------------
_MASS_FLAG = "360"

def validate_cdp_nonllm(name: str) -> Dict[str, Any]:
    tokens = _tok(name)
    checks: List[Dict[str, Any]] = []
    components: Dict[str, Optional[str]] = {
        "subject_area": None,
        "concept": None,
        "sub_concept": None,
        "use_case_business_name": None,
        "mass_adoption": False,
        "use_case_application_id": None
    }

    # Token count
    if not (4 <= len(tokens) <= 6):
        _add(checks, "CDP-GRAMMAR", "fail", "block", "grammar",
             "CDP needs 4–6 tokens: SA[.Concept][.Sub].UseCase[.'360'].AppID")
    else:
        _add(checks, "CDP-GRAMMAR", "pass", "block", "grammar", "Token count within 4–6")

    if not tokens:
        return {"valid": False, "components": components, "checks": checks, "notes": []}

    # Map roles by position (no vocab on SA/Concept/Sub)
    components["subject_area"] = tokens[0]
    i = 1
    # Optionally consume Concept
    if i < len(tokens) - 2 and tokens[i] != _MASS_FLAG and not _ID_RE.fullmatch(tokens[i]):
        components["concept"] = tokens[i]; i += 1
    # Optionally consume Sub-Concept
    if i < len(tokens) - 2 and tokens[i] != _MASS_FLAG and not _ID_RE.fullmatch(tokens[i]):
        components["sub_concept"] = tokens[i]; i += 1
    # UseCaseBusinessName must exist now
    if i < len(tokens) - 1:
        components["use_case_business_name"] = tokens[i]; i += 1
    else:
        _add(checks, "CDP-USECASE", "fail", "block", "use_case_business_name", "Missing UseCaseBusinessName")

    # Optional '360'
    mass_positions = [idx for idx, t in enumerate(tokens) if t == _MASS_FLAG]
    if mass_positions:
        if len(mass_positions) > 1:
            _add(checks, "CDP-360", "fail", "block", "literal", "'360' must appear at most once")
        components["mass_adoption"] = True

    # Final token must be an App ID
    components["use_case_application_id"] = tokens[-1]
    if _ID_RE.fullmatch(tokens[-1]):
        _add(checks, "CDP-ID-LAST", "pass", "block", "use_case_application_id", "Valid ApplicationID at the end")
    else:
        _add(checks, "CDP-ID-LAST", "fail", "block", "use_case_application_id",
             "Final token must be a valid ServiceNow App ID")

    # If '360' present, it must be immediately before the ID
    if _MASS_FLAG in tokens:
        idx_flag = tokens.index(_MASS_FLAG)
        if idx_flag != len(tokens) - 2:
            _add(checks, "CDP-360-POS", "fail", "block", "literal",
                 "'360' must be immediately before the ApplicationID")
        else:
            _add(checks, "CDP-360-POS", "pass", "block", "literal",
                 "'360' correctly placed before ApplicationID")

    valid = not any(c["severity"] == "block" and c["status"] == "fail" for c in checks)
    return {"valid": valid, "components": components, "checks": checks, "notes": []}


# -------------------------
# Helpers shared by LLM checks
# -------------------------
def _load_prompt(fname: str) -> str:
    p = os.path.join(os.path.dirname(__file__), "prompts", fname)
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def _build_llm_payload(name: str, max_len: int = 75) -> Dict[str, Any]:
    tokens = [t for t in (name or "").split(".") if t]
    token_types = ["id" if _ID_RE.fullmatch(t) else "name" for t in tokens]
    sub_tokens = [_camel_split(t) if tt == "name" else [t] for t, tt in zip(tokens, token_types)]
    return {
        "name": name,
        "tokens": tokens,
        "token_types": token_types,
        "sub_tokens": sub_tokens,
        "issues_from_rules": [],
        "max_len": max_len,
    }

def _run_llm_with_prompt(prompt_text: str) -> Dict[str, Any]:
    settings = Settings()
    llm = get_llm(settings)
    raw = llm.generate(prompt_text, temperature=0.0, max_tokens=500)
    return _first_json(raw) or {}

def _guard_llm_suggestion(name: str, payload: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    tokens = payload["tokens"]
    token_types = payload["token_types"]

    suggested = data.get("suggested_name", name)
    s_toks = [t for t in suggested.split(".") if t]
    # hard guards: same token count; IDs unchanged
    if len(s_toks) != len(tokens) or any(
        tt == "id" and s_toks[i] != tokens[i] for i, tt in enumerate(token_types)
    ):
        suggested = name

    # ---- MIRROR labels -> issues (and edits -> issues) ----
    issues = list(data.get("issues") or [])
    token_reviews = data.get("token_reviews") or []
    issue_keys = {(i.get("type"), i.get("token")) for i in issues if isinstance(i, dict)}

    # 1) token_reviews.labels -> issues[]
    for tr in token_reviews:
        tok = tr.get("raw")
        note = tr.get("note", "")
        for lab in tr.get("labels") or []:
            key = (lab, tok)
            if key not in issue_keys:
                issues.append({"type": lab, "token": tok, "note": note})
                issue_keys.add(key)

    # 2) edits[].reason -> issues[] (ensure every edit has a matching issue)
    for ed in data.get("edits") or []:
        idx = ed.get("index")
        reason = ed.get("reason")
        tok_for_edit = tokens[idx] if isinstance(idx, int) and 0 <= idx < len(tokens) else ed.get("from")
        key = (reason, tok_for_edit)
        if reason and tok_for_edit and key not in issue_keys:
            issues.append({"type": reason, "token": tok_for_edit, "note": "Mirrored from edit reason"})
            issue_keys.add(key)

    data["issues"] = issues
    # ---- end mirror ----
    expl = (data.get("llm_explnation") or "").strip()
    verdict = "suggest_changes" if suggested != name else "no_changes"
    return {
        "input_name": name,
        "tokens": tokens,
        "token_types": token_types,
        "suggested_name": suggested,
        "edits": data.get("edits", []),
        "issues": data.get("issues", []),
        "token_reviews": data.get("token_reviews", []),
        "notes": data.get("notes", ""),
        "confidence": data.get("confidence", None),
        "verdict": verdict,
        "suggestion": suggested if verdict == "suggest_changes" else None,
        "llm_explnation": expl,
    }


# -------------------------
# LLM checks by DP type
# -------------------------
def odp_llm_check(name: str, max_len: int = 75) -> Dict[str, Any]:
    payload = _build_llm_payload(name, max_len=max_len)
    tpl = _load_prompt("name_checker_odp.txt")
    prompt = tpl.replace("{payload_json_here}", json.dumps(payload, ensure_ascii=False)).replace("{max_len}", str(max_len))
    data = _run_llm_with_prompt(prompt)
    return _guard_llm_suggestion(name, payload, data)

def fdp_llm_check(name: str, max_len: int = 75) -> Dict[str, Any]:
    payload = _build_llm_payload(name, max_len=max_len)
    tpl = _load_prompt("name_checker_fdp.txt")
    prompt = tpl.replace("{payload_json_here}", json.dumps(payload, ensure_ascii=False)).replace("{max_len}", str(max_len))
    data = _run_llm_with_prompt(prompt)
    return _guard_llm_suggestion(name, payload, data)

def cdp_llm_check(name: str, max_len: int = 75) -> Dict[str, Any]:
    payload = _build_llm_payload(name, max_len=max_len)
    tpl = _load_prompt("name_checker_cdp.txt")
    prompt = tpl.replace("{payload_json_here}", json.dumps(payload, ensure_ascii=False)).replace("{max_len}", str(max_len))
    data = _run_llm_with_prompt(prompt)
    return _guard_llm_suggestion(name, payload, data)

def llm_check_by_type(name: str, dp_type: str, max_len: int = 75) -> Dict[str, Any]:
    t = (dp_type or "").upper()
    if t == "ODP": return odp_llm_check(name, max_len=max_len)
    if t == "FDP": return fdp_llm_check(name, max_len=max_len)
    if t == "CDP": return cdp_llm_check(name, max_len=max_len)
    return {
        "input_name": name, "tokens": [t for t in (name or "").split(".") if t],
        "token_types": [], "suggested_name": name, "edits": [], "issues": [
            {"type":"system","token":dp_type,"note":"Unknown data product type"}
        ],
        "token_reviews": [], "notes": "", "confidence": None,
        "verdict":"no_changes","suggestion": None
    }

# -------------------------
# Connections (warnings-only stubs) by DP type
# -------------------------
def odp_connections_checks(components: Dict[str, Any]) -> Dict[str, Any]:
    checks = []
    app_id = components.get("application_id")
    child_id = components.get("child_application_id")
    if app_id:
        checks.append({
            "system":"SNOW","check":"ApplicationID exists and is active/owned",
            "target": app_id, "status":"not_verified","action":"Lookup in ServiceNow by AppID"
        })
    if child_id:
        checks.append({
            "system":"SNOW","check":"Child ApplicationID exists and is active/owned",
            "target": child_id, "status":"not_verified","action":"Lookup in ServiceNow by AppID"
        })
    checks.append({
        "system":"SOURCE","check":"Single-source principle (one primary AppID)",
        "target": None, "status":"not_verified","action":"Confirm system ownership"
    })
    checks.append({
        "system":"CATALOG","check":"Duplicate ODP full name",
        "target": None, "status":"not_verified","action":"Search catalogue for exact match"
    })
    return {"connections_checks": checks, "notes": []}

def fdp_connections_checks(components: Dict[str, Any]) -> Dict[str, Any]:
    checks = []
    sa = components.get("subject_area"); concept = components.get("concept"); subc = components.get("sub_concept")
    checks.append({
        "system":"CDM","check":"SubjectArea/Concept/Sub-Concept membership in canonical taxonomy",
        "target": f"{sa or '?'} / {concept or '?'} / {subc or '?'}", "status":"not_verified","action":"Lookup in CDM"
    })
    checks.append({
        "system":"LINEAGE","check":"Declared lineage intent to upstream ODPs",
        "target": None, "status":"not_verified","action":"List ODPs this FDP derives from"
    })
    checks.append({
        "system":"CATALOG","check":"Duplicate FDP full name",
        "target": None, "status":"not_verified","action":"Search catalogue for exact match"
    })
    return {"connections_checks": checks, "notes": []}

def cdp_connections_checks(components: Dict[str, Any]) -> Dict[str, Any]:
    checks = []
    app_id = components.get("use_case_application_id")
    mass = components.get("mass_adoption", False)
    checks.append({
        "system":"SNOW","check":"UseCaseApplicationID exists and is active/owned",
        "target": app_id, "status":"not_verified","action":"Lookup in ServiceNow by AppID"
    })
    if mass:
        checks.append({
            "system":"CATALOG","check":"Mass-adoption flag required when '360' present",
            "target":"360", "status":"not_verified","action":"Mark entry as mass-adoption"
        })
    checks.append({
        "system":"LINEAGE","check":"Declared lineage to composed FDPs",
        "target": None, "status":"not_verified","action":"List FDPs used as inputs"
    })
    checks.append({
        "system":"CATALOG","check":"Duplicate CDP full name",
        "target": None, "status":"not_verified","action":"Search catalogue for exact match"
    })
    return {"connections_checks": checks, "notes": []}

def connections_checks_by_type(components: Dict[str, Any], dp_type: str) -> Dict[str, Any]:
    t = (dp_type or "").upper()
    if t == "ODP": return odp_connections_checks(components)
    if t == "FDP": return fdp_connections_checks(components)
    if t == "CDP": return cdp_connections_checks(components)
    return {"connections_checks": [{"system":"system","check":"Unknown type","target":dp_type,"status":"not_verified","action":"Specify ODP/FDP/CDP"}], "notes": []}


def check_name_both(name: str, dp_type: str, max_len: int = 75) -> Dict[str, Any]:
    dp_type_u = (dp_type or "").upper()

    # 1) Universal checks
    det = validate_dp_name(name, max_len=max_len)
    tokens = det.get("tokens", [t for t in (name or "").split(".") if t])
    checks = [{"source": "universal", "severity": "block", "rule": "universal", "detail": e}
              for e in det.get("errors", [])]

    # 2) Type-specific non-LLM (always run)
    type_fn = {"ODP": validate_odp_nonllm, "FDP": validate_fdp_nonllm, "CDP": validate_cdp_nonllm}.get(dp_type_u)
    type_res = type_fn(name) if type_fn else {
        "valid": False, "components": {}, "checks": [{
            "rule": "TYPE", "status": "fail", "severity": "block",
            "target": "category", "detail": f"Unknown data product type '{dp_type}'. Expected ODP/FDP/CDP."
        }], "notes": []
    }
    checks += [{**c, "source": c.get("source", "type_nonllm")} for c in type_res.get("checks", [])]

    # 3) LLM reviews (DP-specific is primary; generic optional for wording help)
    llm_dp = llm_check_by_type(name, dp_type_u, max_len=max_len)        # includes llm_explnation via guard
    llm_gen = llm_review_name(name=name, max_len=max_len)               # generic; keeps same schema
    checks += [{"source": "llm", "severity": "info", "rule": i.get("type"),
                "token": i.get("token"), "detail": i.get("note", "")}
               for i in (llm_dp.get("issues") or [])]

    # 4) Connections (warnings)
    conn = connections_checks_by_type(type_res.get("components", {}), dp_type_u)
    checks += [{"source": "connections", "severity": "info", "system": c.get("system"),
                "rule": c.get("check"), "target": c.get("target"),
                "status": c.get("status"), "action": c.get("action")}
               for c in conn.get("connections_checks", [])]

    # 5) Verdict + suggestion
    invalid = (not det.get("valid", False)) or (not type_res.get("valid", False))
    if invalid:
        verdict, scientific_name = "invalid", None
        suggestion = llm_dp.get("suggestion")
        edits = llm_dp.get("edits", [])
    else:
        if llm_dp.get("suggestion"):
            verdict, scientific_name = "needs_changes", llm_dp.get("suggested_name", name)
            suggestion, edits = llm_dp.get("suggestion"), llm_dp.get("edits", [])
        else:
            verdict, scientific_name, suggestion, edits = "valid", name, None, []

    # 6) Friendly explanation (short + combined)
    dp_expl = (llm_dp or {}).get("llm_explnation") or ""
    gen_expl = (llm_gen or {}).get("llm_explnation") or ""
    combined = dp_expl if dp_expl else gen_expl
    if dp_expl and gen_expl and gen_expl not in dp_expl:
        combined = f"{dp_expl} {gen_expl}"
        if len(combined) > 180:  # keep tidy
            combined = dp_expl
    if invalid:
        combined = f"Structurally invalid—{combined or 'Review grammar/IDs and apply suggested wording if provided.'}"

    return {
        "input_name": name,
        "type": dp_type_u,
        "tokens": tokens,
        "deterministic": det,            # universal
        "type_nonllm": type_res,         # per-type
        "llm_review": llm_dp,            # DP-specific LLM (with llm_explnation)
        "llm_review_generic": llm_gen,   # generic LLM (with llm_explnation)
        "connections": conn,             # warnings
        "overall": {
            "verdict": verdict,                  # invalid | needs_changes | valid
            "scientific_name": scientific_name,
            "suggestion": suggestion,
            "checks": checks,
            "edits": edits,
            "explanation": combined             # friendly one-liner for UI/chat
        }
    }

