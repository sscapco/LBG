import pandas as pd
import numpy as np
import json
import os
import re
from typing import List, Dict, Optional, Tuple
import constants
from cortex_connection import cortex_client


######### Governance Data Loading #########

def load_governance_data():
    """Load governance workflow data from Excel"""
    
    nodes_raw_df = pd.read_excel(constants.EXCEL_PATH, sheet_name="Nodes")
    edges_raw_df = pd.read_excel(constants.EXCEL_PATH, sheet_name="Edges")
    
    # Normalize Nodes columns
    nodes_df = nodes_raw_df.rename(
        columns={
            "Stage Name": "Stage_Name",
            "Step ID": "Step_ID",
            "Step name": "Step_Name",
            "Purpose (Brief one liner to outline purpose of the task)": "Purpose",
        }
    )
    
    # Ensure Automatable + Automation_step exist
    if "Automatable" not in nodes_df.columns:
        nodes_df["Automatable"] = False
    if "Automation_step" not in nodes_df.columns:
        nodes_df["Automation_step"] = ""
    
    # Normalize Edges columns
    edges_df = edges_raw_df.rename(
        columns={
            "Question/Probe": "Question_Probe",
            "Transactional vs Analytical": "Transactional_vs_Analytical",
        }
    )
    
    # Validate required columns
    required_node_cols = ["Stage", "Stage_Name", "Step_ID", "Step_Name", "Purpose", "Description"]
    missing_node_cols = [c for c in required_node_cols if c not in nodes_df.columns]
    if missing_node_cols:
        raise ValueError(f"Missing expected columns in Nodes sheet: {missing_node_cols}")
    
    required_edge_cols = ["from", "to", "edge_type", "guard_condition", "Question_Probe"]
    missing_edge_cols = [c for c in required_edge_cols if c not in edges_df.columns]
    if missing_edge_cols:
        raise ValueError(f"Missing expected columns in Edges sheet: {missing_edge_cols}")
    
    # Normalize Step_IDs and edge endpoints
    nodes_df["Step_ID"] = nodes_df["Step_ID"].astype(str).str.strip().str.upper()
    edges_df["from"] = edges_df["from"].astype(str).str.strip().str.upper()
    edges_df["to"] = edges_df["to"].astype(str).str.strip().str.upper()
    
    # Normalize Automatable & Automation_step
    nodes_df["Automatable"] = (
        nodes_df["Automatable"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "YES", "Y", "1"])
    )
    
    nodes_df["Automation_step"] = nodes_df["Automation_step"].astype(str).str.strip()
    
    return nodes_df, edges_df


# Load data globally
nodes_df, edges_df = load_governance_data()

# Canonical workflow order: row order in Nodes
ORDERED_STEP_IDS: List[str] = list(nodes_df["Step_ID"])


######### Step Information Retrieval #########

def step_index(step_id: str) -> int:
    """Get position of step in canonical order"""
    try:
        return ORDERED_STEP_IDS.index(step_id)
    except ValueError:
        return 10_000


def get_step_record(step_id: str) -> Optional[Dict]:
    """Get metadata for a given step_id"""
    row = nodes_df[nodes_df["Step_ID"] == step_id]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "id": r["Step_ID"],
        "name": r["Step_Name"],
        "purpose": r["Purpose"],
        "description": r["Description"],
        "stage": r["Stage"],
        "stage_name": r["Stage_Name"],
        "automatable": bool(r.get("Automatable", False)),
        "automation_step": (r.get("Automation_step") or "").strip() or None,
    }


######### Embedding Utilities #########

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


######### Embedding Cache Management #########

STEP_EMBEDDINGS: Dict[str, Dict] = {}

def load_or_compute_embeddings():
    """Load embeddings from cache or compute them"""
    global STEP_EMBEDDINGS
    
    if os.path.exists(constants.EMBED_CACHE_PATH):
        try:
            with open(constants.EMBED_CACHE_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for sid, data in raw.items():
                STEP_EMBEDDINGS[str(sid).upper()] = {
                    "embedding": data["embedding"],
                    "text": data.get("text", ""),
                }
            print(f"✅ Loaded {len(STEP_EMBEDDINGS)} step embeddings from cache")
            return
        except Exception as e:
            print(f"⚠️ Failed to load embeddings cache: {e}. Recomputing...")
            STEP_EMBEDDINGS = {}
    else:
        print("ℹ️ No embeddings cache found. Computing step embeddings...")
    
    # Compute embeddings
    for _, row in nodes_df.iterrows():
        sid = str(row["Step_ID"]).strip().upper()
        text_parts = [
            str(row["Stage_Name"]),
            str(row["Step_Name"]),
            str(row["Purpose"]),
            str(row["Description"]),
        ]
        combined_text = ". ".join([p for p in text_parts if p and p != "nan"])
        emb = cortex_client.get_embedding(combined_text)
        STEP_EMBEDDINGS[sid] = {
            "embedding": emb,
            "text": combined_text,
        }
    
    # Save cache
    with open(constants.EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(STEP_EMBEDDINGS, f)
    print(f"✅ Precomputed and cached embeddings for {len(STEP_EMBEDDINGS)} steps")


######### Alias-based deterministic mapping #########

STEP_ALIASES: Dict[str, List[str]] = {}

def build_step_aliases():
    """Build alias dictionary for deterministic step mapping"""
    global STEP_ALIASES
    
    for _, row in nodes_df.iterrows():
        sid = str(row["Step_ID"]).strip().upper()
        name = str(row["Step_Name"]).strip()
        aliases = set()
        
        if sid:
            aliases.add(sid.lower())
        
        if name:
            aliases.add(name.lower())
            
            # contents inside parentheses
            for match in re.findall(r"\(([^)]+)\)", name):
                alias = match.strip()
                if alias:
                    aliases.add(alias.lower())
            
            # acronym from uppercase letters
            caps = "".join(ch for ch in name if ch.isupper())
            if len(caps) >= 3:
                aliases.add(caps.lower())
        
        STEP_ALIASES[sid] = sorted(aliases)


def map_text_to_step_id(text: str, emb_threshold: float = 0.5) -> Tuple[Optional[str], str, float]:
    """
    Map natural-language text to a Step_ID using id, alias, substring, then embeddings
    
    Returns:
        (step_id, method, confidence)
    """
    if not text:
        return None, "none", 0.0
    
    t = text.strip().lower()
    if not t:
        return None, "none", 0.0
    
    # 1) Direct S-identifier
    m = re.search(r"\b(s\d{1,3})\b", t)
    if m:
        sid = m.group(1).upper()
        if sid in STEP_EMBEDDINGS:
            return sid, "id_match", 1.0
    
    # 2) Alias match
    for sid, aliases in STEP_ALIASES.items():
        for alias in aliases:
            if alias in t:
                return sid, "alias_match", 0.95
    
    # 3) Substring match in step name
    for sid in STEP_EMBEDDINGS.keys():
        rec = get_step_record(sid)
        if rec:
            name_lower = rec["name"].lower()
            if len(name_lower) >= 4 and name_lower in t:
                return sid, "substring_match", 0.9
    
    # 4) Embedding similarity
    query_emb = cortex_client.get_embedding(text)
    best_sid = None
    best_sim = 0.0
    
    for sid, data in STEP_EMBEDDINGS.items():
        sim = cosine_similarity(query_emb, data["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_sid = sid
    
    if best_sim >= emb_threshold:
        return best_sid, "embedding_match", best_sim
    
    return None, "no_match", best_sim


######### Initialize on import #########

load_or_compute_embeddings()
build_step_aliases()