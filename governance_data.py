import pandas as pd
import numpy as np
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from cortex_connection import cortex
import constants

# Load and manage governance workflow data and embeddings
class GovernanceDataLoader:
    def __init__(self, excel_path: str = None):
        self.excel_path = excel_path or constants.EXCEL_PATH
        self.embed_cache_path = constants.EMBED_CACHE_PATH
        self._step_index: Dict[str, int] = {}
        self._embedding_matrix: Optional[np.ndarray] = None
        self._embedding_ids: List[str] = []
        self._embedding_norms: Optional[np.ndarray] = None
        
        # Load Excel data
        self._load_excel_data()
        
        # Load or compute embeddings
        self._load_embeddings()
        
        # Build step aliases for deterministic matching
        self._build_step_aliases()
    
    def _load_excel_data(self):
        nodes_raw_df = pd.read_excel(self.excel_path, sheet_name="Nodes")
        edges_raw_df = pd.read_excel(self.excel_path, sheet_name="Edges")
        
        # Normalize Nodes columns
        self.nodes_df = nodes_raw_df.rename(
            columns={
                "Stage Name": "Stage_Name",
                "Step ID": "Step_ID",
                "Step name": "Step_Name",
                "Purpose (Brief one liner to outline purpose of the task)": "Purpose",
            }
        )
        
        # Ensure Automatable + Automation_step exist
        if "Automatable" not in self.nodes_df.columns:
            self.nodes_df["Automatable"] = False
        if "Automation_step" not in self.nodes_df.columns:
            self.nodes_df["Automation_step"] = ""
        
        # Normalize Edges columns
        self.edges_df = edges_raw_df.rename(
            columns={
                "Question/Probe": "Question_Probe",
                "Transactional vs Analytical": "Transactional_vs_Analytical",
            }
        )
        
        # Validate required columns
        required_node_cols = ["Stage", "Stage_Name", "Step_ID", "Step_Name", "Purpose", "Description"]
        missing_node_cols = [c for c in required_node_cols if c not in self.nodes_df.columns]
        if missing_node_cols:
            raise ValueError(f"Missing expected columns in Nodes sheet: {missing_node_cols}")
        
        required_edge_cols = ["from", "to", "edge_type", "guard_condition", "Question_Probe"]
        missing_edge_cols = [c for c in required_edge_cols if c not in self.edges_df.columns]
        if missing_edge_cols:
            raise ValueError(f"Missing expected columns in Edges sheet: {missing_edge_cols}")
        
        # Normalize Step_IDs and edge endpoints
        self.nodes_df["Step_ID"] = self.nodes_df["Step_ID"].astype(str).str.strip().str.upper()
        self.edges_df["from"] = self.edges_df["from"].astype(str).str.strip().str.upper()
        self.edges_df["to"] = self.edges_df["to"].astype(str).str.strip().str.upper()
        
        # Normalize Automatable & Automation_step
        self.nodes_df["Automatable"] = (
            self.nodes_df["Automatable"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(["TRUE", "YES", "Y", "1"])
        )
        
        self.nodes_df["Automation_step"] = self.nodes_df["Automation_step"].astype(str).str.strip()
        
        # Canonical workflow order: row order in Nodes
        self.ordered_step_ids = list(self.nodes_df["Step_ID"])
        self._step_index = {sid: i for i, sid in enumerate(self.ordered_step_ids)}

    def _excel_fingerprint(self) -> str:
        try:
            st = os.stat(self.excel_path)
            return f"{st.st_size}:{int(st.st_mtime)}"
        except Exception:
            return "unknown"
    
    # Load embeddings from cache or compute them
    def _load_embeddings(self):
        self.step_embeddings: Dict[str, Dict] = {}
        
        if os.path.exists(self.embed_cache_path):
            try:
                with open(self.embed_cache_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)

                # Support both legacy caches ({sid: {...}}) and versioned caches.
                if isinstance(raw, dict) and "_meta" in raw and "steps" in raw:
                    meta = raw.get("_meta", {})
                    steps = raw.get("steps", {})

                    # Only reuse cache if it matches the current Excel fingerprint.
                    if meta.get("excel_fingerprint") == self._excel_fingerprint():
                        for sid, data in steps.items():
                            self.step_embeddings[str(sid).upper()] = {
                                "embedding": data["embedding"],
                                "text": data.get("text", ""),
                            }
                        self._build_embedding_index()
                        return
                elif isinstance(raw, dict):
                    for sid, data in raw.items():
                        if sid == "_meta":
                            continue
                        self.step_embeddings[str(sid).upper()] = {
                            "embedding": data["embedding"],
                            "text": data.get("text", ""),
                        }
                    self._build_embedding_index()
                    return
            except Exception as e:
                self.step_embeddings = {}
        else:
            pass
        
        # Compute embeddings with enhanced text for better semantic matching
        for _, row in self.nodes_df.iterrows():
            sid = str(row["Step_ID"]).strip().upper()
            
            # Build rich text for embedding - include multiple representations
            text_parts = []
            
            # Core identification
            text_parts.append(f"Step {sid}")
            text_parts.append(str(row["Step_Name"]))
            
            # Purpose (key for matching user queries)
            purpose = str(row["Purpose"])
            if purpose and purpose != "nan":
                text_parts.append(f"Purpose: {purpose}")
            
            # Description (contains detailed keywords)
            description = str(row["Description"])
            if description and description != "nan":
                text_parts.append(description)
            
            # Stage context
            stage = str(row["Stage_Name"])
            if stage and stage != "nan":
                text_parts.append(f"Stage: {stage}")
            
            # Combine with clear separators for better embedding
            combined_text = " | ".join([p for p in text_parts if p])
            
            emb = cortex.get_embedding(combined_text)
            self.step_embeddings[sid] = {
                "embedding": emb,
                "text": combined_text,
            }

        self._build_embedding_index()
        
        # Save cache
        with open(self.embed_cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "_meta": {
                        "excel_fingerprint": self._excel_fingerprint(),
                    },
                    "steps": self.step_embeddings,
                },
                f,
            )

    def _build_embedding_index(self) -> None:
        ids: List[str] = []
        vectors: List[List[float]] = []
        for sid, data in self.step_embeddings.items():
            emb = data.get("embedding")
            if isinstance(emb, list) and emb:
                ids.append(sid)
                vectors.append(emb)

        if not vectors:
            self._embedding_ids = []
            self._embedding_matrix = None
            self._embedding_norms = None
            return

        mat = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(mat, axis=1)
        self._embedding_ids = ids
        self._embedding_matrix = mat
        self._embedding_norms = norms
    
    # Build alias mappings for deterministic step matching
    def _build_step_aliases(self):
        self.step_aliases: Dict[str, List[str]] = {}
        
        for _, row in self.nodes_df.iterrows():
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
            
            self.step_aliases[sid] = sorted(aliases)
    
    # Get position of step in canonical order
    def step_index(self, step_id: str) -> int:
        return self._step_index.get(step_id, 10_000)
    
    # Get metadata for a given step_id
    def get_step_record(self, step_id: str) -> Optional[Dict]:
        row = self.nodes_df[self.nodes_df["Step_ID"] == step_id]
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
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def deterministic_match(self, text: str) -> Tuple[Optional[str], str, float]:
        if not text:
            return None, "none", 0.0

        t = text.strip().lower()
        if not t:
            return None, "none", 0.0

        # 1) Direct S-identifier
        m = re.search(r"\b(s\d{1,3})\b", t)
        if m:
            sid = m.group(1).upper()
            if sid in self.step_embeddings:
                return sid, "id_match", 1.0

        # 2) Alias exact match
        for sid, aliases in self.step_aliases.items():
            for alias in aliases:
                if alias in t:
                    return sid, "alias_match", 0.95

        # 3) Substring match in step names
        best_substr_sid = None
        best_substr_len = 0

        for _, row in self.nodes_df.iterrows():
            sid = str(row["Step_ID"]).strip().upper()
            name = str(row["Step_Name"]).strip().lower()

            if name in t and len(name) > best_substr_len:
                best_substr_sid = sid
                best_substr_len = len(name)

        if best_substr_sid and best_substr_len >= 5:
            return best_substr_sid, "substring_match", 0.85

        return None, "no_match", 0.0
    
    # Map natural-language text to a Step_ID using multiple strategies 
    def map_text_to_step_id(
        self, 
        text: str, 
        emb_threshold: float = 0.5
    ) -> Tuple[Optional[str], str, float]:
       
        sid, method, confidence = self.deterministic_match(text)
        if sid:
            return sid, method, confidence
        
        # 4) Embedding similarity
        best_emb_sid, best_emb_score = self.best_step_by_embedding(text)
        
        if best_emb_sid and best_emb_score >= emb_threshold:
            return best_emb_sid, "embedding_match", best_emb_score
        
        return None, "no_match", 0.0

    def best_step_by_embedding(self, text: str) -> Tuple[Optional[str], float]:
        candidates = self.semantic_candidates(text, top_k=1)
        if not candidates:
            return None, 0.0
        return candidates[0]["id"], float(candidates[0]["score"])

    def semantic_candidates(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not text:
            return []

        query_emb = cortex.get_embedding(text)
        if self._embedding_matrix is None or self._embedding_norms is None:
            # Fallback: slower Python loop (should be rare).
            scores = []
            for step_id, data in self.step_embeddings.items():
                sim = self.cosine_similarity(query_emb, data["embedding"])
                scores.append({"id": step_id, "score": sim, "text": data.get("text", "")})
            scores.sort(key=lambda x: x["score"], reverse=True)
            return scores[:top_k]

        q = np.asarray(query_emb, dtype=float)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []

        dots = self._embedding_matrix @ q
        sims = dots / (self._embedding_norms * qn)

        idx = np.argsort(-sims)[: max(1, top_k)]
        out: List[Dict[str, Any]] = []
        for i in idx:
            sid = self._embedding_ids[int(i)]
            out.append(
                {
                    "id": sid,
                    "score": float(sims[int(i)]),
                    "text": self.step_embeddings.get(sid, {}).get("text", ""),
                }
            )
        return out
    
    # Get all outgoing edges from a step
    def get_outgoing_edges(self, step_id: str) -> List[Dict]:
        edges = self.edges_df[self.edges_df["from"] == step_id]
        return edges.to_dict('records')
    
    # Get all incoming edges to a step
    def get_incoming_edges(self, step_id: str) -> List[Dict]:
        edges = self.edges_df[self.edges_df["to"] == step_id]
        return edges.to_dict('records')


_governance_data_singleton: Optional[GovernanceDataLoader] = None


def get_governance_data() -> GovernanceDataLoader:
    global _governance_data_singleton
    if _governance_data_singleton is None:
        _governance_data_singleton = GovernanceDataLoader()
    return _governance_data_singleton
