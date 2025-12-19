"""
Governance data loading and embedding management
"""
import pandas as pd
import numpy as np
import json
import os
import re
from typing import Dict, List, Optional, Tuple
from cortex_connection import cortex
import constants


class GovernanceDataLoader:
    """Load and manage governance workflow data and embeddings"""
    
    def __init__(self, excel_path: str = None):
        """
        Initialize the data loader
        
        Args:
            excel_path: Path to nodes_edges_governance.xlsx
        """
        self.excel_path = excel_path or constants.EXCEL_PATH
        self.embed_cache_path = constants.EMBED_CACHE_PATH
        
        # Load Excel data
        self._load_excel_data()
        
        # Load or compute embeddings
        self._load_embeddings()
        
        # Build step aliases for deterministic matching
        self._build_step_aliases()
    
    def _load_excel_data(self):
        """Load and normalize data from Excel file"""
        print(f"📊 Loading governance data from {self.excel_path}...")
        
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
        
        print(f"✅ Loaded {len(self.nodes_df)} steps and {len(self.edges_df)} edges")
    
    def _load_embeddings(self):
        """Load embeddings from cache or compute them"""
        self.step_embeddings: Dict[str, Dict] = {}
        
        if os.path.exists(self.embed_cache_path):
            try:
                with open(self.embed_cache_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for sid, data in raw.items():
                    self.step_embeddings[str(sid).upper()] = {
                        "embedding": data["embedding"],
                        "text": data.get("text", ""),
                    }
                print(f"✅ Loaded {len(self.step_embeddings)} step embeddings from cache")
                return
            except Exception as e:
                print(f"⚠️ Failed to load embeddings cache: {e}. Recomputing...")
                self.step_embeddings = {}
        else:
            print("ℹ️ No embeddings cache found. Computing step embeddings...")
        
        # Compute embeddings
        for _, row in self.nodes_df.iterrows():
            sid = str(row["Step_ID"]).strip().upper()
            text_parts = [
                str(row["Stage_Name"]),
                str(row["Step_Name"]),
                str(row["Purpose"]),
                str(row["Description"]),
            ]
            combined_text = ". ".join([p for p in text_parts if p and p != "nan"])
            emb = cortex.get_embedding(combined_text)
            self.step_embeddings[sid] = {
                "embedding": emb,
                "text": combined_text,
            }
        
        # Save cache
        with open(self.embed_cache_path, "w", encoding="utf-8") as f:
            json.dump(self.step_embeddings, f)
        print(f"✅ Computed and cached embeddings for {len(self.step_embeddings)} steps")
    
    def _build_step_aliases(self):
        """Build alias mappings for deterministic step matching"""
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
    
    def step_index(self, step_id: str) -> int:
        """Get position of step in canonical order"""
        try:
            return self.ordered_step_ids.index(step_id)
        except ValueError:
            return 10_000
    
    def get_step_record(self, step_id: str) -> Optional[Dict]:
        """Get metadata for a given step_id"""
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
        """Calculate cosine similarity between two vectors"""
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)
    
    def map_text_to_step_id(
        self, 
        text: str, 
        emb_threshold: float = 0.5
    ) -> Tuple[Optional[str], str, float]:
        """
        Map natural-language text to a Step_ID using multiple strategies:
        1. Direct S-identifier (S1, S2, etc.)
        2. Alias matching
        3. Substring matching
        4. Embedding similarity
        
        Args:
            text: Natural language text describing a step
            emb_threshold: Minimum similarity threshold for embedding match
            
        Returns:
            Tuple of (step_id, match_method, confidence_score)
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
        
        # 4) Embedding similarity
        query_emb = cortex.get_embedding(text)
        best_emb_sid = None
        best_emb_score = 0.0
        
        for sid, data in self.step_embeddings.items():
            sim = self.cosine_similarity(query_emb, data["embedding"])
            if sim > best_emb_score:
                best_emb_score = sim
                best_emb_sid = sid
        
        if best_emb_sid and best_emb_score >= emb_threshold:
            return best_emb_sid, "embedding_match", best_emb_score
        
        return None, "no_match", 0.0
    
    def get_outgoing_edges(self, step_id: str) -> List[Dict]:
        """Get all outgoing edges from a step"""
        edges = self.edges_df[self.edges_df["from"] == step_id]
        return edges.to_dict('records')
    
    def get_incoming_edges(self, step_id: str) -> List[Dict]:
        """Get all incoming edges to a step"""
        edges = self.edges_df[self.edges_df["to"] == step_id]
        return edges.to_dict('records')


# Global instance
governance_data = GovernanceDataLoader()