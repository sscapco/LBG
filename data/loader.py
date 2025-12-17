"""
Data loading from Excel governance workflow definition.
"""
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path

from config import DataConfig
from core.models import StepRecord


class GovernanceDataLoader:
    """Loads and provides access to governance workflow data"""
    
    def __init__(self, excel_path: Optional[Path] = None):
        self.excel_path = excel_path or DataConfig.EXCEL_PATH
        self.nodes_df: Optional[pd.DataFrame] = None
        self.edges_df: Optional[pd.DataFrame] = None
        self.ordered_step_ids: List[str] = []
        self._load_data()
    
    def _load_data(self):
        """Load and normalize data from Excel"""
        print(f"📂 Loading governance data from {self.excel_path}")
        
        # Load sheets
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
        
        # Ensure automation columns exist
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
        self._validate_columns()
        
        # Normalize data
        self._normalize_data()
        
        # Store canonical step order
        self.ordered_step_ids = list(self.nodes_df["Step_ID"])
        
        print(f"✅ Loaded {len(self.nodes_df)} steps and {len(self.edges_df)} edges")
    
    def _validate_columns(self):
        """Validate required columns exist"""
        required_node_cols = ["Stage", "Stage_Name", "Step_ID", "Step_Name", "Purpose", "Description"]
        missing_node_cols = [c for c in required_node_cols if c not in self.nodes_df.columns]
        if missing_node_cols:
            raise ValueError(f"Missing expected columns in Nodes sheet: {missing_node_cols}")
        
        required_edge_cols = ["from", "to", "edge_type", "guard_condition", "Question_Probe"]
        missing_edge_cols = [c for c in required_edge_cols if c not in self.edges_df.columns]
        if missing_edge_cols:
            raise ValueError(f"Missing expected columns in Edges sheet: {missing_edge_cols}")
    
    def _normalize_data(self):
        """Normalize data types and formats"""
        # Normalize Step IDs and edge endpoints
        self.nodes_df["Step_ID"] = self.nodes_df["Step_ID"].astype(str).str.strip().str.upper()
        self.edges_df["from"] = self.edges_df["from"].astype(str).str.strip().str.upper()
        self.edges_df["to"] = self.edges_df["to"].astype(str).str.strip().str.upper()
        
        # Normalize Automatable
        self.nodes_df["Automatable"] = (
            self.nodes_df["Automatable"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(["TRUE", "YES", "Y", "1"])
        )
        
        # Normalize Automation_step
        self.nodes_df["Automation_step"] = self.nodes_df["Automation_step"].astype(str).str.strip()
    
    def get_step_record(self, step_id: str) -> Optional[StepRecord]:
        """Get step record by ID"""
        row = self.nodes_df[self.nodes_df["Step_ID"] == step_id]
        if row.empty:
            return None
        
        r = row.iloc[0]
        return StepRecord(
            id=r["Step_ID"],
            name=r["Step_Name"],
            purpose=r["Purpose"],
            description=r["Description"],
            stage=r["Stage"],
            stage_name=r["Stage_Name"],
            automatable=bool(r.get("Automatable", False)),
            automation_step=(r.get("Automation_step") or "").strip() or None,
        )
    
    def get_step_index(self, step_id: str) -> int:
        """Get position of step in canonical workflow order"""
        try:
            return self.ordered_step_ids.index(step_id)
        except ValueError:
            return 10_000  # Large number for unknown steps
    
    def get_outgoing_edges(self, step_id: str, edge_type: Optional[str] = None) -> pd.DataFrame:
        """Get edges going out from a step"""
        edges = self.edges_df[self.edges_df["from"] == step_id]
        if edge_type:
            edges = edges[edges["edge_type"].astype(str).str.lower() == edge_type.lower()]
        return edges
    
    def get_incoming_edges(self, step_id: str, edge_type: Optional[str] = None) -> pd.DataFrame:
        """Get edges coming into a step"""
        edges = self.edges_df[self.edges_df["to"] == step_id]
        if edge_type:
            edges = edges[edges["edge_type"].astype(str).str.lower() == edge_type.lower()]
        return edges
    
    def get_all_step_ids(self) -> List[str]:
        """Get all step IDs in canonical order"""
        return self.ordered_step_ids.copy()
    
    def get_step_text_for_embedding(self, step_id: str) -> Optional[str]:
        """Get combined text for embedding generation"""
        row = self.nodes_df[self.nodes_df["Step_ID"] == step_id]
        if row.empty:
            return None
        
        r = row.iloc[0]
        text_parts = [
            str(r["Stage_Name"]),
            str(r["Step_Name"]),
            str(r["Purpose"]),
            str(r["Description"]),
        ]
        combined_text = ". ".join([p for p in text_parts if p and p != "nan"])
        return combined_text
