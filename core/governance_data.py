import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cortex_connection import cortex
import constants


class GovernanceDataLoader:
    def __init__(self, excel_path: str = None):
        self.excel_path = excel_path or constants.EXCEL_PATH
        self.embed_cache_path = constants.EMBED_CACHE_PATH
        self._step_index: Dict[str, int] = {}
        self._embedding_matrix: Optional[np.ndarray] = None
        self._embedding_ids: List[str] = []
        self._embedding_norms: Optional[np.ndarray] = None
        self._doc_tokens: Dict[str, set[str]] = {}
        self._idf: Dict[str, float] = {}

        self._load_excel_data()
        self._load_embeddings()
        self._build_step_aliases()
        self._build_lexical_index()

    def _load_excel_data(self):
        nodes_raw_df = pd.read_excel(self.excel_path, sheet_name="Nodes")
        edges_raw_df = pd.read_excel(self.excel_path, sheet_name="Edges")

        self.nodes_df = nodes_raw_df.rename(
            columns={
                "Stage Name": "Stage_Name",
                "Step ID": "Step_ID",
                "Step name": "Step_Name",
                "Purpose (Brief one liner to outline purpose of the task)": "Purpose",
            }
        )

        if "Automatable" not in self.nodes_df.columns:
            self.nodes_df["Automatable"] = False
        if "Automation_step" not in self.nodes_df.columns:
            self.nodes_df["Automation_step"] = ""

        self.edges_df = edges_raw_df.rename(
            columns={
                "Question/Probe": "Question_Probe",
                "Transactional vs Analytical": "Transactional_vs_Analytical",
            }
        )

        required_node_cols = ["Stage", "Stage_Name", "Step_ID", "Step_Name", "Purpose", "Description"]
        missing_node_cols = [c for c in required_node_cols if c not in self.nodes_df.columns]
        if missing_node_cols:
            raise ValueError(f"Missing expected columns in Nodes sheet: {missing_node_cols}")

        required_edge_cols = ["from", "to", "edge_type", "guard_condition", "Question_Probe"]
        missing_edge_cols = [c for c in required_edge_cols if c not in self.edges_df.columns]
        if missing_edge_cols:
            raise ValueError(f"Missing expected columns in Edges sheet: {missing_edge_cols}")

        self.nodes_df["Step_ID"] = self.nodes_df["Step_ID"].astype(str).str.strip().str.upper()
        self.edges_df["from"] = self.edges_df["from"].astype(str).str.strip().str.upper()
        self.edges_df["to"] = self.edges_df["to"].astype(str).str.strip().str.upper()

        self.nodes_df["Automatable"] = (
            self.nodes_df["Automatable"].astype(str).str.strip().str.upper().isin(["TRUE", "YES", "Y", "1"])
        )
        self.nodes_df["Automation_step"] = self.nodes_df["Automation_step"].astype(str).str.strip()

        self.ordered_step_ids = list(self.nodes_df["Step_ID"])
        self._step_index = {sid: i for i, sid in enumerate(self.ordered_step_ids)}

    def _excel_fingerprint(self) -> str:
        try:
            st = os.stat(self.excel_path)
            return f"{st.st_size}:{int(st.st_mtime)}"
        except Exception:
            return "unknown"

    def _load_embeddings(self):
        self.step_embeddings: Dict[str, Dict] = {}

        if os.path.exists(self.embed_cache_path):
            try:
                with open(self.embed_cache_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)

                if isinstance(raw, dict) and "_meta" in raw and "steps" in raw:
                    meta = raw.get("_meta", {})
                    steps = raw.get("steps", {})
                    # Check version to force regeneration if structure changed
                    if meta.get("excel_fingerprint") == self._excel_fingerprint() and meta.get("version") == "2.0":
                        for sid, data in steps.items():
                            self.step_embeddings[str(sid).upper()] = data
                        self._build_embedding_index()
                        return
            except Exception:
                self.step_embeddings = {}

        # Generate multi-field embeddings
        for _, row in self.nodes_df.iterrows():
            sid = str(row["Step_ID"]).strip().upper()

            # Separate text components for targeted matching
            step_name = str(row["Step_Name"]).strip()
            purpose = str(row["Purpose"]).strip()
            description = str(row["Description"]).strip()
            stage = str(row["Stage_Name"]).strip()

            # Create three separate embeddings for name, purpose, and description
            name_text = f"Step {sid}: {step_name}"
            purpose_text = f"{purpose}" if purpose and purpose != "nan" else step_name

            # For description, include key details but keep it focused
            desc_text = description if description and description != "nan" else purpose_text

            # Combined text for backward compatibility and general matching
            text_parts = [f"Step {sid}", step_name]
            if purpose and purpose != "nan":
                text_parts.append(f"Purpose: {purpose}")
            if description and description != "nan":
                text_parts.append(description)
            if stage and stage != "nan":
                text_parts.append(f"Stage: {stage}")
            combined_text = " | ".join([p for p in text_parts if p])

            # Generate embeddings
            name_emb = cortex.get_embedding(name_text)
            purpose_emb = cortex.get_embedding(purpose_text)
            desc_emb = cortex.get_embedding(desc_text)
            combined_emb = cortex.get_embedding(combined_text)

            self.step_embeddings[sid] = {
                "embedding": combined_emb,  # Default for backward compatibility
                "name_embedding": name_emb,
                "purpose_embedding": purpose_emb,
                "description_embedding": desc_emb,
                "text": combined_text,
                "name_text": name_text,
                "purpose_text": purpose_text,
                "description_text": desc_text,
            }

        self._build_embedding_index()

        with open(self.embed_cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {"_meta": {"excel_fingerprint": self._excel_fingerprint(), "version": "2.0"}, "steps": self.step_embeddings},
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

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        tokens: List[str] = []
        for tok in re.findall(r"[a-z0-9]+(?:[/-][a-z0-9]+)*", text.lower()):
            tokens.append(tok)
            if "/" in tok or "-" in tok:
                for part in re.split(r"[/-]", tok):
                    if part:
                        tokens.append(part)
        return tokens

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for more flexible matching - handles verb tenses, plurals, etc."""
        if not text:
            return ""

        t = text.lower().strip()

        # Common verb normalizations for governance actions
        # Maps past/present/gerund forms to base form
        verb_mappings = {
            "raised": "raise", "raising": "raise", "raises": "raise",
            "created": "create", "creating": "create", "creates": "create",
            "completed": "complete", "completing": "complete", "completes": "complete",
            "delivered": "deliver", "delivering": "deliver", "delivers": "deliver",
            "submitted": "submit", "submitting": "submit", "submits": "submit",
            "obtained": "obtain", "obtaining": "obtain", "obtains": "obtain",
            "finished": "finish", "finishing": "finish", "finishes": "finish",
            "got": "get", "getting": "get", "gets": "get",
            "presented": "present", "presenting": "present", "presents": "present",
            "endorsed": "endorse", "endorsing": "endorse", "endorses": "endorse",
        }

        # Replace verbs with base forms
        for variant, base in verb_mappings.items():
            t = re.sub(r'\b' + variant + r'\b', base, t)

        # Normalize common abbreviations and variations
        t = t.replace("jira", "jira").replace("servicenow", "servicenow")

        return t

    def _build_step_text(self, sid: str, row: pd.Series) -> str:
        text_parts: List[str] = []
        text_parts.append(f"Step {sid}")
        text_parts.append(str(row.get("Step_Name", "")))

        purpose = str(row.get("Purpose", ""))
        if purpose and purpose != "nan":
            text_parts.append(f"Purpose: {purpose}")

        description = str(row.get("Description", ""))
        if description and description != "nan":
            text_parts.append(description)

        stage = str(row.get("Stage_Name", ""))
        if stage and stage != "nan":
            text_parts.append(f"Stage: {stage}")

        return " | ".join([p for p in text_parts if p])

    def _build_lexical_index(self) -> None:
        doc_tokens: Dict[str, set[str]] = {}
        df: Dict[str, int] = {}

        for _, row in self.nodes_df.iterrows():
            sid = str(row["Step_ID"]).strip().upper()
            text = self._build_step_text(sid, row)
            toks = set(self._tokenize(text))
            doc_tokens[sid] = toks
            for tok in toks:
                df[tok] = df.get(tok, 0) + 1

        n_docs = max(1, len(doc_tokens))
        idf: Dict[str, float] = {}
        for tok, count in df.items():
            idf[tok] = float(np.log((n_docs + 1) / (count + 1)) + 1.0)

        self._doc_tokens = doc_tokens
        self._idf = idf

    def lexical_similarity(self, query: str, step_id: str) -> float:
        if not query:
            return 0.0
        q_toks = set(self._tokenize(query))
        if not q_toks:
            return 0.0
        doc = self._doc_tokens.get(step_id)
        if not doc:
            return 0.0

        # Increase from 6 to 20 tokens to handle complex governance queries
        # These queries often have many domain-specific terms that are all important
        ranked = sorted(q_toks, key=lambda t: self._idf.get(t, 1.0), reverse=True)
        top = ranked[: min(20, len(ranked))]

        total = 0.0
        hit = 0.0
        for tok in top:
            w = self._idf.get(tok, 1.0)
            total += w
            if tok in doc:
                hit += w
        if total == 0:
            return 0.0
        return float(hit / total)

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

                for match in re.findall(r"\(([^)]+)\)", name):
                    alias = match.strip()
                    if alias:
                        aliases.add(alias.lower())

                caps = "".join(ch for ch in name if ch.isupper())
                if len(caps) >= 3:
                    aliases.add(caps.lower())

            self.step_aliases[sid] = sorted(aliases)

    def step_index(self, step_id: str) -> int:
        return self._step_index.get(step_id, 10_000)

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
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def deterministic_match(self, text: str) -> Tuple[Optional[str], str, float]:
        if not text:
            return None, "none", 0.0

        t = text.strip().lower()
        if not t:
            return None, "none", 0.0

        # Exact step ID match (highest priority)
        m = re.search(r"\b(s\d{1,3})\b", t)
        if m:
            sid = m.group(1).upper()
            if sid in self.step_embeddings:
                return sid, "id_match", 1.0

        # Alias matching
        for sid, aliases in self.step_aliases.items():
            for alias in aliases:
                if alias in t:
                    return sid, "alias_match", 0.95

        # Normalize both query and step names for flexible matching
        t_normalized = self._normalize_text(t)

        # Try substring matching with normalized text
        best_substr_sid = None
        best_substr_len = 0
        best_match_type = "substring_match"

        for _, row in self.nodes_df.iterrows():
            sid = str(row["Step_ID"]).strip().upper()
            name = str(row["Step_Name"]).strip()
            name_lower = name.lower()
            name_normalized = self._normalize_text(name)

            # Try both original and normalized versions
            # Check if step name appears in query
            if name_lower in t and len(name_lower) > best_substr_len:
                best_substr_sid = sid
                best_substr_len = len(name_lower)
                best_match_type = "substring_match"

            # Check normalized version (handles verb tenses)
            if name_normalized in t_normalized and len(name_normalized) > best_substr_len:
                best_substr_sid = sid
                best_substr_len = len(name_normalized)
                best_match_type = "normalized_match"

            # Also check for significant word overlap (at least 3 consecutive words)
            name_words = name_lower.split()
            if len(name_words) >= 3:
                for i in range(len(name_words) - 2):
                    trigram = " ".join(name_words[i:i+3])
                    if trigram in t and len(trigram) > best_substr_len:
                        best_substr_sid = sid
                        best_substr_len = len(trigram)
                        best_match_type = "partial_match"

        # Lower threshold to 3 characters for better recall
        if best_substr_sid and best_substr_len >= 3:
            # Confidence based on match length and type
            if best_match_type == "substring_match":
                confidence = 0.90
            elif best_match_type == "normalized_match":
                confidence = 0.88
            else:  # partial_match
                confidence = 0.85
            return best_substr_sid, best_match_type, confidence

        return None, "no_match", 0.0

    def map_text_to_step_id(self, text: str, emb_threshold: float = 0.5) -> Tuple[Optional[str], str, float]:
        sid, method, confidence = self.deterministic_match(text)
        if sid:
            return sid, method, confidence

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

        # Fallback for missing index or old cache format
        if self._embedding_matrix is None or self._embedding_norms is None:
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

        # Use the main embedding index for initial filtering
        dots = self._embedding_matrix @ q
        sims = dots / (self._embedding_norms * qn)

        pool_k = max(1, top_k * 5)
        idx = np.argsort(-sims)[:pool_k]

        out: List[Dict[str, Any]] = []
        for i in idx:
            sid = self._embedding_ids[int(i)]
            step_data = self.step_embeddings.get(sid, {})

            # Calculate similarity against all embedding fields
            combined_sim = float(sims[int(i)])

            # Multi-field similarity: check name, purpose, and description separately
            name_sim = 0.0
            purpose_sim = 0.0
            desc_sim = 0.0

            if "name_embedding" in step_data:
                name_sim = self.cosine_similarity(query_emb, step_data["name_embedding"])
            if "purpose_embedding" in step_data:
                purpose_sim = self.cosine_similarity(query_emb, step_data["purpose_embedding"])
            if "description_embedding" in step_data:
                desc_sim = self.cosine_similarity(query_emb, step_data["description_embedding"])

            # Take the maximum similarity across fields (best match wins)
            # This allows matching on ANY aspect: name, purpose, or description
            max_field_sim = max(name_sim, purpose_sim, desc_sim, combined_sim)

            # Weighted combination: best field match gets priority, but combine all signals
            # 40% best field, 30% combined, 15% each for other top fields
            sorted_sims = sorted([name_sim, purpose_sim, desc_sim], reverse=True)
            multi_field_score = (
                0.40 * max_field_sim +
                0.30 * combined_sim +
                0.15 * sorted_sims[0] +
                0.15 * sorted_sims[1]
            )

            # Lexical similarity for exact keyword matches
            lex_score = self.lexical_similarity(text, sid)

            # Final score: balance semantic and lexical (50/50 instead of 75/25)
            # Lexical is more important for governance-specific terminology
            final_score = (0.50 * multi_field_score) + (0.50 * lex_score)

            out.append(
                {
                    "id": sid,
                    "score": final_score,
                    "embedding_score": multi_field_score,
                    "lexical_score": lex_score,
                    "name_similarity": name_sim,
                    "purpose_similarity": purpose_sim,
                    "description_similarity": desc_sim,
                    "text": step_data.get("text", ""),
                }
            )

        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]

    def get_outgoing_edges(self, step_id: str) -> List[Dict]:
        edges = self.edges_df[self.edges_df["from"] == step_id]
        return edges.to_dict("records")

    def get_incoming_edges(self, step_id: str) -> List[Dict]:
        edges = self.edges_df[self.edges_df["to"] == step_id]
        return edges.to_dict("records")


_governance_data_singleton: Optional[GovernanceDataLoader] = None


def get_governance_data() -> GovernanceDataLoader:
    global _governance_data_singleton
    if _governance_data_singleton is None:
        _governance_data_singleton = GovernanceDataLoader()
    return _governance_data_singleton

