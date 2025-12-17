"""
Embedding cache management to avoid recomputing embeddings.
"""
import json
from typing import Dict, List
from pathlib import Path

from config import DataConfig
from core.embeddings import EmbeddingManager
from data.loader import GovernanceDataLoader


class EmbeddingCache:
    """Manages persistent embedding cache"""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        data_loader: GovernanceDataLoader,
        cache_path: Optional[Path] = None
    ):
        self.embedding_manager = embedding_manager
        self.data_loader = data_loader
        self.cache_path = cache_path or DataConfig.EMBEDDING_CACHE_PATH
        self.embeddings: Dict[str, Dict] = {}
        self._load_or_compute()
    
    def _load_or_compute(self):
        """Load from cache or compute embeddings"""
        if self.cache_path.exists():
            self._load_from_cache()
        else:
            print("ℹ️ No embeddings cache found. Computing embeddings...")
            self._compute_all_embeddings()
    
    def _load_from_cache(self):
        """Load embeddings from cache file"""
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            
            for sid, data in raw.items():
                self.embeddings[str(sid).upper()] = {
                    "embedding": data["embedding"],
                    "text": data.get("text", ""),
                }
            
            print(f"✅ Loaded {len(self.embeddings)} step embeddings from cache")
        except Exception as e:
            print(f"⚠️ Failed to load embeddings cache: {e}. Recomputing...")
            self.embeddings = {}
            self._compute_all_embeddings()
    
    def _compute_all_embeddings(self):
        """Compute embeddings for all steps"""
        for step_id in self.data_loader.get_all_step_ids():
            text = self.data_loader.get_step_text_for_embedding(step_id)
            if not text:
                continue
            
            embedding = self.embedding_manager.get_embedding(text)
            self.embeddings[step_id] = {
                "embedding": embedding,
                "text": text,
            }
        
        # Save to cache
        self._save_to_cache()
        print(f"✅ Computed and cached embeddings for {len(self.embeddings)} steps")
    
    def _save_to_cache(self):
        """Save embeddings to cache file"""
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.embeddings, f)
    
    def get_embedding(self, step_id: str) -> Optional[List[float]]:
        """Get embedding for a step"""
        data = self.embeddings.get(step_id.upper())
        return data["embedding"] if data else None
    
    def get_all_embeddings(self) -> Dict[str, List[float]]:
        """Get all embeddings as dict {step_id: embedding}"""
        return {
            sid: data["embedding"]
            for sid, data in self.embeddings.items()
        }
    
    def get_step_text(self, step_id: str) -> Optional[str]:
        """Get the text used for embedding"""
        data = self.embeddings.get(step_id.upper())
        return data["text"] if data else None
