"""
Embedding utilities for semantic matching.
"""
from typing import List
import numpy as np

from core.cortex_client import CortexEmbeddings


class EmbeddingManager:
    """Manages embeddings and similarity calculations"""
    
    def __init__(self, embeddings_client: CortexEmbeddings):
        self.client = embeddings_client
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        text = (text or "").replace("\n", " ")
        return self.client.embed_query(text)
    
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)
    
    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: dict[str, List[float]],
        top_k: int = 3,
        threshold: float = 0.0
    ) -> List[tuple[str, float]]:
        """
        Find most similar candidates to query.
        
        Args:
            query_embedding: Query vector
            candidate_embeddings: Dict of {id: embedding}
            top_k: Number of results to return
            threshold: Minimum similarity score
        
        Returns:
            List of (id, score) tuples sorted by score descending
        """
        scores = []
        for candidate_id, candidate_emb in candidate_embeddings.items():
            score = self.cosine_similarity(query_embedding, candidate_emb)
            if score >= threshold:
                scores.append((candidate_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
