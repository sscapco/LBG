from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    @abstractmethod
    def add(self, embeddings: List[List[float]], docs: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def query(self, embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        pass

class Embeddings(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass

class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
