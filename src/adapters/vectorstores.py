from typing import List, Dict, Any
import numpy as np

class VectorStore:
    def add(self, items: List[Dict[str, Any]]) -> None: ...
    def query(self, vector: List[float], k: int = 5) -> List[Dict[str, Any]]: ...

class LanceDBStore(VectorStore):
    def __init__(self, uri: str, table: str = "chunks"):
        import lancedb
        self.db = lancedb.connect(uri)
        self.table_name = table
        self.tbl = self.db.open_table(table) if table in self.db.table_names() else None

    def _ensure_table(self, items: List[Dict[str, Any]]):
        # Create table from the first batch if it doesn't exist
        if self.tbl is None:
            # Convert vectors to float32 BEFORE creating table so schema infers correctly
            for it in items:
                it["vector"] = np.array(it["vector"], dtype=np.float32)
            self.tbl = self.db.create_table(self.table_name, data=items)
            return True
        return False

    def add(self, items: List[Dict[str, Any]]) -> None:
        if not items:
            return
        # ensure consistent dtype
        for it in items:
            it["vector"] = np.array(it["vector"], dtype=np.float32)

        created_now = self._ensure_table(items)
        if not created_now:
            self.tbl.add(items)

    def query(self, vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        import numpy as np
        if self.tbl is None:
            return []
        return self.tbl.search(np.array(vector, dtype=np.float32)).limit(k).to_list()

def get_vector_store(name: str, uri: str):
    name = (name or "").lower()
    if name == "lancedb":
        return LanceDBStore(uri)
    raise ValueError(f"Unknown vector store: {name}")
