from typing import List, Dict
from src.adapters.embeddings import get_embeddings
from src.adapters.vectorstores import get_vector_store
from src.utils.config import settings

def upsert_docs_record(doc: Dict) -> bool:
    import lancedb
    db = lancedb.connect(settings.store_path)
    table = "docs"

    tbl = db.open_table(table) if table in db.table_names() else None
    if tbl is None:
        tbl = db.create_table(table, data=[doc])
        return True  # newly created

    # Check if doc_id already exists
    existing = tbl.to_pandas()
    existing_doc = existing[existing["doc_id"] == doc["doc_id"]]

    if not existing_doc.empty:
        if existing_doc.iloc[0]["sha256"] == doc["sha256"]:
            return False  # no changes, skip re-indexing
        else:
            tbl.delete(f"doc_id = '{doc['doc_id']}'")

    tbl.add([doc])
    return True  # doc updated


def index_chunks(doc: Dict, chunk_records: List[Dict]):
    # Check doc first — skip if unchanged
    updated = upsert_docs_record(doc)
    if not updated:
        print(f"[SKIPPED] Document already indexed with same content: {doc['title']}")
        return

    # Embed and index
    emb = get_embeddings(settings)
    for c in chunk_records:
        c["vector"] = emb.embed(c["text"])

    store = get_vector_store(settings.vector_store, settings.store_path)
    # Delete old chunks if any
    if hasattr(store, "tbl") and store.tbl is not None:
        store.tbl.delete(f"doc_id = '{doc['doc_id']}'")

    store.add(chunk_records)
    print(f"[INDEXED] {doc['title']} with {len(chunk_records)} chunks")
