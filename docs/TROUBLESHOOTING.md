# Troubleshooting

## Missing `constants.py` or `cortex_connection.py`

Symptoms:
- `ModuleNotFoundError: No module named 'constants'`
- `ModuleNotFoundError: No module named 'cortex_connection'`

Fix:
- Ensure these modules are on `PYTHONPATH` (or installed) in the environment where you run the app.

Expected minimum contract:
- `constants.EXCEL_PATH`: path to `nodes_edges_governance.xlsx`
- `constants.EMBED_CACHE_PATH`: path to embeddings cache JSON
- `cortex_connection.cortex.get_embedding(text) -> list[float]`
- `cortex_connection.cortex.get_chat_response(messages, ...) -> str | OpenAI-like response`

## Excel read errors (openpyxl)

Symptoms:
- `ImportError: Missing optional dependency 'openpyxl'`

Fix:
- `pip install openpyxl`

## Debug candidate scoring

To print top-5 candidates with embedding/lexical breakdown:

- `GOV_DEBUG_MATCHING=1 python3 main.py`

## Cache is stale / wrong matches after workbook edits

The embeddings cache includes an “excel fingerprint” (mtime/size). If you move files or edit in ways that preserve fingerprint, you may need to rebuild:

- `python3 scripts/build_governance_artifacts.py`

## Streamlit conversation titles look generic

The UI tries (in order):
1) deterministic step match → `Sxx: Step Name`
2) detect name validation → `Name Validation`
3) LLM-generated title → sanitized and rejected if too generic
4) fallback heuristic title from the user message

If your Cortex wrapper returns non-text objects, ensure `core/cortex_utils.cortex_chat_text` can extract `content`.

