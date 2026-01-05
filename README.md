# Governance Pipeline (Workflow Q&A + Automation)

This project answers questions about an internal governance workflow and can run workflow-linked automations (currently: data product name validation). The workflow “source of truth” lives in `nodes_edges_governance.xlsx` and is treated as a directed graph (nodes = steps, edges = transitions).

## What It Does

- **Q&A over workflow steps**: “Tell me about S21”, “What do I do next?”, “What’s the purpose of the DOI step?”
- **Navigation**: given a current step, returns the next steps (supports branching).
- **Disambiguation**: when a question could refer to multiple steps, it presents candidates and asks clarifying questions.
- **Automation**: detects automation requests (e.g., “Run the naming check…”) and executes a registered tool.

## External Dependencies (Not In This Repo)

This repo imports (but does not include) these modules:

- `constants.py` (expected to provide at least `EXCEL_PATH`, `EMBED_CACHE_PATH`, and Cortex API values used by `automation_tools/config.py`)
- `cortex_connection.py` (expected to provide `cortex` with:
  - `cortex.get_embedding(text: str) -> list[float]`
  - `cortex.get_chat_response(messages: list[dict], **kwargs) -> str | OpenAI-like response object`)

If these aren’t on your `PYTHONPATH`, the pipeline won’t start.

## Quickstart (Local)

Install deps:
- `pip install -r requirements.txt`

Run the console demo:
- `python3 main.py`

Run Streamlit UI:
- `streamlit run streamlit_cortex.py`

Optional: build/update embeddings cache (offline build step):
- `python3 scripts/build_governance_artifacts.py`

## Repository Structure (High-Level)

- `core/`: core pipeline logic (graph, matching, response generation, automation orchestration).
- `automation_tools/`: implementation of the “name checker” automation (rules + LLM linguistic review prompts).
- `streamlit_cortex.py`: Streamlit UI (thin UI layer calling `core.workflow.process_query_sync`).
- `main.py`: CLI/demo harness.
- `nodes_edges_governance.xlsx`: workflow graph data (Nodes + Edges sheets).

For details:
- `docs/ARCHITECTURE.md`
- `docs/FILES.md`
- `docs/EXCEL_SCHEMA.md`
- `docs/TROUBLESHOOTING.md`

