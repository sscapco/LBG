# File-by-File Guide

This file documents what each file does (excluding `.venv/`, `graveyard/`, and `openai_version/`).

## Entry Points

- `main.py`
  - Demo harness (sync/async demos + interactive mode).
  - Calls into `core.workflow`.

- `streamlit_cortex.py`
  - Streamlit UI.
  - Handles conversation history, titles, and rendering.
  - Calls `core.workflow.process_query_sync()` for responses.

## Core Package (`core/`)

- `core/__init__.py`
  - Convenience exports for the main query functions.

- `core/workflow.py`
  - Defines the LangGraph and session storage.
  - Entry points: `process_query()` (async), `process_query_sync()` (sync).

- `core/state.py`
  - Defines the runtime state (`GovernanceState`) and persisted session shape (`SessionState`).
  - Helpers: `create_initial_state`, `state_to_session_state`, `session_state_to_previous_state`.

- `core/query_analyzer.py`
  - Intent classification and step selection.
  - Runs deterministic matching first, then semantic+lexical reranking.
  - Sets `needs_disambiguation`, `focus_step_id`, `next_step_ids`, `automation_step`, etc.

- `core/response_generator.py`
  - Builds final text answers using the LLM.
  - Handles: greeting, disambiguation, step explanation, next-step guidance, fallback.

- `core/governance_data.py`
  - Loads and normalizes `nodes_edges_governance.xlsx`.
  - Builds step aliases and embeddings cache.
  - Provides matching primitives (`deterministic_match`, `semantic_candidates`).
  - Exposes singleton accessor `get_governance_data()`.

- `core/automation_handler.py`
  - Runs automations inside the graph.
  - Detects “name validation” requests and extracts params.
  - Calls the registry and formats results into user-visible messages.

- `core/automation_registry.py`
  - Registry of available automations.
  - Currently includes `validate_name` (Data Product name checker).

- `core/cortex_utils.py`
  - `cortex_chat_text()`: normalizes different response shapes into plain text.
  - `parse_json_object()`: strict JSON extraction/validation for LLM outputs.

- `core/prompts.py`
  - Prompt builders used by query analysis and response generation.
  - Keeps the rest of the code free of long embedded prompt strings.

## Automation Tools (`automation_tools/`)

- `automation_tools/handler.py`
  - Implements name validation logic:
    - universal deterministic checks
    - ODP/FDP/CDP non-LLM grammar checks
    - LLM-based linguistic review with guardrails
    - “connections” warnings (not verified)
  - Exposes `check_name_both(...)` which returns a structured result.

- `automation_tools/prompts/*.txt`
  - Prompt templates used by the name checker (generic + dp-type specific).

- `automation_tools/llms.py`
  - Cortex API client used by automation tools (separate from `cortex_connection`).
  - Reads config via `automation_tools/config.py`.

- `automation_tools/config.py`
  - Loads YAML config (`automation_tools/app.yaml`).
  - Imports `constants` from the parent path to get Cortex credentials.

- `automation_tools/common.py`
  - Interfaces/ABCs for LLM/Embeddings/VectorStore (minimal scaffolding).

- `automation_tools/manifest.YAML`
  - Metadata manifest for the naming checker tool (id, schema, prompts, routing hints).

## Scripts

- `scripts/build_governance_artifacts.py`
  - “Offline” embeddings cache builder.
  - Constructs `core.governance_data.GovernanceDataLoader`, which writes the cache.

## Top-Level Compatibility Shims

These files re-export symbols from `core/` so older imports keep working:

- `workflow.py`
- `query_analyzer.py`
- `response_generator.py`
- `automation_handler.py`
- `automation_registry.py`
- `state.py`
- `cortex_utils.py`
- `governance_data.py`

