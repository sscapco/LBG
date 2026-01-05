# Architecture & Logic

## Data Model: Workflow Graph

The workflow is represented as a directed graph:

- **Nodes**: steps (`Step_ID` like `S21`) with `Stage`, `Stage_Name`, `Step_Name`, `Purpose`, and `Description`.
- **Edges**: transitions (`from`, `to`) plus metadata such as `edge_type` and `guard_condition`.

The authoritative source is `nodes_edges_governance.xlsx`:
- sheet `Nodes`
- sheet `Edges`

`core/governance_data.py` loads the workbook, normalizes columns, and provides query-time APIs for:
- step lookup (`get_step_record`)
- next steps (`get_outgoing_edges` + sorting via canonical node order)

## Runtime Flow (LangGraph)

The pipeline is a LangGraph state machine (`core/workflow.py`):

1. `analyze_query` (`core/query_analyzer.py`)
2. `automation_handler` (`core/automation_handler.py`)
3. `response_generator` (`core/response_generator.py`)

The graph runs the same way for sync and async entrypoints:
- `core.workflow.process_query_sync()` → `governance_graph.invoke(...)`
- `core.workflow.process_query()` → `governance_graph.ainvoke(...)`

## State

`core/state.py` defines:
- `GovernanceState`: per-turn state (message, intent, matched step, next steps, answer, etc.).
- `SessionState`: persisted (in-memory) snapshot stored in `SESSION_STORE`.

The `SESSION_STORE` is an in-memory dict keyed by `session_id`. It enables “what’s next?” style questions to use prior context.

## Query Analysis Logic

`core/query_analyzer.py` does:

### 1) Intent Classification

It asks the LLM to classify into:
- `greeting`
- `ask_about_step`
- `ask_next_step`
- `ask_previous_step`
- `mark_complete`
- `mark_in_progress`
- `automation_request`
- `clarification_needed`
- `unknown`

The output is parsed strictly as JSON using `core/cortex_utils.parse_json_object`.
If parsing fails, a keyword-based fallback classifier runs.

### 2) Step Identification

Step matching is hybrid:

**Deterministic match** (`GovernanceDataLoader.deterministic_match`):
- explicit `S##` ID
- alias match (derived from step names and parentheses/acronyms)
- substring match on step name

**Semantic match** (`GovernanceDataLoader.semantic_candidates`):
- embedding similarity between user query and cached step embeddings
- lexical reranking: an IDF-weighted overlap signal using tokens from `Step_Name`, `Purpose`, `Description`, and `Stage`

The returned candidates include:
- `embedding_score`
- `lexical_score`
- `score` = blended score used for sorting

### 3) Disambiguation vs Single-Step Selection

When the top-2 candidates are close, the system may set `needs_disambiguation=True` and provide top 3 candidates to the response generator.

To avoid “looks ambiguous but isn’t” cases, a conservative lexical tie-break triggers if the top candidate has clearly stronger lexical evidence than #2.

### 4) Navigation (“what’s next?”)

If intent is `ask_next_step` and the user did not mention a step, the analyzer uses the previous session’s anchor/focus step to compute next steps.

## Response Generation Logic

`core/response_generator.py` produces the final answer:

- If `needs_disambiguation`: call LLM with candidate step details and ask clarifying questions.
- If a step is selected: call LLM with step details and instruct it to quote the `Description` verbatim as bullet points.
- If `ask_next_step`: return a summary of downstream steps.
- Otherwise: fallback clarification prompt.

All LLM responses are normalized to text via `core/cortex_utils.cortex_chat_text` to tolerate different wrapper return types.

## Automation

Automation has two layers:

### Orchestration (Core)

`core/automation_handler.py`:
- runs only when intent is `automation_request`
- resolves which automation to run:
  - from the matched step’s `automation_step`, or
  - inferred from the message (currently: name validation)
- extracts parameters from free-form text
- invokes the registry and formats a user-friendly result

### Implementation (Automation Tools)

`automation_tools/handler.py` implements the name checker:
- deterministic rules (universal + type-specific grammar for ODP/FDP/CDP)
- LLM linguistic review using prompt templates in `automation_tools/prompts/`
- “connections checks” output as warnings (not verified)

The automation registry is in `core/automation_registry.py` (exposed via top-level shim `automation_registry.py`).

## Caching / “Offline Build Step”

Workflow embeddings are cached at `constants.EMBED_CACHE_PATH`.

- `scripts/build_governance_artifacts.py` can be used to prebuild the cache.
- The cache stores an Excel fingerprint (mtime/size) to avoid silently using stale embeddings when the workbook changes.

