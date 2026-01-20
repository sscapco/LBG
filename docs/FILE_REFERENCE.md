# File Reference

Quick reference for the codebase structure.

---

## Core Module (`core/`)

### `governance_data.py`
**Purpose:** Data loader and search engine

**Main Class:** `GovernanceDataLoader`
- Loads Excel data (`cjm_nodes_edges.xlsx`)
- Generates/caches embeddings
- Provides deterministic and semantic search
- Filters NAN rows at load time (lines 75-76)

**Key Methods:**
- `deterministic_match(text)` - ID/alias/substring matching
- `semantic_candidates(text, top_k)` - Hybrid semantic+lexical search
- `lexical_similarity(query, step_id)` - TF-IDF token overlap
- `get_step_record(step_id)` - Fetch step details
- `get_outgoing_edges(step_id)` / `get_incoming_edges(step_id)` - Navigation

**Scoring:** `final_score = (2.0 * multi_field_score + lex_score) / 3.0` (line 413)

---

### `query_analyzer.py`
**Purpose:** Query analysis and step matching

**Main Function:** `analyze_query_node(state)` - Entry point for query processing

**Key Functions:**
- `_classify_intent()` - Determine user intent (greeting, ask_about_step, etc.)
- `_identify_steps()` - Three-tier matching system
- `_llm_validate_scope()` - LLM validation for scores 0.35-0.60

**Critical Thresholds:**
- Embedding: >= 0.5 (line 225)
- Lexical: >= 0.3 (line 225)
- High confidence bypass: >= 0.60 with gap >= 0.15 (line 241)
- LLM validation range: 0.35-0.60 (line 250)

**Intent Types:** greeting, ask_about_step, compare_steps, ask_next_step, automation_request, mark_complete, mark_in_progress

**Debug:** Set `GOV_DEBUG_MATCHING=1` for detailed logs

---

### `response_generator.py`
**Purpose:** Generate natural language responses

**Main Function:** `response_generator_node(state)`

**Response Types:**
- Comparison (2+ referenced IDs)
- Greeting
- Disambiguation (multiple candidates)
- Step explanation (matched step)
- Next step guidance
- Fallback (no match)

**LLM Settings:**
- Disambiguation/Step: 2500 tokens, temp 0.0
- Next step: 1500 tokens, temp 0.0
- Comparison: 2000 tokens, temp 0.0
- Fallback: 800 tokens, temp 0.3

---

### `prompts.py`
**Purpose:** LLM prompt templates

**Key Prompts:**
- `classify_intent_prompt()` - Intent classification
- `validate_scope_prompt()` - Scope validation (hides scores from LLM)
- `step_response_prompt()` - Step explanation
- `disambiguation_prompt()` - Multiple candidate clarification
- `compare_steps_prompt()` - Step comparison
- `fallback_prompt()` - Generic fallback

---

### `state.py`
**Purpose:** State management

**Main Type:** `GovernanceState` (TypedDict)

**Key Fields:** user_message, intent, focus_step_id, candidate_step_ids, match_method, match_confidence, needs_disambiguation, step_details, answer

---

### `workflow.py`
**Purpose:** LangGraph workflow

**Nodes:** analyze_query → handle_automation (conditional) → generate_response

---

### `automation_handler.py` / `automation_registry.py`
**Purpose:** Execute and register automations

Current automations: naming_validation

---

### `cortex_utils.py`
**Purpose:** LLM interaction utilities

Functions: `cortex_chat_text()`, `parse_json_object()`

---

## Automation Tools (`automation_tools/`)

### `config.py`
Settings management - loads from environment variables

### `llms.py`
LLM interface - wraps Vertex AI via Cortex proxy

### `handler.py` / `common.py`
Automation execution and base classes

---

## Entry Points

### `main.py`
CLI interface - `python main.py`

### `streamlit_cortex.py`
Web UI - `streamlit run streamlit_cortex.py`

---

## Debug/Test Scripts

### `debug_single_query.py`
Test single query with debug output
```bash
export GOV_DEBUG_MATCHING=1
python debug_single_query.py "Your query"
```

### Test Scripts (No Longer in Repo)
- `test_changes.py` - Regression testing
- `test_intent_classification.py` - Intent testing
- `verify_nan_fix.py` - NAN filtering verification
- `verify_code_version.py` - Code version check

---

## Data Files

### `cjm_nodes_edges.xlsx`
Governance workflow data - 2 sheets: Nodes (steps) and Edges (connections)

### `embeddings_cache.json`
Cached embeddings (version 2.0) - regenerates when Excel modified

---

## Quick Navigation

**Modify matching logic?**
→ [governance_data.py](core/governance_data.py), [query_analyzer.py](core/query_analyzer.py)

**Change thresholds?**
→ [query_analyzer.py](core/query_analyzer.py) lines 225, 241, 250

**Update prompts?**
→ [prompts.py](core/prompts.py)

**Add automation?**
→ [automation_registry.py](core/automation_registry.py), [handler.py](automation_tools/handler.py)

**Debug matching?**
→ `GOV_DEBUG_MATCHING=1` + [debug_single_query.py](debug_single_query.py)
