# File Reference Guide

Complete reference for every file in the Governance Pipeline project.

---

## Core Module (`core/`)

### `core/governance_data.py`

**Purpose:** Data loader and search engine for governance steps.

**Main Class:** `GovernanceDataLoader`

**Responsibilities:**
1. Load governance data from Excel (`Governance_Process_Updated.xlsx`)
2. Generate and cache multi-field embeddings
3. Provide search interfaces (deterministic, semantic, lexical)
4. Build and maintain search indices (embedding matrix, TF-IDF)

**Key Methods:**

| Method | Purpose | Returns |
|--------|---------|---------|
| `__init__()` | Load Excel, filter NAN rows, load/generate embeddings | - |
| `deterministic_match(text)` | ID/alias/substring matching | (step_id, method, confidence) |
| `semantic_candidates(text, top_k)` | Hybrid semantic+lexical search | List[{id, score, embedding_score, lexical_score, ...}] |
| `best_step_by_embedding(text)` | Pure embedding similarity (legacy) | (step_id, score) |
| `lexical_similarity(query, step_id)` | TF-IDF token overlap | float (0.0-1.0) |
| `get_step_record(step_id)` | Fetch full step details | Dict or None |
| `get_outgoing_edges(step_id)` | Get next steps | List[Dict] |
| `get_incoming_edges(step_id)` | Get previous steps | List[Dict] |

**Important Data Structures:**

```python
self.nodes_df: pd.DataFrame  # Filtered nodes (NAN removed)
self.edges_df: pd.DataFrame  # Workflow edges
self.step_embeddings: Dict[str, Dict]  # {step_id: {name_embedding, purpose_embedding, ...}}
self.step_aliases: Dict[str, List[str]]  # {step_id: [alias1, alias2, ...]}
self.ordered_step_ids: List[str]  # All valid step IDs in order
self._embedding_matrix: np.ndarray  # Pre-computed embedding matrix
self._doc_tokens: Dict[str, set]  # {step_id: {token1, token2, ...}}
self._idf: Dict[str, float]  # {token: idf_score}
```

**Critical Filtering:**
```python
# Line 75-76: Remove NAN rows at load time
valid_mask = (self.nodes_df["Step_ID"] != "NAN") & (self.nodes_df["Step_ID"].notna())
self.nodes_df = self.nodes_df[valid_mask].reset_index(drop=True)
```

**Embedding Cache:**
- File: `embeddings_cache.json`
- Version: 2.0
- Invalidates on Excel file modification
- Structure: `{"_meta": {...}, "steps": {...}}`

**Scoring Formula:**
```python
# Line 413: Final score calculation
final_score = (2.0 * multi_field_score + lex_score) / 3.0
```

---

### `core/query_analyzer.py`

**Purpose:** Analyze user queries to extract intent and match steps.

**Main Function:** `analyze_query_node(state: GovernanceState) -> GovernanceState`

**Workflow:**
1. Extract explicit step references (regex `\bS\d{1,3}\b`)
2. Classify intent via `_classify_intent()`
3. Handle navigation intents (ask_next_step)
4. Match steps via `_identify_steps()`
5. Set disambiguation flag if multiple candidates
6. Enrich state with step details

**Key Functions:**

| Function | Purpose | Returns |
|----------|---------|---------|
| `_classify_intent(message, prev_state)` | Determine user intent | (intent, confidence, metadata) |
| `_identify_steps(message, prev_state, intent)` | Match query to steps | (focus_id, candidates, method, confidence) |
| `_llm_validate_scope(message, candidates)` | LLM scope validation | (step_ids, validation_result, confidence) |
| `_llm_rerank_candidates(message, candidates)` | LLM reranking (legacy) | (best_id, ambiguous_ids) |
| `_extract_comparison_terms(message)` | Extract A vs B comparisons | (term_a, term_b) or None |
| `_map_term_to_step_id(term, gov_data)` | Resolve term to step ID | step_id or None |

**Intent Values:**
- `greeting`: "Hello", "Hi"
- `ask_about_step`: Asking about or describing work (MOST COMMON)
- `compare_steps`: "What's the difference between X and Y?"
- `ask_next_step`: "What comes next?"
- `ask_previous_step`: "What came before?"
- `automation_request`: "Run naming validation"
- `mark_complete`, `mark_in_progress`: Progress tracking
- `clarification_needed`, `unknown`: Fallbacks

**LLM Calls:**

1. **Intent Classification** (Line 132-139)
   - Model: Gemini-2.5-Flash
   - Max tokens: 200
   - Temperature: 0.0
   - Fallback: Heuristics if fails

2. **LLM Validation** (Line 372-379)
   - Model: Gemini-2.5-Flash
   - Max tokens: 1500
   - Temperature: 0.0
   - Fallback: Return "no_match" if fails

**Debug Logging:**
Set `GOV_DEBUG_MATCHING=1` to see:
- Semantic candidates with scores (Line 211-218)
- LLM validation inputs/outputs (Line 361-384)
- Decision results (Line 398-421)

**Critical Logic:**

```python
# Line 223-226: Threshold filtering
valid_candidates = [
    c for c in candidates
    if c.get("embedding_score", 0.0) >= 0.5
    and c.get("lexical_score", 0.0) >= 0.3
]

# Line 240-242: High confidence bypass
if c1["score"] >= 0.60 and score_gap >= 0.15:
    return c1["id"], [c1["id"]], "high_confidence_match", c1["score"]

# Line 250-251: LLM validation for borderline scores
if 0.35 <= c1["score"] < 0.60:
    validated_ids, validation_result, llm_confidence = _llm_validate_scope(...)
```

---

### `core/response_generator.py`

**Purpose:** Generate natural language responses based on query analysis results.

**Main Function:** `response_generator_node(state: GovernanceState) -> GovernanceState`

**Response Routing:**

```python
if len(referenced_ids) >= 2:
    # Comparison of two steps
    return _generate_comparison_response(state, referenced_ids[:2])

elif intent == "greeting":
    return _generate_greeting_response()

elif needs_disambiguation:
    # Multiple candidates - ask user to clarify
    return _generate_disambiguation_response(state)

elif focus_step_id:
    # Explain the matched step
    return _generate_step_response(state)

elif intent == "ask_next_step":
    # Guide to next steps
    return _generate_next_step_response(state)

else:
    # No match - context-aware fallback
    return _generate_fallback_response(state, match_method)
```

**Key Functions:**

| Function | Purpose | Max Tokens | Temp |
|----------|---------|-----------|------|
| `_generate_greeting_response()` | Welcome message | - | - |
| `_generate_disambiguation_response(state)` | Explain multiple candidates | 2500 | 0.0 |
| `_generate_step_response(state)` | Explain single step + next steps | 2500 | 0.0 |
| `_generate_next_step_response(state)` | Guide to what comes next | 1500 | 0.0 |
| `_generate_comparison_response(state, step_ids)` | Compare two steps | 2000 | 0.0 |
| `_generate_fallback_response(state, match_method)` | Context-aware rejection | 800 | 0.3 |

**Fallback Message Routing:**

| match_method | Message Type |
|--------------|--------------|
| `out_of_scope` | "I'm a governance workflow assistant..." |
| `below_threshold` | "I couldn't find a step that matches..." |
| `no_match_validated` | "I understand you're asking about governance..." |
| Other | LLM-generated generic (800 tokens) |

**LLM Calls:**
- All use Vertex AI Gemini-2.5-Flash
- All use temperature=0.0 (except generic fallback: 0.3)
- Token limits increased to handle Gemini-2.5-Flash's capabilities

---

### `core/prompts.py`

**Purpose:** LLM prompt templates for all LLM interactions.

**Functions:**

| Function | Purpose | Used By |
|----------|---------|---------|
| `classify_intent_prompt(message, prev_focus)` | Intent classification | `query_analyzer._classify_intent()` |
| `step_response_prompt(message, step, next_steps, automation, intent)` | Step explanation | `response_generator._generate_step_response()` |
| `next_step_prompt(message, next_steps, anchor)` | Next step guidance | `response_generator._generate_next_step_response()` |
| `disambiguation_prompt(message, candidates)` | Disambiguate between steps | `response_generator._generate_disambiguation_response()` |
| `compare_steps_prompt(message, steps)` | Compare two steps | `response_generator._generate_comparison_response()` |
| `fallback_prompt(message)` | Generic fallback | `response_generator._generate_fallback_response()` |
| `validate_scope_prompt(message, candidates)` | LLM scope validation | `query_analyzer._llm_validate_scope()` |
| `rerank_candidates_prompt(message, candidates)` | LLM reranking | `query_analyzer._llm_rerank_candidates()` |

**Critical Prompt:** `validate_scope_prompt()`

**Key Design Decisions:**
1. **Hides numeric scores** from LLM (Line 197-205)
   - Prevents threshold bias
   - LLM judges on content, not numbers

2. **Balanced instructions** (Line 226-231)
   - "Trust that candidates were pre-filtered"
   - "Focus on content matching, not perfect wording"
   - "Governance queries are often informal - that's OK"

3. **Clear scope definition** (Line 215-217)
   - IN_SCOPE: governance activities, deliverables, approvals
   - OUT_OF_SCOPE: weather, general tech, greetings

4. **Encourages AMBIGUOUS** (Line 223, 231)
   - "Return AMBIGUOUS if 2-3 steps genuinely apply"
   - "When in doubt, return AMBIGUOUS with multiple IDs"

---

### `core/state.py`

**Purpose:** State type definitions and initialization.

**Main Type:** `GovernanceState` (TypedDict)

**State Fields:**

```python
{
    # Input
    "user_message": str,
    "session_id": str,
    "previous_state": Optional[Dict],

    # Query Analysis
    "intent": str,
    "query_analysis": Optional[Dict],
    "focus_step_id": Optional[str],
    "candidate_step_ids": List[str],
    "referenced_ids": List[str],
    "match_method": str,
    "match_confidence": float,
    "needs_disambiguation": bool,

    # Step Details
    "step_details": Optional[Dict],
    "next_step_ids": List[str],
    "anchor_step_id": Optional[str],

    # Automation
    "automatable_step_id": Optional[str],
    "automation_step": Optional[str],
    "automation_result": Optional[Dict],

    # Output
    "answer": str,
    "explained_step_ids": List[str],
}
```

**Function:** `create_initial_state(message, session_id, previous_state=None) -> GovernanceState`

---

### `core/workflow.py`

**Purpose:** LangGraph workflow definition.

**Main Object:** `governance_graph`

**Nodes:**
1. `analyze_query`: `query_analyzer.analyze_query_node`
2. `handle_automation`: `automation_handler.automation_handler_node`
3. `generate_response`: `response_generator.response_generator_node`

**Conditional Edges:**

```python
def should_run_automation(state):
    if state.get("automation_step") and not state.get("automation_result"):
        return "handle_automation"
    return "generate_response"

graph.add_conditional_edges(
    "analyze_query",
    should_run_automation,
    {
        "handle_automation": "handle_automation",
        "generate_response": "generate_response"
    }
)
```

**Entry Point:** `analyze_query`
**Exit Point:** `generate_response`

---

### `core/automation_handler.py`

**Purpose:** Execute automations for steps that support them.

**Main Function:** `automation_handler_node(state: GovernanceState) -> GovernanceState`

**Logic:**
```python
if state.get("automation_result"):
    return state  # Already ran

automation_step = state.get("automation_step")
if not automation_step:
    return state  # No automation for this step

# Execute automation via automation_tools
handler = automation.get_handler(automation_step)
result = handler.execute(params)
state["automation_result"] = result
```

---

### `core/automation_registry.py`

**Purpose:** Registry of available automations.

**Functions:**
- `get_automation_info(automation_name)`: Get metadata
- `register_automation(name, handler, display_name, description)`: Register new automation

**Current Automations:**
- `naming_validation`: Validate artifact names against conventions

---

### `core/cortex_utils.py`

**Purpose:** Utilities for interacting with Cortex (LLM wrapper).

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `cortex_chat_text(response)` | Extract text from LLM response |
| `parse_json_object(text, schema)` | Parse and validate JSON from LLM |

**JSON Parsing:**
- Extracts JSON from markdown code blocks
- Validates against expected schema
- Returns None if parsing fails

---

## Automation Tools Module (`automation_tools/`)

### `automation_tools/config.py`

**Purpose:** Configuration and settings management.

**Class:** `Settings`

**Fields:**
- `llm_provider`: "cortex" or "vertex_ai"
- `llm_model`: Model name (e.g., "vertex_ai/gemini-2.5-flash")
- `embedding_model`: Embedding model name
- `cortex_baseurl`: API base URL
- `cortex_client_id`: Client ID
- `cortex_client_secret`: Client secret
- `cortex_root_ca`: CA certificate path

**Loading:**
```python
settings = Settings()  # Loads from environment variables
```

---

### `automation_tools/llms.py`

**Purpose:** LLM interface implementations.

**Class:** `CortexLLM(LLM)`

**Methods:**
- `generate(prompt, temperature, max_tokens)`: Generate text completion

**Note:** On VM, this wraps Vertex AI models via Cortex proxy.

---

### `automation_tools/handler.py`

**Purpose:** Automation execution handlers.

**Functions:**
- Execute specific automation tasks
- Integrate with LLMs for validation/generation

---

### `automation_tools/common.py`

**Purpose:** Common interfaces and base classes for automations.

---

## Main Entry Points

### `main.py`

**Purpose:** Main CLI entry point for the application.

**Usage:**
```bash
python main.py
```

**Functionality:**
- Interactive command-line interface
- Session management
- Conversation history
- Uses `governance_graph` workflow

---

### `streamlit_cortex.py`

**Purpose:** Streamlit web UI for the application.

**Usage:**
```bash
streamlit run streamlit_cortex.py
```

**Features:**
- Web-based chat interface
- Session state management
- Message history display
- Uses `governance_graph` workflow

---

## Test/Debug Scripts

### `debug_single_query.py`

**Purpose:** Test a single query with debug output.

**Usage:**
```bash
export GOV_DEBUG_MATCHING=1
python debug_single_query.py "Your query here"
```

**Output:**
- Semantic candidates with scores
- LLM validation trace
- Final answer

---

### `test_changes.py`

**Purpose:** Run test queries to validate changes.

**Contains:**
- Test cases for different query types
- Expected vs actual comparison
- Regression testing

---

### `test_intent_classification.py`

**Purpose:** Test intent classification specifically.

**Usage:**
```bash
python test_intent_classification.py
```

---

### `verify_nan_fix.py`

**Purpose:** Verify that NAN filtering is working correctly.

**Checks:**
1. DataFrame has no NAN rows
2. ordered_step_ids doesn't contain NAN
3. step_aliases doesn't have NAN key
4. step_embeddings doesn't have NAN key
5. get_step_record("NAN") returns None
6. deterministic_match never returns NAN
7. Full pipeline test doesn't return NAN

---

### `verify_code_version.py`

**Purpose:** Verify that code changes are actually loaded (not using stale .pyc files).

**Checks:**
1. Intent classification has new logic
2. Multi-field embeddings present
3. Text normalization exists
4. Lexical token limit is 20 (not 6)
5. LLM reranking function exists
6. Embedding structure is correct
7. Workflow graph exists

**Detects:**
- Stale Python cache files
- Multiple Python environments
- Code not pulled on VM

---

## Configuration Files

### `Governance_Process_Updated.xlsx`

**Purpose:** Source data for governance workflow.

**Sheets:**
1. **Nodes:** Governance steps
   - Columns: Step_ID, Step_Name, Purpose, Description, Stage_Name, Automatable, Automation_step
2. **Edges:** Workflow connections
   - Columns: from, to

**Critical:** Rows with Step_ID="NAN" are filtered at load time.

---

### `embeddings_cache.json`

**Purpose:** Cached embeddings to avoid regeneration.

**Structure:**
```json
{
  "_meta": {
    "excel_fingerprint": "12345:1704123456",
    "version": "2.0"
  },
  "steps": {
    "S1": {
      "embedding": [0.1, 0.2, ...],
      "name_embedding": [0.3, 0.4, ...],
      "purpose_embedding": [0.5, 0.6, ...],
      "description_embedding": [0.7, 0.8, ...],
      "text": "...",
      "name_text": "...",
      "purpose_text": "...",
      "description_text": "..."
    }
  }
}
```

**Regeneration:** Delete file or modify Excel to trigger regeneration.

---

## File Dependencies

```
main.py / streamlit_cortex.py
  └─> core/workflow.py
       ├─> core/query_analyzer.py
       │    ├─> core/governance_data.py
       │    │    └─> Governance_Process_Updated.xlsx
       │    │    └─> embeddings_cache.json
       │    └─> core/prompts.py
       ├─> core/automation_handler.py
       │    └─> automation_tools/*
       └─> core/response_generator.py
            └─> core/prompts.py
```

---

## File Modification Checklist

**When modifying matching logic:**
1. Update `core/governance_data.py` if changing scoring
2. Update `core/query_analyzer.py` if changing thresholds
3. Delete `embeddings_cache.json` if changing embedding generation
4. Run `verify_nan_fix.py` to ensure no NAN contamination
5. Run `verify_code_version.py` to ensure changes loaded
6. Run `test_changes.py` for regression testing

**When modifying prompts:**
1. Update `core/prompts.py`
2. Test with `debug_single_query.py` and `GOV_DEBUG_MATCHING=1`
3. Verify LLM responses are well-formed JSON
4. Check token usage doesn't exceed limits

**When modifying response generation:**
1. Update `core/response_generator.py`
2. Check max_tokens limits are appropriate for Gemini-2.5-Flash
3. Test disambiguation flow with multiple candidates
4. Verify fallback messages for all match_method values

**When adding new automations:**
1. Add to `core/automation_registry.py`
2. Implement handler in `automation_tools/handler.py`
3. Update Excel: set Automatable=TRUE, Automation_step=<name>
4. Test via automation_request intent
