# Governance Pipeline Architecture

## System Overview

The Governance Pipeline is a conversational AI system that helps users navigate a complex governance workflow process. It uses hybrid semantic search, LLM validation, and natural language generation to match user queries to specific governance steps and provide contextual guidance.

**Primary Function:** Map informal user queries (e.g., "I raised a JIRA ticket") to specific governance workflow steps and provide detailed explanations.

**Technology Stack:**
- **LLM:** Vertex AI Gemini-2.5-Flash (chat model)
- **Embeddings:** Vertex AI text-embedding-004 (768 dimensions)
- **Framework:** LangGraph (state machine workflow)
- **Data Source:** Excel file (`Governance_Process_Updated.xlsx`)
- **Language:** Python 3.x

---

## High-Level Architecture

```
┌─────────────────┐
│  User Query     │
│  "I raised a    │
│   JIRA ticket"  │
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│  1. QUERY ANALYSIS NODE                        │
│  - Intent classification (greeting, ask_about_ │
│    step, automation_request, etc.)             │
│  - Step matching (deterministic → semantic →   │
│    LLM validation)                              │
│  - Extract referenced step IDs (e.g., "S2")    │
└────────┬───────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│  2. AUTOMATION HANDLER NODE (if applicable)    │
│  - Detects if matched step has automation      │
│  - Runs automation tools (e.g., name validation)│
└────────┬───────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│  3. RESPONSE GENERATOR NODE                    │
│  - Generates natural language response based   │
│    on intent and matched steps                 │
│  - Handles: greetings, step explanations,      │
│    disambiguation, comparisons, fallbacks      │
└────────┬───────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Answer   │
│  to User        │
└─────────────────┘
```

---

## Data Flow

### 1. **Initialization (Load Governance Data)**

```
Excel File (Nodes + Edges sheets)
         │
         ▼
┌────────────────────────────┐
│  GovernanceDataLoader      │
│  - Loads nodes & edges     │
│  - Filters NAN rows        │
│  - Generates embeddings    │
│  - Builds search indices   │
└────────┬───────────────────┘
         │
         ▼
Singleton: get_governance_data()
```

**Key Filtering (CRITICAL):**
- Rows with `Step_ID == "NAN"` are filtered out at load time
- Only valid step IDs are indexed for search

### 2. **Query Processing Pipeline**

#### Step 1: Intent Classification

```python
User Query → _classify_intent() → Intent + Confidence
```

**Intents:**
- `greeting`: "Hello", "Hi"
- `ask_about_step`: "I raised a JIRA ticket" (MOST COMMON)
- `compare_steps`: "What's the difference between S2 and S3?"
- `ask_next_step`: "What comes next?"
- `automation_request`: "Run naming validation"
- `mark_complete`, `mark_in_progress`: Progress tracking
- `clarification_needed`, `unknown`: Fallbacks

**Process:**
1. Call LLM with `classify_intent_prompt` (200 tokens, temp=0.0)
2. Parse JSON response
3. Fallback to heuristics if LLM fails

#### Step 2: Step Matching (Core Logic)

```
Query → _identify_steps() → (focus_step_id, candidates, method, confidence)
```

**Three-Tier Matching System:**

##### **Tier 1: Deterministic Match** (Fast Path)
```python
governance_data.deterministic_match(query)
```

Checks in priority order:
1. **ID Regex Match:** `\b(s\d{1,3})\b` → confidence 1.0
2. **Alias Match:** Predefined aliases → confidence 0.95
3. **Substring Match:** Step name appears in query → confidence 0.85-0.90
4. **Normalized Match:** Verb normalization (e.g., "raised"→"raise") → confidence 0.88

**Only accepts:** ID match or alias match with confidence >= 0.90

##### **Tier 2: Semantic Matching** (Hybrid Search)
```python
governance_data.semantic_candidates(query, top_k=5)
```

**Multi-Field Embedding Scoring:**
- Each step has 4 embeddings: name, purpose, description, combined
- Calculate cosine similarity against all 4
- Weight: 40% best field + 30% combined + 15% each for next 2 fields

**Lexical Scoring (TF-IDF):**
- Token limit: top 20 tokens by IDF
- Weighted token overlap

**Final Score Formula:**
```
final_score = (2.0 * multi_field_embedding_score + lexical_score) / 3.0
```
- 67% semantic, 33% lexical weighting

**Threshold Filtering:**
```python
valid_candidates = [c for c in candidates
                    if c["embedding_score"] >= 0.5
                    and c["lexical_score"] >= 0.3]
```

**Decision Tree:**

```
┌─ valid_candidates empty?
│  └─ YES → return "below_threshold"
│
├─ Top score >= 0.60 AND gap >= 0.15?
│  └─ YES → return immediately (high_confidence_match)
│
├─ Only 1 candidate AND score >= 0.60?
│  └─ YES → return immediately (single_valid_match)
│
├─ Score 0.35-0.59?
│  └─ YES → LLM VALIDATION (see Tier 3)
│
└─ Score < 0.35?
   └─ YES → return "low_confidence"
```

##### **Tier 3: LLM Validation** (Scores 0.35-0.59)
```python
_llm_validate_scope(query, candidates[:3])
```

**Purpose:** Distinguish between:
- Out-of-scope queries (e.g., "What is Python?")
- Ambiguous in-scope queries (e.g., "Tell me about governance")
- Valid matches with rough wording

**Process:**
1. Prepare candidate details (id, name, purpose, description) **WITHOUT scores**
2. Call LLM with `validate_scope_prompt` (1500 tokens, temp=0.0)
3. LLM returns:
   - `scope`: IN_SCOPE | OUT_OF_SCOPE
   - `validation`: VALID | NO_MATCH | AMBIGUOUS
   - `step_ids`: List of matching step IDs
   - `confidence`: 0.0-1.0

**LLM Decision Flow:**
```
OUT_OF_SCOPE → reject (weather, tech questions, etc.)
IN_SCOPE + NO_MATCH → reject with helpful message
IN_SCOPE + VALID + [S1] → return single step
IN_SCOPE + VALID + [S1,S2,S3] → disambiguation
IN_SCOPE + AMBIGUOUS + [S1,S2,S3] → disambiguation
```

**Key Design Choice:**
- Scores are hidden from LLM to prevent threshold bias
- LLM judges based on content matching, not numeric scores
- Promotes AMBIGUOUS when uncertain between steps

### 3. **Response Generation**

```
State → response_generator_node() → Final Answer
```

**Response Types:**

| Condition | Response Type | Handler |
|-----------|---------------|---------|
| `len(referenced_ids) >= 2` | Comparison | `_generate_comparison_response()` |
| `intent == "greeting"` | Greeting | `_generate_greeting_response()` |
| `needs_disambiguation == True` | Disambiguation | `_generate_disambiguation_response()` |
| `focus_step_id` exists | Step Explanation | `_generate_step_response()` |
| `intent == "ask_next_step"` | Next Step Guidance | `_generate_next_step_response()` |
| No match | Fallback | `_generate_fallback_response()` |

**Disambiguation Response:**
- Triggered when `len(candidate_step_ids) > 1`
- LLM explains all candidate steps
- Asks user to clarify which one they mean
- Max tokens: 2500

**Fallback Messages (Context-Aware):**

| match_method | Message Type |
|--------------|--------------|
| `out_of_scope` | "I'm a governance workflow assistant..." |
| `below_threshold` | "I couldn't find a step that matches..." |
| `no_match_validated` | "I understand you're asking about governance..." |
| Other | LLM-generated generic fallback (800 tokens) |

---

## State Management

**GovernanceState TypedDict:**

```python
{
    "user_message": str,           # Original query
    "session_id": str,             # Session identifier
    "previous_state": dict,        # Context from previous turn

    # Query Analysis Results
    "intent": str,                 # Classified intent
    "query_analysis": dict,        # Extra metadata from classifier
    "focus_step_id": str,          # Primary matched step
    "candidate_step_ids": List[str], # All matching candidates
    "referenced_ids": List[str],   # Explicitly mentioned IDs (e.g., S2)
    "match_method": str,           # How match was found
    "match_confidence": float,     # Match confidence score
    "needs_disambiguation": bool,  # Multiple candidates?

    # Step Details
    "step_details": dict,          # Full record of focus step
    "next_step_ids": List[str],    # Downstream steps
    "anchor_step_id": str,         # Reference for navigation

    # Automation
    "automatable_step_id": str,    # If step has automation
    "automation_step": str,        # Automation name
    "automation_result": dict,     # Automation output

    # Response
    "answer": str,                 # Final response text
    "explained_step_ids": List[str] # Steps included in response
}
```

---

## Critical Design Decisions

### 1. **Why Three-Tier Matching?**

- **Tier 1 (Deterministic):** Handles explicit references (e.g., "S2") instantly
- **Tier 2 (Semantic):** Handles variations in wording using embeddings
- **Tier 3 (LLM):** Handles ambiguity and scope validation

### 2. **Why Hide Scores from LLM?**

**Problem:** When LLM saw `"overall_score": 0.57` and was told "only return VALID if >0.7", it rejected valid queries.

**Solution:** Pass only content (id, name, purpose, description) to LLM. Let it judge based on semantic meaning, not numeric thresholds.

### 3. **Why 67/33 Semantic/Lexical Weighting?**

- Semantic embeddings capture meaning (e.g., "raised ticket" ≈ "create JIRA")
- Lexical ensures specific artifacts are matched (e.g., "JIRA" must appear)
- 67/33 ratio prevents false positives from keyword-only overlap

### 4. **Why Force LLM Validation for Scores 0.35-0.59?**

**Problem:** Early version had threshold-based ambiguity detection that bypassed LLM.

**Root Cause:** Out-of-scope queries with scores ~0.4 were marked "ambiguous" without scope checking.

**Solution:** ALL borderline scores go through LLM to validate scope first.

---

## Performance Characteristics

**Latency:**
- Deterministic match: <50ms
- Semantic match: ~500ms (embedding API call)
- LLM validation: ~2-3 seconds (Gemini-2.5-Flash)

**Token Usage (per query):**
- Intent classification: ~300 input, 50 output
- LLM validation: ~1500-3000 input, ~300 output
- Response generation: ~500-2000 input, 500-2500 output

**Cache:**
- Embeddings cached to `embeddings_cache.json` (version 2.0)
- Invalidates on Excel file modification (fingerprint check)

---

## Error Handling

**Graceful Degradation:**
1. LLM intent classification fails → Fallback to heuristics
2. LLM validation fails → Return as "no_match"
3. Embedding API fails → Skip semantic search
4. Response generation fails → Generic fallback message

**Debug Mode:**
```bash
export GOV_DEBUG_MATCHING=1
```

Shows:
- Semantic candidate scores
- LLM validation inputs/outputs
- Decision flow traces

---

## Data Dependencies

**Excel File:** `Governance_Process_Updated.xlsx`

### Nodes Sheet
- `Step_ID`: Unique identifier (e.g., S1, S2)
- `Step_Name`: Human-readable name
- `Purpose`: Why this step exists
- `Description`: Detailed explanation
- `Stage_Name`: Workflow stage
- `Automatable`: TRUE/FALSE
- `Automation_step`: Name of automation function

### Edges Sheet
- `from`: Source step ID
- `to`: Target step ID

**NAN Filtering:** Rows with `Step_ID == "NAN"` are filtered at load time to prevent contamination.

---

## Next Steps for Handover

1. Review [MATCHING_SYSTEM.md](MATCHING_SYSTEM.md) for detailed matching logic
2. Review [FILE_REFERENCE.md](FILE_REFERENCE.md) for file-by-file documentation
3. Review [CONFIGURATION.md](CONFIGURATION.md) for tunable parameters
4. Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
