# Governance Pipeline Matching System

## Overview

The matching system is responsible for mapping user queries to governance workflow steps. It uses a three-tier approach with progressively sophisticated techniques to balance speed, accuracy, and scope validation.

---

## Three-Tier Matching Architecture

### Tier 1: Deterministic Matching
**Purpose:** Fast exact/substring matching
**Latency:** <50ms
**Location:** `core/governance_data.py` - `deterministic_match()`

#### Priority Order:

1. **ID Regex Match** (Highest Priority)
   - Pattern: `\b(s\d{1,3})\b` (case-insensitive)
   - Examples: "S2", "s15", "S123"
   - Confidence: 1.0
   - Method: `"id_match"`

2. **Alias Match**
   - Predefined aliases from step names
   - Extracts: acronyms, parenthesized text, lowercased name
   - Confidence: 0.95
   - Method: `"alias_match"`

3. **Substring Match**
   - Step name appears in query
   - Minimum length: 3 characters
   - Confidence: 0.90
   - Method: `"substring_match"`

4. **Normalized Match**
   - Verb normalization applied
   - Maps: "raised"→"raise", "created"→"create", etc.
   - Confidence: 0.88
   - Method: `"normalized_match"`

5. **Partial Match**
   - 3+ word trigrams from step name found in query
   - Confidence: 0.85
   - Method: `"partial_match"`

#### Text Normalization

```python
verb_mappings = {
    "raised": "raise", "raising": "raise",
    "created": "create", "creating": "create",
    "completed": "complete", "completing": "complete",
    "delivered": "deliver", "delivering": "deliver",
    "submitted": "submit", "submitting": "submit",
    "obtained": "obtain", "obtaining": "obtain",
    "finished": "finish", "finishing": "finish",
    "got": "get", "getting": "get",
    "presented": "present", "presenting": "present",
    "endorsed": "endorse", "endorsing": "endorse",
}
```

#### Acceptance Criteria

**In `_identify_steps()`:**
```python
if focus_id and method in ["id_match", "alias_match"] and confidence >= 0.90:
    return focus_id  # Accept immediately
else:
    # Fall through to Tier 2
```

Only ID and alias matches bypass semantic search.

---

### Tier 2: Semantic Matching
**Purpose:** Hybrid semantic + lexical search
**Latency:** ~500ms (includes embedding API call)
**Location:** `core/governance_data.py` - `semantic_candidates()`

#### Multi-Field Embedding System

Each governance step has **4 separate embeddings**:

1. **Name Embedding**
   - Text: `"Step {sid}: {step_name}"`
   - Example: `"Step S1: Raise JIRA Ticket"`

2. **Purpose Embedding**
   - Text: Step purpose field (verbatim)
   - Captures the "why" of the step

3. **Description Embedding**
   - Text: Full step description
   - Most detailed context

4. **Combined Embedding**
   - Text: `"{sid} | {name} | Purpose: {purpose} | {description} | Stage: {stage}"`
   - Backward compatibility and general matching

#### Embedding Similarity Calculation

```python
# Calculate cosine similarity for each field
name_sim = cosine_similarity(query_emb, step["name_embedding"])
purpose_sim = cosine_similarity(query_emb, step["purpose_embedding"])
desc_sim = cosine_similarity(query_emb, step["description_embedding"])
combined_sim = cosine_similarity(query_emb, step["embedding"])

# Take max similarity (best field wins)
max_field_sim = max(name_sim, purpose_sim, desc_sim, combined_sim)

# Weighted combination (40% best + 30% combined + 15% each for top 2 others)
sorted_sims = sorted([name_sim, purpose_sim, desc_sim], reverse=True)
multi_field_score = (
    0.40 * max_field_sim +
    0.30 * combined_sim +
    0.15 * sorted_sims[0] +
    0.15 * sorted_sims[1]
)
```

**Rationale:** Allows matching on ANY aspect of the step. Query might match the name, purpose, or description specifically.

#### Lexical Similarity (TF-IDF)

**Tokenization:**
```python
# Extracts alphanumeric tokens, preserving internal hyphens/slashes
tokens = re.findall(r"[a-z0-9]+(?:[/-][a-z0-9]+)*", text.lower())

# Also splits compound tokens
# "jira-ticket" → ["jira-ticket", "jira", "ticket"]
```

**Token Weighting:**
- Top 20 tokens by IDF score (increased from 6 to handle complex governance queries)
- IDF: `log(n_docs / (1 + doc_freq))`

**Score Calculation:**
```python
total = sum(idf[tok] for tok in query_tokens[:20])
hit = sum(idf[tok] for tok in query_tokens[:20] if tok in doc_tokens)
lexical_score = hit / total if total > 0 else 0.0
```

#### Final Score Calculation

```python
final_score = (2.0 * multi_field_score + lexical_score) / 3.0
```

**Weighting:**
- 67% semantic (embedding similarity)
- 33% lexical (keyword overlap)

**Rationale:**
- Semantic captures meaning (e.g., "raised ticket" ≈ "create JIRA")
- Lexical ensures specific artifacts mentioned (e.g., "JIRA" must appear)
- Prevents false positives from coincidental keyword overlap

#### Threshold Filtering

```python
valid_candidates = [
    c for c in candidates
    if c["embedding_score"] >= 0.5 and c["lexical_score"] >= 0.3
]
```

**Why These Thresholds?**
- `embedding_score >= 0.5`: Ensures semantic relevance
- `lexical_score >= 0.3`: Ensures meaningful keyword overlap
- Both must pass to avoid:
  - High embedding, low lexical → vague semantic similarity
  - Low embedding, high lexical → coincidental keyword match

#### Score-Based Routing

After threshold filtering, candidates are routed based on top score:

```python
c1 = valid_candidates[0]  # Highest scoring candidate

# Route 1: High Confidence (bypass LLM)
if len(valid_candidates) >= 2:
    c2 = valid_candidates[1]
    score_gap = c1["score"] - c2["score"]
    if c1["score"] >= 0.60 and score_gap >= 0.15:
        return c1["id"]  # Clear winner
elif c1["score"] >= 0.60:
    return c1["id"]  # Single high-confidence candidate

# Route 2: Medium Confidence (LLM validation)
if 0.35 <= c1["score"] < 0.60:
    # Proceed to Tier 3

# Route 3: Low Confidence (reject)
if c1["score"] < 0.35:
    return None  # No match
```

**Why Score >= 0.60 Bypasses LLM?**
- Empirically determined threshold
- High semantic + lexical confidence
- Gap >= 0.15 ensures clear winner (not close scores)

---

### Tier 3: LLM Validation
**Purpose:** Scope validation and ambiguity resolution
**Latency:** ~2-3 seconds
**Location:** `core/query_analyzer.py` - `_llm_validate_scope()`

#### When LLM Validation Triggers

**Condition:** Top candidate score is 0.35-0.59

**Why?**
- Too low to trust blindly (not >= 0.60)
- High enough to warrant investigation (not < 0.35)
- Could be:
  - Out-of-scope query with coincidental keyword overlap
  - Ambiguous in-scope query (multiple steps apply)
  - Valid match with rough/informal wording

#### Input Preparation

**Candidates passed to LLM:**
```python
candidate_info = [{
    "id": "S1",
    "name": "Raise JIRA Ticket on DP Demand Board",
    "purpose": "To initiate the governance process...",
    "description": "The first step requires creating..."
}]
# Note: Numeric scores are HIDDEN from LLM
```

**Why Hide Scores?**

**Problem Observed:**
When LLM saw `"overall_score": 0.57` and was told "only return VALID if >0.7", it rejected valid queries because 0.57 < 0.7.

**Solution:**
LLM now judges based on content matching, not numeric thresholds. It can't be biased by seeing "low" scores.

#### LLM Prompt Structure

**Model:** Vertex AI Gemini-2.5-Flash
**Max Tokens:** 1500
**Temperature:** 0.0
**Location:** `core/prompts.py` - `validate_scope_prompt()`

**Key Instructions:**

1. **Scope Check:**
   - `IN_SCOPE`: Governance activities, deliverables, approvals, assessments
   - `OUT_OF_SCOPE`: Weather, general tech questions, greetings, chitchat

2. **If IN_SCOPE, Validate Match:**
   - `VALID`: Query clearly relates to one or more steps
   - `AMBIGUOUS`: Multiple steps genuinely apply, user needs to clarify
   - `NO_MATCH`: Governance-related but doesn't fit provided steps

3. **Guidelines:**
   - "These candidates passed strict filters - trust they're potentially relevant"
   - "Focus on matching content/intent, not perfect wording"
   - "Governance queries are often informal or vague - that's OK"
   - "Use OUT_OF_SCOPE only for truly unrelated topics"
   - "When in doubt between steps, return AMBIGUOUS with multiple IDs"

#### LLM Response Format

```json
{
  "scope": "IN_SCOPE" | "OUT_OF_SCOPE",
  "validation": "VALID" | "NO_MATCH" | "AMBIGUOUS",
  "step_ids": ["S1"] | ["S1", "S2", "S3"] | null,
  "confidence": 0.85,
  "reasoning": "Query mentions JIRA ticket which matches step S1's purpose"
}
```

#### Response Handling

```python
if scope == "OUT_OF_SCOPE":
    return None, [], "out_of_scope"

elif validation == "NO_MATCH":
    return None, [], "no_match_validated"

elif validation == "AMBIGUOUS" and step_ids:
    # Returns multiple IDs → triggers disambiguation response
    return step_ids[0], step_ids, "llm_validated_ambiguous"

elif validation == "VALID" and step_ids:
    if len(step_ids) == 1:
        return step_ids[0], step_ids, "llm_validated"
    else:
        # Multiple steps validated → also triggers disambiguation
        return step_ids[0], step_ids, "llm_validated_ambiguous"
```

#### Error Handling

```python
try:
    # Call LLM
    response = cortex_chat_text(...)
    data = parse_json_object(response)
    # Process response
except Exception as e:
    # Fallback: treat as no match
    return [], "no_match", 0.0
```

**Rationale:** Fail gracefully rather than crashing. If LLM fails, assume no match rather than accepting potentially wrong match.

---

## Match Method Values

These appear in `state["match_method"]` and indicate how the match was found:

| Method | Source | Confidence | Description |
|--------|--------|------------|-------------|
| `id_match` | Tier 1 | 1.0 | Explicit step ID (e.g., "S2") |
| `alias_match` | Tier 1 | 0.95 | Predefined alias |
| `substring_match` | Tier 1 | 0.90 | Step name in query |
| `normalized_match` | Tier 1 | 0.88 | Verb normalization |
| `partial_match` | Tier 1 | 0.85 | Trigram match |
| `high_confidence_match` | Tier 2 | 0.60+ | Score >= 0.60, gap >= 0.15 |
| `single_valid_match` | Tier 2 | 0.60+ | Single candidate, score >= 0.60 |
| `llm_validated` | Tier 3 | Varies | LLM confirmed single match |
| `llm_validated_ambiguous` | Tier 3 | Varies | LLM returned multiple matches |
| `threshold_match` | Tier 2 | Varies | Fallback for edge cases |
| `below_threshold` | Tier 2 | 0.0 | No candidates passed thresholds |
| `low_confidence` | Tier 2 | 0.0 | Score < 0.35 |
| `out_of_scope` | Tier 3 | 0.0 | LLM rejected as out-of-scope |
| `no_match_validated` | Tier 3 | 0.0 | LLM couldn't find match |
| `no_match` | N/A | 0.0 | No semantic candidates |

---

## Disambiguation Flow

**Trigger Condition:**
```python
if len(candidate_step_ids) > 1 and intent not in {"automation_request", "ask_next_step"}:
    state["needs_disambiguation"] = True
```

**Response:**
- LLM generates disambiguation response (2500 tokens max)
- Explains all candidate steps
- Asks user which one they meant

**Example:**
```
Based on your query, this could relate to a couple of different governance steps:

**S2: Complete DOI Form**
Purpose: To document...
Description: This step involves...

**S5: Security Assessment**
Purpose: To evaluate...
Description: Security team will...

Which of these steps are you asking about, or would you like more details on both?
```

---

## Performance Tuning

### Embedding Cache

**File:** `embeddings_cache.json`
**Version:** 2.0
**Invalidation:** Excel file modification (fingerprint: `{file_size}:{mtime}`)

**Cache Structure:**
```json
{
  "_meta": {
    "excel_fingerprint": "12345:1704123456",
    "version": "2.0"
  },
  "steps": {
    "S1": {
      "embedding": [...],
      "name_embedding": [...],
      "purpose_embedding": [...],
      "description_embedding": [...],
      "text": "...",
      "name_text": "...",
      "purpose_text": "...",
      "description_text": "..."
    }
  }
}
```

### Embedding Index

**Pre-computed matrices for fast similarity:**
```python
self._embedding_matrix = np.array([step["embedding"] for step in steps])
self._embedding_norms = np.linalg.norm(self._embedding_matrix, axis=1)
self._embedding_ids = [step_id for step_id in steps]
```

**Usage:**
```python
dots = self._embedding_matrix @ query_embedding
similarities = dots / (self._embedding_norms * query_norm)
top_k_indices = np.argsort(-similarities)[:top_k]
```

**Performance:** O(n) dot products vs O(n) individual cosine calculations, but vectorized.

---

## Debugging

### Debug Mode

```bash
export GOV_DEBUG_MATCHING=1
python debug_single_query.py "Your query here"
```

**Output Shows:**
1. Semantic candidates with scores
2. LLM validation candidates and prompt length
3. LLM raw response
4. LLM decision result
5. Warnings and errors

**Example Output:**
```
DEBUG: semantic_candidates (top 5)
  S1: score=0.574 (emb=0.612, lex=0.498, name=0.650, desc=0.580)
  S5: score=0.521 (emb=0.559, lex=0.445, name=0.520, desc=0.590)
  S9: score=0.445 (emb=0.480, lex=0.375, name=0.460, desc=0.500)

================================================================================
DEBUG: LLM SCOPE VALIDATION
================================================================================
Query: I raised a JIRA ticket
Candidates passed to LLM: 3
  - S1: Raise JIRA Ticket on DP Demand Board (score=0.574, emb=0.612, lex=0.498)
  - S5: Security Assessment (score=0.521, emb=0.559, lex=0.445)
  - S9: Architecture Review (score=0.445, emb=0.480, lex=0.375)
Prompt length: 2847 chars

LLM Raw Response:
{
  "scope": "IN_SCOPE",
  "validation": "VALID",
  "step_ids": ["S1"],
  "confidence": 0.9,
  "reasoning": "Query explicitly mentions 'raised a JIRA ticket' which directly matches S1's purpose and description"
}
Response length: 156 chars

→ RESULT: VALID (confidence=0.90, step_ids=['S1'])
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Score Inflation

**Problem:** Scores seemed too high (e.g., 0.8 when expected 0.5)
**Cause:** Dividing by 2 instead of 3 in final score formula
**Fix:** `(2*emb + lex) / 3.0` (not `/2.0`)

### Pitfall 2: Out-of-Scope Queries Matching Steps

**Problem:** "What is Python?" matching governance steps
**Cause 1:** Threshold-based ambiguity check bypassing LLM validation
**Cause 2:** LLM seeing low scores and being told to reject if <0.7
**Fix 1:** Remove ambiguity check before LLM validation
**Fix 2:** Hide scores from LLM prompt

### Pitfall 3: Valid Queries Being Rejected

**Problem:** "I raised a JIRA ticket" rejected as out-of-scope
**Cause:** LLM prompt too conservative ("Be CONSERVATIVE, only return if >0.7")
**Fix:** Rebalanced prompt to trust pre-filtered candidates

### Pitfall 4: NAN Step Contamination

**Problem:** Queries matching "NAN" step ID
**Cause:** Excel has rows with Step_ID="NAN" (pandas reads empty as "nan" string)
**Fix:** Filter at load time: `nodes_df = nodes_df[nodes_df["Step_ID"] != "NAN"]`

### Pitfall 5: Long Descriptions Causing Token Limit Exceeded

**Problem:** LLM response cut off mid-JSON
**Cause:** max_tokens too low (400) for long candidate descriptions
**Fix:** Increased to 1500 tokens for Gemini-2.5-Flash

---

## Future Improvements

1. **Adaptive Thresholds:** Learn optimal thresholds from user feedback
2. **Contextual Embeddings:** Include previous conversation turns in embedding
3. **Hybrid Reranking:** Use both LLM and score-based reranking
4. **Caching LLM Validations:** Cache scope decisions for common queries
5. **Fine-tuned Embeddings:** Train custom embedding model on governance domain
