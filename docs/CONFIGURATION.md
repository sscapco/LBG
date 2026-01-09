# Configuration Guide

All tunable parameters, thresholds, and model settings for the Governance Pipeline.

---

## Model Configuration

### LLM Model (Chat)

**Current:** Vertex AI Gemini-2.5-Flash
**Environment Variable:** `LLM_MODEL`
**Value:** `vertex_ai/gemini-2.5-flash`

**Specifications:**
- Context window: 1,048,576 tokens (1M)
- Output tokens: 8,192 max
- Speed: Fast (2-3 seconds per call)
- Cost: Low

**Alternative Models:**
- `vertex_ai/gemini-2.0-pro`: More capable but slower
- `claude-opus-4`: Different provider

**Configuration Location:** `automation_tools/config.py`

---

### Embedding Model

**Current:** Vertex AI text-embedding-004
**Environment Variable:** `EMBEDDING_MODEL`
**Value:** `vertex_ai/text-embedding-004`

**Specifications:**
- Dimensions: 768
- Max input: 2048 tokens per text
- Speed: ~100ms per embedding
- Cost: Low

**Alternative Models:**
- `text-embedding-3-small`: OpenAI alternative
- Custom fine-tuned model

**Configuration Location:** `automation_tools/config.py`

---

## Matching Thresholds

### Tier 2: Semantic Matching

**Location:** `core/query_analyzer.py` lines 223-226

#### Embedding Score Threshold
```python
embedding_score >= 0.5
```

**Purpose:** Ensures semantic relevance
**Range:** 0.0-1.0 (cosine similarity)
**Impact:**
- Lower (0.3-0.4): More recall, more false positives
- Higher (0.6-0.7): More precision, might miss valid queries

**Recommendation:** Keep at 0.5 for balanced performance

#### Lexical Score Threshold
```python
lexical_score >= 0.3
```

**Purpose:** Ensures meaningful keyword overlap
**Range:** 0.0-1.0 (TF-IDF weighted overlap)
**Impact:**
- Lower (0.1-0.2): Accepts queries with minimal keyword match
- Higher (0.4-0.5): Requires strong keyword presence

**Recommendation:** Keep at 0.3 for balanced performance

#### Why BOTH Thresholds?

Prevents:
- High embedding, low lexical → Vague semantic similarity (false positive)
- Low embedding, high lexical → Coincidental keyword match (false positive)

### Tier 2: Score-Based Bypass

**Location:** `core/query_analyzer.py` lines 240-246

#### High Confidence Threshold
```python
score >= 0.60 and score_gap >= 0.15
```

**Purpose:** Bypass LLM for clear winners
**Impact:**
- Lower (0.50): More queries bypass LLM (faster, but might miss out-of-scope)
- Higher (0.70): Fewer queries bypass LLM (slower, but safer)

**Recommendation:** Keep at 0.60 with gap >= 0.15

#### Single Candidate Threshold
```python
score >= 0.60 (only 1 valid candidate)
```

**Purpose:** Accept single candidate if high confidence
**Impact:** Same as above

**Recommendation:** Keep at 0.60

### Tier 3: LLM Validation Range

**Location:** `core/query_analyzer.py` line 250

```python
if 0.35 <= c1["score"] < 0.60:
    # Call LLM validation
```

**Lower Bound (0.35):**
- Purpose: Don't waste LLM calls on very low scores
- Impact: Lower → more LLM calls (slower, more cost)

**Upper Bound (0.60):**
- Purpose: High scores bypass LLM
- Impact: Higher → more LLM calls (safer, but slower)

**Recommendation:** Keep at 0.35-0.60 range

### Low Confidence Threshold

**Location:** `core/query_analyzer.py` line 269

```python
if c1["score"] < 0.35:
    return None, [], "low_confidence", 0.0
```

**Purpose:** Reject very low scores immediately
**Impact:**
- Lower (0.25): More queries go to LLM
- Higher (0.40): More queries rejected immediately

**Recommendation:** Keep at 0.35

---

## Scoring Formula

### Final Score Calculation

**Location:** `core/governance_data.py` line 413

```python
final_score = (2.0 * multi_field_score + lexical_score) / 3.0
```

**Weighting:**
- 67% semantic (embedding similarity)
- 33% lexical (keyword overlap)

**Rationale:**
- Semantic captures meaning ("raised ticket" ≈ "create JIRA")
- Lexical ensures specific artifacts mentioned
- 2:1 ratio prevents keyword-only false positives

**Alternative Formulas:**
```python
# Equal weighting (50/50)
final_score = (multi_field_score + lexical_score) / 2.0

# More lexical (50% semantic, 50% lexical)
final_score = (multi_field_score + lexical_score) / 2.0

# More semantic (80% semantic, 20% lexical)
final_score = (4.0 * multi_field_score + lexical_score) / 5.0
```

**Impact of Changes:**
- More semantic weight → Better for varied wording, worse for specific artifacts
- More lexical weight → Better for exact keywords, worse for paraphrasing

---

## LLM Call Parameters

### Intent Classification

**Location:** `core/query_analyzer.py` lines 132-139

```python
max_tokens=200
temperature=0.0
thinking_enabled=False
```

**Purpose:** Classify user intent (greeting, ask_about_step, etc.)

**Tuning:**
- `max_tokens`: 200 is enough for JSON response
- `temperature`: 0.0 for deterministic classification
- `thinking_enabled`: False to save tokens/latency

### LLM Scope Validation

**Location:** `core/query_analyzer.py` lines 372-379

```python
max_tokens=1500
temperature=0.0
thinking_enabled=False
```

**Purpose:** Validate scope and match steps

**Tuning:**
- `max_tokens`: 1500 handles full JSON with reasoning
  - Lower (800): Risk of truncation with long descriptions
  - Higher (2000): Unnecessary cost
- `temperature`: 0.0 for consistent decisions
- `thinking_enabled`: False to save tokens

### Step Response

**Location:** `core/response_generator.py` lines 129-134

```python
max_tokens=2500
temperature=0.0
thinking_enabled=False
```

**Purpose:** Generate step explanation + next steps

**Tuning:**
- `max_tokens`: 2500 for detailed explanations
- `temperature`: 0.0 for consistency
- `thinking_enabled`: False

### Disambiguation Response

**Location:** `core/response_generator.py` lines 78-85

```python
max_tokens=2500
temperature=0.0
thinking_enabled=False
```

**Purpose:** Explain multiple candidate steps

**Tuning:**
- `max_tokens`: 2500 for multiple step descriptions

### Fallback Response

**Location:** `core/response_generator.py` lines 202-209

```python
max_tokens=800
temperature=0.3
thinking_enabled=False
```

**Purpose:** Generic fallback message

**Tuning:**
- `max_tokens`: 800 for helpful fallback
- `temperature`: 0.3 for slight variation

---

## Lexical Search Parameters

### Token Limit

**Location:** `core/governance_data.py` line 233

```python
top = ranked[: min(20, len(ranked))]
```

**Purpose:** Limit to top 20 tokens by IDF score

**Impact:**
- Lower (6-10): Faster, but might miss important keywords
- Higher (30-50): Slower, might dilute important signals

**Recommendation:** Keep at 20 for complex governance queries

### IDF Calculation

**Location:** `core/governance_data.py` line 258

```python
idf[tok] = math.log(n_docs / (1.0 + df[tok]))
```

**Purpose:** Weight rare tokens higher

**No tuning needed** - standard TF-IDF formula

---

## Text Normalization

### Verb Mappings

**Location:** `core/governance_data.py` lines 191-202

```python
verb_mappings = {
    "raised": "raise",
    "created": "create",
    "completed": "complete",
    "delivered": "deliver",
    "submitted": "submit",
    "obtained": "obtain",
    "finished": "finish",
    "got": "get",
    "presented": "present",
    "endorsed": "endorse",
}
```

**Purpose:** Normalize verb tenses for matching

**To Add New Verbs:**
```python
"approved": "approve",
"validated": "validate",
```

**Impact:** Helps match "I approved the DOI" to "Approve DOI Form" step

---

## Embedding Cache

### Cache File

**Location:** `embeddings_cache.json`

**Version:** 2.0

**Fingerprint:** `{file_size}:{mtime}` of Excel file

**Invalidation:**
```python
if meta.get("excel_fingerprint") == self._excel_fingerprint() and meta.get("version") == "2.0":
    # Use cache
else:
    # Regenerate
```

**Manual Invalidation:**
```bash
rm embeddings_cache.json
python main.py  # Will regenerate on next run
```

---

## Debug Settings

### Debug Matching

**Environment Variable:** `GOV_DEBUG_MATCHING`

```bash
export GOV_DEBUG_MATCHING=1
python debug_single_query.py "Your query"
```

**Output:**
- Semantic candidates with scores
- LLM validation inputs/outputs
- Decision traces
- Warnings and errors

**Location of Debug Code:**
- `core/query_analyzer.py` lines 211-218 (semantic candidates)
- `core/query_analyzer.py` lines 361-432 (LLM validation)

---

## Environment Variables

### Required

```bash
# LLM Configuration
export LLM_PROVIDER=cortex
export LLM_MODEL=vertex_ai/gemini-2.5-flash
export EMBEDDING_MODEL=vertex_ai/text-embedding-004

# Cortex API (VM only)
export CORTEX_BASEURL=https://...
export CORTEX_CLIENT_ID=...
export CORTEX_CLIENT_SECRET=...
export CORTEX_ROOT_CA=/path/to/ca.crt
```

### Optional

```bash
# Debug
export GOV_DEBUG_MATCHING=1
```

**Configuration Loading:** `automation_tools/config.py`

---

## Performance Tuning

### Latency Optimization

**Current Bottlenecks:**
1. Embedding API calls (~500ms)
2. LLM validation calls (~2-3 seconds)

**Optimization Strategies:**

1. **Cache Embeddings** (Already implemented)
   - Embeddings cached to JSON
   - Regenerated only on Excel modification

2. **Reduce LLM Calls**
   - Increase high-confidence threshold (0.60 → 0.65)
   - Risk: More false positives

3. **Parallel LLM Calls**
   - Not applicable (sequential workflow)

4. **Batch Queries**
   - Not applicable (interactive CLI)

### Cost Optimization

**Current Costs (per query):**
- Intent classification: ~300 input, 50 output tokens
- LLM validation: ~1500-3000 input, 300 output tokens
- Response generation: ~500-2000 input, 500-2500 output tokens

**Total:** ~2300-5350 input, 850-2850 output per query

**Optimization Strategies:**

1. **Reduce Token Usage**
   - Truncate descriptions (not recommended - causes issues)
   - Shorter prompts (risk: less context)

2. **Skip LLM Validation**
   - Increase bypass threshold (0.60 → 0.70)
   - Risk: More out-of-scope queries matching

3. **Use Cheaper Model**
   - Switch to smaller model (risk: lower quality)

---

## Threshold Tuning Guide

### Goal: Increase Precision (Reduce False Positives)

**Changes:**
```python
# Stricter thresholds
embedding_score >= 0.6  # Was 0.5
lexical_score >= 0.4    # Was 0.3

# Higher bypass threshold
if c1["score"] >= 0.70 and score_gap >= 0.20:  # Was 0.60, 0.15
```

**Trade-off:** Might reject more valid queries

### Goal: Increase Recall (Catch More Queries)

**Changes:**
```python
# Looser thresholds
embedding_score >= 0.4  # Was 0.5
lexical_score >= 0.2    # Was 0.3

# Lower bypass threshold
if c1["score"] >= 0.50 and score_gap >= 0.10:  # Was 0.60, 0.15
```

**Trade-off:** More false positives

### Goal: Reduce Latency

**Changes:**
```python
# Increase bypass threshold (fewer LLM calls)
if c1["score"] >= 0.50:  # Was 0.60
    return immediately

# Reduce LLM validation range
if 0.40 <= c1["score"] < 0.50:  # Was 0.35-0.60
    call LLM
```

**Trade-off:** More risk of false positives

### Goal: Improve Out-of-Scope Detection

**Changes:**
```python
# Force more queries through LLM
if 0.30 <= c1["score"] < 0.70:  # Was 0.35-0.60
    call LLM

# Or stricter bypass
if c1["score"] >= 0.70:  # Was 0.60
    bypass LLM
```

**Trade-off:** Higher latency, more cost

---

## Recommended Starting Points

**For Production (Balanced):**
```python
embedding_threshold = 0.5
lexical_threshold = 0.3
bypass_score = 0.60
bypass_gap = 0.15
llm_range = (0.35, 0.60)
```

**For High Precision (Finance, Legal):**
```python
embedding_threshold = 0.6
lexical_threshold = 0.4
bypass_score = 0.70
bypass_gap = 0.20
llm_range = (0.35, 0.70)
```

**For High Recall (Customer Support):**
```python
embedding_threshold = 0.4
lexical_threshold = 0.2
bypass_score = 0.50
bypass_gap = 0.10
llm_range = (0.30, 0.50)
```

**For Speed (Demo, Testing):**
```python
embedding_threshold = 0.5
lexical_threshold = 0.3
bypass_score = 0.50  # More bypass
bypass_gap = 0.10
llm_range = (0.40, 0.50)  # Narrow range
```

---

## Monitoring Metrics

### Key Metrics to Track

1. **Match Rate:** % of queries that find a match
2. **Disambiguation Rate:** % triggering disambiguation
3. **Out-of-Scope Rate:** % rejected as out-of-scope
4. **Average Latency:** Time per query
5. **LLM Call Rate:** % of queries calling LLM
6. **Token Usage:** Average tokens per query

### Ideal Ranges

- Match Rate: 70-85%
- Disambiguation Rate: 10-15%
- Out-of-Scope Rate: 5-10%
- Average Latency: <3 seconds
- LLM Call Rate: 30-50%
- Token Usage: <5000 total per query

### Red Flags

- Match Rate <50%: Thresholds too strict
- Out-of-Scope Rate >20%: Thresholds too strict or LLM too conservative
- Average Latency >5s: Too many LLM calls
- LLM Call Rate >70%: Bypass threshold too high
