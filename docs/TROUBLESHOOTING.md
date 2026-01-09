# Troubleshooting Guide

Common issues, debugging techniques, and solutions for the Governance Pipeline.

---

## Common Issues and Solutions

### Issue 1: Out-of-Scope Queries Matching Governance Steps

**Symptoms:**
- Query like "What is Python?" returns a governance step
- Query like "How's the weather?" triggers step explanation

**Root Causes:**

1. **Threshold-based ambiguity bypass (FIXED)**
   - Old code had logic that marked close scores as "ambiguous" without checking scope
   - Bypassed LLM validation entirely

2. **LLM seeing numeric scores**
   - LLM was told "only return VALID if >0.7"
   - Saw scores like 0.57 and rejected valid queries
   - Or saw scores and accepted low-confidence matches

**Solutions:**

✅ **Implemented:**
- Removed threshold-based ambiguity check (commit: a2a1ae8)
- Force ALL scores 0.35-0.59 through LLM validation
- Hide numeric scores from LLM prompt (commit: c90b7c1)

**Verification:**
```bash
export GOV_DEBUG_MATCHING=1
python debug_single_query.py "What is Python?"
```

Expected output:
```
→ RESULT: OUT_OF_SCOPE (confidence=0.95)
```

---

### Issue 2: Valid Queries Being Rejected

**Symptoms:**
- Query like "I raised a JIRA ticket" returns "no match"
- Rough governance queries rejected as out-of-scope

**Root Causes:**

1. **LLM prompt too conservative**
   - Instructions said "Be CONSERVATIVE, only return if >0.7"
   - LLM rejected borderline but valid queries

2. **LLM seeing low scores**
   - Prompt showed `"overall_score": 0.57`
   - LLM thought "0.57 < 0.7 = reject"

**Solutions:**

✅ **Implemented:**
- Rebalanced LLM prompt (commit: c90b7c1)
- Removed "only return if >0.7" instruction
- Added "Trust that candidates were pre-filtered"
- Added "Governance queries are often informal - that's OK"

**Verification:**
```bash
python debug_single_query.py "I raised a JIRA ticket"
```

Expected output:
```
→ RESULT: VALID (confidence=0.90, step_ids=['S1'])
```

---

### Issue 3: NAN Step ID Returned

**Symptoms:**
- Match returns step_id="NAN"
- Errors like `KeyError: 'NAN'` or `TypeError: 'NoneType' object is not subscriptable`

**Root Cause:**
- Excel file has rows with empty Step_ID
- Pandas reads empty cells as "nan" string
- These rows weren't filtered at load time

**Solution:**

✅ **Implemented:**
```python
# Line 75-76 in governance_data.py
valid_mask = (self.nodes_df["Step_ID"] != "NAN") & (self.nodes_df["Step_ID"].notna())
self.nodes_df = self.nodes_df[valid_mask].reset_index(drop=True)
```

**Verification:**
```bash
python verify_nan_fix.py
```

All checks should PASS.

---

### Issue 4: Score Inflation

**Symptoms:**
- Scores seem too high (e.g., 0.8 when expecting 0.5)
- Many queries bypass LLM with high confidence

**Root Cause:**
- Formula was `(2*emb + lex) / 2` instead of `/3`
- Divided by 2 instead of 3, inflating all scores by ~50%

**Solution:**

✅ **Implemented:**
```python
# Line 413 in governance_data.py
final_score = (2.0 * multi_field_score + lex_score) / 3.0  # Not /2.0!
```

**Verification:**
```python
# Example calculation
embedding_score = 0.6
lexical_score = 0.3

# Correct formula
score = (2*0.6 + 0.3) / 3 = 1.5 / 3 = 0.50  ✓

# Wrong formula (old)
score = (2*0.6 + 0.3) / 2 = 1.5 / 2 = 0.75  ✗ (inflated)
```

---

### Issue 5: Response Cut Off Mid-Sentence

**Symptoms:**
- Response like "Thanks for your question, this could relate to..."  then cuts off
- JSON responses incomplete

**Root Cause:**
- `max_tokens` too low (300-400)
- Gemini-2.5-Flash can handle 8192 output tokens
- Long candidate descriptions + prompt exceeded token limit

**Solution:**

✅ **Implemented:**
- Increased max_tokens across all LLM calls:
  - Validation: 400 → 1500
  - Step response: 1500 → 2500
  - Disambiguation: 2000 → 2500
  - Fallback: 300 → 800

**Verification:**
Check that responses are complete and well-formed.

---

### Issue 6: Embeddings Not Regenerating

**Symptoms:**
- Changes to Excel file not reflected in matches
- Old step names still being matched

**Root Cause:**
- Embedding cache (`embeddings_cache.json`) not invalidated
- Fingerprint check didn't detect modification

**Solutions:**

**Option 1: Delete Cache (Recommended)**
```bash
rm embeddings_cache.json
python main.py  # Will regenerate on startup
```

**Option 2: Touch Excel File**
```bash
touch Governance_Process_Updated.xlsx
python main.py  # Will detect modification
```

**Option 3: Check Cache Version**
```bash
grep version embeddings_cache.json
```

Should show `"version": "2.0"`. If "1.0", delete cache.

---

### Issue 7: Code Changes Not Taking Effect

**Symptoms:**
- Made changes to .py files
- Changes don't appear when running
- Old behavior persists

**Root Causes:**

1. **Stale .pyc files**
   - Python bytecode cache is outdated
   - Interpreter loads old code

2. **Multiple Python environments**
   - Editing in one virtualenv, running in another
   - Wrong Python interpreter

3. **Code not pulled on VM**
   - Changes pushed to git but not pulled
   - Working with old code

**Solutions:**

**Step 1: Delete .pyc cache**
```bash
rm -rf core/__pycache__
rm -rf automation_tools/__pycache__
rm -rf __pycache__
```

**Step 2: Verify Python environment**
```bash
which python
python --version
```

**Step 3: Pull latest code (if on VM)**
```bash
git pull origin maggie_dev
```

**Step 4: Verify code loaded**
```bash
python verify_code_version.py
```

All checks should PASS.

---

### Issue 8: High Latency (>5 seconds per query)

**Symptoms:**
- Queries take too long to respond
- User experience feels slow

**Root Causes:**

1. **Too many LLM calls**
   - Every query going through LLM validation
   - Bypass threshold too high

2. **Embedding API slow**
   - Network issues
   - API rate limiting

3. **Large descriptions**
   - Long descriptions = more tokens = slower

**Solutions:**

**Option 1: Increase Bypass Threshold**
```python
# In query_analyzer.py line 241
if c1["score"] >= 0.50 and score_gap >= 0.10:  # Was 0.60, 0.15
    return immediately  # More queries bypass LLM
```

**Trade-off:** More risk of false positives

**Option 2: Reduce LLM Validation Range**
```python
# In query_analyzer.py line 250
if 0.40 <= c1["score"] < 0.50:  # Was 0.35-0.60
    call LLM  # Fewer queries validated
```

**Trade-off:** Less accurate scope detection

**Option 3: Check Embedding Cache**
```bash
ls -lh embeddings_cache.json
# Should exist and be ~500KB-2MB
```

If missing, embeddings are being generated every query → very slow.

---

### Issue 9: Disambiguation Triggered Too Often

**Symptoms:**
- Most queries result in "Which step did you mean?" response
- User frustrated by constant disambiguation

**Root Causes:**

1. **LLM returning AMBIGUOUS too often**
   - Prompt encourages AMBIGUOUS when uncertain
   - LLM being too cautious

2. **Multiple candidates passing thresholds**
   - Thresholds too loose
   - Many steps have similar descriptions

**Solutions:**

**Option 1: Tune LLM Prompt**
```python
# In prompts.py validate_scope_prompt()
# Change from:
"When in doubt between steps, return AMBIGUOUS with multiple IDs"
# To:
"Return AMBIGUOUS only if 2-3 steps are EQUALLY valid. If one is clearly better, return VALID with that step."
```

**Option 2: Stricter Thresholds**
```python
# In query_analyzer.py line 223-226
embedding_score >= 0.6  # Was 0.5
lexical_score >= 0.4    # Was 0.3
```

This filters out weaker candidates before LLM sees them.

---

### Issue 10: JSON Parsing Errors from LLM

**Symptoms:**
- Errors like `ValueError: Invalid JSON`
- LLM response not parsed correctly

**Root Causes:**

1. **LLM returned text before/after JSON**
   - "Sure, here's the validation: {...}"
   - Markdown code blocks wrapping JSON

2. **Incomplete JSON (truncated)**
   - max_tokens too low
   - Response cut off mid-JSON

3. **Malformed JSON**
   - LLM generated invalid syntax
   - Missing quotes, commas, brackets

**Solutions:**

**Step 1: Check max_tokens**
```python
# In query_analyzer.py line 375
max_tokens=1500  # Should be at least 1000
```

**Step 2: Enable Debug**
```bash
export GOV_DEBUG_MATCHING=1
python debug_single_query.py "Your query"
```

Look for "LLM Raw Response:" - check if JSON is complete.

**Step 3: Check parse_json_object()**
```python
# In cortex_utils.py
# Already handles:
# - Markdown code blocks
# - Text before/after JSON
# - Validation against schema
```

**Step 4: Increase Temperature to 0.0**
```python
# All LLM calls should use temperature=0.0
# For deterministic, well-formed responses
```

---

## Debugging Workflows

### Workflow 1: Debug Single Query

**Goal:** Understand why a specific query failed/succeeded

**Steps:**

1. Enable debug mode
```bash
export GOV_DEBUG_MATCHING=1
```

2. Run query
```bash
python debug_single_query.py "Your query here"
```

3. Analyze output
```
DEBUG: semantic_candidates (top 5)
  S1: score=0.574 (emb=0.612, lex=0.498, ...)
  S5: score=0.521 (emb=0.559, lex=0.445, ...)
  ...

DEBUG: LLM SCOPE VALIDATION
Query: Your query here
Candidates passed to LLM: 3
  - S1: ... (score=0.574, emb=0.612, lex=0.498)
  ...
Prompt length: 2847 chars

LLM Raw Response:
{"scope": "IN_SCOPE", "validation": "VALID", ...}

→ RESULT: VALID (confidence=0.90, step_ids=['S1'])
```

4. Check each stage:
   - Are semantic scores reasonable?
   - Did candidates pass thresholds?
   - Did LLM validate correctly?
   - Is final answer appropriate?

---

### Workflow 2: Test Regression

**Goal:** Ensure changes didn't break existing functionality

**Steps:**

1. Run test suite
```bash
python test_changes.py
```

2. Check for failures
```
Test 1: "I raised a JIRA ticket"
  Expected: S1
  Actual: S1
  ✓ PASS

Test 2: "What is Python?"
  Expected: None (out of scope)
  Actual: None
  ✓ PASS
```

3. Investigate failures
```bash
export GOV_DEBUG_MATCHING=1
python debug_single_query.py "Failed query"
```

---

### Workflow 3: Verify System Integrity

**Goal:** Check for known issues (NAN, stale cache, etc.)

**Steps:**

1. Check NAN filtering
```bash
python verify_nan_fix.py
```

All checks should PASS.

2. Check code version
```bash
python verify_code_version.py
```

All checks should PASS.

3. Check embedding cache
```bash
ls -lh embeddings_cache.json
cat embeddings_cache.json | grep version
# Should show "version": "2.0"
```

---

### Workflow 4: Performance Profiling

**Goal:** Identify bottlenecks

**Steps:**

1. Add timing
```python
import time

start = time.time()
# Your code here
print(f"Elapsed: {time.time() - start:.2f}s")
```

2. Profile key operations:
   - Excel loading: ~500ms
   - Embedding generation: ~100ms per step
   - Semantic search: ~500ms
   - LLM validation: ~2-3s
   - Response generation: ~1-2s

3. Identify slow operations
   - If embedding generation >1s per step → check API
   - If LLM calls >5s → check network/model
   - If semantic search >1s → check embedding index

---

## Error Messages and Meanings

### `KeyError: 'NAN'`
**Meaning:** NAN step ID not filtered
**Fix:** Run `verify_nan_fix.py`, ensure line 75-76 in governance_data.py is correct

### `TypeError: 'NoneType' object is not subscriptable`
**Meaning:** Trying to access field on None (e.g., `record["purpose"]` when record is None)
**Fix:** Check that `get_step_record()` returned a valid record

### `ValueError: Invalid JSON`
**Meaning:** LLM response not valid JSON
**Fix:** Check `max_tokens`, enable debug to see raw response

### `FileNotFoundError: Governance_Process_Updated.xlsx`
**Meaning:** Excel file not found
**Fix:** Check path in `governance_data.py`, ensure file exists

### `Exception: Cortex API generation failed`
**Meaning:** LLM API call failed
**Fix:** Check environment variables, network, API credentials

---

## Preventive Measures

### 1. Always Filter NAN
When modifying `governance_data.py`, ensure NAN filtering remains:
```python
valid_mask = (self.nodes_df["Step_ID"] != "NAN") & (self.nodes_df["Step_ID"].notna())
self.nodes_df = self.nodes_df[valid_mask].reset_index(drop=True)
```

### 2. Never Trust Numeric Scores in Prompts
Don't show scores to LLM - it creates threshold bias:
```python
# Good
candidate_info = {
    "id": "S1",
    "name": "...",
    "purpose": "...",
    "description": "..."
}

# Bad
candidate_info = {
    "id": "S1",
    "name": "...",
    "overall_score": 0.57  # LLM will see this and judge based on number!
}
```

### 3. Delete Cache After Formula Changes
If you modify scoring formula, delete cache:
```bash
rm embeddings_cache.json
```

### 4. Test Edge Cases
Always test:
- Out-of-scope queries ("What is Python?")
- Ambiguous queries ("Tell me about governance")
- Clear queries ("I raised a JIRA ticket")
- Rough queries ("we did the security thing")

### 5. Monitor Token Usage
Keep max_tokens appropriate for model:
- Gemini-2.5-Flash: 8192 max output
- Too low → truncation
- Too high → unnecessary cost

---

## Getting Help

### When Stuck

1. **Enable Debug Mode**
   ```bash
   export GOV_DEBUG_MATCHING=1
   ```

2. **Run Verification Scripts**
   ```bash
   python verify_nan_fix.py
   python verify_code_version.py
   ```

3. **Check Recent Changes**
   ```bash
   git log --oneline -10
   git diff HEAD~1
   ```

4. **Review Documentation**
   - [ARCHITECTURE.md](ARCHITECTURE.md): System overview
   - [MATCHING_SYSTEM.md](MATCHING_SYSTEM.md): Matching details
   - [CONFIGURATION.md](CONFIGURATION.md): Tuning parameters

### Logs to Provide

When reporting issues, include:
1. Full debug output (`GOV_DEBUG_MATCHING=1`)
2. Query that failed
3. Expected vs actual behavior
4. Recent code changes (git log)
5. Python version, environment

---

## Quick Reference

### Files to Check First

| Issue | File | Lines |
|-------|------|-------|
| Matching logic | `core/query_analyzer.py` | 197-273 |
| Scoring formula | `core/governance_data.py` | 413 |
| Thresholds | `core/query_analyzer.py` | 223-226, 241, 250 |
| LLM prompt | `core/prompts.py` | 194-247 |
| NAN filtering | `core/governance_data.py` | 75-76 |
| Response routing | `core/response_generator.py` | 12-43 |

### Commands to Run

| Goal | Command |
|------|---------|
| Debug single query | `export GOV_DEBUG_MATCHING=1; python debug_single_query.py "query"` |
| Verify NAN fix | `python verify_nan_fix.py` |
| Verify code version | `python verify_code_version.py` |
| Run tests | `python test_changes.py` |
| Regenerate embeddings | `rm embeddings_cache.json; python main.py` |
| Clear Python cache | `rm -rf core/__pycache__ automation_tools/__pycache__` |

### Critical Values

| Parameter | Value | Location |
|-----------|-------|----------|
| Embedding threshold | 0.5 | `query_analyzer.py:225` |
| Lexical threshold | 0.3 | `query_analyzer.py:225` |
| Bypass score | 0.60 | `query_analyzer.py:241` |
| Bypass gap | 0.15 | `query_analyzer.py:241` |
| LLM range | 0.35-0.60 | `query_analyzer.py:250` |
| Score formula | `(2*emb+lex)/3` | `governance_data.py:413` |
| Max tokens (validation) | 1500 | `query_analyzer.py:375` |
