# Matching System

## Overview

Three-tier matching: deterministic → semantic → LLM validation

---

## Tier 1: Deterministic

**Location:** `governance_data.py:deterministic_match()`
**Latency:** <50ms

**Priority:**
1. ID regex `\b(s\d{1,3})\b` → confidence 1.0
2. Alias match → confidence 0.95
3. Substring match → confidence 0.90
4. Normalized verbs → confidence 0.88
5. Trigrams → confidence 0.85

**Bypass to answer:** Only ID/alias with confidence >= 0.90

## Tier 2: Semantic

**Location:** `governance_data.py:semantic_candidates()`
**Latency:** ~500ms

**Multi-field embeddings:**
- 4 per step: name, purpose, description, combined
- Weighted: 40% best field + 30% combined + 15% + 15%

**Lexical (TF-IDF):**
- Top 20 tokens by IDF
- Weighted overlap score

**Final score:** `(2.0 * embedding_score + lexical_score) / 3.0`
- 67% semantic, 33% lexical

**Threshold filter:**
- embedding >= 0.5 AND lexical >= 0.3

**Routing:**
- score >= 0.60 AND gap >= 0.15 → bypass LLM, return match
- 0.35 <= score < 0.60 → Tier 3
- score < 0.35 → reject

## Tier 3: LLM Validation

**Location:** `query_analyzer.py:_llm_validate_scope()`
**Latency:** ~2-3s

**Trigger:** Score 0.35-0.59

**Model:** Gemini-2.5-Flash, 1500 tokens, temp 0.0

**Input:** Candidate details (id, name, purpose, description) - **scores hidden**

**LLM returns:**
```json
{
  "scope": "IN_SCOPE" | "OUT_OF_SCOPE",
  "validation": "VALID" | "AMBIGUOUS" | "NO_MATCH",
  "step_ids": ["S1"] or ["S1", "S2"],
  "confidence": 0.85
}
```

**Why hide scores?** LLM saw 0.57 and rejected it as "low" despite semantic fit. Now judges content only.

**Handling:**
- OUT_OF_SCOPE → reject
- NO_MATCH → reject
- VALID → return step(s), disambiguate if multiple
- AMBIGUOUS → disambiguate

## Match Methods

Values in `state["match_method"]`:

**Success:**
- `id_match`, `alias_match`, `substring_match`, `normalized_match`, `partial_match` (Tier 1)
- `high_confidence_match`, `single_valid_match` (Tier 2)
- `llm_validated`, `llm_validated_ambiguous` (Tier 3)

**Failure:**
- `below_threshold`, `low_confidence` (Tier 2)
- `out_of_scope`, `no_match_validated` (Tier 3)

---

## Disambiguation

Triggered when multiple candidates returned. LLM explains all steps and asks user to clarify.

---

## Caching

**File:** `embeddings_cache.json` (version 2.0)
**Invalidates:** On Excel file modification
**Structure:** Pre-computed embedding matrix for fast similarity search

---

## Debug Mode

```bash
export GOV_DEBUG_MATCHING=1
python debug_single_query.py "query"
```

Shows: candidate scores, LLM validation I/O, decision traces

---

## Common Issues

1. **Score inflation** - Was dividing by 2 not 3 in formula
2. **Out-of-scope matching** - Fixed by forcing LLM validation for 0.35-0.59
3. **Valid queries rejected** - Fixed by hiding scores from LLM
4. **NAN contamination** - Fixed by filtering at load time
5. **Token limit exceeded** - Fixed by increasing to 1500 tokens
