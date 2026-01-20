# Architecture Overview

## What It Does

Maps informal user queries (e.g., "I raised a JIRA ticket") to governance workflow steps using hybrid search and LLM validation.

**Tech Stack:**
- LLM: Vertex AI Gemini-2.5-Flash
- Embeddings: Vertex AI text-embedding-004 (768D)
- Framework: LangGraph
- Data: Excel file (cjm_nodes_edges.xlsx)

---

## Pipeline Flow

```
User Query → 1. Query Analysis → 2. Automation (optional) → 3. Response → Answer
```

**Node 1: Query Analysis**
- Classify intent (greeting, ask_about_step, etc.)
- Match to steps (3-tier: deterministic → semantic → LLM)
- Extract step IDs

**Node 2: Automation Handler** (conditional)
- Run automations if step supports it

**Node 3: Response Generator**
- Generate answer based on intent and matches

---

## Three-Tier Matching System

### Tier 1: Deterministic (< 50ms)
Fast pattern matching: ID regex (`\bS\d+\b`), aliases, substrings, normalized verbs.
Only ID/alias matches (confidence >= 0.90) bypass Tier 2.

### Tier 2: Semantic (~500ms)
**Hybrid scoring:**
- 4 embeddings per step (name, purpose, description, combined)
- TF-IDF lexical matching (top 20 tokens)
- Final score: `(2.0 * embedding_score + lexical_score) / 3.0` (67% semantic, 33% lexical)

**Thresholds:**
- Must pass: embedding >= 0.5 AND lexical >= 0.3
- Bypass LLM if: score >= 0.60 AND gap >= 0.15
- Score < 0.35: reject immediately
- Score 0.35-0.59: go to Tier 3

### Tier 3: LLM Validation (~2-3s)
For borderline scores (0.35-0.59), LLM validates:
- **Scope:** IN_SCOPE vs OUT_OF_SCOPE
- **Match:** VALID / AMBIGUOUS / NO_MATCH
- Scores hidden from LLM (judges content, not numbers)
- Returns step IDs or disambiguation request

---

## Response Types

| Trigger | Response |
|---------|----------|
| 2+ referenced IDs | Compare steps |
| greeting intent | Welcome message |
| Multiple candidates | Disambiguation |
| Matched step | Step explanation + next steps |
| ask_next_step | Next step guidance |
| No match | Context-aware fallback |

---

## Key Design Choices

1. **Three tiers:** Fast deterministic → semantic hybrid → LLM scope check
2. **Hide scores from LLM:** Prevents threshold bias, judges semantic fit
3. **67/33 weighting:** Semantic captures meaning, lexical ensures artifacts present
4. **Force LLM for 0.35-0.59:** Catches out-of-scope queries with keyword overlap

---

## Data Source

**Excel:** `cjm_nodes_edges.xlsx`
- **Nodes:** Step_ID, Step_Name, Purpose, Description, Stage_Name, Automatable, Automation_step
- **Edges:** from, to
- **Critical:** NAN rows filtered at load time
