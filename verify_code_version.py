#!/usr/bin/env python3
"""
Verify that the actual running code has our changes.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

print("=" * 100)
print("VERIFYING CODE VERSION")
print("=" * 100)

# Check 1: Import and check source code
print("\n[CHECK 1] Verify intent classification has new logic")
import inspect
from core.query_analyzer import _classify_intent

source = inspect.getsource(_classify_intent)
if "has_work_keywords" in source:
    print("✓ PASS: New intent classification fallback logic present")
else:
    print("✗ FAIL: Old intent classification code")

# Check 2: Verify multi-field embeddings
print("\n[CHECK 2] Verify multi-field embeddings in semantic_candidates")
from core.governance_data import GovernanceDataLoader

source = inspect.getsource(GovernanceDataLoader.semantic_candidates)
if "multi_field_score" in source:
    print("✓ PASS: Multi-field embedding logic present")
else:
    print("✗ FAIL: Old single embedding logic")

# Check 3: Verify normalization
print("\n[CHECK 3] Verify text normalization exists")
if hasattr(GovernanceDataLoader, '_normalize_text'):
    print("✓ PASS: _normalize_text method exists")
    source = inspect.getsource(GovernanceDataLoader._normalize_text)
    if "raised" in source and "raise" in source:
        print("✓ PASS: Verb normalization present")
    else:
        print("✗ FAIL: Normalization method exists but missing verb mappings")
else:
    print("✗ FAIL: _normalize_text method missing")

# Check 4: Verify lexical token limit
print("\n[CHECK 4] Verify lexical similarity token limit")
source = inspect.getsource(GovernanceDataLoader.lexical_similarity)
if "min(20," in source:
    print("✓ PASS: Token limit is 20")
elif "min(6," in source:
    print("✗ FAIL: Token limit still 6 (old code)")
else:
    print("⚠ WARNING: Cannot determine token limit")

# Check 5: Verify LLM reranking
print("\n[CHECK 5] Verify LLM reranking exists")
from core.query_analyzer import _llm_rerank_candidates
print(f"✓ PASS: _llm_rerank_candidates function exists")

# Check 6: Actually load governance data and check embeddings
print("\n[CHECK 6] Load governance data and verify embedding structure")
from core.governance_data import get_governance_data

gov_data = get_governance_data()
print(f"  Loaded {len(gov_data.step_embeddings)} steps")

if gov_data.step_embeddings:
    first_step_id = list(gov_data.step_embeddings.keys())[0]
    first_step = gov_data.step_embeddings[first_step_id]

    has_multi = all(k in first_step for k in ["name_embedding", "purpose_embedding", "description_embedding"])
    if has_multi:
        print(f"✓ PASS: Step {first_step_id} has multi-field embeddings")
        print(f"  Keys: {list(first_step.keys())}")
    else:
        print(f"✗ FAIL: Step {first_step_id} missing multi-field embeddings")
        print(f"  Keys: {list(first_step.keys())}")
        print("  This means the cache is old or wasn't regenerated!")

# Check 7: Test actual matching on one query
print("\n[CHECK 7] Test matching on sample query")
test_query = "I've raised a JIRA ticket on the DP Demand Board"

# Deterministic match
sid, method, conf = gov_data.deterministic_match(test_query)
print(f"  Deterministic match: {sid} ({method}, {conf:.2f})")

if sid == "S1":
    print(f"  ✓ PASS: Correctly matched S1")
else:
    # Try semantic
    candidates = gov_data.semantic_candidates(test_query, top_k=3)
    print(f"  Deterministic didn't find S1, trying semantic...")
    if candidates:
        print(f"  Top 3 semantic candidates:")
        for c in candidates[:3]:
            print(f"    {c['id']}: {c['score']:.3f} (lex={c.get('lexical_score', 0):.3f})")

        if candidates[0]['id'] == 'S1':
            print(f"  ✓ PASS: S1 is top semantic match")
        else:
            print(f"  ✗ FAIL: S1 not found. Top match is {candidates[0]['id']}")
    else:
        print(f"  ✗ FAIL: No semantic candidates found")

# Check 8: Verify the actual workflow graph
print("\n[CHECK 8] Verify workflow graph structure")
from core.workflow import governance_graph

print(f"  Workflow graph exists: {governance_graph is not None}")
print(f"  Graph type: {type(governance_graph)}")

# Check 9: Most important - check if pycache might be the issue
print("\n[CHECK 9] Check for stale .pyc files")
import os
pycache_files = list(Path(REPO_ROOT / "core" / "__pycache__").glob("*.pyc"))
print(f"  Found {len(pycache_files)} .pyc files in core/__pycache__/")

for pyc in pycache_files:
    print(f"    {pyc.name}")

print("\n" + "=" * 100)
print("VERIFICATION COMPLETE")
print("=" * 100)
print("\nIf Check 6 FAILED, the embeddings cache is old.")
print("Delete the cache file and rerun.")
print("\nIf other checks FAILED, there may be:")
print("1. Stale .pyc files (delete core/__pycache__/)")
print("2. Multiple Python environments")
print("3. Code not actually committed/pulled on VM")
