#!/usr/bin/env python3
"""
Quick test to verify all changes are working correctly.
Run this on the VM after pulling the latest code.
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

print("=" * 80)
print("TESTING GOVERNANCE PIPELINE CHANGES")
print("=" * 80)

# Test 1: Check intent classification improvements
print("\n[TEST 1] Intent Classification")
print("-" * 80)
from core.query_analyzer import _classify_intent

test_queries = [
    "We created a record in ServiceNow and completed assessment questionnaire",
    "I've raised a JIRA ticket on the DP Demand Board",
    "Hello, how are you?",
]

for query in test_queries:
    intent, confidence, meta = _classify_intent(query)
    print(f"Query: {query[:60]}...")
    print(f"  Intent: {intent} (confidence: {confidence:.2f})")

    # Check for correct intent
    if "ServiceNow" in query or "JIRA" in query:
        if intent == "greeting":
            print("  ❌ FAIL: Should be 'ask_about_step', not 'greeting'")
        else:
            print("  ✓ PASS: Correct intent")
    elif "Hello" in query:
        if intent == "greeting":
            print("  ✓ PASS: Correct greeting detection")
        else:
            print("  ❌ FAIL: Should be 'greeting'")
    print()

# Test 2: Check multi-field embeddings
print("\n[TEST 2] Multi-Field Embeddings")
print("-" * 80)
from core.governance_data import get_governance_data

gov_data = get_governance_data()
if gov_data.step_embeddings:
    first_step_id = list(gov_data.step_embeddings.keys())[0]
    first_step = gov_data.step_embeddings[first_step_id]

    print(f"Sample step ID: {first_step_id}")
    print(f"Keys available: {list(first_step.keys())}")

    has_multi_field = all(k in first_step for k in ["name_embedding", "purpose_embedding", "description_embedding"])
    if has_multi_field:
        print("✓ PASS: Multi-field embeddings present")
    else:
        print("❌ FAIL: Multi-field embeddings missing")
        print("  This means old cache was loaded. Delete cache and retry!")
else:
    print("❌ FAIL: No embeddings loaded")

# Test 3: Check normalization
print("\n[TEST 3] Text Normalization")
print("-" * 80)

test_text = "I raised a JIRA ticket and created a record"
normalized = gov_data._normalize_text(test_text)
print(f"Original:   {test_text}")
print(f"Normalized: {normalized}")

if "raise" in normalized and "create" in normalized:
    print("✓ PASS: Verbs normalized correctly")
else:
    print("❌ FAIL: Normalization not working")

# Test 4: Check lexical token limit
print("\n[TEST 4] Lexical Similarity Token Limit")
print("-" * 80)
import inspect
source = inspect.getsource(gov_data.lexical_similarity)
if "min(20," in source:
    print("✓ PASS: Token limit increased to 20")
elif "min(6," in source:
    print("❌ FAIL: Token limit still at 6 (old code)")
else:
    print("⚠ WARNING: Cannot determine token limit")

# Test 5: Check scoring weights
print("\n[TEST 5] Semantic/Lexical Balance")
print("-" * 80)
source = inspect.getsource(gov_data.semantic_candidates)
if "0.50 * multi_field_score" in source and "0.50 * lex_score" in source:
    print("✓ PASS: 50/50 semantic/lexical weighting")
elif "0.75" in source and "0.25" in source:
    print("❌ FAIL: Old 75/25 weighting still in use")
else:
    print("⚠ WARNING: Cannot determine weighting")

# Test 6: Check LLM reranking
print("\n[TEST 6] LLM Reranking")
print("-" * 80)
from core.query_analyzer import _llm_rerank_candidates
print(f"_llm_rerank_candidates function exists: {callable(_llm_rerank_candidates)}")
print("✓ PASS: LLM reranking available")

# Test 7: Quick semantic match test
print("\n[TEST 7] Semantic Matching Test")
print("-" * 80)

test_query = "I've raised a JIRA ticket on the DP Demand Board"
candidates = gov_data.semantic_candidates(test_query, top_k=3)

if candidates:
    print(f"Query: {test_query}")
    print(f"\nTop 3 candidates:")
    for i, c in enumerate(candidates[:3], 1):
        print(f"  {i}. {c['id']}: {c['score']:.3f} "
              f"(lex={c.get('lexical_score', 0):.3f}, "
              f"name_sim={c.get('name_similarity', 0):.3f})")

    # S1 should be top match
    if candidates[0]['id'] == 'S1':
        print("\n✓ PASS: S1 correctly identified as top match")
    else:
        print(f"\n❌ FAIL: Expected S1, got {candidates[0]['id']}")
else:
    print("❌ FAIL: No candidates returned")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\nIf any tests failed, make sure you:")
print("1. Pulled the latest code: git pull origin maggie_dev")
print("2. Deleted the embedding cache (check constants.EMBED_CACHE_PATH)")
print("3. Are running this script on the VM with the correct Python environment")
