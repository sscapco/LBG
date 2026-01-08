#!/usr/bin/env python3
"""
Verify that NAN filtering is working at all levels.
This script confirms that the comprehensive NAN fix prevents any NAN contamination.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

print("=" * 100)
print("VERIFYING COMPREHENSIVE NAN FIX")
print("=" * 100)

# Check 1: DataFrame should have no NAN rows
print("\n[CHECK 1] Verify nodes_df has NO NAN rows after loading")
from core.governance_data import get_governance_data

gov_data = get_governance_data()

nan_count = (gov_data.nodes_df["Step_ID"] == "NAN").sum()
total_rows = len(gov_data.nodes_df)

print(f"  Total rows in nodes_df: {total_rows}")
print(f"  Rows with Step_ID == 'NAN': {nan_count}")

if nan_count == 0:
    print("  ✓ PASS: DataFrame properly filtered, no NAN rows")
else:
    print(f"  ✗ FAIL: Found {nan_count} NAN rows in DataFrame")

# Check 2: ordered_step_ids should not contain NAN
print("\n[CHECK 2] Verify ordered_step_ids does not contain NAN")
if "NAN" in gov_data.ordered_step_ids:
    print("  ✗ FAIL: 'NAN' found in ordered_step_ids")
else:
    print("  ✓ PASS: No 'NAN' in ordered_step_ids")

# Check 3: step_aliases should not have NAN
print("\n[CHECK 3] Verify step_aliases does not contain NAN key")
if "NAN" in gov_data.step_aliases:
    print("  ✗ FAIL: 'NAN' found in step_aliases")
else:
    print("  ✓ PASS: No 'NAN' in step_aliases")

# Check 4: step_embeddings should not have NAN
print("\n[CHECK 4] Verify step_embeddings does not contain NAN key")
if "NAN" in gov_data.step_embeddings:
    print("  ✗ FAIL: 'NAN' found in step_embeddings")
else:
    print("  ✓ PASS: No 'NAN' in step_embeddings")

# Check 5: get_step_record should return None for NAN
print("\n[CHECK 5] Verify get_step_record('NAN') returns None")
nan_record = gov_data.get_step_record("NAN")
if nan_record is None:
    print("  ✓ PASS: get_step_record('NAN') correctly returns None")
else:
    print("  ✗ FAIL: get_step_record('NAN') returned a record")

# Check 6: deterministic_match should never return NAN
print("\n[CHECK 6] Test deterministic_match doesn't return NAN")
test_queries = [
    "I've raised a JIRA ticket on the DP Demand Board",
    "Our security consultant delivered the Threat Model and SIR rating",
    "We created a record in ServiceNow and completed assessment questionnaire",
]

all_passed = True
for query in test_queries:
    sid, method, conf = gov_data.deterministic_match(query)
    print(f"\n  Query: {query[:60]}...")
    print(f"  Match: {sid} ({method}, {conf:.2f})")

    if sid == "NAN":
        print("  ✗ FAIL: Matched to NAN")
        all_passed = False
    elif sid is None:
        print("  ⚠ INFO: No deterministic match (will try semantic)")
    else:
        print("  ✓ PASS: Valid step ID returned")

if all_passed:
    print("\n  ✓ OVERALL: No NAN matches in deterministic matching")

# Check 7: Full pipeline test
print("\n[CHECK 7] Full pipeline test on Query 2")
from core.state import create_initial_state
from core.query_analyzer import analyze_query_node

query2 = "Our security consultant delivered the Threat Model and Security Design Artefact, and confirmed an SIR rating , what does that mean for our governance status?"

state = create_initial_state(query2, session_id="nan_test")
state["previous_state"] = None
result = analyze_query_node(state)

focus_id = result.get("focus_step_id")
print(f"  Query: {query2[:60]}...")
print(f"  Focus Step ID: {focus_id}")
print(f"  Intent: {result.get('intent')}")
print(f"  Match Method: {result.get('match_method')}")

if focus_id == "NAN":
    print("  ✗ FAIL: Pipeline returned NAN as focus_step_id")
elif focus_id is None:
    print("  ⚠ INFO: No match found (may need semantic/LLM reranking)")
else:
    print(f"  ✓ PASS: Valid step ID '{focus_id}' returned")

    # Also verify we can get the record without errors
    record = gov_data.get_step_record(focus_id)
    if record:
        print(f"  ✓ PASS: get_step_record succeeded")
        print(f"    Name: {record['name']}")
        # Try accessing purpose - this was causing the subscript error
        purpose_preview = str(record.get('purpose', ''))[:50]
        print(f"    Purpose: {purpose_preview}...")
    else:
        print(f"  ✗ FAIL: get_step_record returned None for {focus_id}")

print("\n" + "=" * 100)
print("VERIFICATION COMPLETE")
print("=" * 100)
print("\nKEY CHANGES IN THIS FIX:")
print("1. Filter NAN rows from DataFrame at load time (line 75 in governance_data.py)")
print("2. Add validation in get_step_record to reject NAN (line 349)")
print("3. Skip NAN rows in deterministic_match substring loop (line 408)")
print("\nThis creates defense-in-depth: filter at source + validate in functions.")
