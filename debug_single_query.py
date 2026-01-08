#!/usr/bin/env python3
"""
Debug a single query through the pipeline to see what's happening at each step.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from core.state import create_initial_state
from core.query_analyzer import analyze_query_node
from core.automation_handler import automation_handler_node
from core.response_generator import response_generator_node
from core.governance_data import get_governance_data
import json

# Test queries from your examples
test_queries = [
    "I've raised a JIRA ticket on the DP Demand Board to start the request , where are we in the governance journey right now?",
    "Our security consultant delivered the Threat Model and Security Design Artefact, and confirmed an SIR rating , what does that mean for our governance status?",
    "We created a record in ServiceNow and completed a big assessment questionnaire that generated the tasks , what does this indicate about our governance progress?",
]

print("=" * 100)
print("DEBUGGING PIPELINE EXECUTION")
print("=" * 100)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'=' * 100}")
    print(f"TEST QUERY {i}")
    print(f"{'=' * 100}")
    print(f"Query: {query}\n")

    # Step 1: Create initial state
    initial_state = create_initial_state(query, session_id=f"test_{i}")
    initial_state["previous_state"] = None

    print("[STEP 1] Initial State Created")
    print(f"  user_message: {initial_state['user_message'][:80]}...")
    print()

    # Step 2: Analyze query
    print("[STEP 2] Running analyze_query_node...")
    state_after_analyze = analyze_query_node(initial_state.copy())

    print(f"  Intent: {state_after_analyze.get('intent')}")
    print(f"  Focus Step ID: {state_after_analyze.get('focus_step_id')}")
    print(f"  Candidate Step IDs: {state_after_analyze.get('candidate_step_ids', [])}")
    print(f"  Referenced IDs: {state_after_analyze.get('referenced_ids', [])}")
    print(f"  Match Method: {state_after_analyze.get('match_method')}")
    print(f"  Match Confidence: {state_after_analyze.get('match_confidence')}")
    print(f"  Needs Disambiguation: {state_after_analyze.get('needs_disambiguation')}")
    print()

    # Step 3: Automation handler
    print("[STEP 3] Running automation_handler_node...")
    state_after_automation = automation_handler_node(state_after_analyze.copy())

    print(f"  Focus Step ID (after automation): {state_after_automation.get('focus_step_id')}")
    print(f"  Automation Result: {state_after_automation.get('automation_result')}")
    print()

    # Step 4: Response generator
    print("[STEP 4] Running response_generator_node...")
    final_state = response_generator_node(state_after_automation.copy())

    print(f"  Focus Step ID (final): {final_state.get('focus_step_id')}")
    print(f"  Candidate Step IDs (final): {final_state.get('candidate_step_ids', [])}")
    print(f"  Referenced IDs (final): {final_state.get('referenced_ids', [])}")
    print(f"  Next Step IDs (final): {final_state.get('next_step_ids', [])}")
    print(f"  Explained Step IDs (final): {final_state.get('explained_step_ids', [])}")
    print(f"  Answer length: {len(final_state.get('answer', ''))} chars")
    print()

    # Step 5: What the evaluation sees
    print("[STEP 5] What evaluate_test_cases.py would extract:")

    predicted_ids = []
    focus_id = final_state.get("focus_step_id")
    if focus_id:
        predicted_ids.append(focus_id)

    for key in ("referenced_ids", "candidate_step_ids", "next_step_ids"):
        v = final_state.get(key)
        if isinstance(v, list):
            predicted_ids.extend([str(x).upper() for x in v if x])

    # Dedupe
    seen = set()
    unique_ids = []
    for sid in predicted_ids:
        if sid and sid not in seen:
            seen.add(sid)
            unique_ids.append(sid)

    print(f"  Predicted Step IDs: {unique_ids}")
    print()

    # Show the actual step details if we got a match
    if final_state.get("focus_step_id"):
        gov_data = get_governance_data()
        step_record = gov_data.get_step_record(final_state["focus_step_id"])
        if step_record:
            print("[STEP INFO]")
            print(f"  ID: {step_record['id']}")
            print(f"  Name: {step_record['name']}")
            print(f"  Purpose: {step_record['purpose'][:100]}...")
            print()

    print(f"{'=' * 100}\n")

print("\n" + "=" * 100)
print("DEBUGGING COMPLETE")
print("=" * 100)
print("\nKEY THINGS TO CHECK:")
print("1. Is the intent correct after analyze_query_node?")
print("2. Is focus_step_id populated after analyze_query_node?")
print("3. Does focus_step_id survive through automation_handler and response_generator?")
print("4. Are the predicted_step_ids what you expect?")
