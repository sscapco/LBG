#!/usr/bin/env python3
"""
Test intent classification in isolation to see if the LLM is working correctly.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from core.query_analyzer import _classify_intent

test_cases = [
    ("I've raised a JIRA ticket on the DP Demand Board", "ask_about_step"),
    ("We created a record in ServiceNow and completed assessment questionnaire", "ask_about_step"),
    ("Our security consultant delivered the Threat Model and SIR", "ask_about_step"),
    ("Hello, how are you?", "greeting"),
    ("What's next after this step?", "ask_next_step"),
]

print("=" * 100)
print("TESTING INTENT CLASSIFICATION")
print("=" * 100)

for query, expected in test_cases:
    print(f"\nQuery: {query}")
    print(f"Expected: {expected}")

    intent, confidence, meta = _classify_intent(query)

    print(f"Got: {intent} (confidence: {confidence:.2f})")

    if meta:
        print(f"Reasoning: {meta.get('reasoning', 'N/A')}")

    if intent == expected:
        print("✓ PASS")
    else:
        print(f"✗ FAIL - Expected '{expected}' but got '{intent}'")
        if intent == "greeting" and expected != "greeting":
            print("  WARNING: Incorrectly classified as greeting!")

    print("-" * 100)

print("\n" + "=" * 100)
print("If multiple tests fail, the LLM intent classification might not be working.")
print("Check if the cortex LLM calls are succeeding or timing out.")
print("=" * 100)
