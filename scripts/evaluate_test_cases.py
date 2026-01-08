#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


STEP_ID_RE = re.compile(r"\bS\d{1,3}\b", flags=re.IGNORECASE)

# Ensure repo root is on sys.path when running as a script 
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Expected:
    out_of_process: bool
    likely: Set[str]
    possible: Set[str]


def _norm_step_id(step_id: str) -> str:
    return (step_id or "").strip().upper()


def parse_expected_step_ids(value: str) -> Expected:
    raw = (value or "").strip()
    if not raw:
        return Expected(out_of_process=False, likely=set(), possible=set())

    if raw.strip().upper() == "OUT_OF_PROCESS":
        return Expected(out_of_process=True, likely=set(), possible=set())

    lower = raw.lower()

    # Labeled format: Likely: ... ; Possible: ...
    likely: Set[str] = set()
    possible: Set[str] = set()

    def _extract_ids(text: str) -> Set[str]:
        return {_norm_step_id(m.group(0)) for m in STEP_ID_RE.finditer(text or "")}

    # Split on semicolons to reduce false captures.
    parts = [p.strip() for p in re.split(r"[;|]", lower) if p.strip()]
    used = False
    for part in parts:
        if part.startswith("likely"):
            likely |= _extract_ids(part)
            used = True
        elif part.startswith("possible"):
            possible |= _extract_ids(part)
            used = True

    if used:
        return Expected(out_of_process=False, likely=likely, possible=possible)

    # Unlabeled: treat all IDs as "likely".
    return Expected(out_of_process=False, likely=_extract_ids(raw), possible=set())


def extract_expected_clarifying_questions(value: str) -> List[str]:
    raw = (value or "").strip()
    if not raw:
        return []
    # Allow either JSON array or semicolon/newline-delimited text.
    if raw.startswith("["):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    parts = [p.strip() for p in re.split(r"[;\n]+", raw) if p.strip()]
    return parts


def predicted_step_ids_from_state(final_state: Dict[str, Any]) -> List[str]:
    ids: List[str] = []

    for key in ("focus_step_id",):
        v = final_state.get(key)
        if isinstance(v, str) and v.strip():
            ids.append(_norm_step_id(v))

    for key in ("referenced_ids", "candidate_step_ids", "next_step_ids"):
        v = final_state.get(key)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    ids.append(_norm_step_id(item))

    # De-dupe while keeping order.
    seen: Set[str] = set()
    out: List[str] = []
    for sid in ids:
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def looks_like_clarifying_question(answer: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return False
    # Cheap heuristic: contains '?' or common question starters.
    if "?" in a:
        return True
    if re.search(r"(?i)\b(which|what|when|where|who|can you|could you|do you mean)\b", a):
        return True
    return False


def evaluate_one(
    *,
    test_id: str,
    test_type: str,
    prompt: str,
    expected_step_ids_raw: str,
    expected_step_ids: Expected,
    expected_clarifying_questions: List[str],
    session_id: str,
    session_store: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    # Import here so the script can be imported without side effects.
    from core.state import create_initial_state, session_state_to_previous_state, state_to_session_state
    from core.workflow import governance_graph
    from core.governance_data import get_governance_data

    previous_session = session_store.get(session_id)
    previous_state = session_state_to_previous_state(previous_session)

    initial_state = create_initial_state(prompt, session_id=session_id)
    initial_state["previous_state"] = previous_state

    final_state = governance_graph.invoke(initial_state)
    session_store[session_id] = state_to_session_state(final_state)

    answer = final_state.get("answer") or ""
    predicted = predicted_step_ids_from_state(final_state)
    predicted_set = set(predicted)

    governance_data = get_governance_data()
    top_candidates = governance_data.semantic_candidates(prompt, top_k=10)
    top_by_id = {c["id"]: c for c in top_candidates}
    returned_scores: Dict[str, Dict[str, Any]] = {}
    for sid in predicted:
        c = top_by_id.get(sid)
        if c:
            returned_scores[sid] = {
                "score": c.get("score"),
                "embedding_score": c.get("embedding_score"),
                "lexical_score": c.get("lexical_score"),
            }

    # Scoring
    hit_likely = bool(expected_step_ids.likely & predicted_set)
    hit_possible = bool(expected_step_ids.possible & predicted_set)
    hit_any = hit_likely or hit_possible

    out_of_process_ok = False
    if expected_step_ids.out_of_process:
        # Consider it correct if it doesn't commit to a specific step.
        out_of_process_ok = (final_state.get("focus_step_id") in (None, "")) and not (
            final_state.get("candidate_step_ids") or []
        )

    needs_disambiguation = bool(final_state.get("needs_disambiguation"))
    asked_question = looks_like_clarifying_question(answer)
    expects_disambiguation = bool(expected_clarifying_questions) or bool(expected_step_ids.possible)

    return {
        "test_id": test_id,
        "type": test_type,
        "session_id": session_id,
        "user_prompt": prompt,
        "expected_step_ids_raw": expected_step_ids_raw,
        "expected": {
            "out_of_process": expected_step_ids.out_of_process,
            "likely": sorted(expected_step_ids.likely),
            "possible": sorted(expected_step_ids.possible),
            "clarifying_questions": expected_clarifying_questions,
        },
        "predicted": {
            "intent": final_state.get("intent"),
            "focus_step_id": final_state.get("focus_step_id"),
            "candidate_step_ids": final_state.get("candidate_step_ids", []),
            "referenced_ids": final_state.get("referenced_ids", []),
            "next_step_ids": final_state.get("next_step_ids", []),
            "needs_disambiguation": needs_disambiguation,
            "predicted_step_ids": predicted,
            "returned_step_scores": returned_scores,
            "top_semantic_candidates": top_candidates,
        },
        "scoring": {
            "hit_likely": hit_likely,
            "hit_possible": hit_possible,
            "hit_any": hit_any,
            "out_of_process_ok": out_of_process_ok,
            "expects_disambiguation": expects_disambiguation,
            "needs_disambiguation": needs_disambiguation,
            "asked_question": asked_question,
        },
        "answer": answer,
    }


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0}

    ok_results = [r for r in results if isinstance(r, dict) and "expected" in r and "scoring" in r]
    error_results = [r for r in results if isinstance(r, dict) and "error" in r and ("expected" not in r or "scoring" not in r)]
    error_counts: Dict[str, int] = {}
    for r in error_results:
        err = str(r.get("error") or "unknown")
        error_counts[err] = error_counts.get(err, 0) + 1

    def _count(pred) -> int:
        return sum(1 for r in ok_results if pred(r))

    out_of_process = [r for r in ok_results if r["expected"].get("out_of_process")]
    in_process = [r for r in ok_results if not r["expected"].get("out_of_process")]

    summary: Dict[str, Any] = {
        "total": total,
        "ok": len(ok_results),
        "errors": len(error_results),
        "in_process": len(in_process),
        "out_of_process": len(out_of_process),
        "metrics": {
            "hit_any_rate": _count(lambda r: r["scoring"]["hit_any"]) / max(1, len(in_process)),
            "hit_likely_rate": _count(lambda r: r["scoring"]["hit_likely"]) / max(1, len(in_process)),
            "out_of_process_ok_rate": _count(lambda r: r["scoring"]["out_of_process_ok"]) / max(1, len(out_of_process)),
            "disambiguation_trigger_rate": _count(lambda r: r["scoring"]["needs_disambiguation"]) / max(1, len(ok_results)),
            "clarifying_question_rate": _count(lambda r: r["scoring"]["asked_question"]) / max(1, len(ok_results)),
            "error_rate": len(error_results) / max(1, total),
        },
        "error_samples": dict(sorted(error_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]),
        "by_type": {},
    }

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in ok_results:
        by_type.setdefault(r["type"] or "unknown", []).append(r)

    for t, group in by_type.items():
        in_proc = [r for r in group if not r["expected"]["out_of_process"]]
        oop = [r for r in group if r["expected"]["out_of_process"]]
        summary["by_type"][t] = {
            "total": len(group),
            "hit_any_rate": sum(1 for r in in_proc if r["scoring"]["hit_any"]) / max(1, len(in_proc)),
            "hit_likely_rate": sum(1 for r in in_proc if r["scoring"]["hit_likely"]) / max(1, len(in_proc)),
            "out_of_process_ok_rate": sum(1 for r in oop if r["scoring"]["out_of_process_ok"]) / max(1, len(oop)),
            "needs_disambiguation_rate": sum(1 for r in group if r["scoring"]["needs_disambiguation"]) / max(1, len(group)),
        }

    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate governance pipeline on test_cases.csv")
    parser.add_argument("--csv", required=True, help="Path to test_cases.csv (can be outside this repo).")
    parser.add_argument("--out", default=None, help="Write full JSON results to this file.")
    parser.add_argument("--csv-out", default=None, help="Write a flat CSV report to this file.")
    parser.add_argument("--csv-out-long", default=None, help="Write a long-form (one row per candidate) CSV report to this file.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top semantic candidates to include in CSV output.")
    parser.add_argument("--max", type=int, default=None, help="Max number of rows to run.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first exception.")
    parser.add_argument(
        "--session-col",
        default=None,
        help="Optional column name to use as session id (defaults to test_id).",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 2

    results: List[Dict[str, Any]] = []
    session_store: Dict[str, Dict[str, Any]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"test_id", "type", "user_prompt", "expected_step_ids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(f"Missing required CSV columns: {sorted(missing)}", file=sys.stderr)
            return 2

        for i, row in enumerate(reader, start=1):
            if args.max is not None and len(results) >= args.max:
                break

            test_id = str(row.get("test_id") or "").strip() or f"row_{i}"
            test_type = str(row.get("type") or "").strip()
            prompt = str(row.get("user_prompt") or "").strip()
            expected_raw = str(row.get("expected_step_ids") or "").strip()
            clar_raw = str(row.get("clarifying_questions") or "").strip()

            session_id = test_id
            if args.session_col:
                session_id = str(row.get(args.session_col) or test_id).strip() or test_id

            expected = parse_expected_step_ids(expected_raw)
            expected_clar = extract_expected_clarifying_questions(clar_raw)

            try:
                results.append(
                    evaluate_one(
                        test_id=test_id,
                        test_type=test_type,
                        prompt=prompt,
                        expected_step_ids_raw=expected_raw,
                        expected_step_ids=expected,
                        expected_clarifying_questions=expected_clar,
                        session_id=session_id,
                        session_store=session_store,
                    )
                )
            except Exception as e:
                if args.fail_fast:
                    raise
                results.append(
                    {
                        "test_id": test_id,
                        "type": test_type,
                        "session_id": session_id,
                        "user_prompt": prompt,
                        "expected_step_ids_raw": expected_raw,
                        "expected": {
                            "out_of_process": expected.out_of_process,
                            "likely": sorted(expected.likely),
                            "possible": sorted(expected.possible),
                            "clarifying_questions": expected_clar,
                        },
                        "error": repr(e),
                    }
                )

    summary = summarize(results)
    print(json.dumps(summary, indent=2))

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")

    if args.csv_out:
        csv_out_path = Path(args.csv_out).expanduser()
        csv_out_path.parent.mkdir(parents=True, exist_ok=True)

        top_k = max(1, int(args.top_k))

        base_fields = [
            "test_id",
            "type",
            "session_id",
            "user_prompt",
            "expected_step_ids_raw",
            "expected_out_of_process",
            "expected_likely",
            "expected_possible",
            "predicted_intent",
            "predicted_focus_step_id",
            "predicted_candidate_step_ids",
            "predicted_referenced_ids",
            "predicted_next_step_ids",
            "predicted_predicted_step_ids",
            "predicted_needs_disambiguation",
            "error",
        ]

        cand_fields: List[str] = []
        for i in range(1, top_k + 1):
            cand_fields.extend(
                [
                    f"top{i}_id",
                    f"top{i}_score",
                    f"top{i}_embedding_score",
                    f"top{i}_lexical_score",
                ]
            )

        fieldnames = base_fields + cand_fields

        with csv_out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                expected = r.get("expected") or {}
                predicted = r.get("predicted") or {}
                row: Dict[str, Any] = {
                    "test_id": r.get("test_id"),
                    "type": r.get("type"),
                    "session_id": r.get("session_id"),
                    "user_prompt": r.get("user_prompt"),
                    "expected_step_ids_raw": r.get("expected_step_ids_raw"),
                    "expected_out_of_process": expected.get("out_of_process"),
                    "expected_likely": ";".join(expected.get("likely") or []),
                    "expected_possible": ";".join(expected.get("possible") or []),
                    "predicted_intent": predicted.get("intent"),
                    "predicted_focus_step_id": predicted.get("focus_step_id"),
                    "predicted_candidate_step_ids": ";".join(predicted.get("candidate_step_ids") or []),
                    "predicted_referenced_ids": ";".join(predicted.get("referenced_ids") or []),
                    "predicted_next_step_ids": ";".join(predicted.get("next_step_ids") or []),
                    "predicted_predicted_step_ids": ";".join(predicted.get("predicted_step_ids") or []),
                    "predicted_needs_disambiguation": predicted.get("needs_disambiguation"),
                    "error": r.get("error"),
                }

                top_candidates = predicted.get("top_semantic_candidates") or []
                for i in range(1, top_k + 1):
                    idx = i - 1
                    c = top_candidates[idx] if idx < len(top_candidates) else {}
                    row[f"top{i}_id"] = c.get("id")
                    row[f"top{i}_score"] = c.get("score")
                    row[f"top{i}_embedding_score"] = c.get("embedding_score")
                    row[f"top{i}_lexical_score"] = c.get("lexical_score")

                writer.writerow(row)

    if args.csv_out_long:
        csv_out_path = Path(args.csv_out_long).expanduser()
        csv_out_path.parent.mkdir(parents=True, exist_ok=True)

        top_k = max(1, int(args.top_k))
        fieldnames = [
            "test_id",
            "type",
            "session_id",
            "user_prompt",
            "expected_step_ids_raw",
            "expected_out_of_process",
            "expected_likely",
            "expected_possible",
            "predicted_step_ids",
            "predicted_intent",
            "predicted_focus_step_id",
            "predicted_needs_disambiguation",
            "candidate_rank",
            "candidate_step_id",
            "score",
            "embedding_score",
            "lexical_score",
            "is_predicted_step",
            "is_expected_likely",
            "is_expected_possible",
            "error",
        ]

        with csv_out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                expected = r.get("expected") or {}
                predicted = r.get("predicted") or {}

                predicted_step_ids = predicted.get("predicted_step_ids") or []
                predicted_set = set(predicted_step_ids)
                expected_likely = expected.get("likely") or []
                expected_possible = expected.get("possible") or []
                expected_likely_set = set(expected_likely)
                expected_possible_set = set(expected_possible)

                top_candidates = (predicted.get("top_semantic_candidates") or [])[:top_k]

                # If the row errored before candidates were computed, still emit one row.
                if not top_candidates:
                    writer.writerow(
                        {
                            "test_id": r.get("test_id"),
                            "type": r.get("type"),
                            "session_id": r.get("session_id"),
                            "user_prompt": r.get("user_prompt"),
                            "expected_step_ids_raw": r.get("expected_step_ids_raw"),
                            "expected_out_of_process": expected.get("out_of_process"),
                            "expected_likely": ";".join(expected_likely),
                            "expected_possible": ";".join(expected_possible),
                            "predicted_step_ids": ";".join(predicted_step_ids),
                            "predicted_intent": predicted.get("intent"),
                            "predicted_focus_step_id": predicted.get("focus_step_id"),
                            "predicted_needs_disambiguation": predicted.get("needs_disambiguation"),
                            "candidate_rank": "",
                            "candidate_step_id": "",
                            "score": "",
                            "embedding_score": "",
                            "lexical_score": "",
                            "is_predicted_step": "",
                            "is_expected_likely": "",
                            "is_expected_possible": "",
                            "error": r.get("error"),
                        }
                    )
                    continue

                for idx, c in enumerate(top_candidates, start=1):
                    sid = c.get("id")
                    writer.writerow(
                        {
                            "test_id": r.get("test_id"),
                            "type": r.get("type"),
                            "session_id": r.get("session_id"),
                            "user_prompt": r.get("user_prompt"),
                            "expected_step_ids_raw": r.get("expected_step_ids_raw"),
                            "expected_out_of_process": expected.get("out_of_process"),
                            "expected_likely": ";".join(expected_likely),
                            "expected_possible": ";".join(expected_possible),
                            "predicted_step_ids": ";".join(predicted_step_ids),
                            "predicted_intent": predicted.get("intent"),
                            "predicted_focus_step_id": predicted.get("focus_step_id"),
                            "predicted_needs_disambiguation": predicted.get("needs_disambiguation"),
                            "candidate_rank": idx,
                            "candidate_step_id": sid,
                            "score": c.get("score"),
                            "embedding_score": c.get("embedding_score"),
                            "lexical_score": c.get("lexical_score"),
                            "is_predicted_step": bool(sid and sid in predicted_set),
                            "is_expected_likely": bool(sid and sid in expected_likely_set),
                            "is_expected_possible": bool(sid and sid in expected_possible_set),
                            "error": r.get("error"),
                        }
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
