#!/usr/bin/env python3
"""Proactive surfacing evaluation harness.

Measures how well the proactive engine surfaces the right memories
for a given ambient context — without being explicitly asked.

Usage:
    python evals/proactive_eval.py
    python evals/proactive_eval.py --category health_safety
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_DEFAULT_CASES = Path(__file__).parent / "proactive_cases.jsonl"


def _load_cases(path: str) -> list[dict]:  # type: ignore[type-arg]
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _precision_at_k(
    surfaced: list[str], expected: set[str], k: int,
) -> float:
    """Of top-k surfaced, how many are in expected?"""
    if not expected:
        return 1.0 if not surfaced[:k] else 0.0
    top = surfaced[:k]
    if not top:
        return 0.0
    return sum(1 for s in top if s in expected) / len(top)


def _anti_noise(surfaced: list[str], forbidden: set[str]) -> float:
    """1.0 if no forbidden IDs surfaced, 0.0 otherwise."""
    if not forbidden:
        return 1.0
    return 0.0 if any(s in forbidden for s in surfaced) else 1.0


def _recall_at_k(
    surfaced: list[str], expected: set[str], k: int,
) -> float:
    """Of expected, how many appear in top-k surfaced?"""
    if not expected:
        return 1.0
    top = set(surfaced[:k])
    return sum(1 for e in expected if e in top) / len(expected)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _print_report(results: list[dict]) -> None:  # type: ignore[type-arg]
    """Print evaluation report."""
    n = len(results)
    p3 = sum(r['p3'] for r in results) / n
    r3 = sum(r['r3'] for r in results) / n
    f1 = sum(r['f1'] for r in results) / n
    an = sum(r['anti_noise'] for r in results) / n
    print(f"\n{'=' * 60}")
    print(f"  Proactive Eval: {n} cases")
    print(f"{'=' * 60}")
    print(f"  Precision@3:  {p3:.3f}")
    print(f"  Recall@3:     {r3:.3f}")
    print(f"  F1-proactive: {f1:.3f}")
    print(f"  Anti-noise:   {an:.3f}")
    _print_by_category(results)


def _print_by_category(results: list[dict]) -> None:  # type: ignore[type-arg]
    cats: dict[str, list[dict]] = {}  # type: ignore[type-arg]
    for r in results:
        cats.setdefault(r['category'], []).append(r)
    print(f"\n{'─' * 50}")
    for cat, rows in sorted(cats.items()):
        n = len(rows)
        p3 = sum(r['p3'] for r in rows) / n
        an = sum(r['anti_noise'] for r in rows) / n
        print(f"  {cat:<16}  P@3={p3:.2f}  AN={an:.2f}  ({n})")


def evaluate_case(
    case: dict, surface_fn: object,  # type: ignore[type-arg]
) -> dict:  # type: ignore[type-arg]
    """Evaluate one proactive case. Returns metrics dict."""
    expected = set(case.get("expected_surface_ids", []))
    forbidden = set(case.get("forbidden_surface_ids", []))
    # placeholder: actual engine call will go here
    surfaced: list[str] = []
    p3 = _precision_at_k(surfaced, expected, 3)
    r3 = _recall_at_k(surfaced, expected, 3)
    return {
        "context": case["context"],
        "category": case.get("category", ""),
        "expected": expected,
        "surfaced": surfaced,
        "p3": p3,
        "r3": r3,
        "f1": _f1(p3, r3),
        "anti_noise": _anti_noise(surfaced, forbidden),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Proactive eval")
    parser.add_argument(
        "--cases", default=str(_DEFAULT_CASES), help="Cases JSONL",
    )
    parser.add_argument("--category", help="Filter by category")
    args = parser.parse_args()
    cases = _load_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]
    print(f"Loaded {len(cases)} proactive cases")
    results = [evaluate_case(c, None) for c in cases]
    _print_report(results)


if __name__ == "__main__":
    main()
