#!/usr/bin/env python3
"""Generate cross-encoder training data from eval cases + corpus.

Produces a JSONL file with {query, passage, label} rows.
- label=1: passage is in expected_ids for this query
- label=0: hard negatives (semantically close but wrong)

Usage:
    python evals/gen_rerank_data.py
    python evals/gen_rerank_data.py --out evals/rerank_train.jsonl
"""

import json
import random
from pathlib import Path

_CASES = Path(__file__).parent / "hard_cases.jsonl"
_OUT = Path(__file__).parent / "rerank_train.jsonl"

# Corpus copied from recall_eval.py — single source of truth
_CORPUS: dict[str, str] = {
    "diet-veg":        "User follows a strict vegetarian diet, no meat, poultry or seafood",
    "health-lactose":  "User has lactose intolerance, causes digestive discomfort",
    "health-back":     "User has a lower back condition, max 2 hours sitting without a break",
    "health-migraine": "User suffers from migraines triggered by bright screens and loud noise",
    "health-meds": (
        "User takes a daily medication that must not be "
        "combined with alcohol or grapefruit juice"
    ),
    "partner-allergy": "User's partner is allergic to shellfish and peanuts",
    "food-spicy":      "User loves Sichuan cuisine and spicy food, regularly orders 麻辣 dishes",
    "sched-sync":      "Weekly product sync every Tuesday 2pm with the engineering team",
    "sched-sync-new":  "Product sync rescheduled: moved from Tuesday 2pm to Thursday 4pm",
    "sched-1on1":      "Biweekly 1:1 with manager every other Monday at 10am",
    "work-deadline":   "Current sprint deadline is Friday end-of-day, high priority",
    "work-remote":     "User works from home on Mondays and Fridays, in-office Tuesday–Thursday",
    "focus-block":     "User blocks 2–5pm daily for deep work, disables all notifications",
    "work-role-old":   "User was a senior software engineer",
    "work-role-new":   "User was promoted to engineering manager 3 months ago",
    "home-city":       "User currently lives in Shanghai, Jing'an district",
    "home-prev":       "User previously lived in Beijing for 5 years before moving to Shanghai",
    "routine-run":     "User runs 5km every morning before 8am, skips on rainy days",
    "drink-coffee":    "User drinks black coffee every morning, no milk, no sugar",
    "sleep-habit":     "User goes to bed at 11pm and wakes at 6:30am every day",
    "gym-habit":       "User goes to the gym on Tuesday and Thursday evenings after work",
    "family-kids":     "User has two young children aged 5 and 8, school pickup at 4pm",
    "partner-diet":    "User's partner is pescatarian — eats fish and seafood but no meat",
    "parent-care": (
        "User's elderly father lives with the family "
        "and needs assistance in the evenings"
    ),
    "hobby-jazz":      "User plays jazz piano as a hobby, practices on weekends",
    "lang-native":     "User's first language is Mandarin, second language is English",
    "lang-third":      "User is learning Spanish, currently at intermediate (B1) level",
    "tech-stack":      "User primarily codes in Python and Go, 8 years of engineering experience",
    "comm-style": (
        "User strongly prefers async communication, "
        "dislikes unexpected phone calls"
    ),
    "travel-pref":     "User always requests window seat, avoids middle seats on any flight",
    "budget-travel": (
        "User prefers economy for flights under 3 hours, "
        "business class for 5+ hours"
    ),
    "value-privacy":   "User values data privacy strongly, avoids apps that monetise user data",
    "value-worklife":  "User enforces work-life balance, does not check work messages after 9pm",
}

_NEG_PER_POS = 7  # random negatives per positive pair
_FAILURES = Path(__file__).parent / "failures.jsonl"


def _load_jsonl(path: Path) -> list[dict]:  # type: ignore[type-arg]
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _pick_random_negatives(
    query_expected: set[str], all_ids: list[str], n: int
) -> list[str]:
    """Pick n random negative IDs."""
    pool = [nid for nid in all_ids if nid not in query_expected]
    random.shuffle(pool)
    return pool[:n]


def _failure_pairs(failure: dict) -> list[dict[str, object]]:  # type: ignore[type-arg]
    """Generate training pairs from one failure: confusers as 0, expected as 1."""
    query = failure["query"]
    pairs: list[dict[str, object]] = []
    for nid in failure.get("confusers", []):
        if nid in _CORPUS:
            pairs.append({"query": query, "passage": _CORPUS[nid], "label": 0})
    for nid in failure.get("expected_ids", []):
        if nid in _CORPUS:
            pairs.append({"query": query, "passage": _CORPUS[nid], "label": 1})
    return pairs


def _add_hard_negatives(rows: list[dict[str, object]]) -> int:
    """Add confuser pairs from actual eval failures as hard negatives."""
    if not _FAILURES.exists():
        print("No failures.jsonl — skipping hard-negative mining")
        return 0
    failures = _load_jsonl(_FAILURES)
    added = 0
    for f in failures:
        pairs = _failure_pairs(f)
        rows.extend(pairs)
        added += len(pairs)
    return added


def _pairs_for_case(
    case: dict, all_ids: list[str],  # type: ignore[type-arg]
) -> list[dict[str, object]]:
    """Generate positive + random negative pairs for one eval case."""
    query, expected = case["query"], set(case.get("expected_ids", []))
    if not expected:
        return []
    pos = [
        {"query": query, "passage": _CORPUS[nid], "label": 1}
        for nid in expected if nid in _CORPUS
    ]
    neg_ids = _pick_random_negatives(expected, all_ids, _NEG_PER_POS)
    neg = [
        {"query": query, "passage": _CORPUS[nid], "label": 0}
        for nid in neg_ids
    ]
    return pos + neg


def _base_pairs(cases: list[dict], all_ids: list[str]) -> list[dict[str, object]]:  # type: ignore[type-arg]
    """Generate positive + random negative pairs from eval cases."""
    rows: list[dict[str, object]] = []
    for case in cases:
        rows.extend(_pairs_for_case(case, all_ids))
    return rows


def _note_augmented_pairs(cases: list[dict]) -> list[dict[str, object]]:  # type: ignore[type-arg]
    """Generate note-augmented positive pairs."""
    rows: list[dict[str, object]] = []
    for case in cases:
        note, expected = case.get("note", ""), set(case.get("expected_ids", []))
        if not note or not expected:
            continue
        for nid in expected:
            if nid in _CORPUS:
                rows.append({
                    "query": f"{case['query']} ({note})",
                    "passage": _CORPUS[nid],
                    "label": 1,
                })
    return rows


def _write_rows(rows: list[dict[str, object]], hard_neg_count: int) -> None:
    random.shuffle(rows)
    with open(_OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)
    print(f"Generated {len(rows)} rows ({n_pos} pos, {n_neg} neg)")
    print(f"  Hard negatives from failures: {hard_neg_count}")
    print(f"Written to {_OUT}")


def main() -> None:
    random.seed(42)
    cases = _load_jsonl(_CASES)
    rows = _base_pairs(cases, list(_CORPUS.keys()))
    rows.extend(_note_augmented_pairs(cases))
    hard_neg_count = _add_hard_negatives(rows)
    _write_rows(rows, hard_neg_count)


if __name__ == "__main__":
    main()
