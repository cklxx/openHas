# Engineering Workflow

Updated: 2026-03-18

Use this flow for every engineering task.

See also: [Long-Term Memory](../memory/long-term.md) · [User Patterns](../memory/user-patterns.md)

---

## Core Rules

- Priority order: safety > correctness > maintainability > speed.
- Trust caller and type invariants. Do not add defensive code without a real gap.
- Delete dead code. No deprecated branches, `_unused` names, or commented-out code.
- When requirements change, redesign cleanly. No compatibility shims.
- Follow local patterns before introducing a new one.
- Change only files that matter to the task.

---

## Default Flow

1. Read the target file(s) and neighboring patterns.
2. Implement the minimal correct change.
3. Run `ruff check --fix && basedpyright && pytest` — all must pass.
4. Commit.

---

## Planning

- **Simple task:** inspect target files and implement directly.
- **Non-trivial task:** create `docs/plans/YYYY-MM-DD-short-slug.md` and keep it updated.
- Ask for missing information only when it changes correctness.

Decision order:
1. Explicit rules and hard constraints.
2. Reversibility and safe ordering.
3. Missing information that affects correctness.
4. User preference within the rules.

---

## Implementation Rules

- Keep naming and file structure consistent with nearby code.
- Before writing any helper, check if an equivalent already exists in `src/core/` or `src/domain_types/`.
- If a helper is used in fewer than 2 places, inline it.
- Prefer comprehensions over explicit loops; extract named functions when nesting exceeds one level.
- Independent async I/O operations → `asyncio.gather`. Sequential awaits only when there is a real data dependency.

For `core/` work:
- Inputs are dataclasses or primitives. Output is `Result`. No side effects.
- Do not catch exceptions from injected functions. Adapter Protocols must return `Result` instead of raising.

For `adapters/` work:
- Catch external exceptions at the boundary; convert to `Result` before returning.
- Mutable state lives here. Nowhere else.

---

## Validation

- Prefer TDD for logic changes. Cover edge cases.
- Use closure/lambda stubs in unit tests. No `mock.patch`.
- Use Hypothesis for property-based tests on pure functions.
- Integration tests in `tests/adapters/` run against real dependencies.

Full gate before commit:

```bash
ruff check --fix && basedpyright && pytest
```

---

## Commit Rules

- Small, scoped commits. One logical change per commit.
- Commit only your own changes. Surface unrelated diffs rather than sweeping them in.
- Warn before destructive operations (force push, reset --hard, branch delete).

---

## Experience Records

| Type | Entry path | Summary path |
|------|------------|--------------|
| Error | `docs/error-experience/entries/YYYY-MM-DD-slug.md` | `docs/error-experience/summary/entries/YYYY-MM-DD-slug.md` |
| Win | `docs/good-experience/entries/YYYY-MM-DD-slug.md` | `docs/good-experience/summary/entries/YYYY-MM-DD-slug.md` |

Every entry needs a `## Metadata` block:

```yaml
id: <error|good>-YYYY-MM-DD-<short-slug>
type: error_entry | good_entry | error_summary | good_summary
date: YYYY-MM-DD
tags:
  - <tag>
links:
  related: []
  supersedes: []
```

Index files (`docs/error-experience/summary.md`, `docs/good-experience/summary.md`) stay index-only — no narrative content.

---

## Codex Worker Protocol

Use EXPLORE → PLAN → EXECUTE → REVIEW. Keep worker prompts self-contained. Limit retries to 2 per task.
