# OpenHas — Constraints

## Memory Loading

Always-load at conversation start:
1. `docs/memory/long-term.md`
2. `docs/guides/engineering-workflow.md`
3. Latest 3 from `docs/error-experience/summary/entries/` (date DESC)
4. Latest 3 from `docs/good-experience/summary/entries/` (date DESC)

On-demand — load only when the task needs it:

| Source | When |
|--------|------|
| Full error / good entry | Summary is not enough |
| `docs/memory/user-patterns.md` | Deciding between two approaches |
| `docs/plans/` | Non-trivial task in planning |

---

## Key References

[Engineering Workflow](docs/guides/engineering-workflow.md) · [Long-Term Memory](docs/memory/long-term.md) · [User Patterns](docs/memory/user-patterns.md)

---

## Behavior Rules

- **Self-correct:** On any user correction, codify a preventive rule before resuming.
- **Auto-continue:** Same decision class seen ≥2 times → proceed with an inline note. Ask when ambiguous, irreversible, or no prior match.
- **Minimal change:** Read the target file and neighboring patterns first, then implement the smallest correct change. Touch only relevant files.
- **Simpler wins:** When two approaches are both correct, pick the one with fewer moving parts.
- **Priority:** safety > correctness > maintainability > speed.

---

## Type System

- All domain objects use `frozen dataclass` (`frozen=True, slots=True`). No plain classes.
- No inheritance (Protocol for interface constraints only).
- Mutable state lives in `adapters/` only.
- Error handling: `Result[T, E] = tuple[Literal['ok'], T] | tuple[Literal['err'], E]`.

---

## Project Layout

```
src/
  domain_types/   ← pure dataclass + type alias, zero external imports
  core/           ← pure functions, imports domain_types/ only
  adapters/       ← IO wrappers (DB, HTTP, filesystem)
  entrypoints/    ← CLI / API routes, wires adapters → core
tests/
  core/           ← mirrors core/, pure function tests, no mocks needed
  adapters/       ← integration tests, mocks allowed
docs/
  memory/         ← long-term rules and user patterns
  guides/         ← engineering workflow and coding standards
  error-experience/  ← recorded mistakes and preventive rules
  good-experience/   ← recorded wins and validated approaches
  plans/          ← design docs for non-trivial tasks
```

- `core/` must not import `adapters/` or any side-effectful stdlib module (`os`, `io`, `socket`, `subprocess`).
- All functions in `core/`: inputs are dataclasses or primitives; output is `Result`. No exceptions.

---

## Dependency Injection

- No `dependency-injector` / `fastapi.Depends`. Dependencies passed explicitly through factory functions.
- Each Protocol contains exactly one `__call__`. No fat interfaces.

---

## Error Handling

- `core/` must not `raise` (only `assert` for unreachable paths).
- `adapters/` catches external exceptions at the boundary and converts to `Result` before passing to `core/`.

---

## Lint & Static Analysis

- After every code change: `ruff check --fix && basedpyright`. Must pass. No `# noqa`.
- McCabe complexity limit: 5. Exceed it → split the function.

---

## Module Exports

- All `__init__.py` use explicit re-export (`as` syntax triggers pyright re-export detection).
- Cross-module imports only through `__init__.py`. No `from src.core.auth import _internal_helper`.

---

## Code Comments

- No comments that restate the code (`# loop over nodes` style).
- Only "why" comments for non-obvious decisions.
- Type signatures are documentation. Verbose names over comments.

---

## Abstraction Discipline

- Every abstraction must earn its place: used in fewer than 2 places → inline it.
- Delete dead code immediately. No TODOs in committed code.
- When requirements change, redesign cleanly. No compatibility shims.
- Trust the type system and caller invariants. No unnecessary defensive code.

---

## Python Style

- No `*args`/`**kwargs` (except in `adapters/` wrapping external APIs). Explicit parameters over flexibility.
- String interpolation: f-strings only. No `.format()` or `%`.
- Collection operations: prefer comprehensions. More than one nesting level → extract a named function.
- No monkey-patching. Use Protocol stubs in tests, not `unittest.mock.patch`.

---

## Testing

- Use Hypothesis for property-based tests.
- Pass lambda/closure stubs directly in tests. No `mock.patch`.
- Each test function ≤ 5 lines.

---

## Delivery

- Lint + tests must pass before commit: `ruff check --fix && basedpyright && pytest`.
- Small, scoped commits. Warn before destructive operations.
- Prefer TDD for logic changes. Cover edge cases.
- Record mistakes in `docs/error-experience/` and wins in `docs/good-experience/`.
