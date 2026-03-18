# Long-Term Memory

Updated: 2026-03-18 00:00

## Keep Here

Only durable rules that should survive across tasks.
Prefer short statements with a clear rule or remediation.

---

## Active Rules

- `core/` must never catch exceptions from injected adapter functions. Adapters return `Result`; core processes results.
- All domain objects use `frozen dataclass (frozen=True, slots=True)`. No plain classes, no inheritance.
- Error handling in `core/`: return `('err', ...)`, never `raise`.
- `ruff check --fix && basedpyright && pytest` must all pass before commit. No `# noqa` bypasses.
- McCabe complexity limit is 5. Split functions before exceeding it.
- Every abstraction used in fewer than 2 places must be inlined.
- Dead code is deleted immediately. No TODOs in committed code.
- `type: ignore` comments are a last resort — resolve the underlying type issue first.

## Architecture Defaults

- Prefer context engineering over prompt hacking.
- Keep port and adapter boundaries clean. Cross-layer dependencies go through Protocols only.
- New LLM behavior must not hardcode a single provider path.
- Prefer typed Results over exception-based control flow everywhere.
