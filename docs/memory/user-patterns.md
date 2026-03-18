# User Patterns

Updated: 2026-03-18

## Patterns

Record recurring decisions here once a pattern is seen ≥2 times.
Format: `N. [context] → [decision] (reason)`

1. When two approaches are both correct, pick the one with fewer moving parts — simpler structure is preferred over generality.
2. Except unit tests, never use mock-based validation — inject Protocol stubs or closures; integration checks run against real dependency paths.
3. For type errors, resolve the underlying issue rather than adding `# type: ignore` or `cast` as a first move.
4. Prefer walrus operator `:=` over the `for x in [expr]` let-binding hack in comprehensions.
5. Independent async I/O operations should be parallelized with `asyncio.gather`, not awaited sequentially.
