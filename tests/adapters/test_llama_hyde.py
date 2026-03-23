"""Tests for llama_hyde adapter.

Integration tests require a running llama-server on the predict port (18081).
Skipped automatically when the server is not reachable.
"""

import httpx
import pytest
from src.adapters.llama_hyde import N_SNIPPETS, make_llama_hyde_fn

_PREDICT_URL = "http://localhost:18081"
_MIN_SNIPPET_LEN = 10


def _server_reachable(url: str) -> bool:
    try:
        httpx.get(f"{url}/health", timeout=1.0)
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
    return True


_skip_no_server = pytest.mark.skipif(
    not _server_reachable(_PREDICT_URL),
    reason="llama-server not running on 18081",
)


@_skip_no_server
@pytest.mark.asyncio
async def test_hyde_returns_snippets() -> None:
    fn = make_llama_hyde_fn(_PREDICT_URL)
    snippets = await fn("Could a loud construction site hurt the user's productivity?")
    assert len(snippets) > 0
    assert all(isinstance(s, str) and s for s in snippets)


@_skip_no_server
@pytest.mark.asyncio
async def test_hyde_returns_at_most_n_snippets() -> None:
    fn = make_llama_hyde_fn(_PREDICT_URL)
    snippets = await fn("What are the user's dietary restrictions?")
    assert len(snippets) <= N_SNIPPETS


@_skip_no_server
@pytest.mark.asyncio
async def test_hyde_snippets_are_statements_not_questions() -> None:
    """Snippets should be factual statements (stored memory format), not questions."""
    fn = make_llama_hyde_fn(_PREDICT_URL)
    snippets = await fn("When does the user exercise?")
    assert snippets
    for snippet in snippets:
        assert len(snippet) > _MIN_SNIPPET_LEN
        # Snippets are statements; allow '?' only if embedded mid-sentence
        assert not snippet.strip().endswith('?'), f"Snippet looks like a question: {snippet!r}"
