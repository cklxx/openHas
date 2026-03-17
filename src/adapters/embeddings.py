"""Embedding adapter — wraps external embedding API at IO boundary."""

import hashlib

from src.domain_types.ports import EmbedFn


def make_embed_fn(dim: int = 128) -> EmbedFn:
    """Simple hash-based embedding for local dev. Replace with real API in prod."""

    async def embed(text: str) -> tuple[float, ...]:
        h = hashlib.sha256(text.encode()).digest()
        raw = [b / 255.0 for b in h]
        while len(raw) < dim:
            h = hashlib.sha256(h).digest()
            raw.extend(b / 255.0 for b in h)
        return tuple(raw[:dim])

    return embed  # type: ignore[return-value]
