"""Cross-encoder reranker adapter — pointwise scoring.

Uses a fine-tuned cross-encoder model to score each (query, memory) pair
independently. Much more reliable than listwise LLM ranking on small models.

The model runs in-process — no external server needed.
"""

from sentence_transformers import CrossEncoder

from src.domain_types.memory import MemoryNode


def make_cross_encoder_rerank_fn(
    model_path: str = "BAAI/bge-reranker-v2-m3",
    device: str = "mps",
):  # type: ignore[return]
    """Return a RerankFn backed by a local cross-encoder model.

    Each (query, candidate.content) pair is scored independently (pointwise),
    then candidates are sorted by score descending.
    """
    model = CrossEncoder(model_path, device=device)

    async def rerank(query: str, nodes: list[MemoryNode]) -> list[tuple[str, float]]:
        if not nodes:
            return []
        pairs = [(query, n.content) for n in nodes]
        scores: list[float] = model.predict(pairs)  # type: ignore[assignment]
        ranked = sorted(
            zip(nodes, scores, strict=True), key=lambda x: float(x[1]), reverse=True
        )
        return [(n.id, float(s)) for n, s in ranked]

    return rerank
