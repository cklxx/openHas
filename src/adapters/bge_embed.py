"""BGE-M3 embedding adapter — in-process via sentence-transformers.

Produces 1024-dim dense embeddings, same as Qwen3-0.6B, so no schema change
needed. Uses fp16 on MPS/CUDA for lower memory footprint (~1.5GB).
"""

from sentence_transformers import SentenceTransformer


def make_bge_embed_fn(
    model_name: str = "BAAI/bge-m3",
    device: str = "mps",
):  # type: ignore[return]
    """Return an EmbedFn backed by BGE-M3 in-process.

    The model is loaded once; subsequent calls reuse it.
    """
    model = SentenceTransformer(model_name, device=device)

    async def embed(text: str) -> tuple[float, ...]:
        vec = model.encode(text, normalize_embeddings=True)  # pyright: ignore[reportUnknownMemberType]
        return tuple(float(x) for x in vec)

    return embed
