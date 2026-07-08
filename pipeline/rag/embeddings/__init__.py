from pipeline.rag.embeddings.base import (
    EmbeddingProvider,
    EmbeddingResult,
    SparseVector,
    l2_normalize,
    validate_dimension,
)


def get_provider(name: str, **kwargs) -> EmbeddingProvider:
    """Factory simples baseada em nome. Resolve o provider de config."""
    if name in ("bge-m3", "bge-m3-local", "local"):
        from pipeline.rag.embeddings.local_provider import BGEM3LocalProvider

        return BGEM3LocalProvider(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"provider de embedding desconhecido: {name}")


__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "SparseVector",
    "get_provider",
    "l2_normalize",
    "validate_dimension",
]
