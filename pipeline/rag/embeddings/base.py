"""Camada de embeddings — interface pluggable.

A arquitetura define um `EmbeddingProvider` que deve produzir dense embeddings
(dense vectors L2-normalizados) e, opcionalmente, sparse embeddings (dict
`{idx: peso}`) para busca híbrida no Zvec.

Implementacoes:
  - local_provider.py  -> BGE-M3 (dense 1024d + sparse) via ONNX ou FlagEmbedding,
                         fallback sentence-transformers (dense-only) com aviso.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class SparseVector:
    """Vetor esparso no formato que o Zvec aceita: dict {idx: peso}."""

    values: dict[int, float]


@dataclass(frozen=True)
class EmbeddingResult:
    """Resultado de uma chamada `embed` em lote."""

    dense: np.ndarray  # shape (n, dim), float32, L2-normalizado
    sparse: Optional[list[SparseVector]] = None  # None quando provider só tem dense


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface de provider de embeddings."""

    name: str
    dimension: int
    supports_sparse: bool

    def embed(self, texts: list[str], batch_size: int = 32) -> EmbeddingResult:
        """Embaralha `texts` em batches e retorna dense (+ sparse opcional)."""
        ...

    def embed_query(self, text: str) -> EmbeddingResult:
        """Atalho para embed de uma unica string (query)."""
        ...

    def embed_query_dense(self, text: str) -> np.ndarray:
        """Retorna so o dense da query como vetor 1D (n, ) ou (dim,)."""
        ...


def validate_dimension(vec: np.ndarray, expected: int) -> None:
    """Garante que o vetor esta na dimensao esperada pelo schema do Zvec."""
    if vec.shape[-1] != expected:
        raise ValueError(
            f"dimensao de embedding incompativel: esperado {expected}, obtido {vec.shape[-1]}. "
            "Ajuste a config `embedding.dimension` ou troque o provider."
        )


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normaliza linhas de uma matriz para norma L2 = 1 (cosine ≡ IP)."""
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, eps)
