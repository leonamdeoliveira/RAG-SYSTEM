"""Provider dummy para testes offline (sem modelos ML).

Gera vetores densos deterministas a partir de um hash do texto, de forma que:
  - mesmos textos -> mesmos vetores (reprodutivel)
  - textos parecidos  -> vetores colineares (cao specialised embedding sgnl)
  - dimensao configuravel (default 64 para tests rapidos)
  - sparse opcional usando top-k termos raros via hash TF

Uso: testes unitarios/integracao que precisem da interface `EmbeddingProvider`
sem baixar BGE-M3 (~2GB). Nao deve ser usado em producao.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Optional

import numpy as np

from pipeline.rag.embeddings.base import EmbeddingResult, SparseVector, l2_normalize, validate_dimension
from pipeline.rag.utils.logging import get_logger

log = get_logger("app.embeddings.dummy")

DEFAULT_DIMENSION = 64
_WORD = re.compile(r"\w+", flags=re.UNICODE)


def _hash_to_vec(text: str, dim: int) -> np.ndarray:
    """Vetor denso determinista: cada palavra contribui para buckets por hash."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _WORD.findall(text.lower())
    if not tokens:
        return vec.reshape(1, -1)
    for tok in tokens:
        # dois hashes por token -> banda + sinal
        h_band = hashlib.blake2b(tok.encode(), key=b"band", digest_size=4)
        h_sign = hashlib.blake2b(tok.encode(), key=b"sign", digest_size=4)
        band = int.from_bytes(h_band.digest(), "little") % dim
        sign = 1.0 if (h_sign.digest()[0] & 1) else -1.0
        vec[band] += sign
    return vec.reshape(1, -1)


def _hash_tf(text: str, dim_buckets: int = 4096) -> SparseVector:
    """Sparse 'term frequency' em buckets de hash (saida tipo BGE-M3 sparse)."""
    tokens = _WORD.findall(text.lower())
    counts = Counter(tokens)
    values: dict[int, float] = {}
    for tok, c in counts.items():
        h = hashlib.blake2b(tok.encode(), digest_size=4).digest()
        idx = int.from_bytes(h, "little") % dim_buckets
        values[idx] = values.get(idx, 0.0) + float(c)
    return SparseVector(values=values)


class DummyProvider:
    """EmbeddingProvider offline para testes. Dimensao default 64."""

    name: str = "dummy-hash"
    dimension: int
    supports_sparse: bool

    def __init__(
        self,
        dimension: int = DEFAULT_DIMENSION,
        enable_sparse: bool = True,
        sparse_dim: int = 4096,
    ) -> None:
        self.dimension = dimension
        self.supports_sparse = enable_sparse
        self._sparse_dim = sparse_dim

    def embed(self, texts: list[str], batch_size: int = 32) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(dense=np.zeros((0, self.dimension), dtype=np.float32))
        mats = np.vstack([_hash_to_vec(t, self.dimension) for t in texts])
        mats = l2_normalize(mats)
        validate_dimension(mats, self.dimension)
        sparse: Optional[list[SparseVector]] = None
        if self.supports_sparse:
            sparse = [_hash_tf(t, self._sparse_dim) for t in texts]
        return EmbeddingResult(dense=mats, sparse=sparse)

    def embed_query(self, text: str) -> EmbeddingResult:
        return self.embed([text])

    def embed_query_dense(self, text: str) -> np.ndarray:
        return self.embed([text]).dense[0]
