from __future__ import annotations

import numpy as np

from pipeline.rag.embeddings import get_provider
from pipeline.rag.embeddings.base import validate_dimension
from pipeline.rag.embeddings.dummy_provider import DummyProvider


def test_dummy_dense_shape_and_norm():
    p = DummyProvider(dimension=64, enable_sparse=False)
    res = p.embed(["hello world", "foo bar baz"])
    assert res.dense.shape == (2, 64)
    validate_dimension(res.dense, 64)
    norms = np.linalg.norm(res.dense, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_dummy_deterministic():
    p = DummyProvider(dimension=32, enable_sparse=False)
    a = p.embed(["same text"]).dense
    b = p.embed(["same text"]).dense
    assert np.allclose(a, b)


def test_dummy_similar_texts_more_aligned_than_unrelated():
    p = DummyProvider(dimension=128, enable_sparse=False)
    base = p.embed_query_dense("contrato de servicos prestados")
    near = p.embed_query_dense("contrato de prestacao de servico")
    far = p.embed_query_dense("receita de bolo de chocolate")
    sim_near = float(np.dot(base, near))
    sim_far = float(np.dot(base, far))
    # textos que compartilham palavras-chave devem ter similaridade >= unrelated
    assert sim_near >= sim_far


def test_dummy_sparse_has_indices():
    p = DummyProvider(dimension=64, enable_sparse=True)
    res = p.embed([" Retrieval augmented generation "])
    assert res.sparse is not None
    assert len(res.sparse) == 1
    assert len(res.sparse[0].values) > 0
    assert all(isinstance(k, int) for k in res.sparse[0].values)


def test_factory_returns_dummy():
    p = get_provider("dummy", dimension=16, enable_sparse=False)
    assert p.name == "dummy-hash"
    assert p.dimension == 16


def test_empty_input_returns_zero_rows():
    p = DummyProvider(dimension=8)
    res = p.embed([])
    assert res.dense.shape == (0, 8)