from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.rag.models import Evidence
from pipeline.rag.retrieval.reranker import CrossEncoderReranker


def _ev(chunk_id: str, snippet: str = "", score: float = 0.5) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        doc_id="d1",
        source_path="test.md",
        file_name="test.md",
        score=score,
        chunk_index=0,
        snippet=snippet or f"snippet {chunk_id}",
    )


class TestCrossEncoderReranker:
    def test_disabled_returns_original_order(self):
        reranker = CrossEncoderReranker(enabled=False)
        ev = [_ev("a", score=0.3), _ev("b", score=0.9), _ev("c", score=0.5)]
        result = reranker.rerank("query", ev)
        assert [e.chunk_id for e in result] == ["a", "b", "c"]

    def test_disabled_respects_top_n(self):
        reranker = CrossEncoderReranker(enabled=False)
        ev = [_ev("a"), _ev("b"), _ev("c")]
        result = reranker.rerank("query", ev, top_n=2)
        assert len(result) == 2
        assert [e.chunk_id for e in result] == ["a", "b"]

    def test_empty_evidence(self):
        reranker = CrossEncoderReranker(enabled=True)
        assert reranker.rerank("q", []) == []

    def test_noop_when_cross_encoder_import_fails(self):
        reranker = CrossEncoderReranker(model_name="any/model", enabled=True)
        with patch("builtins.__import__", side_effect=ImportError("mock")):
            reranker._ensure_model()
        assert not reranker._enabled

    def test_noop_when_cross_encoder_load_fails(self):
        reranker = CrossEncoderReranker(model_name="any/model", enabled=True)
        reranker._model = None
        with patch(
            "sentence_transformers.CrossEncoder",
            side_effect=OSError("model not found"),
        ):
            reranker._ensure_model()
        assert not reranker._enabled

    def test_rerank_reorders_by_predict_scores(self):
        reranker = CrossEncoderReranker(enabled=True)
        fake = MagicMock()
        fake.predict.return_value = [0.1, 0.9, 0.5]
        reranker._model = fake
        reranker._enabled = True

        ev = [_ev("a", score=0.3), _ev("b", score=0.2), _ev("c", score=0.7)]
        result = reranker.rerank("query", ev)
        assert [e.chunk_id for e in result] == ["b", "c", "a"]

    def test_rerank_respects_top_n(self):
        reranker = CrossEncoderReranker(enabled=True)
        fake = MagicMock()
        fake.predict.return_value = [0.1, 0.9, 0.5]
        reranker._model = fake
        reranker._enabled = True

        ev = [_ev("a"), _ev("b"), _ev("c")]
        result = reranker.rerank("query", ev, top_n=2)
        assert [e.chunk_id for e in result] == ["b", "c"]

    def test_rerank_falls_back_on_predict_error(self):
        reranker = CrossEncoderReranker(enabled=True)
        fake = MagicMock()
        fake.predict.side_effect = RuntimeError("CUDA OOM")
        reranker._model = fake
        reranker._enabled = True

        ev = [_ev("a", score=0.3), _ev("b", score=0.9)]
        result = reranker.rerank("query", ev)
        assert [e.chunk_id for e in result] == ["a", "b"]

    def test_reranker_init_stores_params(self):
        r = CrossEncoderReranker(
            model_name="my/model", enabled=True, device="cuda", max_length=512
        )
        assert r._model_name == "my/model"
        assert r._enabled is True
        assert r._device == "cuda"
        assert r._max_length == 512


class TestRetrieverSearchK:
    """Garante que search_k >= rerank_top_n_candidates quando rerank ativo."""

    @staticmethod
    def _make_store(tmp_dir: Path):
        """Factory que evita lock de arquivo do Zvec no teardown."""
        from pipeline.rag.storage.zvec_store import ZvecStore
        col_dir = tmp_dir / "zvec_col"
        return ZvecStore.open_or_create(col_dir, dimension=64, enable_sparse=False)

    def test_search_k_expands_for_reranker(self):
        from pipeline.rag.embeddings.dummy_provider import DummyProvider
        from pipeline.rag.retrieval.retriever import RetrievalConfig, Retriever

        cfg = RetrievalConfig(
            top_k=10, rerank=True, rerank_top_n_candidates=20, max_context_chunks=6
        )
        reranker = CrossEncoderReranker(enabled=False)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = self._make_store(Path(tmp))
            provider = DummyProvider(dimension=64)
            retriever = Retriever(
                store=store, provider=provider, config=cfg, reranker=reranker
            )
            captured: dict = {}

            def spy(query, top_k, filter_sql):
                captured["k"] = top_k
                return []

            retriever._search = spy
            retriever.retrieve("test")
            assert captured["k"] == 20

    def test_search_k_uses_original_when_rerank_disabled(self):
        from pipeline.rag.embeddings.dummy_provider import DummyProvider
        from pipeline.rag.retrieval.retriever import RetrievalConfig, Retriever

        cfg = RetrievalConfig(
            top_k=10, rerank=False, rerank_top_n_candidates=20, max_context_chunks=6
        )
        reranker = CrossEncoderReranker(enabled=False)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = self._make_store(Path(tmp))
            provider = DummyProvider(dimension=64)
            retriever = Retriever(
                store=store, provider=provider, config=cfg, reranker=reranker
            )
            captured: dict = {}

            def spy(query, top_k, filter_sql):
                captured["k"] = top_k
                return []

            retriever._search = spy
            retriever.retrieve("test")
            assert captured["k"] == 10

    def test_search_k_uses_explicit_top_k_when_higher(self):
        from pipeline.rag.embeddings.dummy_provider import DummyProvider
        from pipeline.rag.retrieval.retriever import RetrievalConfig, Retriever

        cfg = RetrievalConfig(
            top_k=5, rerank=True, rerank_top_n_candidates=20, max_context_chunks=6
        )
        reranker = CrossEncoderReranker(enabled=False)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = self._make_store(Path(tmp))
            provider = DummyProvider(dimension=64)
            retriever = Retriever(
                store=store, provider=provider, config=cfg, reranker=reranker
            )
            captured: dict = {}

            def spy(query, top_k, filter_sql):
                captured["k"] = top_k
                return []

            retriever._search = spy
            retriever.retrieve("test", top_k=50)
            assert captured["k"] == 50
