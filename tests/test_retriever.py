from __future__ import annotations

import time

import numpy as np
import pytest

from pipeline.rag.embeddings.dummy_provider import DummyProvider
from pipeline.rag.models import Chunk, Document
from pipeline.rag.retrieval.filters import FilterBuilder
from pipeline.rag.retrieval.retriever import RetrievalConfig, Retriever
from pipeline.rag.storage.zvec_store import ZvecStore

pytestmark = pytest.mark.zvec


def _doc(doc_id: str = "d1", tags=("t1",), language="pt", doc_type="manual") -> Document:
    return Document(
        doc_id=doc_id,
        source_path=f"{doc_id}.md",
        file_name=f"{doc_id}.md",
        file_hash="h",
        body="",
        title=f"Doc {doc_id}",
        language=language,
        doc_type=doc_type,
        tags=tags,
        ingested_at=int(time.time() * 1000),
    )


def _chunks(doc_id: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc_id}_c{i}",
            doc_id=doc_id,
            chunk_index=i,
            chunk_text=t,
            char_start=0,
            char_end=len(t),
            token_count=len(t.split()),
            char_count=len(t),
            parents=((1, "Top"), (2, f"Sec {i}")) if i > 0 else ((1, "Top"),),
        )
        for i, t in enumerate(texts)
    ]


def _setup(tmp_path, dim=64, docs=None):
    dummy = DummyProvider(dimension=dim, enable_sparse=False)
    store = ZvecStore.open_or_create(tmp_path / "col", dimension=dim, enable_sparse=False)
    for doc, texts in (docs or []):
        chunks = _chunks(doc.doc_id, texts)
        res = dummy.embed([c.chunk_text for c in chunks])
        store.upsert_chunks(chunks, doc, res.dense)
    store.optimize()
    return store, dummy


def test_retrieve_dense(tmp_path):
    store, dummy = _setup(
        tmp_path,
        docs=[
            (_doc("d1"), ["contrato de servicos prestados", "receita de bolo"]),
            (_doc("d2", tags=("t2",)), ["outro contrato de servico"]),
        ],
    )
    cfg = RetrievalConfig(mode="dense", top_k=5, max_context_chunks=5, max_per_doc=None)
    r = Retriever(store, dummy, cfg)
    hits = r.retrieve("contrato de servico")
    assert hits
    assert all(h.doc_id in ("d1", "d2") for h in hits)


def test_retrieve_fts(tmp_path):
    store, dummy = _setup(
        tmp_path,
        docs=[(_doc("d1"), ["recuperacao aumentada por geracao", "irrelevante"])],
    )
    cfg = RetrievalConfig(mode="fts", top_k=5, max_context_chunks=5, max_per_doc=None)
    r = Retriever(store, dummy, cfg)
    hits = r.retrieve("recuperacao")
    assert hits
    assert any("recuperacao" in h.snippet.lower() for h in hits)


def test_retrieve_hybrid_default(tmp_path):
    store, dummy = _setup(
        tmp_path,
        docs=[(_doc("d1"), ["recuperacao aumentada por geracao", "irrelevante"])],
    )
    r = Retriever(store, dummy, RetrievalConfig(max_context_chunks=5, max_per_doc=None))
    hits = r.retrieve("recuperacao aumentada")
    assert hits
    assert hits[0].doc_id == "d1"


def test_filter_by_doc_id(tmp_path):
    store, dummy = _setup(
        tmp_path,
        docs=[
            (_doc("d1"), ["contrato de servicos"]),
            (_doc("d2"), ["contrato de outro servico"]),
        ],
    )
    r = Retriever(store, dummy, RetrievalConfig(mode="dense", max_context_chunks=10, max_per_doc=None))
    f = FilterBuilder().eq("doc_id", "d2")
    hits = r.retrieve("contrato", filters=f)
    assert hits
    assert all(h.doc_id == "d2" for h in hits)


def test_filter_by_language(tmp_path):
    store, dummy = _setup(
        tmp_path,
        docs=[
            (_doc("d1", language="pt"), ["contrato de servicos"]),
            (_doc("d2", language="en"), ["service contract"]),
        ],
    )
    r = Retriever(store, dummy, RetrievalConfig(mode="dense", max_context_chunks=10, max_per_doc=None))
    f = FilterBuilder().eq("language", "en")
    hits = r.retrieve("contract", filters=f)
    assert hits
    assert all(h.doc_id == "d2" for h in hits)


def test_diversify_max_per_doc(tmp_path):
    # d1 tem 4 chunks, d2 tem 1; max_per_doc=2 deve limitar d1 a 2 hits
    store, dummy = _setup(
        tmp_path,
        docs=[
            (_doc("d1"), ["contrato a", "contrato b", "contrato c", "contrato d"]),
            (_doc("d2"), ["contrato e"]),
        ],
    )
    r = Retriever(
        store,
        dummy,
        RetrievalConfig(mode="dense", top_k=20, max_context_chunks=20, max_per_doc=2),
    )
    hits = r.retrieve("contrato")
    counts: dict[str, int] = {}
    for h in hits:
        counts[h.doc_id] = counts.get(h.doc_id, 0) + 1
    assert counts.get("d1", 0) <= 2


def test_max_context_chunks_limit(tmp_path):
    store, dummy = _setup(
        tmp_path,
        docs=[(_doc("d1"), [f"contrato numero {i}" for i in range(10)])],
    )
    r = Retriever(
        store,
        dummy,
        RetrievalConfig(mode="dense", top_k=20, max_context_chunks=3, max_per_doc=None),
    )
    hits = r.retrieve("contrato")
    assert len(hits) <= 3


def test_retrieve_with_confidence(tmp_path):
    store, dummy = _setup(
        tmp_path,
        docs=[(_doc("d1"), ["contrato de servicos"])],
    )
    r = Retriever(store, dummy, RetrievalConfig(mode="dense", max_context_chunks=5, max_per_doc=None))
    hits, conf = r.retrieve_with_confidence("contrato")
    assert hits
    assert conf == hits[0].score


def test_tags_post_filter(tmp_path):
    # tags nao filtravel no Zvec v0.5 -> post-filtro em memoria
    store, dummy = _setup(
        tmp_path,
        docs=[
            (_doc("d1", tags=("importante",)), ["contrato de servicos"]),
            (_doc("d2", tags=("rascunho",)), ["contrato de outro servico"]),
        ],
    )
    r = Retriever(store, dummy, RetrievalConfig(mode="dense", max_context_chunks=10, max_per_doc=None))
    f = FilterBuilder().with_tags(["importante"])
    hits = r.retrieve("contrato", filters=f)
    # so d1 tem tag 'importante'
    assert hits
    assert all(h.doc_id == "d1" for h in hits)


def test_filter_builder_sql():
    f = FilterBuilder().eq("doc_id", "d1").eq("language", "pt").gte("chunk_index", 2)
    assert f.build() == "doc_id = 'd1' AND language = 'pt' AND chunk_index >= 2"


def test_filter_builder_in():
    f = FilterBuilder().in_("doc_id", ["d1", "d2"])
    assert f.build() == "doc_id IN ('d1', 'd2')"


def test_filter_builder_empty():
    assert FilterBuilder().build() is None
    assert FilterBuilder().is_empty()


def test_filter_builder_bool():
    f = FilterBuilder().eq("is_appendix", False)
    assert f.build() == "is_appendix = FALSE"