from __future__ import annotations

import os
import time

import numpy as np
import pytest

from pipeline.rag.embeddings.dummy_provider import DummyProvider
from pipeline.rag.models import Chunk, Document
from pipeline.rag.storage.zvec_store import ZvecStore

pytestmark = pytest.mark.zvec


def _doc(doc_id: str = "d1", tags=("t1",)) -> Document:
    return Document(
        doc_id=doc_id,
        source_path=f"{doc_id}.md",
        file_name=f"{doc_id}.md",
        file_hash="h1",
        body="",
        title=f"Doc {doc_id}",
        language="pt",
        doc_type="manual",
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
            parents=((1, "Top"),)
            if i == 0
            else ((1, "Top"), (2, f"Sec {i}")),
        )
        for i, t in enumerate(texts)
    ]


def _ingest(store, doc, chunks, dummy: DummyProvider):
    res = dummy.embed([c.chunk_text for c in chunks])
    return res


def test_create_and_reopen(tmp_path, monkeypatch):
    p = tmp_path / "col"
    store = ZvecStore.open_or_create(p, dimension=64, enable_sparse=False)
    assert store.dimension == 64

    # zvec mantem lock RW no processo; reabrir no mesmo processo falha por lock.
    # Validamos apenas que `open_or_create` tenta `zvec.open` (caminho de reopen)
    # quando o path ja existe, sem disparar create_and_open.
    import pipeline.rag.storage.zvec_store as mod

    calls = {"open": 0, "create": 0}

    def fake_open(zvec, path, option=None):
        calls["open"] += 1
        return store.collection  # reusa a colecao ja aberta

    monkeypatch.setattr(mod, "_open_collection", fake_open)
    store2 = ZvecStore.open_or_create(p, dimension=64, enable_sparse=False, read_only=True)
    assert calls["open"] == 1
    assert store2.dimension == 64


def test_upsert_and_search_dense_returns_right_chunk(tmp_path):
    dummy = DummyProvider(dimension=64, enable_sparse=False)
    store = ZvecStore.open_or_create(tmp_path / "c1", dimension=64, enable_sparse=False)
    doc = _doc()
    chunks = _chunks("d1", ["contrato de servicos prestados", "receita de bolo de chocolate"])
    res = dummy.embed([c.chunk_text for c in chunks])
    n = store.upsert_chunks(chunks, doc, res.dense, sparse=None)
    assert n == 2
    store.optimize()

    qres = dummy.embed_query_dense("contrato de servico")
    hits = store.search_dense(qres, top_k=2)
    assert hits
    assert hits[0].doc_id == "d1"
    # top hit deve ser o chunk de contrato (similaridade dummy baseada em hashing compartilhado)
    assert "contrato" in hits[0].snippet.lower()


def test_search_fts_by_keyword(tmp_path):
    dummy = DummyProvider(dimension=64, enable_sparse=False)
    store = ZvecStore.open_or_create(tmp_path / "c2", dimension=64, enable_sparse=False)
    doc = _doc("d2")
    chunks = _chunks("d2", ["maquina de vetores suportando busca semantica", "outra secao irrelevante"])
    res = dummy.embed([c.chunk_text for c in chunks])
    store.upsert_chunks(chunks, doc, res.dense)
    store.optimize()
    hits = store.search_fts("vetores", top_k=5)
    assert hits
    assert any("vetores" in h.snippet.lower() for h in hits)


def test_search_hybrid(tmp_path):
    dummy = DummyProvider(dimension=64, enable_sparse=False)
    store = ZvecStore.open_or_create(tmp_path / "c3", dimension=64, enable_sparse=False)
    doc = _doc("d3")
    chunks = _chunks("d3", ["recuperacao aumentada por geracao", "totalmente fora do topico"])
    res = dummy.embed([c.chunk_text for c in chunks])
    store.upsert_chunks(chunks, doc, res.dense)
    store.optimize()
    qvec = dummy.embed_query_dense("recuperacao aumentada")
    hits = store.search_hybrid(qvec, "recuperacao aumentada geracao", top_k=5)
    assert hits
    assert hits[0].doc_id == "d3"


def test_delete_by_doc(tmp_path):
    dummy = DummyProvider(dimension=64, enable_sparse=False)
    store = ZvecStore.open_or_create(tmp_path / "c4", dimension=64, enable_sparse=False)
    doc = _doc("d4")
    _chunks("d4", ["a", "b"])
    chunks = _chunks("d4", ["primeiro chunk", "segundo chunk tambem"])
    res = dummy.embed([c.chunk_text for c in chunks])
    store.upsert_chunks(chunks, doc, res.dense)
    store.optimize()
    store.delete_by_doc("d4")
    # apos deletar, busca densa nao deve trazer d4
    hits = store.search_dense(dummy.embed_query_dense("primeiro chunk"), top_k=5)
    assert all(h.doc_id != "d4" for h in hits) or not hits


def test_dimension_mismatch_raises(tmp_path):
    dummy = DummyProvider(dimension=32, enable_sparse=False)  # diferente do schema (64)
    store = ZvecStore.open_or_create(tmp_path / "c5", dimension=64, enable_sparse=False)
    chunks = _chunks("d5", ["um texto"])
    res = dummy.embed([c.chunk_text for c in chunks])  # dense 32d
    with pytest.raises(ValueError):
        store.upsert_chunks(chunks, _doc("d5"), res.dense)


def test_fetch_by_id(tmp_path):
    dummy = DummyProvider(dimension=64, enable_sparse=False)
    store = ZvecStore.open_or_create(tmp_path / "c6", dimension=64, enable_sparse=False)
    chunks = _chunks("d6", ["hello world"])
    res = dummy.embed([c.chunk_text for c in chunks])
    store.upsert_chunks(chunks, _doc("d6"), res.dense)
    store.optimize()
    got = store.fetch(["d6_c0"])
    assert any(g.chunk_id == "d6_c0" for g in got)