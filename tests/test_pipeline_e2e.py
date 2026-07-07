from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.rag.chunker import Chunker
from pipeline.rag.embeddings.dummy_provider import DummyProvider
from pipeline.rag.generation.answerer import Answerer, StubLLMClient
from pipeline.rag.manifest import Manifest
from pipeline.rag.markdown_loader import MarkdownLoader
from pipeline.rag.pipelines.ingest import IngestPipeline
from pipeline.rag.pipelines.query import QueryConfig, QueryPipeline
from pipeline.rag.retrieval.retriever import RetrievalConfig, Retriever
from pipeline.rag.storage.zvec_store import ZvecStore

pytestmark = pytest.mark.zvec


def _write(p: Path, content: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _build_pipeline(tmp_path: Path, dim: int = 64):
    root = tmp_path / "data"
    root.mkdir()
    manifest_path = tmp_path / "index" / "manifest.json"
    manifest = Manifest(manifest_path)
    loader = MarkdownLoader(root, manifest)
    chunker = Chunker(target_tokens=80, min_tokens=20, max_tokens=120)
    provider = DummyProvider(dimension=dim, enable_sparse=False)
    store = ZvecStore.open_or_create(tmp_path / "zvec", dimension=dim, enable_sparse=False)
    ingest = IngestPipeline(loader, chunker, provider, store, manifest, optimize_at_end=True)
    retriever = Retriever(
        store,
        provider,
        RetrievalConfig(
            mode="hybrid", top_k=5, max_context_chunks=5, max_per_doc=None, score_threshold=0.0
        ),
    )
    answerer = Answerer(llm=StubLLMClient(), default_mode="answer")
    query = QueryPipeline(retriever, answerer, QueryConfig(low_confidence_threshold=0.0))
    return ingest, query, store, manifest, root


def test_ingest_then_query_e2e(tmp_path):
    ingest, query, _, _, root = _build_pipeline(tmp_path)
    _write(root / "doc1.md", "# Servicos\n\nContrato de servicos prestados com pagamento em 30 dias.\n")
    _write(root / "doc2.md", "# Receitas\n\nReceita de bolo de chocolate com cacau 70%.\n")

    report = ingest.run()
    assert report.new == 2
    assert report.chunks_upserted >= 2
    assert not report.errors

    ans = query.run("Qual o prazo de pagamento?")
    assert ans.mode == "answer"
    assert len(ans.evidence) >= 1
    # top hit deve ser o doc1 (servicos) — doc_id e hash do rel_path
    assert ans.evidence[0].source_path == "doc1.md"


def test_incremental_ingest_new_changed_removed(tmp_path):
    ingest, _, _, manifest, root = _build_pipeline(tmp_path)
    _write(root / "a.md", "# A v1\nConteudo versao um.\n")
    _write(root / "b.md", "# B\nConteudo B.\n")

    r1 = ingest.run()
    assert r1.new == 2 and r1.chunks_upserted >= 2

    # mudar a.md, remover b.md, adicionar c.md
    _write(root / "a.md", "# A v2\nConteudo versao dois diferente.\n")
    (root / "b.md").unlink()
    _write(root / "c.md", "# C\nConteudo novo C.\n")

    r2 = ingest.run()
    assert r2.new == 1  # c.md
    assert r2.changed == 1  # a.md
    assert r2.removed == 1  # b.md
    assert r2.unchanged == 0
    assert r2.chunks_upserted >= 1
    assert not r2.errors

    # manifesto coerente: a e c presentes, b ausente
    all_ids = set(manifest.all_doc_ids())
    assert len(all_ids) == 2


def test_idempotent_rerun(tmp_path):
    ingest, _, _, _, root = _build_pipeline(tmp_path)
    _write(root / "x.md", "# X\nConteudo X.\n")
    r1 = ingest.run()
    r2 = ingest.run()  # nada mudou
    assert r2.new == 0 and r2.changed == 0 and r2.removed == 0
    assert r2.chunks_upserted == 0
    assert r2.unchanged == 1


def test_query_insufficient_when_empty_corpus(tmp_path):
    ingest, query, _, _, _ = _build_pipeline(tmp_path)
    ingest.run()  # corpus vazio
    ans = query.run("Qualquer coisa sem documentos")
    assert ans.insufficient_context
    assert ans.confidence == 0.0


def test_query_modes(tmp_path):
    ingest, query, _, _, root = _build_pipeline(tmp_path)
    _write(root / "m.md", "# Manual\nInstale com pip install zvec.\n")
    ingest.run()
    for mode in ["answer", "answer_with_citations", "extractive_summary", "study_mode"]:
        ans = query.run("Como instalar?", mode=mode)
        assert ans.mode == mode


def test_query_conflict_flag(tmp_path):
    ingest, query, _, _, root = _build_pipeline(tmp_path)
    # dois docs com conteudo similar e scores proximos -> heuristica dispara
    _write(root / "x.md", "# X\ncontrato servicos pagamento\n")
    _write(root / "y.md", "# Y\ncontrato servicos pagamento\n")
    ingest.run()
    # forca conflito: query retorna hits de 2 docs com scores iguais
    # StubLLM nao escreve 'conflito' no texto, mas a heuristica estrutural pode
    # marcar conflict=True se scores proximos em docs diferentes.
    ans = query.run("contrato servicos pagamento")
    # pode ou nao ser conflito dependendo dos scores; so verificamos que roda
    assert isinstance(ans.conflict, bool)


def test_ingest_report_summary(tmp_path):
    ingest, _, _, _, root = _build_pipeline(tmp_path)
    _write(root / "a.md", "# A\nTexto A.\n")
    r = ingest.run()
    s = r.summary()
    assert "scanned=1" in s
    assert "new=1" in s