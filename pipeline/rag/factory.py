"""Factory — wire-up de todas as camadas a partir de Settings.

Centraliza a construção de loader/chunker/provider/store/retriever/answerer/
pipelines para que os CLIs e testes não repitam boilerplate.
"""

from __future__ import annotations

from pipeline.rag.chunker import Chunker
from pipeline.rag.config import Settings, get_settings
from pipeline.rag.embeddings import get_provider
from pipeline.rag.generation.answerer import Answerer, OpenAICompatibleClient, StubLLMClient
from pipeline.rag.manifest import Manifest
from pipeline.rag.markdown_loader import MarkdownLoader
from pipeline.rag.pipelines.ingest import IngestPipeline
from pipeline.rag.pipelines.query import QueryConfig, QueryPipeline
from pipeline.rag.retrieval.retriever import RetrievalConfig, Retriever
from pipeline.rag.retrieval.reranker import CrossEncoderReranker
from pipeline.rag.storage.zvec_store import ZvecStore


def build_chunker(s: Settings) -> Chunker:
    return Chunker(
        target_tokens=s.chunker_target_tokens,
        min_tokens=s.chunker_min_tokens,
        max_tokens=s.chunker_max_tokens,
        overlap_ratio=s.chunker_overlap_ratio,
    )


def build_provider(s: Settings):
    return get_provider(
        "local",
        model_name=s.embedding_model,
        dimension=s.embedding_dimension,
        device=s.embedding_device,
        use_fp16=s.embedding_use_fp16,
        enable_sparse=s.embedding_enable_sparse,
        backend=s.embedding_backend,
        model_name_onnx=s.embedding_model_onnx,
        onnx_device=s.embedding_onnx_device,
    )


def build_store(s: Settings, read_only: bool = False) -> ZvecStore:
    return ZvecStore.open_or_create(
        s.zvec_path,
        dimension=s.embedding_dimension,
        enable_sparse=s.zvec_enable_sparse,
        read_only=read_only,
    )


def build_ingest_pipeline(s: Settings) -> IngestPipeline:
    s.ensure_dirs()
    manifest = Manifest(s.index_dir / "manifest.json")
    loader = MarkdownLoader(s.data_dir, manifest)
    provider = build_provider(s)
    store = build_store(s)

    embed_cache_dir = s.embed_cache_dir if s.embed_cache_enabled else None
    model_id = _get_model_id(s, provider)

    return IngestPipeline(
        loader=loader,
        chunker=build_chunker(s),
        provider=provider,
        store=store,
        manifest=manifest,
        batch_size=s.embedding_batch_size,
        embed_cache_dir=embed_cache_dir,
        model_id=model_id,
    )


def _get_model_id(s: Settings, provider) -> str:
    """Gera identificador unico do modelo para invalidacao de cache."""
    backend = s.embedding_backend
    model = s.embedding_model if backend == "torch" else s.embedding_model_onnx
    provider_name = getattr(provider, "name", "unknown")
    return f"{provider_name}:{model}:{backend}"


def build_query_pipeline(s: Settings):
    s.ensure_dirs()
    provider = build_provider(s)
    store = build_store(s, read_only=True)
    retriever = Retriever(
        store=store,
        provider=provider,
        config=RetrievalConfig(
            top_k=s.retrieval_top_k,
            score_threshold=s.retrieval_score_threshold,
            max_context_chunks=s.retrieval_max_context_chunks,
            max_per_doc=s.retrieval_max_per_doc,
            mode=s.retrieval_mode,
            rerank=s.retrieval_rerank,
            rerank_top_n_candidates=s.rerank_top_n_candidates,
        ),
        reranker=CrossEncoderReranker(
            model_name=s.rerank_model,
            enabled=s.rerank_enabled,
            device=s.rerank_device,
            max_length=s.rerank_max_length,
        ),
    )
    # LLM: se modelo='stub' usa StubLLMClient (offline); senao OpenAI-compatible
    if s.llm_model.lower() == "stub":
        llm = StubLLMClient()
    else:
        llm = OpenAICompatibleClient(
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
            model=s.llm_model,
            timeout=s.llm_timeout,
        )
    answerer = Answerer(
        llm=llm,
        default_mode=s.answer_mode,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
        low_confidence_threshold=s.reliability_low_confidence_threshold,
        max_chars_per_chunk=s.prompt_max_chars_per_chunk,
    )
    query = QueryPipeline(
        retriever=retriever,
        answerer=answerer,
        config=QueryConfig(
            low_confidence_threshold=s.reliability_low_confidence_threshold,
            expand_factor=s.reliability_expand_factor,
            max_retries=s.reliability_max_retries,
        ),
    )
    return query, retriever, store
