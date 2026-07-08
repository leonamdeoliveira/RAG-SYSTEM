"""Camada 5 — Retrieval.

Orquestra:
  - embedding da query (provider)
  - busca no Zvec (dense / fts / hybrid / sparse)
  - aplicacao de filtros escalares (SQL-like) + post-filtro de tags (memoria)
  - threshold de score
  - diversidade max_per_doc (evita saturacao por um documento)
  - reranker leve opcional (cross-encoder)
  - limit max_context_chunks

Config default:
  top_k=20, score_threshold=0.0, max_context_chunks=8, max_per_doc=3,
  mode="hybrid" (dense + FTS).

`retrieve()` retorna `Evidence[]` pronto para a camada de geracao.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pipeline.rag.embeddings.base import EmbeddingProvider, SparseVector
from pipeline.rag.models import Evidence
from pipeline.rag.retrieval.filters import FilterBuilder, post_filter_tags
from pipeline.rag.retrieval.reranker import CrossEncoderReranker
from pipeline.rag.storage.zvec_store import ZvecStore
from pipeline.rag.utils.logging import get_logger

log = get_logger("app.retriever")

DEFAULT_TOP_K = 20
DEFAULT_SCORE_THRESHOLD = 0.0
DEFAULT_MAX_CONTEXT_CHUNKS = 8
DEFAULT_MAX_PER_DOC = 3
DEFAULT_RERANK_TOP_N_CANDIDATES = 20
DEFAULT_QUERY_CACHE_SIZE = 128


@dataclass
class RetrievalConfig:
    top_k: int = DEFAULT_TOP_K
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    max_context_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS
    max_per_doc: Optional[int] = DEFAULT_MAX_PER_DOC  # None = sem diversidade
    mode: str = "hybrid"  # dense | fts | hybrid | sparse
    rerank: bool = True
    rerank_top_n_candidates: int = DEFAULT_RERANK_TOP_N_CANDIDATES
    query_cache_size: int = DEFAULT_QUERY_CACHE_SIZE

    def __post_init__(self) -> None:
        if self.mode not in {"dense", "fts", "hybrid", "sparse", "semantic"}:
            raise ValueError(f"modo de retrieval invalido: {self.mode}")


class Retriever:
    def __init__(
        self,
        store: ZvecStore,
        provider: EmbeddingProvider,
        config: Optional[RetrievalConfig] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.config = config or RetrievalConfig()
        self.reranker = reranker or CrossEncoderReranker(enabled=self.config.rerank)
        self._query_cache: dict[str, "np.ndarray"] = {}
        self._query_cache_max = self.config.query_cache_size
        self._query_cache_hits = 0

    # -------------------------------------------------------- public

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterBuilder] = None,
        top_k: Optional[int] = None,
    ) -> list[Evidence]:
        """Retorna Evidence[] ordenado por relevancia, apos pos-filtros."""
        cfg = self.config
        k = top_k or cfg.top_k
        # com reranker ativo, busca mais candidatos para dar massa ao cross-encoder
        search_k = max(k, cfg.rerank_top_n_candidates) if cfg.rerank else k
        filter_sql = filters.build() if filters else None

        hits = self._search(query, search_k, filter_sql)

        # post-filtro de tags (ARRAY_STRING nao filtravel no Zvec v0.5)
        tags = filters.tags_filter if filters else None
        if tags:
            hits = [e for e in hits if post_filter_tags(e, tags)]

        # threshold (conservador; default 0.0)
        hits = [e for e in hits if e.score >= cfg.score_threshold]

        # diversidade por documento
        if cfg.max_per_doc:
            hits = self._diversify(hits, cfg.max_per_doc)

        # rerank opcional
        if self.reranker and cfg.rerank:
            hits = self.reranker.rerank(query, hits, top_n=cfg.max_context_chunks)

        # limite final de contexto
        hits = hits[: cfg.max_context_chunks]
        log.info(
            "retrieve: query='%s' -> %d evidencias (mode=%s, k=%d)",
            query[:60],
            len(hits),
            cfg.mode,
            k,
        )
        return hits

    def retrieve_with_confidence(
        self,
        query: str,
        filters: Optional[FilterBuilder] = None,
        top_k: Optional[int] = None,
    ) -> tuple[list[Evidence], float]:
        """Retorna evidencias + score maximo (proxy de confianca)."""
        hits = self.retrieve(query, filters, top_k)
        confidence = max((e.score for e in hits), default=0.0)
        return hits, confidence

    # -------------------------------------------------------- interna

    def _embed_query_dense_cached(self, query: str) -> np.ndarray:
        """Embedda query com cache LRU em memoria (dict simples)."""
        if query in self._query_cache:
            self._query_cache_hits += 1
            return self._query_cache[query]
        vec = self.provider.embed_query_dense(query)
        if len(self._query_cache) >= self._query_cache_max:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[query] = vec
        return vec

    def _search(self, query: str, top_k: int, filter_sql: Optional[str]) -> list[Evidence]:
        mode = self.config.mode

        if mode == "dense":
            qv = self._embed_query_dense_cached(query)
            result = self.store.search_dense(qv, top_k=top_k, filter=filter_sql)
        elif mode == "fts":
            result = self.store.search_fts(match_string=query, top_k=top_k, filter=filter_sql)
        elif mode == "sparse":
            result = self._search_sparse(query, top_k, filter_sql)
        elif mode == "semantic":
            res = self.provider.embed_query(query)
            qv = res.dense[0]
            sv = res.sparse[0] if (res.sparse and self.provider.supports_sparse) else None
            result = self.store.search_hybrid(
                qv, query, top_k=top_k, filter=filter_sql,
                use_sparse=True, sparse_vector=sv,
                use_rrf=True,
            )
        else:
            # hybrid (default): dense + FTS com RRF adaptativo
            t_emb = time.perf_counter()
            qv = self._embed_query_dense_cached(query)
            t_emb_done = time.perf_counter()
            rank_k = self._rank_constant_for(query)
            result = self.store.search_hybrid(
                qv, query, top_k=top_k, filter=filter_sql,
                use_rrf=True, rank_constant=rank_k,
            )
            t_search = time.perf_counter()
            log.info("TIMING search: embed=%.1fms  hybrid=%.1fms (rrf_k=%d)  search=%.0fms",
                     (t_emb_done - t_emb) * 1000, (t_search - t_emb_done) * 1000, rank_k, (t_search - t_emb) * 1000)

        return result

    def _search_sparse(self, query: str, top_k: int, filter_sql: Optional[str]) -> list[Evidence]:
        if not self.provider.supports_sparse:
            log.warning("modo=sparse mas provider sem suporte -> fallback FTS")
            return self.store.search_fts(match_string=query, top_k=top_k, filter=filter_sql)
        res = self.provider.embed_query(query)
        sv = res.sparse[0] if res.sparse else None
        if sv is None:
            return self.store.search_fts(match_string=query, top_k=top_k, filter=filter_sql)
        return self.store.search_sparse(sv, top_k=top_k, filter=filter_sql)

    @staticmethod
    def _diversify(hits: list[Evidence], max_per_doc: int) -> list[Evidence]:
        """Mantem top-N por doc_id, preservando ordem original por score."""
        seen: dict[str, int] = {}
        out: list[Evidence] = []
        for e in hits:
            n = seen.get(e.doc_id, 0)
            if n < max_per_doc:
                out.append(e)
                seen[e.doc_id] = n + 1
        return out

    @staticmethod
    def _rank_constant_for(query: str) -> int:
        """RRF rank_constant adaptativo baseado no tipo de query.
        
        Menor constante = ranks importam mais = FTS (termos exatos) pesa mais.
        Maior constante = ranks importam menos = media entre dense e FTS.
        
        Para queries factuais (digitos, nomes): k=5  (FTS domina).
        Para queries conceituais (o que eh):       k=60 (balanceado).
        Default:                                    k=20.
        """
        import re
        has_digits = bool(re.search(r"\d+", query))
        has_proper_noun = bool(re.search(r"[A-Z]{2,}|[A-Z][a-z]+\s[A-Z]", query))
        has_what_is = bool(re.search(r"o que (é|e|eh|significa)|what is|defina|conceito|qual o significado", query.lower()))
        words = len(query.split())
        if has_digits and has_proper_noun:
            return 5
        if has_digits or has_proper_noun:
            return 10
        if has_what_is:
            return 60
        if words <= 3:
            return 10
        if words > 10:
            return 40
        return 20
