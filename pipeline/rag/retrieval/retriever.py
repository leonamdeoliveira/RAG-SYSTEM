"""Camada 5 — Retrieval.

Orquestra:
  - embedding da query (provider)
  - busca no Zvec (dense / fts / hybrid / sparse)
  - aplicacao de filtros escalares (SQL-like) + post-filtro de tags (memoria)
  - threshold de score
  - diversidade max_per_doc (evita saturacao por um documento)
  - limit max_context_chunks
  - reranking feito pela IA do chat (fora desse pipeline)

Config default:
  top_k=20, score_threshold=0.0, max_context_chunks=8, max_per_doc=3,
  mode="hybrid" (dense + FTS).

`retrieve()` retorna `Evidence[]` pronto para a camada de geracao.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pipeline.rag.embeddings.base import EmbeddingProvider, SparseVector
from pipeline.rag.models import Evidence
from pipeline.rag.retrieval.filters import FilterBuilder, post_filter_tags
from pipeline.rag.storage.zvec_store import ZvecStore
from pipeline.rag.utils.logging import get_logger

log = get_logger("app.retriever")

DEFAULT_TOP_K = 20
DEFAULT_SCORE_THRESHOLD = 0.0
DEFAULT_MAX_CONTEXT_CHUNKS = 8
DEFAULT_MAX_PER_DOC = 3
DEFAULT_QUERY_CACHE_SIZE = 128


@dataclass
class RetrievalConfig:
    top_k: int = DEFAULT_TOP_K
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    max_context_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS
    max_per_doc: Optional[int] = DEFAULT_MAX_PER_DOC  # None = sem diversidade
    mode: str = "hybrid"  # dense | fts | hybrid | sparse
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
    ) -> None:
        self.store = store
        self.provider = provider
        self.config = config or RetrievalConfig()
        # LRU cache usando OrderedDict
        self._query_cache: OrderedDict[str, "np.ndarray"] = OrderedDict()
        self._query_cache_max = self.config.query_cache_size
        self._query_cache_hits = 0
        self._query_cache_misses = 0

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
        filter_sql = filters.build() if filters else None

        hits = self._search(query, k, filter_sql)

        # post-filtro de tags (ARRAY_STRING nao filtravel no Zvec v0.5)
        tags = filters.tags_filter if filters else None
        if tags:
            hits = [e for e in hits if post_filter_tags(e, tags)]

        # threshold (conservador; default 0.0)
        hits = [e for e in hits if e.score >= cfg.score_threshold]

        # diversidade por documento
        if cfg.max_per_doc:
            hits = self._diversify(hits, cfg.max_per_doc)

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

    def invalidate_cache(self) -> None:
        """Invalida o cache de queries. Use após reindex para garantir resultados atualizados."""
        cleared = len(self._query_cache)
        self._query_cache.clear()
        self._query_cache_hits = 0
        self._query_cache_misses = 0
        if cleared > 0:
            log.info("Query cache invalidado: %d entries removidas", cleared)
    
    def cache_stats(self) -> dict:
        """Retorna estatísticas do query cache."""
        total = self._query_cache_hits + self._query_cache_misses
        hit_rate = (self._query_cache_hits / total * 100) if total > 0 else 0.0
        usage = (len(self._query_cache) / self._query_cache_max * 100) if self._query_cache_max > 0 else 0.0
        
        return {
            "cache_size": len(self._query_cache),
            "cache_max": self._query_cache_max,
            "cache_hits": self._query_cache_hits,
            "cache_misses": self._query_cache_misses,
            "total_requests": total,
            "hit_rate_pct": round(hit_rate, 2),
            "cache_usage_pct": round(usage, 1),
        }
    
    def print_cache_stats(self) -> None:
        """Imprime estatísticas do cache."""
        s = self.cache_stats()
        log.info("=== Query Cache Stats ===")
        log.info("  Hit rate:  %s%% (%d hits / %d requests)", 
                 s['hit_rate_pct'], s['cache_hits'], s['total_requests'])
        log.info("  Entries:   %d/%d (%s%% full)", 
                 s['cache_size'], s['cache_max'], s['cache_usage_pct'])

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

    def _embed_query_dense_cached(self, query: str) -> "np.ndarray":
        """Embedda query com cache LRU (OrderedDict).
        
        Cache é invalidado automaticamente quando o retriever é recriado.
        """
        if query in self._query_cache:
            self._query_cache_hits += 1
            # LRU: move para o final (marca como recém-usado)
            self._query_cache.move_to_end(query)
            return self._query_cache[query]
        
        self._query_cache_misses += 1
        vec = self.provider.embed_query_dense(query)
        
        if len(self._query_cache) >= self._query_cache_max:
            # Remove o PRIMEIRO (menos recentemente usado)
            self._query_cache.popitem(last=False)
        
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
        """Busca usando sparse embeddings (lexical matching via BGE-M3)."""
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
