"""Pipeline de ingestão: Markdown -> chunks -> embeddings -> Zvec + manifesto.

Orquestra as camadas 1-4 (ARCHITECTURE.md §2.1-2.4) com atualizacao incremental:
  1. MarkdownLoader.scan + Manifest.diff -> new/changed/removed
  2. Para new/changed: load -> Chunker.chunk -> provider.embed -> store.upsert_chunks
  3. Para removed: store.delete_by_doc
  4. store.optimize() ao final (decisao D8: nunca em query-path)
  5. Manifest.save (estado do corpus)

Decisao D6: usa `upsert` (idempotente); reexecutar nao falha.

Cache de embeddings: se embed_cache_dir é fornecido, chunks cujo texto não mudou
(mesmo chunk_id) são recuperados do cache SQLite, evitando re-embeddar documentos
inteiros quando apenas um parágrafo foi alterado.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.rag.chunker import Chunker
from pipeline.rag.embeddings.base import EmbeddingProvider, SparseVector
from pipeline.rag.manifest import Manifest, ManifestEntry
from pipeline.rag.markdown_loader import MarkdownLoader
from pipeline.rag.models import Chunk, Document
from pipeline.rag.storage.zvec_store import ZvecStore
from pipeline.rag.utils.logging import get_logger

log = get_logger("app.pipelines.ingest")


@dataclass
class IngestReport:
    scanned: int = 0
    new: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks_upserted: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: list[str] = field(default_factory=list)

    def has_changes(self) -> bool:
        return bool(self.new or self.changed or self.removed)

    def summary(self) -> str:
        cache_info = ""
        if self.cache_hits or self.cache_misses:
            cache_info = f" cache_hits={self.cache_hits} cache_misses={self.cache_misses}"
        return (
            f"scanned={self.scanned} new={self.new} changed={self.changed} "
            f"removed={self.removed} unchanged={self.unchanged} "
            f"chunks_upserted={self.chunks_upserted}{cache_info} errors={len(self.errors)}"
        )


class IngestPipeline:
    def __init__(
        self,
        loader: MarkdownLoader,
        chunker: Chunker,
        provider: EmbeddingProvider,
        store: ZvecStore,
        manifest: Manifest,
        batch_size: int = 32,
        optimize_at_end: bool = True,
        embed_cache_dir: Optional[Path] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.provider = provider
        self.store = store
        self.manifest = manifest
        self.batch_size = batch_size
        self.optimize_at_end = optimize_at_end

        self._embed_cache = None
        if embed_cache_dir is not None:
            from pipeline.rag.pipelines.embed_cache import EmbedCache
            mid = model_id or getattr(provider, "name", "unknown")
            self._embed_cache = EmbedCache(Path(embed_cache_dir), model_id=mid)
            log.info("embed cache ativo: %s (model=%s)", embed_cache_dir, mid)

    def _embed_with_cache(
        self, chunks: list[Chunk], report: IngestReport
    ) -> tuple[np.ndarray, Optional[list[Optional[SparseVector]]]]:
        """Embedda chunks usando cache quando disponivel.

        Retorna (dense_matrix, sparse_list).
        Cache hit: retorna embedding salvo sem chamar o modelo.
        Cache miss: chama provider.embed(), salva no cache, retorna resultado.
        """
        n = len(chunks)

        if self._embed_cache is None:
            emb = self.provider.embed([c.chunk_text for c in chunks], batch_size=self.batch_size)
            return emb.dense, emb.sparse

        cache = self._embed_cache
        cached = cache.get_many([c.chunk_id for c in chunks])

        dense_results: list[Optional[np.ndarray]] = [None] * n
        sparse_results: list[Optional[SparseVector]] = [None] * n
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, chunk in enumerate(chunks):
            if chunk.chunk_id in cached:
                dense_results[i], sparse_results[i] = cached[chunk.chunk_id]
                report.cache_hits += 1
            else:
                miss_indices.append(i)
                miss_texts.append(chunk.chunk_text)
                report.cache_misses += 1

        if miss_texts:
            emb = self.provider.embed(miss_texts, batch_size=self.batch_size)
            miss_sparse = emb.sparse if self.provider.supports_sparse else None

            to_cache: list[tuple[str, np.ndarray, Optional[SparseVector]]] = []
            for idx, orig_i in enumerate(miss_indices):
                dense_results[orig_i] = emb.dense[idx]
                if miss_sparse is not None:
                    sparse_results[orig_i] = miss_sparse[idx]
                to_cache.append((
                    chunks[orig_i].chunk_id,
                    emb.dense[idx],
                    sparse_results[orig_i],
                ))
            cache.set_many(to_cache)

        dense = np.stack([d for d in dense_results if d is not None], axis=0)
        sparse = sparse_results if any(s is not None for s in sparse_results) else None
        return dense, sparse

    def run(self) -> IngestReport:
        report = IngestReport()
        diff, hashes, doc_ids = self.loader.compute_diff()
        report.scanned = len(hashes)
        report.new = len(diff.new)
        report.changed = len(diff.changed)
        report.removed = len(diff.removed)
        report.unchanged = len(diff.unchanged)

        # 1) removidos primeiro (libera espaco)
        for doc_id in diff.removed:
            try:
                self.store.delete_by_doc(doc_id)
                self.manifest.remove(doc_id)
            except Exception as e:
                report.errors.append(f"remove {doc_id}: {e}")
                log.error("erro removendo doc %s: %s", doc_id, e)

        # 2) new + changed: ingerir
        to_ingest = sorted(set(diff.new) | set(diff.changed))
        for rel in to_ingest:
            abs_path = self.loader.root / rel
            try:
                doc = self.loader.load(abs_path, hashes[rel], doc_ids[rel])
                chunks = self.chunker.chunk(doc)
                if not chunks:
                    log.warning("doc sem chunks: %s", rel)
                    continue
                dense, sparse = self._embed_with_cache(chunks, report)
                if not self.provider.supports_sparse:
                    sparse = None
                n = self.store.upsert_chunks(chunks, doc, dense, sparse=sparse)
                report.chunks_upserted += n
                self.manifest.upsert(
                    ManifestEntry(
                        doc_id=doc.doc_id,
                        source_path=doc.source_path,
                        file_name=doc.file_name,
                        file_hash=doc.file_hash,
                        ingested_at=doc.ingested_at,
                        language=doc.language,
                        doc_type=doc.doc_type,
                        tags=doc.tags,
                    )
                )
                log.info("ingerido %s: %d chunks", rel, n)
            except Exception as e:
                report.errors.append(f"ingest {rel}: {e}")
                log.error("erro ingerindo %s: %s", rel, e)

        # 3) otimizar apenas se houve mudancas (evita custo vazio)
        if self.optimize_at_end and report.has_changes():
            try:
                self.store.optimize()
            except Exception as e:
                report.errors.append(f"optimize: {e}")
                log.error("erro optimize: %s", e)

        # 4) persistir manifesto
        self.manifest.save()

        # 5) fechar cache e store
        if self._embed_cache is not None:
            self._embed_cache.close()
        
        # Fechar store para liberar lock do Zvec
        if hasattr(self.store, 'collection') and hasattr(self.store.collection, 'close'):
            try:
                self.store.collection.close()
            except Exception:
                pass

        log.info("ingest completo: %s", report.summary())
        return report
