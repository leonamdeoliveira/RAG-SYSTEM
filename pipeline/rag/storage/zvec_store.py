"""Camada de persistência vetorial sobre Zvec.

Wrapper fino sobre a API real do Zvec v0.5+ (cf. ARCHITECTURE.md §2.4):
  - build_schema() define o schema completo (campos escalares + FTS + dense + sparse)
  - open_or_create() abre collection persistida em disco (WAL)
  - upsert_chunks() mapeia Chunk+Document+embeddings -> zvec.Doc, usando upsert
    (idempotente para ingestão incremental)
  - search_dense / search_fts / search_hybrid / search_sparse expõem os modos
    de retrieval comuns; resultados normalizados em Evidence
  - delete_by_doc / delete_by_filter para atualização incremental
  - optimize() quando solicitado (nunca na hot-path de query)

Suposicoes marcadas (pontos unicos de ajuste caso a API Zvec varie):
  - zvec.open(path) para reabrir collection existente (a doc descreve "Open" page).
  - zvec.Fts(match_string=...) para FTS; sem `zvec.Fts`, fallback
    `from zvec.model.param.query import Fts`.
  - Resultados de query: objetos com attr `id`, `score`, `fields` (ou dict com mesmas
    chaves). Tratamento defensivo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from pipeline.rag.embeddings.base import SparseVector, validate_dimension
from pipeline.rag.models import Chunk, Document, Evidence
from pipeline.rag.utils.logging import get_logger

log = get_logger("app.storage")

COLLECTION_NAME = "rag_chunks"
DEFAULT_DENSE_INDEX = "hnsw"

# ---- helpers para import zvec (lazy, nao quebra imports offline) ----


def _require_zvec():
    try:
        import zvec

        return zvec
    except ImportError as e:
        raise RuntimeError(
            "zvec nao instalado. Rode: pip install zvec"
        ) from e


def _make_fts(query_text: str):
    """Constroi Fts(match_string=...). Pontos de suposição isolados."""
    zvec = _require_zvec()
    cls = getattr(zvec, "Fts", None)
    if cls is not None:
        return cls(match_string=query_text)
    try:
        from zvec.model.param.query import Fts

        return Fts(match_string=query_text)
    except ImportError as e:
        raise RuntimeError("FTS indisponivel na versao de zvec instalada") from e


def _make_query(field_name: str, **kw):
    zvec = _require_zvec()
    cls = getattr(zvec, "Query", None)
    if cls is None:
        from zvec.model.param.query import Query as cls  # type: ignore
    return cls(field_name=field_name, **kw)


# ---- schema ----


def build_schema(
    dimension: int = 1024,
    enable_sparse: bool = True,
) -> "Any":
    """Constroi o CollectionSchema completo (ver ARCHITECTURE.md §2.4 tabela)."""
    zvec = _require_zvec()
    DT = zvec.DataType

    def invert(**kw):
        return zvec.InvertIndexParam(**kw)

    def fts(**kw):
        return zvec.FtsIndexParam(**kw)

    fields = [
        zvec.FieldSchema(name="doc_id", data_type=DT.STRING, index_param=invert()),
        zvec.FieldSchema(name="chunk_id", data_type=DT.STRING, index_param=invert()),
        zvec.FieldSchema(name="source_path", data_type=DT.STRING),
        zvec.FieldSchema(name="file_name", data_type=DT.STRING),
        zvec.FieldSchema(name="title", data_type=DT.STRING, nullable=True),
        zvec.FieldSchema(name="h1", data_type=DT.STRING, nullable=True),
        zvec.FieldSchema(name="h2", data_type=DT.STRING, nullable=True),
        zvec.FieldSchema(name="h3", data_type=DT.STRING, nullable=True),
        zvec.FieldSchema(name="chunk_index", data_type=DT.INT32, index_param=invert()),
        zvec.FieldSchema(name="token_count", data_type=DT.INT32),
        zvec.FieldSchema(name="language", data_type=DT.STRING, index_param=invert()),
        zvec.FieldSchema(name="file_hash", data_type=DT.STRING),
        zvec.FieldSchema(name="doc_type", data_type=DT.STRING, nullable=True, index_param=invert()),
        zvec.FieldSchema(name="tags", data_type=DT.ARRAY_STRING),
        zvec.FieldSchema(name="is_appendix", data_type=DT.BOOL, nullable=True),
        zvec.FieldSchema(name="ingested_at", data_type=DT.INT64),
        zvec.FieldSchema(
            name="content",
            data_type=DT.STRING,
            index_param=fts(),
        ),
    ]

    vectors = [
        zvec.VectorSchema(
            name="dense_embedding",
            data_type=DT.VECTOR_FP32,
            dimension=dimension,
            index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
        )
    ]
    if enable_sparse:
        vectors.append(
            zvec.VectorSchema(
                name="sparse_embedding",
                data_type=DT.SPARSE_VECTOR_FP32,
            )
        )

    return zvec.CollectionSchema(name=COLLECTION_NAME, fields=fields, vectors=vectors)


# ---- store ----


class ZvecStore:
    """Front-end estável sobre a collection Zvec para o pipeline RAG."""

    def __init__(self, collection: Any, path: Union[str, Path], schema: Any) -> None:
        self.collection = collection
        self.path = str(path)
        self.schema = schema
        self.dimension = _schema_dense_dim(schema)

    # -------------------------------------------------------- fabрика

    @classmethod
    def open_or_create(
        cls,
        path: Union[str, Path],
        dimension: int = 1024,
        enable_sparse: bool = True,
        read_only: bool = False,
    ) -> "ZvecStore":
        zvec = _require_zvec()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = build_schema(dimension=dimension, enable_sparse=enable_sparse)

        option = zvec.CollectionOption(read_only=int(read_only))

        collection = None
        if path.exists() and any(path.iterdir()):
            for attempt in range(5):
                try:
                    collection = _open_collection(zvec, str(path), option=option)
                    log.info("aberta collection existente: %s (read_only=%s)", path, read_only)
                    break
                except Exception as e:
                    if "lock" in str(e).lower():
                        if attempt < 4:
                            import time
                            time.sleep(1.0 * (attempt + 1))
                            continue
                    raise RuntimeError(
                        f"Nao foi possivel abrir a collection em {path}. "
                        f"Outro processo pode estar usando. Aguarde alguns segundos e tente novamente. "
                        f"Erro: {e}"
                    ) from e

        if collection is None:
            if path.exists() and any(path.iterdir()):
                raise RuntimeError(
                    f"Nao foi possivel abrir a collection em {path}. "
                    "Outro processo pode estar usando. Aguarde alguns segundos e tente novamente."
                )
            collection = zvec.create_and_open(path=str(path), schema=schema, option=option)
            log.info("criada nova collection: %s (dim=%d, sparse=%s)", path, dimension, enable_sparse)

        return cls(collection=collection, path=path, schema=schema)

    # -------------------------------------------------------- inserção

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        doc: Document,
        dense: np.ndarray,
        sparse: Optional[Sequence[SparseVector]] = None,
    ) -> int:
        if not chunks:
            return 0
        if dense.shape[0] != len(chunks):
            raise ValueError(
                f"nº de embeddings ({dense.shape[0]}) != nº de chunks ({len(chunks)})"
            )
        validate_dimension(dense, self.dimension)

        zvec = _require_zvec()
        parents_map = _parents_to_h123

        docs = []
        for i, c in enumerate(chunks):
            h1, h2, h3 = parents_map(c.parents)
            vectors: dict[str, Any] = {"dense_embedding": dense[i].astype(np.float32).tolist()}
            if sparse is not None and sparse[i] is not None:
                vectors["sparse_embedding"] = sparse[i].values  # dict {idx: peso}

            fields = {
                "doc_id": doc.doc_id,
                "chunk_id": c.chunk_id,
                "source_path": doc.source_path,
                "file_name": doc.file_name,
                "title": doc.title,
                "h1": h1,
                "h2": h2,
                "h3": h3,
                "chunk_index": int(c.chunk_index),
                "token_count": int(c.token_count),
                "language": doc.language,
                "file_hash": doc.file_hash,
                "doc_type": doc.doc_type,
                "tags": list(doc.tags),
                "is_appendix": bool(c.is_appendix),
                "ingested_at": int(doc.ingested_at),
                "content": c.chunk_text,
            }
            docs.append(zvec.Doc(id=c.chunk_id, vectors=vectors, fields=fields))

        self.collection.upsert(docs)
        log.info("upsert de %d chunks (doc=%s)", len(docs), doc.doc_id)
        return len(docs)

    # -------------------------------------------------------- queries

    def search_dense(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        filter: Optional[str] = None,
    ) -> list[Evidence]:
        q = _make_query(field_name="dense_embedding", vector=np.asarray(vector).astype(np.float32).tolist())
        results = self.collection.query(queries=q, topk=top_k, filter=filter)
        return [self._to_evidence(r) for r in results]

    def search_sparse(
        self,
        sparse_vector: SparseVector,
        top_k: int = 10,
        filter: Optional[str] = None,
    ) -> list[Evidence]:
        q = _make_query(field_name="sparse_embedding", vector=sparse_vector.values)
        results = self.collection.query(queries=q, topk=top_k, filter=filter)
        return [self._to_evidence(r) for r in results]

    def search_fts(
        self,
        match_string: str,
        top_k: int = 10,
        filter: Optional[str] = None,
    ) -> list[Evidence]:
        q = _make_query(field_name="content", fts=_make_fts(match_string))
        results = self.collection.query(queries=q, topk=top_k, filter=filter)
        return [self._to_evidence(r) for r in results]

    def search_hybrid(
        self,
        dense_vector: np.ndarray,
        query_text: str,
        top_k: int = 10,
        top_n: Optional[int] = None,
        filter: Optional[str] = None,
        weights: Optional[list[float]] = None,
        use_sparse: bool = False,
        sparse_vector: Optional[SparseVector] = None,
        use_rrf: bool = True,
        rank_constant: int = 10,
    ) -> list[Evidence]:
        """Hibrido dense + FTS via MultiQuery + RRF/WeightedReRanker.

        Por padrao usa RRF com rank_constant adaptativo (5-60).
        Se `use_sparse`, troca FTS por sparse_embedding (modo semantico).
        """
        zvec = _require_zvec()
        dense_q = _make_query(
            field_name="dense_embedding",
            vector=np.asarray(dense_vector).astype(np.float32).tolist(),
        )
        if use_sparse and sparse_vector is not None:
            second_q = _make_query(field_name="sparse_embedding", vector=sparse_vector.values)
        else:
            second_q = _make_query(field_name="content", fts=_make_fts(query_text))

        if use_rrf:
            reranker = zvec.RrfReRanker(rank_constant=rank_constant)
        else:
            w = weights or [1.0, 0.6]
            reranker = zvec.WeightedReRanker(weights=w)

        results = self.collection.query(
            queries=[dense_q, second_q], topk=top_n or top_k, reranker=reranker, filter=filter
        )
        return [self._to_evidence(r) for r in results]

    # -------------------------------------------------------- delete

    def delete_by_doc(self, doc_id: str) -> None:
        # Sintaxe Zvec usa '=' (nao '==') — verificador isolado em _escape_sq.
        filt = f"doc_id = '{_escape_sq(doc_id)}'"
        self.collection.delete_by_filter(filter=filt)
        log.info("deletados chunks do doc=%s", doc_id)

    def delete_by_filter(self, filter: str) -> None:
        self.collection.delete_by_filter(filter=filter)
        log.info("delete por filtro: %s", filter)

    def fetch(self, chunk_ids: Union[str, Sequence[str]]) -> list[Evidence]:
        ids = [chunk_ids] if isinstance(chunk_ids, str) else list(chunk_ids)
        results = self.collection.fetch(ids=ids)
        # fetch retorna dict[str, Doc] — iterar valores
        docs = list(results.values()) if isinstance(results, dict) else (results or [])
        return [self._to_evidence(d) for d in docs]

    # -------------------------------------------------------- misc

    def purge(self) -> None:
        """Remove todos os documentos da collection."""
        self.collection.delete_by_filter(filter="doc_id != ''")
        log.info("collection purgada: todos os chunks removidos")

    def optimize(self) -> None:
        log.info("optimize() consolidando indice...")
        self.collection.optimize()

    def stats(self) -> Any:
        return getattr(self.collection, "stats", None)

    # -------------------------------------------------------- interna

    def _to_evidence(self, doc: Any) -> Evidence:
        # Trata tanto objeto (attr) quanto dict defensivamente.
        def g(name: str, default=None):
            if isinstance(doc, dict):
                return doc.get(name, default)
            return getattr(doc, name, default)

        fields = g("fields", {}) or {}
        if not isinstance(fields, dict):
            # pode ser objeto com atributos
            fields = {k: getattr(fields, k, None) for k in dir(fields) if not k.startswith("_")}

        def gf(n, d=None):
            v = fields.get(n, d) if isinstance(fields, dict) else getattr(fields, n, d)
            return v

        return Evidence(
            chunk_id=g("id") or gf("chunk_id") or "",
            doc_id=gf("doc_id", ""),
            source_path=gf("source_path", ""),
            file_name=gf("file_name", ""),
            score=float(g("score", 0.0) or 0.0),
            chunk_index=int(gf("chunk_index", 0) or 0),
            snippet=gf("content", "") or "",
            h1=gf("h1"),
            h2=gf("h2"),
            h3=gf("h3"),
            title=gf("title"),
            tags=tuple(gf("tags", []) or []),
        )


# ---- helpers ----


def _schema_dense_dim(schema: Any) -> int:
    for v in getattr(schema, "vectors", []) or []:
        if getattr(v, "name", "") == "dense_embedding":
            return int(v.dimension)
    return 1024


def _parents_to_h123(parents) -> tuple[Optional[str], Optional[str], Optional[str]]:
    h1 = h2 = h3 = None
    for level, text in parents:
        if level == 1 and h1 is None:
            h1 = text
        elif level == 2 and h2 is None:
            h2 = text
        elif level == 3 and h3 is None:
            h3 = text
    return h1, h2, h3


def _escape_sq(s: str) -> str:
    return s.replace("'", "\\'")


def _open_collection(zvec, path: str, option=None):
    """Suposição: zvec.open(path, option) para reabrir. Falha gracefully se lock ocupado."""
    opener = getattr(zvec, "open", None)
    if callable(opener):
        if option is not None:
            return opener(path, option=option)
        return opener(path)
    raise RuntimeError("zvec.open indisponivel nesta versao; recriando")
