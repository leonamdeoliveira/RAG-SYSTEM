"""Cache persistente de embeddings com SQLite (stdlib, zero deps).

Armazena dense + sparse embeddings por chunk_id, permitindo reindex incremental
sem re-embeddar chunks inalterados. Schema versionado por model_id para evitar
incompatibilidade entre backends (torch/onnx/sentence-transformers).

Uso:
    cache = EmbedCache(Path("index/embed_cache.db"), model_id="bge-m3-onnx-int8")
    result = cache.get(chunk_id)  # (dense, sparse) ou None
    cache.set(chunk_id, dense_vec, sparse_vec)
    cache.commit()
    cache.close()
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.rag.embeddings.base import SparseVector
from pipeline.rag.utils.logging import get_logger

log = get_logger("app.embed_cache")

_CREATE = """
CREATE TABLE IF NOT EXISTS embed_cache (
    chunk_id TEXT    NOT NULL,
    model    TEXT    NOT NULL,
    dense    BLOB    NOT NULL,
    sparse   BLOB,
    PRIMARY KEY (chunk_id, model)
);
"""

_BATCH_COMMIT_SIZE = 64


def _encode_sparse(sv: Optional[SparseVector]) -> Optional[bytes]:
    """Serializa SparseVector como BLOB binário: [n_int32 indices][n_float32 weights]."""
    if sv is None or not sv.values:
        return None
    items = sorted(sv.values.items())
    indices = np.array([k for k, _ in items], dtype=np.int32)
    weights = np.array([v for _, v in items], dtype=np.float32)
    header = struct.pack("<I", len(items))
    return header + indices.tobytes() + weights.tobytes()


def _decode_sparse(data: Optional[bytes]) -> Optional[SparseVector]:
    """Desserializa BLOB binário em SparseVector."""
    if data is None:
        return None
    n = struct.unpack_from("<I", data, 0)[0]
    if n == 0:
        return None
    offset = 4
    indices = np.frombuffer(data, dtype=np.int32, count=n, offset=offset)
    weights = np.frombuffer(data, dtype=np.float32, count=n, offset=offset + n * 4)
    return SparseVector(values={int(i): float(w) for i, w in zip(indices, weights)})


class EmbedCache:
    """Cache persistente de embeddings (dense + sparse). Zero deps externas."""

    def __init__(self, path: Path, model_id: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(_CREATE)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.commit()
        self._model_id = model_id
        self._dirty = 0
        self._hits = 0
        self._misses = 0
        self._db_path = path
        log.debug("embed cache inicializado: %s (model=%s)", path, model_id)

    def get(self, chunk_id: str) -> Optional[tuple[np.ndarray, Optional[SparseVector]]]:
        """Retorna (dense, sparse) do cache ou None se não encontrado."""
        row = self._conn.execute(
            "SELECT dense, sparse FROM embed_cache WHERE chunk_id=? AND model=?",
            (chunk_id, self._model_id),
        ).fetchone()
        if row is None:
            self._misses += 1
            return None
        self._hits += 1
        dense = np.frombuffer(row[0], dtype=np.float32).copy()
        return dense, _decode_sparse(row[1])

    def get_many(
        self, chunk_ids: list[str]
    ) -> dict[str, tuple[np.ndarray, Optional[SparseVector]]]:
        """Batch lookup — mais eficiente que get() individual."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        query = f"""
            SELECT chunk_id, dense, sparse FROM embed_cache
            WHERE chunk_id IN ({placeholders}) AND model=?
        """
        params = list(chunk_ids) + [self._model_id]
        rows = self._conn.execute(query, params).fetchall()
        return {
            row[0]: (
                np.frombuffer(row[1], dtype=np.float32).copy(),
                _decode_sparse(row[2]),
            )
            for row in rows
        }

    def set(
        self,
        chunk_id: str,
        dense: np.ndarray,
        sparse: Optional[SparseVector] = None,
    ) -> None:
        """Salva embedding no cache. Commit automático a cada 64 chunks."""
        self._conn.execute(
            "INSERT OR REPLACE INTO embed_cache VALUES (?,?,?,?)",
            (
                chunk_id,
                self._model_id,
                dense.astype(np.float32).tobytes(),
                _encode_sparse(sparse),
            ),
        )
        self._dirty += 1
        if self._dirty >= _BATCH_COMMIT_SIZE:
            self._conn.commit()
            self._dirty = 0

    def set_many(
        self,
        items: list[tuple[str, np.ndarray, Optional[SparseVector]]],
    ) -> None:
        """Batch insert — mais eficiente que set() individual."""
        if not items:
            return
        data = [
            (chunk_id, self._model_id, dense.astype(np.float32).tobytes(), _encode_sparse(sparse))
            for chunk_id, dense, sparse in items
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embed_cache VALUES (?,?,?,?)", data
        )
        self._conn.commit()
        self._dirty = 0

    def commit(self) -> None:
        """Força commit de pendências."""
        self._conn.commit()
        self._dirty = 0

    def close(self) -> None:
        """Commit e fecha conexão."""
        self.commit()
        self._conn.close()

    def invalidate_model(self, model_id: Optional[str] = None) -> int:
        """Remove entradas de um modelo específico (ou todos se None)."""
        if model_id is None:
            cursor = self._conn.execute("DELETE FROM embed_cache")
        else:
            cursor = self._conn.execute(
                "DELETE FROM embed_cache WHERE model=?", (model_id,)
            )
        self._conn.commit()
        return cursor.rowcount

    def count(self, model_id: Optional[str] = None) -> int:
        """Retorna número de entradas no cache."""
        mid = model_id or self._model_id
        return self._conn.execute(
            "SELECT COUNT(*) FROM embed_cache WHERE model=?", (mid,)
        ).fetchone()[0]

    def __len__(self) -> int:
        return self.count()
    
    def stats(self) -> dict:
        """Retorna estatísticas do cache."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        
        # Tamanho do arquivo
        try:
            db_size = self._db_path.stat().st_size
        except (OSError, AttributeError):
            db_size = 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate_pct": round(hit_rate, 2),
            "cache_entries": self.count(),
            "db_size_mb": round(db_size / (1024*1024), 2),
            "model_id": self._model_id,
        }
    
    def print_stats(self) -> None:
        """Imprime estatísticas formatadas."""
        s = self.stats()
        log.info("=== Embed Cache Stats ===")
        log.info("  Hit rate: %s%% (%d/%d)", s['hit_rate_pct'], s['hits'], s['total_requests'])
        log.info("  Entries:  %d", s['cache_entries'])
        log.info("  Size:     %s MB", s['db_size_mb'])
        log.info("  Model:    %s", s['model_id'])
