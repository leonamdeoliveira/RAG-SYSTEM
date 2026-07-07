"""Reranker cross-encoder via sentence-transformers.

Quando `retrieval.rerank=true`, apos a busca no Zvec os top-k candidatos sao
re-ordenados por um cross-encoder que pontua (query, chunk) em pares.
Isso tipicamente melhora precision@k.

Implementacao default: `BAAI/bge-reranker-v2-m3` (multilingue, forte em PT/EN).
Pluggable via `model_name` e totalmente configuravel via env vars RAG_RERANK_*.

Quando sentence-transformers nao esta disponivel ou `enabled=False`, o reranker
e no-op (devolve a lista original) — nao quebra o pipeline.
"""

from __future__ import annotations

from typing import Optional, Sequence

from pipeline.rag.models import Evidence
from pipeline.rag.utils.logging import get_logger

log = get_logger("app.reranker")

DEFAULT_MODEL = "suhaan7988/bge-reranker-v2-m3-int8-onnx"
DEFAULT_MAX_LENGTH = 1024


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        enabled: bool = False,
        device: str = "cpu",
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self._model_name = model_name
        self._enabled = enabled
        self._device = device
        self._max_length = max_length
        self._model: Optional[object] = None

    # -------------------------------------------------------- public

    def rerank(
        self, query: str, evidence: Sequence[Evidence], top_n: Optional[int] = None
    ) -> list[Evidence]:
        """Reordena `evidence` por relevancia cross-encoder. No-op se desativado."""
        if not evidence:
            return list(evidence)
        if not self._enabled:
            return list(evidence)[: (top_n or len(evidence))]

        self._ensure_model()
        if not self._enabled or self._model is None:
            return list(evidence)[: (top_n or len(evidence))]

        pairs = [(query, e.snippet) for e in evidence]
        try:
            scores = self._model.predict(
                pairs,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as e:
            log.warning("reranker.predict falhou (%s) -> no-op", e)
            return list(evidence)[: (top_n or len(evidence))]

        ranked = sorted(
            zip(evidence, scores), key=lambda x: float(x[1]), reverse=True
        )
        out = [e for e, _ in ranked]
        if top_n is not None:
            out = out[:top_n]
        return out

    # -------------------------------------------------------- private

    def _ensure_model(self) -> None:
        if self._model is not None or not self._enabled:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                max_length=self._max_length,
            )
            log.info(
                "cross-encoder carregado: %s (device=%s, max_length=%d)",
                self._model_name,
                self._device,
                self._max_length,
            )
        except ImportError:
            log.warning(
                "sentence-transformers ausente -> reranker desativado (no-op). "
                "Instale com: pip install sentence-transformers"
            )
            self._enabled = False
        except (OSError, ValueError, RuntimeError) as e:
            log.warning(
                "falha ao carregar cross-encoder '%s' (%s) -> reranker desativado (no-op). "
                "Verifique se o modelo foi baixado ou se ha conexao com internet.",
                self._model_name,
                e,
            )
            self._enabled = False
        except Exception as e:
            log.warning(
                "erro inesperado ao carregar cross-encoder (%s) -> reranker desativado (no-op)",
                e,
            )
            self._enabled = False
