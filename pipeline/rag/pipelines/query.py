"""Pipeline de query: pergunta -> retrieval -> resposta (com Reliability Layer).

Reliability Layer (ARCHITECTURE.md §2.7):
  - Nunca presume que o primeiro retrieval basta.
  - Se confidence < threshold:
      1. expande top_k (factor configuravel)
      2. se modo era 'dense', tenta 'hybrid'
      3. se persistir -> marca insufficient_context
  - Se chunks conflitantes (mesma pergunta, respostas opostas com scores altos):
       repassa flag conflict para o Answerer (system prompt ja instrui apontar).

Retorna `Answer` com `evidence`, `confidence`, `insufficient_context`, `conflict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pipeline.rag.generation.answerer import Answerer
from pipeline.rag.models import Answer
from pipeline.rag.retrieval.filters import FilterBuilder
from pipeline.rag.retrieval.retriever import Retriever
from pipeline.rag.utils.logging import get_logger

log = get_logger("app.pipelines.query")


@dataclass
class QueryConfig:
    low_confidence_threshold: float = 0.02
    expand_factor: int = 2
    max_retries: int = 1
    conflict_score_gap: float = 0.15  # diferenca minima de score para NAO ser conflito


class QueryPipeline:
    def __init__(
        self,
        retriever: Retriever,
        answerer: Answerer,
        config: Optional[QueryConfig] = None,
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.config = config or QueryConfig()

    def run(
        self,
        question: str,
        filters: Optional[FilterBuilder] = None,
        mode: Optional[str] = None,
    ) -> Answer:
        cfg = self.config
        rc = self.retriever.config

        # 1) retrieval inicial
        hits, confidence = self.retriever.retrieve_with_confidence(question, filters)
        log.info("retrieval inicial: %d hits, conf=%.3f, mode=%s", len(hits), confidence, rc.mode)

        attempts = 0
        old_topk = rc.top_k
        old_mode = rc.mode
        try:
            # 2) retry se baixa confianca e ainda nao saturou
            while (
                attempts < cfg.max_retries
                and (not hits or confidence < cfg.low_confidence_threshold)
            ):
                attempts += 1
                rc.top_k = rc.top_k * cfg.expand_factor
                if rc.mode == "dense":
                    rc.mode = "hybrid"
                    log.info("retry %d: expandindo top_k=%d e modo -> hybrid", attempts, rc.top_k)
                else:
                    log.info("retry %d: expandindo top_k=%d", attempts, rc.top_k)
                hits, confidence = self.retriever.retrieve_with_confidence(question, filters)
        finally:
            # restaura config original (stateless entre queries)
            rc.top_k = old_topk
            rc.mode = old_mode

        # 3) gerar resposta
        answer = self.answerer.answer(question, hits, mode=mode)

        # 4) deteccao de conflito (heuristica de scores proximos em chunks diferentes)
        if self._has_potential_conflict(hits, cfg.conflict_score_gap):
            answer = Answer(
                text=answer.text,
                mode=answer.mode,
                evidence=answer.evidence,
                confidence=answer.confidence,
                insufficient_context=answer.insufficient_context,
                conflict=True,
            )
            log.info("possivel conflito detectado entre evidencias")

        if answer.insufficient_context:
            log.info("contexto insuficiente para '%s'", question[:60])

        return answer

    @staticmethod
    def _has_potential_conflict(hits, gap: float) -> bool:
        """Heuristica: 2+ chunks de docs diferentes com scores muito proximos.
        (A confirmacao real de conflito semantico cabe ao LLM no system prompt.)
        """
        if len(hits) < 2:
            return False
        sorted_hits = sorted(hits, key=lambda e: e.score, reverse=True)
        a, b = sorted_hits[0], sorted_hits[1]
        if a.doc_id == b.doc_id:
            return False
        return abs(a.score - b.score) <= gap
