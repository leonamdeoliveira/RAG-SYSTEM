import logging
from typing import Optional

from pipeline.ocr.ocr_engine.base import (
    OCREngineBase,
    OCREngineError,
    EngineNotAvailableError,
    EngineResult,
    timed_extract,
)
from pipeline.ocr.ocr_engine.config import HybridOCRConfig
from pipeline.ocr.ocr_engine.quality import QualityScorer
from pipeline.ocr.ocr_engine.text_stats import tokenize, extract_numbers
from pipeline.ocr.pdf_utils import PDFPage, has_meaningful_text

logger = logging.getLogger(__name__)


class OCRRouter:

    def __init__(
        self,
        engines: dict[str, OCREngineBase],
        config: HybridOCRConfig,
    ):
        self.engines = engines
        self.config = config
        self.scorer = QualityScorer()

    def _evaluate_result(self, page: PDFPage, result: EngineResult, accept_threshold: float) -> tuple[bool, dict]:
        native = page.native_text.strip()
        ocr = result.text.strip()

        if len(native) >= 20:
            native_words = tokenize(native)
            ocr_words = tokenize(ocr)
            shared = native_words & ocr_words
            word_recall = len(shared) / max(len(native_words), 1)

            native_nums = extract_numbers(native)
            ocr_nums = extract_numbers(ocr)
            num_shared = native_nums & ocr_nums
            num_recall = len(num_shared) / max(len(native_nums), 1) if native_nums else 1.0

            combined = word_recall * 0.7 + num_recall * 0.3

            logger.info(
                "Router [page %d]: reference eval word=%.1f%% num=%.1f%% combined=%.4f (accept>=%.2f)",
                page.page_num, word_recall * 100, num_recall * 100, combined, accept_threshold,
            )

            report = {
                "method": "reference",
                "word_recall": round(word_recall, 4),
                "num_recall": round(num_recall, 4),
                "combined": round(combined, 4),
            }
            return combined >= accept_threshold, report

        qreport = self.scorer.score(ocr, engine_confidence=result.confidence)
        logger.info(
            "Router [page %d]: heuristic score=%.4f conf=%.4f (accept>=%.2f)",
            page.page_num, qreport.score, result.confidence, accept_threshold,
        )

        report = {
            "method": "heuristic",
            "score": round(qreport.score, 4),
            "engine_confidence": round(result.confidence, 4),
            "details": qreport.to_dict(),
        }
        return qreport.acceptable(accept_threshold), report

    def process_with_fallback(self, page: PDFPage, **kwargs) -> EngineResult:
        mode = self.config.mode

        if mode in ("legacy", "ai_only"):
            return self._run_ai(page, **kwargs)

        if mode == "classic_only":
            return self._run_classic_only(page, **kwargs)

        return self._run_hybrid(page, **kwargs)

    def _run_ai(self, page: PDFPage, **kwargs) -> EngineResult:
        ai_engine = self._get_ai_engine()
        logger.info("Router: using AI engine (%s) for page %d", ai_engine.name, page.page_num)
        return timed_extract(ai_engine, page, **kwargs)

    def _run_classic_only(self, page: PDFPage, **kwargs) -> EngineResult:
        engine_name = self.config.classic_engine
        engine = self._get_engine(engine_name)
        if engine is None:
            raise EngineNotAvailableError(
                f"Classic engine '{self.config.classic_engine}' not available"
            )
        if not engine.is_available():
            raise EngineNotAvailableError(
                f"Engine '{engine.name}' is not available (missing dependencies)"
            )
        logger.info("Router: using %s (classic_only) for page %d", engine.name, page.page_num)
        return timed_extract(engine, page, **kwargs)

    def _run_hybrid(self, page: PDFPage, **kwargs) -> EngineResult:
        accept = self.config.quality_threshold_accept

        if has_meaningful_text(page.native_text, min_chars=50):
            logger.info("Router [page %d]: native text OK, skipping OCR", page.page_num)
            return EngineResult(
                text=page.native_text,
                engine_name="native",
                confidence=1.0,
                processing_time=0.0,
                metadata={"source": "native_text", "chars": len(page.native_text)},
            )

        classic_order = ["tesseract"]
        for engine_name in classic_order:
            engine = self._get_engine(engine_name)
            if engine is None or not engine.is_available():
                continue
            if not engine.supports(page):
                continue

            logger.info("Router [page %d]: trying %s ...", page.page_num, engine_name)
            try:
                result = timed_extract(engine, page, **kwargs)
            except OCREngineError as e:
                logger.warning("Router [page %d]: %s failed: %s", page.page_num, engine_name, e)
                continue

            accepted, quality_report = self._evaluate_result(page, result, accept)
            result.metadata["quality"] = quality_report

            if accepted:
                logger.info("Router [page %d]: accepted %s result", page.page_num, engine_name)
                return result

            logger.info(
                "Router [page %d]: %s rejected, trying next engine",
                page.page_num, engine_name,
            )

        if self.config.enable_glm_fallback:
            logger.info("Router [page %d]: all classic/VLM engines failed, falling back to AI", page.page_num)
            ai_engine = self._get_ai_engine()
            if not ai_engine.is_available():
                raise EngineNotAvailableError(
                    f"[Pagina {page.page_num}] OCR classico nao atingiu qualidade minima "
                    f"e o fallback para IA (GLM-OCR) falhou pois o LM Studio nao esta rodando.\n"
                    f"Para processar esta pagina com maxima qualidade:\n"
                    f"  1. Abra o LM Studio\n"
                    f"  2. Carregue o modelo de OCR (ex: glm-ocr)\n"
                    f"  3. Inicie o servidor em http://localhost:1234/v1\n"
                    f"  4. Execute novamente com --ocr-mode hybrid --resume\n"
                    f"Enquanto isso, as demais paginas ja foram processadas com OCR classico."
                )
            return self._run_ai(page, **kwargs)

        raise OCREngineError(
            f"Router [page {page.page_num}]: no engine could process this page "
            f"(classic engines failed, GLM fallback disabled)"
        )

    def _get_ai_engine(self) -> OCREngineBase:
        for name, engine in self.engines.items():
            if name.startswith("ai:"):
                return engine
        raise EngineNotAvailableError("No AI engine registered")

    def _get_engine(self, name: str) -> Optional[OCREngineBase]:
        for key, engine in self.engines.items():
            if key == name or key.endswith(f":{name}"):
                return engine
        return None
