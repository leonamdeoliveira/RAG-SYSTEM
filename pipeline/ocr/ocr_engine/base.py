import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pipeline.ocr.pdf_utils import PDFPage


class OCREngineError(Exception):
    """Base exception for all OCR engine errors."""


class EngineNotAvailableError(OCREngineError):
    """Raised when the engine's dependencies are not installed."""


@dataclass
class EngineResult:
    text: str
    engine_name: str
    confidence: float
    processing_time: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "engine": self.engine_name,
            "confidence": round(self.confidence, 4),
            "processing_time_s": round(self.processing_time, 2),
            "chars": len(self.text),
            "metadata": self.metadata,
        }


class OCREngineBase(ABC):

    @abstractmethod
    def extract_text(self, page: PDFPage, **kwargs) -> EngineResult:
        ...

    @abstractmethod
    def supports(self, page: PDFPage) -> bool:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


def timed_extract(engine: OCREngineBase, page: PDFPage, **kwargs) -> EngineResult:
    start = time.time()
    result = engine.extract_text(page, **kwargs)
    elapsed = time.time() - start
    result.processing_time = elapsed
    return result
