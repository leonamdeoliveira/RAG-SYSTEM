import logging
import os
import tempfile
import time
from pathlib import Path

from pipeline.ocr.ocr_engine.base import (
    OCREngineBase,
    OCREngineError,
    EngineNotAvailableError,
    EngineResult,
)
from pipeline.ocr.pdf_utils import PDFPage

logger = logging.getLogger(__name__)


class TesseractEngine(OCREngineBase):

    _COMMON_PATHS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]

    def __init__(self, langs: str = "por+eng", timeout: int = 120):
        self._langs = langs
        self._timeout = timeout
        self._available = None

    @property
    def name(self) -> str:
        return "tesseract"

    @staticmethod
    def _find_tesseract() -> str:
        for path in TesseractEngine._COMMON_PATHS:
            p = Path(path)
            if p.exists():
                return str(p)
        return ""

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import pytesseract
            tesseract_cmd = os.environ.get("TESSERACT_CMD") or self._find_tesseract()
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            pytesseract.get_tesseract_version()
            self._available = True
        except Exception:
            logger.warning("Tesseract not available. Install pytesseract and Tesseract binary.")
            self._available = False
        return self._available

    def supports(self, page: PDFPage) -> bool:
        return True

    def extract_text(self, page: PDFPage, **kwargs) -> EngineResult:
        if not self.is_available():
            raise EngineNotAvailableError("Tesseract is not installed")

        import pytesseract

        start = time.time()
        try:
            lang = kwargs.get("langs", self._langs)
            data = pytesseract.image_to_data(page.image, lang=lang, output_type=pytesseract.Output.DICT, timeout=self._timeout)
            confidences = [c for c in data["conf"] if c > 0]
            avg_conf = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0

            words = []
            for i, word in enumerate(data["text"]):
                w = word.strip()
                if w:
                    words.append(w)
            text = " ".join(words)
        except RuntimeError as e:
            raise OCREngineError(f"Tesseract failed: {e}") from e

        elapsed = time.time() - start
        return EngineResult(
            text=text.strip(),
            engine_name=self.name,
            confidence=avg_conf,
            processing_time=elapsed,
            metadata={"langs": self._langs, "method": "image_to_data", "mean_word_conf": round(avg_conf, 4)},
        )

    @staticmethod
    def ocrmypdf_pdf(pdf_path: Path, langs: str = "por+eng", timeout: int = 120) -> str:
        try:
            import ocrmypdf
        except ImportError:
            raise EngineNotAvailableError("OCRmyPDF is not installed")

        import fitz

        start = time.time()
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp_path = Path(tmp.name)
            tmp.close()

            ocrmypdf.ocr(
                str(pdf_path),
                str(tmp_path),
                language=langs,
                force_ocr=True,
                skip_text=False,
                timeout=timeout,
                progress_bar=False,
            )

            doc = fitz.open(str(tmp_path))
            text = "\n".join(page.get_text("text") for page in doc)
            doc.close()
        except Exception as e:
            raise OCREngineError(f"OCRmyPDF failed: {e}") from e
        finally:
            if tmp_path.exists():
                try:
                    os.unlink(str(tmp_path))
                except OSError:
                    pass

        return text.strip()
