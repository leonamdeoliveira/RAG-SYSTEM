import logging
import time

from pipeline.ocr.lmstudio_client import LMStudioClient, LMStudioClientError
from pipeline.ocr.model_loader import load_model_config, load_model_prompts
from pipeline.ocr.ocr_engine.base import (
    OCREngineBase,
    OCREngineError,
    EngineResult,
)
from pipeline.ocr.pdf_utils import PDFPage

logger = logging.getLogger(__name__)


class AIEngine(OCREngineBase):

    def __init__(
        self,
        client: LMStudioClient,
        model_name: str = "glm-ocr",
        mode: str = "text-first",
    ):
        self._client = client
        self._model_name = model_name
        self._mode = mode
        self._model_config = load_model_config(model_name)
        self._prompts = load_model_prompts(model_name)
        self._output_is_html = self._model_config.get("default_prompt_format", "html") == "html"
        self._use_system_prompt = self._model_config.get("use_system_prompt", True)
        self._use_native_ref = self._model_config.get("use_native_text_reference", True)

    @property
    def name(self) -> str:
        return f"ai:{self._model_name}"

    def is_available(self) -> bool:
        try:
            return self._client.is_server_available()
        except Exception:
            return False

    def supports(self, page: PDFPage) -> bool:
        return True

    def extract_text(self, page: PDFPage, **kwargs) -> EngineResult:
        fmt = kwargs.get("fmt", "markdown")
        prompt_fmt = self._model_config.get("default_prompt_format", "html")
        actual_fmt = prompt_fmt if fmt == "markdown" else fmt
        prompt = self._prompts.get(actual_fmt)
        if prompt is None:
            available = list(self._prompts.keys())
            raise OCREngineError(
                f"Prompt format '{actual_fmt}' not found for model '{self._model_name}'. "
                f"Available: {available}"
            )

        system_prompt = "" if not self._use_system_prompt else None

        start = time.time()
        try:
            if self._mode == "text-first" and page.native_text and self._use_native_ref:
                enhanced = (
                    f"{prompt}\n\n"
                    f"As reference, here is the raw extracted text from this page "
                    f"(may have OCR errors or wrong order - use the image as source of truth):\n"
                    f"---\n{page.native_text}\n---"
                )
                content = self._client.ocr_image(enhanced, page.image, system_prompt=system_prompt)
            else:
                content = self._client.ocr_image(prompt, page.image, system_prompt=system_prompt)
        except LMStudioClientError as e:
            raise OCREngineError(f"AI engine failed: {e}") from e

        elapsed = time.time() - start
        return EngineResult(
            text=content,
            engine_name=self.name,
            confidence=1.0,
            processing_time=elapsed,
            metadata={"model": self._model_name, "mode": self._mode},
        )
