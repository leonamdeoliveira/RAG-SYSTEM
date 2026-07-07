import logging
import time
from typing import Optional

from pipeline.ocr.ocr_engine.base import (
    OCREngineBase,
    OCREngineError,
    EngineNotAvailableError,
    EngineResult,
)
from pipeline.ocr.pdf_utils import PDFPage

logger = logging.getLogger(__name__)


class VLMEngine(OCREngineBase):

    def __init__(
        self,
        model_id: str = "ibm-granite/granite-docling-258M",
        device: str = "cpu",
    ):
        self._model_id = model_id
        self._device = device
        self._model = None
        self._processor = None
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "vlm:granitedocling"

    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            import torch

            logger.info("Loading GraniteDocling model (%s) on %s ...", self._model_id, self._device)
            self._processor = AutoProcessor.from_pretrained(
                self._model_id, trust_remote_code=True,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_id,
                dtype=torch.float32 if self._device == "cpu" else torch.float16,
                trust_remote_code=True,
            ).to(self._device)
            self._model.eval()
            logger.info("GraniteDocling model loaded successfully")
        except ImportError:
            raise EngineNotAvailableError(
                "GraniteDocling requires transformers and torch. "
                "Run: pip install transformers torch"
            )
        except Exception as e:
            raise EngineNotAvailableError(
                f"Failed to load GraniteDocling model '{self._model_id}': {e}"
            )

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            self._ensure_loaded()
            self._available = True
        except EngineNotAvailableError:
            self._available = False
        return self._available

    def supports(self, page: PDFPage) -> bool:
        return True

    def extract_text(self, page: PDFPage, **kwargs) -> EngineResult:
        if not self.is_available():
            raise EngineNotAvailableError("GraniteDocling model is not available")

        start = time.time()
        try:
            image = page.image
            if image.mode != "RGB":
                image = image.convert("RGB")

            confidences = []
            bboxes = []

            inputs = None
            if hasattr(self._processor, "apply_chat_template"):
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Convert this document page to markdown."}
                    ]
                }]
                try:
                    inputs = self._processor.apply_chat_template(
                        messages,
                        images=[image],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt",
                    )
                except Exception:
                    pass

            if inputs is None:
                inputs = self._processor(
                    images=image,
                    text="<image>\nConvert this document page to markdown.",
                    return_tensors="pt",
                )
            if self._device != "cpu":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                )

            generated = self._processor.batch_decode(
                outputs, skip_special_tokens=True,
            )[0]

            if hasattr(self._processor, "token2bbox"):
                try:
                    generated, confidences, bboxes = self._processor.token2bbox(generated)
                except Exception:
                    pass

            text = generated.strip() if generated else ""

        except Exception as e:
            raise OCREngineError(f"GraniteDocling VLM failed: {e}") from e

        elapsed = time.time() - start
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.8

        return EngineResult(
            text=text,
            engine_name=self.name,
            confidence=avg_conf,
            processing_time=elapsed,
            metadata={
                "model": self._model_id,
                "device": self._device,
                "bbox_count": len(bboxes),
            },
        )
