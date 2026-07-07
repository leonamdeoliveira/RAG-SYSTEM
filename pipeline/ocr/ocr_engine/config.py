from dataclasses import dataclass
from typing import ClassVar


@dataclass
class HybridOCRConfig:
    mode: str = "hybrid"
    classic_engine: str = "tesseract"
    enable_glm_fallback: bool = True
    langs: str = "por+eng"
    quality_threshold_accept: float = 0.70
    ocr_timeout: int = 120

    VALID_MODES: ClassVar[set[str]] = {"legacy", "hybrid", "classic_only", "ai_only"}

    def __post_init__(self):
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of: {self.VALID_MODES}")
        self.quality_threshold_accept = max(0.0, min(1.0, self.quality_threshold_accept))
