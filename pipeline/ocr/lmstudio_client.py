import time
import logging
import base64
from io import BytesIO
from typing import Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class LMStudioClientError(Exception):
    pass


class LMStudioClient:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "chandra-ocr-2",
        api_key: str = "",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_tokens: int = 4096,
        extra_params: Optional[dict] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_tokens = max_tokens
        self.extra_params = extra_params or {}

    def _encode_image(self, image: Image.Image, fmt: str = "PNG") -> str:
        buffer = BytesIO()
        image.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_multimodal_payload(
        self, prompt: str, image: Optional[Image.Image] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        default_system = "You are an OCR engine. Extract all text from the document image faithfully. Do not invent, summarize, or translate."
        messages = [{"role": "system", "content": system_prompt or default_system}]

        content = []

        if image is not None:
            b64 = self._encode_image(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    },
                }
            )

        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        if self.extra_params:
            payload.update(self.extra_params)
        return payload

    def _build_text_payload(self, prompt: str) -> dict:
        messages = [
            {"role": "system", "content": "You are an OCR engine. Extract all text from the document faithfully."},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        if self.extra_params:
            payload.update(self.extra_params)
        return payload

    def _request(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise LMStudioClientError(f"Model error: {data['error']}")
                return data
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "Attempt %d/%d failed: %s", attempt, self.max_retries, e
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
                else:
                    raise LMStudioClientError(
                        f"Request failed after {self.max_retries} attempts: {e}"
                    ) from e

    def _extract_content(self, data: dict) -> str:
        try:
            msg = data["choices"][0]["message"]
            content = msg.get("content", "") or ""
            if not content:
                content = msg.get("reasoning_content", "") or ""
            if not content:
                logger.warning("Model returned empty content. finish_reason=%s",
                               data["choices"][0].get("finish_reason"))
            return content
        except (KeyError, IndexError) as e:
            logger.error("Unexpected API response: %s", data)
            raise LMStudioClientError(f"Unexpected API response format: {e}") from e

    def ocr_image(self, prompt: str, image: Image.Image, system_prompt: Optional[str] = None) -> str:
        payload = self._build_multimodal_payload(prompt, image, system_prompt=system_prompt)
        data = self._request(payload)
        return self._extract_content(data)

    def ocr_text(self, prompt: str) -> str:
        payload = self._build_text_payload(prompt)
        data = self._request(payload)
        return self._extract_content(data)

    def ocr_images(self, prompt: str, images: list[Image.Image], system_prompt: Optional[str] = None) -> str:
        default_system = "You are an OCR engine. Extract all text from the document images faithfully. Do not invent, summarize, or translate."
        messages = [{"role": "system", "content": system_prompt or default_system}]
        content = []
        for img in images:
            b64 = self._encode_image(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        payload = {
            "model": self.model, "messages": messages,
            "max_tokens": self.max_tokens, "temperature": 0.0,
        }
        if self.extra_params:
            payload.update(self.extra_params)
        data = self._request(payload)
        return self._extract_content(data)

    def is_server_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False
