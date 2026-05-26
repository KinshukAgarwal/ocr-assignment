import logging
import time
from base64 import b64encode
from io import BytesIO
from typing import Any

import httpx
from google.api_core.exceptions import DeadlineExceeded, GoogleAPIError, ServiceUnavailable
from google.cloud import vision
from PIL import Image

from passport_ocr_api.config import Settings
from passport_ocr_api.errors import OcrFailedError, OcrTimeoutError, OcrUnavailableError
from passport_ocr_api.logging_config import log_extra
from passport_ocr_api.services.circuit_breaker import CircuitBreaker
from passport_ocr_api.services.types import OcrLine, OcrResult

logger = logging.getLogger(__name__)

VISION_FEATURE_TYPE = "DOCUMENT_TEXT_DETECTION"
RETRY_BASE_DELAY_SECONDS = 0.2
RETRY_MAX_DELAY_SECONDS = 1.0


class GoogleVisionOcrEngine:
    def __init__(self, settings: Settings, circuit_breaker: CircuitBreaker | None = None) -> None:
        self._settings = settings
        self._client: vision.ImageAnnotatorClient | None = None
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            reset_seconds=settings.circuit_breaker_reset_seconds,
        )

    def extract(self, image: Image.Image, timeout_seconds: int, request_id: str) -> OcrResult:
        if not self._settings.google_fallback_enabled:
            raise OcrUnavailableError("Google Vision fallback is disabled.")
        if not self._circuit_breaker.can_call():
            raise OcrUnavailableError("Google Vision fallback circuit is open.")

        attempts = self._settings.google_max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                result = self._extract_once(image, timeout_seconds)
                self._circuit_breaker.record_success()
                return result
            except httpx.TimeoutException as exc:
                last_error = exc
                self._circuit_breaker.record_failure()
            except httpx.HTTPError as exc:
                last_error = exc
                self._circuit_breaker.record_failure()
                if not _is_retryable_http_error(exc):
                    break
            except DeadlineExceeded as exc:
                last_error = exc
                self._circuit_breaker.record_failure()
            except ServiceUnavailable as exc:
                last_error = exc
                self._circuit_breaker.record_failure()
            except GoogleAPIError as exc:
                last_error = exc
                self._circuit_breaker.record_failure()
                break

            if attempt < attempts:
                sleep_seconds = min(RETRY_BASE_DELAY_SECONDS * attempt, RETRY_MAX_DELAY_SECONDS)
                logger.warning(
                    "google vision retry attempt=%s",
                    attempt,
                    **log_extra(request_id),
                )
                time.sleep(sleep_seconds)

        if isinstance(last_error, DeadlineExceeded | httpx.TimeoutException):
            raise OcrTimeoutError("Google Vision OCR timed out.") from last_error
        raise OcrFailedError("Google Vision OCR failed.") from last_error

    def _extract_once(self, image: Image.Image, timeout_seconds: int) -> OcrResult:
        if self._settings.google_vision_api_key:
            return self._extract_once_with_api_key(image, timeout_seconds)
        return self._extract_once_with_adc(image, timeout_seconds)

    def _extract_once_with_adc(self, image: Image.Image, timeout_seconds: int) -> OcrResult:
        client = self._get_client()
        response = client.document_text_detection(
            image=vision.Image(content=_image_to_png_bytes(image)),
            timeout=timeout_seconds,
        )
        if response.error.message:
            raise OcrFailedError("Google Vision returned an OCR error.")

        full_text = response.full_text_annotation.text if response.full_text_annotation else ""
        lines = tuple(
            OcrLine(text=line.strip(), confidence=1.0)
            for line in full_text.splitlines()
            if line.strip()
        )
        confidence = 1.0 if full_text.strip() else 0.0
        return OcrResult(engine="google_vision", text=full_text, confidence=confidence, lines=lines)

    def _extract_once_with_api_key(self, image: Image.Image, timeout_seconds: int) -> OcrResult:
        payload = _build_vision_request(image)
        params = {"key": self._settings.google_vision_api_key}
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(
                self._settings.google_vision_rest_endpoint,
                params=params,
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        error = _extract_response_error(data)
        if error is not None:
            raise OcrFailedError("Google Vision returned an OCR error.")

        full_text = _extract_response_text(data)
        lines = tuple(
            OcrLine(text=line.strip(), confidence=1.0)
            for line in full_text.splitlines()
            if line.strip()
        )
        confidence = 1.0 if full_text.strip() else 0.0
        return OcrResult(engine="google_vision", text=full_text, confidence=confidence, lines=lines)

    def _get_client(self) -> vision.ImageAnnotatorClient:
        if self._client is None:
            self._client = vision.ImageAnnotatorClient()
        return self._client


def _build_vision_request(image: Image.Image) -> dict[str, Any]:
    encoded_image = b64encode(_image_to_png_bytes(image)).decode("ascii")
    return {
        "requests": [
            {
                "image": {"content": encoded_image},
                "features": [{"type": VISION_FEATURE_TYPE}],
            }
        ]
    }


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _extract_response_text(data: dict[str, Any]) -> str:
    responses = data.get("responses")
    if not isinstance(responses, list) or not responses:
        return ""

    first_response = responses[0]
    if not isinstance(first_response, dict):
        return ""

    annotation = first_response.get("fullTextAnnotation")
    if not isinstance(annotation, dict):
        return ""

    text = annotation.get("text")
    return text if isinstance(text, str) else ""


def _extract_response_error(data: dict[str, Any]) -> dict[str, Any] | None:
    responses = data.get("responses")
    if not isinstance(responses, list) or not responses:
        return None

    first_response = responses[0]
    if not isinstance(first_response, dict):
        return None

    error = first_response.get("error")
    return error if isinstance(error, dict) else None


def _is_retryable_http_error(exc: httpx.HTTPError) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return True
    return exc.response.status_code in {429, 500, 502, 503, 504}
