import logging
import time
from io import BytesIO

from google.api_core.exceptions import DeadlineExceeded, GoogleAPIError, ServiceUnavailable
from google.cloud import vision
from PIL import Image

from passport_ocr_api.config import Settings
from passport_ocr_api.errors import OcrFailedError, OcrTimeoutError, OcrUnavailableError
from passport_ocr_api.logging_config import log_extra
from passport_ocr_api.services.circuit_breaker import CircuitBreaker
from passport_ocr_api.services.types import OcrLine, OcrResult

logger = logging.getLogger(__name__)


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
                sleep_seconds = min(0.2 * attempt, 1.0)
                logger.warning(
                    "google vision retry attempt=%s",
                    attempt,
                    **log_extra(request_id),
                )
                time.sleep(sleep_seconds)

        if isinstance(last_error, DeadlineExceeded):
            raise OcrTimeoutError("Google Vision OCR timed out.") from last_error
        raise OcrFailedError("Google Vision OCR failed.") from last_error

    def _extract_once(self, image: Image.Image, timeout_seconds: int) -> OcrResult:
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

    def _get_client(self) -> vision.ImageAnnotatorClient:
        if self._client is None:
            self._client = vision.ImageAnnotatorClient()
        return self._client


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
