import logging
import time

from PIL import Image

from passport_ocr_api.config import Settings
from passport_ocr_api.errors import OcrFailedError, OcrTimeoutError, OcrUnavailableError
from passport_ocr_api.logging_config import log_extra
from passport_ocr_api.schemas import ConfidenceInfo, OcrInfo, OrientationInfo, PassportOcrResponse
from passport_ocr_api.services.document_loader import DocumentLoader
from passport_ocr_api.services.image_preprocessing import to_black_and_white_text_image
from passport_ocr_api.services.orientation import OrientationCorrector
from passport_ocr_api.services.types import (
    CloudOcrEngine,
    OcrResult,
    PassportImageExtractor,
    PassportParser,
    PassportValidator,
)
from passport_ocr_api.services.upload_validator import UploadedDocument

logger = logging.getLogger(__name__)
TEXT_FALLBACK_BINARY_THRESHOLD = 100


class PassportOcrPipeline:
    def __init__(
        self,
        settings: Settings,
        document_loader: DocumentLoader,
        orientation_corrector: OrientationCorrector,
        parser: PassportParser,
        validator: PassportValidator,
        image_extractor: PassportImageExtractor,
        cloud_ocr: CloudOcrEngine,
    ) -> None:
        self._settings = settings
        self._document_loader = document_loader
        self._orientation_corrector = orientation_corrector
        self._parser = parser
        self._validator = validator
        self._image_extractor = image_extractor
        self._cloud_ocr = cloud_ocr

    def extract(self, document: UploadedDocument, request_id: str) -> PassportOcrResponse:
        started = time.monotonic()
        pages = self._document_loader.load_pages(document)
        first_page = pages[0]
        try:
            orientation = self._orientation_corrector.correct(first_page.image)
        except (OcrFailedError, OcrTimeoutError, OcrUnavailableError):
            if not self._settings.google_fallback_enabled:
                raise
            logger.warning("local ocr unavailable; using cloud fallback", **log_extra(request_id))
            return self._extract_with_cloud_only(first_page.image, request_id, started)

        images = self._image_extractor.extract(orientation.image)
        text_image = to_black_and_white_text_image(
            orientation.image,
            threshold=TEXT_FALLBACK_BINARY_THRESHOLD,
        )
        parsed = self._parser.parse(orientation.local_ocr.text, orientation.local_ocr.confidence)
        fallback_result = self._fallback_if_needed(
            text_image,
            orientation.local_ocr,
            parsed.confidence,
            request_id,
        )

        final_ocr = fallback_result or orientation.local_ocr
        if fallback_result is not None:
            parsed = self._parser.parse(final_ocr.text, final_ocr.confidence)

        validation = self._validator.validate(parsed.extraction)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "passport ocr completed engine=%s fallback=%s confidence=%.3f elapsed_ms=%s",
            final_ocr.engine,
            fallback_result is not None,
            parsed.confidence,
            elapsed_ms,
            **log_extra(request_id),
        )

        return PassportOcrResponse(
            request_id=request_id,
            orientation=OrientationInfo(
                detected_degrees=orientation.detected_degrees,
                corrected=orientation.corrected,
                method=orientation.method,
            ),
            extraction=parsed.extraction,
            images=images,
            confidence=ConfidenceInfo(
                overall=parsed.confidence,
                fields=parsed.field_confidence,
            ),
            validation=validation,
            ocr=OcrInfo(
                engine="hybrid" if fallback_result is not None else orientation.local_ocr.engine,
                fallback_used=fallback_result is not None,
                raw_text_snippets=self._snippets(final_ocr),
            ),
        )

    def _extract_with_cloud_only(
        self,
        image: Image.Image,
        request_id: str,
        started: float,
    ) -> PassportOcrResponse:
        images = self._image_extractor.extract(image)
        text_image = to_black_and_white_text_image(
            image,
            threshold=TEXT_FALLBACK_BINARY_THRESHOLD,
        )
        cloud_ocr = self._cloud_ocr.extract(
            text_image,
            self._settings.google_ocr_timeout_seconds,
            request_id,
        )
        parsed = self._parser.parse(cloud_ocr.text, cloud_ocr.confidence)
        validation = self._validator.validate(parsed.extraction)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "passport ocr completed engine=%s fallback=%s confidence=%.3f elapsed_ms=%s",
            cloud_ocr.engine,
            True,
            parsed.confidence,
            elapsed_ms,
            **log_extra(request_id),
        )

        return PassportOcrResponse(
            request_id=request_id,
            orientation=OrientationInfo(
                detected_degrees=0,
                corrected=False,
                method="local_orientation_unavailable",
            ),
            extraction=parsed.extraction,
            images=images,
            confidence=ConfidenceInfo(
                overall=parsed.confidence,
                fields=parsed.field_confidence,
            ),
            validation=validation,
            ocr=OcrInfo(
                engine=cloud_ocr.engine,
                fallback_used=True,
                raw_text_snippets=self._snippets(cloud_ocr),
            ),
        )

    def _fallback_if_needed(
        self,
        image: Image.Image,
        local_ocr: OcrResult,
        parsed_confidence: float,
        request_id: str,
    ) -> OcrResult | None:
        if not self._settings.google_fallback_enabled:
            return None
        if (
            local_ocr.confidence >= self._settings.low_confidence_threshold
            and parsed_confidence >= self._settings.low_confidence_threshold
        ):
            return None

        return self._cloud_ocr.extract(
            image,
            self._settings.google_ocr_timeout_seconds,
            request_id,
        )

    def _snippets(self, ocr_result: OcrResult) -> list[str]:
        snippets: list[str] = []
        for line in ocr_result.lines:
            if len(snippets) >= self._settings.max_raw_snippets:
                break
            snippet = line.text[: self._settings.max_raw_snippet_chars]
            if snippet:
                snippets.append(snippet)
        return snippets
