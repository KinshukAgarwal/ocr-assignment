from PIL import Image

from passport_ocr_api.config import Settings
from passport_ocr_api.services.local_tesseract import TesseractOcrEngine
from passport_ocr_api.services.mrz_parser import score_passport_text
from passport_ocr_api.services.types import OcrResult, OrientationResult, RotationDegrees

ROTATION_CANDIDATES: tuple[RotationDegrees, ...] = (0, 90, 180, 270)


class OrientationCorrector:
    def __init__(self, local_ocr: TesseractOcrEngine, settings: Settings) -> None:
        self._local_ocr = local_ocr
        self._settings = settings

    def correct(self, image: Image.Image) -> OrientationResult:
        osd_rotation = self._local_ocr.detect_orientation(
            image,
            self._settings.local_ocr_timeout_seconds,
        )
        return self._score_rotations(image, osd_rotation)

    def _score_rotations(self, image: Image.Image, osd_rotation: int | None) -> OrientationResult:
        best_rotation: RotationDegrees = 0
        best_image = image
        best_ocr: OcrResult | None = None
        best_score = -1.0

        for rotation in ROTATION_CANDIDATES:
            rotated = _rotate(image, rotation)
            ocr_result = self._local_ocr.extract(
                rotated,
                self._settings.local_ocr_timeout_seconds,
            )
            score = score_passport_text(ocr_result.text) + ocr_result.confidence
            if score > best_score:
                best_rotation = rotation
                best_image = rotated
                best_ocr = ocr_result
                best_score = score

        assert best_ocr is not None, "rotation candidates must produce an OCR result"
        method = "tesseract_osd" if best_rotation == osd_rotation else "rotation_scoring"
        return OrientationResult(
            image=best_image,
            detected_degrees=best_rotation,
            corrected=best_rotation != 0,
            method=method,
            local_ocr=best_ocr,
        )


def _rotate(image: Image.Image, degrees: RotationDegrees) -> Image.Image:
    if degrees == 0:
        return image
    return image.rotate(-degrees, expand=True)
