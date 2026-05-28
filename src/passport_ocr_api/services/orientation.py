import re

from PIL import Image

from passport_ocr_api.config import Settings
from passport_ocr_api.services.local_tesseract import TesseractOcrEngine
from passport_ocr_api.services.mrz_parser import score_passport_text
from passport_ocr_api.services.types import OcrResult, OrientationResult, RotationDegrees

ROTATION_CANDIDATES: tuple[RotationDegrees, ...] = (0, 90, 180, 270)
MRZ_SCORE_WEIGHT = 12.0
FIELD_LABEL_SCORE_WEIGHT = 0.35
LINE_SHAPE_SCORE_WEIGHT = 0.45
CONFIDENCE_SCORE_WEIGHT = 1.5
OSD_AGREEMENT_BONUS = 0.20
MAX_FIELD_LABEL_SCORE = 4.0
MAX_LINE_SHAPE_SCORE = 3.0
MIN_LINE_LENGTH_FOR_SHAPE = 8
MIN_READABLE_LINES_FOR_SHAPE_BONUS = 2
READABLE_LINE_SHAPE_BONUS = 0.5
MRZ_SHAPED_LINE_PATTERN = re.compile(r"^[A-Z0-9<]{25,44}$")
PASSPORT_FIELD_LABELS = (
    "PASSPORT",
    "SURNAME",
    "GIVEN",
    "NATIONALITY",
    "DATE",
    "BIRTH",
    "EXPIRY",
    "SEX",
    "PLACE",
    "COUNTRY",
    "TYPE",
)


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
            score = _orientation_quality_score(ocr_result, rotation, osd_rotation)
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


def _orientation_quality_score(
    ocr_result: OcrResult,
    rotation: RotationDegrees,
    osd_rotation: int | None,
) -> float:
    score = score_passport_text(ocr_result.text) * MRZ_SCORE_WEIGHT
    score += _field_label_score(ocr_result.text) * FIELD_LABEL_SCORE_WEIGHT
    score += _line_shape_score(ocr_result) * LINE_SHAPE_SCORE_WEIGHT
    score += ocr_result.confidence * CONFIDENCE_SCORE_WEIGHT
    if osd_rotation == rotation:
        score += OSD_AGREEMENT_BONUS
    return score


def _field_label_score(text: str) -> float:
    normalized = text.upper()
    matches = sum(1 for label in PASSPORT_FIELD_LABELS if label in normalized)
    return min(float(matches), MAX_FIELD_LABEL_SCORE)


def _line_shape_score(ocr_result: OcrResult) -> float:
    shaped_lines = 0
    readable_lines = 0
    for line in ocr_result.lines:
        normalized = line.text.upper().replace(" ", "")
        if len(normalized) >= MIN_LINE_LENGTH_FOR_SHAPE:
            readable_lines += 1
        if MRZ_SHAPED_LINE_PATTERN.fullmatch(normalized) is not None:
            shaped_lines += 1

    score = min(float(shaped_lines), MAX_LINE_SHAPE_SCORE)
    if readable_lines >= MIN_READABLE_LINES_FOR_SHAPE_BONUS:
        score += READABLE_LINE_SHAPE_BONUS
    return min(score, MAX_LINE_SHAPE_SCORE)
