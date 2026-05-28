import re
import shutil
from dataclasses import dataclass
from typing import Any, SupportsFloat, SupportsIndex

import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output, TesseractError, TesseractNotFoundError

from passport_ocr_api.errors import OcrFailedError, OcrTimeoutError, OcrUnavailableError
from passport_ocr_api.services.image_preprocessing import to_black_and_white_text_image
from passport_ocr_api.services.mrz_parser import score_passport_text
from passport_ocr_api.services.types import OcrLine, OcrResult

TESSERACT_CONFIG = "--oem 3 --psm 6"
MRZ_TESSERACT_CONFIG = (
    "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
)
MRZ_SCALE_FACTOR = 2
MRZ_BINARY_THRESHOLD = 145
MRZ_BAND_TOP_RATIO = 0.58
FULL_PAGE_TEXT_BINARY_THRESHOLD = 100
MIN_JOINED_MRZ_LENGTH = 25
MRZ_WORD_PATTERN = re.compile(r"^[A-Z0-9<]+$")


@dataclass(frozen=True)
class OcrVariant:
    image: Image.Image
    config: str


class TesseractOcrEngine:
    def __init__(self, command_path: str | None = None) -> None:
        self._command_path = command_path or shutil.which("tesseract")
        if self._command_path is not None:
            pytesseract.pytesseract.tesseract_cmd = self._command_path

    def extract(self, image: Image.Image, timeout_seconds: int) -> OcrResult:
        if self._command_path is None:
            raise OcrUnavailableError("Tesseract is not installed or is not on PATH.")

        variants = _ocr_variants(image)
        per_variant_timeout = max(1, timeout_seconds // len(variants))
        results: list[OcrResult] = []
        last_error: OcrFailedError | OcrTimeoutError | OcrUnavailableError | None = None
        for variant in variants:
            try:
                results.append(self._extract_variant(variant, per_variant_timeout))
            except (OcrFailedError, OcrTimeoutError, OcrUnavailableError) as exc:
                last_error = exc

        if results:
            return max(results, key=_ocr_quality_score)

        if last_error is not None:
            raise last_error
        raise OcrFailedError("Local OCR failed.")

    def _extract_variant(self, variant: OcrVariant, timeout_seconds: int) -> OcrResult:
        try:
            data = pytesseract.image_to_data(
                variant.image,
                output_type=Output.DICT,
                config=variant.config,
                timeout=timeout_seconds,
            )
        except TesseractNotFoundError as exc:
            raise OcrUnavailableError("Tesseract is not installed or is not on PATH.") from exc
        except RuntimeError as exc:
            if "timeout" in str(exc).lower():
                raise OcrTimeoutError("Local OCR timed out.") from exc
            raise OcrFailedError("Local OCR failed.") from exc
        except TesseractError as exc:
            raise OcrFailedError("Local OCR failed.") from exc

        lines = _parse_lines(data)
        text = "\n".join(line.text for line in lines if line.text)
        confidence = _average_confidence([line.confidence for line in lines])
        return OcrResult(engine="tesseract", text=text, confidence=confidence, lines=tuple(lines))

    def detect_orientation(self, image: Image.Image, timeout_seconds: int) -> int | None:
        if self._command_path is None:
            return None

        try:
            osd = pytesseract.image_to_osd(
                image,
                output_type=Output.DICT,
                timeout=timeout_seconds,
            )
        except OcrTimeoutError:
            raise
        except RuntimeError as exc:
            if "timeout" in str(exc).lower():
                raise OcrTimeoutError("Local orientation detection timed out.") from exc
            return None
        except Exception:
            return None

        rotate = osd.get("rotate")
        if not isinstance(rotate, int):
            return None
        if rotate not in {0, 90, 180, 270}:
            return None
        return rotate


def _parse_lines(data: dict[str, list[Any]]) -> list[OcrLine]:
    grouped = _parse_grouped_lines(data)
    if grouped:
        return grouped

    return _parse_token_lines(data)


def _parse_grouped_lines(data: dict[str, list[Any]]) -> list[OcrLine]:
    texts = data.get("text", [])
    confidences = data.get("conf", [])
    pages = data.get("page_num", [])
    blocks = data.get("block_num", [])
    paragraphs = data.get("par_num", [])
    line_numbers = data.get("line_num", [])
    value_count = min(
        len(texts),
        len(confidences),
        len(pages),
        len(blocks),
        len(paragraphs),
        len(line_numbers),
    )
    grouped_words: dict[tuple[object, object, object, object], list[tuple[str, float]]] = {}

    for index in range(value_count):
        text = str(texts[index]).strip()
        confidence = _normalize_confidence(confidences[index])
        if text:
            key = (pages[index], blocks[index], paragraphs[index], line_numbers[index])
            grouped_words.setdefault(key, []).append((text, confidence))

    return [
        OcrLine(
            text=_join_line_words([word for word, _ in words]),
            confidence=_average_confidence([confidence for _, confidence in words]),
        )
        for words in grouped_words.values()
        if words
    ]


def _parse_token_lines(data: dict[str, list[Any]]) -> list[OcrLine]:
    texts = data.get("text", [])
    confidences = data.get("conf", [])
    line_count = min(len(texts), len(confidences))
    return [
        OcrLine(text=text, confidence=confidence)
        for index in range(line_count)
        if (text := str(texts[index]).strip())
        and (confidence := _normalize_confidence(confidences[index])) >= 0.0
    ]


def _join_line_words(words: list[str]) -> str:
    if _looks_like_mrz_words(words):
        return "".join(words)
    return " ".join(words)


def _looks_like_mrz_words(words: list[str]) -> bool:
    if any("<" in word for word in words):
        return True
    combined_length = sum(len(word) for word in words)
    return combined_length >= MIN_JOINED_MRZ_LENGTH and all(
        MRZ_WORD_PATTERN.fullmatch(word) for word in words
    )


def _normalize_confidence(value: object) -> float:
    if not isinstance(value, str | SupportsFloat | SupportsIndex):
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0:
        return 0.0
    return min(numeric / 100, 1.0)


def _average_confidence(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _ocr_variants(image: Image.Image) -> tuple[OcrVariant, ...]:
    return (
        OcrVariant(
            image=to_black_and_white_text_image(
                image,
                threshold=FULL_PAGE_TEXT_BINARY_THRESHOLD,
            ),
            config=TESSERACT_CONFIG,
        ),
        OcrVariant(image=to_black_and_white_text_image(image), config=TESSERACT_CONFIG),
        OcrVariant(image=_preprocess_mrz_image(image), config=MRZ_TESSERACT_CONFIG),
        OcrVariant(image=_preprocess_mrz_band_image(image), config=MRZ_TESSERACT_CONFIG),
    )


def _preprocess_mrz_image(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(grayscale)
    scaled = contrasted.resize(
        (contrasted.width * MRZ_SCALE_FACTOR, contrasted.height * MRZ_SCALE_FACTOR),
    )
    thresholded = scaled.point(lambda pixel: 255 if pixel > MRZ_BINARY_THRESHOLD else 0)
    return thresholded.convert("RGB")


def _preprocess_mrz_band_image(image: Image.Image) -> Image.Image:
    top = int(image.height * MRZ_BAND_TOP_RATIO)
    band = image.crop((0, top, image.width, image.height))
    return _preprocess_mrz_image(band)


def _ocr_quality_score(result: OcrResult) -> float:
    return (score_passport_text(result.text) * 10.0) + result.confidence
