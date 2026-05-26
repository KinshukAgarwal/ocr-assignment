import shutil
from typing import Any, SupportsFloat, SupportsIndex

import pytesseract
from PIL import Image
from pytesseract import Output, TesseractError, TesseractNotFoundError

from passport_ocr_api.errors import OcrFailedError, OcrTimeoutError, OcrUnavailableError
from passport_ocr_api.services.types import OcrLine, OcrResult

TESSERACT_CONFIG = "--oem 3 --psm 6"


class TesseractOcrEngine:
    def __init__(self, command_path: str | None = None) -> None:
        self._command_path = command_path or shutil.which("tesseract")
        if self._command_path is not None:
            pytesseract.pytesseract.tesseract_cmd = self._command_path

    def extract(self, image: Image.Image, timeout_seconds: int) -> OcrResult:
        if self._command_path is None:
            raise OcrUnavailableError("Tesseract is not installed or is not on PATH.")

        try:
            data = pytesseract.image_to_data(
                image,
                output_type=Output.DICT,
                config=TESSERACT_CONFIG,
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
    texts = data.get("text", [])
    confidences = data.get("conf", [])
    line_count = min(len(texts), len(confidences))
    lines: list[OcrLine] = []

    for index in range(line_count):
        text = str(texts[index]).strip()
        confidence = _normalize_confidence(confidences[index])
        if text:
            lines.append(OcrLine(text=text, confidence=confidence))

    return lines


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
