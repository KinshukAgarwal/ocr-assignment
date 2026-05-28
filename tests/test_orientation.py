from typing import cast

from PIL import Image
from pytest import MonkeyPatch

from passport_ocr_api.config import Settings
from passport_ocr_api.services import orientation as orientation_module
from passport_ocr_api.services.orientation import OrientationCorrector
from passport_ocr_api.services.types import OcrLine, OcrResult, RotationDegrees

EXPECTED_CORRECT_ROTATION: RotationDegrees = 90


def test_orientation_prefers_passport_structure_over_raw_confidence(
    monkeypatch: MonkeyPatch,
) -> None:
    def fake_rotate(image: Image.Image, degrees: RotationDegrees) -> Image.Image:
        rotated = image.copy()
        rotated.info["rotation"] = degrees
        return rotated

    monkeypatch.setattr(orientation_module, "_rotate", fake_rotate)
    fake_ocr = FakeOcr(
        {
            0: _ocr("high confidence document noise", 0.99),
            90: _ocr(
                "\n".join(
                    [
                        "PASSPORT",
                        "SURNAME ERIKSSON",
                        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
                        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
                    ],
                ),
                0.58,
            ),
            180: _ocr("DATE DATE DATE", 0.72),
            270: _ocr("plain rotated text", 0.80),
        },
    )

    corrector = OrientationCorrector(fake_ocr, Settings())  # type: ignore[arg-type]
    result = corrector.correct(Image.new("RGB", (300, 200)))

    assert result.detected_degrees == EXPECTED_CORRECT_ROTATION
    assert result.corrected is True
    assert result.method == "rotation_scoring"


class FakeOcr:
    def __init__(self, results: dict[RotationDegrees, OcrResult]) -> None:
        self._results = results

    def detect_orientation(self, _image: Image.Image, _timeout_seconds: int) -> int | None:
        return 0

    def extract(self, image: Image.Image, _timeout_seconds: int) -> OcrResult:
        rotation = cast(RotationDegrees, image.info.get("rotation", 0))
        assert rotation in self._results
        return self._results[rotation]


def _ocr(text: str, confidence: float) -> OcrResult:
    return OcrResult(
        engine="tesseract",
        text=text,
        confidence=confidence,
        lines=tuple(OcrLine(text=line, confidence=confidence) for line in text.splitlines()),
    )
