from typing import Any

import pytest
from PIL import Image

from passport_ocr_api.services import local_tesseract
from passport_ocr_api.services.local_tesseract import TesseractOcrEngine

EXPECTED_AVERAGE_CONFIDENCE = 0.85
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def test_parse_lines_groups_tesseract_words_by_line() -> None:
    data: dict[str, list[Any]] = {
        "text": ["P<UTO", "ERIKSSON<<ANNA", "", "L898902C36", "UTO7408122F1204159"],
        "conf": ["80", "90", "-1", "70", "75"],
        "page_num": [1, 1, 1, 1, 1],
        "block_num": [1, 1, 1, 1, 1],
        "par_num": [1, 1, 1, 1, 1],
        "line_num": [1, 1, 1, 2, 2],
    }

    lines = local_tesseract._parse_lines(data)

    assert [line.text for line in lines] == [
        "P<UTOERIKSSON<<ANNA",
        "L898902C36UTO7408122F1204159",
    ]
    assert lines[0].confidence == pytest.approx(EXPECTED_AVERAGE_CONFIDENCE)


def test_extract_prefers_mrz_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_image_to_data(*args: object, **kwargs: object) -> dict[str, list[Any]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _tesseract_data(["REPUBLIC OF EXAMPLE"], ["95"])
        return _tesseract_data(
            [
                "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
                "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
            ],
            ["70", "72"],
        )

    monkeypatch.setattr(
        "passport_ocr_api.services.local_tesseract.pytesseract.image_to_data",
        fake_image_to_data,
    )

    result = TesseractOcrEngine(command_path="/usr/bin/tesseract").extract(
        Image.new("RGB", (16, 16), "white"),
        timeout_seconds=10,
    )

    assert "P<UTOERIKSSON" in result.text


def test_extract_uses_black_and_white_full_image_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_images: list[Image.Image] = []

    def fake_image_to_data(
        image: Image.Image,
        *args: object,
        **kwargs: object,
    ) -> dict[str, list[Any]]:
        captured_images.append(image.copy())
        return _tesseract_data(["REPUBLIC OF EXAMPLE"], ["95"])

    monkeypatch.setattr(
        "passport_ocr_api.services.local_tesseract.pytesseract.image_to_data",
        fake_image_to_data,
    )

    image = Image.new("RGB", (4, 4), (245, 235, 210))
    image.putpixel((0, 0), (25, 45, 70))

    TesseractOcrEngine(command_path="/usr/bin/tesseract").extract(image, timeout_seconds=10)

    assert captured_images
    assert _unique_rgb_pixels(captured_images[0]) <= {BLACK, WHITE}


def _tesseract_data(lines: list[str], confidences: list[str]) -> dict[str, list[Any]]:
    return {
        "text": lines,
        "conf": confidences,
        "page_num": [1] * len(lines),
        "block_num": [1] * len(lines),
        "par_num": [1] * len(lines),
        "line_num": list(range(1, len(lines) + 1)),
    }


def _unique_rgb_pixels(image: Image.Image) -> set[tuple[int, int, int]]:
    data = image.convert("RGB").tobytes()
    return {
        (data[index], data[index + 1], data[index + 2])
        for index in range(0, len(data), 3)
    }
