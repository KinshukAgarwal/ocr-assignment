import shutil
from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from passport_ocr_api.config import Settings
from passport_ocr_api.main import app
from passport_ocr_api.services.dependencies import build_pipeline
from passport_ocr_api.services.upload_validator import UploadedDocument

HTTP_OK = 200
IMAGE_WIDTH = 1700
IMAGE_HEIGHT = 450
MRZ_LEFT = 80
MRZ_TOP = 260
MRZ_FONT_SIZE = 34
MRZ_LINE_SPACING = 12
LOCAL_OCR_TIMEOUT_SECONDS = 10
PASSPORT_NUMBER = "L898902C3"
SURNAME = "ERIKSSON"
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
SAMPLE_MRZ = "\n".join(
    [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    ]
)


pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None or not FONT_PATH.exists(),
    reason="Tesseract and DejaVu mono font are required for OCR integration tests.",
)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_pipeline_extracts_passport_mrz_from_rotated_png(
    rotation: Literal[0, 90, 180, 270],
) -> None:
    uploaded = UploadedDocument(
        filename="synthetic-passport.png",
        content_type="image/png",
        data=_make_png_bytes(rotation),
    )

    response = build_pipeline(_integration_settings()).extract(uploaded, request_id="test-request")

    assert response.extraction.passport_number == PASSPORT_NUMBER
    assert response.extraction.surname == SURNAME
    assert response.validation.status == "passed"
    assert response.orientation.detected_degrees == rotation
    assert response.orientation.corrected is (rotation != 0)


def test_pipeline_extracts_passport_mrz_from_pdf() -> None:
    uploaded = UploadedDocument(
        filename="synthetic-passport.pdf",
        content_type="application/pdf",
        data=_make_pdf_bytes(),
    )

    response = build_pipeline(_integration_settings()).extract(uploaded, request_id="test-request")

    assert response.extraction.passport_number == PASSPORT_NUMBER
    assert response.extraction.surname == SURNAME
    assert response.validation.status == "passed"


def test_api_extracts_passport_mrz_from_png_upload() -> None:
    response = TestClient(app).post(
        "/v1/passports/ocr",
        files={
            "file": (
                "synthetic-passport.png",
                _make_png_bytes(rotation=0),
                "image/png",
            )
        },
    )

    body = response.json()

    assert response.status_code == HTTP_OK
    assert "issuing_country" not in body["extraction"]
    assert body["extraction"]["passport_number"] == PASSPORT_NUMBER
    assert body["extraction"]["surname"] == SURNAME
    assert body["validation"]["status"] == "passed"


def _integration_settings() -> Settings:
    return Settings(
        local_ocr_timeout_seconds=LOCAL_OCR_TIMEOUT_SECONDS,
        google_fallback_enabled=False,
    )


def _make_png_bytes(rotation: Literal[0, 90, 180, 270]) -> bytes:
    image = _make_passport_image()
    if rotation != 0:
        image = image.rotate(rotation, expand=True)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_pdf_bytes() -> bytes:
    buffer = BytesIO()
    _make_passport_image().save(buffer, format="PDF", resolution=200)
    return buffer.getvalue()


def _make_passport_image() -> Image.Image:
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
    font = ImageFont.truetype(str(FONT_PATH), MRZ_FONT_SIZE)
    draw = ImageDraw.Draw(image)
    draw.text(
        (MRZ_LEFT, MRZ_TOP),
        SAMPLE_MRZ,
        fill="black",
        font=font,
        spacing=MRZ_LINE_SPACING,
    )
    return image
