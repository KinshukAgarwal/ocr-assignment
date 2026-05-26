import anyio
import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from passport_ocr_api.config import Settings
from passport_ocr_api.errors import PayloadTooLargeError, UnsupportedMediaTypeError
from passport_ocr_api.services.upload_validator import read_upload


def make_upload(content: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        filename="passport.png",
        file=__import__("io").BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )


def test_read_upload_accepts_allowed_type() -> None:
    settings = Settings(max_upload_bytes=10)

    uploaded = anyio.run(read_upload, make_upload(b"123", "image/png"), settings)

    assert uploaded.filename == "passport.png"
    assert uploaded.content_type == "image/png"
    assert uploaded.data == b"123"


def test_read_upload_rejects_unsupported_type() -> None:
    settings = Settings()

    with pytest.raises(UnsupportedMediaTypeError):
        anyio.run(read_upload, make_upload(b"123", "text/plain"), settings)


def test_read_upload_rejects_large_payload() -> None:
    settings = Settings(max_upload_bytes=2)

    with pytest.raises(PayloadTooLargeError):
        anyio.run(read_upload, make_upload(b"123", "image/png"), settings)
