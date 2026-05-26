from dataclasses import dataclass

from fastapi import UploadFile

from passport_ocr_api.config import Settings
from passport_ocr_api.errors import BadUploadError, PayloadTooLargeError, UnsupportedMediaTypeError

CHUNK_SIZE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class UploadedDocument:
    filename: str
    content_type: str
    data: bytes


async def read_upload(file: UploadFile, settings: Settings) -> UploadedDocument:
    content_type = file.content_type or ""
    if content_type not in settings.allowed_mime_types:
        raise UnsupportedMediaTypeError(f"Unsupported content type: {content_type or 'unknown'}")

    chunks: list[bytes] = []
    total_size = 0

    while True:
        chunk = await file.read(CHUNK_SIZE_BYTES)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > settings.max_upload_bytes:
            raise PayloadTooLargeError("Uploaded document is larger than the configured limit.")
        chunks.append(chunk)

    if total_size == 0:
        raise BadUploadError("Uploaded document is empty.")

    filename = file.filename or "upload"
    return UploadedDocument(filename=filename, content_type=content_type, data=b"".join(chunks))
