from dataclasses import dataclass
from enum import StrEnum


class ErrorCode(StrEnum):
    BAD_UPLOAD = "bad_upload"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    OCR_UNAVAILABLE = "ocr_unavailable"
    OCR_FAILED = "ocr_failed"
    OCR_TIMEOUT = "ocr_timeout"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass(frozen=True)
class AppError(Exception):
    code: ErrorCode
    message: str
    http_status: int


class BadUploadError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.BAD_UPLOAD, message, 400)


class UnsupportedMediaTypeError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.UNSUPPORTED_MEDIA_TYPE, message, 415)


class PayloadTooLargeError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.PAYLOAD_TOO_LARGE, message, 413)


class OcrUnavailableError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.OCR_UNAVAILABLE, message, 503)


class OcrFailedError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.OCR_FAILED, message, 502)


class OcrTimeoutError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.OCR_TIMEOUT, message, 504)


class ConfigurationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.CONFIGURATION_ERROR, message, 500)
