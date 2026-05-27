from dataclasses import dataclass
from typing import Literal, Protocol

from PIL import Image

from passport_ocr_api.schemas import PassportExtraction, PassportImageExtraction, ValidationInfo

RotationDegrees = Literal[0, 90, 180, 270]
OcrEngineName = Literal["tesseract", "google_vision", "hybrid"]


@dataclass(frozen=True)
class RenderedPage:
    image: Image.Image
    page_number: int


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float


@dataclass(frozen=True)
class OcrResult:
    engine: OcrEngineName
    text: str
    confidence: float
    lines: tuple[OcrLine, ...]


@dataclass(frozen=True)
class OrientationResult:
    image: Image.Image
    detected_degrees: RotationDegrees
    corrected: bool
    method: str
    local_ocr: OcrResult


@dataclass(frozen=True)
class ParsedPassport:
    extraction: PassportExtraction
    confidence: float
    field_confidence: dict[str, float]


class OcrEngine(Protocol):
    def extract(self, image: Image.Image, timeout_seconds: int) -> OcrResult:
        raise NotImplementedError


class CloudOcrEngine(Protocol):
    def extract(self, image: Image.Image, timeout_seconds: int, request_id: str) -> OcrResult:
        raise NotImplementedError


class PassportParser(Protocol):
    def parse(self, text: str, confidence_hint: float) -> ParsedPassport:
        raise NotImplementedError


class PassportValidator(Protocol):
    def validate(self, extraction: PassportExtraction) -> ValidationInfo:
        raise NotImplementedError


class PassportImageExtractor(Protocol):
    def extract(self, image: Image.Image) -> PassportImageExtraction:
        raise NotImplementedError
