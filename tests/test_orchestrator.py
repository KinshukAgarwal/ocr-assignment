from dataclasses import dataclass

from PIL import Image

from passport_ocr_api.config import Settings
from passport_ocr_api.errors import OcrUnavailableError
from passport_ocr_api.schemas import (
    ExtractedImage,
    PassportExtraction,
    PassportImageExtraction,
    ValidationInfo,
)
from passport_ocr_api.services.orchestrator import PassportOcrPipeline
from passport_ocr_api.services.types import OcrLine, OcrResult, OrientationResult, ParsedPassport
from passport_ocr_api.services.upload_validator import UploadedDocument

COLOR_SOURCE_PIXEL = (40, 150, 220)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class FakeLoader:
    def load_pages(self, document: UploadedDocument) -> tuple[object, ...]:
        return (
            type(
                "Page",
                (),
                {"image": Image.new("RGB", (8, 8), COLOR_SOURCE_PIXEL), "page_number": 1},
            )(),
        )


class FakeOrientation:
    def __init__(self, ocr: OcrResult) -> None:
        self.ocr = ocr

    def correct(self, image: Image.Image) -> OrientationResult:
        return OrientationResult(
            image=image,
            detected_degrees=0,
            corrected=False,
            method="test",
            local_ocr=self.ocr,
        )


class UnavailableOrientation:
    def correct(self, image: Image.Image) -> OrientationResult:
        raise OcrUnavailableError("local unavailable")


@dataclass
class FakeParser:
    confidence: float

    def parse(self, text: str, confidence_hint: float) -> ParsedPassport:
        return ParsedPassport(
            extraction=PassportExtraction(passport_number="A1234567"),
            confidence=self.confidence,
            field_confidence={"passport_number": self.confidence},
        )


class FakeValidator:
    def validate(self, extraction: PassportExtraction) -> ValidationInfo:
        return ValidationInfo(status="not_evaluated", issues=[])


class FakeImageExtractor:
    def __init__(self) -> None:
        self.received_image: Image.Image | None = None

    def extract(self, image: Image.Image) -> PassportImageExtraction:
        self.received_image = image.copy()
        return PassportImageExtraction(
            portrait=ExtractedImage(
                present=False,
                data_base64=None,
                bounding_box=None,
                confidence=0.0,
                method="test",
            ),
            signature=ExtractedImage(
                present=False,
                data_base64=None,
                bounding_box=None,
                confidence=0.0,
                method="test",
            ),
        )


class FakeCloud:
    def __init__(self) -> None:
        self.called = False
        self.received_image: Image.Image | None = None

    def extract(self, image: Image.Image, timeout_seconds: int, request_id: str) -> OcrResult:
        self.called = True
        self.received_image = image.copy()
        return OcrResult(
            engine="google_vision",
            text="cloud",
            confidence=0.99,
            lines=(OcrLine(text="cloud", confidence=0.99),),
        )


def test_pipeline_skips_google_when_confidence_is_high() -> None:
    cloud = FakeCloud()
    pipeline = PassportOcrPipeline(
        settings=Settings(google_fallback_enabled=True, low_confidence_threshold=0.7),
        document_loader=FakeLoader(),  # type: ignore[arg-type]
        orientation_corrector=FakeOrientation(_ocr_result(0.9)),  # type: ignore[arg-type]
        parser=FakeParser(confidence=0.9),
        validator=FakeValidator(),
        image_extractor=FakeImageExtractor(),
        cloud_ocr=cloud,
    )

    response = pipeline.extract(_document(), request_id="req-1")

    assert response.ocr.fallback_used is False
    assert cloud.called is False


def test_pipeline_uses_google_when_confidence_is_low() -> None:
    cloud = FakeCloud()
    pipeline = PassportOcrPipeline(
        settings=Settings(google_fallback_enabled=True, low_confidence_threshold=0.7),
        document_loader=FakeLoader(),  # type: ignore[arg-type]
        orientation_corrector=FakeOrientation(_ocr_result(0.2)),  # type: ignore[arg-type]
        parser=FakeParser(confidence=0.2),
        validator=FakeValidator(),
        image_extractor=FakeImageExtractor(),
        cloud_ocr=cloud,
    )

    response = pipeline.extract(_document(), request_id="req-1")

    assert response.ocr.fallback_used is True
    assert response.ocr.engine == "hybrid"
    assert cloud.called is True


def test_pipeline_keeps_media_color_and_sends_black_white_image_to_fallback() -> None:
    cloud = FakeCloud()
    image_extractor = FakeImageExtractor()
    pipeline = PassportOcrPipeline(
        settings=Settings(google_fallback_enabled=True, low_confidence_threshold=0.7),
        document_loader=FakeLoader(),  # type: ignore[arg-type]
        orientation_corrector=FakeOrientation(_ocr_result(0.2)),  # type: ignore[arg-type]
        parser=FakeParser(confidence=0.2),
        validator=FakeValidator(),
        image_extractor=image_extractor,
        cloud_ocr=cloud,
    )

    pipeline.extract(_document(), request_id="req-1")

    assert image_extractor.received_image is not None
    assert image_extractor.received_image.getpixel((0, 0)) == COLOR_SOURCE_PIXEL
    assert cloud.received_image is not None
    assert _unique_rgb_pixels(cloud.received_image) <= {BLACK, WHITE}


def test_pipeline_uses_google_when_local_ocr_is_unavailable() -> None:
    cloud = FakeCloud()
    pipeline = PassportOcrPipeline(
        settings=Settings(google_fallback_enabled=True, low_confidence_threshold=0.7),
        document_loader=FakeLoader(),  # type: ignore[arg-type]
        orientation_corrector=UnavailableOrientation(),  # type: ignore[arg-type]
        parser=FakeParser(confidence=0.8),
        validator=FakeValidator(),
        image_extractor=FakeImageExtractor(),
        cloud_ocr=cloud,
    )

    response = pipeline.extract(_document(), request_id="req-1")

    assert response.orientation.method == "local_orientation_unavailable"
    assert response.ocr.fallback_used is True
    assert response.ocr.engine == "google_vision"
    assert cloud.called is True


def _ocr_result(confidence: float) -> OcrResult:
    return OcrResult(
        engine="tesseract",
        text="local",
        confidence=confidence,
        lines=(OcrLine(text="local", confidence=confidence),),
    )


def _document() -> UploadedDocument:
    return UploadedDocument(filename="x.png", content_type="image/png", data=b"x")


def _unique_rgb_pixels(image: Image.Image) -> set[tuple[int, int, int]]:
    data = image.convert("RGB").tobytes()
    return {
        (data[index], data[index + 1], data[index + 2])
        for index in range(0, len(data), 3)
    }
