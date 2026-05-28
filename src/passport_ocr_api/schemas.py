from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

ConfidenceValue = Annotated[float, Field(ge=0.0, le=1.0)]


class Sex(StrEnum):
    MALE = "M"
    FEMALE = "F"
    UNSPECIFIED = "X"


class OrientationInfo(BaseModel):
    detected_degrees: Literal[0, 90, 180, 270]
    corrected: bool
    method: str


class PassportExtraction(BaseModel):
    type: str | None = None
    country_code: str | None = None
    passport_number: str | None = None
    surname: str | None = None
    given_names: str | None = None
    nationality: str | None = None
    date_of_birth: str | None = None
    sex: Sex | None = None
    place_of_birth: str | None = None
    date_of_issue: str | None = None
    date_of_expiry: str | None = None
    mrz_line_1: str | None = None
    mrz_line_2: str | None = None


class ConfidenceInfo(BaseModel):
    overall: ConfidenceValue
    fields: dict[str, ConfidenceValue] = Field(default_factory=dict)


class ValidationInfo(BaseModel):
    status: Literal["passed", "failed", "not_evaluated"]
    issues: list[str] = Field(default_factory=list)


class OcrInfo(BaseModel):
    engine: Literal["tesseract", "google_vision", "hybrid"]
    fallback_used: bool
    raw_text_snippets: list[str] = Field(default_factory=list)


class BoundingBox(BaseModel):
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ExtractedImage(BaseModel):
    present: bool
    content_type: Literal["image/jpeg"] = "image/jpeg"
    data_base64: str | None = None
    bounding_box: BoundingBox | None = None
    confidence: ConfidenceValue
    method: str


class PassportImageExtraction(BaseModel):
    portrait: ExtractedImage
    signature: ExtractedImage


class PassportOcrResponse(BaseModel):
    request_id: str
    document_type: Literal["passport"] = "passport"
    orientation: OrientationInfo
    extraction: PassportExtraction
    images: PassportImageExtraction
    confidence: ConfidenceInfo
    validation: ValidationInfo
    ocr: OcrInfo


class ErrorResponse(BaseModel):
    request_id: str
    code: str
    message: str
