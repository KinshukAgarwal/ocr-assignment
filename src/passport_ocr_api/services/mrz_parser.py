import re
from datetime import date

from passport_ocr_api.schemas import PassportExtraction, Sex
from passport_ocr_api.services.types import ParsedPassport

MRZ_CHAR_PATTERN = re.compile(r"^[A-Z0-9<]+$")
MRZ_LINE_PATTERN = re.compile(r"[A-Z0-9<]{30,44}")
TD3_LINE_LENGTH = 44
MIN_MRZ_LINE_LENGTH = 30
REQUIRED_MRZ_LINES = 2
MRZ_DATE_LENGTH = 6
TWO_DIGIT_YEAR_CURRENT_CENTURY_CUTOFF = 50
COMPLETE_MRZ_SCORE = 2.0
PARTIAL_MRZ_SCORE = 1.0
MRZ_PATTERN_SCORE = 0.5


class MrzPassportParser:
    def parse(self, text: str, confidence_hint: float) -> ParsedPassport:
        lines = find_mrz_lines(text)
        if len(lines) < REQUIRED_MRZ_LINES:
            return ParsedPassport(
                extraction=PassportExtraction(),
                confidence=0.0,
                field_confidence={},
            )

        line_1, line_2 = lines[0], lines[1]
        extraction = _parse_td3_lines(line_1, line_2)
        field_confidence = _field_confidence(extraction, confidence_hint)
        confidence = _overall_confidence(field_confidence)
        return ParsedPassport(
            extraction=extraction,
            confidence=confidence,
            field_confidence=field_confidence,
        )


def find_mrz_lines(text: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in text.splitlines():
        normalized = normalize_mrz_line(raw_line)
        if _looks_like_mrz_line(normalized):
            candidates.append(normalized[:TD3_LINE_LENGTH])
    return _best_mrz_pair(candidates)


def normalize_mrz_line(value: str) -> str:
    upper = value.upper().replace(" ", "")
    return "".join(char for char in upper if char.isalnum() or char == "<")


def score_passport_text(text: str) -> float:
    lines = find_mrz_lines(text)
    if len(lines) >= REQUIRED_MRZ_LINES:
        return COMPLETE_MRZ_SCORE
    if "P<" in text.upper():
        return PARTIAL_MRZ_SCORE
    if MRZ_LINE_PATTERN.search(text.upper().replace(" ", "")):
        return MRZ_PATTERN_SCORE
    return 0.0


def _looks_like_mrz_line(value: str) -> bool:
    if len(value) < MIN_MRZ_LINE_LENGTH:
        return False
    if MRZ_CHAR_PATTERN.fullmatch(value) is None:
        return False
    return "<" in value


def _best_mrz_pair(candidates: list[str]) -> list[str]:
    if len(candidates) < REQUIRED_MRZ_LINES:
        return candidates

    for index in range(len(candidates) - 1):
        first = candidates[index]
        second = candidates[index + 1]
        if first.startswith("P<") and len(second) >= MIN_MRZ_LINE_LENGTH:
            return [_pad_mrz(first), _pad_mrz(second)]

    return [_pad_mrz(candidates[0]), _pad_mrz(candidates[1])]


def _pad_mrz(value: str) -> str:
    if len(value) >= TD3_LINE_LENGTH:
        return value[:TD3_LINE_LENGTH]
    return value.ljust(TD3_LINE_LENGTH, "<")


def _parse_td3_lines(line_1: str, line_2: str) -> PassportExtraction:
    issuing_country = _empty_to_none(line_1[2:5].replace("<", ""))
    surname, given_names = _parse_names(line_1[5:44])
    sex = _parse_sex(line_2[20:21])

    return PassportExtraction(
        passport_number=_empty_to_none(line_2[0:9].replace("<", "")),
        issuing_country=issuing_country,
        surname=surname,
        given_names=given_names,
        nationality=_empty_to_none(line_2[10:13].replace("<", "")),
        date_of_birth=_parse_date(line_2[13:19]),
        sex=sex,
        date_of_expiry=_parse_date(line_2[21:27]),
        mrz_line_1=line_1,
        mrz_line_2=line_2,
    )


def _parse_names(name_section: str) -> tuple[str | None, str | None]:
    parts = name_section.split("<<", maxsplit=1)
    surname = _clean_name(parts[0])
    given_names = _clean_name(parts[1]) if len(parts) > 1 else None
    return surname, given_names


def _clean_name(value: str) -> str | None:
    cleaned = " ".join(part for part in value.split("<") if part)
    return _empty_to_none(cleaned)


def _parse_sex(value: str) -> Sex | None:
    if value == "M":
        return Sex.MALE
    if value == "F":
        return Sex.FEMALE
    if value in {"X", "<"}:
        return Sex.UNSPECIFIED
    return None


def _parse_date(value: str) -> str | None:
    if not value.isdigit() or len(value) != MRZ_DATE_LENGTH:
        return None

    year = int(value[0:2])
    month = int(value[2:4])
    day = int(value[4:6])
    full_year = 2000 + year if year < TWO_DIGIT_YEAR_CURRENT_CENTURY_CUTOFF else 1900 + year

    try:
        return date(full_year, month, day).isoformat()
    except ValueError:
        return None


def _empty_to_none(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _field_confidence(extraction: PassportExtraction, confidence_hint: float) -> dict[str, float]:
    confidence = max(0.0, min(confidence_hint, 1.0))
    return {
        field: confidence
        for field, value in extraction.model_dump().items()
        if value is not None and not field.startswith("mrz_line")
    }


def _overall_confidence(field_confidence: dict[str, float]) -> float:
    if not field_confidence:
        return 0.0
    return sum(field_confidence.values()) / len(field_confidence)
