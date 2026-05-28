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
VISUAL_DATE_YEAR_LENGTH = 4
VISUAL_DATE_DAY_LENGTH = 2
COUNTRY_CODE_LENGTH = 3
TD3_PREFIX_LENGTH = 2
MRZ_NAME_SECTION_LENGTH = 39
MRZ_PASSPORT_NUMBER_LENGTH = 9
MRZ_OPTIONAL_DATA_LENGTH = 14
MIN_PASSPORT_NUMBER_LENGTH = 6
MAX_PASSPORT_NUMBER_LENGTH = 12
DATE_CONTEXT_LINE_LIMIT = 3
FIRST_FOLLOWING_LINE = 1
SECOND_FOLLOWING_LINE = 2
TWO_DIGIT_YEAR_CURRENT_CENTURY_CUTOFF = 50
COMPLETE_MRZ_SCORE = 2.0
PARTIAL_MRZ_SCORE = 1.0
MRZ_PATTERN_SCORE = 0.5
MRZ_CHECK_DIGIT_WEIGHTS = (7, 3, 1)
DOCUMENT_TYPE_CODES = {"P", "D", "S"}
FIELD_CONFIDENCE_FLOOR = 0.35
MISREAD_DAY_PREFIXES = {"4", "7"}
DATE_PATTERN = re.compile(
    r"(?<!\d)([0-9OQDISL]{1,2})[^0-9A-Z]{0,4}"
    r"([0-9OQDISL]{1,2})[^0-9A-Z]{0,4}((?:19|20)\d{2})(?!\d)",
)
PASSPORT_NUMBER_PATTERN = re.compile(r"\b[A-Z][A-Z0-9][0-9][A-Z0-9]{5,9}\b")
COUNTRY_CODE_PATTERN = re.compile(r"\b[A-Z01L]{3}\b")
LOCATION_PATTERN = re.compile(r"\b([A-Z]{3,}(?:\s+[A-Z]{3,})?),\s*([A-Z]{3,})\b")
LETTER_TOKEN_PATTERN = re.compile(r"[A-Z]{2,}")
DOCUMENT_NOISE_WORDS = {
    "AUTHORITY",
    "BIRTH",
    "CODE",
    "COUNTRY",
    "DATE",
    "EXPIRY",
    "GIVEN",
    "GOVERNMENT",
    "INDIA",
    "ISSUE",
    "NAMES",
    "NATIONALITY",
    "PASSPORT",
    "PLACE",
    "REPUBLIC",
    "SEX",
    "SURNAME",
    "TYPE",
}


class MrzPassportParser:
    def parse(self, text: str, confidence_hint: float) -> ParsedPassport:
        visible_extraction = _parse_visible_text(text)
        lines = find_mrz_lines(text)
        if len(lines) < REQUIRED_MRZ_LINES:
            confidence = _confidence_for_extraction(visible_extraction, confidence_hint)
            field_confidence = _field_confidence(visible_extraction, confidence)
            return ParsedPassport(
                extraction=visible_extraction,
                confidence=_overall_confidence(field_confidence),
                field_confidence=field_confidence,
            )

        line_1, line_2 = lines[0], lines[1]
        mrz_extraction = _parse_td3_lines(line_1, line_2)
        extraction = _merge_extractions(mrz_extraction, visible_extraction)
        extraction = _repair_mrz_lines_from_fields(extraction)
        confidence = _confidence_for_extraction(extraction, confidence_hint)
        field_confidence = _field_confidence(extraction, confidence)
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


def calculate_mrz_check_digit(value: str) -> str:
    total = 0
    for index, char in enumerate(value):
        weight = MRZ_CHECK_DIGIT_WEIGHTS[index % len(MRZ_CHECK_DIGIT_WEIGHTS)]
        total += _mrz_character_value(char) * weight
    return str(total % 10)


def has_valid_mrz_check_digit(value: str, check_digit: str) -> bool:
    if not check_digit.isdigit():
        return False
    return calculate_mrz_check_digit(value) == check_digit


def _looks_like_mrz_line(value: str) -> bool:
    if len(value) < MIN_MRZ_LINE_LENGTH:
        return False
    if MRZ_CHAR_PATTERN.fullmatch(value) is None:
        return False
    return "<" in value


def _mrz_character_value(char: str) -> int:
    if char == "<":
        return 0
    if char.isdigit():
        return int(char)
    if "A" <= char <= "Z":
        return ord(char) - ord("A") + 10
    return 0


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
    line_1 = _repair_td3_line_1(line_1)
    line_2 = _repair_td3_line_2(line_2)
    country_code = _normalize_country_code(line_1[2:5].replace("<", ""))
    surname, given_names = _parse_names(line_1[5:44])
    sex = _parse_sex(line_2[20:21])

    return PassportExtraction(
        type=_parse_document_type(line_1[0:1]),
        country_code=country_code,
        passport_number=_empty_to_none(line_2[0:9].replace("<", "")),
        surname=surname,
        given_names=given_names,
        nationality=_normalize_country_code(line_2[10:13].replace("<", "")),
        date_of_birth=_parse_mrz_date(line_2[13:19]),
        sex=sex,
        date_of_expiry=_parse_mrz_date(line_2[21:27]),
        mrz_line_1=line_1,
        mrz_line_2=line_2,
    )


def _repair_td3_line_1(value: str) -> str:
    repaired = value
    if repaired.startswith("<"):
        repaired = f"P{repaired}"
    elif _should_force_td3_prefix(repaired):
        repaired = f"P<{repaired[2:]}"

    repaired = _pad_mrz(repaired)
    country_code = _normalize_country_code(repaired[2:5])
    if country_code is None:
        return repaired
    return f"{repaired[0:2]}{country_code}{repaired[5:]}"


def _should_force_td3_prefix(value: str) -> bool:
    has_country_at_td3_position = (
        len(value) >= COUNTRY_CODE_LENGTH + TD3_PREFIX_LENGTH
        and value[TD3_PREFIX_LENGTH : TD3_PREFIX_LENGTH + COUNTRY_CODE_LENGTH].isalnum()
    )
    has_bad_td3_separator = value.startswith("P") and len(value) > 1 and value[1] != "<"
    return has_country_at_td3_position or has_bad_td3_separator


def _repair_td3_line_2(value: str) -> str:
    repaired = _pad_mrz(value)
    nationality = _normalize_country_code(repaired[10:13])
    if nationality is not None:
        repaired = f"{repaired[0:10]}{nationality}{repaired[13:]}"

    birth_date = _normalize_numeric_ocr(repaired[13:19])
    expiry_date = _normalize_numeric_ocr(repaired[21:27])
    return f"{repaired[0:13]}{birth_date}{repaired[19:21]}{expiry_date}{repaired[27:]}"


def _parse_document_type(value: str) -> str | None:
    normalized = value.upper().strip()
    if normalized in DOCUMENT_TYPE_CODES:
        return normalized
    return None


def _normalize_country_code(value: str) -> str | None:
    normalized = (
        value.upper()
        .replace("0", "O")
        .replace("1", "I")
        .replace("L", "I")
        .replace("<", "")
    )
    if len(normalized) != COUNTRY_CODE_LENGTH or not normalized.isalpha():
        return None
    return normalized


def _normalize_numeric_ocr(value: str) -> str:
    return (
        value.upper()
        .replace("O", "0")
        .replace("Q", "0")
        .replace("D", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
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


def _parse_mrz_date(value: str) -> str | None:
    normalized = _normalize_numeric_ocr(value)
    if not normalized.isdigit() or len(normalized) != MRZ_DATE_LENGTH:
        return None

    year = int(normalized[0:2])
    month = int(normalized[2:4])
    day = int(normalized[4:6])
    full_year = 2000 + year if year < TWO_DIGIT_YEAR_CURRENT_CENTURY_CUTOFF else 1900 + year

    try:
        return date(full_year, month, day).isoformat()
    except ValueError:
        return None


def _empty_to_none(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _parse_visible_text(text: str) -> PassportExtraction:
    lines = _clean_visible_lines(text)
    indexed_dates = _extract_indexed_dates(lines)
    country_code = _find_visible_country_code(lines)
    surname, given_names = _find_visible_names(lines)
    date_of_birth = _find_labeled_date(lines, indexed_dates, "BIRTH")
    date_of_issue = _find_labeled_date(lines, indexed_dates, "ISSUE")
    date_of_expiry = _find_labeled_date(lines, indexed_dates, "EXPIR")
    ordered_dates = [value for _, value in indexed_dates]

    if date_of_birth is None and ordered_dates:
        date_of_birth = ordered_dates[0]
    if date_of_issue is None and len(ordered_dates) >= SECOND_FOLLOWING_LINE:
        date_of_issue = ordered_dates[1]
    if date_of_expiry is None and len(ordered_dates) >= DATE_CONTEXT_LINE_LIMIT:
        date_of_expiry = ordered_dates[2]

    return PassportExtraction(
        type=_find_visible_document_type(lines),
        country_code=country_code,
        passport_number=_find_visible_passport_number(lines),
        surname=surname,
        given_names=given_names,
        nationality=_find_visible_nationality(lines),
        date_of_birth=date_of_birth,
        sex=_find_visible_sex(lines),
        place_of_birth=_find_labeled_location(lines, "BIRTH") or _find_first_location(lines),
        date_of_issue=date_of_issue,
        date_of_expiry=date_of_expiry,
    )


def _clean_visible_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _normalize_visible_line(raw_line)
        if line and not _looks_like_mrz_line(normalize_mrz_line(line)):
            lines.append(line)
    return lines


def _normalize_visible_line(value: str) -> str:
    upper = value.upper().replace("|", " ")
    return re.sub(r"\s+", " ", upper).strip()


def _find_visible_document_type(lines: list[str]) -> str | None:
    for line in lines:
        if "TYPE" not in line and "PASSPORT" not in line:
            continue
        tokens = _line_tokens(line)
        for token in tokens:
            if token in DOCUMENT_TYPE_CODES:
                return token
    return None


def _find_visible_country_code(lines: list[str]) -> str | None:
    for line in lines:
        if "INDIAN" in line and "IND" in line:
            return "IND"
        if not _is_country_code_context(line):
            continue
        for match in COUNTRY_CODE_PATTERN.finditer(line):
            country_code = _normalize_country_code(match.group(0))
            if country_code is not None and country_code not in DOCUMENT_NOISE_WORDS:
                return country_code
    return None


def _is_country_code_context(line: str) -> bool:
    return "CODE" in line or "NATIONALITY" in line or "TYPE" in line


def _find_visible_passport_number(lines: list[str]) -> str | None:
    for line in lines:
        for match in PASSPORT_NUMBER_PATTERN.finditer(line):
            passport_number = match.group(0)
            if MIN_PASSPORT_NUMBER_LENGTH <= len(passport_number) <= MAX_PASSPORT_NUMBER_LENGTH:
                return passport_number
    return None


def _find_visible_names(lines: list[str]) -> tuple[str | None, str | None]:
    candidates = _name_candidates(lines)
    given = _find_name_after_label(lines, "GIVEN")
    surname = _find_name_after_label(lines, "SURNAME")

    if surname is None and given is not None:
        surname = _candidate_before(candidates, given)
    if surname is None and given is None and candidates:
        surname = candidates[0]
    if given is None:
        given = _first_candidate_after(candidates, surname)
    return surname, given


def _find_visible_nationality(lines: list[str]) -> str | None:
    for line in lines:
        if "INDIAN" in line:
            return "INDIAN"
        if "NATIONALITY" in line:
            value = _value_after_label(line, "NATIONALITY")
            if value is not None:
                return value
    return None


def _find_visible_sex(lines: list[str]) -> Sex | None:
    for line in lines:
        if _is_date_or_birth_context(line):
            for token in _line_tokens(line):
                parsed = _parse_sex(token)
                if parsed is not None:
                    return parsed
    return None


def _find_labeled_date(
    lines: list[str],
    indexed_dates: list[tuple[int, str]],
    label: str,
) -> str | None:
    for index, line in enumerate(lines):
        if label not in line:
            continue
        date = _first_date_in_context(index, indexed_dates)
        if date is not None:
            return date
    return None


def _first_date_in_context(
    label_index: int,
    indexed_dates: list[tuple[int, str]],
) -> str | None:
    for index, value in indexed_dates:
        if label_index <= index <= label_index + DATE_CONTEXT_LINE_LIMIT:
            return value
    return None


def _extract_indexed_dates(lines: list[str]) -> list[tuple[int, str]]:
    dates: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        for value in _extract_dates_from_line(line):
            dates.append((index, value))
    return dates


def _extract_dates_from_line(line: str) -> list[str]:
    dates: list[str] = []
    for match in DATE_PATTERN.finditer(line):
        parsed = _parse_visual_date(match.group(1), match.group(2), match.group(3))
        if parsed is not None:
            dates.append(parsed)
    return dates


def _parse_visual_date(day_value: str, month_value: str, year_value: str) -> str | None:
    day = _normalize_numeric_ocr(day_value)
    month = _normalize_numeric_ocr(month_value)
    year = _normalize_numeric_ocr(year_value)
    parsed = _date_from_parts(day, month, year)
    if parsed is not None:
        return parsed
    if len(day) == VISUAL_DATE_DAY_LENGTH and day[0] in MISREAD_DAY_PREFIXES:
        return _date_from_parts(f"1{day[1]}", month, year)
    return None


def _date_from_parts(day_value: str, month_value: str, year_value: str) -> str | None:
    if not (day_value.isdigit() and month_value.isdigit() and year_value.isdigit()):
        return None
    if len(year_value) != VISUAL_DATE_YEAR_LENGTH:
        return None
    try:
        return date(int(year_value), int(month_value), int(day_value)).isoformat()
    except ValueError:
        return None


def _find_labeled_location(lines: list[str], label: str) -> str | None:
    for index, line in enumerate(lines):
        if label not in line or "PLACE" not in line:
            continue
        for candidate in lines[
            index + FIRST_FOLLOWING_LINE : index + DATE_CONTEXT_LINE_LIMIT + 1
        ]:
            location = _extract_location(candidate)
            if location is not None:
                return location
    return None


def _find_first_location(lines: list[str]) -> str | None:
    for line in lines:
        location = _extract_comma_location(line)
        if location is not None:
            return location
    return None


def _extract_location(line: str) -> str | None:
    if any(char.isdigit() for char in line):
        return None
    comma_location = _extract_comma_location(line)
    if comma_location is not None:
        return comma_location
    words = [
        word
        for word in _line_tokens(line)
        if word.isalpha() and word not in DOCUMENT_NOISE_WORDS and len(word) >= COUNTRY_CODE_LENGTH
    ]
    if words:
        return " ".join(words)
    return None


def _extract_comma_location(line: str) -> str | None:
    match = LOCATION_PATTERN.search(line)
    if match is not None:
        return f"{match.group(1).strip()}, {match.group(2).strip()}"
    return None


def _value_after_label(
    line: str,
    label: str,
) -> str | None:
    position = line.find(label)
    if position < 0:
        return None
    return _clean_free_text(line[position + len(label) :])


def _clean_free_text(value: str) -> str | None:
    words = [word for word in _line_tokens(value) if word not in DOCUMENT_NOISE_WORDS]
    if not words:
        return None
    return " ".join(words)


def _name_candidates(lines: list[str]) -> list[str]:
    candidates: list[str] = []
    for line in lines:
        candidate = _name_candidate(line)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _name_candidate(line: str) -> str | None:
    if _is_non_name_line(line):
        return None
    words = [
        word
        for word in LETTER_TOKEN_PATTERN.findall(line)
        if word not in DOCUMENT_NOISE_WORDS and len(word) > 1
    ]
    if not words:
        return None
    return " ".join(words)


def _is_non_name_line(line: str) -> bool:
    if any(char.isdigit() for char in line):
        return True
    if any(word in line for word in DOCUMENT_NOISE_WORDS):
        return True
    return "," in line or "/" in line


def _find_name_after_label(lines: list[str], label: str) -> str | None:
    for index, line in enumerate(lines):
        if label not in line:
            continue
        for candidate in lines[index + FIRST_FOLLOWING_LINE : index + DATE_CONTEXT_LINE_LIMIT + 1]:
            name = _name_candidate(candidate)
            if name is not None:
                return name
    return None


def _candidate_before(candidates: list[str], value: str) -> str | None:
    try:
        index = candidates.index(value)
    except ValueError:
        return None
    if index == 0:
        return None
    return candidates[index - FIRST_FOLLOWING_LINE]


def _first_candidate_after(candidates: list[str], value: str | None) -> str | None:
    if not candidates:
        return None
    if value is None:
        return candidates[0]
    try:
        index = candidates.index(value)
    except ValueError:
        return candidates[0]
    next_index = index + FIRST_FOLLOWING_LINE
    if next_index >= len(candidates):
        return None
    return candidates[next_index]


def _is_date_or_birth_context(line: str) -> bool:
    return "BIRTH" in line or bool(_extract_dates_from_line(line))


def _line_tokens(line: str) -> list[str]:
    return re.findall(r"[A-Z0-9]+", line)


def _merge_extractions(
    mrz_extraction: PassportExtraction,
    visible_extraction: PassportExtraction,
) -> PassportExtraction:
    merged = mrz_extraction.model_copy()
    visible_data = visible_extraction.model_dump()
    for field, visible_value in visible_data.items():
        if visible_value is not None and not field.startswith("mrz_line"):
            setattr(merged, field, visible_value)
    return merged


def _repair_mrz_lines_from_fields(extraction: PassportExtraction) -> PassportExtraction:
    repaired = extraction.model_copy()
    if repaired.mrz_line_1 is not None:
        repaired.mrz_line_1 = _repair_returned_mrz_line_1(repaired)
    if repaired.mrz_line_2 is not None:
        repaired.mrz_line_2 = _repair_returned_mrz_line_2(repaired)
    return repaired


def _repair_returned_mrz_line_1(extraction: PassportExtraction) -> str:
    line = _pad_mrz(extraction.mrz_line_1 or "")
    document_type = _parse_document_type(extraction.type or line[0:1]) or "P"
    country_code = _mrz_country_code(extraction.country_code)
    if country_code is None:
        country_code = _normalize_country_code(line[2:5]) or line[2:5]

    name_section = _format_mrz_name_section(extraction.surname, extraction.given_names)
    if name_section is None:
        name_section = line[5:TD3_LINE_LENGTH]
    return _pad_mrz(f"{document_type}<{country_code}{name_section}")


def _repair_returned_mrz_line_2(extraction: PassportExtraction) -> str:
    line = _pad_mrz(extraction.mrz_line_2 or "")
    passport_number = _format_mrz_value(
        extraction.passport_number,
        MRZ_PASSPORT_NUMBER_LENGTH,
    )
    if passport_number is None:
        passport_number = _format_mrz_value(line[0:9], MRZ_PASSPORT_NUMBER_LENGTH) or line[0:9]
    passport_check = calculate_mrz_check_digit(passport_number)

    nationality = _mrz_country_code(extraction.nationality)
    if nationality is None:
        nationality = _mrz_country_code(extraction.country_code)
    if nationality is None:
        nationality = _normalize_country_code(line[10:13]) or line[10:13]

    birth_date = _format_mrz_date(extraction.date_of_birth) or _normalize_numeric_ocr(line[13:19])
    birth_date = _pad_mrz(birth_date)[:MRZ_DATE_LENGTH]
    birth_check = calculate_mrz_check_digit(birth_date)

    sex = _format_mrz_sex(extraction.sex) or line[20:21]

    expiry_date = _format_mrz_date(extraction.date_of_expiry) or _normalize_numeric_ocr(line[21:27])
    expiry_date = _pad_mrz(expiry_date)[:MRZ_DATE_LENGTH]
    expiry_check = calculate_mrz_check_digit(expiry_date)

    optional_data = _format_mrz_value(line[28:42], MRZ_OPTIONAL_DATA_LENGTH)
    if optional_data is None:
        optional_data = "<" * MRZ_OPTIONAL_DATA_LENGTH
    optional_check = calculate_mrz_check_digit(optional_data)

    composite_value = (
        passport_number
        + passport_check
        + birth_date
        + birth_check
        + expiry_date
        + expiry_check
        + optional_data
        + optional_check
    )
    composite_check = calculate_mrz_check_digit(composite_value)
    return (
        passport_number
        + passport_check
        + nationality
        + birth_date
        + birth_check
        + sex
        + expiry_date
        + expiry_check
        + optional_data
        + optional_check
        + composite_check
    )


def _mrz_country_code(value: str | None) -> str | None:
    if value is None:
        return None
    return _normalize_country_code(value)


def _format_mrz_name_section(surname: str | None, given_names: str | None) -> str | None:
    surname_value = _format_mrz_name_value(surname)
    given_value = _format_mrz_name_value(given_names)
    if surname_value is None and given_value is None:
        return None
    name_section = f"{surname_value or ''}<<{given_value or ''}"
    return _pad_mrz(name_section)[:MRZ_NAME_SECTION_LENGTH]


def _format_mrz_name_value(value: str | None) -> str | None:
    if value is None:
        return None
    words = re.findall(r"[A-Z]+", value.upper())
    if not words:
        return None
    return "<".join(words)


def _format_mrz_value(value: str | None, length: int) -> str | None:
    if value is None:
        return None
    normalized = normalize_mrz_line(value)
    if not normalized:
        return None
    return normalized[:length].ljust(length, "<")


def _format_mrz_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return f"{parsed.year % 100:02d}{parsed.month:02d}{parsed.day:02d}"


def _format_mrz_sex(value: Sex | None) -> str | None:
    if value is None:
        return None
    return value.value


def _confidence_for_extraction(extraction: PassportExtraction, confidence_hint: float) -> float:
    if not _has_extracted_fields(extraction):
        return 0.0
    return max(FIELD_CONFIDENCE_FLOOR, min(confidence_hint, 1.0))


def _has_extracted_fields(extraction: PassportExtraction) -> bool:
    return any(
        value is not None
        for field, value in extraction.model_dump().items()
        if not field.startswith("mrz_line")
    )


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
