from passport_ocr_api.schemas import PassportExtraction, ValidationInfo
from passport_ocr_api.services.mrz_parser import TD3_LINE_LENGTH, has_valid_mrz_check_digit

PASSPORT_NUMBER_FIELD = "passport_number_check_digit"
BIRTH_DATE_FIELD = "date_of_birth_check_digit"
EXPIRY_DATE_FIELD = "date_of_expiry_check_digit"
PERSONAL_NUMBER_FIELD = "personal_number_check_digit"
COMPOSITE_FIELD = "composite_check_digit"


class PassportMrzValidator:
    def validate(self, extraction: PassportExtraction) -> ValidationInfo:
        if extraction.mrz_line_1 is None or extraction.mrz_line_2 is None:
            return ValidationInfo(status="not_evaluated", issues=["mrz_missing"])

        line_1 = extraction.mrz_line_1
        line_2 = extraction.mrz_line_2
        issues = _validate_td3_lines(line_1, line_2)
        if issues:
            return ValidationInfo(status="failed", issues=issues)
        return ValidationInfo(status="passed", issues=[])


def _validate_td3_lines(line_1: str, line_2: str) -> list[str]:
    issues: list[str] = []
    if len(line_1) != TD3_LINE_LENGTH or len(line_2) != TD3_LINE_LENGTH:
        return ["mrz_invalid_length"]

    checks = (
        (line_2[0:9], line_2[9], PASSPORT_NUMBER_FIELD),
        (line_2[13:19], line_2[19], BIRTH_DATE_FIELD),
        (line_2[21:27], line_2[27], EXPIRY_DATE_FIELD),
        (line_2[28:42], line_2[42], PERSONAL_NUMBER_FIELD),
        (line_2[0:10] + line_2[13:20] + line_2[21:43], line_2[43], COMPOSITE_FIELD),
    )

    for value, check_digit, issue in checks:
        if not has_valid_mrz_check_digit(value, check_digit):
            issues.append(issue)

    return issues
