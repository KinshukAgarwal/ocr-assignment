from passport_ocr_api.schemas import PassportExtraction
from passport_ocr_api.services.mrz_parser import MrzPassportParser, calculate_mrz_check_digit
from passport_ocr_api.services.validation import PassportMrzValidator

SAMPLE_MRZ = "\n".join(
    [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    ]
)


def test_calculate_mrz_check_digit() -> None:
    assert calculate_mrz_check_digit("L898902C3") == "6"


def test_validation_passes_valid_td3_mrz() -> None:
    extraction = _parse_sample()

    validation = PassportMrzValidator().validate(extraction)

    assert validation.status == "passed"
    assert validation.issues == []


def test_validation_fails_bad_td3_check_digit() -> None:
    extraction = _parse_sample()
    assert extraction.mrz_line_2 is not None
    extraction.mrz_line_2 = f"{extraction.mrz_line_2[:-1]}9"

    validation = PassportMrzValidator().validate(extraction)

    assert validation.status == "failed"
    assert "composite_check_digit" in validation.issues


def test_validation_is_not_evaluated_without_mrz() -> None:
    extraction = _parse_sample()
    extraction.mrz_line_1 = None

    validation = PassportMrzValidator().validate(extraction)

    assert validation.status == "not_evaluated"
    assert validation.issues == ["mrz_missing"]


def _parse_sample() -> PassportExtraction:
    return MrzPassportParser().parse(SAMPLE_MRZ, confidence_hint=0.9).extraction
