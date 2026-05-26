from passport_ocr_api.services.mrz_parser import (
    COMPLETE_MRZ_SCORE,
    PARTIAL_MRZ_SCORE,
    MrzPassportParser,
    find_mrz_lines,
    score_passport_text,
)

HIGH_CONFIDENCE = 0.91


SAMPLE_MRZ = "\n".join(
    [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    ]
)


def test_find_mrz_lines_returns_td3_pair() -> None:
    lines = find_mrz_lines(f"noise\n{SAMPLE_MRZ}\nmore noise")

    assert lines == [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    ]


def test_parser_extracts_icao_fields() -> None:
    parsed = MrzPassportParser().parse(SAMPLE_MRZ, confidence_hint=HIGH_CONFIDENCE)

    assert parsed.extraction.passport_number == "L898902C3"
    assert parsed.extraction.issuing_country == "UTO"
    assert parsed.extraction.surname == "ERIKSSON"
    assert parsed.extraction.given_names == "ANNA MARIA"
    assert parsed.extraction.nationality == "UTO"
    assert parsed.extraction.date_of_birth == "1974-08-12"
    assert parsed.extraction.sex == "F"
    assert parsed.extraction.date_of_expiry == "2012-04-15"
    assert parsed.confidence == HIGH_CONFIDENCE


def test_parser_handles_missing_mrz() -> None:
    parsed = MrzPassportParser().parse("not a passport", confidence_hint=0.9)

    assert parsed.extraction.passport_number is None
    assert parsed.confidence == 0.0
    assert parsed.field_confidence == {}


def test_score_passport_text_prefers_complete_mrz() -> None:
    assert score_passport_text(SAMPLE_MRZ) == COMPLETE_MRZ_SCORE
    assert score_passport_text("P<UTO") == PARTIAL_MRZ_SCORE
    assert score_passport_text("plain text") == 0.0
