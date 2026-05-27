import pytest

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

    assert parsed.extraction.type == "P"
    assert parsed.extraction.country_code == "UTO"
    assert parsed.extraction.passport_number == "L898902C3"
    assert parsed.extraction.issuing_country == "UTO"
    assert parsed.extraction.surname == "ERIKSSON"
    assert parsed.extraction.given_names == "ANNA MARIA"
    assert parsed.extraction.nationality == "UTO"
    assert parsed.extraction.date_of_birth == "1974-08-12"
    assert parsed.extraction.sex == "F"
    assert parsed.extraction.date_of_expiry == "2012-04-15"
    assert parsed.confidence == pytest.approx(HIGH_CONFIDENCE)


def test_parser_merges_visible_common_passport_fields() -> None:
    text = "\n".join(
        [
            "REPUBLIC OF UTOPIA",
            "Type P Code UTO Nationality UTOPIAN",
            "Passport No L898902C3",
            "Surname",
            "ERIKSSON",
            "Given Name(s)",
            "ANNA MARIA",
            "Date of Birth / Sex",
            "12/08/1974 F",
            "Place of Birth",
            "SAMPLE CITY",
            "Date of Issue",
            "01/01/2020",
            "Date of Expiry",
            "15/04/2012",
            SAMPLE_MRZ,
        ],
    )

    parsed = MrzPassportParser().parse(text, confidence_hint=HIGH_CONFIDENCE)

    assert parsed.extraction.type == "P"
    assert parsed.extraction.country_code == "UTO"
    assert parsed.extraction.passport_number == "L898902C3"
    assert parsed.extraction.surname == "ERIKSSON"
    assert parsed.extraction.given_names == "ANNA MARIA"
    assert parsed.extraction.nationality == "UTOPIAN"
    assert parsed.extraction.date_of_birth == "1974-08-12"
    assert parsed.extraction.sex == "F"
    assert parsed.extraction.place_of_birth == "SAMPLE CITY"
    assert parsed.extraction.date_of_issue == "2020-01-01"
    assert parsed.extraction.date_of_expiry == "2012-04-15"


def test_parser_repairs_returned_mrz_lines_from_visible_fields() -> None:
    text = "\n".join(
        [
            "Type P Code UTO Nationality UTOPIAN",
            "Passport No L898902C3",
            "Surname",
            "ERIKSSON",
            "Given Name(s)",
            "ANNA MARIA",
            "Date of Birth / Sex",
            "12/08/1974 F",
            "Date of Expiry",
            "15/04/2012",
            "P<UTOERIKSS0N<<ANNA<MAR1A<<<<<<<<<<<<<<<<<<<",
            "L898902C30UTO7408120F1204150ZE184226B<<<<<00",
        ],
    )

    parsed = MrzPassportParser().parse(text, confidence_hint=HIGH_CONFIDENCE)

    assert parsed.extraction.mrz_line_1 == "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    assert parsed.extraction.mrz_line_2 == "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def test_parser_handles_missing_mrz() -> None:
    parsed = MrzPassportParser().parse("not a passport", confidence_hint=0.9)

    assert parsed.extraction.passport_number is None
    assert parsed.confidence == 0.0
    assert parsed.field_confidence == {}


def test_score_passport_text_prefers_complete_mrz() -> None:
    assert score_passport_text(SAMPLE_MRZ) == COMPLETE_MRZ_SCORE
    assert score_passport_text("P<UTO") == PARTIAL_MRZ_SCORE
    assert score_passport_text("plain text") == 0.0
