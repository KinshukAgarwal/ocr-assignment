from fastapi.testclient import TestClient

from passport_ocr_api.main import app

HTTP_OK = 200
HTTP_UNSUPPORTED_MEDIA_TYPE = 415


def test_healthz() -> None:
    response = TestClient(app).get("/healthz")

    assert response.status_code == HTTP_OK
    assert response.json() == {"status": "ok"}


def test_manual_tester_page_is_available() -> None:
    response = TestClient(app).get("/tester")

    assert response.status_code == HTTP_OK
    assert "Passport OCR Tester" in response.text
    assert "/v1/passports/ocr" in response.text


def test_ocr_rejects_unsupported_media_type() -> None:
    response = TestClient(app).post(
        "/v1/passports/ocr",
        files={"file": ("passport.txt", b"hello", "text/plain")},
    )

    assert response.status_code == HTTP_UNSUPPORTED_MEDIA_TYPE
    assert response.json()["code"] == "unsupported_media_type"
