from typing import Any

import httpx
import pytest
from PIL import Image

from passport_ocr_api.config import Settings
from passport_ocr_api.errors import OcrFailedError, OcrTimeoutError, OcrUnavailableError
from passport_ocr_api.services.google_vision import GoogleVisionOcrEngine, _build_vision_request

TIMEOUT_SECONDS = 3
REQUEST_ID = "request-1"


def test_google_vision_uses_api_key_rest_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, timeout: int) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def post(
            self,
            url: str,
            params: dict[str, str | None],
            json: dict[str, Any],
        ) -> httpx.Response:
            captured["url"] = url
            captured["params"] = params
            captured["json"] = json
            return httpx.Response(
                status_code=200,
                request=httpx.Request("POST", url),
                json={"responses": [{"fullTextAnnotation": {"text": "P<UTO\nLINE2"}}]},
            )

    monkeypatch.setattr("passport_ocr_api.services.google_vision.httpx.Client", FakeClient)

    result = GoogleVisionOcrEngine(_settings(api_key="demo-api-key")).extract(
        Image.new("RGB", (16, 16), "white"),
        TIMEOUT_SECONDS,
        REQUEST_ID,
    )

    assert result.engine == "google_vision"
    assert result.text == "P<UTO\nLINE2"
    assert result.confidence == 1.0
    assert captured["timeout"] == TIMEOUT_SECONDS
    assert captured["params"] == {"key": "demo-api-key"}
    assert captured["json"]["requests"][0]["features"][0]["type"] == "DOCUMENT_TEXT_DETECTION"


def test_google_vision_maps_rest_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class TimeoutClient:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def __enter__(self) -> "TimeoutClient":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def post(
            self,
            url: str,
            params: dict[str, str | None],
            json: dict[str, Any],
        ) -> httpx.Response:
            raise httpx.TimeoutException("timeout")

    monkeypatch.setattr("passport_ocr_api.services.google_vision.httpx.Client", TimeoutClient)

    with pytest.raises(OcrTimeoutError):
        GoogleVisionOcrEngine(_settings(api_key="demo-api-key")).extract(
            Image.new("RGB", (16, 16), "white"),
            TIMEOUT_SECONDS,
            REQUEST_ID,
        )


def test_google_vision_maps_rest_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class ErrorClient:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def __enter__(self) -> "ErrorClient":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def post(
            self,
            url: str,
            params: dict[str, str | None],
            json: dict[str, Any],
        ) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                request=httpx.Request("POST", url),
                json={"responses": [{"error": {"message": "bad key"}}]},
            )

    monkeypatch.setattr("passport_ocr_api.services.google_vision.httpx.Client", ErrorClient)

    with pytest.raises(OcrFailedError):
        GoogleVisionOcrEngine(_settings(api_key="demo-api-key")).extract(
            Image.new("RGB", (16, 16), "white"),
            TIMEOUT_SECONDS,
            REQUEST_ID,
        )


def test_google_vision_rejects_disabled_fallback() -> None:
    with pytest.raises(OcrUnavailableError):
        GoogleVisionOcrEngine(Settings(google_fallback_enabled=False)).extract(
            Image.new("RGB", (16, 16), "white"),
            TIMEOUT_SECONDS,
            REQUEST_ID,
        )


def test_google_vision_request_contains_base64_image() -> None:
    payload = _build_vision_request(Image.new("RGB", (16, 16), "white"))

    image_content = payload["requests"][0]["image"]["content"]

    assert isinstance(image_content, str)
    assert len(image_content) > 0


def _settings(api_key: str) -> Settings:
    return Settings(
        google_fallback_enabled=True,
        google_vision_api_key=api_key,
        google_max_retries=0,
    )
