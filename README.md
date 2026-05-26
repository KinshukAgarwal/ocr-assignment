# Passport OCR API

FastAPI module for passport OCR extraction with bounded upload handling, orientation correction, local Tesseract OCR, Google Vision fallback, structured JSON output, and a validation placeholder.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
uvicorn passport_ocr_api.main:app --reload
```

Tesseract must be installed on the host for local OCR:

```bash
sudo apt-get install tesseract-ocr
```

Google Vision fallback uses standard Google Application Default Credentials when `PASSPORT_OCR_GOOGLE_FALLBACK_ENABLED=true`.

## Endpoint

```http
POST /v1/passports/ocr
Content-Type: multipart/form-data
```

Upload field: `file`

Accepted content types:

- `image/jpeg`
- `image/png`
- `image/webp`
- `application/pdf`

## Safety defaults

- Max upload size: 10 MB
- Max PDF pages: 2
- Local OCR timeout: 30 seconds
- Google OCR timeout: 15 seconds
- Google fallback retries: 1
- No persistence of uploaded documents or OCR results

## Quality checks

```bash
ruff check .
mypy src tests
pytest
pip-audit
```
