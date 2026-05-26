# Passport OCR API

FastAPI module for passport OCR extraction with bounded upload handling, orientation correction, local Tesseract OCR, Google Vision fallback, structured JSON output, and ICAO TD3 MRZ validation.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
uvicorn passport_ocr_api.main:app --reload
```

Tesseract and a monospace font must be installed on the host for local OCR and integration tests:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd fonts-dejavu-core
```

Google Vision fallback uses standard Google Application Default Credentials when `PASSPORT_OCR_GOOGLE_FALLBACK_ENABLED=true`.

## Run with Docker

The Docker image installs system Tesseract so OCR does not depend on host packages.

```bash
docker build -t passport-ocr-api:local .
docker run --rm -p 8000:8000 passport-ocr-api:local
```

Or use Compose:

```bash
docker compose up --build
```

Then verify:

```bash
curl http://127.0.0.1:8000/healthz
```

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

## Validation

The response validates TD3 passport MRZ check digits when both MRZ lines are extracted:

- `passed`: all supported MRZ check digits match
- `failed`: at least one supported MRZ check digit is invalid
- `not_evaluated`: MRZ lines were not available

## Quality checks

```bash
ruff check .
mypy src tests
pytest
pip-audit
```
