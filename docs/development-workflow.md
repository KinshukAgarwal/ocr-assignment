# Development Workflow

## Branching

- `main` holds reviewed, runnable work.
- New implementation work starts from a feature branch named `feature/<short-name>`.
- Commit completed work before starting the next feature.
- Push `main` and feature branches after an `origin` remote is configured.

## Local Checks

Run these before committing:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd fonts-dejavu-core
ruff check .
mypy src tests
pytest
pip-audit
```

## Docker Checks

Build the runtime image:

```bash
docker build -t passport-ocr-api:local .
```

Run the container:

```bash
docker run --rm -p 8000:8000 passport-ocr-api:local
```

Verify health:

```bash
curl http://127.0.0.1:8000/healthz
```

## Remote Setup

After creating a remote repository, connect it once:

```bash
git remote add origin <remote-url>
git push -u origin main
git push -u origin feature/docker-runtime
```
