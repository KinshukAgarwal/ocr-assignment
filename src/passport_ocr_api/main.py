import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from passport_ocr_api.api.manual_tester import router as manual_tester_router
from passport_ocr_api.api.routes import router
from passport_ocr_api.config import get_settings
from passport_ocr_api.errors import AppError, ErrorCode
from passport_ocr_api.logging_config import configure_logging, log_extra
from passport_ocr_api.middleware import RequestIdMiddleware
from passport_ocr_api.schemas import ErrorResponse

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
app.add_middleware(RequestIdMiddleware)
app.include_router(router)
app.include_router(manual_tester_router)


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    request_id = str(getattr(request.state, "request_id", "-"))
    logger.warning(
        "request failed code=%s status=%s",
        exc.code,
        exc.http_status,
        **log_extra(request_id),
    )
    body = ErrorResponse(request_id=request_id, code=exc.code, message=exc.message)
    return JSONResponse(status_code=exc.http_status, content=body.model_dump())


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = str(getattr(request.state, "request_id", "-"))
    logger.exception("unexpected request failure", **log_extra(request_id))
    body = ErrorResponse(
        request_id=request_id,
        code=ErrorCode.OCR_FAILED,
        message="The OCR request failed unexpectedly.",
    )
    return JSONResponse(status_code=500, content=body.model_dump())


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
