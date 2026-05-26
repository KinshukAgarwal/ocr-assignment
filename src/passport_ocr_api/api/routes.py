import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile

from passport_ocr_api.config import Settings, get_settings
from passport_ocr_api.logging_config import log_extra
from passport_ocr_api.schemas import PassportOcrResponse
from passport_ocr_api.services.dependencies import build_pipeline
from passport_ocr_api.services.orchestrator import PassportOcrPipeline
from passport_ocr_api.services.upload_validator import UploadedDocument, read_upload

router = APIRouter(prefix="/v1/passports", tags=["passport-ocr"])
logger = logging.getLogger(__name__)


def get_pipeline(settings: Annotated[Settings, Depends(get_settings)]) -> PassportOcrPipeline:
    return build_pipeline(settings)


@router.post("/ocr", response_model=PassportOcrResponse)
async def extract_passport_ocr(
    request: Request,
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
    pipeline: Annotated[PassportOcrPipeline, Depends(get_pipeline)],
) -> PassportOcrResponse:
    request_id = str(request.state.request_id)
    uploaded: UploadedDocument = await read_upload(file, settings)
    logger.info(
        "passport ocr accepted content_type=%s bytes=%s",
        uploaded.content_type,
        len(uploaded.data),
        **log_extra(request_id),
    )
    return pipeline.extract(uploaded, request_id)
