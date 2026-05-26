from passport_ocr_api.config import Settings
from passport_ocr_api.services.document_loader import DocumentLoader
from passport_ocr_api.services.google_vision import GoogleVisionOcrEngine
from passport_ocr_api.services.local_tesseract import TesseractOcrEngine
from passport_ocr_api.services.mrz_parser import MrzPassportParser
from passport_ocr_api.services.orchestrator import PassportOcrPipeline
from passport_ocr_api.services.orientation import OrientationCorrector
from passport_ocr_api.services.validation import PassportMrzValidator


def build_pipeline(settings: Settings) -> PassportOcrPipeline:
    local_ocr = TesseractOcrEngine()
    return PassportOcrPipeline(
        settings=settings,
        document_loader=DocumentLoader(settings),
        orientation_corrector=OrientationCorrector(local_ocr, settings),
        parser=MrzPassportParser(),
        validator=PassportMrzValidator(),
        cloud_ocr=GoogleVisionOcrEngine(settings),
    )
