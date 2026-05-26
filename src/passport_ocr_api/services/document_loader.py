from io import BytesIO

import pypdfium2 as pdfium
from PIL import Image, UnidentifiedImageError

from passport_ocr_api.config import Settings
from passport_ocr_api.errors import BadUploadError
from passport_ocr_api.services.types import RenderedPage
from passport_ocr_api.services.upload_validator import UploadedDocument


class DocumentLoader:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def load_pages(self, document: UploadedDocument) -> tuple[RenderedPage, ...]:
        if document.content_type == "application/pdf":
            return self._load_pdf(document.data)
        return (RenderedPage(image=self._load_image(document.data), page_number=1),)

    def _load_image(self, data: bytes) -> Image.Image:
        try:
            image = Image.open(BytesIO(data))
            image.load()
        except UnidentifiedImageError as exc:
            raise BadUploadError("Uploaded image could not be decoded.") from exc

        return image.convert("RGB")

    def _load_pdf(self, data: bytes) -> tuple[RenderedPage, ...]:
        try:
            pdf = pdfium.PdfDocument(data)
        except Exception as exc:
            raise BadUploadError("Uploaded PDF could not be decoded.") from exc

        page_count = len(pdf)
        if page_count == 0:
            raise BadUploadError("Uploaded PDF has no pages.")
        if page_count > self._settings.max_pdf_pages:
            raise BadUploadError("Uploaded PDF has more pages than the configured limit.")

        pages: list[RenderedPage] = []
        scale = self._settings.max_render_dpi / 72

        for page_index in range(page_count):
            page = pdf[page_index]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil().convert("RGB")
            pages.append(RenderedPage(image=pil_image, page_number=page_index + 1))

        return tuple(pages)
