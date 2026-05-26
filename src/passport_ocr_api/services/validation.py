from passport_ocr_api.schemas import PassportExtraction, ValidationInfo


class PlaceholderPassportValidator:
    def validate(self, extraction: PassportExtraction) -> ValidationInfo:
        return ValidationInfo(status="not_implemented", issues=[])
