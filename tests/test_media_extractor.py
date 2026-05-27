import base64
from io import BytesIO

from PIL import Image, ImageDraw

from passport_ocr_api.services.media_extractor import PassportMediaExtractor

IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 700
PORTRAIT_LEFT = 80
PORTRAIT_TOP = 180
PORTRAIT_RIGHT = 310
PORTRAIT_BOTTOM = 500
SIGNATURE_LEFT = 420
SIGNATURE_TOP = 470
SIGNATURE_RIGHT = 760
SIGNATURE_BOTTOM = 535
LOW_SIGNATURE_LEFT = 120
LOW_SIGNATURE_TOP = 455
LOW_SIGNATURE_RIGHT = 440
LOW_SIGNATURE_BOTTOM = 510
NEARBY_TEXT_LEFT = 560
STROKE_CONFIDENCE = 0.75


def test_media_extractor_returns_portrait_and_signature_crops() -> None:
    image = _passport_like_image()

    result = PassportMediaExtractor().extract(image)

    assert result.portrait.present is True
    assert result.signature.present is True
    assert result.portrait.bounding_box is not None
    assert result.signature.bounding_box is not None
    assert result.portrait.data_base64 is not None
    assert result.signature.data_base64 is not None
    assert _decoded_image_size(result.portrait.data_base64)[0] > 0
    assert _decoded_image_size(result.signature.data_base64)[0] > 0
    assert result.portrait.bounding_box.left < PORTRAIT_RIGHT
    assert result.portrait.bounding_box.top < PORTRAIT_BOTTOM
    assert result.portrait.bounding_box.left + result.portrait.bounding_box.width > PORTRAIT_LEFT
    assert result.portrait.bounding_box.top + result.portrait.bounding_box.height > PORTRAIT_TOP
    assert result.signature.bounding_box.left < SIGNATURE_RIGHT
    assert result.signature.bounding_box.left + result.signature.bounding_box.width > SIGNATURE_LEFT


def test_media_extractor_prefers_stroke_density_signature_below_portrait() -> None:
    image = _passport_like_image_with_signature_below_portrait()

    result = PassportMediaExtractor().extract(image)

    assert result.signature.present is True
    assert result.signature.bounding_box is not None
    assert result.signature.method == "stroke_density_scan"
    assert result.signature.confidence == STROKE_CONFIDENCE

    signature_box = result.signature.bounding_box
    signature_right = signature_box.left + signature_box.width
    assert signature_box.left < LOW_SIGNATURE_RIGHT
    assert signature_right > LOW_SIGNATURE_LEFT
    assert signature_box.top < LOW_SIGNATURE_BOTTOM
    assert signature_right < NEARBY_TEXT_LEFT


def _passport_like_image() -> Image.Image:
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (PORTRAIT_LEFT, PORTRAIT_TOP, PORTRAIT_RIGHT, PORTRAIT_BOTTOM),
        fill=(190, 210, 235),
        outline="black",
    )
    draw.ellipse((130, 230, 250, 350), fill=(215, 170, 135))
    draw.rectangle((145, 350, 235, 470), fill=(35, 80, 145))
    draw.line(
        [
            (SIGNATURE_LEFT, 500),
            (480, 470),
            (550, 525),
            (630, 480),
            (SIGNATURE_RIGHT, 515),
        ],
        fill=(20, 30, 120),
        width=6,
    )
    draw.text((420, 120), "P UTO L898902C3", fill="black")
    return image


def _passport_like_image_with_signature_below_portrait() -> Image.Image:
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 120, 330, 430), fill=(190, 210, 235), outline="black")
    draw.ellipse((135, 170, 275, 300), fill=(215, 170, 135))
    draw.rectangle((150, 300, 260, 410), fill=(35, 80, 145))
    draw.line(
        [
            (LOW_SIGNATURE_LEFT, 485),
            (190, LOW_SIGNATURE_TOP),
            (260, LOW_SIGNATURE_BOTTOM),
            (340, 465),
            (LOW_SIGNATURE_RIGHT, 500),
        ],
        fill=(10, 20, 70),
        width=6,
    )
    draw.text((NEARBY_TEXT_LEFT, 470), "17/06/2025", fill="black")
    draw.text((NEARBY_TEXT_LEFT, 520), "16/06/2035", fill="black")
    return image


def _decoded_image_size(data_base64: str) -> tuple[int, int]:
    image_bytes = base64.b64decode(data_base64)
    with Image.open(BytesIO(image_bytes)) as image:
        return image.size
