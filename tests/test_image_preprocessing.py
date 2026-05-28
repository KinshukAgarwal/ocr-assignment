from PIL import Image, ImageDraw

from passport_ocr_api.services.image_preprocessing import to_black_and_white_text_image

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def test_black_and_white_text_image_contains_only_binary_pixels() -> None:
    image = Image.new("RGB", (24, 12), (245, 235, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 10, 9), fill=(20, 35, 60))
    draw.rectangle((14, 2, 21, 9), fill=(170, 80, 80))

    processed = to_black_and_white_text_image(image)

    assert processed.mode == "RGB"
    assert _unique_rgb_pixels(processed) <= {BLACK, WHITE}


def _unique_rgb_pixels(image: Image.Image) -> set[tuple[int, int, int]]:
    data = image.tobytes()
    return {
        (data[index], data[index + 1], data[index + 2])
        for index in range(0, len(data), 3)
    }
