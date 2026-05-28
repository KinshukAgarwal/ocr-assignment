from PIL import Image, ImageOps

HISTOGRAM_BUCKETS = 256
DEFAULT_BINARY_THRESHOLD = 180
MIN_BINARY_THRESHOLD = 80
MAX_BINARY_THRESHOLD = 220


def to_black_and_white_text_image(
    image: Image.Image,
    threshold: int | None = None,
) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(grayscale)
    binary_threshold = _clamp_threshold(threshold)
    if binary_threshold is None:
        binary_threshold = _otsu_threshold(contrasted.histogram())
    thresholded = contrasted.point(lambda pixel: 255 if pixel > binary_threshold else 0)
    return thresholded.convert("RGB")


def _clamp_threshold(threshold: int | None) -> int | None:
    if threshold is None:
        return None
    return max(MIN_BINARY_THRESHOLD, min(MAX_BINARY_THRESHOLD, threshold))


def _otsu_threshold(histogram: list[int]) -> int:
    if len(histogram) != HISTOGRAM_BUCKETS:
        return DEFAULT_BINARY_THRESHOLD

    total = sum(histogram)
    if total <= 0:
        return DEFAULT_BINARY_THRESHOLD

    weighted_total = sum(index * count for index, count in enumerate(histogram))
    background_weight = 0
    background_sum = 0
    best_threshold = DEFAULT_BINARY_THRESHOLD
    best_variance = -1.0

    for index, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue

        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break

        background_sum += index * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_total - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = index

    return max(MIN_BINARY_THRESHOLD, min(MAX_BINARY_THRESHOLD, best_threshold))
