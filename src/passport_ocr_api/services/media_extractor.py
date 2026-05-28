import base64
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

from PIL import Image, ImageOps

from passport_ocr_api.schemas import BoundingBox, ExtractedImage, PassportImageExtraction

GRID_COLUMNS = 64
GRID_ROWS = 48
STROKE_MAX_WIDTH = 360
STROKE_MAX_HEIGHT = 180
MAX_COMPONENTS = 256
MAX_COMPONENT_CELLS = GRID_COLUMNS * GRID_ROWS
MAX_STROKE_COMPONENTS = 512
MAX_STROKE_COMPONENT_PIXELS = STROKE_MAX_WIDTH * STROKE_MAX_HEIGHT
MIN_RGB_TUPLE_LENGTH = 3
MIN_PORTRAIT_WIDTH_RATIO = 0.10
MIN_PORTRAIT_HEIGHT_RATIO = 0.18
MAX_PORTRAIT_WIDTH_RATIO = 0.42
MAX_PORTRAIT_HEIGHT_RATIO = 0.68
MIN_SIGNATURE_WIDTH_RATIO = 0.06
MAX_SIGNATURE_HEIGHT_RATIO = 0.20
PORTRAIT_MIN_ASPECT_RATIO = 0.45
PORTRAIT_MAX_ASPECT_RATIO = 1.20
SIGNATURE_MIN_ASPECT_RATIO = 2.0
MIN_SIGNATURE_COMPONENT_WIDTH_RATIO = 0.04
PORTRAIT_SCORE_THRESHOLD = 0.18
SIGNATURE_DARKNESS_THRESHOLD = 0.05
SIGNATURE_COMPONENT_CONFIDENCE = 0.55
SIGNATURE_STROKE_CONFIDENCE = 0.75
PORTRAIT_COMPONENT_CONFIDENCE = 0.65
FALLBACK_CONFIDENCE = 0.2
EXCLUDED_REGION_OVERLAP_RATIO = 0.65
JPEG_QUALITY = 85
MAX_CROP_EDGE_PIXELS = 512
PORTRAIT_OVERSHOOT_X_RATIO = 0.035
PORTRAIT_OVERSHOOT_Y_RATIO = 0.030
SIDE_BAND_TOP_RATIO = 0.10
SIDE_BAND_BOTTOM_RATIO = 0.86
LEFT_BAND_RIGHT_RATIO = 0.45
RIGHT_BAND_LEFT_RATIO = 0.55
SIGNATURE_TOP_RATIO = 0.50
SIGNATURE_BOTTOM_RATIO = 0.76
SIGNATURE_LEFT_RATIO = 0.05
SIGNATURE_RIGHT_RATIO = 0.92
FALLBACK_SIGNATURE_LEFT_RATIO = 0.32
FALLBACK_SIGNATURE_TOP_RATIO = 0.58
FALLBACK_SIGNATURE_WIDTH_RATIO = 0.46
FALLBACK_SIGNATURE_HEIGHT_RATIO = 0.14
PORTRAIT_SIGNATURE_LEFT_PADDING_RATIO = 0.02
PORTRAIT_SIGNATURE_TOP_RATIO = 0.90
PORTRAIT_SIGNATURE_WIDTH_RATIO = 1.20
PORTRAIT_SIGNATURE_HEIGHT_RATIO = 0.22
SKIN_RED_GREEN_MIN = 25
SKIN_RED_BLUE_MIN = 20
SKIN_CHANNEL_SPREAD_MIN = 15
SKIN_BRIGHTNESS_MIN = 85
SKIN_BRIGHTNESS_MAX = 245
MIN_SKIN_CELLS = 2
SKIN_RANK_WEIGHT = 1.5
PORTRAIT_AREA_RANK_WEIGHT = 1.0
BLUE_WATERMARK_PENALTY_WEIGHT = 15.0
BLUE_DOMINANCE_MIN = 20
GHOST_PORTRAIT_BLUE_THRESHOLD = 0.018
ESTIMATED_PORTRAIT_SIDE_MARGIN_RATIO = 0.06
ESTIMATED_PORTRAIT_TOP_RATIO = 0.22
ESTIMATED_PORTRAIT_WIDTH_RATIO = 0.24
ESTIMATED_PORTRAIT_HEIGHT_RATIO = 0.42
FACE_HORIZONTAL_EXPANSION_RATIO = 0.95
FACE_TOP_EXPANSION_RATIO = 0.85
FACE_BOTTOM_EXPANSION_RATIO = 1.8
FACE_BOTTOM_COMPONENT_OVERFLOW_RATIO = 0.10
SIGNATURE_PORTRAIT_LEFT_PADDING_RATIO = 0.04
SIGNATURE_PORTRAIT_RIGHT_PADDING_RATIO = 0.12
SIGNATURE_PORTRAIT_TOP_RATIO = 0.55
SIGNATURE_PORTRAIT_BOTTOM_RATIO = 1.12
SIGNATURE_PORTRAIT_BOTTOM_PADDING_RATIO = 0.16
SIGNATURE_DISTANCE_WEIGHT = 1.2
SIGNATURE_STROKE_TOP_OVERLAP_RATIO = 0.18
SIGNATURE_STROKE_BOTTOM_PADDING_RATIO = 0.26
SIGNATURE_STROKE_LEFT_PADDING_RATIO = 0.28
SIGNATURE_STROKE_RIGHT_PADDING_RATIO = 0.20
SIGNATURE_STROKE_MIN_WIDTH_RATIO = 0.09
SIGNATURE_STROKE_MAX_HEIGHT_RATIO = 0.22
SIGNATURE_STROKE_MIN_ASPECT_RATIO = 1.20
SIGNATURE_STROKE_MIN_PIXELS = 6
SIGNATURE_STROKE_MIN_UNION_PIXELS = 16
SIGNATURE_STROKE_MAX_FILL_RATIO = 0.36
SIGNATURE_STROKE_MAX_COMPONENT_HEIGHT_RATIO = 0.62
SIGNATURE_STROKE_MIN_COMPONENT_WIDTH_RATIO = 0.012
SIGNATURE_STROKE_CLUSTER_VERTICAL_RATIO = 0.20
SIGNATURE_STROKE_THRESHOLD_FLOOR = 45
SIGNATURE_STROKE_THRESHOLD_CEILING = 150
SIGNATURE_STROKE_THRESHOLD_OFFSET = 18
SIGNATURE_STROKE_EXPAND_LEFT_RATIO = 0.055
SIGNATURE_STROKE_EXPAND_RIGHT_RATIO = 0.030
SIGNATURE_STROKE_EXPAND_Y_RATIO = 0.025
TRIM_SCAN_MAX_EDGE_PIXELS = 240
TRIM_MIN_EDGE_PIXELS = 3
PORTRAIT_TRIM_MAX_EDGE_RATIO = 0.18
SIGNATURE_TRIM_MAX_EDGE_RATIO = 0.32
PORTRAIT_TRIM_KEEP_RATIO = 0.025
SIGNATURE_TRIM_KEEP_RATIO = 0.045
PORTRAIT_TRIM_MIN_ACTIVE_RATIO = 0.060
SIGNATURE_TRIM_MIN_ACTIVE_RATIO = 0.006
PORTRAIT_BLANK_BRIGHTNESS = 238
PORTRAIT_BLANK_SPREAD = 18
SIGNATURE_BLUE_INK_DELTA = 22
SIGNATURE_COLORED_INK_SPREAD = 42
SIGNATURE_INK_BRIGHTNESS_MAX = 225

CellKind = Literal["photo", "dark"]
MediaKind = Literal["portrait", "signature"]


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class Component:
    box: Box
    cell_count: int
    score: float


@dataclass(frozen=True)
class MediaCandidate:
    box: Box | None
    confidence: float
    method: str


@dataclass(frozen=True)
class AnalysisGrid:
    image: Image.Image
    cell_width: float
    cell_height: float


@dataclass(frozen=True)
class StrokeMask:
    width: int
    height: int
    mask: tuple[bool, ...]
    scale_x: float
    scale_y: float
    origin: Box


@dataclass(frozen=True)
class StrokeComponent:
    box: Box
    pixel_count: int


class PassportMediaExtractor:
    def extract(self, image: Image.Image) -> PassportImageExtraction:
        portrait_box = self._find_portrait_box(image)
        signature = self._find_signature(image, portrait_box)

        return PassportImageExtraction(
            portrait=_build_image_result(
                image=image,
                box=portrait_box,
                confidence=PORTRAIT_COMPONENT_CONFIDENCE if portrait_box is not None else 0.0,
                method="layout_component_scan" if portrait_box is not None else "not_found",
            ),
            signature=_build_image_result(
                image=image,
                box=signature.box,
                confidence=signature.confidence,
                method=signature.method,
            ),
        )

    def _find_portrait_box(self, image: Image.Image) -> Box | None:
        grid = _build_grid(image)
        bands = (
            _fractional_box(
                image,
                0.0,
                SIDE_BAND_TOP_RATIO,
                LEFT_BAND_RIGHT_RATIO,
                SIDE_BAND_BOTTOM_RATIO,
            ),
            _fractional_box(
                image,
                RIGHT_BAND_LEFT_RATIO,
                SIDE_BAND_TOP_RATIO,
                1.0,
                SIDE_BAND_BOTTOM_RATIO,
            ),
        )
        components = _find_components(grid, bands, "photo")
        candidates = [
            component for component in components if _is_portrait_candidate(component.box, image)
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda component: _portrait_rank(component, image))
        if (
            _blue_dominance_score(best.box, image) >= GHOST_PORTRAIT_BLUE_THRESHOLD
            and _skin_score(best.box, image) == 0.0
        ):
            estimated = _estimated_main_portrait_box(image, best.box)
            return _overshoot_and_trim_media_box(image, estimated, "portrait")
        refined = _refine_portrait_box(best.box, image)
        return _overshoot_and_trim_media_box(image, refined, "portrait")

    def _find_signature(self, image: Image.Image, portrait_box: Box | None) -> MediaCandidate:
        stroke_candidate = _find_signature_stroke_candidate(image, portrait_box)
        if stroke_candidate.box is not None:
            return stroke_candidate

        grid = _build_grid(image)
        search_boxes = _signature_search_boxes(image, portrait_box)
        components = _find_components(grid, search_boxes, "dark")
        candidates = [
            component
            for component in components
            if _is_signature_candidate(component.box, image)
            and not _overlaps_excluded_region(component.box, portrait_box)
        ]
        if not candidates:
            return MediaCandidate(
                box=_fallback_signature_box(image, portrait_box),
                confidence=FALLBACK_CONFIDENCE,
                method="layout_estimate",
            )
        best = max(
            candidates,
            key=lambda component: _signature_rank(component, image, portrait_box),
        )
        return MediaCandidate(
            box=_overshoot_and_trim_media_box(image, best.box, "signature"),
            confidence=SIGNATURE_COMPONENT_CONFIDENCE,
            method="stroke_component_scan",
        )


def _build_grid(image: Image.Image) -> AnalysisGrid:
    return AnalysisGrid(
        image=image.resize((GRID_COLUMNS, GRID_ROWS)).convert("RGB"),
        cell_width=image.width / GRID_COLUMNS,
        cell_height=image.height / GRID_ROWS,
    )


def _find_components(
    grid: AnalysisGrid,
    allowed_boxes: tuple[Box, ...],
    kind: CellKind,
) -> list[Component]:
    visited: set[tuple[int, int]] = set()
    components: list[Component] = []

    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            if len(components) >= MAX_COMPONENTS:
                return components
            if (column, row) in visited or not _cell_allowed(column, row, grid, allowed_boxes):
                continue
            score = _cell_score(_rgb_at(grid.image, column, row), kind)
            if score <= 0.0:
                continue
            components.append(_collect_component(grid, column, row, kind, allowed_boxes, visited))

    return components


def _collect_component(
    grid: AnalysisGrid,
    start_column: int,
    start_row: int,
    kind: CellKind,
    allowed_boxes: tuple[Box, ...],
    visited: set[tuple[int, int]],
) -> Component:
    queue: deque[tuple[int, int]] = deque([(start_column, start_row)])
    visited.add((start_column, start_row))
    cells: list[tuple[int, int]] = []
    score_total = 0.0

    while queue and len(cells) < MAX_COMPONENT_CELLS:
        column, row = queue.popleft()
        cells.append((column, row))
        score_total += _cell_score(_rgb_at(grid.image, column, row), kind)
        for neighbor in _neighbors(column, row):
            if neighbor in visited:
                continue
            neighbor_column, neighbor_row = neighbor
            if not _cell_allowed(neighbor_column, neighbor_row, grid, allowed_boxes):
                continue
            if _cell_score(_rgb_at(grid.image, neighbor_column, neighbor_row), kind) <= 0.0:
                continue
            visited.add(neighbor)
            queue.append(neighbor)

    return _component_from_cells(cells, grid, score_total)


def _component_from_cells(
    cells: list[tuple[int, int]],
    grid: AnalysisGrid,
    score_total: float,
) -> Component:
    columns = [column for column, _ in cells]
    rows = [row for _, row in cells]
    left = int(min(columns) * grid.cell_width)
    top = int(min(rows) * grid.cell_height)
    right = int((max(columns) + 1) * grid.cell_width)
    bottom = int((max(rows) + 1) * grid.cell_height)
    return Component(
        box=Box(left=left, top=top, right=right, bottom=bottom),
        cell_count=len(cells),
        score=score_total / max(1, len(cells)),
    )


def _neighbors(column: int, row: int) -> tuple[tuple[int, int], ...]:
    return (
        (column - 1, row),
        (column + 1, row),
        (column, row - 1),
        (column, row + 1),
    )


def _cell_allowed(column: int, row: int, grid: AnalysisGrid, boxes: tuple[Box, ...]) -> bool:
    if not (0 <= column < GRID_COLUMNS and 0 <= row < GRID_ROWS):
        return False
    x = int(column * grid.cell_width)
    y = int(row * grid.cell_height)
    return any(box.left <= x < box.right and box.top <= y < box.bottom for box in boxes)


def _rgb_at(image: Image.Image, column: int, row: int) -> tuple[int, int, int]:
    pixel = image.getpixel((column, row))
    if isinstance(pixel, tuple) and len(pixel) >= MIN_RGB_TUPLE_LENGTH:
        return int(pixel[0]), int(pixel[1]), int(pixel[2])
    value = int(pixel) if isinstance(pixel, int | float) else 0
    return value, value, value


def _cell_score(pixel: tuple[int, int, int], kind: CellKind) -> float:
    red, green, blue = pixel
    brightness = (red + green + blue) / 765
    colorfulness = (max(pixel) - min(pixel)) / 255
    darkness = 1.0 - brightness
    if kind == "photo":
        score = (colorfulness * 0.7) + (darkness * 0.3)
        return score if score >= PORTRAIT_SCORE_THRESHOLD else 0.0
    return darkness if darkness >= SIGNATURE_DARKNESS_THRESHOLD else 0.0


def _is_portrait_candidate(box: Box, image: Image.Image) -> bool:
    if box.width < image.width * MIN_PORTRAIT_WIDTH_RATIO:
        return False
    if box.height < image.height * MIN_PORTRAIT_HEIGHT_RATIO:
        return False
    if box.width > image.width * MAX_PORTRAIT_WIDTH_RATIO:
        return False
    if box.height > image.height * MAX_PORTRAIT_HEIGHT_RATIO:
        return False
    aspect_ratio = box.width / box.height
    return PORTRAIT_MIN_ASPECT_RATIO <= aspect_ratio <= PORTRAIT_MAX_ASPECT_RATIO


def _portrait_rank(component: Component, image: Image.Image) -> tuple[float, int]:
    side_distance = min(component.box.left, image.width - component.box.right)
    edge_score = 1.0 - min(side_distance / max(1, image.width // 2), 1.0)
    area_score = component.box.area / max(1, image.width * image.height)
    rank = (
        edge_score
        + component.score
        + (_skin_score(component.box, image) * SKIN_RANK_WEIGHT)
        + (area_score * PORTRAIT_AREA_RANK_WEIGHT)
        - (_blue_dominance_score(component.box, image) * BLUE_WATERMARK_PENALTY_WEIGHT)
    )
    return (rank, component.box.area)


def _refine_portrait_box(component_box: Box, image: Image.Image) -> Box:
    skin_box = _skin_bbox(component_box, image)
    if skin_box is None:
        return component_box

    horizontal_padding = int(skin_box.width * FACE_HORIZONTAL_EXPANSION_RATIO)
    top_padding = int(skin_box.height * FACE_TOP_EXPANSION_RATIO)
    bottom_padding = int(skin_box.height * FACE_BOTTOM_EXPANSION_RATIO)
    face_box = Box(
        left=max(0, skin_box.left - horizontal_padding),
        top=max(0, skin_box.top - top_padding),
        right=min(image.width, skin_box.right + horizontal_padding),
        bottom=min(image.height, skin_box.bottom + bottom_padding),
    )
    max_bottom = component_box.bottom + int(
        component_box.height * FACE_BOTTOM_COMPONENT_OVERFLOW_RATIO,
    )
    return Box(
        left=min(component_box.left, face_box.left),
        top=min(component_box.top, face_box.top),
        right=max(component_box.right, face_box.right),
        bottom=max(component_box.bottom, min(face_box.bottom, max_bottom)),
    )


def _skin_bbox(box: Box, image: Image.Image) -> Box | None:
    skin_cells = _skin_cells(box, image)
    if len(skin_cells) < MIN_SKIN_CELLS:
        return None

    columns = [column for column, _ in skin_cells]
    rows = [row for _, row in skin_cells]
    return Box(
        left=int(min(columns) * image.width / GRID_COLUMNS),
        top=int(min(rows) * image.height / GRID_ROWS),
        right=int((max(columns) + 1) * image.width / GRID_COLUMNS),
        bottom=int((max(rows) + 1) * image.height / GRID_ROWS),
    )


def _skin_score(box: Box, image: Image.Image) -> float:
    cells = _skin_cells(box, image)
    box_cell_area = max(
        1,
        int(box.width / max(1, image.width / GRID_COLUMNS))
        * int(box.height / max(1, image.height / GRID_ROWS)),
    )
    return min(len(cells) / box_cell_area, 1.0)


def _skin_cells(box: Box, image: Image.Image) -> list[tuple[int, int]]:
    grid = _build_grid(image)
    cells: list[tuple[int, int]] = []
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            x = int(column * grid.cell_width)
            y = int(row * grid.cell_height)
            if (
                box.left <= x < box.right
                and box.top <= y < box.bottom
                and _looks_like_skin(_rgb_at(grid.image, column, row))
            ):
                cells.append((column, row))
    return cells


def _blue_dominance_score(box: Box, image: Image.Image) -> float:
    grid = _build_grid(image)
    total = 0.0
    count = 0
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            x = int(column * grid.cell_width)
            y = int(row * grid.cell_height)
            if box.left <= x < box.right and box.top <= y < box.bottom:
                red, green, blue = _rgb_at(grid.image, column, row)
                dominance = blue - max(red, green)
                if dominance >= BLUE_DOMINANCE_MIN:
                    total += dominance / 255
                count += 1
    return total / max(1, count)


def _estimated_main_portrait_box(image: Image.Image, rejected_box: Box) -> Box:
    top = int(image.height * ESTIMATED_PORTRAIT_TOP_RATIO)
    width = int(image.width * ESTIMATED_PORTRAIT_WIDTH_RATIO)
    height = int(image.height * ESTIMATED_PORTRAIT_HEIGHT_RATIO)
    if rejected_box.left > image.width // 2:
        left = int(image.width * ESTIMATED_PORTRAIT_SIDE_MARGIN_RATIO)
    else:
        left = int(image.width * (1 - ESTIMATED_PORTRAIT_SIDE_MARGIN_RATIO)) - width
    return Box(
        left=max(0, left),
        top=max(0, top),
        right=min(image.width, left + width),
        bottom=min(image.height, top + height),
    )


def _looks_like_skin(pixel: tuple[int, int, int]) -> bool:
    red, green, blue = pixel
    brightness = max(pixel)
    spread = max(pixel) - min(pixel)
    return (
        SKIN_BRIGHTNESS_MIN <= brightness <= SKIN_BRIGHTNESS_MAX
        and red > blue
        and green > blue
        and red - green >= SKIN_RED_GREEN_MIN
        and red - blue >= SKIN_RED_BLUE_MIN
        and spread >= SKIN_CHANNEL_SPREAD_MIN
    )


def _find_signature_stroke_candidate(
    image: Image.Image,
    portrait_box: Box | None,
) -> MediaCandidate:
    candidates: list[tuple[Box, int]] = []
    for search_box in _signature_stroke_search_boxes(image, portrait_box):
        candidate = _signature_stroke_candidate_in_box(image, search_box, portrait_box)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return MediaCandidate(box=None, confidence=0.0, method="stroke_density_scan")

    best_box, _ = max(candidates, key=lambda candidate: _signature_stroke_rank(candidate[0], image))
    return MediaCandidate(
        box=best_box,
        confidence=SIGNATURE_STROKE_CONFIDENCE,
        method="stroke_density_scan",
    )


def _signature_stroke_search_boxes(
    image: Image.Image,
    portrait_box: Box | None,
) -> tuple[Box, ...]:
    if portrait_box is None:
        return (
            _fractional_box(
                image,
                SIGNATURE_LEFT_RATIO,
                SIGNATURE_TOP_RATIO,
                SIGNATURE_RIGHT_RATIO,
                SIGNATURE_BOTTOM_RATIO,
            ),
        )

    left = portrait_box.left - int(portrait_box.width * SIGNATURE_STROKE_LEFT_PADDING_RATIO)
    top = portrait_box.bottom - int(portrait_box.height * SIGNATURE_STROKE_TOP_OVERLAP_RATIO)
    right = portrait_box.right + int(portrait_box.width * SIGNATURE_STROKE_RIGHT_PADDING_RATIO)
    bottom = portrait_box.bottom + int(portrait_box.height * SIGNATURE_STROKE_BOTTOM_PADDING_RATIO)
    portrait_local_box = _clamp_box(Box(left=left, top=top, right=right, bottom=bottom), image)
    return (
        portrait_local_box,
        _fractional_box(
            image,
            SIGNATURE_LEFT_RATIO,
            SIGNATURE_TOP_RATIO,
            SIGNATURE_RIGHT_RATIO,
            SIGNATURE_BOTTOM_RATIO,
        ),
    )


def _signature_stroke_candidate_in_box(
    image: Image.Image,
    search_box: Box,
    portrait_box: Box | None,
) -> tuple[Box, int] | None:
    if search_box.width <= 0 or search_box.height <= 0:
        return None

    stroke_mask = _build_stroke_mask(image, search_box)
    components = [
        component
        for component in _find_stroke_components(stroke_mask)
        if _is_stroke_component(component, stroke_mask)
        and _is_in_signature_band(component, stroke_mask, portrait_box)
    ]
    selected_components = _select_signature_stroke_cluster(components, stroke_mask, image)
    if not selected_components:
        return None

    union_box = _union_component_boxes([component.box for component in selected_components])
    if union_box is None:
        return None

    image_box = _stroke_box_to_image_box(union_box, stroke_mask)
    expanded = _expand_signature_stroke_box(image_box, image)
    total_pixels = sum(component.pixel_count for component in selected_components)
    if not _is_signature_stroke_candidate(expanded, total_pixels, image):
        return None
    return (expanded, total_pixels)


def _build_stroke_mask(image: Image.Image, search_box: Box) -> StrokeMask:
    crop = image.crop((search_box.left, search_box.top, search_box.right, search_box.bottom))
    crop = crop.convert("RGB")
    crop.thumbnail((STROKE_MAX_WIDTH, STROKE_MAX_HEIGHT))
    grayscale = ImageOps.grayscale(crop)
    pixels = list(grayscale.tobytes())
    threshold = _stroke_threshold(pixels)
    mask = tuple(pixel <= threshold for pixel in pixels)
    return StrokeMask(
        width=grayscale.width,
        height=grayscale.height,
        mask=mask,
        scale_x=search_box.width / max(1, grayscale.width),
        scale_y=search_box.height / max(1, grayscale.height),
        origin=search_box,
    )


def _stroke_threshold(pixels: list[int]) -> int:
    if not pixels:
        return SIGNATURE_STROKE_THRESHOLD_FLOOR
    sorted_pixels = sorted(pixels)
    percentile_index = min(len(sorted_pixels) - 1, max(0, len(sorted_pixels) // 4))
    percentile_value = sorted_pixels[percentile_index]
    threshold = percentile_value + SIGNATURE_STROKE_THRESHOLD_OFFSET
    return max(
        SIGNATURE_STROKE_THRESHOLD_FLOOR,
        min(SIGNATURE_STROKE_THRESHOLD_CEILING, threshold),
    )


def _find_stroke_components(stroke_mask: StrokeMask) -> list[StrokeComponent]:
    visited: set[tuple[int, int]] = set()
    components: list[StrokeComponent] = []

    for row in range(stroke_mask.height):
        for column in range(stroke_mask.width):
            if len(components) >= MAX_STROKE_COMPONENTS:
                return components
            if (column, row) in visited or not _stroke_at(stroke_mask, column, row):
                continue
            components.append(_collect_stroke_component(stroke_mask, column, row, visited))

    return components


def _collect_stroke_component(
    stroke_mask: StrokeMask,
    start_column: int,
    start_row: int,
    visited: set[tuple[int, int]],
) -> StrokeComponent:
    queue: deque[tuple[int, int]] = deque([(start_column, start_row)])
    visited.add((start_column, start_row))
    pixels: list[tuple[int, int]] = []

    while queue and len(pixels) < MAX_STROKE_COMPONENT_PIXELS:
        column, row = queue.popleft()
        pixels.append((column, row))
        for neighbor_column, neighbor_row in _neighbors(column, row):
            if (neighbor_column, neighbor_row) in visited:
                continue
            if not _stroke_at(stroke_mask, neighbor_column, neighbor_row):
                continue
            visited.add((neighbor_column, neighbor_row))
            queue.append((neighbor_column, neighbor_row))

    return StrokeComponent(box=_box_from_points(pixels), pixel_count=len(pixels))


def _stroke_at(stroke_mask: StrokeMask, column: int, row: int) -> bool:
    if not (0 <= column < stroke_mask.width and 0 <= row < stroke_mask.height):
        return False
    index = row * stroke_mask.width + column
    return stroke_mask.mask[index]


def _box_from_points(points: list[tuple[int, int]]) -> Box:
    columns = [column for column, _ in points]
    rows = [row for _, row in points]
    return Box(
        left=min(columns),
        top=min(rows),
        right=max(columns) + 1,
        bottom=max(rows) + 1,
    )


def _is_stroke_component(component: StrokeComponent, stroke_mask: StrokeMask) -> bool:
    if component.pixel_count < SIGNATURE_STROKE_MIN_PIXELS:
        return False
    min_width = max(
        2,
        int(stroke_mask.width * SIGNATURE_STROKE_MIN_COMPONENT_WIDTH_RATIO),
    )
    if component.box.width < min_width:
        return False
    if component.box.height > stroke_mask.height * SIGNATURE_STROKE_MAX_COMPONENT_HEIGHT_RATIO:
        return False
    fill_ratio = component.pixel_count / max(1, component.box.area)
    return fill_ratio <= SIGNATURE_STROKE_MAX_FILL_RATIO


def _is_in_signature_band(
    component: StrokeComponent,
    stroke_mask: StrokeMask,
    portrait_box: Box | None,
) -> bool:
    if portrait_box is None:
        return True

    portrait_bottom_in_mask = int(
        (portrait_box.bottom - stroke_mask.origin.top) / stroke_mask.scale_y,
    )
    band_start = max(0, portrait_bottom_in_mask - int(stroke_mask.height * 0.12))
    component_center_y = component.box.top + (component.box.height / 2)
    return component_center_y >= band_start


def _select_signature_stroke_cluster(
    components: list[StrokeComponent],
    stroke_mask: StrokeMask,
    image: Image.Image,
) -> list[StrokeComponent]:
    best_cluster: list[StrokeComponent] = []
    best_rank: tuple[float, int] = (-1.0, 0)

    for seed in components:
        cluster = [
            component
            for component in components
            if _same_signature_cluster(seed, component, stroke_mask)
        ]
        union_box = _union_component_boxes([component.box for component in cluster])
        if union_box is None:
            continue

        image_box = _expand_signature_stroke_box(
            _stroke_box_to_image_box(union_box, stroke_mask),
            image,
        )
        total_pixels = sum(component.pixel_count for component in cluster)
        if not _is_signature_stroke_candidate(image_box, total_pixels, image):
            continue

        rank = _signature_stroke_rank(image_box, image)
        if rank > best_rank:
            best_cluster = cluster
            best_rank = rank

    return best_cluster


def _same_signature_cluster(
    seed: StrokeComponent,
    component: StrokeComponent,
    stroke_mask: StrokeMask,
) -> bool:
    seed_center_y = seed.box.top + (seed.box.height / 2)
    component_center_y = component.box.top + (component.box.height / 2)
    max_vertical_distance = max(
        seed.box.height,
        int(stroke_mask.height * SIGNATURE_STROKE_CLUSTER_VERTICAL_RATIO),
    )
    return abs(seed_center_y - component_center_y) <= max_vertical_distance


def _union_component_boxes(boxes: list[Box]) -> Box | None:
    if not boxes:
        return None
    return Box(
        left=min(box.left for box in boxes),
        top=min(box.top for box in boxes),
        right=max(box.right for box in boxes),
        bottom=max(box.bottom for box in boxes),
    )


def _stroke_box_to_image_box(box: Box, stroke_mask: StrokeMask) -> Box:
    left = stroke_mask.origin.left + int(box.left * stroke_mask.scale_x)
    top = stroke_mask.origin.top + int(box.top * stroke_mask.scale_y)
    right = stroke_mask.origin.left + int(box.right * stroke_mask.scale_x)
    bottom = stroke_mask.origin.top + int(box.bottom * stroke_mask.scale_y)
    return Box(left=left, top=top, right=right, bottom=bottom)


def _expand_signature_stroke_box(box: Box, image: Image.Image) -> Box:
    expanded = _expand_box_asymmetric(
        box,
        image,
        left_ratio=SIGNATURE_STROKE_EXPAND_LEFT_RATIO,
        top_ratio=SIGNATURE_STROKE_EXPAND_Y_RATIO,
        right_ratio=SIGNATURE_STROKE_EXPAND_RIGHT_RATIO,
        bottom_ratio=SIGNATURE_STROKE_EXPAND_Y_RATIO,
    )
    return _trim_low_activity_edges(image, expanded, "signature")


def _is_signature_stroke_candidate(box: Box, pixel_count: int, image: Image.Image) -> bool:
    if pixel_count < SIGNATURE_STROKE_MIN_UNION_PIXELS:
        return False
    if box.width < image.width * SIGNATURE_STROKE_MIN_WIDTH_RATIO:
        return False
    if box.height > image.height * SIGNATURE_STROKE_MAX_HEIGHT_RATIO:
        return False
    return box.width / max(1, box.height) >= SIGNATURE_STROKE_MIN_ASPECT_RATIO


def _signature_stroke_rank(box: Box, image: Image.Image) -> tuple[float, int]:
    area_score = box.area / max(1, image.width * image.height)
    compact_height_score = 1.0 - min(box.height / max(1, image.height * 0.25), 1.0)
    return (area_score + compact_height_score, box.width)


def _is_signature_candidate(box: Box, image: Image.Image) -> bool:
    if box.width < image.width * MIN_SIGNATURE_COMPONENT_WIDTH_RATIO:
        return False
    if box.height > image.height * MAX_SIGNATURE_HEIGHT_RATIO:
        return False
    return box.width / max(1, box.height) >= SIGNATURE_MIN_ASPECT_RATIO


def _signature_search_boxes(image: Image.Image, portrait_box: Box | None) -> tuple[Box, ...]:
    global_box = _fractional_box(
        image,
        SIGNATURE_LEFT_RATIO,
        SIGNATURE_TOP_RATIO,
        SIGNATURE_RIGHT_RATIO,
        SIGNATURE_BOTTOM_RATIO,
    )
    if portrait_box is None:
        return (global_box,)

    portrait_local_box = Box(
        left=max(0, portrait_box.left - int(image.width * SIGNATURE_PORTRAIT_LEFT_PADDING_RATIO)),
        top=max(0, portrait_box.top + int(portrait_box.height * SIGNATURE_PORTRAIT_TOP_RATIO)),
        right=min(
            image.width,
            portrait_box.right + int(image.width * SIGNATURE_PORTRAIT_RIGHT_PADDING_RATIO),
        ),
        bottom=min(
            image.height,
            portrait_box.top
            + int(portrait_box.height * SIGNATURE_PORTRAIT_BOTTOM_RATIO)
            + int(image.height * SIGNATURE_PORTRAIT_BOTTOM_PADDING_RATIO),
        ),
    )
    return (portrait_local_box, global_box)


def _signature_rank(
    component: Component,
    image: Image.Image,
    portrait_box: Box | None,
) -> tuple[float, int]:
    if portrait_box is None:
        return (component.score, component.box.width)

    portrait_center_x = portrait_box.left + (portrait_box.width / 2)
    portrait_signature_y = portrait_box.top + (
        portrait_box.height * SIGNATURE_PORTRAIT_BOTTOM_RATIO
    )
    component_center_x = component.box.left + (component.box.width / 2)
    component_center_y = component.box.top + (component.box.height / 2)
    horizontal_distance = abs(component_center_x - portrait_center_x)
    vertical_distance = abs(component_center_y - portrait_signature_y)
    horizontal_score = 1.0 - min(horizontal_distance / max(1.0, image.width / 2), 1.0)
    vertical_score = 1.0 - min(vertical_distance / max(1.0, image.height / 3), 1.0)
    distance_score = (horizontal_score + vertical_score) * SIGNATURE_DISTANCE_WEIGHT
    return (component.score + distance_score, component.box.width)


def _overlaps_excluded_region(box: Box, excluded: Box | None) -> bool:
    if excluded is None:
        return False
    intersection_left = max(box.left, excluded.left)
    intersection_top = max(box.top, excluded.top)
    intersection_right = min(box.right, excluded.right)
    intersection_bottom = min(box.bottom, excluded.bottom)
    if intersection_left >= intersection_right or intersection_top >= intersection_bottom:
        return False
    intersection_area = (intersection_right - intersection_left) * (
        intersection_bottom - intersection_top
    )
    return intersection_area / max(1, box.area) >= EXCLUDED_REGION_OVERLAP_RATIO


def _fallback_signature_box(image: Image.Image, portrait_box: Box | None) -> Box | None:
    if portrait_box is not None:
        return _overshoot_and_trim_media_box(
            image,
            _portrait_relative_signature_box(image, portrait_box),
            "signature",
        )

    box = _fractional_box(
        image,
        FALLBACK_SIGNATURE_LEFT_RATIO,
        FALLBACK_SIGNATURE_TOP_RATIO,
        FALLBACK_SIGNATURE_LEFT_RATIO + FALLBACK_SIGNATURE_WIDTH_RATIO,
        FALLBACK_SIGNATURE_TOP_RATIO + FALLBACK_SIGNATURE_HEIGHT_RATIO,
    )
    if _overlaps_excluded_region(box, portrait_box):
        return None
    return _overshoot_and_trim_media_box(image, box, "signature")


def _portrait_relative_signature_box(image: Image.Image, portrait_box: Box) -> Box:
    left = max(0, portrait_box.left - int(image.width * PORTRAIT_SIGNATURE_LEFT_PADDING_RATIO))
    top = portrait_box.top + int(portrait_box.height * PORTRAIT_SIGNATURE_TOP_RATIO)
    width = int(portrait_box.width * PORTRAIT_SIGNATURE_WIDTH_RATIO)
    height = int(portrait_box.height * PORTRAIT_SIGNATURE_HEIGHT_RATIO)
    return Box(
        left=left,
        top=min(image.height - 1, max(0, top)),
        right=min(image.width, left + width),
        bottom=min(image.height, top + height),
    )


def _fractional_box(
    image: Image.Image,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> Box:
    return Box(
        left=int(image.width * left),
        top=int(image.height * top),
        right=int(image.width * right),
        bottom=int(image.height * bottom),
    )


def _overshoot_and_trim_media_box(image: Image.Image, box: Box, kind: MediaKind) -> Box:
    if kind == "portrait":
        expanded = _expand_box_asymmetric(
            box,
            image,
            left_ratio=PORTRAIT_OVERSHOOT_X_RATIO,
            top_ratio=PORTRAIT_OVERSHOOT_Y_RATIO,
            right_ratio=PORTRAIT_OVERSHOOT_X_RATIO,
            bottom_ratio=PORTRAIT_OVERSHOOT_Y_RATIO,
        )
        return _trim_low_activity_edges(image, expanded, kind)

    expanded = _expand_box_asymmetric(
        box,
        image,
        left_ratio=SIGNATURE_STROKE_EXPAND_LEFT_RATIO,
        top_ratio=SIGNATURE_STROKE_EXPAND_Y_RATIO,
        right_ratio=SIGNATURE_STROKE_EXPAND_RIGHT_RATIO,
        bottom_ratio=SIGNATURE_STROKE_EXPAND_Y_RATIO,
    )
    return _trim_low_activity_edges(image, expanded, kind)


def _expand_box_asymmetric(
    box: Box,
    image: Image.Image,
    *,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> Box:
    return _clamp_box(
        Box(
            left=box.left - int(image.width * left_ratio),
            top=box.top - int(image.height * top_ratio),
            right=box.right + int(image.width * right_ratio),
            bottom=box.bottom + int(image.height * bottom_ratio),
        ),
        image,
    )


def _trim_low_activity_edges(image: Image.Image, box: Box, kind: MediaKind) -> Box:
    clamped = _clamp_box(box, image)
    if clamped.width <= TRIM_MIN_EDGE_PIXELS or clamped.height <= TRIM_MIN_EDGE_PIXELS:
        return clamped

    scan = image.crop((clamped.left, clamped.top, clamped.right, clamped.bottom)).convert("RGB")
    scan.thumbnail((TRIM_SCAN_MAX_EDGE_PIXELS, TRIM_SCAN_MAX_EDGE_PIXELS))
    if scan.width <= TRIM_MIN_EDGE_PIXELS or scan.height <= TRIM_MIN_EDGE_PIXELS:
        return clamped

    activity = _activity_mask(scan, kind)
    left = _edge_trim_pixels(activity, scan.width, scan.height, "left", kind)
    right = _edge_trim_pixels(activity, scan.width, scan.height, "right", kind)
    top = _edge_trim_pixels(activity, scan.width, scan.height, "top", kind)
    bottom = _edge_trim_pixels(activity, scan.width, scan.height, "bottom", kind)
    return _map_scan_trim_to_box(clamped, scan.size, left, right, top, bottom, kind)


def _activity_mask(scan: Image.Image, kind: MediaKind) -> tuple[bool, ...]:
    raw_pixels = scan.tobytes()
    pixels = [
        (raw_pixels[index], raw_pixels[index + 1], raw_pixels[index + 2])
        for index in range(0, len(raw_pixels), MIN_RGB_TUPLE_LENGTH)
    ]
    if kind == "signature":
        grayscale = [int(sum(pixel) / MIN_RGB_TUPLE_LENGTH) for pixel in pixels]
        threshold = _stroke_threshold(grayscale)
        return tuple(_looks_like_signature_ink(pixel, threshold) for pixel in pixels)

    return tuple(_looks_like_portrait_content(pixel) for pixel in pixels)


def _looks_like_signature_ink(pixel: tuple[int, int, int], threshold: int) -> bool:
    red, green, blue = pixel
    brightness = int(sum(pixel) / MIN_RGB_TUPLE_LENGTH)
    spread = max(pixel) - min(pixel)
    blue_delta = blue - max(red, green)
    return (
        brightness <= threshold
        or (blue_delta >= SIGNATURE_BLUE_INK_DELTA and brightness <= SIGNATURE_INK_BRIGHTNESS_MAX)
        or (
            spread >= SIGNATURE_COLORED_INK_SPREAD
            and brightness <= SIGNATURE_INK_BRIGHTNESS_MAX
        )
    )


def _looks_like_portrait_content(pixel: tuple[int, int, int]) -> bool:
    brightness = int(sum(pixel) / MIN_RGB_TUPLE_LENGTH)
    spread = max(pixel) - min(pixel)
    return brightness <= PORTRAIT_BLANK_BRIGHTNESS or spread >= PORTRAIT_BLANK_SPREAD


def _edge_trim_pixels(
    activity: tuple[bool, ...],
    width: int,
    height: int,
    edge: Literal["left", "right", "top", "bottom"],
    kind: MediaKind,
) -> int:
    vertical_edge = edge in {"left", "right"}
    scan_length = width if vertical_edge else height
    cross_length = height if vertical_edge else width
    min_active = max(1, int(cross_length * _trim_min_active_ratio(kind)))
    keep = max(TRIM_MIN_EDGE_PIXELS, int(scan_length * _trim_keep_ratio(kind)))
    max_steps = min(scan_length, int(scan_length * _trim_max_edge_ratio(kind)))

    for offset in range(max_steps):
        if _edge_active_count(activity, width, height, edge, offset) >= min_active:
            return max(0, offset - keep)
    return 0


def _edge_active_count(
    activity: tuple[bool, ...],
    width: int,
    height: int,
    edge: Literal["left", "right", "top", "bottom"],
    offset: int,
) -> int:
    if edge == "left":
        return sum(int(activity[(row * width) + offset]) for row in range(height))
    if edge == "right":
        column = width - 1 - offset
        return sum(int(activity[(row * width) + column]) for row in range(height))
    if edge == "top":
        return sum(int(activity[(offset * width) + column]) for column in range(width))

    row = height - 1 - offset
    return sum(int(activity[(row * width) + column]) for column in range(width))


def _map_scan_trim_to_box(
    box: Box,
    scan_size: tuple[int, int],
    left: int,
    right: int,
    top: int,
    bottom: int,
    kind: MediaKind,
) -> Box:
    scan_width, scan_height = scan_size
    max_trim_x = int(box.width * _trim_max_edge_ratio(kind))
    max_trim_y = int(box.height * _trim_max_edge_ratio(kind))
    left_trim = min(int(left * box.width / scan_width), max_trim_x)
    right_trim = min(int(right * box.width / scan_width), max_trim_x)
    top_trim = min(int(top * box.height / scan_height), max_trim_y)
    bottom_trim = min(int(bottom * box.height / scan_height), max_trim_y)
    trimmed = Box(
        left=box.left + left_trim,
        top=box.top + top_trim,
        right=box.right - right_trim,
        bottom=box.bottom - bottom_trim,
    )
    if trimmed.width < box.width * (1.0 - (2 * _trim_max_edge_ratio(kind))):
        return box
    if trimmed.height < box.height * (1.0 - (2 * _trim_max_edge_ratio(kind))):
        return box
    if trimmed.left >= trimmed.right or trimmed.top >= trimmed.bottom:
        return box
    return trimmed


def _trim_max_edge_ratio(kind: MediaKind) -> float:
    if kind == "portrait":
        return PORTRAIT_TRIM_MAX_EDGE_RATIO
    return SIGNATURE_TRIM_MAX_EDGE_RATIO


def _trim_keep_ratio(kind: MediaKind) -> float:
    if kind == "portrait":
        return PORTRAIT_TRIM_KEEP_RATIO
    return SIGNATURE_TRIM_KEEP_RATIO


def _trim_min_active_ratio(kind: MediaKind) -> float:
    if kind == "portrait":
        return PORTRAIT_TRIM_MIN_ACTIVE_RATIO
    return SIGNATURE_TRIM_MIN_ACTIVE_RATIO


def _clamp_box(box: Box, image: Image.Image) -> Box:
    return Box(
        left=max(0, min(image.width - 1, box.left)),
        top=max(0, min(image.height - 1, box.top)),
        right=max(1, min(image.width, box.right)),
        bottom=max(1, min(image.height, box.bottom)),
    )


def _build_image_result(
    image: Image.Image,
    box: Box | None,
    confidence: float,
    method: str,
) -> ExtractedImage:
    if box is None:
        return ExtractedImage(
            present=False,
            data_base64=None,
            bounding_box=None,
            confidence=0.0,
            method=method,
        )

    return ExtractedImage(
        present=True,
        data_base64=_encode_crop(image, box),
        bounding_box=BoundingBox(
            left=box.left,
            top=box.top,
            width=box.width,
            height=box.height,
        ),
        confidence=confidence,
        method=method,
    )


def _encode_crop(image: Image.Image, box: Box) -> str:
    crop = image.crop((box.left, box.top, box.right, box.bottom)).convert("RGB")
    crop.thumbnail((MAX_CROP_EDGE_PIXELS, MAX_CROP_EDGE_PIXELS))
    buffer = BytesIO()
    crop.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
