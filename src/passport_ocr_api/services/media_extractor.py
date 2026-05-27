import base64
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

from PIL import Image

from passport_ocr_api.schemas import BoundingBox, ExtractedImage, PassportImageExtraction

GRID_COLUMNS = 64
GRID_ROWS = 48
MAX_COMPONENTS = 256
MAX_COMPONENT_CELLS = GRID_COLUMNS * GRID_ROWS
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
PORTRAIT_COMPONENT_CONFIDENCE = 0.65
FALLBACK_CONFIDENCE = 0.2
EXCLUDED_REGION_OVERLAP_RATIO = 0.65
JPEG_QUALITY = 85
MAX_CROP_EDGE_PIXELS = 512
PORTRAIT_EXPAND_PADDING_RATIO = 0.005
SIGNATURE_EXPAND_PADDING_RATIO = 0.03
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
SIGNATURE_PORTRAIT_LEFT_PADDING_RATIO = 0.04
SIGNATURE_PORTRAIT_RIGHT_PADDING_RATIO = 0.12
SIGNATURE_PORTRAIT_TOP_RATIO = 0.55
SIGNATURE_PORTRAIT_BOTTOM_RATIO = 1.12
SIGNATURE_PORTRAIT_BOTTOM_PADDING_RATIO = 0.16
SIGNATURE_DISTANCE_WEIGHT = 1.2

CellKind = Literal["photo", "dark"]


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
            return _estimated_main_portrait_box(image, best.box)
        refined = _refine_portrait_box(best.box, image)
        return _expand_box(refined, image, PORTRAIT_EXPAND_PADDING_RATIO)

    def _find_signature(self, image: Image.Image, portrait_box: Box | None) -> MediaCandidate:
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
            box=_expand_box(best.box, image, SIGNATURE_EXPAND_PADDING_RATIO),
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
    return Box(
        left=max(0, skin_box.left - horizontal_padding),
        top=max(0, skin_box.top - top_padding),
        right=min(image.width, skin_box.right + horizontal_padding),
        bottom=min(image.height, skin_box.bottom + bottom_padding),
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
        return _portrait_relative_signature_box(image, portrait_box)

    box = _fractional_box(
        image,
        FALLBACK_SIGNATURE_LEFT_RATIO,
        FALLBACK_SIGNATURE_TOP_RATIO,
        FALLBACK_SIGNATURE_LEFT_RATIO + FALLBACK_SIGNATURE_WIDTH_RATIO,
        FALLBACK_SIGNATURE_TOP_RATIO + FALLBACK_SIGNATURE_HEIGHT_RATIO,
    )
    if _overlaps_excluded_region(box, portrait_box):
        return None
    return box


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


def _expand_box(box: Box, image: Image.Image, padding_ratio: float) -> Box:
    padding_x = int(image.width * padding_ratio)
    padding_y = int(image.height * padding_ratio)
    return Box(
        left=max(0, box.left - padding_x),
        top=max(0, box.top - padding_y),
        right=min(image.width, box.right + padding_x),
        bottom=min(image.height, box.bottom + padding_y),
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
