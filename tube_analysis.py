"""Image measurements and batch exports, independent of the desktop interface.

Coordinates always refer to the decoded, full resolution image. Box right and
bottom edges are exclusive, matching NumPy slices. Heights are vertical spans.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
TUBE_IDS = range(1, 13)


def format_height(height_cm: float | None, missing: str = "") -> str:
    """Report centimetres with one decimal; use the caller's missing-value label."""
    return f"{height_cm:.1f}" if height_cm is not None else missing


def validate_tubes(tubes: list[TubeInfo]) -> None:
    """Validate IDs, labels and inclusion for both setup files and the editor.

    Labels are unique without regard to case, including excluded tubes, so they
    remain unambiguous when those tubes are enabled again.
    """
    for tube in tubes:
        tube.validate()
    if len({tube.number for tube in tubes}) != len(tubes):
        raise ValueError("Each tube must have a unique physical number.")
    if len({tube.name.casefold() for tube in tubes}) != len(tubes):
        raise ValueError("Tube names must be unique.")
    if not any(tube.enabled for tube in tubes):
        raise ValueError("Include at least one tube for analysis.")


@dataclass(frozen=True)
class Box:
    """A nonempty image-space rectangle with exclusive right and bottom edges."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Return the horizontal span in original image pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Return the vertical span in original image pixels."""
        return self.y2 - self.y1

    def validate(self, image_size: tuple[int, int]) -> None:
        """Raise ValueError unless integer coordinates lie inside the given image."""
        width, height = image_size
        if any(type(value) is not int for value in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("Region coordinates must be integers.")
        if not (0 <= self.x1 < self.x2 <= width and 0 <= self.y1 < self.y2 <= height):
            raise ValueError(f"Invalid region {self}; image size is {width} × {height}.")

    @classmethod
    def from_points(cls, start, end, image_size: tuple[int, int]) -> Box:
        """Normalize either drag direction, clamp to the image, and reject empty boxes."""
        width, height = image_size
        xs = sorted(max(0, min(width, float(p[0]))) for p in (start, end))
        ys = sorted(max(0, min(height, float(p[1]))) for p in (start, end))
        box = cls(math.floor(xs[0]), math.floor(ys[0]), math.ceil(xs[1]), math.ceil(ys[1]))
        box.validate(image_size)
        return box

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return a validated NumPy view of the original image region."""
        self.validate((image.shape[1], image.shape[0]))
        return image[self.y1:self.y2, self.x1:self.x2]


@dataclass(frozen=True)
class Reference:
    """A green tape region paired with its known length in centimetres."""
    length_cm: float
    box: Box


@dataclass
class Settings:
    """Color thresholds and geometry options shared by previews and batch runs."""
    red_hue_max: int = 15  # Includes orange-red columns while excluding more beige background.
    red_hue_min_high: int = 160
    red_saturation_min: int = 60
    red_value_min: int = 35
    green_hue_min: int = 35
    green_hue_max: int = 85
    green_saturation_min: int = 45
    green_value_min: int = 35
    min_component_area: int = 12
    recalibrate_each_image: bool = True
    align_each_image: bool = True
    calibration_tolerance: float = 0.05

    def validate(self) -> None:
        """Reject unsupported HSV ranges, component sizes and calibration options."""
        for name in ("red_hue_max", "red_hue_min_high", "green_hue_min", "green_hue_max"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 179:
                raise ValueError(f"{name} must be an integer from 0 to 179.")
        if self.red_hue_max >= self.red_hue_min_high or self.green_hue_min >= self.green_hue_max:
            raise ValueError("Hue range limits must be in increasing order.")
        for name in ("red_saturation_min", "red_value_min", "green_saturation_min", "green_value_min"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError(f"{name} must be an integer from 0 to 255.")
        if type(self.min_component_area) is not int or self.min_component_area < 1:
            raise ValueError("Minimum component area must be a positive integer.")
        if type(self.recalibrate_each_image) is not bool or type(self.align_each_image) is not bool:
            raise ValueError("Calibration and alignment options must be true or false.")
        if not math.isfinite(self.calibration_tolerance) or not 0 < self.calibration_tolerance < 1:
            raise ValueError("Calibration tolerance must be between 0 and 1.")


@dataclass
class TubeInfo:
    """Physical tube identity, output label, inclusion and persisted naming choice."""
    number: int
    name: str = ""
    enabled: bool = True
    use_default_name: bool | None = None

    def __post_init__(self):
        # Infer the mode for older setups that did not store the naming choice.
        """Resolve numeric default labels and infer naming mode for older setups."""
        if self.use_default_name is None:
            self.use_default_name = self.name in ("", str(self.number), f"Tube {self.number}")
        if self.use_default_name:
            self.name = str(self.number)

    def validate(self) -> None:
        """Reject invalid IDs, blank labels, and inconsistent naming choices."""
        if type(self.number) is not int or self.number not in TUBE_IDS:
            raise ValueError("Tube numbers must be integers from 1 to 12.")
        if not isinstance(self.name, str) or not self.name.strip() or self.name != self.name.strip():
            raise ValueError("Tube names must be nonempty, without leading/trailing spaces.")
        if type(self.enabled) is not bool:
            raise ValueError("Tube inclusion must be true or false.")
        if type(self.use_default_name) is not bool:
            raise ValueError("Default naming must be true or false.")
        if self.use_default_name and self.name != str(self.number):
            raise ValueError("A default tube name must match its physical ID.")


@dataclass
class Setup:
    """Reusable full-resolution regions, tube identities and measurement settings."""
    image_size: tuple[int, int]
    references: list[Reference] = field(default_factory=list)
    tubes: list[Box] = field(default_factory=list)
    settings: Settings = field(default_factory=Settings)
    sample_name: str = ""
    tube_info: list[TubeInfo] = field(default_factory=list)

    def __post_init__(self):
        """Supply sequential tube identities when loading older geometry-only setups."""
        if not self.tube_info:
            self.tube_info = [TubeInfo(i) for i in range(1, len(self.tubes) + 1)]

    def selected_tubes(self) -> list[tuple[Box, TubeInfo]]:
        """Return included regions and metadata in their saved output order."""
        return [(box, info) for box, info in zip(self.tubes, self.tube_info) if info.enabled]

    def validate(self) -> None:
        """Check reference lengths, geometry, settings and tube metadata before use."""
        if len(self.image_size) != 2 or any(type(v) is not int or v <= 0 for v in self.image_size):
            raise ValueError("Image width and height must be positive integers.")
        if len(self.references) != 2:
            raise ValueError("Select both green tape references: 64.5 cm and 30 cm.")
        if sorted(r.length_cm for r in self.references) != [30.0, 64.5]:
            raise ValueError("The green tape reference lengths must be 64.5 cm and 30 cm.")
        if not self.tubes:
            raise ValueError("Select at least one tube, or split a rack into tube regions.")
        if len(self.tube_info) != len(self.tubes):
            raise ValueError("Each tube region must have its own identity and name.")
        validate_tubes(self.tube_info)
        self.settings.validate()
        for box in self.tubes + [r.box for r in self.references]:
            box.validate(self.image_size)

    def save(self, path: str | Path) -> None:
        """Validate and atomically replace a version 2 JSON setup file."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 2, **asdict(self)}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> Setup:
        """Read version 1 or 2 JSON, migrate defaults and validate before returning."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data["version"] not in (1, 2):
                raise ValueError("Unsupported setup version.")
            setup = cls(
                image_size=tuple(data["image_size"]),
                references=[Reference(r["length_cm"], Box(**r["box"])) for r in data["references"]],
                tubes=[Box(**b) for b in data["tubes"]],
                settings=Settings(**data["settings"]),
                sample_name=data.get("sample_name", ""),
                tube_info=[TubeInfo(**info) for info in data["tube_info"]] if data["version"] == 2 else [],
            )
            setup.validate()
            return setup
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid setup file: {exc}") from exc


@dataclass(frozen=True)
class Detection:
    """Largest color component, vertical pixel span and detection quality flags."""
    box: Box | None
    height_px: int | None
    status: str
    component_count: int = 0


@dataclass(frozen=True)
class Calibration:
    """Combined pixels-per-centimetre scale, individual tape scales and quality."""
    pixels_per_cm: float
    scales: tuple[float, ...]
    status: str


@dataclass(frozen=True)
class Measurement:
    """One physical tube estimate, retaining full precision and quality flags."""
    tube: int
    height_cm: float | None
    detection: Detection
    status: str
    tube_name: str = ""


def list_images(folder: str | Path) -> list[Path]:
    """List supported visible image files in natural filename order; reject empty folders."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Image folder does not exist: {folder}")
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")]
    files.sort(key=lambda p: [int(s) if s.isdigit() else s.casefold() for s in re.split(r"(\d+)", p.name)])
    if not files:
        raise ValueError(f"No supported images in {folder}.")
    return files


def read_image(path: str | Path) -> np.ndarray:
    # imdecode also handles non-ASCII Windows paths reliably.
    """Decode a BGR image, supporting Unicode paths and reporting read/decode errors."""
    try:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as exc:
        raise ValueError(f"Cannot read image: {path}") from exc
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def color_mask(image: np.ndarray, color: str, settings: Settings) -> np.ndarray:
    """Build a red/orange or green HSV mask and remove isolated noise once."""
    if image.size == 0:
        raise ValueError("The selected region is empty.")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color == "red":
        low = cv2.inRange(hsv, (0, settings.red_saturation_min, settings.red_value_min), (settings.red_hue_max, 255, 255))
        high = cv2.inRange(hsv, (settings.red_hue_min_high, settings.red_saturation_min, settings.red_value_min), (179, 255, 255))
        mask = cv2.bitwise_or(low, high)
    elif color == "green":
        mask = cv2.inRange(hsv, (settings.green_hue_min, settings.green_saturation_min, settings.green_value_min), (settings.green_hue_max, 255, 255))
    else:
        raise ValueError(f"Unknown color: {color}")
    # One small opening removes isolated noise without repeatedly eroding the height.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def detect_span(image: np.ndarray, roi: Box, color: str, settings: Settings) -> Detection:
    """Measure the largest qualifying component and flag clipping or ambiguity."""
    mask = color_mask(roi.crop(image), color, settings)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= settings.min_component_area]
    if not components:
        return Detection(None, None, "not_detected")
    components.sort(key=lambda s: int(s[cv2.CC_STAT_AREA]), reverse=True)
    x, y, w, h, area = map(int, components[0])
    warnings = []
    if y <= 1 or y + h >= roi.height - 1:
        warnings.append("clipped")
    # A second substantial region can mean reflections, an interruption, or a valve.
    if len(components) > 1 and components[1][cv2.CC_STAT_AREA] >= area * 0.2:
        warnings.append("multiple_regions")
    if color == "red" and h < roi.height * 0.05:
        warnings.append("small_region")
    box = Box(roi.x1 + x, roi.y1 + y, roi.x1 + x + w, roi.y1 + y + h)
    return Detection(box, h, ";".join(warnings) or "ok", len(components))


def calibrate(image: np.ndarray, setup: Setup) -> Calibration:
    """Fit pixels = scale × centimetres through the origin using both green tapes.

    Reject missing or clipped tape detections. Differences between the two
    individual scales remain visible as a calibration_disagreement warning.
    """
    setup.validate()
    spans = [detect_span(image, ref.box, "green", setup.settings) for ref in setup.references]
    for ref, span in zip(setup.references, spans):
        if span.height_px is None or span.status != "ok":
            raise ValueError(f"{ref.length_cm:g} cm tape: {span.status}. Adjust its region so the whole tape is isolated.")
    scales = tuple(span.height_px / ref.length_cm for ref, span in zip(setup.references, spans))
    # Least squares fit of pixels = scale × centimetres, with no hidden correction.
    scale = sum(ref.length_cm * span.height_px for ref, span in zip(setup.references, spans)) / sum(ref.length_cm ** 2 for ref in setup.references)
    disagreement = (max(scales) - min(scales)) / (sum(scales) / len(scales))
    status = "calibration_disagreement" if disagreement > setup.settings.calibration_tolerance else "ok"
    return Calibration(scale, scales, status)


def measure_image(image: np.ndarray, setup: Setup, calibration: Calibration | None = None, visible_polygon: np.ndarray | None = None, alignment_status: str = "ok") -> tuple[Calibration, list[Measurement]]:
    """Measure included red columns in setup coordinates, preserving missing heights.

    A supplied calibration reuses a fixed scale; otherwise both tapes are
    measured in this frame. Regions outside visible_polygon stay blank.
    Detection, calibration and alignment warnings are retained together;
    rounding is deferred until display or CSV export.
    """
    setup.validate()
    if (image.shape[1], image.shape[0]) != setup.image_size:
        raise ValueError("image_size_mismatch")
    calibration = calibration or calibrate(image, setup)
    measurements = []
    for roi, info in setup.selected_tubes():
        if visible_polygon is not None and any(cv2.pointPolygonTest(visible_polygon, (float(x), float(y)), False) < 0 for x, y in ((roi.x1, roi.y1), (roi.x2 - 1, roi.y1), (roi.x1, roi.y2 - 1), (roi.x2 - 1, roi.y2 - 1))):
            status = ";".join(s for s in ("outside_frame", alignment_status) if s != "ok")
            measurements.append(Measurement(info.number, None, Detection(None, None, "outside_frame"), status, info.name))
            continue
        detection = detect_span(image, roi, "red", setup.settings)
        height = detection.height_px / calibration.pixels_per_cm if detection.height_px is not None else None
        statuses = [s for s in (detection.status, calibration.status, alignment_status) if s != "ok"]
        measurements.append(Measurement(info.number, height, detection, ";".join(statuses) or "ok", info.name))
    return calibration, measurements


def split_rack(box: Box, count: int, fill: float = 0.55) -> list[Box]:
    """Divide a rack into equally spaced, narrow regions, including empty tubes."""
    if not 1 <= count <= 200:
        raise ValueError("Tube count must be between 1 and 200.")
    if not 0 < fill <= 1 or box.width / count * fill < 3:
        raise ValueError("Rack region is too narrow for that many tubes.")
    pitch = box.width / count
    inset = pitch * (1 - fill) / 2
    return [Box(round(box.x1 + i * pitch + inset), box.y1, round(box.x1 + (i + 1) * pitch - inset), box.y2) for i in range(count)]


def suggest_references(image: np.ndarray, settings: Settings) -> list[Reference]:
    """Propose two tall green components with matching 64.5:30 scale.

    Suggestions must be reviewed; other green objects can have similar geometry.
    """
    height, width = image.shape[:2]
    factor = min(1.0, 1200 / max(width, height))
    small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    _, _, stats, _ = cv2.connectedComponentsWithStats(color_mask(small, "green", settings))
    candidates = [s for s in stats[1:] if s[3] > small.shape[0] * 0.12 and s[3] > s[2] * 5 and s[4] > s[2] * s[3] * 0.35]
    candidates.sort(key=lambda s: int(s[4]), reverse=True)
    pairs = []
    for long in candidates[:20]:
        for short in candidates[:20]:
            if long[3] <= short[3]:
                continue
            ratio_error = abs((long[3] / short[3]) / (64.5 / 30) - 1)
            if ratio_error < 0.15:
                pairs.append((ratio_error, long, short))
    if not pairs:
        raise ValueError("Could not find a reliable pair of green tapes. Draw their regions manually.")
    _, long, short = min(pairs, key=lambda pair: pair[0])
    references = []
    for length, stat in ((64.5, long), (30.0, short)):
        x, y, w, h = (float(v) / factor for v in stat[:4])
        padding = max(8, w * 0.2)
        references.append(Reference(length, Box.from_points((x - padding, y - padding), (x + w + padding, y + h + padding), (width, height))))
    return references


def reference_anchors(image: np.ndarray, setup: Setup, references=None) -> np.ndarray:
    """Return top/bottom tape centres ordered by known length for alignment."""
    points = []
    references = setup.references if references is None else references
    for ref in sorted(references, key=lambda r: r.length_cm, reverse=True):
        detection = detect_span(image, ref.box, "green", setup.settings)
        if detection.box is None or detection.status != "ok":
            raise ValueError(f"alignment_failed: {ref.length_cm:g} cm tape {detection.status}")
        box = detection.box
        points.extend([((box.x1 + box.x2) / 2, box.y1), ((box.x1 + box.x2) / 2, box.y2)])
    return np.asarray(points, dtype=np.float64)


def align_image(image: np.ndarray, setup: Setup, anchors: np.ndarray) -> tuple[np.ndarray, str, np.ndarray]:
    """Fit translation, uniform scale and rotation from the two tape endpoints.

    This preserves relative tube heights and cannot correct perspective/parallax.
    Large or inconsistent changes are rejected rather than guessed.
    """
    if (image.shape[1], image.shape[0]) != setup.image_size:
        raise ValueError("image_size_mismatch")
    references = suggest_references(image, setup.settings)
    source = reference_anchors(image, setup, references)
    source_mean, target_mean = source.mean(axis=0), anchors.mean(axis=0)
    a, b = source - source_mean, anchors - target_mean
    u, singular, vt = np.linalg.svd(a.T @ b)
    correction = np.ones(2)
    correction[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(correction) @ vt
    scale = float(sum(singular * correction) / np.sum(a * a))
    shift = target_mean - scale * source_mean @ rotation
    residual = float(np.sqrt(np.mean(np.sum((scale * source @ rotation + shift - anchors) ** 2, axis=1))))
    long_span = np.linalg.norm(anchors[1] - anchors[0])
    angle = abs(math.degrees(math.atan2(rotation[0, 1], rotation[0, 0])))
    if not 0.5 <= scale <= 2 or angle > 10 or residual > max(5, long_span * 0.02):
        raise ValueError(f"alignment_failed: camera geometry changed too much (residual {residual:.1f} px, rotation {angle:.1f}°)")
    matrix = np.column_stack((scale * rotation.T, shift))
    aligned = cv2.warpAffine(image, matrix, setup.image_size, flags=cv2.INTER_LINEAR)
    width, height = setup.image_size
    corners = np.asarray([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]], dtype=np.float64)
    visible_polygon = cv2.transform(corners, matrix)[0].astype(np.float32)
    status = "alignment_uncertain" if residual > max(3, long_span * 0.0075) else "ok"
    return aligned, status, visible_polygon


@dataclass(frozen=True)
class PreparedFrame:
    """Image in setup coordinates, with alignment quality and visible footprint."""

    image: np.ndarray
    alignment_status: str = "ok"
    visible_polygon: np.ndarray | None = None


class AnalysisSession:
    """Share reference preparation and measurement between preview and batches.

    A batch creates one session and reuses its anchors and optional fixed scale.
    The GUI creates a fresh session after edits so it never uses stale settings.
    Image arrays use OpenCV's BGR order and retain their decoded resolution.
    """

    def __init__(self, setup: Setup, sample_image: np.ndarray | None = None):
        """Validate the setup and prepare only the reference data it requires."""
        setup.validate()
        self.setup = setup
        self.anchors = None
        self.fixed_calibration = None
        needs_sample = setup.settings.align_each_image or not setup.settings.recalibrate_each_image
        if needs_sample:
            if sample_image is None:
                raise ValueError("The setup sample image is required for alignment or fixed calibration.")
            if (sample_image.shape[1], sample_image.shape[0]) != setup.image_size:
                raise ValueError("The calibration sample has a different image size.")
            if setup.settings.align_each_image:
                self.anchors = reference_anchors(sample_image, setup)
            if not setup.settings.recalibrate_each_image:
                self.fixed_calibration = calibrate(sample_image, setup)

    @classmethod
    def from_files(cls, setup: Setup, files: list[Path]) -> AnalysisSession:
        """Load the named sample when needed, rejecting a missing reference early."""
        sample_image = None
        if setup.settings.align_each_image or not setup.settings.recalibrate_each_image:
            sample = next((path for path in files if path.name == setup.sample_name), None)
            if sample is None:
                raise ValueError(f"Setup sample {setup.sample_name!r} is missing from the folder. Choose a new sample in the desktop app.")
            sample_image = read_image(sample)
        return cls(setup, sample_image)

    def prepare(self, image: np.ndarray, *, is_sample: bool = False) -> PreparedFrame:
        """Align a new frame when enabled; the original sample needs no transform."""
        if (image.shape[1], image.shape[0]) != self.setup.image_size:
            raise ValueError("image_size_mismatch")
        if self.anchors is not None and not is_sample:
            return PreparedFrame(*align_image(image, self.setup, self.anchors))
        return PreparedFrame(image)

    def measure(self, frame: PreparedFrame) -> tuple[Calibration, list[Measurement]]:
        """Measure included tubes and combine detection, calibration and alignment flags."""
        return measure_image(frame.image, self.setup, self.fixed_calibration, frame.visible_polygon, frame.alignment_status)


def annotate(image: np.ndarray, setup: Setup, calibration: Calibration, measurements: list[Measurement]) -> np.ndarray:
    """Return an image copy with reference regions, one-decimal heights and span lines."""
    canvas = image.copy()
    thickness = max(2, round(max(image.shape[:2]) / 1200))
    font_scale = max(0.5, max(image.shape[:2]) / 3400)
    def draw(box, color, label):
        """Draw one labelled region on the output copy using original image coordinates."""
        cv2.rectangle(canvas, (box.x1, box.y1), (box.x2 - 1, box.y2 - 1), color, thickness)
        cv2.putText(canvas, label, (box.x1, max(25, box.y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    for ref in setup.references:
        draw(ref.box, (40, 220, 40), f"{ref.length_cm:g} cm")
    regions = {info.number: roi for roi, info in setup.selected_tubes()}
    for measurement in measurements:
        roi = regions[measurement.tube]
        color = (255, 210, 40) if measurement.status == "ok" else (0, 170, 255)
        height = f"{format_height(measurement.height_cm)}cm" if measurement.height_cm is not None else "n/a"
        draw(roi, color, f"T{measurement.tube}:{height}")
        if measurement.detection.box:
            box = measurement.detection.box
            for y in (box.y1, box.y2 - 1):
                cv2.line(canvas, (roi.x1, y), (roi.x2 - 1, y), (255, 255, 0), thickness)
    cv2.putText(canvas, f"{calibration.pixels_per_cm:.3f} px/cm | {calibration.status}", (25, canvas.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def _write_measurements(wide, detail, image_name: str, measurements: list[Measurement], calibration: Calibration | None, error: str = "") -> None:
    """Write matching wide and detailed rows, including blank failed measurements."""
    wide.writerow([image_name] + [format_height(m.height_cm) for m in measurements])
    for measurement in measurements:
        box = measurement.detection.box
        detail.writerow({
            "image": image_name,
            "tube": measurement.tube,
            "tube_name": measurement.tube_name,
            "height_cm": format_height(measurement.height_cm),
            "height_px": measurement.detection.height_px,
            "top_y": box.y1 if box else "",
            "bottom_y": box.y2 if box else "",
            "pixels_per_cm": round(calibration.pixels_per_cm, 6) if calibration else "",
            "status": measurement.status,
            "detail": error,
        })


def run_batch(
    files: list[Path], setup: Setup, output: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    save_previews: bool = True,
    save_plots: bool = True,
) -> dict:
    """Stream a batch to fresh CSV files; retain marked partial output on cancel."""
    setup.validate()
    selected = setup.selected_tubes()
    if not files:
        raise ValueError("No images to analyze.")
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output folder is not empty. Choose a new folder: {output}")
    session = AnalysisSession.from_files(setup, files)
    output.mkdir(parents=True, exist_ok=True)
    setup.save(output / "setup.json")
    summary = {"total_images": len(files), "processed_images": 0, "failed_images": 0, "flagged_measurements": 0, "cancelled": False, "complete": False, "output": str(output)}
    preview_indices = {0, len(files) // 2, len(files) - 1} if save_previews else set()
    fields = ["image", "tube", "tube_name", "height_cm", "height_px", "top_y", "bottom_y", "pixels_per_cm", "status", "detail"]
    with (output / "heights.csv").open("w", newline="", encoding="utf-8") as wide_file, (output / "measurements.csv").open("w", newline="", encoding="utf-8") as detail_file:
        wide = csv.writer(wide_file)
        detail = csv.DictWriter(detail_file, fieldnames=fields)
        wide.writerow(["image"] + [f"{info.name} (cm)" for _, info in selected])
        detail.writeheader()
        try:
            for index, path in enumerate(files):
                if cancelled and cancelled():
                    summary["cancelled"] = True
                    break
                error = ""
                calibration = None
                try:
                    frame = session.prepare(read_image(path), is_sample=path.name == setup.sample_name)
                    calibration, measurements = session.measure(frame)
                except (ValueError, cv2.error) as exc:
                    error = str(exc)
                    summary["failed_images"] += 1
                    measurements = [Measurement(info.number, None, Detection(None, None, "image_error"), "image_error", info.name) for _, info in selected]
                _write_measurements(wide, detail, path.name, measurements, calibration, error)
                summary["flagged_measurements"] += sum(m.status != "ok" for m in measurements)
                if not error and index in preview_indices:
                    preview = annotate(frame.image, setup, calibration, measurements)
                    success, encoded = cv2.imencode(".jpg", preview)
                    if not success:
                        raise OSError("Could not encode preview image.")
                    encoded.tofile(str(output / f"preview_{index + 1:04d}_{path.stem}.jpg"))
                summary["processed_images"] += 1
                if progress:
                    progress(index + 1, len(files), path.name)
            summary["complete"] = not summary["cancelled"] and summary["processed_images"] == len(files)
        except Exception as exc:
            summary["error"] = str(exc)
            raise
        finally:
            (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if save_plots and summary["processed_images"]:
        try:
            from tube_plots import export_plots
            summary["plots"] = export_plots(output / "measurements.csv", output)
        except (ImportError, OSError, ValueError) as exc:
            summary["plot_error"] = str(exc)
        (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
