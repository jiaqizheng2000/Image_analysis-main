"""Shared macOS-friendly Tk canvas selection, with image-space coordinates.

OpenCV is used for image processing only. Tk owns all pointer events, geometry,
and overlays, avoiding resized HighGUI/Retina mouse-coordinate inconsistencies.
"""

from __future__ import annotations

import base64
import math
import tkinter as tk
from dataclasses import dataclass, replace

import cv2
import numpy as np

from tube_analysis import Box, TubeInfo, format_height


@dataclass(frozen=True)
class Region:
    """Immutable canvas geometry with optional physical tube metadata."""

    kind: str
    box: Box
    tube_number: int | None = None
    tube_name: str = ""
    enabled: bool = True
    use_default_name: bool | None = None

    def tube_info(self, fallback_number: int = 1) -> TubeInfo:
        """Convert canvas metadata to a tube identity, supplying missing old IDs."""
        if self.kind != "tube":
            raise ValueError("Only tube regions have tube identities.")
        return TubeInfo(
            self.tube_number or fallback_number,
            self.tube_name,
            self.enabled,
            self.use_default_name,
        )


@dataclass
class ViewTransform:
    """Map between original image coordinates and the zoomed/panned canvas."""

    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def to_image(self, x, y):
        """Convert a canvas position to original floating-point image coordinates."""
        return ((x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale)

    def to_canvas(self, x, y):
        """Convert an original image position to its displayed canvas position."""
        return (x * self.scale + self.offset_x, y * self.scale + self.offset_y)

    def zoom(self, factor, x, y):
        """Change scale while keeping the image point under the pointer stationary."""
        ix, iy = self.to_image(x, y)
        self.scale *= factor
        self.offset_x = x - ix * self.scale
        self.offset_y = y - iy * self.scale


class ImageCanvas(tk.Canvas):
    """Native Tk selection canvas with live overlays, zoom, pan and undoable edits."""

    def __init__(self, master, on_box=None, on_change=None, **kwargs):
        """Initialize canvas state and bind native pointer and keyboard gestures."""
        super().__init__(master, background="#20282d", highlightthickness=0, **kwargs)
        self.image = None
        self.regions: list[Region] = []
        self.history: list[list[Region]] = []
        self.selected = None
        self.mode = "tube"
        self.transform = ViewTransform()
        self.on_box = on_box
        self.on_change = on_change
        self.locked = False
        self.preview = []
        self._photo = None
        self._gesture = None
        self._space = False
        self._resize_job = None
        self.bind("<Configure>", self._resize)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        for button in (2, 3):
            self.bind(f"<ButtonPress-{button}>", self._pan_press)
            self.bind(f"<B{button}-Motion>", self._motion)
            self.bind(f"<ButtonRelease-{button}>", self._release)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda e: self.zoom(1.12, e.x, e.y))
        self.bind("<Button-5>", lambda e: self.zoom(1 / 1.12, e.x, e.y))
        self.bind("<KeyPress-space>", lambda e: setattr(self, "_space", True))
        self.bind("<KeyRelease-space>", lambda e: setattr(self, "_space", False))
        self.bind("<FocusOut>", lambda e: setattr(self, "_space", False))
        self.bind("<Escape>", self._cancel_gesture)
        self.bind("<Delete>", lambda e: self.delete_selected())
        self.bind("<BackSpace>", lambda e: self.delete_selected())
        for key in ("<Command-z>", "<Control-z>"):
            self.bind(key, lambda e: self.undo())

    @property
    def image_size(self):
        """Return decoded image width and height, independent of display scaling."""
        return (self.image.shape[1], self.image.shape[0])

    def set_image(self, image, fit=True):
        """Display a source image and optionally refit it to the current viewport."""
        self.image = image
        if fit:
            self.fit()
        else:
            self._render()

    def set_regions(self, regions, remember=True):
        """Replace the complete region state and optionally record it for undo."""
        if self.locked:
            return
        if remember and regions != self.regions:
            self._remember(self.regions)
        self.regions = list(regions)
        self.selected = None
        self._changed()

    def _remember(self, regions: list[Region]) -> None:
        """Store a complete region snapshot and retain at most 100 undo entries."""
        self.history.append(regions.copy())
        self.history = self.history[-100:]

    def _changed(self):
        """Clear old measurements, redraw regions and notify the owner of an edit."""
        self.preview = []
        self._draw_regions()
        if self.on_change:
            self.on_change()

    def show_measurements(self, measurements):
        """Replace only measurement overlays, leaving the source image and history
        intact.
        """
        self.preview = list(measurements)
        self._draw_regions()

    def undo(self):
        """Restore the preceding recorded region state when editing is unlocked."""
        if not self.locked and self.history:
            self.regions = self.history.pop()
            self.selected = None
            self._changed()

    def delete_selected(self):
        """Remove the current region through the shared undoable state update."""
        if not self.locked and self.selected is not None:
            self.set_regions(
                [r for i, r in enumerate(self.regions) if i != self.selected]
            )

    def select(self, index):
        """Select a region and activate move/resize editing."""
        self.selected = index
        self.mode = "edit"
        self._draw_regions()

    def fit(self):
        """Centre the whole source image in the current canvas with a small margin."""
        if self.image is None:
            return
        w, h = max(1, self.winfo_width()), max(1, self.winfo_height())
        iw, ih = self.image_size
        scale = min((w - 20) / iw, (h - 20) / ih)
        scale = max(0.001, scale)
        self.transform = ViewTransform(
            scale, (w - iw * scale) / 2, (h - ih * scale) / 2
        )
        self._render()

    def zoom(self, factor, x=None, y=None):
        """Apply bounded zoom around a pointer position or the viewport centre."""
        if self.image is None:
            return
        scale = self.transform.scale
        fit_scale = min(
            self.winfo_width() / self.image_size[0],
            self.winfo_height() / self.image_size[1],
        )
        target = max(fit_scale * 0.25, min(4.0, scale * factor))
        self.transform.zoom(
            target / scale,
            self.winfo_width() / 2 if x is None else x,
            self.winfo_height() / 2 if y is None else y,
        )
        self._render()

    def _wheel(self, event):
        """Translate native scroll direction into pointer-anchored zoom."""
        if event.delta:
            self.zoom(1.12 if event.delta > 0 else 1 / 1.12, event.x, event.y)

    def _resize(self, event):
        """Debounce resize events so fitting waits for the window geometry to settle."""
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(100, self._after_resize)

    def _after_resize(self):
        """Refit the image after the pending resize delay completes."""
        self._resize_job = None
        self.fit()

    def _render(self):
        """Render the visible image viewport and redraw vector overlays above it."""
        if self.image is None:
            return
        t = self.transform
        matrix = np.array(
            [[t.scale, 0, t.offset_x], [0, t.scale, t.offset_y]], dtype=np.float64
        )
        # Render only the visible viewport, even when zoomed into a 20 MP photo.
        viewport = cv2.warpAffine(
            self.image,
            matrix,
            (max(1, self.winfo_width()), max(1, self.winfo_height())),
            flags=cv2.INTER_LINEAR,
            borderValue=(45, 40, 32),
        )
        encoded = cv2.imencode(".png", viewport)[1]
        self._photo = tk.PhotoImage(master=self, data=base64.b64encode(encoded))
        self.delete("raster")
        self.create_image(0, 0, image=self._photo, anchor="nw", tags="raster")
        self.tag_lower("raster")
        self._draw_regions()

    def _draw_regions(self):
        """Draw regions, physical IDs, quality colors, span lines and resize handles."""
        self.delete("region")
        tube_index = 0
        measurements = {m.tube: m for m in self.preview}
        for i, region in enumerate(self.regions):
            box = region.box
            x1, y1 = self.transform.to_canvas(box.x1, box.y1)
            x2, y2 = self.transform.to_canvas(box.x2, box.y2)
            color = "#66e0a3" if region.kind != "tube" else "#6fcfff"
            label = region.kind + " cm"
            if region.kind == "tube":
                tube_index += 1
                number = region.tube_number or tube_index
                label = f"T{number}"
                if not region.enabled:
                    color = "#929a9f"
                    label += " (off)"
                if region.enabled and number in measurements:
                    measurement = measurements[number]
                    label += f": {format_height(measurement.height_cm, 'n/a')}"
                    if measurement.status != "ok":
                        color = "#ffbb55"
                    if measurement.detection.box:
                        for y in (
                            measurement.detection.box.y1,
                            measurement.detection.box.y2 - 1,
                        ):
                            _, cy = self.transform.to_canvas(0, y)
                            self.create_line(
                                x1, cy, x2, cy, fill="#ffe36b", width=3, tags="region"
                            )
            self.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=color,
                width=3 if i == self.selected else 2,
                dash=() if region.enabled else (4, 4),
                tags="region",
            )
            label_y = max(2, y1 - 21)
            text = self.create_text(
                x1 + 2,
                label_y,
                text=label,
                anchor="nw",
                fill=color,
                font=("TkDefaultFont", 11, "bold"),
                tags="region",
            )
            background = self.create_rectangle(
                *self.bbox(text), fill="#20282d", outline="", tags="region"
            )
            self.tag_lower(background, text)
            if i == self.selected:
                for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                    self.create_rectangle(
                        x - 5,
                        y - 5,
                        x + 5,
                        y + 5,
                        fill=color,
                        outline="#20282d",
                        tags="region",
                    )

    def _point(self, event):
        """Translate a pointer event into image coordinates clamped to the source
        bounds.
        """
        x, y = self.transform.to_image(event.x, event.y)
        width, height = self.image_size
        return max(0, min(width, x)), max(0, min(height, y))

    def _pan_press(self, event):
        """Remember the starting pointer and offsets for a viewport pan gesture."""
        if self.image is not None:
            self.focus_set()
            self._gesture = (
                "pan",
                event.x,
                event.y,
                self.transform.offset_x,
                self.transform.offset_y,
            )

    def _press(self, event):
        """Begin drawing, moving, resizing or panning from a native pointer event."""
        self.focus_set()
        if self.image is None or self.locked:
            return
        if self._space:
            return self._pan_press(event)
        point = self._point(event)
        if self.mode == "edit":
            # The selected region's corner handles take precedence over overlaps.
            if self.selected is not None:
                box = self.regions[self.selected].box
                for x, y, opposite in (
                    (box.x1, box.y1, (box.x2, box.y2)),
                    (box.x2, box.y1, (box.x1, box.y2)),
                    (box.x1, box.y2, (box.x2, box.y1)),
                    (box.x2, box.y2, (box.x1, box.y1)),
                ):
                    cx, cy = self.transform.to_canvas(x, y)
                    if math.hypot(cx - event.x, cy - event.y) <= 10:
                        self._gesture = ("resize", opposite, self.regions.copy())
                        return
            self.selected = next(
                (
                    i
                    for i in reversed(range(len(self.regions)))
                    if self.regions[i].box.x1 <= point[0] <= self.regions[i].box.x2
                    and self.regions[i].box.y1 <= point[1] <= self.regions[i].box.y2
                ),
                None,
            )
            if self.selected is not None:
                self._gesture = ("move", point, self.regions.copy())
            self._draw_regions()
        else:
            self._gesture = ("draw", point)

    def _motion(self, event):
        """Update the active gesture live without committing an undo-history entry."""
        if not self._gesture:
            return
        kind, *args = self._gesture
        if kind == "pan":
            x, y, ox, oy = args
            self.transform.offset_x = ox + event.x - x
            self.transform.offset_y = oy + event.y - y
            self._render()
            return
        point = self._point(event)
        if kind == "draw":
            self.delete("draft")
            a = self.transform.to_canvas(*args[0])
            b = self.transform.to_canvas(*point)
            self.create_rectangle(
                *a, *b, outline="#ffffff", width=2, dash=(5, 3), tags="draft"
            )
        elif kind in ("resize", "move"):
            start, before = args
            original = before[self.selected]
            try:
                if kind == "resize":
                    box = Box.from_points(start, point, self.image_size)
                else:
                    box = original.box
                    dx = round(
                        max(
                            -box.x1,
                            min(self.image_size[0] - box.x2, point[0] - start[0]),
                        )
                    )
                    dy = round(
                        max(
                            -box.y1,
                            min(self.image_size[1] - box.y2, point[1] - start[1]),
                        )
                    )
                    box = Box(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy)
                self.regions[self.selected] = replace(original, box=box)
                self.preview = []
                self._draw_regions()
            except ValueError:
                pass

    def _release(self, event):
        """Commit a valid draw or geometry edit once the pointer is released."""
        if not self._gesture:
            return
        self._motion(event)
        kind, *args = self._gesture
        self._gesture = None
        self.delete("draft")
        if kind == "draw":
            try:
                box = Box.from_points(args[0], self._point(event), self.image_size)
            except ValueError:
                return
            if box.width >= 3 and box.height >= 3 and self.on_box:
                self.on_box(box)
        elif kind in ("resize", "move") and self.regions != args[1]:
            self._remember(args[1])
            self._changed()

    def _cancel_gesture(self, event=None):
        """Cancel a draft or restore pre-drag geometry without committing the edit."""
        if self._gesture and self._gesture[0] in ("move", "resize"):
            self.regions = self._gesture[2]
        self._gesture = None
        self.delete("draft")
        self._draw_regions()
