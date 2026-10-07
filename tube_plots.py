"""Height plots from detailed exports, with stable tube IDs and visible flags."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter


@dataclass
class TubeSeries:
    """Ordered heights and quality flags for one stable physical tube ID."""
    number: int
    name: str
    heights: np.ndarray
    statuses: list[str]


@dataclass
class PlotData:
    """Image-sequence labels and per-tube series loaded from a detailed CSV."""
    images: list[str]
    series: list[TubeSeries]


def load_measurements(path: str | Path) -> PlotData:
    """Read current and legacy detailed CSVs; missing values stay NaN."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"image", "tube", "height_cm", "status"}.issubset(reader.fieldnames or []):
            raise ValueError("Choose measurements.csv, which contains tube identities and quality flags.")
        rows = list(reader)
    if not rows:
        raise ValueError("The measurements file contains no processed images.")
    images = list(dict.fromkeys(row["image"] for row in rows))
    image_indices = {image: i for i, image in enumerate(images)}
    series = {}
    seen = set()
    for row in rows:
        number = int(row["tube"])
        name = row.get("tube_name") or str(number)
        if number not in series:
            series[number] = TubeSeries(number, name, np.full(len(images), np.nan), ["not_detected"] * len(images))
        if series[number].name != name:
            raise ValueError(f"Tube {number} has inconsistent names in the CSV.")
        key = (number, row["image"])
        if key in seen:
            raise ValueError(f"Duplicate measurement for tube {number}, image {row['image']}.")
        seen.add(key)
        index = image_indices[row["image"]]
        if row["height_cm"].strip():
            height = float(row["height_cm"])
            if not math.isfinite(height) or height < 0:
                raise ValueError("Heights must be nonnegative finite numbers or blank.")
            series[number].heights[index] = height
        series[number].statuses[index] = row["status"] or "not_detected"
    return PlotData(images, list(series.values()))


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#6A3D9A", "#8C564B", "#E377C2", "#4C6B22", "#7F7F7F", "#17365D"]


def build_figure(data: PlotData, tube_numbers=None, individual=False, include_flagged=True) -> Figure:
    """Build a combined plot or individual panels with shared height limits.

    Accepted detections form lines with gaps; flagged estimates use crosses.
    Tube colors follow physical IDs, and height labels use one decimal place.
    """
    selected = set(tube_numbers) if tube_numbers is not None else {s.number for s in data.series}
    series = [s for s in data.series if s.number in selected]
    columns = min(3, max(1, len(series))) if individual else 1
    rows = max(1, math.ceil(len(series) / columns)) if individual else 1
    figure = Figure(figsize=(12, rows * 2.8 + 1.5 if individual else 7), facecolor="white")
    axes = figure.subplots(rows, columns, squeeze=False)
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.2 if not individual else 0.15, top=0.8 if not individual else 0.9, hspace=0.6, wspace=0.18)
    if not series:
        axes[0, 0].text(0.5, 0.5, "Select at least one tube to plot", ha="center", va="center", transform=axes[0, 0].transAxes)
        return figure
    x = np.arange(1, len(data.images) + 1)
    max_height = 1.0
    for series_index, tube in enumerate(series):
        ax = axes.flat[series_index] if individual else axes[0, 0]
        color = COLORS[(tube.number - 1) % len(COLORS)]
        good = np.asarray([status == "ok" for status in tube.statuses]) & np.isfinite(tube.heights)
        flagged = ~good & np.isfinite(tube.heights)
        # Keep gaps for unavailable/flagged samples; never interpolate or fill zero.
        ax.plot(x, np.where(good, tube.heights, np.nan), color=color, linewidth=1.6, marker=".", markersize=3, label=tube.name)
        if include_flagged and flagged.any():
            ax.scatter(x[flagged], tube.heights[flagged], marker="x", s=18, linewidths=0.8, color=color, alpha=0.65)
        displayed = good | (flagged if include_flagged else False)
        if displayed.any():
            max_height = max(max_height, float(np.nanmax(tube.heights[displayed])))
        else:
            ax.text(0.5, 0.5, "No accepted detections" if not include_flagged else "No detected heights", ha="center", va="center", color="#666666", transform=ax.transAxes)
        if individual:
            title = tube.name if tube.name in (str(tube.number), f"Tube {tube.number}") else f"{tube.name} · Tube {tube.number}"
            ax.set_title(title, fontsize=11, weight="bold")
    used_axes = list(axes.flat[:len(series)]) if individual else [axes[0, 0]]
    ticks = np.unique(np.linspace(1, max(1, len(data.images)), min(5 if individual else 7, len(data.images)), dtype=int))
    for ax in used_axes:
        ax.set_ylabel("Height (cm)", fontsize=10)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xlabel("Image sequence", fontsize=9)
        ax.set_ylim(0, max_height * 1.08)
        ax.set_xlim(0.5, max(1.5, len(data.images) + 0.5))
        labels = [str(tick) for tick in ticks] if individual else [f"{tick}\n{data.images[tick - 1]}" for tick in ticks]
        ax.set_xticks(ticks, labels, fontsize=8)
        ax.grid(True, color="#DDE2E7", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
    if individual:
        for ax in list(axes.flat)[len(series):]:
            ax.set_visible(False)
        figure.suptitle("Column heights by tube", fontsize=16, weight="bold", y=0.98)
    else:
        ax = axes[0, 0]
        ax.set_title("Column height across images", fontsize=16, weight="bold", pad=12)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=min(4, len(series)), frameon=False, fontsize=9)
    note = "Lines / dots: accepted detections.  ×: flagged estimates." if include_flagged else "Only accepted detections are shown. Flagged estimates are hidden."
    figure.text(0.075, 0.025, note + "  Gaps are preserved; missing heights are never filled with zero.", fontsize=8, color="#555555")
    if individual:
        figure.text(0.075, 0.065, f"Image 1: {data.images[0]}     Image {len(data.images)}: {data.images[-1]}", fontsize=8, color="#555555")
    return figure


def export_plots(measurements: str | Path, output: str | Path) -> list[str]:
    """Save combined and per-tube figures as PNG/PDF, returning their filenames."""
    data = load_measurements(measurements)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for individual, stem in ((False, "height_trends"), (True, "tube_panels")):
        figure = build_figure(data, individual=individual)
        FigureCanvasAgg(figure)
        for extension in ("png", "pdf"):
            path = output / f"{stem}.{extension}"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            paths.append(path.name)
        figure.clear()
    return paths


def show_plot_window(parent, measurements: str | Path):
    """Open native plots with per-tube visibility, quality filtering and export."""
    import tkinter as tk
    from tkinter import filedialog, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    data = load_measurements(measurements)
    window = tk.Toplevel(parent)
    window.title(f"Tube height plots · {Path(measurements).parent.name}")
    window.geometry(f"{min(1250, window.winfo_screenwidth() - 60)}x{min(900, window.winfo_screenheight() - 100)}")
    window.minsize(900, 600)
    controls = ttk.Frame(window, padding=12)
    controls.pack(side="left", fill="y")
    ttk.Label(controls, text="Visible tubes", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(0, 8))
    variables = {}
    figures = []
    plot_frames = []
    include_flagged = tk.BooleanVar(window, value=True)
    def redraw():
        """Rebuild both plot tabs using the current tube and quality filters."""
        numbers = [number for number, variable in variables.items() if variable.get()]
        figures.clear()
        for individual, frame in enumerate(plot_frames):
            for widget in frame.winfo_children():
                widget.destroy()
            figure = build_figure(data, numbers, individual=bool(individual), include_flagged=include_flagged.get())
            figures.append(figure)
            canvas = FigureCanvasTkAgg(figure, master=frame)
            NavigationToolbar2Tk(canvas, frame, pack_toolbar=False).pack(side="bottom", fill="x")
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()
    for series in data.series:
        variables[series.number] = tk.BooleanVar(window, value=True)
        ttk.Checkbutton(controls, text=series.name, variable=variables[series.number], command=redraw).pack(anchor="w", pady=3)
    def select_all(value):
        """Set every plot visibility checkbox, then redraw both tabs once."""
        for variable in variables.values():
            variable.set(value)
        redraw()
    buttons = ttk.Frame(controls)
    buttons.pack(fill="x", pady=10)
    ttk.Button(buttons, text="All", command=lambda: select_all(True)).pack(side="left")
    ttk.Button(buttons, text="None", command=lambda: select_all(False)).pack(side="left", padx=4)
    ttk.Checkbutton(controls, text="Show flagged estimates (×)", variable=include_flagged, command=redraw).pack(anchor="w", pady=8)
    ttk.Label(controls, text=f"{len(data.images)} images\nX axis: image sequence\nY axis: height in cm\n\nUse the toolbar to pan,\nzoom and inspect plots.\n\nPlot visibility does not change\nwhich tubes were analyzed.", justify="left").pack(anchor="w", pady=10)
    notebook = ttk.Notebook(window)
    notebook.pack(side="right", fill="both", expand=True, padx=(0, 10), pady=10)
    for title in ("Combined", "Individual tubes"):
        frame = ttk.Frame(notebook)
        plot_frames.append(frame)
        notebook.add(frame, text=title)
    def save():
        """Export the active plot tab to the selected PNG or PDF path."""
        path = filedialog.asksaveasfilename(parent=window, initialdir=Path(measurements).parent, initialfile="height_plot.png", defaultextension=".png", filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf")])
        if path:
            figures[notebook.index("current")].savefig(path, dpi=180, bbox_inches="tight")
    ttk.Button(controls, text="Export displayed plot…", command=save).pack(fill="x", pady=10)
    redraw()
    return window
