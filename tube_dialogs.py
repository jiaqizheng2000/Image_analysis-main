"""Modal editors for detection settings, tube labels and analysis inclusion.

Editors work on local drafts. They validate before invoking the supplied apply
callback, so closing a dialog never changes the main application's setup.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, ttk
from typing import Callable

from tube_analysis import TUBE_IDS, Settings, TubeInfo, validate_tubes
from tube_selector import Region


def _modal_dialog(parent, title: str) -> tuple[tk.Toplevel, ttk.Frame]:
    """Create a parent-owned modal window with a consistently padded body."""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    body = ttk.Frame(dialog, padding=16)
    body.pack(fill="both", expand=True)
    return dialog, body


class _TubeRow:
    """Keep one tube's editable draft and naming-mode transitions together."""

    def __init__(self, body, row: int, index: int, region: Region):
        """Place row controls without modifying the original region."""
        self.index = index
        self.region = region
        info = region.tube_info(row - 1)
        self.included = tk.BooleanVar(body, value=info.enabled)
        self.number = tk.IntVar(body, value=info.number)
        self.use_default = tk.BooleanVar(body, value=info.use_default_name)
        self.name = tk.StringVar(body, value=info.name)
        self._was_default = info.use_default_name
        self._custom_name = "" if info.use_default_name else info.name
        ttk.Checkbutton(body, variable=self.included).grid(
            row=row, column=0, padx=8, pady=4
        )
        ttk.Combobox(
            body,
            textvariable=self.number,
            values=list(TUBE_IDS),
            state="readonly",
            width=5,
        ).grid(row=row, column=1, padx=8)
        ttk.Checkbutton(body, variable=self.use_default).grid(row=row, column=2, padx=8)
        self.entry = ttk.Entry(
            body,
            textvariable=self.name,
            width=28,
            state="readonly" if info.use_default_name else "normal",
        )
        self.entry.grid(row=row, column=3, padx=8)
        self.use_default.trace_add("write", self._refresh_name)
        self.number.trace_add("write", self._refresh_name)

    def _refresh_name(self, *_):
        """Follow ID edits in default mode, retaining custom drafts on toggles."""
        if self.use_default.get():
            if not self._was_default:
                self._custom_name = self.name.get()
            self.name.set(str(self.number.get()))
            self.entry.configure(state="readonly")
        else:
            if self._was_default:
                self.name.set(self._custom_name or str(self.number.get()))
            self.entry.configure(state="normal")
        self._was_default = self.use_default.get()

    def read(self) -> Region:
        """Return the edited region with a normalized label and explicit mode."""
        info = TubeInfo(
            self.number.get(),
            self.name.get().strip(),
            self.included.get(),
            self.use_default.get(),
        )
        return replace(
            self.region,
            tube_number=info.number,
            tube_name=info.name,
            enabled=info.enabled,
            use_default_name=info.use_default_name,
        )


def show_tube_editor(
    parent, regions: list[Region], on_apply: Callable[[list[Region]], None]
) -> tk.Toplevel:
    """Edit identities, labels and inclusion while preserving region geometry."""
    dialog, body = _modal_dialog(parent, "Tube names and analysis selection")
    ttk.Label(
        body,
        text=(
            "Default labels are 1–12, matching the physical tube IDs.\n"
            "Uncheck Default to enter a name or label. "
            "Only included tubes appear in CSVs and plots."
        ),
    ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))
    for column, text in enumerate(
        ("Include", "Tube ID (1–12)", "Default", "CSV / plot name or label")
    ):
        ttk.Label(body, text=text).grid(row=1, column=column, sticky="w", padx=8)
    rows = []
    for index, region in enumerate(regions):
        if region.kind == "tube":
            rows.append(_TubeRow(body, len(rows) + 2, index, region))
    bar = ttk.Frame(body)
    bar.grid(row=len(rows) + 2, column=0, columnspan=4, sticky="ew", pady=(12, 0))

    def include_all(value: bool) -> None:
        """Change every inclusion checkbox without altering names or geometry."""
        for row in rows:
            row.included.set(value)

    def default_names() -> None:
        """Reset all row labels to their current physical tube IDs."""
        for row in rows:
            row.use_default.set(True)

    def apply() -> None:
        """Validate the complete draft and commit it through the owner callback."""
        updated = regions.copy()
        try:
            for row in rows:
                updated[row.index] = row.read()
            validate_tubes(
                [region.tube_info() for region in updated if region.kind == "tube"]
            )
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Tube selection", str(exc), parent=dialog)
            return
        on_apply(updated)
        dialog.destroy()

    ttk.Button(bar, text="Include all", command=lambda: include_all(True)).pack(
        side="left"
    )
    ttk.Button(bar, text="Include none", command=lambda: include_all(False)).pack(
        side="left", padx=6
    )
    ttk.Button(bar, text="Use default labels", command=default_names).pack(
        side="left", padx=6
    )
    ttk.Button(bar, text="Apply", command=apply).pack(side="right")
    return dialog


def show_settings_editor(
    parent, settings: Settings, on_apply: Callable[[Settings], None]
) -> tk.Toplevel:
    """Edit a settings copy and commit only validated thresholds and options."""
    dialog, body = _modal_dialog(parent, "Detection settings")
    labels = (
        ("red_hue_max", "Red/orange upper hue (0–179)"),
        ("red_hue_min_high", "Second red range lower hue (0–179)"),
        ("red_saturation_min", "Red minimum saturation (0–255)"),
        ("red_value_min", "Red minimum brightness (0–255)"),
        ("green_hue_min", "Green lower hue (0–179)"),
        ("green_hue_max", "Green upper hue (0–179)"),
        ("green_saturation_min", "Green minimum saturation (0–255)"),
        ("green_value_min", "Green minimum brightness (0–255)"),
        ("min_component_area", "Minimum component area (pixels)"),
    )
    fields = {}
    for row, (name, label) in enumerate(labels):
        ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=4)
        fields[name] = tk.StringVar(dialog, value=str(getattr(settings, name)))
        ttk.Entry(body, textvariable=fields[name], width=8).grid(
            row=row, column=1, padx=10
        )
    per_image = tk.BooleanVar(dialog, value=settings.recalibrate_each_image)
    alignment = tk.BooleanVar(dialog, value=settings.align_each_image)
    ttk.Checkbutton(
        body, text="Measure both tapes in every image", variable=per_image
    ).grid(row=9, column=0, columnspan=2, sticky="w", pady=10)
    ttk.Checkbutton(
        body, text="Align camera movement using both green tapes", variable=alignment
    ).grid(row=10, column=0, columnspan=2, sticky="w", pady=4)
    ttk.Label(
        body,
        text=(
            "Alignment corrects translation, uniform scale and small rotation.\n"
            "It cannot correct perspective changes. Inspect early and late previews.\n"
            "With per-image calibration off, frames use the setup sample's scale."
        ),
    ).grid(row=11, column=0, columnspan=2, sticky="w")

    def apply() -> None:
        """Reject invalid values before notifying the main application."""
        try:
            updated = replace(
                settings,
                **{name: int(variable.get()) for name, variable in fields.items()},
                recalibrate_each_image=per_image.get(),
                align_each_image=alignment.get(),
            )
            updated.validate()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=dialog)
            return
        dialog.destroy()
        on_apply(updated)

    ttk.Button(body, text="Apply settings", command=apply).grid(
        row=12, column=0, columnspan=2, sticky="ew", pady=(12, 0)
    )
    return dialog
