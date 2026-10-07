"""Desktop setup, visual verification and batch analysis. Run this file to start."""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from tube_analysis import (
    AnalysisSession,
    PreparedFrame,
    Reference,
    Settings,
    Setup,
    TUBE_IDS,
    color_mask,
    format_height,
    list_images,
    read_image,
    run_batch,
    split_rack,
    suggest_references,
)
from tube_dialogs import show_settings_editor, show_tube_editor
from tube_selector import ImageCanvas, Region


PROJECT = Path(__file__).resolve().parent


class TubeApp:
    """Coordinate desktop setup, measurement previews and a cancellable batch worker."""

    def __init__(self, root, folder=None):
        """Initialize window state and schedule folder loading on the Tk event loop."""
        self.root = root
        root.title("Tube height analysis")
        root.geometry(
            f"{min(1380, root.winfo_screenwidth() - 60)}x"
            f"{min(920, root.winfo_screenheight() - 100)}"
        )
        root.minsize(1000, 700)
        self.folder = None
        self.files = []
        self.image = None
        self.settings = Settings()
        self.worker = None
        self.running = False
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.closing = False
        self.last_output = None
        self.dirty = False
        self.mask_visible = False
        self.visible_polygon = None
        self.alignment_status = "ok"
        self.folder_text = tk.StringVar(root, value="Choose an image folder to start")
        self.sample = tk.StringVar(root)
        self.setup_sample = ""
        self.mode = tk.StringVar(root, value="tube")
        self.count = tk.IntVar(root, value=12)
        self.draw_tube_number = tk.IntVar(root, value=1)
        self.status = tk.StringVar(
            root, value="Select the two tapes, select tubes, then preview and analyze."
        )
        self.calibration_text = tk.StringVar(
            root, value="Calibration: select both green tapes"
        )
        self._build()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(100, self._poll)
        if folder:
            root.after(150, lambda: self.load_folder(folder))

    def _build(self):
        """Arrange the folder header, sidebar and image viewer in the main window."""
        header = ttk.Frame(self.root, padding=12)
        header.pack(fill="x")
        ttk.Button(
            header, text="Choose image folder…", command=self.choose_folder
        ).pack(side="left")
        ttk.Label(header, textvariable=self.folder_text, padding=(12, 0)).pack(
            side="left", fill="x", expand=True
        )
        sample_bar = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        sample_bar.pack(fill="x")
        ttk.Label(sample_bar, text="Sample image:").pack(side="left")
        self.sample_combo = ttk.Combobox(
            sample_bar, textvariable=self.sample, state="readonly", width=35
        )
        self.sample_combo.pack(side="left", padx=8)
        self.sample_combo.bind("<<ComboboxSelected>>", lambda e: self.change_sample())
        ttk.Label(
            sample_bar,
            text=(
                "Other frames align to your setup using the green tapes. "
                "Check early and late images."
            ),
        ).pack(side="left", padx=8)
        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12)
        sidebar = ttk.Frame(body, width=310)
        body.add(sidebar, weight=0)
        right = ttk.Frame(body)
        body.add(right, weight=1)

        self._build_sidebar(sidebar)
        self._build_viewer(right)
        ttk.Label(
            self.root, textvariable=self.status, padding=12, wraplength=1280
        ).pack(fill="x")

    def _build_sidebar(self, parent):
        """Build the setup, preview and batch controls in the left notebook."""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)
        setup_tab, analyze_tab = ttk.Frame(notebook, padding=6), ttk.Frame(
            notebook, padding=6
        )
        notebook.add(setup_tab, text="Set up regions")
        notebook.add(analyze_tab, text="Preview / analyze")
        tools = ttk.LabelFrame(setup_tab, text="1 · Set up regions", padding=10)
        tools.pack(fill="x", pady=(0, 8))
        ttk.Button(
            tools, text="Find green tapes automatically", command=self.find_tapes
        ).pack(fill="x", pady=(0, 6))
        for text, value in (
            ("Draw 64.5 cm tape region", "64.5"),
            ("Draw 30 cm tape region", "30"),
            ("Draw one tube region", "tube"),
            ("Draw rack and divide into tubes", "rack"),
            ("Move / resize existing regions", "edit"),
        ):
            ttk.Radiobutton(
                tools,
                text=text,
                value=value,
                variable=self.mode,
                command=self.change_mode,
            ).pack(anchor="w", pady=2)
        count_row = ttk.Frame(tools)
        count_row.pack(fill="x", pady=6)
        ttk.Label(count_row, text="Tubes in rack:").pack(side="left")
        ttk.Spinbox(count_row, from_=1, to=12, textvariable=self.count, width=5).pack(
            side="left", padx=8
        )
        id_row = ttk.Frame(tools)
        id_row.pack(fill="x", pady=(0, 6))
        ttk.Label(id_row, text="Tube ID to draw:").pack(side="left")
        ttk.Combobox(
            id_row,
            textvariable=self.draw_tube_number,
            values=list(TUBE_IDS),
            state="readonly",
            width=5,
        ).pack(side="left", padx=8)
        ttk.Label(
            tools,
            text=(
                "Draw a rack from half a tube spacing before the\n"
                "first centre to half a spacing after the last.\n"
                "Keep valves and lower reservoirs outside it.\n"
                "Review and adjust every suggested tube region."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 6))
        row = ttk.Frame(tools)
        row.pack(fill="x")
        for text, command in (
            ("Undo", self.undo),
            ("Delete", self.delete_region),
            ("Clear", self.clear),
        ):
            ttk.Button(row, text=text, command=command, width=7).pack(
                side="left", padx=(0, 4)
            )

        self.region_list = tk.Listbox(setup_tab, height=8, exportselection=False)
        self.region_list.pack(fill="both", expand=True, pady=(0, 6))
        self.region_list.bind("<<ListboxSelect>>", self.select_region)
        ttk.Button(
            setup_tab, text="Tube names / include in analysis…", command=self.edit_tubes
        ).pack(fill="x", pady=(0, 6))
        ttk.Button(
            setup_tab, text="Sort regions left → right", command=self.order_tubes
        ).pack(fill="x", pady=(0, 6))
        save_row = ttk.Frame(setup_tab)
        save_row.pack(fill="x", pady=(0, 8))
        ttk.Button(save_row, text="Save setup", command=self.save_setup).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(save_row, text="Load setup…", command=self.load_setup).pack(
            side="left", fill="x", expand=True, padx=(6, 0)
        )

        verify = ttk.LabelFrame(analyze_tab, text="2 · Check measurements", padding=10)
        verify.pack(fill="x", pady=(0, 8))
        ttk.Label(verify, textvariable=self.calibration_text, wraplength=280).pack(
            anchor="w", pady=(0, 6)
        )
        ttk.Button(verify, text="Preview heights", command=self.preview).pack(
            fill="x", pady=2
        )
        ttk.Button(
            verify, text="Show / hide color mask", command=self.toggle_mask
        ).pack(fill="x", pady=2)
        ttk.Button(verify, text="Detection settings…", command=self.edit_settings).pack(
            fill="x", pady=2
        )

        batch = ttk.LabelFrame(analyze_tab, text="3 · Analyze folder", padding=10)
        batch.pack(fill="x")
        self.run_button = ttk.Button(
            batch, text="Analyze all images", command=self.start_batch
        )
        self.run_button.pack(fill="x", pady=2)
        self.cancel_button = ttk.Button(
            batch,
            text="Stop after current image",
            command=self.stop.set,
            state="disabled",
        )
        self.cancel_button.pack(fill="x", pady=2)
        ttk.Button(batch, text="Show latest results", command=self.show_results).pack(
            fill="x", pady=2
        )
        ttk.Button(
            batch, text="View plots / open results CSV…", command=self.show_plots
        ).pack(fill="x", pady=2)
        self.progress = ttk.Progressbar(batch, mode="determinate")
        self.progress.pack(fill="x", pady=(6, 0))

    def _build_viewer(self, parent):
        """Build image navigation, the selection canvas and measurement table."""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", pady=(0, 6), padx=(10, 0))
        for text, command in (
            ("Fit image", lambda: self.canvas.fit()),
            ("+", lambda: self.canvas.zoom(1.3)),
            ("−", lambda: self.canvas.zoom(1 / 1.3)),
        ):
            ttk.Button(toolbar, text=text, command=command).pack(
                side="left", padx=(0, 4)
            )
        ttk.Label(
            toolbar,
            text=(
                "Scroll to zoom · Space + drag or right drag to pan · "
                "Corner handles resize"
            ),
        ).pack(side="left", padx=8)
        self.canvas = ImageCanvas(
            parent, on_box=self.add_box, on_change=self.regions_changed
        )
        self.canvas.pack(fill="both", expand=True, padx=(10, 0))
        result_frame = ttk.Frame(parent)
        result_frame.pack(fill="x", pady=(6, 0), padx=(10, 0))
        self.result_list = ttk.Treeview(
            result_frame,
            columns=("tube", "height", "status"),
            show="headings",
            height=5,
        )
        for name, title, width in (
            ("tube", "Tube", 80),
            ("height", "Height (cm)", 130),
            ("status", "Detection status", 300),
        ):
            self.result_list.heading(name, text=title)
            self.result_list.column(name, width=width)
        self.result_list.pack(side="left", fill="x", expand=True)
        result_scroll = ttk.Scrollbar(
            result_frame, orient="vertical", command=self.result_list.yview
        )
        result_scroll.pack(side="right", fill="y")
        self.result_list.configure(yscrollcommand=result_scroll.set)

    def busy(self) -> bool:
        """Report whether a running batch currently prevents setup changes."""
        if self.running:
            self.status.set("Analysis is running. Stop it before changing the setup.")
            return True
        return False

    def choose_folder(self):
        """Prompt for an image folder and load it when a selection is made."""
        if self.busy():
            return
        folder = filedialog.askdirectory(
            parent=self.root,
            initialdir=self.folder or PROJECT / "images",
            title="Choose the folder containing your images",
        )
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        """Load naturally ordered images and restore the folder setup when present."""
        if self.busy():
            return
        if self.dirty and not messagebox.askyesno(
            "Unsaved setup",
            "Discard unsaved setup changes and open another folder?",
            parent=self.root,
        ):
            return
        try:
            files = list_images(folder)
            image = read_image(files[0])
        except (ValueError, cv2.error) as exc:
            return self.error(exc)
        self.folder, self.files, self.image = (
            Path(folder).expanduser().resolve(),
            files,
            image,
        )
        self.folder_text.set(f"{self.folder}  ·  {len(files)} images")
        self.sample_combo.configure(values=[p.name for p in files])
        self.sample.set(files[0].name)
        self.loaded_sample = self.setup_sample = files[0].name
        self.settings = Settings()
        self.canvas.history.clear()
        self.canvas.set_regions([], remember=False)
        self.canvas.set_image(image)
        self.dirty = False
        self.mask_visible = False
        self.visible_polygon = None
        self.alignment_status = "ok"
        saved = self.folder / "tube_setup.json"
        if saved.is_file():
            self.load_setup(saved)
        else:
            self.status.set(
                "Folder loaded. Find the green tapes, "
                "then draw tube regions or divide a rack."
            )

    def change_sample(self):
        """Display the chosen frame in setup coordinates, keeping saved regions."""
        if self.busy():
            return
        try:
            image = read_image(self.folder / self.sample.get())
            if self.canvas.regions and image.shape[:2] != self.image.shape[:2]:
                raise ValueError(
                    "This image has a different size. "
                    "Clear regions or use a separate setup for it."
                )
            alignment_status = "ok"
            visible_polygon = None
            if (
                self.canvas.regions
                and self.settings.align_each_image
                and self.sample.get() != self.setup_sample
            ):
                setup = self.get_setup()
                session = AnalysisSession.from_files(setup, self.files)
                frame = session.prepare(image)
                image, alignment_status, visible_polygon = (
                    frame.image,
                    frame.alignment_status,
                    frame.visible_polygon,
                )
        except (ValueError, cv2.error) as exc:
            self.sample.set(getattr(self, "loaded_sample", self.files[0].name))
            return self.error(exc)
        self.image = image
        self.visible_polygon = visible_polygon
        self.alignment_status = alignment_status
        self.loaded_sample = self.sample.get()
        if not self.canvas.regions:
            self.setup_sample = self.sample.get()
        self.mask_visible = False
        self.canvas.preview = []
        self.canvas.set_image(image)
        self.result_list.delete(*self.result_list.get_children())
        self.status.set(
            f"Image changed; alignment: {alignment_status}. "
            "Preview to check that the regions still align."
        )

    def change_mode(self):
        """Apply the selected drawing or editing tool to the canvas."""
        self.canvas.mode = self.mode.get()

    def add_box(self, box):
        """Commit a drawn tape, tube or rack while preserving existing tube metadata."""
        if self.busy():
            return
        mode = self.mode.get()
        regions = self.canvas.regions.copy()
        if mode == "rack":
            try:
                count = self.count.get()
                if count not in TUBE_IDS:
                    raise ValueError(
                        "This rack has tube IDs 1–12. Choose a count from 1 to 12."
                    )
                boxes = split_rack(box, count)
            except (ValueError, tk.TclError) as exc:
                return self.error(exc)
            existing = {r.tube_number: r for r in regions if r.kind == "tube"}
            regions = [r for r in regions if r.kind != "tube"] + [
                replace(existing[i], box=b)
                if i in existing
                else Region("tube", b, i, str(i), use_default_name=True)
                for i, b in enumerate(boxes, 1)
            ]
            self.status.set(
                "Rack divided. Review tube centres, "
                "resize any misaligned regions, and preview."
            )
        elif mode in ("64.5", "30"):
            regions = [r for r in regions if r.kind != mode] + [Region(mode, box)]
        else:
            number = self.draw_tube_number.get()
            existing = next(
                (r for r in regions if r.kind == "tube" and r.tube_number == number),
                None,
            )
            region = (
                replace(existing, box=box)
                if existing
                else Region("tube", box, number, str(number), use_default_name=True)
            )
            regions = [
                r for r in regions if not (r.kind == "tube" and r.tube_number == number)
            ] + [region]
            used = {r.tube_number for r in regions if r.kind == "tube"}
            self.draw_tube_number.set(
                next((i for i in TUBE_IDS if i not in used), number)
            )
        self.canvas.set_regions(regions)

    def regions_changed(self):
        """Mark setup edits as unsaved and clear stale preview results and masks."""
        self.dirty = True
        self.region_list.delete(0, "end")
        count = 0
        for region in self.canvas.regions:
            if region.kind == "tube":
                count += 1
                number = region.tube_number or count
                label = (
                    f"{'[x]' if region.enabled else '[ ]'} "
                    f"{region.tube_name or str(number)} · ID {number}"
                )
            else:
                label = f"Green tape · {region.kind} cm"
            self.region_list.insert("end", label)
        self.result_list.delete(*self.result_list.get_children())
        self.calibration_text.set("Calibration: preview to verify current regions")
        if self.mask_visible:
            self.mask_visible = False
            self.canvas.set_image(self.image, fit=False)

    def select_region(self, event=None):
        """Select the listed region for canvas movement or corner resizing."""
        selection = self.region_list.curselection()
        if selection and not self.busy():
            self.mode.set("edit")
            self.canvas.select(selection[0])
            region = self.canvas.regions[selection[0]]
            if region.kind == "tube":
                self.draw_tube_number.set(region.tube_number or 1)

    def undo(self):
        """Restore the preceding complete region state when analysis is idle."""
        if not self.busy():
            self.canvas.undo()

    def delete_region(self):
        """Remove the selected canvas region as an undoable edit."""
        if not self.busy():
            self.canvas.delete_selected()

    def clear(self):
        """Clear all regions while allowing the user to restore them with Undo."""
        if not self.busy():
            self.canvas.set_regions([])
            self.status.set("Regions cleared. Undo restores the previous setup.")

    def order_tubes(self):
        """Sort tube regions left to right while preserving physical IDs and labels."""
        if not self.busy():
            refs = [r for r in self.canvas.regions if r.kind != "tube"]
            tubes = sorted(
                (r for r in self.canvas.regions if r.kind == "tube"),
                key=lambda r: (r.box.x1 + r.box.x2) / 2,
            )
            self.canvas.set_regions(refs + tubes)

    def get_setup(self) -> Setup:
        """Build and validate a setup snapshot from the current canvas state."""
        if self.image is None:
            raise ValueError("Choose an image folder first.")
        regions = [r for r in self.canvas.regions if r.kind == "tube"]
        info = [region.tube_info(number) for number, region in enumerate(regions, 1)]
        setup = Setup(
            image_size=(self.image.shape[1], self.image.shape[0]),
            references=[
                Reference(float(r.kind), r.box)
                for r in self.canvas.regions
                if r.kind != "tube"
            ],
            tubes=[r.box for r in regions],
            settings=replace(self.settings),
            sample_name=self.setup_sample,
            tube_info=info,
        )
        setup.validate()
        return setup

    def find_tapes(self):
        """Suggest both green references while keeping existing tube regions intact."""
        if self.image is None or self.busy():
            return
        try:
            refs = suggest_references(self.image, self.settings)
        except ValueError as exc:
            return self.error(exc)
        self.canvas.set_regions(
            [Region(f"{r.length_cm:g}", r.box) for r in refs]
            + [r for r in self.canvas.regions if r.kind == "tube"]
        )
        self.status.set(
            "Green tapes suggested. Check the 64.5 cm and 30 cm boxes before analyzing."
        )

    def save_setup(self):
        """Save the validated setup beside the input images and clear the dirty flag."""
        if self.busy():
            return
        try:
            path = self.folder / "tube_setup.json" if self.folder else None
            self.get_setup().save(path)
        except (ValueError, OSError) as exc:
            return self.error(exc)
        self.dirty = False
        self.status.set(f"Setup saved: {path}")

    def load_setup(self, path=None):
        """Restore a compatible setup and its original coordinate-reference image."""
        if self.image is None or self.busy():
            return
        if path is None:
            path = filedialog.askopenfilename(
                parent=self.root,
                initialdir=self.folder,
                filetypes=[("Tube setup", "*.json")],
            )
        if not path:
            return
        try:
            setup = Setup.load(path)
            if setup.image_size != (self.image.shape[1], self.image.shape[0]):
                raise ValueError(
                    "Setup image size differs from this folder. Create a new setup."
                )
            sample_path = next(
                (p for p in self.files if p.name == setup.sample_name), None
            )
            if sample_path:
                image = read_image(sample_path)
                if (image.shape[1], image.shape[0]) != setup.image_size:
                    raise ValueError("The setup sample image has different dimensions.")
                self.image = image
                self.sample.set(sample_path.name)
                self.loaded_sample = sample_path.name
                self.setup_sample = sample_path.name
            else:
                raise ValueError(
                    f"The setup sample {setup.sample_name!r} "
                    "is missing from this folder."
                )
            self.settings = setup.settings
            self.canvas.set_regions(
                [Region(f"{r.length_cm:g}", r.box) for r in setup.references]
                + [
                    Region(
                        "tube",
                        b,
                        info.number,
                        info.name,
                        info.enabled,
                        info.use_default_name,
                    )
                    for b, info in zip(setup.tubes, setup.tube_info)
                ]
            )
            self.mask_visible = False
            self.visible_polygon = None
            self.alignment_status = "ok"
            self.canvas.set_image(self.image)
        except (ValueError, OSError) as exc:
            return self.error(exc)
        self.dirty = False
        self.status.set(f"Setup loaded. Preview before running: {path}")

    def preview(self) -> bool:
        """Measure the displayed aligned frame and update overlays and one-decimal
        results.
        """
        if self.busy():
            return False
        try:
            setup = self.get_setup()
            session = AnalysisSession.from_files(setup, self.files)
            frame = PreparedFrame(
                self.image, self.alignment_status, self.visible_polygon
            )
            calibration, measurements = session.measure(frame)
        except (ValueError, cv2.error) as exc:
            self.error(exc)
            return False
        self.canvas.show_measurements(measurements)
        self.result_list.delete(*self.result_list.get_children())
        for m in measurements:
            self.result_list.insert(
                "",
                "end",
                values=(m.tube_name, format_height(m.height_cm, "—"), m.status),
            )
        scales = ", ".join(
            f"{r.length_cm:g} cm: {s:.3f}"
            for r, s in zip(setup.references, calibration.scales)
        )
        self.calibration_text.set(
            f"{calibration.pixels_per_cm:.3f} pixels/cm\n{scales}\n{calibration.status}"
        )
        flagged = sum(m.status != "ok" for m in measurements)
        self.status.set(
            f"Preview: {len(measurements)} tubes, {flagged} flagged; "
            f"alignment: {self.alignment_status}. "
            "Yellow lines show detected top and bottom. "
            "Review flags and the color mask."
        )
        return True

    def toggle_mask(self):
        """Toggle a diagnostic color overlay while preserving source pixels."""
        if self.image is None or self.busy():
            return
        self.mask_visible = not self.mask_visible
        if self.mask_visible:
            image = (self.image * 0.2).astype(np.uint8)
            for region in self.canvas.regions:
                if region.kind == "tube" and not region.enabled:
                    continue
                mask = color_mask(
                    region.box.crop(self.image),
                    "red" if region.kind == "tube" else "green",
                    self.settings,
                )
                crop = region.box.crop(image)
                crop[mask > 0] = (
                    (80, 230, 255) if region.kind == "tube" else (100, 240, 100)
                )
        else:
            image = self.image
        self.canvas.set_image(image, fit=False)

    def edit_settings(self):
        """Open the settings draft editor and refresh the preview state on apply."""
        if self.busy():
            return

        def apply(settings):
            """Commit validated settings and realign the displayed frame if needed."""
            self.settings = settings
            self.regions_changed()
            if self.canvas.regions and self.sample.get() != self.setup_sample:
                self.change_sample()
            self.status.set(
                "Detection settings updated. Preview and inspect the color mask."
            )

        return show_settings_editor(self.root, self.settings, apply)

    def start_batch(self):
        """Snapshot the setup and launch folder processing on a background thread."""
        if self.busy() or not self.preview():
            return
        setup = self.get_setup()
        try:
            setup.save(self.folder / "tube_setup.json")
        except OSError as exc:
            return self.error(exc)
        self.dirty = False
        output = (
            PROJECT
            / "results"
            / f"{self.folder.name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        )
        self.last_output = output
        self.stop.clear()
        self._set_running(True)
        self.progress.configure(maximum=len(self.files), value=0)
        files = self.files.copy()

        def run():
            """Send worker progress, completion or failure through the event queue."""
            try:
                result = run_batch(
                    files,
                    setup,
                    output,
                    progress=lambda done, total, name: self.events.put(
                        ("progress", (done, total, name))
                    ),
                    cancelled=self.stop.is_set,
                )
                self.events.put(("done", result))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()
        self.status.set(f"Analyzing {len(files)} images. Output: {output}")

    def _set_running(self, running: bool) -> None:
        """Synchronize the busy flag and editable controls around a worker run."""
        self.running = running
        self.canvas.locked = running
        self.sample_combo.configure(state="disabled" if running else "readonly")
        self.run_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")

    def _poll(self):
        """Consume worker events on the Tk thread and open plots after completion."""
        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "progress":
                    done, total, name = value
                    self.progress.configure(value=done)
                    self.status.set(f"Analyzing {done}/{total}: {name}")
                else:
                    self._set_running(False)
                    if event == "error":
                        self.error(value)
                    else:
                        state = (
                            "Stopped; partial results saved"
                            if value["cancelled"]
                            else "Analysis finished"
                        )
                        self.status.set(
                            f"{state}: {value['processed_images']}/"
                            f"{value['total_images']} images, "
                            f"{value['failed_images']} failed, "
                            f"{value['flagged_measurements']} flagged measurements. "
                            f"{self.last_output}"
                        )
                        if value.get("plot_error"):
                            self.status.set(
                                self.status.get()
                                + f" Plot export failed: {value['plot_error']}"
                            )
                        if value["processed_images"] and not self.closing:
                            self.show_plots()
                    if self.closing:
                        self.root.destroy()
                        return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def edit_tubes(self):
        """Open the tube editor and commit valid drafts as one undoable change."""
        if self.busy():
            return
        if not any(region.kind == "tube" for region in self.canvas.regions):
            self.status.set(
                "Draw tube regions first. "
                "You can choose IDs 1–12 before drawing a tube."
            )
            return

        def apply(regions):
            """Update labels and inclusion without replacing measurement regions."""
            self.canvas.set_regions(regions)
            self.status.set(
                "Tube identities, names and inclusion updated. Save setup to keep them."
            )

        return show_tube_editor(self.root, self.canvas.regions.copy(), apply)

    def show_plots(self, path=None):
        """Open the latest or chosen detailed CSV in the native plot viewer."""
        if self.running:
            return
        path = (
            Path(path)
            if path
            else (self.last_output / "measurements.csv" if self.last_output else None)
        )
        if path is None or not path.is_file():
            path = filedialog.askopenfilename(
                parent=self.root,
                initialdir=PROJECT / "results",
                title="Open detailed measurements for plotting",
                filetypes=[("Detailed measurements CSV", "*.csv")],
            )
        if not path:
            return
        try:
            from tube_plots import show_plot_window

            return show_plot_window(self.root, path)
        except (ValueError, OSError, ImportError) as exc:
            self.error(exc)

    def show_results(self):
        """Open the latest result folder in the operating system file manager."""
        if self.last_output and self.last_output.exists():
            try:
                if sys.platform == "darwin":
                    subprocess.Popen(["open", str(self.last_output)])
                elif sys.platform == "win32":
                    import os

                    os.startfile(self.last_output)
                else:
                    subprocess.Popen(["xdg-open", str(self.last_output)])
            except OSError as exc:
                self.error(exc)
        else:
            self.status.set(
                "Run an analysis first. Results include heights.csv, "
                "detailed measurements and annotated previews."
            )

    def error(self, exc):
        """Show an actionable error in both the status bar and a parent-owned dialog."""
        self.status.set(str(exc))
        messagebox.showerror("Tube height analysis", str(exc), parent=self.root)

    def _close(self):
        """Stop safely after the current image or offer to save unsaved setup edits."""
        if self.running:
            self.closing = True
            self.stop.set()
            self.status.set(
                "Stopping after the current image and saving partial results…"
            )
        elif self.dirty and self.canvas.regions:
            answer = messagebox.askyesnocancel(
                "Save setup?", "Save your setup before closing?", parent=self.root
            )
            if answer is None:
                return
            if answer:
                self.save_setup()
                if self.dirty:
                    return
            self.root.destroy()
        else:
            self.root.destroy()


def main():
    """Parse launch options, create the desktop app and enter the Tk event loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Image folder to open")
    parser.add_argument("--setup", type=Path, help="Optional saved setup to load")
    parser.add_argument(
        "--plots", type=Path, help="Open plots from an existing measurements.csv"
    )
    args = parser.parse_args()
    default = PROJECT / "images" / "OCT7"
    root = tk.Tk()
    app = TubeApp(root, args.input or (default if default.is_dir() else None))
    if args.setup:
        root.after(350, lambda: app.load_setup(args.setup))
    if args.plots:
        app.last_output = args.plots.resolve().parent
        root.after(500, lambda: app.show_plots(args.plots))
    root.mainloop()


if __name__ == "__main__":
    main()
