# Tube height analysis

Measure the **vertical height of the red/orange column** inside each tube, using green tape references of **64.5 cm** and **30 cm**. The application runs locally; images are not uploaded.

## Start on macOS

The existing project virtual environment already has the required packages:

```sh
.venv/bin/python tube_app.py
```

You can also run `tube_app.py` in PyCharm with the project's `.venv` interpreter. OCT7 opens automatically if that folder exists. Use `analyze_tubes.py` for command-line batch processing.

For a new environment (Python 3.10 or newer):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tube_app.py --input images/OCT7
```

Tkinter comes with the python.org macOS installer. If your interpreter has no Tkinter, use a Python installation with Tk support. The core CLI does not need Tkinter or a display.

## Set up once, reuse for the folder

1. **Choose image folder.** Only image files in that folder are analyzed; subfolders, hidden files and non-image files are skipped. Filenames use natural ordering (`frame2` before `frame10`).
2. On **Set up regions**, click **Find green tapes automatically**. Check that the long tape is labelled **64.5 cm** and the short tape **30 cm**. Alternatively draw each tape region manually. Include a small margin around the whole tape and exclude other green objects.
3. Draw one region around each tube. **Tube ID to draw** selects the physical tube number (1–12); it advances to the next unused ID after each new tube. To replace a tube's region, select its ID and draw again. Include the full possible column height while keeping red valves and lower reservoirs outside the regions. To speed this up, set **Tubes in rack**, choose **Draw rack and divide into tubes**, then draw across the rack. Start half a tube spacing before the first tube centre and end half a spacing after the last. This produces equally spaced regions labelled 1, 2, …, including empty tubes. Adjust them individually when spacing is uneven; it does not detect tube centres automatically.
4. Choose **Move / resize existing regions** or select a region in the list. Drag its interior to move it or a corner handle to resize it. **Delete**, **Undo**, and **Clear** operate on the actual saved coordinates. **Tube names / include in analysis…** lets you correct physical IDs, choose default or custom labels, and uncheck unused tubes. **Default** uses the physical ID (`1`–`12`); uncheck it to type a name or label. **Use default labels** resets all names to their IDs. Excluding a tube preserves its region for later use and omits it from measurements, CSVs and plots. **Sort regions left → right** changes their order while preserving IDs/names. For example, selecting only physical tubes 2, 5 and 12 retains labels 2, 5 and 12.
5. Open **Preview / analyze** and click **Preview heights**. Yellow lines mark the measured top and bottom. Inspect the result table and **Show / hide color mask**. Adjust color thresholds in **Detection settings** if needed.
6. Check a later image using the image dropdown. Once regions exist, the original setup image remains the coordinate reference; other images are aligned to it using the green tapes. Inspect the alignment before running the folder.
7. Click **Analyze all images**. The setup, tube names and inclusion choices are saved to `INPUT/tube_setup.json` and reloaded when you next open the folder. Every run gets a fresh timestamped directory under the project's `results/` folder. Combined and individual height plots open when analysis finishes. The interface remains responsive; **Stop after current image** saves marked partial results.

**Controls:** scroll to zoom at the pointer; Space + drag or right drag to pan; **Fit image** to reset the view. Drag rectangles in either direction. Escape cancels the active canvas gesture. Command-Z / Control-Z undoes a region change while the canvas is focused; Delete / Backspace removes the selected region.

**Save setup** requires both references and at least one tube. **Load setup…** also accepts another run's `setup.json`, provided its sample image and image dimensions match this folder. To choose a different coordinate reference, clear the regions, choose the new image, then create the setup again.

## Outputs

| File | Contents |
| --- | --- |
| `heights.csv` | One row per image; filename and included labels such as `2 (cm)`, `5 (cm)`, `Outlet (cm)`. Excluded tubes are omitted. |
| `measurements.csv` | One row per included image/tube; physical `tube` ID, `tube_name`, estimated cm, pixel span, top/bottom coordinates, pixels/cm, status and error detail. Coordinates refer to the aligned setup image. The bottom edge is exclusive. |
| `setup.json` | Exact regions, reference lengths, sample filename and detection settings used for this run. |
| `run_summary.json` | Processed/failed image counts, flagged measurement count, completion/cancellation state. `complete` describes processing completion, not scientific validity. |
| `preview_*.jpg` | Annotated first/middle/last images when those frames can be measured. |
| `height_trends.png` / `.pdf` | Combined height plot for included tubes with stable names/colors. |
| `tube_panels.png` / `.pdf` | Separate panels for each included tube, using a common height scale. |

Heights are reported with **one decimal place** in the preview, annotated images, both CSV files and plot height labels (for example, `6.4 cm` or `6.5 cm`). Calculations retain full precision until reporting.

Missing detections, regions outside the frame, and failed image processing have **blank heights**, not zero. A blank cannot distinguish an empty tube from an unreadable or hidden column. **Estimates with warnings remain numeric** in both CSVs; use `measurements.csv` to filter or review them before using the wide table for analysis. No result files are appended to previous runs or silently overwritten.

| Status | Meaning / action |
| --- | --- |
| `ok` | Detection passed the implemented checks; still inspect representative previews. |
| `not_detected` | No red component passed the color/area filters. Check whether the tube is empty or the settings need adjustment. |
| `outside_frame` | Part of the tube region falls outside the source photo after alignment. Height is blank. |
| `clipped` | The detected color reaches the top or bottom of its region. Enlarge/reposition the region or check for fittings/reservoir contamination. |
| `multiple_regions` | Another colored component has at least 20% of the largest component's area. Reflections, interruptions or unrelated red objects may affect the estimate. |
| `small_region` | Detected red spans less than 5% of the tube region height. Inspect for staining or background objects, especially in empty tubes. |
| `calibration_disagreement` | The scales from the two tapes differ by more than 5% of their mean. Check reference regions and camera geometry. |
| `alignment_uncertain` | Tape endpoint alignment residual exceeds 0.75% of the long tape's span (minimum 3 pixels). Inspect that frame's tube locations. |
| `image_error` | Corrupt image, different dimensions, missing/clipped/fragmented tape, or failed alignment. The detail column records the cause. |

Several flags can appear together, separated by semicolons. The preview is a check of the currently displayed frame; the batch recalibrates and checks every frame by default.

## View and export plots

Click **View plots / open results CSV…** on the **Preview / analyze** tab. It opens the latest run, or lets you choose an existing `measurements.csv`. The viewer has **Combined** and **Individual tubes** tabs, per-tube visibility checkboxes, pan/zoom controls and **Export displayed plot…** for PNG/PDF. Plot visibility affects the display only; use **Tube names / include in analysis…** to change the next batch's selection.

The X axis is image sequence in natural filename order, not elapsed time. The Y axis is height in cm. Lines/dots show measurements with status `ok`; crosses show flagged estimates and can be hidden. Missing/flagged samples break the accepted line; values are not interpolated or filled with zero. Automatically saved plots include warning crosses.

Open a saved result directly:

```sh
.venv/bin/python tube_app.py --plots results/my_run/measurements.csv
```

Default labels are `1` through `12`. Each tube can instead have a custom name or label, which appears in CSV headers, the preview table and plot legends. The naming choice is saved with the setup. Default labels follow changes to the physical ID; custom labels stay as entered. Names must be nonempty and unique. Earlier setups with automatic `Tube N` names load with numeric defaults; other custom labels are preserved, and existing result CSVs keep their recorded labels. Version 1 setups remain readable, with sequential default names because they did not store physical IDs. When opening an old partial-rack setup, verify its physical IDs in the tube editor. The supplied six-region OCT7 setup was migrated to IDs **2, 3, 4, 5, 7, 8**, with the previous file preserved as `images/OCT7/tube_setup.v1.json`.

## Calibration and measurement assumptions

Each green tape is measured as the vertical pixel span of its largest connected green component. Scale is a least-squares fit through the origin:

```text
pixels_per_cm = (64.5 * long_tape_pixels + 30 * short_tape_pixels)
                / (64.5**2 + 30**2)
height_cm = red_vertical_span_pixels / pixels_per_cm
```

There are no empirical camera multipliers or forced nominal tube heights. The HSV mask covers both red hue ranges, including orange-red columns. One 3×3 opening removes isolated noise; the old repeated erosion and mask stretching are removed. Measurements use the largest connected colored component, so an interrupted column can be underestimated and needs inspection.

**Align camera movement using both green tapes** is enabled by default. It finds tall green candidates, matches a 64.5:30 length ratio, and fits a similarity transform from their endpoints to the setup image. This corrects translation, uniform scale and rotation up to 10 degrees; scale changes outside 0.5–2 or fit residuals above 2% of the long tape span (minimum 5 pixels) fail the frame. Suggestions may select the wrong green objects in a different scene; review previews.

Alignment does not correct perspective, lens distortion or depth differences between tapes and tubes. The images must have the same decoded dimensions, and tapes and tubes should lie in approximately the same plane. Heights are **vertical projections**, not lengths along tilted tubes. For a substantial camera viewpoint change, use a separate setup. Disabling alignment reuses fixed image coordinates. Disabling per-image calibration uses the saved setup sample's scale; use that only when appropriate for your capture geometry.

The OCT7 photographs show **12 tubes in the first photo** and a camera-position change after the early frames. Some later photos no longer contain the rightmost tube(s). Alignment can recover positions within the photograph but cannot recover missing image content. Small red/brown patches at the tube bases can resemble liquid in empty tubes, so inspect `small_region` flags.

## Command-line batch processing

After saving a setup in the desktop application:

```sh
.venv/bin/python analyze_tubes.py --input images/OCT7
```

Or specify the setup and a **new or empty** output directory:

```sh
.venv/bin/python analyze_tubes.py \
  --input images/OCT7 \
  --setup images/OCT7/tube_setup.json \
  --output results/my_run
```

Add `--no-previews` to skip annotated JPEGs or `--no-plots` to skip plot exports. The CLI exits with status 1 for invalid setup/output arguments and 2 if the batch completed with image failures. Measurement warnings are recorded in the exports. A plotting failure is recorded separately as `plot_error` in the run summary; the measurements remain saved.

## Checks

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Native pointer-event checks require desktop access (on macOS, run from Terminal or PyCharm):

```sh
RUN_GUI_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
```

The checks cover known synthetic heights, both red hue ranges, noise, clipping, fragmented columns, two-tape disagreement, missing tapes, saved setup validation, reversed selection, zoom transforms, camera translation/scale/rotation, out-of-frame regions, natural file ordering, corrupt files, output protection, cancellation, and native drag/move/resize/undo/sample switching. Synthetic checks establish behavior, not the physical accuracy of experimental photos; compare several real measurements with manual readings before scientific use.

## Project structure and maintenance

| Module | Responsibility |
| --- | --- |
| `tube_app.py` | Main window, folder/setup workflow, previews and background batch progress. |
| `tube_dialogs.py` | Validated settings and tube-label drafts; closing an editor discards its draft. |
| `tube_selector.py` | Native Tk canvas, image/display coordinate mapping, selection and undo history. |
| `tube_analysis.py` | Shared data models, image processing, calibration, alignment, measurement and CSV export. |
| `analyze_tubes.py` | Display-independent command-line entry point. |
| `tube_plots.py` | Detailed CSV loading, shared figure builder, PNG/PDF export and native plot viewer. |
| `tests/` | Synthetic measurement, export, plotting and optional native GUI checks. |

Preview and batch processing use the same `AnalysisSession`. It prepares the sample's reference data, aligns other frames into setup coordinates, and measures included tubes with consistent warning flags. A batch reuses one session; the GUI creates a fresh session after edits. `PreparedFrame` carries the aligned image and its visible footprint. Keep calculation changes in this shared path so preview and exported measurements agree.

`TubeInfo.validate()` and `validate_tubes()` enforce the same identity and label rules in setup files and the editor. `format_height()` controls one-decimal reporting across the table, canvas, annotated images and CSV exports. All functions and classes have docstrings describing their purpose; the more involved measurement functions also document coordinate, missing-data and calibration behavior.

The obsolete compatibility entry points and unused standalone selection dialogs have been removed. Run `tube_app.py` for the desktop workflow and `analyze_tubes.py` for batch processing.

Python formatting follows Black with the settings in `pyproject.toml`. With Black available, format and check the source using:

```sh
black *.py tests
black --check *.py tests
```
