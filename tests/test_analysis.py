"""Known-geometry checks for measurement, alignment, setup and batch exports."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tube_analysis import (
    AnalysisSession,
    Box,
    PreparedFrame,
    Reference,
    Settings,
    Setup,
    TubeInfo,
    calibrate,
    detect_span,
    list_images,
    measure_image,
    run_batch,
    split_rack,
    suggest_references,
)
from tube_selector import ViewTransform


def scene():
    """Known geometry: 4 pixels/cm, 64.5/30 cm tapes, 40/25 cm columns."""
    image = np.zeros((420, 600, 3), np.uint8)
    image[40:298, 40:52] = (0, 200, 0)
    image[40:160, 260:272] = (0, 200, 0)
    image[150:310, 120:140] = (0, 0, 220)
    orange = cv2.cvtColor(np.uint8([[[15, 210, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    image[210:310, 340:360] = orange
    setup = Setup(
        (600, 420),
        [
            Reference(64.5, Box(30, 30, 65, 310)),
            Reference(30.0, Box(250, 30, 285, 175)),
        ],
        [Box(100, 30, 160, 330), Box(320, 30, 380, 330), Box(420, 30, 480, 330)],
        sample_name="frame1.png",
    )
    return image, setup


class MeasurementTests(unittest.TestCase):
    """Check calibration, detection, identities and geometry against known synthetic
    data.
    """

    def test_subset_preserves_physical_ids_names_and_saved_inclusion(self):
        """Subset preserves physical IDs names and saved inclusion."""
        image, setup = scene()
        setup.tube_info = [TubeInfo(2), TubeInfo(5, "Control", False), TubeInfo(12)]
        _, measurements = measure_image(image, setup)
        self.assertEqual([m.tube for m in measurements], [2, 12])
        self.assertEqual([m.tube_name for m in measurements], ["2", "12"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "setup.json"
            setup.save(path)
            self.assertEqual(Setup.load(path), setup)
            self.assertEqual(json.loads(path.read_text())["version"], 2)

    def test_old_setup_loads_with_default_names(self):
        """Old setup loads with default names."""
        _, setup = scene()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "setup.json"
            setup.save(path)
            data = json.loads(path.read_text())
            data["version"] = 1
            del data["tube_info"]
            path.write_text(json.dumps(data))
            restored = Setup.load(path)
            self.assertEqual(
                [info.name for info in restored.tube_info], ["1", "2", "3"]
            )

    def test_default_labels_cover_all_twelve_ids(self):
        """Default labels cover all twelve IDs."""
        self.assertEqual(
            [TubeInfo(i).name for i in range(1, 13)], [str(i) for i in range(1, 13)]
        )
        self.assertTrue(all(TubeInfo(i).use_default_name for i in range(1, 13)))

    def test_legacy_names_migrate_and_explicit_custom_names_round_trip(self):
        """Legacy names migrate and explicit custom names round trip."""
        _, setup = scene()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "setup.json"
            setup.save(path)
            data = json.loads(path.read_text())
            data["tube_info"] = [
                {"number": 2, "name": "Tube 2", "enabled": True},
                {"number": 5, "name": "Control, α", "enabled": False},
                {
                    "number": 12,
                    "name": "Tube 12",
                    "enabled": True,
                    "use_default_name": False,
                },
            ]
            path.write_text(json.dumps(data))
            restored = Setup.load(path)
            self.assertEqual(
                [info.name for info in restored.tube_info],
                ["2", "Control, α", "Tube 12"],
            )
            self.assertEqual(
                [info.use_default_name for info in restored.tube_info],
                [True, False, False],
            )
            restored.save(path)
            self.assertEqual(Setup.load(path), restored)

    def test_duplicate_ids_names_and_empty_selection_are_rejected(self):
        """Duplicate IDs names and empty selection are rejected."""
        _, setup = scene()
        for info in (
            [TubeInfo(2), TubeInfo(2), TubeInfo(12)],
            [TubeInfo(2, "Sample"), TubeInfo(5, "sample"), TubeInfo(12)],
            [
                TubeInfo(2, enabled=False),
                TubeInfo(5, enabled=False),
                TubeInfo(12, enabled=False),
            ],
        ):
            setup.tube_info = info
            with self.assertRaises(ValueError):
                setup.validate()

    def test_two_reference_calibration_and_red_orange_columns(self):
        """Two reference calibration and red orange columns."""
        image, setup = scene()
        calibration, measurements = measure_image(image, setup)
        self.assertEqual(calibration.pixels_per_cm, 4)
        self.assertEqual(calibration.scales, (4, 4))
        self.assertEqual([m.height_cm for m in measurements], [40, 25, None])
        self.assertEqual([m.status for m in measurements], ["ok", "ok", "not_detected"])

    def test_second_red_hue_range(self):
        """Second red hue range."""
        image, setup = scene()
        image[150:310, 120:140] = cv2.cvtColor(
            np.uint8([[[175, 210, 220]]]), cv2.COLOR_HSV2BGR
        )[0, 0]
        self.assertEqual(measure_image(image, setup)[1][0].height_cm, 40)

    def test_clipped_and_fragmented_columns_are_flagged(self):
        """Clipped and fragmented columns are flagged."""
        image, setup = scene()
        clipped = detect_span(image, Box(100, 150, 160, 310), "red", setup.settings)
        self.assertEqual(clipped.status, "clipped")
        image[210:220, 120:140] = 0
        fragmented = detect_span(image, setup.tubes[0], "red", setup.settings)
        self.assertIn("multiple_regions", fragmented.status)

    def test_missing_or_clipped_tape_fails_calibration(self):
        """Missing or clipped tape fails calibration."""
        image, setup = scene()
        image[40:160, 260:272] = 0
        with self.assertRaisesRegex(ValueError, "30 cm tape: not_detected"):
            calibrate(image, setup)
        image, setup = scene()
        setup.references[0] = Reference(64.5, Box(30, 40, 65, 298))
        with self.assertRaisesRegex(ValueError, "clipped"):
            calibrate(image, setup)

    def test_tape_disagreement_is_visible(self):
        """Tape disagreement is visible."""
        image, setup = scene()
        image[160:174, 260:272] = (0, 200, 0)
        setup.references[1] = Reference(30, Box(250, 30, 285, 190))
        self.assertEqual(calibrate(image, setup).status, "calibration_disagreement")

    def test_dimension_mismatch_is_rejected(self):
        """Dimension mismatch is rejected."""
        image, setup = scene()
        with self.assertRaisesRegex(ValueError, "image_size_mismatch"):
            measure_image(image[:-1], setup)

    def test_reversed_drag_and_image_boundary_clamping(self):
        """Reversed drag and image boundary clamping."""
        box = Box.from_points((610, 430), (-2, 10), (600, 420))
        self.assertEqual(box, Box(0, 10, 600, 420))
        with self.assertRaises(ValueError):
            Box.from_points((20, 20), (20, 20), (600, 420))
        with self.assertRaises(ValueError):
            Box(1.5, 1, 20, 20).validate((600, 420))

    def test_split_includes_empty_tubes_and_has_no_fixed_count(self):
        """Split includes empty tubes and has no fixed count."""
        for count in (1, 11, 12, 15):
            boxes = split_rack(Box(0, 20, 600, 400), count)
            self.assertEqual(len(boxes), count)
            for box in boxes:
                box.validate((600, 420))
        with self.assertRaises(ValueError):
            split_rack(Box(0, 0, 10, 100), 12)

    def test_setup_round_trip_and_malformed_input(self):
        """Setup round trip and malformed input."""
        _, setup = scene()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "setup.json"
            setup.save(path)
            self.assertEqual(Setup.load(path), setup)
            path.write_text('{"version": 999}')
            with self.assertRaises(ValueError):
                Setup.load(path)
        setup.settings.red_hue_max = 180
        with self.assertRaises(ValueError):
            setup.validate()

    def test_noise_does_not_outrank_column(self):
        """Noise does not outrank column."""
        image, setup = scene()
        image[40, 110] = (0, 0, 255)
        image[325, 145] = (0, 0, 255)
        self.assertEqual(measure_image(image, setup)[1][0].height_cm, 40)

    def test_green_suggestions_and_camera_translation_scale_rotation(self):
        """Green suggestions and camera translation scale rotation."""
        image, setup = scene()
        refs = suggest_references(image, setup.settings)
        self.assertEqual([r.length_cm for r in refs], [64.5, 30])
        session = AnalysisSession(setup, image)
        matrix = cv2.getRotationMatrix2D((300, 210), 1.2, 0.95)
        matrix[:, 2] += (30, 8)
        moved = cv2.warpAffine(image, matrix, (600, 420))
        frame = session.prepare(moved)
        calibration, measurements = session.measure(frame)
        self.assertEqual(frame.alignment_status, "ok")
        self.assertAlmostEqual(calibration.pixels_per_cm, 4, delta=0.04)
        self.assertAlmostEqual(measurements[0].height_cm, 40, delta=0.6)
        self.assertAlmostEqual(measurements[1].height_cm, 25, delta=0.6)

    def test_session_preserves_combined_flags_and_missing_outside_heights(self):
        """The shared preview/batch path must retain every applicable quality flag."""
        image, setup = scene()
        setup.tubes[0] = Box(100, 150, 160, 310)
        image[160:174, 260:272] = (0, 200, 0)
        setup.references[1] = Reference(30, Box(250, 30, 285, 190))
        polygon = np.array([[0, 0], [400, 0], [400, 419], [0, 419]], np.float32)
        session = AnalysisSession(setup, image)
        _, measurements = session.measure(
            PreparedFrame(image, "alignment_uncertain", polygon)
        )
        self.assertEqual(
            measurements[0].status,
            "clipped;calibration_disagreement;alignment_uncertain",
        )
        self.assertEqual(
            measurements[1].status, "calibration_disagreement;alignment_uncertain"
        )
        self.assertEqual(measurements[2].status, "outside_frame;alignment_uncertain")
        self.assertIsNone(measurements[2].height_cm)

    def test_out_of_frame_tube_is_blank(self):
        """Out of frame tube is blank."""
        image, setup = scene()
        polygon = np.array([[0, 0], [400, 0], [400, 419], [0, 419]], np.float32)
        _, measurements = measure_image(image, setup, visible_polygon=polygon)
        self.assertIsNone(measurements[2].height_cm)
        self.assertEqual(measurements[2].status, "outside_frame")

    def test_canvas_transform_round_trip_and_pointer_anchored_zoom(self):
        """Canvas transform round trip and pointer anchored zoom."""
        transform = ViewTransform(0.25, 83, 41)
        point = (1090, 2090)
        canvas_point = transform.to_canvas(*point)
        self.assertEqual(transform.to_image(*canvas_point), point)
        transform.zoom(1.5, *canvas_point)
        np.testing.assert_allclose(transform.to_image(*canvas_point), point)


class BatchTests(unittest.TestCase):
    """Check complete and partial batch exports, error reporting and output protection."""

    def test_named_subset_csv_and_error_rows_do_not_renumber(self):
        """Named subset CSV and error rows do not renumber."""
        image, setup = scene()
        image[310:313, 120:140] = (0, 0, 220)  # 40.75 cm, exported as 40.8.
        setup.tube_info = [
            TubeInfo(2),
            TubeInfo(5, "Reference column", False),
            TubeInfo(12, "Outlet"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            sample = folder / setup.sample_name
            cv2.imwrite(str(sample), image)
            broken = folder / "broken.png"
            broken.write_bytes(b"broken")
            output = folder / "out"
            result = run_batch(
                [sample, broken], setup, output, save_previews=False, save_plots=False
            )
            with (output / "heights.csv").open() as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["image", "2 (cm)", "Outlet (cm)"])
            self.assertEqual(rows[1], ["frame1.png", "40.8", ""])
            with (output / "measurements.csv").open() as handle:
                details = list(csv.DictReader(handle))
            self.assertEqual(
                [(r["tube"], r["tube_name"]) for r in details],
                [("2", "2"), ("12", "Outlet"), ("2", "2"), ("12", "Outlet")],
            )
            self.assertEqual([r["height_cm"] for r in details], ["40.8", "", "", ""])
            self.assertEqual(result["flagged_measurements"], 3)

    def test_streamed_csv_filenames_order_failure_status_and_output_protection(self):
        """Streamed CSV filenames order failure status and output protection."""
        image, setup = scene()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name in ("frame10.png", "frame2.png", "frame1.png"):
                cv2.imwrite(str(folder / name), image)
            (folder / "frame3.JPG").write_bytes(b"broken image")
            (folder / ".hidden.png").write_bytes(b"skip")
            (folder / "notes.txt").write_text("skip")
            files = list_images(folder)
            self.assertEqual(
                [p.name for p in files],
                ["frame1.png", "frame2.png", "frame3.JPG", "frame10.png"],
            )
            calls = []
            output = folder / "results"
            result = run_batch(
                files, setup, output, progress=lambda *args: calls.append(args)
            )
            self.assertTrue(result["complete"])
            self.assertEqual(result["failed_images"], 1)
            self.assertEqual(len(calls), 4)
            with (output / "heights.csv").open() as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[1], ["frame1.png", "40.0", "25.0", ""])
            self.assertEqual(rows[3], ["frame3.JPG", "", "", ""])
            with (output / "measurements.csv").open() as handle:
                details = list(csv.DictReader(handle))
            self.assertEqual(details[6]["status"], "image_error")
            self.assertIn("Cannot decode", details[6]["detail"])
            self.assertEqual(Setup.load(output / "setup.json"), setup)
            with self.assertRaisesRegex(ValueError, "not empty"):
                run_batch(files, setup, output)

    def test_cancel_marks_partial_run_and_preserves_completed_rows(self):
        """Cancel marks partial run and preserves completed rows."""
        image, setup = scene()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            files = [folder / f"frame{i}.png" for i in range(1, 4)]
            for path in files:
                cv2.imwrite(str(path), image)
            count = [0]
            result = run_batch(
                files,
                setup,
                folder / "out",
                progress=lambda *args: count.__setitem__(0, count[0] + 1),
                cancelled=lambda: count[0] == 1,
            )
            self.assertTrue(result["cancelled"])
            self.assertFalse(result["complete"])
            self.assertEqual(result["processed_images"], 1)
            self.assertEqual(
                json.loads((folder / "out/run_summary.json").read_text()), result
            )

    def test_fixed_calibration_uses_named_sample(self):
        """Fixed calibration uses named sample."""
        image, setup = scene()
        setup.settings.align_each_image = False
        setup.settings.recalibrate_each_image = False
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            reference = folder / setup.sample_name
            cv2.imwrite(str(reference), image)
            without_tapes = image.copy()
            without_tapes[:, :80] = 0
            without_tapes[:, 250:285] = 0
            other = folder / "frame2.png"
            cv2.imwrite(str(other), without_tapes)
            result = run_batch(
                [other, reference], setup, folder / "out", save_previews=False
            )
            self.assertEqual(result["failed_images"], 0)

    def test_missing_sample_is_rejected_before_export(self):
        """Missing sample is rejected before export."""
        _, setup = scene()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "out"
            with self.assertRaisesRegex(ValueError, "missing"):
                run_batch([Path(folder) / "other.png"], setup, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
