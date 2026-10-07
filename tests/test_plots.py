"""Check scientific plot data, label preservation, filtering and file exports."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tube_plots import build_figure, export_plots, load_measurements


class PlotTests(unittest.TestCase):
    """Check plotted labels, quality flags, missing-data gaps and exported figures."""

    def setUp(self):
        """Create isolated fixtures before each regression check."""
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.path = self.folder / "measurements.csv"
        with self.path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "tube", "tube_name", "height_cm", "status"])
            writer.writerows(
                [
                    ["image1.png", 2, "Tube 2", 40, "ok"],
                    ["image1.png", 12, "Outlet", "", "not_detected"],
                    ["image2.png", 2, "Tube 2", 45, "clipped"],
                    ["image2.png", 12, "Outlet", 20, "ok"],
                    ["image3.png", 2, "Tube 2", 30, "ok"],
                    ["image3.png", 12, "Outlet", "", "image_error"],
                ]
            )

    def tearDown(self):
        """Release temporary files or native windows after each regression check."""
        self.temporary.cleanup()

    def test_names_ids_gaps_and_flags_are_preserved(self):
        """Names IDs gaps and flags are preserved."""
        data = load_measurements(self.path)
        self.assertEqual([s.number for s in data.series], [2, 12])
        self.assertEqual([s.name for s in data.series], ["Tube 2", "Outlet"])
        self.assertTrue(np.isnan(data.series[1].heights[0]))
        figure = build_figure(data)
        line = figure.axes[0].lines[0]
        np.testing.assert_allclose(line.get_ydata(), [40, np.nan, 30], equal_nan=True)
        np.testing.assert_allclose(
            figure.axes[0].collections[0].get_offsets(), [[2, 45]]
        )
        self.assertEqual(
            [text.get_text() for text in figure.axes[0].get_legend().get_texts()],
            ["Tube 2", "Outlet"],
        )

    def test_plot_filters_and_individual_panels(self):
        """Plot filters and individual panels."""
        data = load_measurements(self.path)
        figure = build_figure(data, [12], individual=True, include_flagged=False)
        self.assertEqual(len(figure.axes), 1)
        self.assertIn("Outlet", figure.axes[0].get_title())
        self.assertEqual(len(figure.axes[0].collections), 0)
        empty = build_figure(data, [])
        self.assertIn("Select at least one tube", empty.axes[0].texts[0].get_text())

    def test_exports_png_and_pdf_and_supports_legacy_csv(self):
        """Exports PNG and PDF and supports legacy CSV."""
        paths = export_plots(self.path, self.folder)
        self.assertEqual(len(paths), 4)
        for name in paths:
            data = (self.folder / name).read_bytes()
            self.assertGreater(len(data), 1000)
            self.assertTrue(
                data.startswith(b"%PDF")
                if name.endswith("pdf")
                else data.startswith(b"\x89PNG")
            )
        legacy = self.folder / "legacy.csv"
        legacy.write_text("image,tube,height_cm,status\na.jpg,7,12,ok\n")
        self.assertEqual(load_measurements(legacy).series[0].name, "7")

    def test_wide_csv_and_duplicate_rows_are_rejected(self):
        """Wide CSV and duplicate rows are rejected."""
        wrong = self.folder / "heights.csv"
        wrong.write_text("image,Tube 2 (cm)\na.jpg,40\n")
        with self.assertRaises(ValueError):
            load_measurements(wrong)
        with self.path.open("a") as handle:
            handle.write("image1.png,2,Tube 2,50,ok\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_measurements(self.path)
