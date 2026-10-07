"""Native Tk interaction checks. Opt in on a desktop: RUN_GUI_TESTS=1.

Tk requires desktop access on macOS, so ordinary/headless test runs skip these.
"""

import os
import tkinter as tk
import unittest

import numpy as np

from tube_analysis import Box, Setup, TubeInfo, run_batch
from tube_selector import ImageCanvas, Region


@unittest.skipUnless(os.environ.get("RUN_GUI_TESTS") == "1", "requires an accessible desktop")
class NativeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.geometry("820x620")
        self.boxes = []
        self.canvas = ImageCanvas(self.root, on_box=self.boxes.append)
        self.canvas.pack(fill="both", expand=True)
        self.root.update()
        self.canvas.set_image(np.zeros((1200, 1600, 3), np.uint8))
        self.root.update()
        self.canvas.fit()

    def tearDown(self):
        self.root.destroy()

    def drag(self, start, end):
        x1, y1 = self.canvas.transform.to_canvas(*start)
        x2, y2 = self.canvas.transform.to_canvas(*end)
        self.canvas.event_generate("<ButtonPress-1>", x=round(x1), y=round(y1))
        self.canvas.event_generate("<B1-Motion>", x=round(x2), y=round(y2))
        self.root.update()
        if self.canvas.mode != "edit":
            draft = self.canvas.find_withtag("draft")
            self.assertTrue(draft, "live drag preview must exist before release")
            expected = (round(min(x1, x2)), round(min(y1, y2)), round(max(x1, x2)), round(max(y1, y2)))
            np.testing.assert_allclose(self.canvas.coords(draft[0]), expected, atol=1)
        self.canvas.event_generate("<ButtonRelease-1>", x=round(x2), y=round(y2))
        self.root.update()

    def test_live_drag_reverse_direction_zoom_and_resize(self):
        for zoom in (1.0, 1.6):
            self.canvas.zoom(zoom)
            self.drag((1100, 900), (400, 200))
            box = self.boxes[-1]
            np.testing.assert_allclose((box.x1, box.y1, box.x2, box.y2), (400, 200, 1100, 900), atol=3)
        self.root.geometry("1050x720")
        self.root.update()
        self.canvas.fit()
        self.drag((400, 200), (1100, 900))
        box = self.boxes[-1]
        np.testing.assert_allclose((box.x1, box.y1, box.x2, box.y2), (400, 200, 1100, 900), atol=3)

    def test_move_resize_delete_undo_keep_region_coordinates_synchronized(self):
        original = Region("tube", Box(300, 200, 600, 800))
        self.canvas.set_regions([original])
        self.canvas.select(0)
        self.drag((450, 500), (550, 550))
        moved = self.canvas.regions[0].box
        np.testing.assert_allclose((moved.x1, moved.y1), (400, 250), atol=3)
        self.canvas.undo()
        self.assertEqual(self.canvas.regions, [original])
        self.canvas.select(0)
        self.drag((600, 800), (800, 900))
        resized = self.canvas.regions[0].box
        np.testing.assert_allclose((resized.x2, resized.y2), (800, 900), atol=3)
        self.canvas.delete_selected()
        self.assertEqual(self.canvas.regions, [])
        self.canvas.undo()
        self.assertEqual(self.canvas.regions[0].box, resized)

    def test_app_preview_sample_switch_and_saved_setup(self):
        import tempfile
        from pathlib import Path
        import cv2
        from tube_app import TubeApp
        from test_analysis import scene

        self.canvas.destroy()
        app = TubeApp(self.root)
        image, setup = scene()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cv2.imwrite(str(folder / setup.sample_name), image)
            moved = cv2.warpAffine(image, np.float64([[1, 0, 30], [0, 1, 8]]), (600, 420))
            cv2.imwrite(str(folder / "frame2.png"), moved)
            setup.save(folder / "tube_setup.json")
            app.load_folder(folder)
            self.root.update()
            self.assertTrue(app.preview())
            app.sample.set("frame2.png")
            app.change_sample()
            self.root.update()
            self.assertTrue(app.preview())
            self.assertEqual(app.get_setup().sample_name, "frame1.png")
            self.assertEqual(len(app.result_list.get_children()), 3)
            app.save_setup()
            self.assertEqual(Setup.load(folder / "tube_setup.json").sample_name, "frame1.png")

    def test_fixed_calibration_preview_matches_batch_behavior(self):
        import tempfile
        from pathlib import Path
        import cv2
        from tube_app import TubeApp
        from test_analysis import scene

        self.canvas.destroy()
        app = TubeApp(self.root)
        image, setup = scene()
        setup.settings.align_each_image = False
        setup.settings.recalibrate_each_image = False
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cv2.imwrite(str(folder / setup.sample_name), image)
            image[:, :80] = 0
            image[:, 250:285] = 0
            cv2.imwrite(str(folder / "frame2.png"), image)
            setup.save(folder / "tube_setup.json")
            app.load_folder(folder)
            app.sample.set("frame2.png")
            app.change_sample()
            self.assertTrue(app.preview())
            first = app.result_list.item(app.result_list.get_children()[0], "values")
            self.assertEqual(first[1], "40.0")

    def test_named_subset_preview_sort_undo_and_native_plot_window(self):
        import tempfile
        from pathlib import Path
        import cv2
        from tube_app import TubeApp
        from test_analysis import scene

        self.canvas.destroy()
        app = TubeApp(self.root)
        image, setup = scene()
        setup.tube_info = [TubeInfo(2), TubeInfo(5, enabled=False), TubeInfo(12)]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            sample = folder / setup.sample_name
            cv2.imwrite(str(sample), image)
            setup.save(folder / "tube_setup.json")
            app.load_folder(folder)
            self.root.update()
            self.assertTrue(app.preview())
            names = [app.result_list.item(row, "values")[0] for row in app.result_list.get_children()]
            self.assertEqual(names, ["2", "12"])
            app.order_tubes()
            self.assertEqual([info.number for info in app.get_setup().tube_info], [2, 5, 12])
            app.canvas.select(2)
            app.delete_region()
            app.undo()
            self.assertEqual([info.number for info in app.get_setup().tube_info], [2, 5, 12])
            run_batch([sample], setup, folder / "out", save_previews=False, save_plots=False)
            window = app.show_plots(folder / "out/measurements.csv")
            self.root.update()
            self.assertIn("Tube height plots", window.title())
            window.destroy()

    def test_default_custom_label_editor_persistence_csv_and_plot_labels(self):
        import csv
        import tempfile
        from pathlib import Path
        import cv2
        from tube_app import TubeApp
        from tube_plots import build_figure, load_measurements
        from test_analysis import scene

        self.canvas.destroy()
        app = TubeApp(self.root)
        image, setup = scene()
        setup.tube_info = [TubeInfo(2), TubeInfo(5), TubeInfo(12)]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            sample = folder / setup.sample_name
            cv2.imwrite(str(sample), image)
            setup.save(folder / "tube_setup.json")
            app.load_folder(folder)
            dialog = app.edit_tubes()
            self.root.update()
            body = dialog.winfo_children()[0]
            first_id = body.grid_slaves(row=2, column=1)[0]
            first_name = body.grid_slaves(row=2, column=3)[0]
            self.assertEqual(first_name.get(), "2")
            self.assertEqual(str(first_name.cget("state")), "readonly")
            first_id.set("4")
            self.assertEqual(first_name.get(), "4")
            body.grid_slaves(row=3, column=0)[0].invoke()  # Exclude physical tube 5.
            custom_option = body.grid_slaves(row=4, column=2)[0]
            custom_name = body.grid_slaves(row=4, column=3)[0]
            custom_option.invoke()
            self.assertEqual(str(custom_name.cget("state")), "normal")
            custom_name.delete(0, "end")
            custom_name.insert(0, "Outlet, α")
            body.grid_slaves(row=4, column=1)[0].set("11")
            self.assertEqual(custom_name.get(), "Outlet, α")
            custom_option.invoke()
            self.assertEqual(custom_name.get(), "11")
            custom_option.invoke()
            self.assertEqual(custom_name.get(), "Outlet, α")
            bar = body.grid_slaves(row=5, column=0)[0]
            next(w for w in bar.winfo_children() if w.cget("text") == "Apply").invoke()
            app.save_setup()
            restored = Setup.load(folder / "tube_setup.json")
            self.assertEqual([(i.number, i.name, i.enabled, i.use_default_name) for i in restored.tube_info], [(4, "4", True, True), (5, "5", False, True), (11, "Outlet, α", True, False)])
            app.load_setup(folder / "tube_setup.json")
            self.assertEqual(app.get_setup(), restored)
            run_batch([sample], restored, folder / "out", save_previews=False, save_plots=False)
            with (folder / "out/heights.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.reader(handle)), ["image", "4 (cm)", "Outlet, α (cm)"])
            data = load_measurements(folder / "out/measurements.csv")
            figure = build_figure(data)
            self.assertEqual([t.get_text() for t in figure.axes[0].get_legend().get_texts()], ["4", "Outlet, α"])
            figure.clear()
            dialog = app.edit_tubes()
            body = dialog.winfo_children()[0]
            bar = body.grid_slaves(row=5, column=0)[0]
            next(w for w in bar.winfo_children() if w.cget("text") == "Use default labels").invoke()
            self.assertEqual(body.grid_slaves(row=4, column=3)[0].get(), "11")
            next(w for w in bar.winfo_children() if w.cget("text") == "Apply").invoke()
            self.assertTrue(all(i.use_default_name for i in app.get_setup().tube_info))
