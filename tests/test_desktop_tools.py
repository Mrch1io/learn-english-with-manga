"""Independent utility boundaries; no GPU, model inference, or network calls.

Run: .venv\\Scripts\\python.exe -m unittest tests.test_desktop_tools -v
"""
import copy
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from ui import tools as desktop_tools
from ui import app as gui
from tests.test_gui import fake_editor_session


class RevisionInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def save(self, filename, record, stamp):
        path = self.folder / filename
        path.write_text(json.dumps(record), encoding="utf-8")
        os.utime(path, ns=(stamp, stamp))
        return path

    def page(self, name="page", **flags):
        return {"input_path": str(self.folder / f"{name}.jpg"),
                "output_path": str(self.folder / f"{name}.png"),
                "work": "Daily sample", "blocks": [], **flags}

    def test_scan_reads_each_json_once_and_never_changes_saved_bytes(self):
        self.save("manual.json", self.page(manual_edits=True), 1_000_000_000)
        self.save("ocr.json", self.page("ocr", ocr_candidate_edits=True), 1_000_000_000)
        self.save("ordinary.json", self.page("ordinary"), 1_000_000_000)
        self.save("run-one.json", {"pages": [dict(self.page(), status="needs_review", message="Terms changed; retained.")]}, 2_000_000_000)
        before = {p: p.read_bytes() for p in self.folder.glob("*.json")}
        calls, original = [], Path.read_text

        def read(path, *args, **kwargs):
            calls.append(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", autospec=True, side_effect=read):
            rows, errors = desktop_tools.collect_revisions(self.folder)
        self.assertFalse(errors)
        self.assertEqual(len(rows), 2)
        self.assertCountEqual(calls, before)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(next(r for r in rows if r["page"] == "page.jpg")["note"], "Terms changed; retained.")
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_newer_saved_edit_is_not_misrepresented_as_current_batch_warning(self):
        self.save("page.json", self.page(manual_edits=True), 3_000_000_000)
        self.save("run-old.json", {"pages": [dict(self.page(), status="needs_review", message="Older warning")]}, 1_000_000_000)
        self.save("run-new.json", {"pages": [dict(self.page(), status="needs_review", message="Newer warning")]}, 2_000_000_000)
        rows, _ = desktop_tools.collect_revisions(self.folder)
        self.assertEqual(rows[0]["note"], "Saved after the last batch note. Previous note: Newer warning")

    def test_cancelled_scan_reads_no_records_and_invalid_json_is_reported(self):
        self.save("page.json", self.page(manual_edits=True), 1_000_000_000)
        (self.folder / "broken.json").write_text("{broken", encoding="utf-8")
        rows, errors = desktop_tools.collect_revisions(self.folder)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(errors), 1)
        stop = threading.Event()
        stop.set()
        with patch.object(Path, "read_text", side_effect=AssertionError("Cancelled scans must not read files")):
            self.assertEqual(desktop_tools.collect_revisions(self.folder, stop), ([], []))

    def test_geometry_is_source_scaled_and_rejects_artwork_outside_bubble(self):
        self.assertEqual(desktop_tools.cleanup_rectangle((20, 25), (10, 10), (0.5, 0.5), (10, 10, 50, 50)), (20, 20, 40, 50))
        for end in ((0, 0), (100, 100), (20, 20)):
            with self.subTest(end=end), self.assertRaises(ValueError):
                desktop_tools.cleanup_rectangle((20, 20), end, (1, 1), (10, 10, 50, 50))
        with self.assertRaises(ValueError):
            desktop_tools.cleanup_rectangle((10, 10), (20, 20), (0, 1), (0, 0, 50, 50))

    def test_relative_paths_do_not_mix_notes_from_different_folders(self):
        for name in ("first", "second"):
            (self.folder / name).mkdir()
            record = dict(self.page(), input_path="page.jpg", output_path="page.png", manual_edits=True)
            self.save(f"{name}/page.json", record, 1_000_000_000)
            self.save(f"{name}/run-one.json", {"pages": [dict(record, status="needs_review", message=name)]}, 2_000_000_000)
        rows, errors = desktop_tools.collect_revisions(self.folder)
        self.assertFalse(errors)
        self.assertEqual({Path(row["path"]).parent.name: row["note"] for row in rows},
                         {"first": "first", "second": "second"})

    def test_recorded_times_survive_copying_older_files(self):
        record = self.page(manual_edits=True, last_edit={"at": "2000-01-01T09:00:00+00:00"})
        self.save("page.json", record, 9_000_000_000)
        for name, recorded_time, stamp in (("old", "2000-01-01T08:00:00+00:00", 8_000_000_000),
                                           ("new", "2000-01-01T10:00:00+00:00", 1_000_000_000)):
            self.save(f"run-{name}.json", {"started_at": recorded_time, "pages": [
                dict(self.page(), status="needs_review", message=name)]}, stamp)
        rows, _ = desktop_tools.collect_revisions(self.folder)
        self.assertEqual(rows[0]["note"], "new")

    def test_malformed_record_paths_do_not_hide_valid_protected_pages(self):
        self.save("page.json", self.page(manual_edits=True), 1_000_000_000)
        self.save("bad-page.json", dict(self.page(manual_edits=True), input_path=["invalid"]), 1_000_000_000)
        self.save("run-invalid.json", {"pages": [dict(self.page(), output_path={"invalid": "path"})]}, 2_000_000_000)
        rows, errors = desktop_tools.collect_revisions(self.folder)
        self.assertFalse(errors)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["page"], "page.jpg")


class DesktopToolWindowTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        gc.collect()
        self._cleaned = False
        self.addCleanup(self._cleanup)
        try:
            self.root = gui.tk.Tk()
        except gui.tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.root.withdraw()
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.image = self.folder / "page.png"
        Image.new("RGB", (120, 100), "white").save(self.image)
        self.calls = []
        self.patch = patch.dict(sys.modules, {"pipeline.editing": types.SimpleNamespace(PageEditorSession=fake_editor_session(self.image, self.calls))})
        self.patch.start()
        self.tool = None
        self.closed = False
        self.callback_threads = []

    def on_closed(self):
        if self.tool.worker is not None:
            self.assertFalse(self.tool.worker.is_alive())
        self.callback_threads.append(threading.get_ident())
        self.closed = True

    def pump(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertTrue(condition(), "Tk/worker condition timed out")

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        try:
            if getattr(self, "tool", None) is not None:
                try:
                    if not self.closed:
                        self.tool.request_close()
                        self.pump(lambda: self.closed)
                finally:
                    if self.tool.worker is not None:
                        self.tool.worker.join(timeout=5)
                        self.assertFalse(self.tool.worker.is_alive())
        finally:
            try:
                if hasattr(self, "root"):
                    self.root.destroy()
            finally:
                if hasattr(self, "patch"):
                    self.patch.stop()
                if hasattr(self, "temp"):
                    self.temp.cleanup()
                self.tool = self.root = self.patch = None
                gc.collect()

    def test_cleanup_uses_one_session_preserves_text_and_releases_worker_before_callback(self):
        main_thread = threading.get_ident()
        self.tool = desktop_tools.CleanupWindow(self.root, self.image, on_closed=self.on_closed)
        self.pump(lambda: not self.tool.busy)
        before = copy.deepcopy(self.tool.record["blocks"])
        sx, sy = self.tool.image_scale
        with patch.object(self.tool.canvas, "canvasx", side_effect=lambda x: x), patch.object(self.tool.canvas, "canvasy", side_effect=lambda y: y):
            self.tool._start_region(types.SimpleNamespace(x=20*sx, y=20*sy))
            self.tool._finish_region(types.SimpleNamespace(x=40*sx, y=40*sy))
        self.assertEqual(self.tool.region, (20, 20, 40, 40))
        self.tool._reclean()
        self.pump(lambda: not self.tool.busy)
        self.assertEqual(self.tool.record["blocks"], before)
        self.assertEqual([c[0] for c in self.calls], ["open", "reclean"])
        self.assertEqual(self.calls[1][2], [(20, 20, 40, 40)])
        self.assertEqual(len({c[1] for c in self.calls}), 1)
        self.assertNotEqual(self.calls[0][1], main_thread)
        self.tool.request_close()
        self.pump(lambda: self.closed)
        self.assertEqual(self.calls[-1][0], "close")
        self.assertEqual(self.callback_threads, [main_thread])

    def test_cleanup_rejects_outside_selection_without_enqueuing_work(self):
        self.tool = desktop_tools.CleanupWindow(self.root, self.image, on_closed=self.on_closed)
        self.pump(lambda: not self.tool.busy)
        with patch.object(self.tool.canvas, "canvasx", side_effect=lambda x: x), patch.object(self.tool.canvas, "canvasy", side_effect=lambda y: y):
            self.tool._start_region(types.SimpleNamespace(x=20, y=20))
            self.tool._finish_region(types.SimpleNamespace(x=120, y=99))
        self.assertIsNone(self.tool.region)
        self.tool._reclean()
        self.assertEqual([c[0] for c in self.calls], ["open"])

    def test_close_before_mapping_never_starts_cleanup_session(self):
        self.tool = desktop_tools.CleanupWindow(self.root, self.image, on_closed=self.on_closed)
        self.tool.request_close()
        self.pump(lambda: self.closed)
        self.assertFalse(self.calls)

    def test_revision_scan_runs_off_tk_thread_and_close_waits_for_cancellation(self):
        started, cancelled = threading.Event(), threading.Event()
        scan_threads = []

        def scan(folder, cancel):
            scan_threads.append(threading.get_ident())
            started.set()
            self.assertTrue(cancel.wait(4), "Close must cancel an in-flight scan")
            cancelled.set()
            return [], []
        with patch.object(desktop_tools, "collect_revisions", side_effect=scan):
            self.tool = desktop_tools.RevisionManagerWindow(self.root, self.folder, on_closed=self.on_closed)
            self.pump(started.is_set)
            self.tool.request_close()
            self.assertFalse(self.closed)
            self.pump(lambda: self.closed)
        self.assertTrue(cancelled.is_set())
        self.assertNotEqual(scan_threads, [threading.get_ident()])
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
