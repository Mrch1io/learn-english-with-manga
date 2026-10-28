"""Three-page showcase navigation; real Tk, with no models, network or speech.

Run the Tk case through the private Windows desktop runner when other reader
windows are open. The GUI handoff tests do not create any Tk windows.
"""
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageTk

from ui import app as gui
import presentation
import reader
from core.runtime import Config
import tkinter as tk
from reader import worker as reader_worker


class DemoHandoffTests(unittest.TestCase):
    def test_open_demo_passes_all_three_materialized_pages_to_reader(self):
        paths = [Path("sample") / name for name in ("page-4.json", "page-5.json", "page-6.json")]
        app = types.SimpleNamespace(root=None, _open_reader=Mock(), _show_tasks=Mock())
        with patch.object(presentation, "ensure_demo_pages", return_value=paths, create=True) as materialize, \
             patch.object(presentation, "ensure_demo", side_effect=AssertionError("Do not load just one sample")):
            gui.TranslatorApp._open_demo(app)
        materialize.assert_called_once_with()
        app._open_reader.assert_called_once_with([str(path) for path in paths])
        app._show_tasks.assert_not_called()

    def test_start_reading_uses_the_same_demo_entry_point(self):
        app = types.SimpleNamespace(_open_demo=Mock())
        gui.TranslatorApp.start_reading(app)
        app._open_demo.assert_called_once_with()


class DemoNavigationTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.root = tk.Tk()
        self.root.withdraw()
        self.window = None
        self.closed = False
        self.patches = []
        self.temporary = tempfile.TemporaryDirectory(prefix="manga-demo-navigation-")
        self.addCleanup(self._cleanup)
        self.folder = Path(self.temporary.name)
        self.paths = presentation.ensure_demo_pages(self.folder / "sample")
        self.assertEqual(len(self.paths), 3)
        self.records = [json.loads(path.read_text(encoding="utf-8")) for path in self.paths]

        class Catalog:
            def __init__(self, *args, **kwargs):
                pass

            def list_voices(self):
                return []

        def forbidden(*args, **kwargs):
            raise AssertionError("Reading saved demo pages must not start models, synthesis or playback")

        self.forbidden = Mock(side_effect=forbidden)
        speech = types.SimpleNamespace(VoicevoxClient=Catalog,
            synthesize_cached=self.forbidden, start_local_voicevox=self.forbidden,
            play_wav=self.forbidden, stop_playback=lambda: None)
        self.patches = [patch.dict(sys.modules, {"audio.speech": speech}),
                        patch.object(reader_worker, "SETTINGS_PATH", self.folder / "settings.json"),
                        patch("core.runtime.ModelPool", self.forbidden)]
        for patcher in self.patches:
            patcher.start()
        first = self.records[0]
        self.window = reader.ReaderWindow(self.root, self.paths,
            config=Config(work=first["work"], memory_db=first["config"]["memory_db"]),
            on_closed=self._closed)
        self.window.autoplay.set(False)
        self.window.window.geometry("1250x850")
        self._wait_for_page(0)

    def _closed(self):
        self.closed = True

    def _pump(self, condition, timeout=10):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertTrue(condition(), "Demo navigation did not settle before the timeout")

    def _wait_for_page(self, index):
        expected = str(self.paths[index].resolve())
        self._pump(lambda: self.window.page_index == index and not self.window.loading
                   and self.window.current_result_path() == expected and self.window.photo is not None)
        self._pump(lambda: not self.window.resizing and self.window.after_resize is None)

    def _assert_translated_page(self, index):
        self._wait_for_page(index)
        self.assertEqual(self.window.page_position.get(), f"{index + 1} / 3")
        self.assertEqual(self.window.page_list.curselection(), (index,))
        self.assertEqual(self.window.view.get(), "英文译图")
        self.assertEqual(self.window.view_button.cget("text"), "Show Original")
        self.assertEqual(self.window.record["input_sha256"], self.records[index]["input_sha256"])
        with Image.open(self.paths[index].with_suffix(".png")) as opened:
            saved = opened.convert("RGB")
        self.assertEqual(self.window.original.size, saved.size)
        self.assertEqual(self.window.original.tobytes(), saved.tobytes())
        thumbnail = ImageTk.getimage(self.window.thumbnail).convert("RGB")
        factor = min(132 / saved.width, 184 / saved.height)
        expected = saved.resize((round(saved.width * factor), round(saved.height * factor)), Image.Resampling.LANCZOS)
        self.assertEqual(thumbnail.size, expected.size)
        self.assertEqual(thumbnail.tobytes(), expected.tobytes())
        self.assertEqual(self.window.thumbnail_label.tk.splitlist(self.window.thumbnail_label.cget("image")),
                         (str(self.window.thumbnail),))
        return hashlib.sha256(thumbnail.tobytes()).hexdigest()

    def test_next_previous_pages_keep_translated_pixels_and_valid_selection(self):
        self.assertEqual([page["path"] for page in self.window.pages],
                         [str(path.resolve()) for path in self.paths])
        self.assertEqual(self.window.page_list.size(), 3)
        self.assertIn("disabled", self.window.previous_button.state())
        self.assertNotIn("disabled", self.window.next_button.state())
        thumbnails = [self._assert_translated_page(0)]
        first_key = self.records[0]["blocks"][1]["block_id"]
        self.window.bubble_list.selection_set(first_key)
        self.window._select_bubble()
        self.assertEqual(self.window.block_id, first_key)
        first_role = self.window.role.get()

        for index in (1, 2):
            self.window.next_button.invoke()
            thumbnails.append(self._assert_translated_page(index))
            self.assertNotIn(first_key, self.window.blocks)
            self.assertIn(self.window.block_id, self.window.blocks)
        self.assertEqual(len(set(thumbnails)), 3, "Each page needs its own visible thumbnail")
        self.assertIn("disabled", self.window.next_button.state())
        self.assertNotIn("disabled", self.window.previous_button.state())

        for index in (1, 0):
            self.window.previous_button.invoke()
            self.assertEqual(self._assert_translated_page(index), thumbnails[index])
        self.window.bubble_list.selection_set(first_key)
        self.window._select_bubble()
        self.assertEqual(self.window.block_id, first_key)
        self.assertEqual(self.window.role.get(), first_role)
        self.assertEqual(self.window.selected_bubble.get(), "Dialogue selected")
        self.assertEqual(len(self.window.canvas.find_withtag("bubble")), 1)
        self.forbidden.assert_not_called()

    def _cleanup(self):
        try:
            if self.window is not None and not self.closed:
                self.window.close()
                self._pump(lambda: self.closed)
            if self.window is not None:
                self.window.worker.join(timeout=3)
                self.assertFalse(self.window.worker.is_alive())
        finally:
            self.root.destroy()
            for patcher in reversed(self.patches):
                patcher.stop()
            self.temporary.cleanup()
            self.window = self.root = None
            gc.collect()


if __name__ == "__main__":
    unittest.main()
