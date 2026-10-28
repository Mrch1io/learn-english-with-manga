"""GUI boundary checks without loading any ML libraries or using a GPU."""

import queue
import copy
import gc
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ui import app as gui


def fake_editor_session(image_path, calls):
    """A deterministic editing contract; it never imports a model library."""
    class FakeSession:
        def __init__(self, result_path):
            calls.append(("open", threading.get_ident(), result_path))
            self.record = {"output_path": str(image_path), "input_path": str(image_path), "blocks": [
                {"block_id": "b-one", "box": [10, 10, 50, 50], "status": "translated",
                 "source_text": "原文一", "translated_text": "First", "metadata": {"ocr_box_audit": [
                     {"classification": "expansion_candidate", "crop_box": [5, 5, 55, 55]}]}},
                {"block_id": "b-two", "box": [60, 10, 110, 80], "status": "translated",
                 "source_text": "原文二", "translated_text": "Second"}]}
            self.candidates = {}

        def rerender(self, block_id, text, *, cancel=None):
            cancel()
            calls.append(("rerender", threading.get_ident(), text))
            if text == "__overflow__":
                raise ValueError("text overflow")
            block = next(block for block in self.record["blocks"] if block["block_id"] == block_id)
            block["translated_text"] = text
            return self.record

        def retranslate(self, block_id, text, *, cancel=None):
            cancel()
            calls.append(("retranslate", threading.get_ident(), text))
            block = next(block for block in self.record["blocks"] if block["block_id"] == block_id)
            block.update(source_text=text, translated_text="Translated: " + text)
            return self.record

        def close(self):
            calls.append(("close", threading.get_ident(), None))

        def preview_ocr(self, block_id, crop_box, *, enabled=False, cancel=None):
            cancel()
            if not enabled:
                raise ValueError("disabled")
            calls.append(("preview_ocr", threading.get_ident(), crop_box))
            block = next(block for block in self.record["blocks"] if block["block_id"] == block_id)
            candidate = {"candidate_id": f"candidate-{len(self.candidates) + 1}", "crop_box": crop_box, "crop_sha256": "fixture",
                         "text": "候補原文", "raw_text": "候補原文", "status": "ok", "error": "", "seconds": 0.01,
                         "base_source_text": block["source_text"], "base_translated_text": block["translated_text"]}
            self.candidates[candidate["candidate_id"]] = candidate
            block.setdefault("metadata", {}).setdefault("ocr_candidates", []).append(candidate)
            return candidate

        def list_ocr_candidates(self, block_id):
            block = next(block for block in self.record["blocks"] if block["block_id"] == block_id)
            candidates = copy.deepcopy(block.get("metadata", {}).get("ocr_candidates", []))
            for candidate in candidates:
                candidate["usable"] = (not candidate.get("accepted") and candidate.get("status") == "ok"
                    and bool(candidate.get("text", "").strip())
                    and candidate.get("base_source_text") == block["source_text"]
                    and candidate.get("base_translated_text") == block["translated_text"])
                candidate["unavailable_reason"] = "" if candidate["usable"] else "候选已过期或不可采用。"
            return candidates

        def preview_region(self, block_id, crop_box=None, *, cancel=None):
            from PIL import Image
            cancel()
            calls.append(("preview_region", threading.get_ident(), crop_box))
            block = next(block for block in self.record["blocks"] if block["block_id"] == block_id)
            crop_box = list(crop_box or block["box"])
            with Image.open(image_path) as opened:
                original = opened.convert("RGB").crop(crop_box)
            return {"block_id": block_id, "crop_box": crop_box, "original": original,
                    "translated": original.copy() if self.record.get("exported", True) else None}

        def apply_ocr_candidate(self, block_id, candidate_id, *, enabled=False, cancel=None):
            cancel()
            if not enabled:
                raise ValueError("disabled")
            calls.append(("apply_ocr_candidate", threading.get_ident(), candidate_id))
            block = next(block for block in self.record["blocks"] if block["block_id"] == block_id)
            block.update(source_text=self.candidates[candidate_id]["text"], translated_text="Candidate translation")
            self.candidates[candidate_id]["accepted"] = True
            return self.record

        def reclean(self, block_id, regions, *, cancel=None):
            cancel()
            calls.append(("reclean", threading.get_ident(), regions))
            self.record["regions"] = regions
            return self.record

        def export_image(self, *, cancel=None):
            cancel()
            calls.append(("export", threading.get_ident(), None))
            self.record["exported"] = True
            return self.record

    return FakeSession


class GuiBoundaryTests(unittest.TestCase):
    def test_output_is_separate_and_aliases_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "1.png"
            source.touch()
            self.assertEqual(Path(gui.default_output(str(source))), Path(directory) / "translated_output")
            self.assertEqual(Path(gui.default_output(directory)), Path(directory) / "translated_output")
        self.assertEqual(gui.parse_aliases("名, なまえ，名\n別名\n"), ("名", "なまえ", "別名"))

    def test_worker_reports_progress_and_result_without_touching_tk(self):
        events = queue.Queue()
        cancellation = threading.Event()
        caller = threading.get_ident()
        called_threads = []

        def fake_batch(input_path, output_dir, config, *, progress, cancel_event):
            called_threads.append(threading.get_ident())
            self.assertEqual((input_path, output_dir, config), ("input", "output", "config"))
            self.assertIs(cancel_event, cancellation)
            progress({"type": "page_done", "current": 1, "total": 1})
            return {"completed": 1}

        with patch.dict(sys.modules, {"pipeline.batch": types.SimpleNamespace(run_batch=fake_batch)}):
            worker = threading.Thread(target=gui._run_worker,
                                      args=("input", "output", "config", events, cancellation))
            worker.start()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        self.assertNotEqual(called_threads, [caller])
        self.assertEqual(events.get_nowait(), ("progress", {"type": "page_done", "current": 1, "total": 1}))
        self.assertEqual(events.get_nowait(), ("finished", {"completed": 1}))

    def test_worker_failure_always_returns_terminal_event(self):
        for error in (RuntimeError("sample error"), SystemExit("dependency missing")):
            with self.subTest(error=type(error).__name__):
                events = queue.Queue()

                def fake_batch(*args, **kwargs):
                    raise error

                with patch.dict(sys.modules, {"pipeline.batch": types.SimpleNamespace(run_batch=fake_batch)}):
                    gui._run_worker("input", "output", None, events, threading.Event())
                event, message = events.get_nowait()
                self.assertEqual(event, "failed")
                self.assertIn(str(error), message)

    def test_window_constructs_without_models(self):
        try:
            root = gui.tk.Tk()
        except gui.tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        root.withdraw()
        app = None
        try:
            app = gui.TranslatorApp(root)
            self.assertFalse(app.running)
            self.assertEqual(app.input_path.get(), "")
            self.assertEqual(app.language_box["values"], ("English (en)",))
            self.assertEqual(app._config().translation_backend, "vntl")
            self.assertEqual(app._config().translation_mode, "auto")
            self.assertTrue(app._config().use_fragment_links)
            app.use_fragment_links.set(False)
            self.assertFalse(app._config().use_fragment_links)
            self.assertFalse(app._config().use_hybrid_ocr)
            self.assertEqual(app._config().ocr_prob_threshold, .1)
            app.use_hybrid_ocr.set(True)
            app.ocr_prob_threshold.set("0.2")
            self.assertTrue(app._config().use_hybrid_ocr)
            self.assertEqual(app._config().ocr_prob_threshold, .2)
            self.assertFalse(app._config().ocr_candidate_review)
            app.ocr_candidate_review.set(True)
            self.assertTrue(app._config().ocr_candidate_review)
            app._set_running(True)
            self.assertIn("disabled", app.start_button.state())
            self.assertIn("disabled", app.translator_box.state())
            self.assertIn("disabled", app.hybrid_check.state())
            self.assertIn("disabled", app.hybrid_threshold_entry.state())
            app._cancel()
            self.assertTrue(app.cancel_event.is_set())
            app._set_running(False)
            self.assertIn("readonly", app.language_box.state())
            self.assertIn("readonly", app.translator_box.state())
            self.assertNotIn("disabled", app.translator_box.state())
            app.translator.set("Qwen 3B + LoRA（旧版）")
            self.assertEqual(app._config().translation_backend, "qwen")
            self.assertEqual(app._config().translation_mode, "auto")
        finally:
            if app is not None:
                app._destroy()
            else:
                root.destroy()
            app = root = None
            gc.collect()

    def test_multiple_input_selection_and_text_only_config(self):
        root = gui.tk.Tk()
        root.withdraw()
        app = gui.TranslatorApp(root)
        try:
            with patch.object(gui.filedialog, "askopenfilenames", return_value=("2.png", "10.png")):
                app._choose_image()
            self.assertEqual(app.input_paths, ["2.png", "10.png"])
            app.text_only.set(True)
            self.assertFalse(app._config().export_images)
            app.input_path.set("manually-changed.png")
            self.assertIsNone(app.input_paths)
        finally:
            app._destroy()
            app = root = None
            gc.collect()

    def test_editor_session_and_preview_stay_on_one_worker_with_bounded_cancel(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "page.png"
            Image.new("RGB", (1200, 900), "white").save(image_path)
            calls, events, commands, cancel = [], queue.Queue(), queue.Queue(), threading.Event()
            session_type = fake_editor_session(image_path, calls)
            with patch.dict(sys.modules, {"pipeline.editing": types.SimpleNamespace(PageEditorSession=session_type)}):
                worker = threading.Thread(target=gui._page_editor_worker,
                                          args=(str(image_path), commands, events, cancel))
                worker.start()
                try:
                    kind, opened = events.get(timeout=5)
                    self.assertEqual(kind, "opened")
                    self.assertEqual(opened["image_size"], (1200, 900))
                    self.assertEqual(opened["preview"].size, (720, 540))
                    commands.put(("rerender", "b-one", ""))
                    kind, saved = events.get(timeout=5)
                    self.assertEqual(kind, "saved")
                    self.assertEqual(saved["record"]["blocks"][0]["translated_text"], "")
                    # Event records are snapshots; later session writes cannot mutate them.
                    self.assertEqual(opened["record"]["blocks"][0]["translated_text"], "First")
                    commands.put(("retranslate", "b-one", "訂正"))
                    self.assertEqual(events.get(timeout=5)[0], "saved")
                    cancel.set()
                    commands.put(("rerender", "b-one", "never committed"))
                    kind, error = events.get(timeout=5)
                    self.assertEqual(kind, "error")
                    self.assertIn("Cancelled", error)
                finally:
                    commands.put(("close", None, None))
                    worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
                self.assertEqual(events.get(timeout=5)[0], "closed")
            self.assertEqual([call[0] for call in calls], ["open", "rerender", "retranslate", "close"])
            self.assertEqual(len({call[1] for call in calls}), 1)
            self.assertNotEqual(calls[0][1], threading.get_ident())

    def test_region_preview_uses_editor_worker_without_save_or_model_actions(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "page.png"
            Image.new("RGB", (120, 100), "white").save(image_path)
            calls, events, commands, cancel = [], queue.Queue(), queue.Queue(), threading.Event()
            base = fake_editor_session(image_path, calls)
            class TextOnlySession(base):
                def __init__(self, path):
                    super().__init__(path)
                    self.record["exported"] = False
            with patch.dict(sys.modules, {"pipeline.editing": types.SimpleNamespace(PageEditorSession=TextOnlySession)}):
                worker = threading.Thread(target=gui._page_editor_worker,
                                          args=(str(image_path), commands, events, cancel))
                worker.start()
                try:
                    self.assertEqual(events.get(timeout=5)[0], "opened")
                    commands.put(("preview_region", "b-one", [0, 0, 60, 70]))
                    kind, result = events.get(timeout=5)
                    self.assertEqual(kind, "region_preview")
                    self.assertEqual(result["crop_box"], [0, 0, 60, 70])
                    self.assertEqual(result["original"].size, (60, 70))
                    self.assertIsNone(result["translated"])
                finally:
                    commands.put(("close", None, None))
                    worker.join(timeout=5)
            self.assertEqual([call[0] for call in calls], ["open", "preview_region", "close"])
            self.assertEqual(len({call[1] for call in calls}), 1)
            self.assertNotEqual(calls[0][1], threading.get_ident())


class LauncherFlowTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self._cleaned = False
        self.addCleanup(self._cleanup)
        self.root = gui.tk.Tk()
        self.root.withdraw()
        self.app = gui.TranslatorApp(self.root)
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.app.output_dir.set(str(self.output))
        self.factory = Mock()
        self.reader_patch = patch.dict(sys.modules, {"reader": types.SimpleNamespace(ReaderWindow=self.factory)})
        self.reader_patch.start()

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        try:
            if hasattr(self, "app"):
                self.app._destroy()
            elif hasattr(self, "root"):
                self.root.destroy()
        finally:
            if hasattr(self, "reader_patch"):
                self.reader_patch.stop()
            if hasattr(self, "temp"):
                self.temp.cleanup()
            for name in ("app", "root", "factory", "reader_patch"):
                setattr(self, name, None)
            gc.collect()

    def finish(self, pages, *, cancelled=False):
        self.app._set_running(True)
        self.app.events.put(("finished", {"pages": pages, "cancelled": cancelled}))
        deadline = time.monotonic() + 3
        while self.app.running and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertFalse(self.app.running)

    def page(self, name, status="done"):
        return {"input_path": str(self.output / name), "output_path": str(self.output / (name + ".png")),
                "status": status, "message": "2/2 bubbles", "blocks": []}

    def test_finished_opens_only_this_runs_readable_pages(self):
        pages = [self.page("2", "done"), self.page("3", "partial"), self.page("4", "failed"),
                 self.page("5", "skipped"), self.page("6", "needs_review")]
        self.finish(pages)
        expected = [str(self.output / (name + ".json")) for name in ("2", "3", "5", "6")]
        self.assertEqual(self.factory.call_args.kwargs["paths"], expected)
        self.assertEqual(self.app.last_result_paths, expected)
        self.assertEqual(len(self.app.result_tree.get_children()), 5)
        self.assertEqual(self.app.result_tree.item("page-2", "values")[1], "Failed")
        self.assertIn("disabled", self.app.start_button.state())
        self.factory.call_args.kwargs["on_closed"]()
        self.assertNotIn("disabled", self.app.start_button.state())

    def test_failed_or_cancelled_run_does_not_open_reader_automatically(self):
        self.finish([self.page("broken", "failed")])
        self.factory.assert_not_called()
        self.assertEqual(self.app.last_result_paths, [])
        self.finish([self.page("completed-before-cancel")], cancelled=True)
        self.factory.assert_not_called()
        self.assertEqual(len(self.app.last_result_paths), 1)

    def test_opt_out_then_open_selected_result(self):
        self.app.auto_read.set(False)
        self.finish([self.page("2"), self.page("10")])
        self.factory.assert_not_called()
        self.app.result_tree.selection_set("page-1")
        self.app._open_selected_result()
        self.assertEqual(self.factory.call_args.kwargs["paths"], [str(self.output / "10.json")])

    def test_manual_reader_opens_source_with_current_model_and_output_settings(self):
        self.assertTrue(self.app.auto_read.get())
        self.assertTrue(self.app._config().export_images)
        self.app.input_path.set(str(self.output / "selected-page.png"))
        self.app._open_reader()
        self.assertEqual(self.factory.call_args.kwargs["paths"], self.app.input_path.get())
        self.assertEqual(self.factory.call_args.kwargs["output_dir"], str(self.output))
        self.assertEqual(self.factory.call_args.kwargs["config"].translation_backend, "vntl")

    def test_empty_input_does_not_create_translation_worker(self):
        with patch.object(gui.messagebox, "showerror") as error, \
                patch.object(gui.threading, "Thread") as worker:
            self.app._start()
        error.assert_called_once()
        worker.assert_not_called()
        self.factory.assert_not_called()
        self.assertFalse(self.app.running)

    def test_reader_prompts_for_folder_when_input_is_empty(self):
        with patch.object(gui.filedialog, "askdirectory", return_value=str(self.output)) as choose:
            self.app._open_reader()
        choose.assert_called_once()
        self.factory.assert_called_once()
        self.assertEqual(self.factory.call_args.kwargs["paths"], str(self.output))

    def test_cancelled_reader_input_does_not_open_a_session(self):
        with patch.object(gui.filedialog, "askdirectory", return_value="") as choose:
            self.app._open_reader()
        choose.assert_called_once()
        self.factory.assert_not_called()
        self.assertIsNone(self.app.reader_window)
        self.assertFalse(self.app.running)

    def test_unreadable_page_double_click_does_not_open_failed_record(self):
        self.app.auto_read.set(False)
        self.finish([self.page("bad", "failed")])
        self.app.result_tree.selection_set("page-0")
        self.app._open_selected_result()
        self.factory.assert_not_called()


class PageEditorTests(unittest.TestCase):
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
        self.image_path = Path(self.temp.name) / "page.png"
        Image.new("RGB", (120, 100), "white").save(self.image_path)
        self.calls = []
        self.session_patch = patch.dict(sys.modules, {"pipeline.editing": types.SimpleNamespace(
            PageEditorSession=fake_editor_session(self.image_path, self.calls))})
        self.session_patch.start()
        self.errors = patch.object(gui.messagebox, "showerror")
        self.errors.start()
        self.closed = False
        self.editor = gui.PageEditor(self.root, str(self.image_path), on_closed=self.mark_closed)
        self.pump_until(lambda: not self.editor.busy)

    def mark_closed(self):
        self.closed = True

    def pump_until(self, condition, timeout=4):
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
            if hasattr(self, "editor"):
                try:
                    if not self.closed:
                        with patch.object(gui.messagebox, "askyesno", return_value=True):
                            self.editor.request_close()
                        self.pump_until(lambda: self.closed)
                finally:
                    if self.editor.worker is not None:
                        self.editor.worker.join(timeout=5)
                        self.assertFalse(self.editor.worker.is_alive())
        finally:
            try:
                if hasattr(self, "main_app"):
                    self.main_app._destroy()
                elif hasattr(self, "root"):
                    self.root.destroy()
            finally:
                for name in ("errors", "session_patch"):
                    patcher = getattr(self, name, None)
                    if patcher is not None:
                        patcher.stop()
                        setattr(self, name, None)
                patcher = None
                if hasattr(self, "temp"):
                    self.temp.cleanup()
                self.editor = self.main_app = self.root = None
                gc.collect()

    def test_close_removes_coordinate_traces_and_releases_preview_images(self):
        self.assertTrue(all(value.trace_info() for value in self.editor.ocr_coordinates))
        self.assertIsNotNone(self.editor.photo)
        self.editor.request_close()
        self.pump_until(lambda: self.closed)
        self.assertTrue(all(not value.trace_info() for value in self.editor.ocr_coordinates))
        self.assertEqual(self.editor._ocr_coordinate_traces, [])
        self.assertIsNone(self.editor.photo)
        self.assertIsNone(self.editor.thumbnail_photo)
        self.assertEqual(self.editor.comparison_photos, [])

    @staticmethod
    def replace_text(widget, value):
        widget.delete("1.0", "end")
        widget.insert("1.0", value)

    def test_preview_selection_preserves_drafts_and_empty_translation_is_valid(self):
        self.assertEqual(self.editor.selected_id, "b-one")
        self.replace_text(self.editor.source_text, "OCR draft")
        self.replace_text(self.editor.translated_text, "")
        self.editor._click_preview(types.SimpleNamespace(
            x=70-self.editor.canvas.canvasx(0), y=20-self.editor.canvas.canvasy(0)))
        self.assertEqual(self.editor.selected_id, "b-two")
        self.editor.table.selection_set("b-one")
        self.editor._select()
        self.assertEqual(self.editor.source_text.get("1.0", "end-1c"), "OCR draft")
        self.assertEqual(self.editor.translated_text.get("1.0", "end-1c"), "")
        self.editor._save("rerender")
        self.pump_until(lambda: not self.editor.busy)
        self.assertEqual(self.editor.blocks["b-one"]["translated_text"], "")
        self.assertEqual(self.editor.blocks["b-one"]["source_text"], "原文一")
        self.assertEqual(self.editor.drafts["b-one"][0], "OCR draft")
        self.assertTrue(self.editor._is_dirty("b-one"))

    def test_overflow_preserves_draft_and_old_record_then_can_retry(self):
        self.replace_text(self.editor.translated_text, "__overflow__")
        self.editor._save("rerender")
        self.pump_until(lambda: not self.editor.busy)
        self.assertIn("overflow", self.editor.notice.get())
        self.assertTrue(self.editor.notice.get().startswith("Operation failed; drafts are retained."))
        self.assertEqual(self.editor.blocks["b-one"]["translated_text"], "First")
        self.assertEqual(self.editor.translated_text.get("1.0", "end-1c"), "__overflow__")
        self.replace_text(self.editor.translated_text, "Fits")
        self.editor._save("rerender")
        self.pump_until(lambda: not self.editor.busy)
        self.assertEqual(self.editor.blocks["b-one"]["translated_text"], "Fits")
        self.assertEqual(sum(call[0] == "open" for call in self.calls), 1)

    def test_empty_ocr_is_rejected_and_retranslation_updates_only_selected_bubble(self):
        self.replace_text(self.editor.source_text, "  \n")
        self.editor._save("retranslate")
        self.assertFalse(self.editor.busy)
        self.assertIn("cannot be empty", self.editor.notice.get())
        self.assertFalse(any(call[0] == "retranslate" for call in self.calls))
        self.replace_text(self.editor.source_text, "訂正済み")
        self.editor._save("retranslate")
        self.pump_until(lambda: not self.editor.busy)
        self.assertEqual(self.editor.blocks["b-one"]["translated_text"], "Translated: 訂正済み")
        self.assertEqual(self.editor.blocks["b-two"]["translated_text"], "Second")
        self.assertFalse(self.editor._is_dirty("b-one"))

    def test_close_can_be_declined_and_callback_follows_session_release(self):
        self.replace_text(self.editor.translated_text, "Unsaved")
        with patch.object(gui.messagebox, "askyesno", return_value=False):
            self.assertFalse(self.editor.request_close())
        self.assertFalse(self.editor.closing)
        self.assertFalse(self.closed)
        with patch.object(gui.messagebox, "askyesno", return_value=True):
            self.assertTrue(self.editor.request_close())
        self.pump_until(lambda: self.closed)
        self.assertEqual(self.calls[-1][0], "close")

    def test_main_window_disables_batch_and_terms_until_editor_releases(self):
        app = gui.TranslatorApp(self.root)
        self.main_app = app
        app.page_editor = self.editor
        app._set_running(False)
        self.assertIn("disabled", app.start_button.state())
        self.assertIn("disabled", app.edit_button.state())
        with patch.object(app, "_config", side_effect=AssertionError("must not start")):
            app._start()
            app._open_terms()
        app.page_editor = None
        app._set_running(False)
        self.assertNotIn("disabled", app.start_button.state())

    def test_original_preview_has_no_cleanup_drag_and_export_keeps_text_draft(self):
        self.replace_text(self.editor.source_text, "Unsubmitted OCR")
        self.editor.show_original.set(True)
        self.editor._display_preview()
        self.assertEqual(self.editor.canvas.bind("<B1-Motion>"), "")
        self.assertEqual(self.editor.canvas.bind("<ButtonRelease-1>"), "")
        self.assertFalse(hasattr(self.editor, "reclean_button"))
        self.assertNotIn("Repair", [self.editor.tabs.tab(tab, "text") for tab in self.editor.tabs.tabs()])
        self.assertFalse(any(call[0] == "reclean" for call in self.calls))
        self.editor._save("export")
        self.pump_until(lambda: not self.editor.busy)
        self.assertTrue(self.editor.record["exported"])
        self.assertEqual(self.editor.drafts["b-one"][0], "Unsubmitted OCR")

    def test_text_only_snapshot_uses_original_without_missing_translation_warning(self):
        record = dict(self.editor.record, output_path=str(Path(self.temp.name)/"not-exported.png"), exported=False)
        snapshot = gui._editor_snapshot(types.SimpleNamespace(record=record))
        self.assertIsNone(snapshot["preview"])
        self.assertIsNotNone(snapshot["original_preview"])
        self.assertEqual(snapshot["warning"], "")
        self.editor._apply_snapshot(snapshot)
        self.assertTrue(self.editor.show_original.get())
        self.assertIsNotNone(self.editor.photo)

    def enable_ocr(self):
        self.editor.ocr_enabled.set(True)
        self.editor._toggle_ocr()

    def preview_ocr(self):
        self.editor._preview_ocr()
        self.pump_until(lambda: not self.editor.busy)

    def test_local_ocr_default_off_and_separate_from_cleanup_coordinates(self):
        self.assertFalse(self.editor.ocr_enabled.get())
        self.assertIn("disabled", self.editor.ocr_preview_button.state())
        self.assertTrue(all("disabled" in entry.state() for entry in self.editor.ocr_entries))
        self.editor._preview_ocr()
        self.assertFalse(any(call[0] == "preview_ocr" for call in self.calls))
        self.enable_ocr()
        self.assertTrue(self.editor.show_original.get())
        self.assertIn("Expansion candidate", self.editor.ocr_audit.get())
        self.assertFalse(hasattr(self.editor, "reclean_button"))
        self.editor.ocr_crop_choice.set(next(name for name in self.editor.ocr_crop_choices if "Expansion candidate" in name))
        self.editor._choose_ocr_crop()
        self.assertEqual(self.editor._read_ocr_crop(), [5, 5, 55, 55])
        self.assertEqual(self.editor.canvas.bind("<B1-Motion>"), "")
        self.assertEqual(self.editor.blocks["b-one"]["box"], [10, 10, 50, 50])
        self.assertEqual(len(self.editor.canvas.find_withtag("ocr_crop")), 1)
        self.editor.ocr_enabled.set(False)
        self.editor._toggle_ocr()
        self.assertFalse(self.editor.canvas.find_withtag("ocr_crop"))

    def test_local_ocr_worker_preserves_drafts_and_requires_explicit_apply(self):
        self.enable_ocr()
        self.replace_text(self.editor.source_text, "Manual source")
        self.replace_text(self.editor.translated_text, "Manual translation")
        self.preview_ocr()
        self.assertEqual(self.editor.blocks["b-one"]["source_text"], "原文一")
        self.assertEqual(self.editor.source_text.get("1.0", "end-1c"), "Manual source")
        self.assertEqual(self.editor.translated_text.get("1.0", "end-1c"), "Manual translation")
        self.assertEqual(self.editor.ocr_candidate_text.get("1.0", "end-1c"), "候補原文")
        self.assertEqual(self.editor.ocr_candidate_text["state"], "disabled")
        self.editor._apply_ocr_candidate()
        self.assertIn("unsaved drafts", self.editor.notice.get())
        self.assertFalse(any(call[0] == "apply_ocr_candidate" for call in self.calls))
        self.replace_text(self.editor.source_text, "原文一")
        self.replace_text(self.editor.translated_text, "First")
        self.editor._apply_ocr_candidate()
        self.pump_until(lambda: not self.editor.busy)
        self.assertEqual(self.editor.blocks["b-one"]["source_text"], "候補原文")
        self.assertEqual(self.editor.translated_text.get("1.0", "end-1c"), "Candidate translation")
        self.assertEqual(self.editor.blocks["b-two"]["translated_text"], "Second")
        self.assertFalse(self.editor._is_dirty("b-one"))
        self.assertIsNone(self.editor.ocr_candidate)
        model_calls = [call for call in self.calls if call[0] in ("open", "preview_ocr", "apply_ocr_candidate")]
        self.assertEqual([call[0] for call in model_calls], ["open", "preview_ocr", "apply_ocr_candidate"])
        self.assertEqual(len({call[1] for call in model_calls}), 1)
        self.assertNotEqual(model_calls[0][1], threading.get_ident())

    def test_local_ocr_coordinate_or_bubble_change_invalidates_candidate(self):
        self.enable_ocr()
        self.preview_ocr()
        self.assertNotIn("disabled", self.editor.ocr_apply_button.state())
        self.editor.ocr_coordinates[0].set("5")
        self.assertIsNone(self.editor.ocr_candidate)
        self.assertIn("disabled", self.editor.ocr_apply_button.state())
        self.editor._apply_ocr_candidate()
        self.assertFalse(any(call[0] == "apply_ocr_candidate" for call in self.calls))
        self.preview_ocr()
        self.editor.table.selection_set("b-two")
        self.editor._select()
        self.assertIsNone(self.editor.ocr_candidate)
        self.assertEqual(self.editor._read_ocr_crop(), [60, 10, 110, 80])
        self.assertIn("disabled", self.editor.ocr_apply_button.state())

    def test_local_ocr_rejects_invalid_crop_and_failed_or_empty_candidate(self):
        self.enable_ocr()
        for coordinates in ([0, 0, 121, 100], [10, 10, 10, 20], ["bad", 0, 10, 10]):
            self.editor._set_ocr_crop(coordinates)
            self.editor._preview_ocr()
            self.assertFalse(self.editor.busy)
        self.assertFalse(any(call[0] == "preview_ocr" for call in self.calls))
        self.editor._set_ocr_crop([10, 10, 50, 50])
        self.preview_ocr()
        for status, text in (("empty", ""), ("failed", "untrusted"), ("ok", "  ")):
            self.editor.ocr_candidate.update(status=status, text=text)
            self.editor._set_busy(False)
            self.assertIn("disabled", self.editor.ocr_apply_button.state())
            self.editor._apply_ocr_candidate()
        self.assertFalse(any(call[0] == "apply_ocr_candidate" for call in self.calls))

    def test_saved_ocr_can_be_reselected_after_switching_bubbles_without_inference(self):
        self.enable_ocr()
        self.preview_ocr()
        before_calls = len(self.calls)
        self.replace_text(self.editor.source_text, "Source draft")
        self.replace_text(self.editor.translated_text, "Translation draft")
        self.editor.table.selection_set("b-two")
        self.editor._select()
        self.editor.table.selection_set("b-one")
        self.editor._select()
        self.assertIsNone(self.editor.ocr_candidate)
        self.assertEqual(len(self.editor.saved_ocr_candidates), 1)
        self.editor.saved_ocr_choice.set(next(iter(self.editor.saved_ocr_candidates)))
        self.editor._choose_saved_ocr()
        self.assertEqual(len(self.calls), before_calls)
        self.assertEqual(self.editor.ocr_candidate_text.get("1.0", "end-1c"), "候補原文")
        self.assertEqual(self.editor.source_text.get("1.0", "end-1c"), "Source draft")
        self.assertEqual(self.editor.translated_text.get("1.0", "end-1c"), "Translation draft")
        self.editor._apply_ocr_candidate()
        self.assertIn("unsaved drafts", self.editor.notice.get())
        self.assertEqual(len(self.calls), before_calls)

    def test_opened_snapshot_exposes_saved_candidates_without_ocr(self):
        candidate = {"candidate_id": "persisted", "crop_box": [5, 5, 55, 55], "status": "ok",
                     "text": "Persisted OCR", "usable": True, "unavailable_reason": "",
                     "base_source_text": "原文一"}
        snapshot = dict(self.editor.snapshot, ocr_candidates={"b-one": [candidate]})
        self.editor._apply_snapshot(snapshot)
        self.enable_ocr()
        self.editor.saved_ocr_choice.set(next(iter(self.editor.saved_ocr_candidates)))
        self.editor._choose_saved_ocr()
        self.assertEqual(self.editor.ocr_candidate["candidate_id"], "persisted")
        self.assertEqual(self.editor._read_ocr_crop(), [5, 5, 55, 55])
        self.assertEqual(self.editor.ocr_candidate_text.get("1.0", "end-1c"), "Persisted OCR")
        self.assertEqual([call[0] for call in self.calls], ["open"])
        self.assertNotIn("disabled", self.editor.ocr_apply_button.state())

    def test_saved_candidate_remains_readable_after_text_edit_but_cannot_be_adopted(self):
        self.enable_ocr()
        self.preview_ocr()
        self.replace_text(self.editor.translated_text, "New human correction")
        self.editor._save("rerender")
        self.pump_until(lambda: not self.editor.busy)
        self.assertEqual(self.editor.ocr_candidate_text.get("1.0", "end-1c"), "候補原文")
        self.assertIn("过期", self.editor.ocr_candidate_status.get())
        self.assertIn("disabled", self.editor.ocr_apply_button.state())
        self.editor._apply_ocr_candidate()
        self.assertFalse(any(call[0] == "apply_ocr_candidate" for call in self.calls))
        self.assertEqual(sum(call[0] == "preview_ocr" for call in self.calls), 1)

    def test_unusable_saved_candidate_with_invalid_geometry_is_readable_only(self):
        snapshot = dict(self.editor.snapshot, ocr_candidates={"b-one": [{
            "candidate_id": "bad", "crop_box": None, "status": "ok", "text": "Damaged candidate",
            "usable": False, "unavailable_reason": "裁剪坐标无效。"}]})
        self.editor._apply_snapshot(snapshot)
        self.enable_ocr()
        self.editor.saved_ocr_choice.set(next(iter(self.editor.saved_ocr_candidates)))
        self.editor._choose_saved_ocr()
        self.assertEqual(self.editor.ocr_candidate_text.get("1.0", "end-1c"), "Damaged candidate")
        self.assertIn("disabled", self.editor.ocr_apply_button.state())
        self.assertEqual([call[0] for call in self.calls], ["open"])

    def test_local_comparison_keeps_drafts_and_uses_matching_read_only_crops(self):
        self.replace_text(self.editor.source_text, "Unsaved source")
        self.replace_text(self.editor.translated_text, "Unsaved translation")
        record = copy.deepcopy(self.editor.record)
        self.editor._preview_region()
        self.assertTrue(self.editor.busy)
        self.editor.table.selection_set("b-two")
        self.editor._select()
        self.assertEqual(self.editor.selected_id, "b-one")
        self.pump_until(lambda: not self.editor.busy)
        self.assertIsNotNone(self.editor.comparison_window)
        self.assertEqual(len(self.editor.comparison_photos), 2)
        self.assertEqual([(photo.width(), photo.height()) for photo in self.editor.comparison_photos], [(40, 40), (40, 40)])
        self.assertEqual(self.editor.record, record)
        self.assertEqual(self.editor.source_text.get("1.0", "end-1c"), "Unsaved source")
        self.assertEqual(self.editor.translated_text.get("1.0", "end-1c"), "Unsaved translation")
        self.assertEqual([call[0] for call in self.calls], ["open", "preview_region"])
        self.assertIsNone(self.calls[-1][2])
        self.assertNotEqual(self.calls[-1][1], threading.get_ident())
        self.editor._close_region_preview()
        self.assertIsNone(self.editor.comparison_window)
        self.assertEqual(self.editor.comparison_photos, [])

    def test_local_comparison_uses_ocr_crop_and_handles_text_only_stale_or_cancelled_results(self):
        from PIL import Image
        self.enable_ocr()
        self.editor._set_ocr_crop([5, 6, 55, 76])
        self.editor._preview_region()
        self.pump_until(lambda: not self.editor.busy)
        self.assertEqual(self.calls[-1][2], [5, 6, 55, 76])
        self.editor._close_region_preview()
        payload = {"block_id": "b-one", "crop_box": [5, 6, 55, 76],
                   "original": Image.new("RGB", (768, 768)), "translated": None}
        self.assertTrue(self.editor._show_region_preview(payload))
        self.assertEqual(len(self.editor.comparison_photos), 1)
        self.assertIn("No exported translation", self.editor.comparison_captions[1])
        self.root.deiconify()
        self.pump_until(lambda: self.editor.comparison_window.winfo_ismapped())
        self.root.update()
        self.assertLessEqual(self.editor.comparison_window.winfo_width(), self.root.winfo_screenwidth())
        self.assertLessEqual(self.editor.comparison_window.winfo_height(), self.root.winfo_screenheight())
        self.editor._close_region_preview()
        self.assertFalse(self.editor._show_region_preview(dict(payload, block_id="b-two")))
        self.editor.cancel_event.set()
        self.assertFalse(self.editor._show_region_preview(payload))
        self.assertIsNone(self.editor.comparison_window)

    def test_candidate_controls_fit_default_editor_size_with_status_text(self):
        self.root.deiconify()
        self.pump_until(lambda: self.editor.window.winfo_ismapped())
        panel = self.editor.ocr_apply_button.master
        panel.master.select(panel)
        self.editor.ocr_candidate_status.set("候选已经过期：原文或译文已经修改，请重新预览，避免覆盖人工校对。")
        self.root.update()
        window_bottom = self.editor.window.winfo_rooty() + self.editor.window.winfo_height()
        for widget in (self.editor.saved_ocr_box, self.editor.ocr_apply_button,
                       self.editor.export_button, self.editor.compare_button):
            self.assertTrue(widget.winfo_ismapped())
            self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), window_bottom)


if __name__ == "__main__":
    unittest.main()
