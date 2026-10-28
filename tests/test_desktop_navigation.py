"""Desktop mode handoffs: one session at a time, without models or network.

Run: .venv\\Scripts\\python.exe -m unittest tests.test_desktop_navigation -v
"""

import copy
import gc
import re
import sys
import tempfile
import time
import types
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from ui import app as gui


def non_english_button_labels(widget):
    labels = []
    if widget.winfo_class() in {"TButton", "Button", "TCheckbutton", "Checkbutton", "TMenubutton"}:
        text = str(widget.cget("text"))
        if re.search(r"[\u3400-\u9fff]", text):
            labels.append(text)
    for child in widget.winfo_children():
        labels.extend(non_english_button_labels(child))
    return labels


class DesktopNavigationTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self._cleaned = False
        self.addCleanup(self._cleanup)
        try:
            self.root = gui.tk.Tk()
        except gui.tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.root.withdraw()
        self.app = gui.TranslatorApp(self.root)
        self.temp = tempfile.TemporaryDirectory()
        self.page = str(Path(self.temp.name) / "daily-page.json")
        self.other_page = str(Path(self.temp.name) / "next-page.json")
        self.reader = Mock()
        self.reader.pages = [
            {"path": self.page, "input": self.page.replace(".json", ".jpg")},
            {"path": self.other_page, "input": self.other_page.replace(".json", ".jpg")},
        ]
        self.reader.page_index = 0
        self.reader.config = self.app._config()
        self.reader.work = gui.tk.StringVar(master=self.root, value=self.reader.config.work)
        self.reader.output_dir = gui.tk.StringVar(master=self.root, value=self.app.output_dir.get())
        self.reader_factory = Mock(return_value=self.reader)
        self.editor = Mock()
        self.editor.request_close.return_value = True
        self.editor_factory = Mock(return_value=self.editor)
        self.term_factory = Mock()
        self.reader_patch = patch.dict(sys.modules, {"reader": types.SimpleNamespace(ReaderWindow=self.reader_factory)})
        self.editor_patch = patch.object(gui, "PageEditor", self.editor_factory)
        self.term_patch = patch.object(gui, "TermEditor", self.term_factory)
        self.reader_patch.start()
        self.editor_patch.start()
        self.term_patch.start()

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
            for name in ("term_patch", "editor_patch", "reader_patch"):
                patcher = getattr(self, name, None)
                if patcher is not None:
                    patcher.stop()
                    setattr(self, name, None)
            patcher = None
            if hasattr(self, "temp"):
                self.temp.cleanup()
            for name in ("app", "root", "reader", "editor", "reader_factory", "editor_factory", "term_factory"):
                setattr(self, name, None)
            gc.collect()

    def reader_closed(self):
        """Deliver the callback only after the fake worker has released its session."""
        self.reader_factory.call_args.kwargs["on_closed"]()

    def editor_closed(self):
        self.editor_factory.call_args.kwargs["on_closed"]()

    def test_launch_starts_reading_with_demo_and_exports_new_pages_by_default(self):
        demo = types.SimpleNamespace(ensure_demo_pages=Mock(return_value=[Path(self.page)]))
        with patch.dict(sys.modules, {"presentation": demo}):
            self.app.start_reading()
        demo.ensure_demo_pages.assert_called_once()
        self.reader_factory.assert_called_once()
        self.assertEqual([str(path) for path in self.reader_factory.call_args.kwargs["paths"]], [self.page])
        self.editor_factory.assert_not_called()
        self.assertEqual(self.root.state(), "withdrawn")
        self.assertTrue(self.app._config().export_images)

    def test_processing_controls_have_english_labels(self):
        self.assertEqual(non_english_button_labels(self.root), [])

    def test_reader_to_workbench_waits_for_session_release(self):
        self.app._open_reader([self.page])
        self.app._switch_from_reader("workbench", self.page)
        self.reader.close.assert_called_once()
        self.editor_factory.assert_not_called()
        self.assertIs(self.app.reader_window, self.reader)
        self.assertIn("disabled", self.app.start_button.state())
        self.reader_closed()
        self.assertIsNone(self.app.reader_window)
        self.editor_factory.assert_called_once()
        self.assertEqual(self.editor_factory.call_args.args[1], self.page)
        self.assertEqual(self.root.state(), "withdrawn")

    def test_tasks_stay_hidden_until_reader_closes(self):
        self.app._open_reader([self.page])
        self.app._switch_from_reader("tasks")
        self.assertEqual(self.root.state(), "withdrawn")
        self.reader_closed()
        self.assertEqual(self.root.state(), "normal")
        self.assertIsNone(self.app.reader_window)
        self.assertNotIn("disabled", self.app.start_button.state())
        self.editor_factory.assert_not_called()

    def custom_reader_config(self):
        folder = Path(self.temp.name)
        return replace(self.reader.config,
            work="stored-work", device="cuda:1", memory_policy="resident",
            memory_db=str(folder / "private-memory.sqlite3"), translation_backend="qwen",
            model_path=str(folder / "custom-qwen"), gguf_path=str(folder / "custom.gguf"),
            llama_server_path=str(folder / "llama-server.exe"), llama_context_size=4096,
            model_name="custom-model", adapter_path=str(folder / "custom-adapter"),
            detection_conf=.77, detect_free_text=False, translation_mode="grouped",
            group_size=6, context_chars=2048, use_continuations=False,
            use_translation_history=False, use_speaker_context=False, max_new_tokens=384,
            retries=0, extract_terms=False, local_files_only=True,
            font_path=str(folder / "font.ttf"), min_font_size=15, max_font_size=35,
            skip_inpaint=True, export_images=False, debug_mask=True,
            ocr_candidate_review=True, use_hybrid_ocr=True, ocr_prob_threshold=.25,
            use_fragment_links=False)

    def test_tasks_preserve_reader_scope_output_and_unexposed_model_options(self):
        config = self.reader.config = self.custom_reader_config()
        saved_config = asdict(config)
        expected = replace(config, work="Cafe Notes", export_images=True)
        output_dir = str(Path(self.temp.name) / "custom-output")
        self.reader.work.set("Cafe Notes")
        self.reader.output_dir.set(output_dir)
        self.reader.page_index = 1
        self.app._open_reader([self.page, self.other_page], config=config)
        self.app._switch_from_reader("tasks")
        self.reader_closed()
        self.assertEqual(self.app.output_dir.get(), output_dir)
        self.assertEqual(self.app.input_path.get(), self.reader.pages[1]["input"])
        self.assertEqual(self.app.input_paths, [page["input"] for page in self.reader.pages])
        self.assertEqual(asdict(self.app._config()), asdict(expected))
        self.assertEqual(asdict(config), saved_config)

    def test_handoff_preserves_tasks_text_only_choice_instead_of_reader_override(self):
        self.app.text_only.set(True)
        self.reader.config = replace(self.custom_reader_config(), export_images=True)
        self.app._open_reader([self.page], config=self.reader.config)
        self.app._switch_from_reader("tasks")
        self.reader_closed()
        self.assertTrue(self.app.text_only.get())
        self.assertFalse(self.app._config().export_images)

    def test_empty_reader_output_does_not_erase_task_output(self):
        output_dir = str(Path(self.temp.name) / "task-output")
        self.app.output_dir.set(output_dir)
        self.reader.output_dir.set("")
        self.app._open_reader([self.page])
        self.app._switch_from_reader("tasks")
        self.reader_closed()
        self.assertEqual(self.app.output_dir.get(), output_dir)

    def test_resume_keeps_reader_output_scope_and_current_page(self):
        config = self.reader.config = self.custom_reader_config()
        output_dir = str(Path(self.temp.name) / "reader-output")
        self.reader.output_dir.set(output_dir)
        self.reader.work.set(config.work)
        self.reader.page_index = 1
        self.app._open_reader([self.page, self.other_page], config=config)
        self.app._switch_from_reader("tasks")
        self.reader_closed()
        self.reader_factory.reset_mock()
        self.app._resume_reading()
        self.reader_factory.assert_called_once()
        arguments = self.reader_factory.call_args.kwargs
        self.assertEqual(arguments["output_dir"], output_dir)
        self.assertEqual(arguments["paths"], [self.page, self.other_page])
        self.assertEqual(arguments["initial_path"], self.other_page)
        self.assertEqual(asdict(arguments["config"]), asdict(config))

    def test_terms_wait_for_reader_release_before_opening_database_editor(self):
        self.app._open_reader([self.page])
        self.app._switch_from_reader("terms")
        self.term_factory.assert_not_called()
        self.reader.close.assert_called_once()
        self.reader_closed()
        self.term_factory.assert_called_once()
        self.assertIs(self.term_factory.call_args.args[1], self.reader.config)
        self.assertIsNone(self.app.reader_window)

    def test_untranslated_page_workbench_opens_processing_without_record_dialog(self):
        self.app._open_reader([self.page])
        with patch.object(gui.filedialog, "askopenfilename") as chooser:
            self.app._switch_from_reader("workbench", None)
            self.reader_closed()
        chooser.assert_not_called()
        self.editor_factory.assert_not_called()
        self.assertEqual(self.root.state(), "normal")

    def test_declined_unsaved_draft_keeps_workbench_and_does_not_open_reader(self):
        self.app._open_page_editor(self.page)
        self.editor.request_close.return_value = False
        self.app._switch_from_editor("reading")
        self.editor.request_close.assert_called_once()
        self.reader_factory.assert_not_called()
        self.assertIs(self.app.page_editor, self.editor)
        self.assertEqual(self.root.state(), "withdrawn")

    def test_workbench_to_reading_waits_then_reloads_same_page_set(self):
        self.app.reader_paths = [self.page, self.other_page]
        self.app._open_page_editor(self.page)
        self.app._switch_from_editor("reading")
        self.reader_factory.assert_not_called()
        self.assertIs(self.app.page_editor, self.editor)
        self.editor_closed()
        self.assertIsNone(self.app.page_editor)
        self.reader_factory.assert_called_once()
        self.assertEqual(self.reader_factory.call_args.kwargs["paths"], [self.page, self.other_page])
        self.assertEqual(self.root.state(), "withdrawn")

    def test_workbench_to_tasks_waits_for_editor_close_callback(self):
        self.app._open_page_editor(self.page)
        self.app._switch_from_editor("tasks")
        self.assertEqual(self.root.state(), "withdrawn")
        self.editor_closed()
        self.assertEqual(self.root.state(), "normal")
        self.assertNotIn("disabled", self.app.start_button.state())
        self.reader_factory.assert_not_called()

    def test_opening_another_record_from_tasks_returns_to_that_record(self):
        self.app.reader_paths = [self.other_page]
        self.app._open_page_editor(self.page)
        self.app._switch_from_editor("reading")
        self.editor_closed()
        self.assertIn(self.page, self.reader_factory.call_args.kwargs["paths"])
        self.assertEqual(self.reader_factory.call_args.kwargs["initial_path"], self.page)

    def test_repeated_mode_click_does_not_spawn_duplicate_sessions(self):
        self.app._open_reader([self.page])
        self.app._switch_from_reader("workbench", self.page)
        self.app._switch_from_reader("tasks")
        self.reader.close.assert_called_once()
        self.reader_closed()
        self.editor_factory.assert_called_once()
        self.assertEqual(self.root.state(), "withdrawn")

    def test_application_close_during_reader_handoff_cancels_destination(self):
        self.app._open_reader([self.page])
        self.app._switch_from_reader("workbench", self.page)
        with patch.object(self.app, "_destroy") as destroy:
            self.app._close()
            destroy.assert_not_called()
            self.reader_closed()
            destroy.assert_called_once()
        self.editor_factory.assert_not_called()

    def test_application_close_during_editor_handoff_cancels_destination(self):
        self.app._open_page_editor(self.page)
        self.app._switch_from_editor("reading")
        with patch.object(self.app, "_destroy") as destroy:
            self.app._close()
            destroy.assert_not_called()
            self.editor_closed()
            destroy.assert_called_once()
        self.reader_factory.assert_not_called()


    def test_reader_cleanup_handoff_waits_and_retains_current_page(self):
        cleanup = Mock()
        factory = Mock(return_value=cleanup)
        with patch.dict(sys.modules, {"ui.tools": types.SimpleNamespace(
                CleanupWindow=factory, RevisionManagerWindow=Mock())}):
            self.app._open_reader([self.page, self.other_page])
            self.reader_factory.call_args.kwargs["on_tools"]("cleanup", self.page)
            factory.assert_not_called()
            self.reader_closed()
        self.assertIs(self.app.tool_window, cleanup)
        self.assertEqual(factory.call_args.args[1], self.page)
        self.assertEqual(self.app.reader_resume_path, self.page)
        self.assertIn("disabled", self.app.start_button.state())

    def test_workbench_tool_navigation_preserves_unsaved_draft_confirmation(self):
        self.app._open_page_editor(self.page)
        self.editor.result_path = self.page
        self.editor.request_close.return_value = False
        self.editor_factory.call_args.kwargs["on_tools"]("cleanup")
        self.assertIsNone(self.app.pending_view)
        self.editor.request_close.return_value = True
        self.editor_factory.call_args.kwargs["on_tools"]("cleanup")
        self.assertEqual(self.app.pending_view, ("cleanup", self.page))
        self.assertIsNone(self.app.tool_window)

    def test_cleanup_reading_handoff_waits_for_tool_session_release(self):
        cleanup = Mock()
        cleanup.request_close.return_value = True
        factory = Mock(return_value=cleanup)
        with patch.dict(sys.modules, {"ui.tools": types.SimpleNamespace(
                CleanupWindow=factory, RevisionManagerWindow=Mock())}):
            self.app._open_tool("cleanup", self.other_page)
        factory.call_args.kwargs["on_read"]()
        self.reader_factory.assert_not_called()
        cleanup.request_close.assert_called_once()
        factory.call_args.kwargs["on_closed"]()
        self.assertIsNone(self.app.tool_window)
        self.assertEqual(self.reader_factory.call_args.kwargs["initial_path"], self.other_page)

    def test_revision_manager_opens_record_only_after_scan_closes(self):
        manager = Mock()
        manager.request_close.return_value = True
        factory = Mock(return_value=manager)
        with patch.dict(sys.modules, {"ui.tools": types.SimpleNamespace(
                CleanupWindow=Mock(), RevisionManagerWindow=factory)}):
            self.app._open_tool("revisions", self.page)
        self.assertEqual(Path(factory.call_args.args[1]), Path(self.page).parent)
        factory.call_args.kwargs["on_edit"](self.other_page)
        self.editor_factory.assert_not_called()
        factory.call_args.kwargs["on_closed"]()
        self.assertEqual(self.editor_factory.call_args.args[1], self.other_page)

    def test_tool_blocks_batch_and_duplicate_navigation(self):
        cleanup = Mock()
        cleanup.request_close.return_value = True
        self.app.tool_window = cleanup
        self.app._start()
        self.assertIsNone(self.app.worker)
        self.app._open_reader([self.page])
        self.reader_factory.assert_not_called()
        self.app._switch_from_tool("reading")
        self.app._switch_from_tool("workbench", self.page)
        cleanup.request_close.assert_called_once()
        self.assertEqual(self.app.pending_view, ("reading", None))

    def test_application_close_during_tool_handoff_cancels_destination(self):
        cleanup = Mock()
        cleanup.request_close.return_value = True
        self.app.tool_window = cleanup
        self.app._switch_from_tool("reading")
        with patch.object(self.app, "_destroy") as destroy:
            self.app._close()
            destroy.assert_not_called()
            self.app._tool_closed()
            destroy.assert_called_once()
        self.reader_factory.assert_not_called()


class WorkbenchLinkedSentenceTests(unittest.TestCase):
    """The workspace shows saved linked text while retaining unsubmitted drafts."""

    def setUp(self):
        from PIL import Image
        from tests.test_gui import fake_editor_session

        gc.collect()
        self._cleaned = False
        self.addCleanup(self._cleanup)
        try:
            self.root = gui.tk.Tk()
        except gui.tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.root.withdraw()
        self.temp = tempfile.TemporaryDirectory()
        self.image = Path(self.temp.name) / "linked-page.png"
        Image.new("RGB", (1200, 1000), "white").save(self.image)
        self.calls = []
        base_session = fake_editor_session(self.image, self.calls)

        class LinkedSession(base_session):
            def __init__(self, result_path):
                super().__init__(result_path)
                unit = {"members": ["b-one", "b-two"], "sources": ["原文一", "原文二"],
                        "translated_text": "Stale saved sentence."}
                for block in self.record["blocks"]:
                    block.setdefault("metadata", {})["translation_unit"] = copy.deepcopy(unit)

        self.session_patch = patch.dict(sys.modules, {"pipeline.editing": types.SimpleNamespace(PageEditorSession=LinkedSession)})
        self.session_patch.start()
        self.closed = False
        self.editor = gui.PageEditor(self.root, str(self.image), on_closed=self.mark_closed)
        self.pump(lambda: not self.editor.busy)

    def mark_closed(self):
        self.closed = True

    def pump(self, condition, timeout=4):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertTrue(condition(), "desktop worker condition timed out")

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
                        self.pump(lambda: self.closed)
                finally:
                    if self.editor.worker is not None:
                        self.editor.worker.join(timeout=5)
                        self.assertFalse(self.editor.worker.is_alive())
        finally:
            try:
                if hasattr(self, "root"):
                    self.root.destroy()
            finally:
                if hasattr(self, "session_patch"):
                    self.session_patch.stop()
                if hasattr(self, "temp"):
                    self.temp.cleanup()
                self.editor = self.root = self.session_patch = None
                gc.collect()

    @staticmethod
    def set_text(widget, text):
        widget.delete("1.0", "end")
        widget.insert("1.0", text)

    def select(self, block_id):
        self.editor.table.selection_set(block_id)
        self.editor._select()

    def test_workbench_opens_from_hidden_root(self):
        self.pump(lambda: bool(self.editor.window.winfo_viewable()))
        self.assertEqual(str(self.editor.window.transient()), "")
        self.assertEqual(self.editor.window.state(), "normal")

    def test_first_preview_fits_mapped_canvas_without_becoming_a_thumbnail(self):
        preview = self.editor.snapshot["preview"]
        self.assertGreater(preview.width, 200)
        self.assertGreater(preview.height, 200)
        self.assertLessEqual(preview.width, self.editor.canvas.winfo_width())
        self.assertLessEqual(preview.height, self.editor.canvas.winfo_height())

    def test_close_before_window_mapping_does_not_start_a_session(self):
        closed = []
        opened_before = sum(call[0] == "open" for call in self.calls)
        editor = gui.PageEditor(self.root, str(self.image), on_closed=lambda: closed.append(True))
        self.assertTrue(editor.request_close())
        self.pump(lambda: bool(closed))
        self.assertIsNone(editor.worker)
        self.assertEqual(sum(call[0] == "open" for call in self.calls), opened_before)

    def test_workbench_controls_have_english_labels(self):
        self.assertEqual(non_english_button_labels(self.editor.window), [])

    def test_english_draft_is_not_misrepresented_as_saved_linked_sentence(self):
        self.assertEqual(self.editor.linked_sentence.get(), "First Second")
        self.set_text(self.editor.translated_text, "Updated")
        self.select("b-two")
        self.assertEqual(self.editor.linked_sentence.get(), "First Second")
        self.select("b-one")
        self.assertEqual(self.editor.translated_text.get("1.0", "end-1c"), "Updated")
        self.editor._save("rerender")
        self.pump(lambda: not self.editor.busy)
        self.assertEqual(self.editor.linked_sentence.get(), "Updated Second")
        self.assertEqual([call[0] for call in self.calls], ["open", "rerender"])

    def test_changed_japanese_invalidates_old_link_for_both_members(self):
        self.set_text(self.editor.source_text, "訂正済み")
        self.editor._save("retranslate")
        self.pump(lambda: not self.editor.busy)
        self.assertEqual(self.editor.linked_sentence.get(), "No linked sentence for this region.")
        self.select("b-two")
        self.assertEqual(self.editor.linked_sentence.get(), "No linked sentence for this region.")
        self.assertEqual([call[0] for call in self.calls], ["open", "retranslate"])


if __name__ == "__main__":
    unittest.main()
