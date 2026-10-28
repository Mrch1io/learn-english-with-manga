r"""Reader presentation regressions, with real Tk and no model or network work.

Run: .venv\Scripts\python.exe -m unittest tests.test_reader_ui -v
"""
from unittest import TestCase
from unittest.mock import Mock, patch
from pathlib import Path
import queue
import tempfile
import threading
import types

from PIL import Image
import reader
from reader.messages import PreviewRequest
from reader.worker import ReaderWorker

from tests import test_reader as reader_tests


class ReaderPresentationTests(TestCase):
    def setUp(self):
        self.fixture = reader_tests.ReaderTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.window = self.fixture.window

    def tearDown(self):
        self.fixture.tearDown()

    def test_reading_sidebar_has_only_character_voice_controls(self):
        self.assertEqual(self.window.japanese.winfo_manager(), "")
        self.assertEqual(self.window.english.winfo_manager(), "")
        self.assertEqual(self.window.tools.winfo_manager(), "")
        visible_labels = []
        buttons = []
        pending = [self.window.window]
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            if widget.winfo_class() == "TButton":
                buttons.append(str(widget.cget("text")))
            if widget.winfo_class() == "TLabel" and widget.winfo_manager():
                visible_labels.append(str(widget.cget("text")))
        self.assertTrue(all(text.isascii() for text in buttons), buttons)
        self.assertIn("Character & Voice", visible_labels)
        self.assertNotIn("English", visible_labels)
        self.assertNotIn("关联整句", visible_labels)

    def test_bubble_ids_are_hidden_while_page_selection_keeps_full_id(self):
        key = "presentation:everyday:t003"
        self.window.blocks[key] = {**self.window.blocks["b-one"], "block_id": key}
        self.window.bubble_list.insert("", "end", iid=key)
        self.window._refresh_rows()
        self.assertTrue(self.window.bubble_list.exists(key))
        self.assertEqual(self.window.bubble_list.winfo_manager(), "")
        self.assertFalse(self.window.bubble_list.winfo_viewable())
        self.window.autoplay.set(False)
        self.window.bubble_list.selection_set(key)
        self.window._select_bubble()
        self.assertEqual(self.window.block_id, key)
        self.assertEqual(self.window.selected_bubble.get(), "Dialogue selected")
        self.assertTrue(self.window.canvas.find_withtag("bubble"))

    def test_voice_diagnostics_stay_in_settings_without_expanding_reading_footer(self):
        self.fixture.wait_for_fit()
        height = self.window.canvas.winfo_height()
        error = "Port 50021 timed out.\nAt launcher.ps1:23 char:13\n" * 30
        self.window._show_voice_error("Voice unavailable.", error)
        self.fixture.root.update()
        self.assertEqual(self.window.status.get(), "Voice unavailable. See Settings for details.")
        self.assertIn(error, self.window.voice_status.get())
        self.assertEqual(self.window.canvas.winfo_height(), height)
        self.window.settings_button.invoke()
        self.fixture.root.update()
        self.assertGreater(self.window.voice_status_canvas.bbox("all")[3],
                           self.window.voice_status_canvas.winfo_height())

    def test_display_title_does_not_change_the_memory_namespace(self):
        self.window.work.set("presentation:tencho_isoro")
        self.window.record["presentation_fixture"] = {"title": "A quiet afternoon"}
        self.window._update_display_title()
        self.assertEqual(self.window.display_title.get(), "A quiet afternoon")
        self.assertEqual(self.window.work.get(), "presentation:tencho_isoro")
        self.assertEqual(self.window.config.work, "reader-test")
        self.window.record["presentation_fixture"] = None
        self.window._update_display_title()
        self.assertEqual(self.window.display_title.get(), "presentation:tencho_isoro")

    def test_toolbar_buttons_fit_minimum_width_with_terms_enabled(self):
        closed = []
        extra = reader.ReaderWindow(self.fixture.root, config=self.window.config,
                                    on_terms=Mock(), on_closed=lambda: closed.append(True))
        try:
            extra.record["presentation_fixture"] = {"title": "A quiet afternoon at the bookshop"}
            extra._update_display_title()
            extra.window.geometry("900x650")
            self.fixture.root.update()
            left = extra.toolbar.winfo_rootx()
            right = left + extra.toolbar.winfo_width()
            buttons = [widget for widget in extra.toolbar.winfo_children()
                       if widget.winfo_class() in ("TButton", "TMenubutton")]
            self.assertEqual(extra.import_menu.entrycget(0, "label"), "Images / JSON...")
            self.assertEqual(extra.import_menu.entrycget(1, "label"), "Folder...")
            for toggle_label in ("Show Original", "Show Translation"):
                extra.view_button.configure(text=toggle_label)
                self.fixture.root.update_idletasks()
                for button in buttons:
                    with self.subTest(view=toggle_label, button=button.cget("text")):
                        self.assertTrue(button.winfo_viewable())
                        self.assertGreaterEqual(button.winfo_rootx(), left)
                        self.assertLessEqual(button.winfo_rootx() + button.winfo_width(), right)
                        self.assertGreaterEqual(button.winfo_width(), button.winfo_reqwidth())
        finally:
            extra.close()
            self.fixture.pump(lambda: bool(closed))
            extra.worker.join(timeout=1)

    def test_open_prefers_verified_translation_and_toggle_never_reruns_models(self):
        path = self.window.current_result_path()
        self.fixture.exported_record(path)
        with patch("core.runtime.ModelPool", side_effect=AssertionError("reading loaded a model")):
            self.window.open_paths([path])
            self.fixture.pump(lambda: not self.window.loading and self.window.view.get() == "英文译图")
            self.assertEqual(self.fixture.displayed_pixel(), (20, 120, 200))
            self.assertEqual(self.window.view_button.cget("text"), "Show Original")
            self.window._toggle_view()
            self.fixture.pump(lambda: not self.window.loading and self.window.view.get() == "原图")
            self.assertEqual(self.fixture.displayed_pixel(), (255, 255, 255))
            self.assertEqual(self.window.view_button.cget("text"), "Show Translation")
            self.window._toggle_view()
            self.fixture.pump(lambda: not self.window.loading and self.window.view.get() == "英文译图")
            self.assertEqual(self.fixture.displayed_pixel(), (20, 120, 200))
        self.assertFalse(self.fixture.synthesized)

    def test_text_only_open_is_original_without_exporting(self):
        self.assertEqual(self.window.view.get(), "原图")
        self.assertIn("Text-only page", self.window.status.get())
        self.assertEqual(self.window.view_button.cget("text"), "Show Translation")
        self.assertFalse(self.window.record["exported"])
        with patch.object(self.window, "_export_page") as export:
            self.window._toggle_view()
        export.assert_not_called()
        self.assertEqual(self.window.view.get(), "原图")
        self.assertIn("No translated image yet", self.window.status.get())

    def test_workbench_callback_receives_current_path_and_no_path_for_raw_pages(self):
        callback = self.window.on_workbench = Mock()
        self.window._open_workbench()
        callback.assert_called_once_with(self.window.current_result_path())
        self.window.pages[self.window.page_index]["path"] = None
        self.window._open_workbench()
        callback.assert_called_with(None)

    def test_terms_navigation_waits_for_stable_page_scope(self):
        callback = self.window.on_terms = Mock()
        for busy in ("loading", "translation_busy", "magi_busy"):
            with self.subTest(busy=busy):
                setattr(self.window, busy, True)
                self.window._open_terms()
                callback.assert_not_called()
                setattr(self.window, busy, False)
        self.window.closing.set()
        try:
            self.window._open_terms()
            callback.assert_not_called()
        finally:
            self.window.closing.clear()
        self.window._open_terms()
        callback.assert_called_once_with()

    def test_worker_pool_cleanup_failure_is_reported_before_window_closes(self):
        pool = Mock()
        pool.close.side_effect = RuntimeError("sample cleanup failure")
        with patch("core.runtime.ModelPool", return_value=pool), \
             patch("pipeline.editing.PageEditorSession", self.fixture.exporter([])), \
             patch("reader.events.messagebox.showerror") as show_error:
            self.window._export_page()
            self.fixture.pump(lambda: not self.window.translation_busy and not self.window.loading)
            self.window.close()
            self.fixture.pump(lambda: self.fixture.closed)
            self.window.worker.join(timeout=1)
        pool.close.assert_called_once_with()
        show_error.assert_called_once()
        self.assertEqual(show_error.call_args.args[0], "Resource cleanup failed")
        self.assertIn("sample cleanup failure", show_error.call_args.args[1])
        self.assertFalse(self.window.worker.is_alive())

    def test_review_state_is_preserved_without_page_badge(self):
        self.window.pages[0]["state"] = "needs_review"
        self.window._refresh_page_rows()
        self.assertEqual(self.window.page_list.get(0), self.window.pages[0]["label"])
        self.assertEqual(self.window.pages[0]["state"], "needs_review")

    def test_processing_remains_available_and_rehides_for_reading(self):
        self.window._show_tools("Processing")
        self.assertEqual(self.window.tools.winfo_manager(), "grid")
        self.assertEqual(self.window.tools.tab(self.window.tools.select(), "text"), "Processing")
        self.assertEqual(self.window.translate_button.cget("text"), "Translate Current Page")
        self.window._focus_reading()
        self.assertEqual(self.window.tools.winfo_manager(), "")

    def test_initial_page_is_consumed_once_when_pages_open(self):
        desired = self.window.pages[1]["path"]
        self.window.initial_path = desired
        self.window.open_paths(self.fixture.folder)
        self.fixture.pump(lambda: self.window.current_result_path() == desired and not self.window.loading)
        self.assertIsNone(self.window.initial_path)
        self.assertEqual(self.window.page_position.get(), "2 / 2")
        self.window.open_paths(self.fixture.folder)
        self.fixture.pump(lambda: self.window.page_index == 0 and not self.window.loading)

    def test_short_window_keeps_voice_controls_reachable_by_sidebar_scroll(self):
        self.window.window.geometry("900x650")
        self.fixture.root.update()
        self.assertTrue(self.window.stop_button.winfo_viewable())
        self.window.side_canvas.yview_moveto(1)
        self.assertAlmostEqual(self.window.side_canvas.yview()[1], 1)
        self.assertLessEqual(self.window.sidebar.winfo_rootx() + self.window.sidebar.winfo_width(),
                             self.window.window.winfo_rootx() + self.window.window.winfo_width())

    def test_collapse_expands_canvas_without_reloading_page_or_interrupting_audio(self):
        self.fixture.root.update()
        # Finish initial sidebar image work before asserting that a resize
        # itself never reopens image files.
        self.fixture.pump(lambda: self.window.thumbnail_pending is None and all(
            index in self.window.thumbnail_cache or index in self.window.thumbnail_errors
            for index in self.window.thumbnail_cards))
        width = self.window.canvas.winfo_width()
        before = (self.window.block_id, self.window.role.get(), self.window.voice.get(), self.window.token)
        worker, pixels = self.window.worker, self.window.original
        with patch.object(self.window, "_invalidate", side_effect=AssertionError("collapse stopped audio")), \
                patch("PIL.Image.open", side_effect=AssertionError("collapse reopened a file")), \
                patch("core.runtime.ModelPool", side_effect=AssertionError("collapse loaded a model")):
            self.window.collapse_sidebar_button.invoke()
            self.fixture.root.update_idletasks()
            self.assertEqual(self.window.sidebar.winfo_manager(), "")
            self.assertEqual(self.window.sidebar_rail.winfo_manager(), "grid")
            self.assertGreater(self.window.canvas.winfo_width(), width + 180)
            self.window._resize_to_canvas()
            self.fixture.pump(lambda: not self.window.resizing)
            self.window.expand_sidebar_button.invoke()
            self.fixture.root.update_idletasks()
            self.assertEqual(self.window.sidebar.winfo_manager(), "grid")
            self.assertEqual(self.window.sidebar_rail.winfo_manager(), "")
            self.assertEqual(self.window.canvas.winfo_width(), width)
        self.assertEqual((self.window.block_id, self.window.role.get(), self.window.voice.get(), self.window.token), before)
        self.assertIs(self.window.worker, worker)
        self.assertIs(self.window.original, pixels)

    def test_settings_remain_accessible_when_collapsed_and_keep_unsaved_values(self):
        for widget in (self.window.autoplay_button, self.window.preview_voice_button,
                       self.window.character_details_button, self.window.connect_button,
                       self.window.refresh_button):
            self.assertTrue(str(widget).startswith(str(self.window.settings_window) + "."))
            self.assertFalse(widget.winfo_viewable())
        self.window._toggle_sidebar()
        self.window.settings_button.invoke()
        self.fixture.root.update()
        self.assertTrue(self.window.autoplay_button.winfo_viewable())
        self.window.autoplay_button.invoke()
        self.window.character_details_button.invoke()
        self.window.role_name.set("Unsaved character")
        self.window._hide_settings()
        self.window.settings_button.invoke()
        self.fixture.root.update()
        self.assertTrue(self.window.sidebar_collapsed)
        self.assertFalse(self.window.autoplay.get())
        self.assertEqual(self.window.role_name.get(), "Unsaved character")
        self.assertEqual(self.window.character_details.winfo_manager(), "grid")
        self.window._focus_reading()
        self.assertEqual(self.window.settings_window.state(), "withdrawn")

    def test_main_close_also_destroys_open_settings(self):
        self.window.settings_button.invoke()
        self.fixture.root.update()
        self.window.close()
        self.fixture.pump(lambda: self.fixture.closed)
        self.assertFalse(self.window.settings_window.winfo_exists())

    def test_settings_controls_fit_minimum_window_with_character_management_open(self):
        self.window.settings_button.invoke()
        self.window.character_details_button.invoke()
        settings = self.window.settings_window
        settings.geometry("640x420")
        self.fixture.root.update()
        left, top = settings.winfo_rootx(), settings.winfo_rooty()
        right, bottom = left + settings.winfo_width(), top + settings.winfo_height()
        pending = list(settings.winfo_children())
        while pending:
            widget = pending.pop()
            children = widget.winfo_children()
            pending.extend(children)
            if children or not widget.winfo_viewable():
                continue
            with self.subTest(widget=widget.winfo_class(), path=str(widget)):
                self.assertGreaterEqual(widget.winfo_rootx(), left)
                self.assertGreaterEqual(widget.winfo_rooty(), top)
                self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(), right)
                self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), bottom)
                self.assertGreaterEqual(widget.winfo_height(), widget.winfo_reqheight())

    def test_long_voice_connection_errors_scroll_inside_settings(self):
        self.window.settings_button.invoke()
        self.window.voice_status.set("VOICEVOX connection failed. " * 80)
        self.fixture.root.update()
        canvas = self.window.voice_status_canvas
        self.assertGreater(canvas.bbox("all")[3], canvas.winfo_height())
        self.assertLessEqual(canvas.bbox("all")[2], canvas.winfo_width())
        canvas.yview_moveto(1)
        self.assertAlmostEqual(canvas.yview()[1], 1)

    def test_collapsed_sidebar_keeps_magi_cancellation_reachable(self):
        self.window._toggle_sidebar()
        with patch.object(self.window.commands, "put"):
            self.window._suggest_characters()
            self.assertTrue(self.window.magi_busy)
            self.assertEqual(str(self.window.cancel_button.cget("state")), "normal")
            self.assertTrue(self.window.cancel_button.winfo_viewable())
            self.window.cancel_button.invoke()
            self.assertTrue(self.window.operation_cancel.is_set())
            self.window._set_magi_busy(False)
            self.assertEqual(str(self.window.cancel_button.cget("state")), "disabled")


class ReaderThumbnailTests(TestCase):
    def test_initial_one_pixel_canvas_does_not_destroy_thumbnail_detail(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "portrait.png"
            image = Image.new("RGB", (100, 200), "red")
            image.paste("blue", (50, 0, 100, 200))
            image.save(source)
            commands, events = queue.Queue(), queue.Queue()
            commands.put(("page", PreviewRequest(None, str(source), 0, (1, 1)), 1))
            commands.put(("close", None, 2))
            speech = types.SimpleNamespace(VoicevoxClient=Mock(), synthesize_cached=Mock())
            with patch.dict("sys.modules", {"audio.speech": speech}):
                ReaderWorker(commands, events, threading.Event(), threading.Event()).run()
            action, result, token = events.get_nowait()
            self.assertEqual((action, token), ("page", 1))
            self.assertEqual(result.preview.size, (1, 1))
            thumbnail = result.thumbnail
            self.assertEqual(thumbnail.size, (92, 184))
            self.assertEqual(thumbnail.getpixel((10, 40)), (255, 0, 0))
            self.assertEqual(thumbnail.getpixel((80, 40)), (0, 0, 255))
            self.assertEqual(events.get_nowait(), ("closed", None, 2))

