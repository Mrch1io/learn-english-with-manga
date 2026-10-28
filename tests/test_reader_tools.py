r"""Reader tool navigation and English chrome; real Tk, no models or network.

Run: .venv\Scripts\python.exe -m unittest tests.test_reader_tools -v
"""
from copy import deepcopy
from unittest import TestCase
from unittest.mock import Mock, patch

import reader
from reader.roles import _character_name
from tests import test_reader as reader_tests


class ReaderToolsTests(TestCase):
    def setUp(self):
        self.fixture = reader_tests.ReaderTests()
        self.addCleanup(self.fixture.doCleanups)
        self.on_tools, self.on_tasks, self.on_terms = Mock(), Mock(), Mock()
        constructor = reader.ReaderWindow

        def with_navigation(*args, **kwargs):
            return constructor(*args, on_tools=self.on_tools, on_tasks=self.on_tasks,
                               on_terms=self.on_terms, **kwargs)

        with patch.object(reader, "ReaderWindow", side_effect=with_navigation):
            self.fixture.setUp()
        self.window = self.fixture.window

    def tearDown(self):
        self.fixture.tearDown()

    def test_toolbar_is_compact_and_existing_tools_remain_in_menu(self):
        labels = [str(widget.cget("text")) for widget in self.window.toolbar.winfo_children()
                  if widget.winfo_class() in ("TButton", "TMenubutton")]
        self.assertEqual(labels, ["Import", "Tools", "Show Translation", "Settings"])
        menu = self.window.tools_menu
        entries = [menu.entrycget(index, "label") for index in range(menu.index("end") + 1)
                   if menu.type(index) == "command"]
        self.assertEqual(entries, ["Tasks", "Terms", "Processing", "Settings",
                                  "Translate Current Page", "Export Saved Translation",
                                  "Image Cleanup", "Revision Manager"])
        self.assertEqual(self.window.tools.winfo_manager(), "")

    def test_tool_navigation_passes_page_scope_without_closing_or_loading_models(self):
        path = self.window.current_result_path()
        with patch.object(self.window, "close") as close, \
                patch("core.runtime.ModelPool", side_effect=AssertionError("navigation loaded a model")):
            self.window.tools_menu.invoke("Image Cleanup")
            self.on_tools.assert_called_once_with("cleanup", path)
            self.window.tools_menu.invoke("Revision Manager")
            self.on_tools.assert_called_with("revisions", path)
        close.assert_not_called()
        self.assertFalse(self.window.closing.is_set())

    def test_tool_navigation_defers_during_page_or_model_work_and_shutdown(self):
        for busy in ("loading", "magi_busy", "translation_busy"):
            with self.subTest(busy=busy):
                setattr(self.window, busy, True)
                self.window._open_tool("cleanup")
                self.on_tools.assert_not_called()
                setattr(self.window, busy, False)
        self.window.closing.set()
        try:
            self.window._open_tool("revisions")
            self.on_tools.assert_not_called()
        finally:
            self.window.closing.clear()
        self.window._open_tool("cleanup")
        self.on_tools.assert_called_once_with("cleanup", self.window.current_result_path())

    def test_raw_page_navigation_preserves_absent_result(self):
        self.window.pages[self.window.page_index]["path"] = None
        self.window._open_tool("cleanup")
        self.on_tools.assert_called_once_with("cleanup", None)

    def test_tasks_terms_and_settings_keep_their_existing_destinations(self):
        self.window.tools_menu.invoke("Tasks")
        self.on_tasks.assert_called_once_with()
        self.window.tools_menu.invoke("Terms")
        self.on_terms.assert_called_once_with()
        self.window.tools_menu.invoke("Processing")
        self.assertEqual(self.window.tools.tab(self.window.tools.select(), "text"), "Processing")
        self.assertEqual(self.window.tools.winfo_manager(), "grid")
        self.window.tools_menu.invoke("Settings")
        self.assertEqual(self.window.settings_window.state(), "normal")
        self.window._focus_reading()
        self.assertEqual(self.window.tools.winfo_manager(), "")
        self.assertEqual(self.window.settings_window.state(), "withdrawn")

    def test_translate_and_export_use_existing_worker_queue(self):
        for label, action in (("Translate Current Page", "translate"),
                              ("Export Saved Translation", "export")):
            with self.subTest(action=action), patch.object(self.window.commands, "put") as enqueue, \
                    patch("core.runtime.ModelPool", side_effect=AssertionError("UI loaded a model")):
                self.window.tools_menu.invoke(label)
                enqueue.assert_called_once()
                self.assertEqual(enqueue.call_args.args[0][0], action)
                self.assertTrue(self.window.translation_busy)
                self.assertEqual(str(self.window.tools_button.cget("state")), "disabled")
                self.window._translation_finished()
                self.assertEqual(str(self.window.tools_button.cget("state")), "normal")

    def test_default_character_labels_are_english_without_changing_saved_names(self):
        before = deepcopy(self.window.characters)
        labels = self.window.role_box.cget("values")
        self.assertIn("Default voice", labels)
        self.assertIn("Narrator", labels)
        self.assertEqual(self.window.role_name.get(), "Default voice")
        self.assertEqual(self.window.characters, before)
        self.assertEqual(_character_name({"id": "unknown", "name": "Other name"}), "Other name")
        self.assertEqual(_character_name({"id": "custom", "name": "旁白"}), "旁白")

    def test_duplicate_character_names_remain_selectable_without_exposing_ids(self):
        self.window.characters += [
            {"id": "first-uuid", "name": "Meru", "voice": {}},
            {"id": "second-uuid", "name": "Meru", "voice": {}},
            {"id": "third-uuid", "name": "Meru (2)", "voice": {}},
        ]
        self.window._set_roles("second-uuid")
        labels = list(self.window.role_box.cget("values"))
        self.assertEqual(len(labels), len(set(labels)))
        self.assertIn("Meru (2)", labels)
        self.assertNotIn("uuid", " ".join(labels))
        self.assertEqual(self.window.characters[self.window.role_box.current()]["id"], "second-uuid")
        with patch.object(self.window.commands, "put") as enqueue:
            self.window._bind_role()
        self.assertEqual(enqueue.call_args.args[0][1].character_id, "second-uuid")

    def test_reader_generated_voice_and_playback_status_is_english(self):
        self.assertIn("Temporary voice", self.window.voice_status.get())
        self.window.stop()
        self.assertIn("Stopped", self.window.status.get())
        self.window.autoplay.set(False)
        self.window._auto_changed()
        self.assertEqual(self.window.status.get(), "Play on bubble click disabled.")
        self.window.speed.set("3")
        with self.assertRaisesRegex(ValueError, "Choose a speed between"):
            self.window._settings()

    def test_character_management_is_collapsed_without_losing_actions(self):
        panel = self.window.character_details
        self.assertEqual(panel.winfo_manager(), "")
        self.window.character_details_button.invoke()
        self.assertEqual(panel.winfo_manager(), "grid")
        labels = [str(widget.cget("text")) for widget in panel.winfo_children()
                  if widget.winfo_class() == "TButton"]
        self.assertEqual(labels, ["Save Name & Default Voice", "Suggest Characters (Magi)"])
        self.window.role_name.set("Preview character")
        self.window.character_details_button.invoke()
        self.assertEqual(panel.winfo_manager(), "")
        self.assertEqual(self.window.role_name.get(), "Preview character")

    def test_close_releases_owned_traces_and_worker_before_callback(self):
        variables = (self.window.speed, self.window.url, self.window.work, self.window.view)
        self.assertTrue(all(value.trace_info() for value in variables))
        callback = self.window.on_closed
        observed = []

        def closed():
            observed.append(self.window.worker.is_alive())
            callback()

        self.window.on_closed = closed
        self.window.close()
        self.fixture.pump(lambda: self.fixture.closed)
        self.assertEqual(observed, [False])
        self.assertTrue(all(not value.trace_info() for value in variables))
        self.assertIsNone(self.window.on_closed)
