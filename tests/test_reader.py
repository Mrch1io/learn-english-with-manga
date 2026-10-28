"""Reader UI and stale-audio tests; no GPU, server or audible playback."""
import json
import gc
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import reader
from reader.messages import CatalogResult, ErrorResult
from core.runtime import Config
import tkinter as tk
import queue
from reader import images as reader_images
from reader import records as reader_records
from reader import worker as reader_worker
from reader.worker import ReaderWorker
from core.memory import GlossaryMemory
from pipeline.batch import file_hash


VOICE = {"name": "Test / Normal", "engine_uuid": "engine", "engine_version": "1",
         "core_version": "1", "speaker_uuid": "speaker", "speaker_version": "1", "style_id": 7, "params": {}}


class ReaderTests(unittest.TestCase):
    def setUp(self):
        # Earlier fixtures may retain Tk objects in mock/closure cycles. Collect
        # them here on the UI thread, before this fixture starts its worker.
        gc.collect()
        self._cleaned = False
        self.addCleanup(self._cleanup)
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.withdraw()
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.db = self.folder / "memory.sqlite3"
        self.pages = []
        for number in (10, 2):
            original = self.folder / f"original-{number}.png"
            Image.new("RGB", (1000, 500), "white").save(original)
            record = {"input_path": str(original), "output_path": str(self.folder / f"missing-{number}.png"),
                      "input_sha256": file_hash(original),
                      "work": "reader-test", "target_language": "en", "exported": False,
                      "config": {"memory_db": str(self.db)}, "blocks": [
                          {"block_id": "b-one", "box": [20, 20, 200, 200], "source_text": "校正後の日本語", "translated_text": "Corrected English"},
                          {"block_id": "b-two", "box": [600, 100, 800, 300], "source_text": "第二の吹き出し", "translated_text": "Second bubble"}]}
            path = self.folder / f"page-{number}.json"
            path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            self.pages.append(path)
        (self.folder / "run-report.json").write_text('{"pages": []}', encoding="utf-8")
        self.started, self.gate = threading.Event(), threading.Event()
        self.gate.set()
        self.synthesized, self.played, self.stops = [], [], []
        self.catalog_calls, self.startup_calls = [], []
        self.catalog_error = None
        self.engine_started = threading.Event()
        self.wait_for_engine_close = False
        outer = self

        class FakeClient:
            def __init__(self, base_url, timeout=30):
                self.base_url = base_url

            def list_voices(self):
                outer.catalog_calls.append((self.base_url, threading.get_ident()))
                if outer.catalog_error is not None:
                    raise outer.catalog_error
                return [dict(VOICE), {**VOICE, "style_id": 8, "name": "Test / Soft"}]

        def start_engine(url, *, cancel_event):
            outer.startup_calls.append((url, threading.get_ident()))
            outer.engine_started.set()
            if outer.wait_for_engine_close:
                if not cancel_event.wait(3):
                    raise TimeoutError("test close did not cancel startup wait")
                raise RuntimeError("startup wait cancelled")
            return FakeClient(url).list_voices()

        def synthesize(client, text, **kwargs):
            outer.started.set()
            if not outer.gate.wait(4):
                raise TimeoutError("test synthesis gate")
            outer.synthesized.append((text, threading.get_ident(), kwargs))
            return {"wav_path": str(outer.folder/"fake.wav"), "duration_seconds": 0.03, "cached": True}

        speech = types.SimpleNamespace(VoicevoxClient=FakeClient, synthesize_cached=synthesize,
            start_local_voicevox=start_engine,
            play_wav=lambda path: self.played.append((path, threading.get_ident())),
            stop_playback=lambda: self.stops.append(threading.get_ident()))
        self.speech_patch = patch.dict(sys.modules, {"audio.speech": speech})
        self.settings_patch = patch.object(reader_worker, "SETTINGS_PATH", self.folder/"settings.json")
        self.speech_patch.start()
        self.settings_patch.start()
        self.closed = False
        self.window = reader.ReaderWindow(self.root, self.folder,
            config=Config(work="reader-test", memory_db=str(self.db)), on_closed=self.mark_closed)
        self.pump(lambda: bool(self.window.blocks) and not self.window.loading and self.window.voice_connected)

    def mark_closed(self):
        self.closed = True

    def pump(self, condition, timeout=5):
        until = time.monotonic()+timeout
        while not condition() and time.monotonic() < until:
            self.root.update()
            time.sleep(0.01)
        self.assertTrue(condition(), "reader condition timed out")

    def click_bubble(self, block_id):
        x1, y1, x2, y2 = self.window.blocks[block_id]["box"]
        with patch.object(self.window.canvas, "canvasx", return_value=(x1+x2)/2*self.window.scale[0]), \
             patch.object(self.window.canvas, "canvasy", return_value=(y1+y2)/2*self.window.scale[1]):
            self.window._click(types.SimpleNamespace(x=0, y=0))

    def exported_record(self, path, color=(20, 120, 200)):
        """Write visibly different, checksummed pixels at the export boundary."""
        path = Path(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        with Image.open(record["input_path"]) as original:
            size = original.size
        output = path.with_suffix(".png")
        Image.new("RGB", size, color).save(output)
        record.update(output_path=str(output), output_sha256=file_hash(output),
                      image_size=list(size), exported=True)
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return record

    def exporter(self, calls, *, failure=None, started=None, wait_for_cancel=False):
        """Fake only the expensive export boundary; use the real reader and files."""
        outer = self

        class ExportSession:
            def __init__(self, path, *, pool=None):
                self.path = Path(path)
                self.pool = pool

            def export_image(self, *, cancel=None):
                calls.append((self.path, threading.get_ident(), self.pool))
                if started is not None:
                    started.set()
                if wait_for_cancel:
                    if not outer.window.operation_cancel.wait(3):
                        raise TimeoutError("test export cancellation gate")
                if cancel:
                    cancel()
                if failure is not None:
                    raise failure
                return outer.exported_record(self.path)

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.close()

        return ExportSession

    def displayed_pixel(self):
        from PIL import ImageTk

        return ImageTk.getimage(self.window.photo).convert("RGB").getpixel((0, 0))

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        if hasattr(self, "gate"):
            self.gate.set()
        try:
            if hasattr(self, "window") and not self.closed:
                self.window.close()
                self.pump(lambda: self.closed)
            if hasattr(self, "window"):
                self.window.worker.join(timeout=5)
                self.assertFalse(self.window.worker.is_alive())
        finally:
            if hasattr(self, "root"):
                self.root.destroy()
            for name in ("settings_patch", "speech_patch"):
                patcher = getattr(self, name, None)
                if patcher is not None:
                    patcher.stop()
                    setattr(self, name, None)
            if hasattr(self, "temp"):
                self.temp.cleanup()
            self.window = self.root = None
            gc.collect()

    def test_text_only_pages_sort_naturally_and_preview_original_corrected_text(self):
        self.assertEqual([page["label"] for page in self.window.pages], ["original-2.png", "original-10.png"])
        self.assertEqual(self.window.japanese.get("1.0", "end-1c"), "校正後の日本語")
        self.assertEqual(self.window.english.get("1.0", "end-1c"), "Corrected English")
        from PIL import ImageTk
        self.assertEqual(ImageTk.getimage(self.window.photo).convert("RGB").getpixel((0, 0)), (255, 255, 255))
        self.assertFalse(Path(self.window.record["output_path"]).exists())

    def test_hidden_standalone_root_keeps_reader_window_visible(self):
        self.assertEqual(self.root.state(), "withdrawn")
        self.assertEqual(str(self.window.window.transient()), "")
        self.pump(lambda: bool(self.window.window.winfo_viewable()))
        self.assertEqual(self.window.window.state(), "normal")

    @unittest.skipUnless(sys.platform == "win32", "Native minimize commands are Windows-specific")
    def test_native_minimize_and_restore_preserve_reader_session(self):
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.SendMessageW.restype = ctypes.c_ssize_t
        self.wait_for_fit()
        window = self.window.window
        hwnd = int(window.wm_frame(), 0)
        before = self.window.page_index, self.window.block_id, self.window.token
        worker, original = self.window.worker, self.window.original
        stops = len(self.stops)
        with patch.object(reader_images, "original_preview", side_effect=AssertionError("restore reread original")), \
             patch.object(reader_records, "open_memory", side_effect=AssertionError("restore reread database")):
            for settings_open in (False, True):
                with self.subTest(settings_open=settings_open):
                    if settings_open:
                        self.window._show_tools("Settings")
                        self.root.update()
                    # iconify() bypasses the system-command path blocked by a Tk grab.
                    user32.SendMessageW(hwnd, 0x0112, 0xF020, 0)  # WM_SYSCOMMAND / SC_MINIMIZE
                    self.pump(lambda: window.state() == "iconic")
                    self.assertEqual(self.root.state(), "withdrawn")
                    user32.SendMessageW(hwnd, 0x0112, 0xF120, 0)  # SC_RESTORE
                    self.pump(lambda: window.state() == "normal" and window.winfo_viewable())
                    if settings_open:
                        self.pump(lambda: bool(self.window.settings_window.winfo_viewable()))
                        self.window._hide_settings()
                    self.wait_for_fit()
        self.assertEqual((self.window.page_index, self.window.block_id, self.window.token), before)
        self.assertIs(self.window.worker, worker)
        self.assertTrue(worker.is_alive())
        self.assertIs(self.window.original, original)
        self.assertEqual(len(self.stops), stops)
        self.assertFalse(self.synthesized)

    def test_visible_main_window_keeps_reader_transient_and_visible(self):
        self.root.deiconify()
        self.root.update()
        closed = []
        extra = reader.ReaderWindow(self.root, on_closed=lambda: closed.append(True))
        try:
            self.pump(lambda: bool(extra.window.winfo_viewable()))
            self.assertEqual(str(extra.window.transient()), str(self.root))
        finally:
            extra.close()
            self.pump(lambda: bool(closed))
            extra.worker.join(timeout=1)
            self.assertFalse(extra.worker.is_alive())
            self.root.withdraw()

    def wait_for_fit(self):
        self.root.update()
        self.pump(lambda: (not self.window.loading and not self.window.resizing
                          and self.window.after_resize is None and self.window.photo is not None
                          and self.window.thumbnail_pending is None
                          and all(index in self.window.thumbnail_cache or index in self.window.thumbnail_errors
                                  for index in self.window.thumbnail_cards)
                          and (self.window.photo.width(), self.window.photo.height())
                          == reader_images.preview_size(self.window.original.size, 0, self.window._canvas_size())))

    def test_window_starts_inside_screen_and_wide_tall_small_pages_fit_viewport(self):
        self.assertLessEqual(self.window.window.winfo_width(), self.window.window.winfo_screenwidth())
        self.assertLessEqual(self.window.window.winfo_height(), self.window.window.winfo_screenheight())
        for size in ((2400, 400), (400, 12000), (100, 50)):
            with self.subTest(size=size):
                source = self.folder/f"fit-{size[0]}-{size[1]}.png"
                Image.new("RGB", size, "white").save(source)
                self.window.open_paths([source])
                self.pump(lambda: self.window.record.get("input_path") == str(source.resolve()) and not self.window.loading)
                self.wait_for_fit()
                width, height = self.window._canvas_size()
                self.assertLessEqual(self.window.photo.width(), width)
                self.assertLessEqual(self.window.photo.height(), height)
                self.assertTrue(self.window.photo.width() == width or self.window.photo.height() == height)
        self.assertFalse(self.synthesized)

    def test_window_resize_uses_cached_pixels_without_cancelling_or_restarting_audio(self):
        self.wait_for_fit()
        self.gate.clear()
        self.click_bubble("b-one")
        self.pump(lambda: self.started.is_set())
        token, stops = self.window.token, len(self.stops)
        before = self.window.photo.width(), self.window.photo.height()
        with patch.object(reader_images, "original_preview", side_effect=AssertionError("resize reread original")), \
             patch.object(reader_records, "open_memory", side_effect=AssertionError("resize reread database")):
            self.window.window.geometry("900x650")
            self.root.update()
            self.pump(lambda: self.window.resizing)
            self.gate.set()
            self.wait_for_fit()
            self.pump(lambda: len(self.played) == 1)
        self.assertNotEqual((self.window.photo.width(), self.window.photo.height()), before)
        self.assertEqual(self.window.token, token)
        self.assertEqual(len(self.stops), stops)
        self.assertEqual(len(self.synthesized), 1)
        self.assertEqual(self.window.block_id, "b-one")

    def test_resize_events_are_debounced_and_last_viewport_wins(self):
        self.wait_for_fit()
        with patch.object(reader_images, "resize_original", wraps=reader_images.resize_original) as resized:
            self.window.window.geometry("900x650")
            self.root.update()
            for _ in range(12):
                self.window._canvas_resized()
            self.wait_for_fit()
        self.assertEqual(resized.call_count, 1)
        self.assertFalse(self.synthesized)

    def test_manual_zoom_keeps_scale_on_resize_and_real_scrolled_click_hits_bubble(self):
        self.window.autoplay.set(False)
        self.window.zoom.set("200%")
        self.window._load_page(zoom_only=True)
        self.pump(lambda: not self.window.loading)
        self.assertEqual((self.window.photo.width(), self.window.photo.height()), (2000, 1000))
        self.window.window.geometry("900x650")
        self.root.update()
        # The Configure handler must leave explicit magnification unchanged.
        self.window._resize_to_canvas()
        self.assertEqual(self.window.scale, (2.0, 2.0))
        width, height = self.window._canvas_size()
        self.window.canvas.xview_moveto((1400-width/2)/2000)
        self.window.canvas.yview_moveto(max(0, (400-height/2)/1000))
        x = round(1400-self.window.canvas.canvasx(0))
        y = round(400-self.window.canvas.canvasy(0))
        self.window.canvas.event_generate("<Button-1>", x=x, y=y)
        self.root.update()
        self.assertEqual(self.window.block_id, "b-two")
        self.assertFalse(self.synthesized)

    def test_scrolled_scaled_click_selects_correct_bubble(self):
        self.window.scale = (2.0, 2.0)
        with patch.object(self.window.canvas, "canvasx", return_value=1300), \
             patch.object(self.window.canvas, "canvasy", return_value=300):
            self.window._click(types.SimpleNamespace(x=15, y=25))
        self.assertEqual(self.window.block_id, "b-two")

    def test_open_connects_without_starting_or_playing_and_temporary_voice_is_not_saved(self):
        self.assertTrue(self.window.autoplay.get())
        self.assertEqual(len(self.catalog_calls), 1)
        self.assertNotEqual(self.catalog_calls[0][1], threading.get_ident())
        self.assertFalse(self.startup_calls)
        self.assertFalse(self.synthesized)
        self.assertFalse(self.played)
        self.assertTrue(self.window.temporary_voice)
        self.assertEqual(self.window.current_voice["style_id"], 7)
        self.assertIn("Temporary voice", self.window.voice_status.get())
        with GlossaryMemory(self.db, "reader-test", "en") as memory:
            self.assertTrue(all(not character.get("voice") for character in memory.list_characters()))

    def test_clicks_play_selected_bubble_once_and_repeated_click_replays(self):
        for index, block_id in enumerate(("b-one", "b-two", "b-two"), 1):
            self.click_bubble(block_id)
            self.pump(lambda: len(self.played) == index)
            self.root.update()
            self.assertEqual(len(self.synthesized), index)
        self.assertEqual([call[0] for call in self.synthesized],
                         ["校正後の日本語", "第二の吹き出し", "第二の吹き出し"])

    def test_click_starts_disconnected_engine_then_plays_without_another_click(self):
        self.window.voice_connected = False
        self.click_bubble("b-one")
        self.pump(lambda: len(self.played) == 1)
        self.assertEqual(len(self.startup_calls), 1)
        self.assertEqual(self.synthesized[0][0], "校正後の日本語")

    def test_connection_wait_keeps_only_latest_clicked_bubble(self):
        self.window.voice_connected = False
        self.window._set_voice_connecting(True)
        request = self.window.catalog_request
        self.click_bubble("b-one")
        self.click_bubble("b-two")
        self.window.events.put(("catalog", CatalogResult(self.window.url.get(), [VOICE]), request))
        self.pump(lambda: len(self.played) == 1)
        self.assertEqual([call[0] for call in self.synthesized], ["第二の吹き出し"])
        self.assertFalse(self.startup_calls)

    def test_stop_or_disable_autoplay_cancels_click_waiting_for_connection(self):
        for disable in (False, True):
            with self.subTest(disable=disable):
                self.window.voice_connected = False
                self.window._set_voice_connecting(True)
                request = self.window.catalog_request
                self.click_bubble("b-one")
                self.assertIsNotNone(self.window.pending_audio)
                if disable:
                    self.window.autoplay.set(False)
                    self.window._auto_changed()
                else:
                    self.window.stop()
                self.window.events.put(("catalog", CatalogResult(self.window.url.get(), [VOICE]), request))
                self.pump(lambda: not self.window.voice_connecting)
                self.assertIsNone(self.window.pending_audio)
                self.assertFalse(self.synthesized)
                self.assertFalse(self.played)

    def test_click_waiting_for_initial_failed_connection_starts_engine_once(self):
        self.window.voice_connected = False
        self.window._set_voice_connecting(True)
        request = self.window.catalog_request
        self.click_bubble("b-two")
        self.window.events.put(("error", ErrorResult("catalog", "not running"), request))
        self.pump(lambda: len(self.played) == 1)
        self.assertEqual(len(self.startup_calls), 1)
        self.assertEqual(self.synthesized[0][0], "第二の吹き出し")

    def test_engine_start_failure_keeps_autoplay_enabled_and_next_click_can_retry(self):
        self.window.voice_connected = False
        self.catalog_error = RuntimeError("cannot start engine")
        self.click_bubble("b-one")
        self.pump(lambda: not self.window.voice_connecting)
        self.assertTrue(self.window.autoplay.get())
        self.assertIsNone(self.window.pending_audio)
        self.assertIn("cannot start engine", self.window.voice_status.get())
        self.assertEqual(self.window.status.get(), "Voice unavailable. See Settings for details.")
        self.assertEqual(len(self.startup_calls), 1)
        self.assertFalse(self.played)
        self.catalog_error = None
        self.click_bubble("b-one")
        self.pump(lambda: len(self.played) == 1)
        self.assertEqual(len(self.startup_calls), 2)

    def test_playback_failure_does_not_silently_disable_click_to_speak(self):
        with patch.object(sys.modules["audio.speech"], "play_wav", side_effect=RuntimeError("audio unavailable")):
            self.click_bubble("b-one")
            self.pump(lambda: "Playback failed" in self.window.status.get())
        self.assertTrue(self.window.autoplay.get())
        self.click_bubble("b-one")
        self.pump(lambda: len(self.played) == 1)

    def test_play_works_with_temporary_actual_voice_without_saving_a_role(self):
        self.window.play()
        self.pump(lambda: len(self.played) == 1)
        self.assertEqual(self.synthesized[0][2]["voice"]["style_id"], 7)
        with GlossaryMemory(self.db, "reader-test", "en") as memory:
            self.assertTrue(all(not character.get("voice") for character in memory.list_characters()))

    def test_page_and_catalog_arrival_orders_both_select_an_unsaved_default(self):
        self.window.voices = []
        self.window.voice_connected = False
        self.window._show_bubble()
        self.assertIsNone(self.window.current_voice)
        self.window.refresh_voices()
        self.pump(lambda: self.window.voice_connected)
        self.assertTrue(self.window.temporary_voice)
        self.assertEqual(self.window.current_voice["style_id"], 7)

        self.window._clear_page()
        self.window.refresh_voices()
        self.pump(lambda: self.window.voice_connected)
        self.assertTrue(self.window.temporary_voice)
        self.window._load_page()
        self.pump(lambda: not self.window.loading)
        self.assertTrue(self.window.temporary_voice)
        self.assertEqual(self.window.current_voice["style_id"], 7)

    def test_automatic_catalog_never_replaces_persisted_character_voice(self):
        saved = {**VOICE, "style_id": 8, "name": "Saved / Soft"}
        with GlossaryMemory(self.db, "reader-test", "en") as memory:
            memory.upsert_character("unknown", "未知角色", saved)
        self.window._load_page()
        self.pump(lambda: not self.window.loading)
        self.window.refresh_voices()
        self.pump(lambda: self.window.voice_connected)
        self.assertEqual(self.window.current_voice, saved)
        self.assertFalse(self.window.temporary_voice)
        self.assertNotIn("Temporary voice", self.window.voice_status.get())

    def test_engine_button_runs_on_reader_worker_and_keeps_independent_status(self):
        self.window.start_voice_engine()
        self.pump(lambda: self.window.voice_connected and bool(self.startup_calls))
        self.assertEqual(self.startup_calls[0][0], "http://127.0.0.1:50021")
        self.assertNotEqual(self.startup_calls[0][1], threading.get_ident())
        voice_status = self.window.voice_status.get()
        self.window.stop()
        self.assertEqual(self.window.voice_status.get(), voice_status)
        self.assertEqual(str(self.window.connect_button.cget("state")), "normal")

    def test_connection_failure_keeps_page_text_and_can_be_retried(self):
        before = self.window.english.get("1.0", "end-1c")
        self.catalog_error = RuntimeError("engine not running")
        self.window.refresh_voices()
        self.pump(lambda: not self.window.voice_connecting)
        self.assertFalse(self.window.voice_connected)
        self.assertIn("engine not running", self.window.voice_status.get())
        self.assertEqual(self.window.english.get("1.0", "end-1c"), before)
        self.assertTrue(self.window.blocks)
        self.catalog_error = None
        self.window.start_voice_engine()
        self.pump(lambda: self.window.voice_connected)

    def test_changed_address_discards_a_stale_catalog_result(self):
        previous = self.window.catalog_request
        self.window.url.set("http://127.0.0.1:50022")
        self.window.events.put(("catalog", CatalogResult("http://127.0.0.1:50021", [VOICE]), previous))
        self.pump(lambda: self.window.events.empty())
        self.assertEqual(self.window.voices, [])
        self.assertFalse(self.window.voice_connected)
        self.assertIsNone(self.window.current_voice)
        self.assertIn("Address changed", self.window.voice_status.get())

    def test_close_cancels_engine_readiness_wait_without_playing(self):
        self.wait_for_engine_close = True
        self.window.start_voice_engine()
        self.pump(lambda: self.engine_started.is_set())
        self.window.close()
        self.pump(lambda: self.closed)
        self.window.worker.join(timeout=1)
        self.assertFalse(self.window.worker.is_alive())
        self.assertFalse(self.played)

    def test_magi_uncovered_regions_are_visible_but_not_playable_bubbles(self):
        page = Path(self.window.pages[self.window.page_index]["path"])
        record = json.loads(page.read_text(encoding="utf-8"))
        record["magi_coverage"] = {"detected_text_count": 3, "covered_text_count": 2,
                                   "unmatched_texts": [{"text_index": 2, "box": [300, 350, 500, 450]}]}
        page.write_text(json.dumps(record), encoding="utf-8")
        self.window.show_regions.set(True)
        self.window._load_page()
        self.pump(lambda: not self.window.loading)
        self.assertIn("Magi detected 1 additional text regions", self.window.status.get())
        items = self.window.canvas.find_withtag("unmatched-text")
        self.assertEqual(len(items), 2)
        self.assertEqual(self.window.canvas.type(items[0]), "rectangle")
        self.assertTrue(self.window.canvas.itemcget(items[0], "dash"))
        before = self.window.block_id
        with patch.object(self.window.canvas, "canvasx", return_value=400*self.window.scale[0]), \
             patch.object(self.window.canvas, "canvasy", return_value=400*self.window.scale[1]):
            self.window._click(types.SimpleNamespace(x=0, y=0))
        self.assertEqual(self.window.block_id, before)
        self.assertEqual(len(self.window.blocks), 2)
        self.window._draw_boxes()
        self.assertEqual(len(self.window.canvas.find_withtag("unmatched-text")), 2)

    def test_free_text_detection_review_is_visible_beside_translation(self):
        page = Path(self.window.pages[self.window.page_index]["path"])
        record = json.loads(page.read_text(encoding="utf-8"))
        record["blocks"][0]["metadata"] = {"detection_review": "框外文字需要复核。"}
        record["config"]["detect_free_text"] = True
        record["config"]["magi_model_path"] = "retained-magi-model"
        page.write_text(json.dumps(record), encoding="utf-8")
        self.window._load_page()
        self.pump(lambda: not self.window.loading)
        self.assertIn("框外文字需要复核。", self.window.bubble_notice.get())
        self.assertEqual(self.window.english.get("1.0", "end-1c"), "Corrected English")
        self.assertTrue(self.window.record["config"]["detect_free_text"])
        self.assertEqual(self.window.record["config"]["magi_model_path"], "retained-magi-model")
        self.assertFalse(self.synthesized)

    def test_changed_original_is_rejected_on_zoom_and_old_coordinates_cleared(self):
        Image.new("RGB", (1000, 500), "black").save(self.window.record["input_path"])
        self.window.zoom.set("100%")
        self.window._load_page(zoom_only=True)
        self.pump(lambda: not self.window.loading)
        self.assertIn("The original image has changed", self.window.status.get())
        self.assertIsNone(self.window.photo)
        self.assertIsNone(self.window.block_id)
        self.assertFalse(self.window.blocks)
        self.assertFalse(self.window.bubble_list.get_children())
        self.assertEqual(self.window.japanese.get("1.0", "end-1c"), "")

    def test_missing_original_checksum_is_explicitly_rejected(self):
        page = Path(self.window.pages[self.window.page_index]["path"])
        record = json.loads(page.read_text(encoding="utf-8"))
        del record["input_sha256"]
        page.write_text(json.dumps(record), encoding="utf-8")
        self.window._load_page()
        self.pump(lambda: not self.window.loading)
        self.assertIn("missing the original-image checksum", self.window.status.get())
        self.assertFalse(self.window.blocks)

    def test_rapid_bubble_selection_plays_only_latest_completed_audio(self):
        self.gate.clear()
        self.click_bubble("b-one")
        self.pump(lambda: self.started.is_set())
        self.click_bubble("b-two")
        self.gate.set()
        self.pump(lambda: len(self.played) == 1)
        self.assertEqual([call[0] for call in self.synthesized], ["校正後の日本語", "第二の吹き出し"])
        self.assertEqual(self.window.block_id, "b-two")

    def test_close_waits_for_running_synthesis_and_discards_queued_audio(self):
        self.window.current_voice = dict(VOICE)
        self.gate.clear()
        self.window.play()
        self.pump(lambda: self.started.is_set())
        self.window.play()
        self.window.close()
        self.assertFalse(self.closed)
        self.assertTrue(self.window.worker.is_alive())
        self.gate.set()
        self.pump(lambda: self.closed)
        self.window.worker.join(timeout=1)
        self.assertFalse(self.window.worker.is_alive())
        self.assertEqual(len(self.synthesized), 1)
        self.assertFalse(self.played)

    def test_stopped_synthesis_cannot_play_but_replay_uses_current_token(self):
        self.window.current_voice = dict(VOICE)
        self.gate.clear()
        self.window.play()
        self.pump(lambda: self.started.is_set())
        self.window.stop()
        self.gate.set()
        self.pump(lambda: bool(self.synthesized))
        self.root.update()
        self.assertEqual(self.played, [])
        self.window.play()
        self.pump(lambda: len(self.played) == 1)
        self.assertEqual(self.played[0][1], threading.get_ident())
        self.assertNotEqual(self.synthesized[0][1], threading.get_ident())
        self.assertEqual(self.synthesized[0][0], "校正後の日本語")
        self.assertEqual(self.window.page_index, 0)
        self.assertEqual(self.window.block_id, "b-one")

    def test_voice_change_invalidates_pending_audio_and_finished_audio_never_advances(self):
        self.window.refresh_voices()
        self.pump(lambda: len(self.window.voices) == 2)
        self.window.voice_box.current(0)
        self.window._voice_changed()
        self.gate.clear()
        self.window.play()
        self.pump(lambda: self.started.is_set())
        self.window.voice_box.current(1)
        self.window._voice_changed()
        self.gate.set()
        self.pump(lambda: bool(self.synthesized))
        self.root.update()
        self.assertEqual(self.played, [])
        self.window.autoplay.set(True)
        self.window._audio_finished(self.window.token)
        self.assertEqual((self.window.page_index, self.window.block_id), (0, "b-one"))

    def test_unknown_default_voice_new_role_and_manual_binding_persist(self):
        self.window.refresh_voices()
        self.pump(lambda: len(self.window.voices) == 2)
        self.window.voice_box.current(0)
        self.window._voice_changed()
        self.window._save_role()
        self.pump(lambda: "default voice saved" in self.window.status.get())
        with GlossaryMemory(self.db, "reader-test", "en") as memory:
            unknown = next(c for c in memory.list_characters() if c["id"] == "unknown")
            self.assertEqual(unknown["voice"]["style_id"], 7)
        self.window.role_name.set("主角")
        self.window._save_role(new=True)
        self.pump(lambda: any(c["name"] == "主角" for c in self.window.characters))
        character_id = self.window.characters[self.window.role_box.current()]["id"]
        self.window._bind_role()
        self.pump(lambda: "binding saved" in self.window.status.get())
        with GlossaryMemory(self.db, "reader-test", "en") as memory:
            binding = memory.get_character_binding(self.window.record["input_path"], "b-one")
            self.assertEqual(binding, {"character_id": character_id, "confirmed": True})

    def test_audio_setting_change_does_not_discard_pending_page_load(self):
        started, ready = threading.Event(), threading.Event()
        preview = reader_images.original_preview

        def delayed(*args, **kwargs):
            started.set()
            if not ready.wait(3):
                raise TimeoutError("test preview gate")
            return preview(*args, **kwargs)

        with patch.object(reader_images, "original_preview", side_effect=delayed):
            self.window.page_index = 1
            self.window._load_page()
            self.pump(lambda: started.is_set())
            self.window.speed.set("1.2")
            ready.set()
            self.pump(lambda: not self.window.loading)
        self.assertTrue(self.window.record["input_path"].endswith("original-10.png"))
        self.assertTrue(self.window.blocks)

    def test_magi_uses_same_worker_locks_controls_and_can_be_cancelled(self):
        from core.runtime import Cancelled

        started = threading.Event()
        calls = []

        def analyze(paths, *, model_path, device, progress, cancel):
            calls.append((paths, threading.get_ident()))
            started.set()
            progress({"message": "test Magi phase"})
            deadline = time.monotonic()+3
            while not self.window.operation_cancel.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
            cancel()
            self.fail("Magi cancellation should raise")

        with patch.dict(sys.modules, {"vision.characters": types.SimpleNamespace(analyze_pages=analyze)}):
            self.window._suggest_characters()
            self.assertTrue(self.window.magi_busy)
            self.assertEqual(str(self.window.role_box.cget("state")), "disabled")
            self.pump(lambda: started.is_set())
            self.window.stop()
            self.pump(lambda: not self.window.magi_busy)
        self.assertEqual(len(calls[0][0]), 2)
        self.assertNotEqual(calls[0][1], threading.get_ident())
        self.assertIn("Cancelled", self.window.status.get())
        self.assertEqual(str(self.window.role_box.cget("state")), "readonly")

    def test_raw_images_open_sorted_without_models_or_audio_then_reuse_worker_pool(self):
        raw = self.folder / "chapter"
        raw.mkdir()
        for name in ("10.png", "2.png"):
            Image.new("RGB", (300, 400), "white").save(raw / name)
        self.window.output_dir.set(str(self.folder / "reader-output"))
        self.window.config.memory_db = str(self.db)
        calls, exports = [], []
        fake_pool = Mock()

        def translate(source, output, config, *, pool, cancel_event, progress):
            self.assertFalse(config.export_images)
            self.assertIs(pool, fake_pool)
            self.assertFalse(cancel_event.is_set())
            calls.append((source, threading.get_ident()))
            progress({"type": "stage", "message": "translate"})
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            image_path = output / (Path(source).stem + ".png")
            record = {"input_path": source, "input_sha256": file_hash(source), "blocks": [
                {"block_id": "b-one", "box": [20, 20, 200, 200], "source_text": "待って", "translated_text": "Wait!"}],
                "work": config.work, "target_language": "en", "config": {"memory_db": str(self.db)},
                "status": "done", "output_path": str(image_path), "exported": False}
            image_path.with_suffix(".json").write_text(json.dumps(record), encoding="utf-8")
            return {"pages": [record], "cancelled": False}

        with patch("core.runtime.ModelPool", return_value=fake_pool) as factory, \
             patch("pipeline.batch.run_batch", side_effect=translate), \
             patch("pipeline.editing.PageEditorSession", self.exporter(exports)):
            self.window.open_paths(raw)
            self.pump(lambda: self.window.record.get("untranslated") is True and not self.window.loading)
            self.assertEqual([page["label"] for page in self.window.pages], ["2.png", "10.png"])
            self.assertTrue(self.window.photo)
            self.assertFalse(self.window.blocks)
            factory.assert_not_called()
            self.assertFalse(self.synthesized)
            self.window._translate_page()
            self.pump(lambda: not self.window.translation_busy and not self.window.loading and bool(self.window.blocks))
            self.assertEqual(self.window.page_list.get(0), "2.png")
            self.assertEqual(self.window.pages[0]["state"], "done")
            self.assertEqual(self.window.english.get("1.0", "end-1c"), "Wait!")
            self.assertTrue(self.window.record["exported"])
            self.assertEqual(self.window.view.get(), "英文译图")
            self.assertEqual(self.displayed_pixel(), (20, 120, 200))
            self.window.page_list.selection_clear(0, "end")
            self.window.page_list.selection_set(1)
            self.window._select_page()
            self.pump(lambda: not self.window.loading)
            self.window._translate_page()
            self.pump(lambda: not self.window.translation_busy and not self.window.loading)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(exports), 2)
            self.assertTrue(all(call[2] is fake_pool for call in exports))
            self.assertEqual(len({thread for _, thread in calls}), 1)
            self.assertNotEqual(calls[0][1], threading.get_ident())
            self.window.close()
            self.pump(lambda: self.closed)
            fake_pool.close.assert_called_once()

    def test_translation_cancel_keeps_preview_and_can_retry_without_starting_audio(self):
        raw = self.folder / "untranslated.png"
        Image.new("RGB", (300, 400), "white").save(raw)
        self.window.open_paths(raw)
        self.pump(lambda: self.window.record.get("untranslated") is True and not self.window.loading)
        started = threading.Event()

        def translate(*args, cancel_event, **kwargs):
            started.set()
            self.assertTrue(cancel_event.wait(3))
            return {"pages": [], "cancelled": True}

        with patch("core.runtime.ModelPool"), patch("pipeline.batch.run_batch", side_effect=translate):
            self.window._translate_page()
            self.pump(lambda: started.is_set())
            self.assertEqual(str(self.window.translate_button.cget("state")), "disabled")
            self.assertEqual(str(self.window.cancel_button.cget("state")), "normal")
            self.window.stop()
            self.pump(lambda: not self.window.translation_busy)
        self.assertIn("Cancelled", self.window.page_list.get(0))
        self.assertTrue(self.window.photo)
        self.assertEqual(str(self.window.translate_button.cget("state")), "normal")
        self.assertFalse(self.synthesized)

    def test_manual_corrections_are_not_overwritten_by_translate_page(self):
        page = Path(self.window.pages[self.window.page_index]["path"])
        self.window.record["manual_edits"] = True
        page.write_text(json.dumps(self.window.record), encoding="utf-8")
        original_blocks = json.loads(page.read_text(encoding="utf-8"))["blocks"]
        exports = []
        with patch("core.runtime.ModelPool"), patch("pipeline.batch.run_batch") as batch, \
             patch("pipeline.editing.PageEditorSession", self.exporter(exports)):
            self.window._translate_page()
            self.pump(lambda: not self.window.translation_busy and not self.window.loading)
            batch.assert_not_called()
        self.assertEqual(len(exports), 1)
        self.assertEqual(self.window.record["blocks"], original_blocks)
        self.assertTrue(self.window.record["manual_edits"])
        self.assertEqual(self.window.view.get(), "英文译图")
        self.assertEqual(self.displayed_pixel(), (20, 120, 200))
        self.assertFalse(self.synthesized)

    def test_ocr_candidates_are_not_overwritten_by_translate_page(self):
        page = Path(self.window.pages[self.window.page_index]["path"])
        self.window.record["ocr_candidate_edits"] = True
        self.window.record["blocks"][0]["metadata"] = {
            "ocr_candidates": [{"candidate_id": "candidate-1", "text": "候補"}],
            "ocr_review": "OCR 候选尚未采用。"}
        page.write_text(json.dumps(self.window.record), encoding="utf-8")
        original_blocks = json.loads(page.read_text(encoding="utf-8"))["blocks"]
        exports = []
        with patch("core.runtime.ModelPool"), patch("pipeline.batch.run_batch") as batch, \
             patch("pipeline.editing.PageEditorSession", self.exporter(exports)):
            self.window._translate_page()
            self.pump(lambda: not self.window.translation_busy and not self.window.loading)
            batch.assert_not_called()
        self.assertEqual(len(exports), 1)
        self.assertEqual(self.window.record["blocks"], original_blocks)
        self.assertTrue(self.window.record["ocr_candidate_edits"])
        self.assertFalse(self.window.record.get("manual_edits"))
        self.assertIn("OCR 候选尚未采用。", self.window.bubble_notice.get())
        self.assertEqual(self.window.english.get("1.0", "end-1c"), "Corrected English")
        self.assertFalse(self.synthesized)

    def test_switching_saved_text_to_english_exports_once_without_translation_or_audio(self):
        exports = []
        original_blocks = self.window.record["blocks"]
        with patch("core.runtime.ModelPool"), \
             patch("pipeline.batch.run_batch", side_effect=AssertionError("view switch reran translation")) as batch, \
             patch("pipeline.editing.PageEditorSession", self.exporter(exports)):
            self.window.view.set("英文译图")
            self.window._view_changed()
            self.pump(lambda: not self.window.translation_busy and not self.window.loading)
            self.assertEqual(self.displayed_pixel(), (20, 120, 200))
            self.assertEqual(self.window.record["blocks"], original_blocks)
            for view, pixel in (("原图", (255, 255, 255)), ("英文译图", (20, 120, 200))):
                self.window.view.set(view)
                self.window._view_changed()
                self.pump(lambda: not self.window.loading)
                self.assertEqual(self.displayed_pixel(), pixel)
            batch.assert_not_called()
        self.assertEqual(len(exports), 1)
        self.assertNotEqual(exports[0][1], threading.get_ident())
        self.assertFalse(self.synthesized)
        self.assertFalse(self.played)

    def test_existing_english_pixels_keep_scrolled_click_mapping_and_speak_japanese(self):
        page = Path(self.window.pages[self.window.page_index]["path"])
        self.exported_record(page)
        self.window._load_page()
        self.pump(lambda: not self.window.loading)
        with patch("core.runtime.ModelPool") as pool, \
             patch("pipeline.batch.run_batch") as batch, \
             patch("pipeline.editing.PageEditorSession") as export:
            self.window.view.set("英文译图")
            self.window._view_changed()
            self.pump(lambda: not self.window.loading)
            self.assertEqual(self.displayed_pixel(), (20, 120, 200))
            self.assertFalse(self.synthesized)
            self.window.zoom.set("200%")
            self.window._load_page(zoom_only=True)
            self.pump(lambda: not self.window.loading)
            self.assertEqual(self.window.scale, (2.0, 2.0))
            self.assertEqual(self.displayed_pixel(), (20, 120, 200))
            width, height = self.window._canvas_size()
            self.window.canvas.xview_moveto((1400-width/2)/2000)
            self.window.canvas.yview_moveto(max(0, (400-height/2)/1000))
            x = round(1400-self.window.canvas.canvasx(0))
            y = round(400-self.window.canvas.canvasy(0))
            self.window.canvas.event_generate("<Button-1>", x=x, y=y)
            self.pump(lambda: len(self.played) == 1)
            self.assertEqual(self.window.block_id, "b-two")
            self.assertEqual(self.synthesized[0][0], "第二の吹き出し")
            pool.assert_not_called()
            batch.assert_not_called()
            export.assert_not_called()

    def test_export_failure_and_cancellation_keep_saved_text_and_original_pixels(self):
        page = Path(self.window.pages[self.window.page_index]["path"])
        before = page.read_bytes()
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                exports, started = [], threading.Event()
                exporter = self.exporter(exports, started=started,
                    failure=None if cancelled else RuntimeError("layout overflow"),
                    wait_for_cancel=cancelled)
                with patch("core.runtime.ModelPool"), patch("pipeline.batch.run_batch") as batch, \
                     patch("pipeline.editing.PageEditorSession", exporter):
                    self.window.view.set("英文译图")
                    self.window._view_changed()
                    self.pump(lambda: started.is_set())
                    if cancelled:
                        self.window.stop()
                    self.pump(lambda: not self.window.translation_busy and not self.window.loading)
                    batch.assert_not_called()
                self.assertEqual(page.read_bytes(), before)
                self.assertFalse(page.with_suffix(".png").exists())
                self.assertEqual(self.window.view.get(), "原图")
                self.assertEqual(self.displayed_pixel(), (255, 255, 255))
                self.assertEqual(self.window.english.get("1.0", "end-1c"), "Corrected English")
                self.assertIn("cancelled" if cancelled else "layout overflow", self.window.status.get())
                self.assertFalse(self.synthesized)

    def test_context_reallocation_and_term_conflict_remain_visible_during_playback(self):
        self.window.blocks["b-one"].update(translated_text="Do not read", status="translated")
        self.window.blocks["b-two"].update(translated_text="pirated copies.", status="translated")
        self.window.blocks["b-one"]["metadata"] = {
            "context_review": "关联气泡译文重新分配，请复核未选气泡。",
            "translation_unit": {"members": ["b-one", "b-two"],
                "sources": [self.window.blocks[key]["source_text"] for key in ("b-one", "b-two")],
                "translated_text": "Do not read pirated copies."},
            "term_conflicts": ["Expected locked name: Misaki Takanashi"]}
        self.window._show_bubble()
        self.assertIn("关联气泡译文重新分配", self.window.bubble_notice.get())
        self.assertIn("Misaki Takanashi", self.window.bubble_notice.get())
        self.assertEqual(self.window.bubble_context.get(), "关联整句：Do not read pirated copies.")
        self.window.play()
        self.pump(lambda: len(self.played) == 1)
        self.assertEqual(self.synthesized[-1][0], self.window.blocks["b-one"]["source_text"])
        self.assertIn("关联气泡译文重新分配", self.window.bubble_notice.get())
        self.window.block_id = "b-two"
        self.window._show_bubble()
        self.assertEqual(self.window.bubble_notice.get(), "")
        self.assertEqual(self.window.bubble_context.get(), "")

    def test_exported_page_retranslation_keeps_original_json_and_clean_reference(self):
        page = Path(self.window.pages[self.window.page_index]["path"])
        record = json.loads(page.read_text(encoding="utf-8"))
        record.update(exported=True, clean_image={"file": "old.clean.png", "sha256": "saved-hash"})
        page.write_text(json.dumps(record), encoding="utf-8")
        before = page.read_bytes()
        destinations = []

        def translate(source, output, config, **kwargs):
            destinations.append(Path(output))
            self.assertFalse(config.export_images)
            return {"pages": [], "cancelled": True}

        with patch("core.runtime.ModelPool"), patch("pipeline.batch.run_batch", side_effect=translate):
            self.window._translate_page()
            self.pump(lambda: not self.window.translation_busy)
        self.assertEqual(destinations, [page.parent / "reader_text"])
        self.assertEqual(page.read_bytes(), before)


class PreviewSizeTests(unittest.TestCase):
    def test_fit_has_no_five_percent_floor_or_four_hundred_pixel_minimum(self):
        self.assertEqual(reader_images.preview_size((1000, 20000), 0, (300, 400)), (20, 400))
        self.assertEqual(reader_images.preview_size((1000, 500), 0, (120, 80)), (120, 60))
        self.assertEqual(reader_images.preview_size((100, 50), 0, (600, 500)), (600, 300))

    def test_manual_scale_retains_pixel_budget(self):
        self.assertEqual(reader_images.preview_size((1000, 500), 0.25, (120, 80)), (250, 125))
        width, height = reader_images.preview_size((10000, 10000), 2, (800, 600))
        self.assertLessEqual(width*height, 16_000_000)


class TranslatedPreviewTests(unittest.TestCase):
    def test_display_requires_verified_output_and_matching_original_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original, page = root/"original.png", root/"translated.json"
            output = page.with_suffix(".png")
            Image.new("RGB", (120, 60), "white").save(original)
            Image.new("RGB", (120, 60), "navy").save(output)
            record = {"input_path": str(original), "input_sha256": file_hash(original),
                      "output_path": str(output), "output_sha256": file_hash(output),
                      "image_size": [120, 60], "exported": True, "blocks": []}
            preview, pixels = reader_images.translated_preview(record, page, 1, (120, 60))
            self.assertEqual(preview.getpixel((0, 0)), (0, 0, 128))
            self.assertEqual(pixels.size, (120, 60))
            with self.subTest(problem="missing output checksum"), self.assertRaises(ValueError):
                reader_images.translated_preview({**record, "output_sha256": None}, page, 1, (120, 60))
            Image.new("RGB", (120, 60), "black").save(output)
            with self.subTest(problem="changed output"), self.assertRaises(ValueError):
                reader_images.translated_preview(record, page, 1, (120, 60))
            Image.new("RGB", (60, 120), "navy").save(output)
            record["output_sha256"] = file_hash(output)
            with self.subTest(problem="different output dimensions"), self.assertRaises(ValueError):
                reader_images.translated_preview(record, page, 1, (120, 60))
            Image.new("RGB", (120, 60), "navy").save(output)
            record["output_sha256"] = file_hash(output)
            Image.new("RGB", (120, 60), "gray").save(original)
            with self.subTest(problem="changed original"), self.assertRaises(ValueError):
                reader_images.translated_preview(record, page, 1, (120, 60))


class ReaderCollectionTests(unittest.TestCase):
    def test_saved_result_and_clean_assets_do_not_duplicate_raw_page(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "2.jpg"
            Image.new("RGB", (100, 100)).save(source)
            output = root / "reader_output"
            output.mkdir()
            for name in ("translated.png", "translated.clean.png", "translated.mask.png"):
                Image.new("RGB", (100, 100)).save(output/name)
            record = {"input_path": str(source), "blocks": [], "status": "done",
                      "output_path": str(output/"translated.png"), "clean_image": {"file": "translated.clean.png"}}
            (output/"translated.json").write_text(json.dumps(record), encoding="utf-8")
            pages, errors = reader.collect_pages(root, output)
            self.assertEqual(errors, [])
            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0]["input"], str(source.resolve()))
            self.assertEqual(pages[0]["state"], "done")

    def test_raw_image_can_reopen_separate_output_result_with_manual_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "2.jpg"
            source.touch()
            output = root / "results"
            output.mkdir()
            record = {"input_path": str(source), "blocks": [], "status": "done", "manual_edits": True}
            target = output/"saved.json"
            target.write_text(json.dumps(record), encoding="utf-8")
            pages, errors = reader.collect_pages(source, output)
            self.assertEqual(errors, [])
            self.assertEqual(pages[0]["path"], str(target.resolve()))

    def test_collection_prefers_manual_text_then_candidate_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "page.jpg"
            source.touch()
            output = root / "results"
            output.mkdir()
            base = {"input_path": str(source), "blocks": [], "status": "done"}
            candidate = output / "candidate.json"
            candidate.write_text(json.dumps({**base, "ocr_candidate_edits": True}), encoding="utf-8")
            (output / "new-automatic.json").write_text(json.dumps(base), encoding="utf-8")
            pages, errors = reader.collect_pages(source, output)
            self.assertFalse(errors)
            self.assertEqual(pages[0]["path"], str(candidate.resolve()))
            manual = output / "manual.json"
            manual.write_text(json.dumps({**base, "manual_edits": True}), encoding="utf-8")
            # Candidate previews retain their metadata without outranking a
            # different result that contains actual manual text corrections.
            candidate.write_text(json.dumps({**base, "ocr_candidate_edits": True}), encoding="utf-8")
            pages, errors = reader.collect_pages(source, output)
            self.assertFalse(errors)
            self.assertEqual(pages[0]["path"], str(manual.resolve()))
            # Both records contain manual corrections: a saved candidate must
            # not make an older correction outrank a newer manual-only result.
            candidate.write_text(json.dumps({**base, "manual_edits": True,
                                             "ocr_candidate_edits": True}), encoding="utf-8")
            os.utime(candidate, ns=(1_000_000_000, 1_000_000_000))
            os.utime(manual, ns=(2_000_000_000, 2_000_000_000))
            pages, errors = reader.collect_pages(source, output)
            self.assertFalse(errors)
            self.assertEqual(pages[0]["path"], str(manual.resolve()))

    def test_raw_resume_isolated_by_work_language_and_database(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = root / "2.jpg"
            source.touch()
            output = root / "results"
            output.mkdir()
            config = Config(work="comic-a", memory_db=str(root/"a.db"))
            matching = {"input_path": str(source), "blocks": [], "work": config.work,
                        "target_language": "en", "config": {"memory_db": config.memory_db}}
            target = output/"0-correct.json"
            target.write_text(json.dumps(matching), encoding="utf-8")
            for name, changed in (("work", {"work": "comic-b"}),
                                  ("language", {"target_language": "zh"}),
                                  ("database", {"config": {"memory_db": str(root/"b.db")}})):
                record = {**matching, **changed, "manual_edits": True}
                (output/f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
            pages, errors = reader.collect_pages(root, output, config=config)
            self.assertEqual(errors, [])
            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0]["path"], str(target))
            # Explicitly opening another work's result is still allowed and uses its own scope.
            pages, _ = reader.collect_pages(output/"work.json", output, config=config)
            self.assertEqual(pages[0]["path"], str(output/"work.json"))


class ReaderSettingsTests(unittest.TestCase):
    def test_settings_worker_keeps_event_polling_after_invalid_file(self):
        cases = [None, "{", "[]", "null", "42", '"text"', "false", "{}",
                 '{"url": "http://localhost:50021", "speed": 1.25}']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reader-settings.json"
            for contents in cases:
                with self.subTest(contents=contents):
                    if contents is not None:
                        path.write_text(contents, encoding="utf-8")
                    commands, events = queue.Queue(), queue.Queue()
                    commands.put(("settings", None, 1))
                    commands.put(("close", None, 2))
                    with patch.object(reader_worker, "SETTINGS_PATH", path):
                        ReaderWorker(commands, events, threading.Event(), threading.Event()).run()
                    settings = events.get_nowait()
                    self.assertEqual(events.get_nowait(), ("closed", None, 2))
                    events.put(settings)
                    window = types.SimpleNamespace(events=events, closing=threading.Event(),
                                                  url=Mock(), speed=Mock(), refresh_voices=Mock(),
                                                  window=Mock(), _drain=Mock())
                    reader.ReaderWindow._drain(window)
                    expected = json.loads(contents) if contents and contents.startswith('{"url"') else {}
                    self.assertEqual(settings, ("settings", expected, 1))
                    window.url.set.assert_called_once_with(expected.get("url", window.url.get()))
                    window.speed.set.assert_called_once_with(str(expected.get("speed", 1.0)))
                    window.refresh_voices.assert_called_once_with()
                    window.window.after.assert_called_once_with(80, window._drain)
                    if contents is None:
                        self.assertFalse(path.exists())
                    else:
                        self.assertEqual(path.read_text(encoding="utf-8"), contents)


if __name__ == "__main__":
    unittest.main()
