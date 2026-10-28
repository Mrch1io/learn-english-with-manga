"""Left thumbnail rail collapse: real pixels grow without file/model/audio work."""
import threading
from unittest import TestCase
from unittest.mock import patch

from reader import images as reader_images
from tests import test_page_thumbnails as thumbnail_tests


class ThumbnailCollapseTests(TestCase):
    def setUp(self):
        self.fixture = thumbnail_tests.PageThumbnailTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.window = self.fixture.window
        self.root = self.fixture.root
        self._settle()

    def _pump(self, condition):
        self.fixture._pump(condition)

    def _settle(self):
        self.root.update()
        self._pump(lambda: not self.window.loading and not self.window.resizing
                   and self.window.after_resize is None)

    def _select_shopkeeper(self):
        key = self.fixture.fixture.records[0]["blocks"][1]["block_id"]
        self.window.bubble_list.selection_set(key)
        self.window._select_bubble()
        self.root.update()
        self.assertEqual(self.window.role.get(), "Shopkeeper")

    def test_collapse_enlarges_actual_pixels_and_restores_state_without_io(self):
        self._select_shopkeeper()
        old_width = self.window.photo.width()
        old_panel_width = self.window.pages_panel.winfo_width()
        state = (self.window.page_index, self.window.block_id, self.window.role.get(),
                 self.window.voice.get(), self.window.token)
        cache = dict(self.window.thumbnail_cache)
        original, worker = self.window.original, self.window.worker
        with patch("PIL.Image.open", side_effect=AssertionError("Collapse reopened an image")) as image_open, \
             patch.object(self.window, "_invalidate", side_effect=AssertionError("Collapse interrupted audio")) as invalidate, \
             patch("core.runtime.ModelPool", side_effect=AssertionError("Collapse started a model")) as model:
            self.window.collapse_pages_button.invoke()
            self._settle()
            self.assertTrue(self.window.pages_collapsed)
            self.assertFalse(self.window.pages_panel.winfo_viewable())
            self.assertTrue(self.window.pages_rail.winfo_viewable())
            self.assertLess(self.window.pages_rail.winfo_width(), old_panel_width)
            self.assertEqual(self.window.zoom.get(), "Fit to Width")
            self.assertGreater(self.window.photo.width(), old_width + 20)
            self.assertAlmostEqual(self.window.photo.width(), self.window.canvas.winfo_width(), delta=1)
            scroll_top = self.window.canvas.yview()[0]
            self.window.canvas.event_generate("<MouseWheel>", delta=-120)
            self.root.update_idletasks()
            self.assertGreater(self.window.canvas.yview()[0], scroll_top)
            self.assertEqual((self.window.page_index, self.window.block_id, self.window.role.get(),
                              self.window.voice.get(), self.window.token), state)
            self.window.expand_pages_button.invoke()
            self._settle()
            self.assertFalse(self.window.pages_collapsed)
            self.assertTrue(self.window.pages_panel.winfo_viewable())
            self.assertFalse(self.window.pages_rail.winfo_viewable())
            self.assertEqual(self.window.zoom.get(), "Fit to Window")
            self.assertEqual(self.window.photo.width(), old_width)
            image_open.assert_not_called()
            invalidate.assert_not_called()
            model.assert_not_called()
        self.assertEqual((self.window.page_index, self.window.block_id, self.window.role.get(),
                          self.window.voice.get(), self.window.token), state)
        self.assertIs(self.window.original, original)
        self.assertIs(self.window.worker, worker)
        self.assertEqual(set(self.window.thumbnail_cache), set(cache))
        for index, photo in cache.items():
            self.assertIs(self.window.thumbnail_cache[index], photo)

    def test_manual_zoom_is_restored_when_thumbnail_rail_expands(self):
        self.window.zoom.set("75%")
        self.window._load_page(zoom_only=True)
        self._settle()
        old_size = (self.window.photo.width(), self.window.photo.height())
        token = self.window.token
        with patch("PIL.Image.open", side_effect=AssertionError("Panel toggles must use cached pixels")) as image_open:
            self.window.collapse_pages_button.invoke()
            self._settle()
            self.assertEqual(self.window.zoom.get(), "Fit to Width")
            self.window.expand_pages_button.invoke()
            self._settle()
            self.assertEqual(self.window.zoom.get(), "75%")
            self.assertEqual((self.window.photo.width(), self.window.photo.height()), old_size)
            self.assertEqual(self.window.token, token)
            image_open.assert_not_called()

    def test_left_and_right_panels_toggle_independently(self):
        self.window.collapse_pages_button.invoke()
        self._settle()
        self.assertTrue(self.window.pages_collapsed)
        self.assertFalse(self.window.sidebar_collapsed)
        left_only_width = self.window.photo.width()
        self.window.collapse_sidebar_button.invoke()
        self._settle()
        self.assertTrue(self.window.pages_collapsed)
        self.assertTrue(self.window.sidebar_collapsed)
        self.assertGreater(self.window.photo.width(), left_only_width)
        self.window.expand_pages_button.invoke()
        self._settle()
        self.assertFalse(self.window.pages_collapsed)
        self.assertTrue(self.window.sidebar_collapsed)
        self.window.expand_sidebar_button.invoke()
        self._settle()
        self.assertFalse(self.window.pages_collapsed)
        self.assertFalse(self.window.sidebar_collapsed)
        self.assertTrue(self.window.pages_panel.winfo_viewable())
        self.assertTrue(self.window.sidebar.winfo_viewable())

    def test_expand_during_pending_width_resize_restores_manual_zoom(self):
        self.window.zoom.set("75%")
        self.window._load_page(zoom_only=True)
        self._settle()
        original_size = (self.window.photo.width(), self.window.photo.height())
        started, release = threading.Event(), threading.Event()
        real_resize = reader_images.resize_original

        def delayed_resize(original, scale, fit, **kwargs):
            if kwargs.get("fit_width"):
                started.set()
                if not release.wait(5):
                    raise TimeoutError("Test did not release the pending width resize")
            return real_resize(original, scale, fit, **kwargs)

        with patch("reader.images.resize_original", side_effect=delayed_resize):
            try:
                self.window.collapse_pages_button.invoke()
                self._pump(started.is_set)
                self.assertTrue(self.window.resizing)
                self.window.expand_pages_button.invoke()
                self.root.update()
                self.assertFalse(self.window.pages_collapsed)
                self.assertEqual(self.window.zoom.get(), "75%")
            finally:
                release.set()
            self._settle()
        self.assertEqual((self.window.photo.width(), self.window.photo.height()), original_size)
        self.assertFalse(self.window.pages_collapsed)
        self.assertEqual(self.window.zoom.get(), "75%")

    def test_fit_to_width_survives_page_navigation_and_resizes_from_cached_pixels(self):
        self.window.collapse_pages_button.invoke()
        self._settle()
        self.window.next_button.invoke()
        self.fixture.fixture._wait_for_page(1)
        self._settle()
        self.assertEqual(self.window.page_position.get(), "2 / 3")
        self.assertEqual(self.window.zoom.get(), "Fit to Width")
        self.assertEqual(self.window.view_button.cget("text"), "Show Original")
        self.assertAlmostEqual(self.window.photo.width(), self.window.canvas.winfo_width(), delta=1)
        width = self.window.photo.width()
        original, token = self.window.original, self.window.token
        with patch("PIL.Image.open", side_effect=AssertionError("Width fit must resize cached pixels")) as image_open:
            self.window.window.geometry("1100x850")
            self._settle()
            self.assertLess(self.window.photo.width(), width)
            self.assertAlmostEqual(self.window.photo.width(), self.window.canvas.winfo_width(), delta=1)
            self.assertIs(self.window.original, original)
            self.assertEqual(self.window.token, token)
            image_open.assert_not_called()
        self.window.expand_pages_button.invoke()
        self._settle()
        self.assertEqual(self.window.zoom.get(), "Fit to Window")
        self.assertEqual(self.window.page_position.get(), "2 / 3")
        self.fixture.fixture.forbidden.assert_not_called()

    def test_new_page_starts_at_top_while_window_resize_keeps_reading_position(self):
        self.window.collapse_pages_button.invoke()
        self._settle()
        self.window.canvas.yview_moveto(1)
        self.root.update_idletasks()
        self.assertGreater(self.window.canvas.yview()[0], 0.1)
        self.window.next_button.invoke()
        self.fixture.fixture._wait_for_page(1)
        self._settle()
        self.assertEqual(self.window.page_position.get(), "2 / 3")
        self.assertEqual(self.window.zoom.get(), "Fit to Width")
        self.assertAlmostEqual(self.window.canvas.yview()[0], 0, delta=0.001)
        self.window.canvas.yview_moveto(0.25)
        self.root.update_idletasks()
        self.assertGreater(self.window.canvas.yview()[0], 0.1)
        self.window.window.geometry("1100x850")
        self._settle()
        self.assertGreater(self.window.canvas.yview()[0], 0.1)
        self.assertEqual(self.window.page_position.get(), "2 / 3")
