"""Clickable page thumbnails with bounded decoding; no models or real speech."""
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from PIL import Image, ImageTk

from dataclasses import replace
from reader.messages import ThumbnailResult
from ui.theme import PALETTE
from reader import images as reader_images
from tests import test_demo_navigation as demo_tests


class PageThumbnailTests(TestCase):
    def setUp(self):
        self.fixture = demo_tests.DemoNavigationTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.window = self.fixture.window
        self.root = self.fixture.root
        self.window.window.geometry("1250x850")
        self.root.update()
        self._settle_thumbnails({0, 1, 2})

    def _pump(self, condition, timeout=10):
        self.fixture._pump(condition, timeout)

    def _settle_thumbnails(self, indices):
        self._pump(lambda: indices.issubset(self.window.thumbnail_cache)
                   and self.window.thumbnail_pending is None)
        self.root.update_idletasks()

    def _assert_selected(self, index):
        self.assertEqual(self.window.page_index, index)
        self.assertEqual(self.window.page_list.curselection(), (index,))
        canvas = self.window.page_thumbnails
        for number, card in self.window.thumbnail_cards.items():
            fill = canvas.itemcget(card["background"], "fill")
            outline = canvas.itemcget(card["background"], "outline")
            if number == index:
                self.assertEqual(fill, PALETTE["selection"])
                self.assertEqual(outline, PALETTE["accent"])
            else:
                self.assertNotEqual(outline, PALETTE["accent"])

    def test_every_demo_page_has_its_own_visible_translated_thumbnail(self):
        self.assertFalse(self.window.page_list.winfo_viewable())
        self.assertFalse(self.window.thumbnail_label.winfo_viewable())
        canvas = self.window.page_thumbnails
        viewport_top = canvas.canvasy(0)
        viewport_bottom = canvas.canvasy(canvas.winfo_height())
        self.assertEqual(set(self.window.thumbnail_cards), {0, 1, 2})
        for index, path in enumerate(self.fixture.paths):
            with self.subTest(page=index):
                card = self.window.thumbnail_cards[index]
                self.assertEqual(canvas.itemcget(card["label"], "text"), "")
                self.assertTrue(canvas.itemcget(card["image"], "image"))
                bounds = canvas.bbox(card["image"])
                self.assertGreaterEqual(bounds[1], viewport_top)
                self.assertLessEqual(bounds[3], viewport_bottom)
                actual = ImageTk.getimage(self.window.thumbnail_cache[index]).convert("RGB")
                with Image.open(path.with_suffix(".png")) as opened:
                    expected = opened.convert("RGB")
                ratio = min(reader_images.THUMBNAIL_SIZE[0] / expected.width,
                            reader_images.THUMBNAIL_SIZE[1] / expected.height)
                direct = expected.resize((round(expected.width * ratio), round(expected.height * ratio)),
                                         Image.Resampling.LANCZOS)
                expected.thumbnail(reader_images.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                # The active page can reuse its already decoded preview; other
                # cards may use Pillow's reducing-gap thumbnail optimization.
                self.assertEqual(actual.size, direct.size)
                self.assertIn(actual.tobytes(), (direct.tobytes(), expected.tobytes()))
        self._assert_selected(0)

    def test_click_and_previous_next_keep_page_selection_highlight_in_sync(self):
        canvas = self.window.page_thumbnails
        box = canvas.bbox(self.window.thumbnail_cards[1]["image"])
        canvas.event_generate("<Button-1>", x=round((box[0] + box[2]) / 2 - canvas.canvasx(0)),
                              y=round((box[1] + box[3]) / 2 - canvas.canvasy(0)))
        self.fixture._assert_translated_page(1)
        self._settle_thumbnails({0, 1, 2})
        self._assert_selected(1)
        self.window.next_button.invoke()
        self.fixture._assert_translated_page(2)
        self._assert_selected(2)
        self.window.previous_button.invoke()
        self.fixture._assert_translated_page(1)
        self._assert_selected(1)
        self.fixture.forbidden.assert_not_called()

    def test_long_library_decodes_visible_pages_and_keeps_cache_bounded(self):
        folder = self.fixture.folder / "many-pages"
        folder.mkdir()
        paths = []
        for index in range(120):
            path = folder / f"page-{index:03}.png"
            Image.new("RGB", (24, 32), (index, 100, 200)).save(path)
            paths.append(path)
        real_open = Image.open
        decoded = []
        resolved_folder = folder.resolve()

        def count_open(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve().parent == resolved_folder:
                decoded.append(Path(path).name)
            return real_open(path, *args, **kwargs)

        with patch("PIL.Image.open", side_effect=count_open):
            self.window.open_paths(paths)
            self._pump(lambda: len(self.window.pages) == 120 and not self.window.loading
                       and self.window.photo is not None)
            self._settle_thumbnails({0, 1, 2})
            self.assertLessEqual(len(decoded), 12, "Opening 120 pages must decode only the visible range")
            self.assertLessEqual(len(self.window.thumbnail_cards), 7)
            initial_decodes = len(decoded)
            peak_cache = len(self.window.thumbnail_cache)
            for index in (0, 15, 30, 45, 60, 75, 90, 105, 119):
                self.window._see_thumbnail(index)
                self.root.update()
                self._settle_thumbnails({index})
                peak_cache = max(peak_cache, len(self.window.thumbnail_cache))
                self.assertLessEqual(len(self.window.thumbnail_cache), 24)
                self.assertLessEqual(len(self.window.thumbnail_cards), 7)
            self.assertGreater(self.window.page_thumbnails.yview()[0], 0.8)
            self.assertGreater(len(decoded), 24, "Exercise eviction after more than a cache of decodes")
            self.window._select_thumbnail(119)
            self._pump(lambda: self.window.page_index == 119 and not self.window.loading)
            self._settle_thumbnails({119})
            self._assert_selected(119)
            self.assertNotIn(0, self.window.thumbnail_cache)
            before_return = len(decoded)
            self.window._see_thumbnail(0)
            self.root.update()
            self._settle_thumbnails({0})
            self.assertGreater(len(decoded), before_return)
            self.assertEqual(ImageTk.getimage(self.window.thumbnail_cache[0]).convert("RGB").getpixel((10, 10)),
                             (0, 100, 200))
            self.assertLessEqual(len(self.window.thumbnail_cache), 24)
            print(f"Thumbnail bounds: 120 pages, initial decodes={initial_decodes}, "
                  f"total decodes={len(decoded)}, peak cache={peak_cache}.")
        self.fixture.forbidden.assert_not_called()

    def test_zoom_and_thumbnail_clicks_cannot_supersede_an_incoming_library(self):
        self.window.open_paths(self.fixture.paths)
        self.assertTrue(self.window.opening)
        token = self.window.page_token
        self.window.zoom.set("100%")
        self.window._load_page(zoom_only=True)
        self.window._select_thumbnail(2)
        self.assertEqual(self.window.page_token, token)
        self._pump(lambda: not self.window.opening and not self.window.loading
                   and self.window.current_result_path() == str(self.fixture.paths[0].resolve()))
        self._settle_thumbnails({0, 1, 2})
        self._assert_selected(0)
        self.assertEqual(self.window.page_position.get(), "1 / 3")

    def test_reimport_rejects_old_generation_thumbnail_and_error(self):
        old_generation = self.window.thumbnail_generation
        old_page = dict(self.window.pages[0])
        self.window.open_paths(self.fixture.paths)
        self._pump(lambda: not self.window.loading and len(self.window.pages) == 3
                   and self.window.photo is not None)
        self._settle_thumbnails({0, 1, 2})
        self.assertGreater(self.window.thumbnail_generation, old_generation)
        expected_photo = self.window.thumbnail_cache[0]
        payload = ThumbnailResult(0, (old_page["path"], old_page["input"]),
                                  image=Image.new("RGB", (3, 3), "red"), view="translated")
        self.window.events.put(("thumbnail", payload, (old_generation, 0)))
        self.window.events.put(("thumbnail_error", replace(payload, message="stale test failure"),
                                (old_generation, 0)))
        self._pump(lambda: self.window.events.empty())
        self.assertIs(self.window.thumbnail_cache[0], expected_photo)
        self.assertNotIn(0, self.window.thumbnail_errors)
        self.assertEqual(self.window.page_position.get(), "1 / 3")
