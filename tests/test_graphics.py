"""Run: python -m unittest discover -s tests -p test_graphics.py -v"""
import importlib.util
from copy import deepcopy
import io
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision import graphics


def block(box=(10, 10, 150, 110), text="Hello world", status="translated", **kwargs):
    return SimpleNamespace(box=box, source_text="Original", translated_text=text,
                           status=status, block_id="b0001", error="", safe_box=None,
                           metadata={}, **kwargs)


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font_path = graphics.resolve_font()

    def test_english_words_are_never_broken(self):
        text = "This guy is connected to Hayes!!"
        layout = graphics.layout_text(text, 130, 230, self.font_path)
        self.assertFalse(layout.overflow)
        self.assertEqual(" ".join(layout.lines).split(), text.split())
        self.assertIn("connected", " ".join(layout.lines).split())
        self.assertLessEqual(layout.width, 130)
        self.assertLessEqual(layout.height, 230)

    def test_very_long_word_is_explicit_overflow(self):
        word = "supercalifragilisticexpialidocious"
        layout = graphics.layout_text(word, 40, 200, self.font_path)
        self.assertTrue(layout.overflow)
        self.assertEqual(layout.lines, [word])
        self.assertEqual(layout.font_size, 12)
        self.assertIn("Unbreakable", layout.reason)

    def test_binary_search_and_largest_fitting_size(self):
        layout = graphics.layout_text("Readable words fit here", 160, 90, self.font_path,
                                      min_font_size=12, max_font_size=100)
        self.assertFalse(layout.overflow)
        self.assertLessEqual(layout.measurements, 1 + math.ceil(math.log2(100 - 12 + 1)))
        next_size = layout.font_size + 1
        larger = graphics.layout_text("Readable words fit here", 160, 90, self.font_path,
                                      min_font_size=next_size, max_font_size=next_size)
        self.assertTrue(larger.overflow)

    def test_empty_text_and_explicit_paragraphs(self):
        self.assertEqual(graphics.layout_text(" \n ", 1, 1).lines, [])
        layout = graphics.layout_text("First paragraph.\n\nSecond paragraph.", 300, 300, self.font_path)
        self.assertIn("", layout.lines)
        self.assertEqual(" ".join(layout.lines).split(), "First paragraph. Second paragraph.".split())

    def test_vertical_overflow_never_shrinks_below_minimum(self):
        layout = graphics.layout_text("one two three four five six", 50, 15, self.font_path)
        self.assertTrue(layout.overflow)
        self.assertEqual(layout.font_size, 12)

    def test_missing_font_and_missing_glyph_are_clear_errors(self):
        with self.assertRaises(graphics.FontError):
            graphics.layout_text("Hello", 100, 100, "/no/such/font.ttf")
        with self.assertRaisesRegex(graphics.FontError, "lacks"):
            graphics.layout_text("\U0010ffff", 100, 100, self.font_path)

    def test_chinese_punctuation_and_paragraphs(self):
        candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simsun.ttc"),
                      Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
        path = next((str(p) for p in candidates if p.exists()), None)
        if path is None:
            self.skipTest("A CJK font is not installed; no CJK model support is implied")
        text = "你好（世界），这是一个测试。\n下一段，继续！"
        layout = graphics.layout_text(text, 100, 350, path, "zh", 16, 16)
        self.assertFalse(layout.overflow, layout.reason)
        self.assertEqual("".join(layout.lines), text.replace("\n", ""))
        for line in layout.lines:
            if line:
                self.assertNotIn(line[0], graphics._CLOSE_PUNCT)
                self.assertNotIn(line[-1], graphics._OPEN_PUNCT)

    def test_rendered_pixels_stay_in_safe_box(self):
        original = Image.new("RGB", (200, 160), "white")
        b = block((20, 20, 180, 140), "Words stay inside this rectangle.")
        b.safe_box = (30, 30, 170, 130)
        rendered = graphics.render(original, [b], self.font_path)
        changed = np.any(np.asarray(rendered) != np.asarray(original), axis=2)
        self.assertTrue(changed.any())
        self.assertFalse(changed[:30].any())
        self.assertFalse(changed[130:].any())
        self.assertFalse(changed[:, :30].any())
        self.assertFalse(changed[:, 170:].any())

    def test_failed_empty_and_overflow_blocks_are_not_rendered(self):
        original = Image.new("RGB", (160, 120), "white")
        failed = block(text="Hello", status="failed")
        empty = block(text="")
        overflow = block((20, 20, 40, 40), "A word that cannot fit")
        result = graphics.render(original, [failed, empty, overflow], self.font_path)
        self.assertTrue(np.array_equal(np.asarray(original), np.asarray(result)))
        self.assertEqual(overflow.status, "overflow")
        self.assertTrue(overflow.error)

    def test_layout_gray_preparation_only_converts_roi_and_keeps_legacy_pixels(self):
        rgb = np.random.default_rng(12345).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        for mode in ("RGB", "RGBA", "L"):
            with self.subTest(mode=mode):
                image = Image.fromarray(rgb).convert(mode)
                target = block((30, 40, 160, 180))
                expected = np.asarray(image.convert("L"))[40:180, 30:160]
                original_convert = Image.Image.convert
                converted_sizes = []
                def convert(source, *args, **kwargs):
                    converted_sizes.append(source.size)
                    return original_convert(source, *args, **kwargs)
                with patch.object(Image.Image, "convert", convert), \
                     patch("vision.graphics._uniform_interior", return_value=None) as interior:
                    graphics.safe_layout_box(image, target)
                np.testing.assert_array_equal(interior.call_args.args[0], expected)
                self.assertEqual(converted_sizes, [(130, 140)])


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is required for cleanup tests")
class CleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import before patch.dict(sys.modules), whose restoration otherwise
        # unloads newly imported torch Python modules but not native bindings.
        __import__("torch")

    def make_image(self, dark=False, background=None):
        bg = (0 if dark else 255) if background is None else background
        image = Image.new("RGB", (180, 110), (bg, bg, bg))
        font = ImageFont.truetype(graphics.resolve_font(), 22)
        ImageDraw.Draw(image).text((40, 35), "HELLO", font=font, fill="white" if dark else "black")
        return image, block((20, 15, 160, 95), "Hello")

    def test_both_polarities_fill_aa_and_preserve_unmasked_pixels(self):
        for dark in (False, True):
            with self.subTest(dark=dark):
                image, b = self.make_image(dark)
                mask = graphics.build_text_mask(image, [b])
                cleaned = graphics.inpaint(image, [b])
                self.assertTrue(mask.any())
                expected = 0 if dark else 255
                difference = np.abs(np.asarray(cleaned).astype(int) - expected)
                self.assertLessEqual(difference.max(), 8)
                self.assertTrue(np.array_equal(np.asarray(cleaned)[mask == 0], np.asarray(image)[mask == 0]))
                self.assertEqual(b.metadata["cleanup"]["method"], "solid_fill")

    def test_cleanup_plan_analyzes_once_with_identical_masks_images_and_metadata(self):
        pool = SimpleNamespace(acquire=lambda *args: lambda crop, _: Image.new("RGB", crop.size, "red"))
        for dark, background in ((False, None), (True, None), (False, 160)):
            with self.subTest(dark=dark, background=background), \
                 patch.dict(sys.modules, {"simple_lama_inpainting": SimpleNamespace(SimpleLama=object)}):
                image, original = self.make_image(dark, background)
                reference, actual = deepcopy(original), deepcopy(original)
                expected_image = graphics.inpaint(image, [reference], pool=pool)
                expected_mask = graphics.build_text_mask(image, [reference])
                with patch("vision.graphics._mask_patches", wraps=graphics._mask_patches) as analyze:
                    plan = graphics.prepare_cleanup(image, [actual])
                    actual_image = graphics.inpaint(image, [actual], pool=pool, plan=plan)
                    actual_mask = graphics.build_text_mask(image, [actual], plan=plan)
                    analyze.assert_called_once()
                np.testing.assert_array_equal(actual_image, expected_image)
                np.testing.assert_array_equal(actual_mask, expected_mask)
                self.assertEqual(actual.metadata, reference.metadata)
                self.assertIs(plan._source_image, image)
                for _, mask, _, color in plan.patches:
                    self.assertFalse(mask.flags.writeable)
                    self.assertFalse(color.flags.writeable)

    def test_cleanup_plan_rejects_other_image_and_changed_targets_before_model_access(self):
        image, target = self.make_image(background=160)
        plan = graphics.prepare_cleanup(image, [target])
        pool = SimpleNamespace(acquire=Mock(side_effect=AssertionError("Must reject stale plan first")))
        changed_targets = []
        for name, value in (("block_id", "other"), ("box", (10, 15, 160, 95)),
                            ("status", "failed"), ("translated_text", ""), ("translated_text", "New words")):
            changed = deepcopy(target)
            setattr(changed, name, value)
            changed_targets.append([changed])
        changed = deepcopy(target)
        changed.metadata["kind"] = "free_text"
        changed_targets.extend(([changed], [], [target, block()]))
        cases = [(image.copy(), [target], plan), (Image.new("RGB", (1, 1)), [target], plan),
                 (image, [target], object())]
        cases.extend((image, targets, plan) for targets in changed_targets)
        with patch("vision.graphics._mask_patches", side_effect=AssertionError("Must not reanalyze a stale plan")):
            for source, targets, supplied_plan in cases:
                with self.subTest(source=source.size, targets=len(targets)):
                    with self.assertRaisesRegex(ValueError, "Cleanup plan"):
                        graphics.inpaint(source, targets, pool=pool, plan=supplied_plan)
                    with self.assertRaisesRegex(ValueError, "Cleanup plan"):
                        graphics.build_text_mask(source, targets, plan=supplied_plan)
        pool.acquire.assert_not_called()

    def test_cleanup_plan_ignores_ocr_and_layout_metadata_and_keeps_union(self):
        image, first = self.make_image()
        second = block((30, 20, 170, 100))
        second.block_id = "second"
        blocks = [first, second]
        plan = graphics.prepare_cleanup(image, blocks)
        expected = graphics.build_text_mask(image, blocks)
        first.metadata["ocr_crop_box"] = [0, 0, 180, 110]
        first.safe_box = (40, 30, 140, 80)
        first.metadata["ocr_candidates"] = [{"text": "different OCR"}]
        with patch("vision.graphics._mask_patches", side_effect=AssertionError("Plan already prepared")):
            actual = graphics.build_text_mask(image, blocks, plan=plan)
        np.testing.assert_array_equal(actual, expected)

    def test_cleanup_plan_rejects_in_place_image_size_or_mode_changes(self):
        for change in (lambda image: image.thumbnail((90, 55)), lambda image: image.putalpha(255)):
            image, target = self.make_image()
            plan = graphics.prepare_cleanup(image, [target])
            change(image)
            with self.assertRaisesRegex(ValueError, "Cleanup plan"):
                graphics.build_text_mask(image, [target], plan=plan)

    def test_free_text_uses_background_relative_polarity_on_colored_panels(self):
        font = ImageFont.truetype(graphics.resolve_font(), 24)
        # Cover both cases an absolute 127 cutoff gets wrong, plus the usual
        # black/white backgrounds. The mask must stay on letters, not the panel.
        for background, ink, expected in (
                ((120, 185, 120), "white", "dark"),
                ((80, 110, 140), "black", "light"),
                ((255, 255, 255), "black", "light"),
                ((0, 0, 0), "white", "dark")):
            with self.subTest(background=background, ink=ink):
                image = Image.new("RGB", (240, 60), background)
                drawing = ImageDraw.Draw(image)
                drawing.text((28, 15), "COLOR TEXT!", font=font, fill=ink)
                b = block((20, 10, 210, 50), "Colored text")
                b.metadata["kind"] = "free_text"
                mask = graphics.build_text_mask(image, [b]) > 0
                pixels = np.asarray(image)
                letters = np.max(np.abs(pixels.astype(int) - background), axis=2) > 20
                self.assertTrue(letters.any())
                self.assertGreater(np.mean(mask[letters]), .99)
                self.assertLess(mask[10:50, 20:210].mean(), .50)
                self.assertFalse(mask[:10].any())
                self.assertFalse(mask[50:].any())
                self.assertFalse(mask[:, :20].any())
                self.assertFalse(mask[:, 210:].any())
                self.assertEqual(b.metadata["cleanup"]["background"], expected)
                self.assertEqual(b.metadata["cleanup"]["polarity_source"], "text_border_otsu")
                if background not in ((255, 255, 255), (0, 0, 0)):
                    self.assertEqual(b.metadata["cleanup"]["method"], "lama")

    def test_colored_text_complex_cleanup_only_composites_glyph_mask(self):
        image = Image.new("RGB", (240, 60), (120, 185, 120))
        ImageDraw.Draw(image).text((28, 15), "COLOR TEXT!",
                                  font=ImageFont.truetype(graphics.resolve_font(), 24), fill="white")
        b = block((20, 10, 210, 50), "Colored text")
        b.metadata["kind"] = "free_text"
        mask = graphics.build_text_mask(image, [b])
        fake_pool = SimpleNamespace(acquire=lambda *args: lambda crop, _: Image.new("RGB", crop.size, "red"))
        with patch.dict(sys.modules, {"simple_lama_inpainting": SimpleNamespace(SimpleLama=object)}):
            cleaned = graphics.inpaint(image, [b], pool=fake_pool)
        before, after = np.asarray(image), np.asarray(cleaned)
        np.testing.assert_array_equal(before[mask == 0], after[mask == 0])
        self.assertTrue(np.all(after[mask > 0] == (255, 0, 0)))

    def test_box_coordinates_and_boundary_outline_protection(self):
        image, b = self.make_image()
        ImageDraw.Draw(image).rectangle((20, 15, 159, 94), outline="black", width=2)
        mask = graphics.build_text_mask(image, [b])
        self.assertTrue(mask[35:65, 40:120].any())
        self.assertFalse(mask[:15].any())
        self.assertFalse(mask[:, :20].any())
        self.assertFalse(mask[15:17, 20:160].any())
        self.assertFalse(mask[93:95, 20:160].any())

    def test_ellipse_interior_protects_external_art_and_text_layout(self):
        image = Image.new("RGB", (180, 160), (150, 150, 150))
        drawing = ImageDraw.Draw(image)
        drawing.ellipse((20, 10, 160, 150), fill="white", outline="black", width=2)
        drawing.text((58, 48), "HELLO", font=ImageFont.truetype(graphics.resolve_font(), 18), fill="black")
        drawing.rectangle((14, 132, 26, 139), fill="black")  # Outside the bubble, inside detection.
        b = block((10, 5, 170, 155), "Every word stays inside")
        mask = graphics.build_text_mask(image, [b])
        self.assertFalse(mask[132:140, 14:27].any())
        graphics.prepare_layout(image, [b])
        self.assertEqual(b.metadata["safe_region_method"], "rectangle_inside_uniform_background")
        x1, y1, x2, y2 = b.metadata["layout"]["safe_box"]
        interior = graphics._uniform_interior(np.asarray(image.convert("L"))[5:155, 10:170])
        self.assertTrue(np.all(interior[y1 - 5:y2 - 5, x1 - 10:x2 - 10]))

    def test_preflight_rectangle_is_reused_after_erasing(self):
        image, b = self.make_image()
        graphics.prepare_layout(image, [b])
        before = dict(b.metadata["layout"])
        cleaned = graphics.inpaint(image, [b])
        graphics.render(cleaned, [b])
        self.assertEqual(before, b.metadata["layout"])

    def test_explicit_cleanup_rectangles_preserve_bubble_outline_and_outside_pixels(self):
        for dark in (False, True):
            with self.subTest(dark=dark):
                image, b = self.make_image(dark)
                drawing = ImageDraw.Draw(image)
                drawing.rectangle((20, 15, 159, 94), outline="white" if dark else "black", width=2)
                region = (20, 15, 130, 70)
                pool = SimpleNamespace(acquire=Mock(side_effect=AssertionError("Flat cleanup needs no model")))
                result = graphics.clean_regions(image, b, [region], pool=pool)
                before, after = np.asarray(image), np.asarray(result)
                expected = 0 if dark else 255
                self.assertTrue(np.all(after[40:60, 40:120] == expected))
                np.testing.assert_array_equal(before[15:17, 20:160], after[15:17, 20:160])
                np.testing.assert_array_equal(before[70:], after[70:])
                np.testing.assert_array_equal(before[:, 130:], after[:, 130:])
                self.assertEqual(b.metadata["manual_cleanup_regions"], [list(region)])
                pool.acquire.assert_not_called()

    def test_overlapping_masks_use_union(self):
        image, first = self.make_image()
        second = block((30, 20, 170, 100))
        expected = graphics.build_text_mask(image, [first]) | graphics.build_text_mask(image, [second])
        actual = graphics.build_text_mask(image, [first, second])
        np.testing.assert_array_equal(actual, expected)

    def test_overflow_preflight_preserves_original(self):
        image, b = self.make_image()
        b.translated_text = "X" * 100
        graphics.prepare_layout(image, [b])
        self.assertEqual(b.status, "overflow")
        cleaned = graphics.inpaint(image, [b])
        np.testing.assert_array_equal(cleaned, image)
        self.assertFalse(graphics.build_text_mask(image, [b]).any())

    def test_failed_and_empty_blocks_are_never_erased(self):
        image, b = self.make_image()
        b.status = "failed"
        empty = block(text="")
        np.testing.assert_array_equal(graphics.inpaint(image, [b, empty]), image)

    def test_missing_glyph_fails_only_its_bubble_and_preserves_source(self):
        image = Image.new("RGB", (360, 110), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(graphics.resolve_font(), 22)
        draw.text((40, 35), "HELLO", font=font, fill="black")
        draw.text((220, 35), "WORLD", font=font, fill="black")
        bad = block((20, 15, 160, 95), "Missing glyph \U0010ffff")
        good = block((200, 15, 340, 95), "World")
        good.block_id = "b0002"
        graphics.prepare_layout(image, [bad, good])
        self.assertEqual(bad.status, "failed")
        self.assertIn("Font unsupported", bad.error)
        self.assertEqual(good.status, "translated")
        mask = graphics.build_text_mask(image, [bad, good])
        self.assertFalse(mask[:, :180].any())
        self.assertTrue(mask[:, 180:].any())
        cleaned = graphics.inpaint(image, [bad, good])
        np.testing.assert_array_equal(np.asarray(cleaned)[:, :180], np.asarray(image)[:, :180])
        self.assertFalse(np.array_equal(np.asarray(cleaned)[:, 180:], np.asarray(image)[:, 180:]))
        rendered = graphics.render(cleaned, [bad, good])
        np.testing.assert_array_equal(np.asarray(rendered)[:, :180], np.asarray(image)[:, :180])
        self.assertEqual(good.status, "translated")

    def test_lama_padding_is_cropped_and_only_mask_is_composited(self):
        image, b = self.make_image(background=160)
        mask = graphics.build_text_mask(image, [b])
        self.assertTrue(mask.any())
        calls = []
        def acquire(name, loader, mover):
            calls.append(name)
            return lambda source, _: Image.new("RGB", (184, 112), "red")
        fake_pool = SimpleNamespace(acquire=acquire)
        fake_module = SimpleNamespace(SimpleLama=object)
        with patch.dict(sys.modules, {"simple_lama_inpainting": fake_module}):
            repaired = graphics.inpaint(image, [b], pool=fake_pool)
        self.assertEqual(calls, ["lama"])
        self.assertEqual(repaired.size, image.size)
        np.testing.assert_array_equal(np.asarray(repaired)[mask == 0], np.asarray(image)[mask == 0])
        self.assertTrue(np.all(np.asarray(repaired)[mask > 0] == (255, 0, 0)))

    def test_cancellation_is_checked_before_cleanup(self):
        image, b = self.make_image()
        def cancel():
            raise InterruptedError("cancelled")
        with self.assertRaises(InterruptedError):
            graphics.inpaint(image, [b], cancel=cancel)

    def test_offline_lama_fails_before_attempting_model_download(self):
        image, b = self.make_image(background=160)
        constructor = Mock()
        pool = SimpleNamespace(local_files_only=True, acquire=lambda name, loader, mover: loader())
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"LAMA_MODEL": str(Path(directory) / "missing.pt")}):
                with patch.dict(sys.modules, {"simple_lama_inpainting": SimpleNamespace(SimpleLama=constructor)}):
                    with self.assertRaisesRegex(FileNotFoundError, "Offline LaMa checkpoint missing"):
                        graphics.inpaint(image, [b], pool=pool)
        constructor.assert_not_called()

    def test_offline_lama_accepts_standard_torch_checkpoint_cache(self):
        image, b = self.make_image(background=160)
        constructor = Mock(return_value=lambda source, _: source)
        pool = SimpleNamespace(local_files_only=True, acquire=lambda name, loader, mover: loader())
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "checkpoints"
            cache.mkdir()
            (cache / "test.pt").write_bytes(b"test stub, constructor mocked")
            with patch.dict(os.environ, {"LAMA_MODEL": "", "LAMA_MODEL_URL": "https://example.invalid/test.pt"}):
                with patch("torch.hub.get_dir", return_value=directory):
                    with patch.dict(sys.modules, {"simple_lama_inpainting": SimpleNamespace(SimpleLama=constructor)}):
                        graphics.inpaint(image, [b], pool=pool)
        constructor.assert_called_once()


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is required for cleanup tests")
class HaloCleanupTests(unittest.TestCase):
    """Faint glyph halos beyond the two-pixel fringe in flat-filled bubbles."""
    BOX = (5, 5, 215, 165)

    @staticmethod
    def bubble(dark=False, halo_value=None, line=False):
        import cv2
        ink, paper = (255, 0) if dark else (0, 255)
        image = Image.new("L", (220, 170), 150)
        draw = ImageDraw.Draw(image)
        draw.ellipse((10, 10, 210, 160), fill=paper, outline=ink, width=3)
        if line:  # artwork inside the bubble, long enough to be rejected
            draw.line((40, 128, 182, 128), fill=ink, width=1)
        text = Image.new("L", image.size, 0)
        ImageDraw.Draw(text).text((62, 55), "HELLO", font=ImageFont.truetype(graphics.resolve_font(), 30), fill=255)
        strokes = np.asarray(text) > 0
        pixels = np.asarray(image).copy()
        if halo_value is not None:
            distance = cv2.distanceTransform((~strokes).astype(np.uint8), cv2.DIST_L2, 3)
            pixels[(distance > 2.5) & (distance <= 4.0)] = halo_value
        pixels[strokes] = np.where(np.asarray(text)[strokes] > 128, ink, (ink + paper) // 2)
        near = cv2.dilate(strokes.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
        return Image.fromarray(pixels).convert("RGB"), strokes, near, paper

    @staticmethod
    def jpeg(image, quality):
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=quality)
        return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")

    @staticmethod
    def without_halo():
        return patch("vision.graphics._halo", side_effect=lambda mask, *_: np.zeros_like(mask))

    def masks(self, image, box=BOX):
        """Mask, the same mask without the halo step, and cleanup metadata."""
        target = block(box)
        mask = graphics.build_text_mask(image, [target]) > 0
        with self.without_halo():
            base = graphics.build_text_mask(image, [block(box)]) > 0
        return mask, base, target.metadata["cleanup"]

    def test_faint_ring_is_removed_for_both_polarities(self):
        for dark in (False, True):
            with self.subTest(dark=dark):
                image, _, near, paper = self.bubble(dark, halo_value=7 if dark else 248)
                target = block(self.BOX)
                cleaned = np.asarray(graphics.inpaint(image, [target])).astype(int)
                with self.without_halo():
                    before = np.asarray(graphics.inpaint(image, [block(self.BOX)])).astype(int)
                mask = graphics.build_text_mask(image, [block(self.BOX)]) > 0
                self.assertGreaterEqual(np.abs(before[near] - paper).max(), 7)
                self.assertLessEqual(np.abs(cleaned[near] - paper).max(), 2)
                self.assertEqual(target.metadata["cleanup"]["method"], "solid_fill")
                self.assertGreater(target.metadata["cleanup"]["halo_pixels"], 0)
                np.testing.assert_array_equal(cleaned[~mask], np.asarray(image).astype(int)[~mask])

    def test_outline_and_rejected_line_art_keep_their_pixels(self):
        import cv2
        near_border = Image.new("RGB", (170, 90), (150, 150, 150))
        draw = ImageDraw.Draw(near_border)
        draw.rectangle((8, 8, 160, 80), fill="white", outline="black", width=2)
        draw.text((13, 30), "HELLO", font=ImageFont.truetype(graphics.resolve_font(), 26), fill="black")
        cases = [(self.bubble(halo_value=248, line=True)[0], self.BOX), (self.jpeg(near_border, 80), (4, 4, 166, 86))]
        for image, box in cases:
            with self.subTest(size=image.size):
                source = np.asarray(image.convert("L")).astype(int)
                mask, base, _ = self.masks(image, box)
                added = mask & ~base
                self.assertTrue(added.any())
                self.assertTrue(np.array_equal(mask & base, base))
                art = cv2.dilate(((source < 235) & ~base).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                self.assertFalse((added & art).any())
                self.assertTrue((source[added] > 235).all())

    def test_light_hatching_below_otsu_is_guarded(self):
        import cv2
        image, _, _, _ = self.bubble(halo_value=248)
        draw = ImageDraw.Draw(image)
        for x in range(150, 175, 4):  # light pencil hatching beside the text
            draw.line((x, 60, x + 10, 100), fill=(205, 205, 205), width=1)
        source = np.asarray(image.convert("L")).astype(int)
        mask, base, _ = self.masks(image)
        hatching = cv2.dilate(((source <= 235) & ~base).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.assertTrue((mask & ~base).any())
        self.assertFalse((mask & ~base & hatching).any())

    def test_screentone_paper_is_not_flattened_into_a_pad(self):
        import cv2
        image, _, _, _ = self.bubble()
        pixels = np.asarray(image).copy()
        paper = (pixels == 255).all(axis=2)
        tone = np.random.default_rng(67890).integers(245, 256, paper.sum())
        pixels[paper] = np.repeat(tone[:, None], 3, axis=1)
        mask, base, cleanup = self.masks(Image.fromarray(pixels))
        self.assertEqual(cleanup["method"], "solid_fill")
        band = (cv2.dilate(base.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~base & paper
        self.assertLess((mask & ~base)[band].mean(), 0.15)

    def test_lama_regions_get_no_halo(self):
        image = Image.new("RGB", (180, 110), (160, 160, 160))
        ImageDraw.Draw(image).text((40, 35), "HELLO", font=ImageFont.truetype(graphics.resolve_font(), 22), fill="black")
        mask, base, cleanup = self.masks(image, (20, 15, 160, 95))
        self.assertEqual(cleanup["method"], "lama")
        self.assertEqual(cleanup["halo_pixels"], 0)
        np.testing.assert_array_equal(mask, base)

    def test_jpeg_ringing_near_text_is_reduced(self):
        for quality, remaining in ((90, 0), (75, None)):
            with self.subTest(quality=quality):
                image, strokes, near, paper = self.bubble()
                image = self.jpeg(image, quality)
                cleaned = np.asarray(graphics.inpaint(image, [block(self.BOX)])).astype(int)
                with self.without_halo():
                    before = np.asarray(graphics.inpaint(image, [block(self.BOX)])).astype(int)
                zone = near & ~strokes
                after_count = int((np.abs(cleaned[zone] - paper) > 3).sum())
                before_count = int((np.abs(before[zone] - paper) > 3).sum())
                self.assertLess(after_count * 2, before_count)
                if remaining is not None:
                    self.assertEqual(after_count, remaining)


if __name__ == "__main__":
    unittest.main()
