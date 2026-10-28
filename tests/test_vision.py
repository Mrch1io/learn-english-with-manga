"""Reading order remains explicit without merging neighboring bubbles."""
import unittest
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw
from core.runtime import Cancelled, Config, TextBlock
from vision import detection as vision
from vision.detection import (add_text_regions, audit_text_regions, detect, detect_page_text, locate_wide_line,
                    order_blocks, preview_ocr_crop, run_ocr, validate_ocr_crop)


class ReadingOrderTests(unittest.TestCase):
    def test_direction_orders_rows_and_preserves_independent_ids(self):
        blocks = [TextBlock((10, 80, 40, 110), block_id="lower"),
                  TextBlock((10, 10, 40, 40), block_id="left"),
                  TextBlock((50, 15, 80, 45), block_id="right")]
        for direction, expected in [("rtl", ["right", "left", "lower"]),
                                    ("ltr", ["left", "right", "lower"])]:
            with self.subTest(direction=direction):
                ordered = order_blocks(blocks, direction)
                self.assertEqual([b.block_id for b in ordered], expected)
                self.assertEqual([b.reading_order for b in ordered], [0, 1, 2])
                self.assertEqual(len(ordered), 3)
        self.assertEqual(order_blocks([]), [])


class PageTextTests(unittest.TestCase):
    def test_outside_text_enters_same_blocks_and_inside_columns_are_not_duplicated(self):
        bubble = TextBlock((20, 20, 120, 180), block_id="existing-id")
        regions = [{"box": [30, 30, 60, 160], "score": .7},
                   {"box": [65, 30, 100, 160], "score": .6},
                   {"box": [200, 220, 450, 260], "score": .5}]
        blocks = add_text_regions([bubble], regions, (500, 300))
        self.assertEqual(len(blocks), 2)
        self.assertIs(blocks[0], bubble)
        self.assertEqual(bubble.block_id, "existing-id")
        self.assertEqual(blocks[1].metadata["kind"], "free_text")
        self.assertEqual(blocks[1].box, (200, 220, 450, 260))
        repeated = add_text_regions([bubble], list(reversed(regions)), (500, 300))
        self.assertEqual(blocks[1].block_id, repeated[1].block_id)

    def test_page_without_bubbles_still_has_text_and_clips_boundary_proposals(self):
        blocks = add_text_regions([], [{"box": [-1.2, 10.1, 121.3, 49.9], "score": .5}], (100, 80))
        self.assertEqual(blocks[0].box, (0, 10, 100, 50))
        with self.assertRaisesRegex(ValueError, "coordinates"):
            add_text_regions([], [{"box": [float("nan"), 0, 20, 20], "score": .5}], (100, 80))
        self.assertEqual(add_text_regions([], [{"box": [120, 0, 140, 30], "score": .5}], (100, 80)), [])

    def test_distinct_nearby_text_regions_stay_separate_and_nested_proposals_deduplicate(self):
        regions = [{"box": [10, 10, 50, 80], "score": .8},
                   {"box": [15, 15, 45, 75], "score": .5},
                   {"box": [52, 10, 90, 80], "score": .7}]
        blocks = add_text_regions([], regions, (100, 100))
        self.assertEqual(len(blocks), 2)
        self.assertEqual([b.box for b in blocks], [(10, 10, 50, 80), (52, 10, 90, 80)])

    def test_free_text_keeps_edge_glyphs_and_uses_existing_ocr(self):
        image = Image.new("RGB", (100, 80))
        block = TextBlock((0, 10, 100, 50), metadata={"kind": "free_text"})
        mocr = Mock(return_value="枠のない文字")
        pool = SimpleNamespace(acquire=lambda *args: mocr)
        with patch("vision.detection.shrink_crop_to_text") as shrink:
            run_ocr(image, [block], pool=pool, config=Config())
        shrink.assert_not_called()
        self.assertEqual(mocr.call_args.args[0].size, (100, 48))
        self.assertEqual(block.source_text, "枠のない文字")
        self.assertEqual(block.status, "pending")


class WideLineTests(unittest.TestCase):
    @staticmethod
    def line_image(light=False):
        image = Image.new("RGB", (420, 80), "#444444" if light else "white")
        draw = ImageDraw.Draw(image)
        for x in range(32, 376, 22):
            draw.rectangle((x, 28, x + 11, 47), fill="white" if light else "black")
        return image

    def test_both_polarities_find_line_and_preserve_all_pixels_at_segment_seams(self):
        for light in (False, True):
            with self.subTest(light=light):
                image = self.line_image(light)
                result = locate_wide_line(image, [8, 16, 408, 64])
                self.assertIsNotNone(result)
                x1, y1, x2, y2 = result["box"]
                self.assertLessEqual(x1, 32)
                self.assertGreaterEqual(x2, 373)
                self.assertLessEqual(y1, 28)
                self.assertGreaterEqual(y2, 48)
                segments = result["ocr_segments"]
                self.assertGreater(len(segments), 1)
                self.assertLessEqual(len(segments), 8)
                self.assertEqual(segments[0][0], x1)
                self.assertEqual(segments[-1][2], x2)
                for first, second in zip(segments, segments[1:]):
                    self.assertEqual(first[2], second[0])
                self.assertEqual(sum(segment[2] - segment[0] for segment in segments), x2 - x1)
                # The synthetic glyph gaps are empty; no in-glyph cut is needed.
                self.assertFalse(result["ocr_split_review"])

    def test_blank_narrow_and_solid_border_crops_do_not_create_text(self):
        image = Image.new("RGB", (420, 80), "white")
        self.assertIsNone(locate_wide_line(image, [8, 16, 408, 64]))
        self.assertIsNone(locate_wide_line(image, [8, 16, 60, 64]))
        ImageDraw.Draw(image).rectangle((0, 16, 419, 64), fill="black")
        self.assertIsNone(locate_wide_line(image, [8, 16, 408, 64]))

    def test_wide_multiline_proposal_never_silently_crops_to_one_row(self):
        for light in (False, True):
            image = Image.new("RGB", (820, 110), "#444444" if light else "white")
            draw = ImageDraw.Draw(image)
            for y in (25, 65):
                for x in range(35, 770, 22):
                    draw.rectangle((x, y, x + 11, y + 17), fill="white" if light else "black")
            box = [10, 10, 810, 100]
            for score in (.12, .6):
                with self.subTest(light=light, score=score), \
                     patch("vision.characters.load_magi", return_value=object()), \
                     patch("vision.detection._text_proposals", return_value=[{"box": box, "score": score}]):
                    regions = detect_page_text(image, object(), Config())
                self.assertEqual(regions[0]["box"], box)
                self.assertNotIn("ocr_segments", regions[0])
                self.assertIn("多行", regions[0]["detection_review"])

    def test_tiny_endpoint_comma_period_and_detached_exclamation_dot_are_retained(self):
        for light in (False, True):
            for punctuation in ("period", "exclamation"):
                with self.subTest(light=light, punctuation=punctuation):
                    image = self.line_image(light)
                    draw = ImageDraw.Draw(image)
                    ink = "white" if light else "black"
                    # Five-pixel comma and three-pixel period are intentionally
                    # below the primary six-pixel minimum glyph height.
                    draw.rectangle((18, 44, 20, 48), fill=ink)
                    if punctuation == "exclamation":
                        draw.rectangle((387, 29, 389, 41), fill=ink)
                    draw.rectangle((387, 47, 389, 49), fill=ink)
                    result = locate_wide_line(image, [8, 16, 408, 64])
                    self.assertIsNotNone(result)
                    box = result["box"]
                    self.assertLessEqual(box[0], 18)
                    self.assertGreaterEqual(box[2], 390)
                    self.assertGreaterEqual(box[3], 50)
                    self.assertEqual(result["ocr_segments"][0][0], box[0])
                    self.assertEqual(result["ocr_segments"][-1][2], box[2])

    def test_endpoint_recovery_does_not_include_long_background_lines(self):
        image = self.line_image()
        before = locate_wide_line(image, [8, 16, 408, 64])["box"]
        ImageDraw.Draw(image).rectangle((378, 44, 401, 45), fill="black")
        ImageDraw.Draw(image).rectangle((10, 44, 13, 46), fill="black")
        after = locate_wide_line(image, [8, 16, 408, 64])["box"]
        self.assertEqual(after, before)

    def test_one_forward_keeps_high_and_reviews_only_localized_low_wide_line(self):
        image = self.line_image(True)
        proposals = [{"box": [10, 1, 20, 15], "score": .6},
                     {"box": [8, 16, 408, 64], "score": .12},
                     {"box": [200, 5, 220, 20], "score": .14}]
        with patch("vision.characters.load_magi", return_value=object()), patch("vision.detection._text_proposals", return_value=proposals) as infer:
            regions = detect_page_text(image, object(), Config())
        infer.assert_called_once()
        self.assertEqual(infer.call_args.kwargs["threshold"], .1)
        self.assertEqual(len(regions), 2)
        low = next(region for region in regions if region["score"] < .3)
        self.assertTrue(low["detection_review"])
        self.assertIn("ocr_segments", low)
        block = add_text_regions([], [low], image.size)[0]
        self.assertEqual(block.metadata["ocr_segments"], low["ocr_segments"])
        self.assertTrue(block.metadata["detection_review"])
        self.assertIsInstance(block.metadata["detection_review"], str)
        self.assertEqual(block.metadata["detection_score"], .12)

    def test_split_and_boolean_candidate_review_are_readable_strings(self):
        region = {"box": [0, 0, 90, 30], "score": .5,
                  "ocr_split_review": True, "detection_review": True}
        block = add_text_regions([], [region], (90, 30))[0]
        self.assertIsInstance(block.metadata["detection_review"], str)
        self.assertIn("切分点", block.metadata["detection_review"])
        self.assertIn("OCR", block.metadata["detection_review"])

    def test_overlapping_low_regions_are_not_rechecked_and_work_is_bounded(self):
        image = Image.new("RGB", (1000, 600))
        high = {"box": [10, 10, 90, 70], "score": .6}
        overlap = {"box": [0, 10, 600, 70], "score": .2}
        lows = [{"box": [0, y, 600, y + 30], "score": .15} for y in range(100, 500, 50)]
        with patch("vision.characters.load_magi", return_value=object()), patch("vision.detection._text_proposals", return_value=[high, overlap] + lows), patch("vision.detection.locate_wide_line", return_value=None) as locate:
            self.assertEqual(detect_page_text(image, object(), Config()), [high])
        self.assertEqual(locate.call_count, 5)  # One high region plus four low proposals.
        self.assertNotIn(overlap["box"], [call.args[1] for call in locate.call_args_list])

    def test_segment_ocr_uses_one_model_and_returns_one_block(self):
        image = Image.new("RGB", (90, 30))
        segments = [[0, 0, 30, 30], [30, 0, 60, 30], [60, 0, 90, 30]]
        block = TextBlock((0, 0, 90, 30), metadata={"kind": "free_text", "ocr_segments": segments})
        mocr, acquire = Mock(side_effect=["前半", "中間", "後半"]), Mock()
        acquire.return_value = mocr
        with patch("vision.detection.shrink_crop_to_text") as shrink:
            result = run_ocr(image, [block], pool=SimpleNamespace(acquire=acquire), config=Config())
        acquire.assert_called_once()
        shrink.assert_not_called()
        self.assertEqual(result, [block])
        self.assertEqual(block.source_text, "前半中間後半")
        self.assertEqual([call.args[0].size for call in mocr.call_args_list], [(30, 30)] * 3)

    def test_segment_cancellation_propagates_before_next_ocr(self):
        image = Image.new("RGB", (60, 30))
        block = TextBlock((0, 0, 60, 30), metadata={"ocr_segments": [[0, 0, 30, 30], [30, 0, 60, 30]]})
        mocr, cancel = Mock(return_value="前半"), Mock(side_effect=[None, None, Cancelled()])
        with self.assertRaises(Cancelled):
            run_ocr(image, [block], pool=SimpleNamespace(acquire=lambda *args: mocr), config=Config(), cancel=cancel)
        mocr.assert_called_once()
        self.assertEqual(block.source_text, "")

    def test_empty_or_out_of_page_segment_never_becomes_partial_success(self):
        for segments, replies in [([[0, 0, 30, 30], [30, 0, 60, 30]], ["前半", ""]),
                                  ([[0, 0, 70, 30]], ["unused"])]:
            with self.subTest(segments=segments):
                block = TextBlock((0, 0, 60, 30), metadata={"ocr_segments": segments})
                mocr = Mock(side_effect=replies)
                run_ocr(Image.new("RGB", (60, 30)), [block],
                        pool=SimpleNamespace(acquire=lambda *args: mocr), config=Config())
                self.assertEqual(block.status, "failed")
                self.assertEqual(block.source_text, "")
                self.assertTrue(block.error)


class BoxAuditTests(unittest.TestCase):
    def test_relations_use_original_snapshot_and_are_order_independent(self):
        blocks = [TextBlock((10, 10, 30, 50), block_id="a"),
                  TextBlock((60, 10, 80, 50), block_id="b")]
        regions = [{"box": [12, 12, 28, 48], "score": .9},
                   {"box": [5, 5, 35, 55], "score": .8},
                   {"box": [0, 0, 90, 60], "score": .7},
                   {"box": [25, 40, 45, 70], "score": .6},
                   {"box": [40, 70, 60, 90], "score": .5}]
        before = deepcopy((blocks, regions))
        entries = audit_text_regions(blocks, regions, (100, 100))
        by_box = {tuple(entry["crop_box"]): entry for entry in entries}
        self.assertEqual(by_box[(12, 12, 28, 48)]["classification"], "covered")
        expansion = by_box[(5, 5, 35, 55)]
        self.assertEqual(expansion["classification"], "expansion_candidate")
        self.assertEqual(expansion["related_ids"], ["a"])
        self.assertEqual(expansion["overlaps"][0]["block_coverage"], 1)
        self.assertEqual(expansion["overlaps"][0]["intersection_area"], 800)
        self.assertEqual(by_box[(0, 0, 90, 60)]["classification"], "conflict")
        self.assertEqual(by_box[(0, 0, 90, 60)]["related_ids"], ["a", "b"])
        self.assertEqual(by_box[(25, 40, 45, 70)]["classification"], "conflict")
        self.assertEqual(by_box[(40, 70, 60, 90)]["classification"], "new")
        self.assertEqual(entries, audit_text_regions(list(reversed(blocks)), list(reversed(regions)), (100, 100)))
        self.assertEqual((blocks, regions), before)
        self.assertEqual(json.loads(json.dumps(entries, allow_nan=False)), entries)

    def test_invalid_coordinates_rejected_and_outside_regions_match_detector_clipping(self):
        for box in ([float("nan"), 0, 10, 10], [0, 0, float("inf"), 10],
                    [0, 0, 0, 10], [10, 0, 5, 10], [False, 0, 10, 10], [0, 0, 10]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                audit_text_regions([], [{"box": box, "score": .5}], (100, 100))
        entries = audit_text_regions([], [{"box": [-2.5, 2.4, 11.1, 10.6], "score": .5},
                                          {"box": [120, 0, 140, 30], "score": .8}], (100, 100))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["crop_box"], [0, 2, 12, 11])
        with self.assertRaisesRegex(ValueError, "score"):
            audit_text_regions([], [{"box": [0, 0, 10, 10], "score": float("nan")}], (100, 100))

    def test_detection_default_skips_audit_and_opt_in_preserves_boxes_and_model_calls(self):
        image = Image.new("RGB", (100, 100))
        proposals = [{"box": [5, 5, 35, 55], "score": .8},
                     {"box": [60, 65, 80, 90], "score": .6}]
        coords, model = Mock(), Mock()
        coords.detach.return_value.cpu.return_value.tolist.return_value = [[10, 10, 30, 50]]
        model.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(xyxy=coords))]
        pool = SimpleNamespace(device="cuda:0", acquire=Mock(return_value=model))
        with TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            path.touch()
            config = Config(model_path=str(path))
            with patch("vision.detection.detect_page_text", return_value=proposals) as page_text, \
                    patch("vision.detection.audit_text_regions") as audit:
                baseline = detect(image, pool=pool, config=config)
            audit.assert_not_called()
            page_text.assert_called_once()
            self.assertTrue(all("ocr_box_audit" not in block.metadata for block in baseline))
            config.ocr_candidate_review = True
            with patch("vision.detection.detect_page_text", return_value=proposals) as page_text:
                reviewed = detect(image, pool=pool, config=config)
            page_text.assert_called_once()
        self.assertEqual(model.predict.call_count, 2)  # One detector call per invocation, no preview calls.
        self.assertEqual([(b.box, b.block_id, b.reading_order) for b in reviewed],
                         [(b.box, b.block_id, b.reading_order) for b in baseline])
        self.assertEqual([entry["classification"] for entry in reviewed[0].metadata["ocr_box_audit"]],
                         ["expansion_candidate"])
        self.assertTrue(all(block.metadata["ocr_box_audit"] for block in reviewed))

    def test_disabled_free_text_does_not_add_magi_inference_for_audit(self):
        model = Mock()
        model.predict.return_value = []
        pool = SimpleNamespace(device="cuda:0", acquire=Mock(return_value=model))
        with TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            path.touch()
            config = Config(model_path=str(path), detect_free_text=False)
            config.ocr_candidate_review = True
            with patch("vision.detection.detect_page_text") as page_text:
                self.assertEqual(detect(Image.new("RGB", (10, 10)), pool=pool, config=config), [])
            page_text.assert_not_called()


class OCRCandidateTests(unittest.TestCase):
    def test_exact_crop_is_recognized_once_without_padding_shrink_or_source_changes(self):
        image = Image.new("RGB", (80, 60), "white")
        ImageDraw.Draw(image).rectangle((11, 7, 39, 29), fill="black")
        block = TextBlock((15, 10, 25, 20), source_text="手動修正", translated_text="Edited",
                          block_id="b1", status="done", safe_box=(16, 11, 24, 19),
                          metadata={"manual_edit": True, "ocr_segments": [[15, 10, 25, 20]]})
        before, pixels = deepcopy(block), image.tobytes()
        mocr = Mock(return_value="  新しい原文\n")
        pool = SimpleNamespace(acquire=Mock(return_value=mocr))
        config = Config()
        with patch("vision.detection.shrink_crop_to_text") as shrink, \
                patch("vision.detection._acquire_ocr", wraps=vision._acquire_ocr) as acquire:
            result = preview_ocr_crop(image, block, [11, 7, 40, 30], pool=pool, config=config)
        shrink.assert_not_called()
        pool.acquire.assert_called_once()
        mocr.assert_called_once()
        crop = mocr.call_args.args[0]
        self.assertEqual(crop.size, (29, 23))
        self.assertEqual(crop.tobytes(), image.crop((11, 7, 40, 30)).tobytes())
        self.assertEqual(block, before)
        self.assertEqual(image.tobytes(), pixels)
        self.assertEqual(result["raw_text"], "  新しい原文\n")
        self.assertEqual(result["text"], "新しい原文")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["crop_box"], [11, 7, 40, 30])
        self.assertEqual(result["crop_sha256"], hashlib.sha256(b"RGB\x0029\x0023\x00" + crop.tobytes()).hexdigest())
        self.assertIsNone(result["confidence"])
        self.assertGreaterEqual(result["seconds"], 0)
        self.assertTrue(acquire.call_args.args[1].local_files_only)
        self.assertFalse(config.local_files_only)
        json.dumps(result, allow_nan=False)

    def test_bad_crop_fails_before_loading_model(self):
        pool = SimpleNamespace(acquire=Mock())
        for box in ([0, 0, 2.0, 2], [False, 0, 2, 2], [0, 0, float("inf"), 2],
                    [-1, 0, 2, 2], [0, 0, 11, 2], [2, 0, 2, 2], [0, 0, 2], "0,0,2,2"):
            with self.subTest(box=box), self.assertRaises(ValueError):
                preview_ocr_crop(Image.new("RGB", (10, 10)), TextBlock((0, 0, 2, 2)), box,
                                 pool=pool, config=Config())
        pool.acquire.assert_not_called()
        self.assertEqual(validate_ocr_crop([0, 0, 10, 10], (10, 10)), (0, 0, 10, 10))

    def test_empty_and_inference_error_are_records_without_retry(self):
        for reply, expected in [(" \n", "empty"), (ValueError("bad OCR"), "failed"), (None, "failed")]:
            with self.subTest(reply=reply):
                mocr = Mock(side_effect=reply) if isinstance(reply, Exception) else Mock(return_value=reply)
                block = TextBlock((0, 0, 5, 5), source_text="keep")
                result = preview_ocr_crop(Image.new("RGB", (10, 10)), block, (0, 0, 5, 5),
                                          pool=SimpleNamespace(acquire=Mock(return_value=mocr)), config=Config())
                self.assertEqual(result["status"], expected)
                self.assertTrue(result["error"])
                self.assertEqual(block.source_text, "keep")
                mocr.assert_called_once()

    def test_cancellation_and_oom_propagate_without_retry(self):
        for cancel_sequence, calls in [([Cancelled()], 0), ([None, Cancelled()], 0),
                                        ([None, None, Cancelled()], 1)]:
            mocr, acquire = Mock(return_value="text"), Mock()
            acquire.return_value = mocr
            with self.subTest(calls=calls), self.assertRaises(Cancelled):
                preview_ocr_crop(Image.new("RGB", (10, 10)), TextBlock((0, 0, 5, 5)), (0, 0, 5, 5),
                                 pool=SimpleNamespace(acquire=acquire), config=Config(),
                                 cancel=Mock(side_effect=cancel_sequence))
            self.assertEqual(mocr.call_count, calls)
            if len(cancel_sequence) == 1:
                acquire.assert_not_called()
        mocr = Mock(side_effect=RuntimeError("CUDA out of memory"))
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            preview_ocr_crop(Image.new("RGB", (10, 10)), TextBlock((0, 0, 5, 5)), (0, 0, 5, 5),
                             pool=SimpleNamespace(acquire=Mock(return_value=mocr)), config=Config())
        mocr.assert_called_once()

    def test_local_only_lookup_is_enforced_when_cache_misses(self):
        hub, manga, torch = Mock(), Mock(), Mock()
        hub.snapshot_download.side_effect = FileNotFoundError("no local OCR")
        torch.cuda.device.return_value = nullcontext()
        pool = SimpleNamespace(device="cuda:0", acquire=lambda name, load, move: load())
        with patch("vision.detection.require", side_effect=lambda name, *args: {
                "huggingface_hub": hub, "manga_ocr": manga, "torch": torch}[name]):
            result = preview_ocr_crop(Image.new("RGB", (10, 10)), TextBlock((0, 0, 5, 5)),
                                     (0, 0, 5, 5), pool=pool, config=Config(local_files_only=False))
        self.assertEqual(result["status"], "failed")
        hub.snapshot_download.assert_called_once()
        self.assertTrue(hub.snapshot_download.call_args.kwargs["local_files_only"])
        manga.MangaOcr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
