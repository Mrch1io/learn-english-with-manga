"""CPU-only editing regression tests; no detection, OCR or GPU model loading.

Run: .venv\\Scripts\\python.exe -m unittest discover -s tests -p test_editing.py -v
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from core.runtime import Cancelled
from pipeline.editing import PageEditorSession
from vision import graphics
from pipeline.batch import file_hash
from tests.editing_fixture import EditingFixture


class PageEditingTests(EditingFixture, unittest.TestCase):
    def test_legacy_page_keeps_qwen_and_new_page_keeps_selected_translator(self):
        for saved_backend in (None, "vntl", "qwen"):
            with self.subTest(saved_backend=saved_backend):
                record = self.make_result()
                if saved_backend is None:
                    record["config"].pop("translation_backend")
                    record["config"].pop("translation_mode")
                else:
                    record["config"]["translation_backend"] = saved_backend
                self.record_path.write_text(json.dumps(record), encoding="utf-8")
                session = self.open_session()
                self.assertEqual(session.config.translation_backend, saved_backend or "qwen")
                self.assertEqual(session.config.translation_mode, "auto" if saved_backend else "individual")

    def test_clean_reflow_uses_no_models_and_preserves_neighbor(self):
        self.make_result(inferred_regions=True)
        session = self.open_session()
        before_pixels = self.output_pixels()
        neighbor = deepcopy(session.record["blocks"][1])
        clean_bytes = self.clean_path.read_bytes()
        with patch("pipeline.editing.inpaint", side_effect=AssertionError("Already cleaned")), \
             patch("pipeline.editing.translate", side_effect=AssertionError("Reflow must not translate")), \
             patch("vision.detection.detect", side_effect=AssertionError("Reflow must not detect")), \
             patch("vision.detection.run_ocr", side_effect=AssertionError("Reflow must not OCR")):
            result = session.rerender("left", "Corrected words")
        np.testing.assert_array_equal(self.output_pixels()[:, 320:], before_pixels[:, 320:])
        self.assertEqual(result["blocks"][1], neighbor)
        self.assertEqual(self.clean_path.read_bytes(), clean_bytes)
        self.assertEqual(result["blocks"][0]["translated_text"], "Corrected words")
        self.assertEqual(result["output_sha256"], file_hash(self.output_path))
        self.assertEqual(result["clean_image"]["sha256"], file_hash(self.clean_path))
        self.pool.acquire.assert_not_called()

    def fake_candidate(self, *, text="校正した原文", status="ok"):
        return {"engine": "manga-ocr", "model": "kha-white/manga-ocr-base",
                "crop_box": [0, 10, 320, 250], "crop_sha256": "fixture-crop",
                "raw_text": text, "text": text, "status": status, "error": "",
                "seconds": 0.01, "confidence": None}

    def saved_candidate(self, session, **kwargs):
        with patch("pipeline.editing.preview_ocr_crop", return_value=self.fake_candidate(**kwargs)):
            return session.preview_ocr("left", (0, 10, 320, 250), enabled=True)

    def test_ocr_preview_disabled_and_invalid_inputs_make_no_request_or_write(self):
        self.make_result()
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        cases = [("left", [0, 0, 640, 260], False), ("missing", [0, 0, 640, 260], True),
                 ("left", [-1, 0, 20, 30], True), ("left", [True, 0, 20, 30], True)]
        with patch("pipeline.editing.preview_ocr_crop") as preview:
            for block_id, crop, enabled in cases:
                with self.subTest(block_id=block_id, crop=crop, enabled=enabled), self.assertRaises(ValueError):
                    session.preview_ocr(block_id, crop, enabled=enabled)
                self.assert_uncommitted(session, before_disk, before_record)
            preview.assert_not_called()
        session.original = None
        with self.assertRaisesRegex(ValueError, "原图"):
            session.preview_ocr("left", [0, 0, 30, 30], enabled=True)

    def test_ocr_preview_only_persists_candidate_and_preserves_manual_revision_and_pixels(self):
        self.make_result()
        session = self.open_session()
        session.rerender("left", "Manually corrected words")
        before = deepcopy(session.record)
        clean_bytes, output_bytes = self.clean_path.read_bytes(), self.output_path.read_bytes()
        with patch("pipeline.editing.preview_ocr_crop", return_value=self.fake_candidate()) as preview, \
             patch("pipeline.editing.translate", side_effect=AssertionError("Preview must not translate")), \
             patch("pipeline.editing.render", side_effect=AssertionError("Preview must not render")), \
             patch("pipeline.editing.inpaint", side_effect=AssertionError("Preview must not clean")):
            candidate = session.preview_ocr("left", (0, 10, 320, 250), enabled=True)
        preview.assert_called_once()
        self.assertIs(preview.call_args.kwargs["pool"], self.pool)
        self.assertEqual(preview.call_args.args[2], (0, 10, 320, 250))
        self.assertEqual(session.record["edit_revision"], before["edit_revision"])
        self.assertEqual(session.record["last_edit"], before["last_edit"])
        self.assertEqual(session.record["blocks"][1], before["blocks"][1])
        for name in ("box", "safe_box", "source_text", "translated_text", "status"):
            self.assertEqual(session.record["blocks"][0][name], before["blocks"][0][name])
        self.assertEqual(self.clean_path.read_bytes(), clean_bytes)
        self.assertEqual(self.output_path.read_bytes(), output_bytes)
        self.assertTrue(session.record["ocr_candidate_edits"])
        self.assertFalse(candidate["accepted"])
        reopened = self.open_session()
        self.assertEqual(reopened.blocks[0].metadata["ocr_candidates"], [candidate])

    def test_ocr_preview_oom_cancel_and_commit_failure_leave_saved_state_intact(self):
        self.make_result()
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        for exc in (Cancelled("cancelled"), RuntimeError("CUDA out of memory")):
            with self.subTest(exc=exc), patch("pipeline.editing.preview_ocr_crop", side_effect=exc) as preview:
                with self.assertRaises(type(exc)):
                    session.preview_ocr("left", [0, 10, 320, 250], enabled=True)
                preview.assert_called_once()
                self.assert_uncommitted(session, before_disk, before_record)
        self.pool.offload.assert_not_called()
        with patch("pipeline.editing.preview_ocr_crop", return_value=self.fake_candidate()), \
             patch("pipeline.editing._commit_files", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                session.preview_ocr("left", [0, 10, 320, 250], enabled=True)
        self.assert_uncommitted(session, before_disk, before_record)

    def test_ocr_adoption_requires_enabled_successful_fresh_candidate(self):
        self.make_result()
        session = self.open_session()
        candidate = self.saved_candidate(session)
        with patch("pipeline.editing.translate") as translate:
            with self.assertRaises(ValueError):
                session.apply_ocr_candidate("left", candidate["candidate_id"])
            for block_id, candidate_id in (("right", candidate["candidate_id"]), ("left", "missing")):
                with self.assertRaises(ValueError):
                    session.apply_ocr_candidate(block_id, candidate_id, enabled=True)
            for status in ("empty", "failed"):
                invalid = self.saved_candidate(session, text="", status=status)
                with self.assertRaises(ValueError):
                    session.apply_ocr_candidate("left", invalid["candidate_id"], enabled=True)
            session.rerender("left", "A newer manual edit")
            before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
            with self.assertRaisesRegex(ValueError, "过期"):
                session.apply_ocr_candidate("left", candidate["candidate_id"], enabled=True)
            self.assert_uncommitted(session, before_disk, before_record)
            translate.assert_not_called()

    def test_ocr_adoption_retranslates_only_target_without_expanding_cleanup_or_neighbor(self):
        self.make_result()
        session = self.open_session()
        candidate = self.saved_candidate(session)
        before = deepcopy(session.record)
        clean_bytes, before_pixels = self.clean_path.read_bytes(), self.output_pixels()
        def translated(blocks, **kwargs):
            self.assertEqual(kwargs["target_ids"], {"left"})
            self.assertEqual(blocks[0].source_text, candidate["text"])
            blocks[0].translated_text, blocks[0].status = "New words", "translated"
            return {"generation_calls": 1}
        with patch("pipeline.editing.translate", side_effect=translated) as translate, \
             patch("pipeline.editing.preview_ocr_crop", side_effect=AssertionError("Adoption must not OCR")), \
             patch("pipeline.editing.inpaint", side_effect=AssertionError("Already cleaned")):
            result = session.apply_ocr_candidate("left", candidate["candidate_id"], enabled=True)
        translate.assert_called_once()
        self.assertEqual(result["blocks"][1], before["blocks"][1])
        self.assertEqual(result["blocks"][0]["box"], before["blocks"][0]["box"])
        self.assertEqual(result["blocks"][0]["safe_box"], before["blocks"][0]["safe_box"])
        metadata = result["blocks"][0]["metadata"]
        self.assertEqual(metadata["ocr_crop_box"], candidate["crop_box"])
        self.assertTrue(metadata["ocr_candidates"][0]["accepted"])
        self.assertEqual(metadata["original_source_text"], before["blocks"][0]["source_text"])
        self.assertNotIn("ocr_review", metadata)
        self.assertEqual(self.clean_path.read_bytes(), clean_bytes)
        np.testing.assert_array_equal(self.output_pixels()[:, 320:], before_pixels[:, 320:])
        with self.assertRaises(ValueError):
            session.apply_ocr_candidate("left", candidate["candidate_id"], enabled=True)

    def test_failed_ocr_candidate_translation_does_not_accept_or_replace_anything(self):
        self.make_result()
        session = self.open_session()
        candidate = self.saved_candidate(session)
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        def failed(blocks, **kwargs):
            blocks[0].status, blocks[0].error = "failed", "candidate translation failure"
            return {"generation_calls": 1}
        with patch("pipeline.editing.translate", side_effect=failed), self.assertRaisesRegex(ValueError, "candidate translation failure"):
            session.apply_ocr_candidate("left", candidate["candidate_id"], enabled=True)
        self.assert_uncommitted(session, before_disk, before_record)

    def test_ocr_adoption_does_not_restart_translation_retry_budget_after_oom(self):
        self.make_result()
        session = self.open_session()
        candidate = self.saved_candidate(session)
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        with patch("pipeline.editing.translate", side_effect=RuntimeError("CUDA out of memory")) as translate:
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                session.apply_ocr_candidate("left", candidate["candidate_id"], enabled=True)
        translate.assert_called_once()
        self.pool.offload.assert_not_called()
        self.assert_uncommitted(session, before_disk, before_record)

    def test_saved_candidate_listing_is_read_only_reusable_and_reports_staleness(self):
        self.make_result()
        session = self.open_session()
        candidate = self.saved_candidate(session)
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        with patch("pipeline.editing.preview_ocr_crop", side_effect=AssertionError("Listing must not OCR")), \
             patch("pipeline.editing.translate", side_effect=AssertionError("Listing must not translate")):
            rows = session.list_ocr_candidates("left")
            self.assertTrue(rows[0]["usable"])
            self.assertEqual(rows[0]["candidate_id"], candidate["candidate_id"])
            self.assertEqual(rows[0]["unavailable_reason"], "")
            rows[0]["text"] = "UI cannot mutate a saved candidate"
            self.assertEqual(session.list_ocr_candidates("left")[0]["text"], candidate["text"])
            self.assertEqual(session.list_ocr_candidates("right"), [])
            self.assert_uncommitted(session, before_disk, before_record)
            session.rerender("left", "A later manual correction")
            row = session.list_ocr_candidates("left")[0]
            self.assertFalse(row["usable"])
            self.assertIn("过期", row["unavailable_reason"])

    def test_saved_failed_accepted_and_malformed_candidates_are_not_reusable(self):
        self.make_result()
        session = self.open_session()
        good = self.saved_candidate(session)
        variants = [dict(good, accepted=True), dict(good, text=""), dict(good, status="failed"),
                    dict(good, crop_box=[-1, 0, 20, 20]), dict(good, candidate_id=""), None]
        session.blocks[0].metadata["ocr_candidates"] = variants
        with patch("pipeline.editing.translate") as translate:
            for row in session.list_ocr_candidates("left"):
                self.assertFalse(row["usable"])
                self.assertTrue(row["unavailable_reason"])
            with self.assertRaises(ValueError):
                session.apply_ocr_candidate("left", good["candidate_id"], enabled=True)
            translate.assert_not_called()

    def test_local_comparison_returns_aligned_bounded_images_without_saving_or_models(self):
        self.make_result()
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        with patch("pipeline.editing.preview_ocr_crop", side_effect=AssertionError("Comparison must not OCR")), \
             patch("pipeline.editing.translate", side_effect=AssertionError("Comparison must not translate")), \
             patch("pipeline.editing.render", side_effect=AssertionError("Comparison must not render")):
            for box, size in [([20, 25, 300, 235], (768, 576)),
                              ([632, 255, 640, 260], (24, 15)),
                              ([0, 0, 640, 260], (768, 312))]:
                with self.subTest(box=box):
                    result = session.preview_region("left", box)
                    self.assertEqual(result["crop_box"], box)
                    self.assertEqual(result["original"].size, size)
                    self.assertEqual(result["translated"].size, size)
                    for image, original in ((result["original"], self.original),
                                            (result["translated"], self.fixture_output)):
                        # Separate crop then resize is the reference for spatial alignment.
                        expected = original.crop(box).resize(size, Image.Resampling.LANCZOS)
                        np.testing.assert_array_equal(image, expected)
            self.assertEqual(session.preview_region("left")["crop_box"], [20, 25, 300, 235])
        self.assert_uncommitted(session, before_disk, before_record)

    def test_local_comparison_uses_exif_corrected_original_coordinates(self):
        record = self.make_result()
        exif = Image.Exif()
        exif[274] = 6
        self.original.transpose(Image.Transpose.ROTATE_90).save(self.input_path, exif=exif)
        record["input_sha256"] = file_hash(self.input_path)
        self.record_path.write_text(json.dumps(record), encoding="utf-8")
        session = self.open_session()
        result = session.preview_region("left", [70, 100, 100, 120])
        expected = self.original.crop((70, 100, 100, 120)).resize((90, 60), Image.Resampling.LANCZOS)
        np.testing.assert_array_equal(result["original"], expected)
        self.pool.acquire.assert_not_called()

    def test_local_comparison_text_only_failure_and_cancel_leave_state_intact(self):
        self.make_reading_result()
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        self.assertIsNone(session.preview_region("left")["translated"])
        for block_id, box in (("missing", None), ("left", [0, 0, 0, 1]),
                              ("left", [0, 0, 700, 100]), ("left", [False, 0, 10, 10])):
            with self.assertRaises(ValueError):
                session.preview_region(block_id, box)
        with self.assertRaises(Cancelled):
            session.preview_region("left", cancel=Mock(side_effect=[None, Cancelled("cancel")]))
        session.original = None
        with self.assertRaisesRegex(ValueError, "原图"):
            session.preview_region("left")
        self.assert_uncommitted(session, before_disk, before_record)

    def test_recovery_reuses_one_cleanup_analysis_for_inpaint_and_composite(self):
        self.make_result(left_failed=True)
        session = self.open_session()
        with patch("vision.graphics._mask_patches", wraps=graphics._mask_patches) as analyze:
            session.rerender("left", "Recovered")
        analyze.assert_called_once()
        self.assertEqual([b.block_id for b in analyze.call_args.args[1]], ["left"])
        self.pool.acquire.assert_not_called()

    def test_only_reviewed_source_clears_recovered_ocr_warning(self):
        record = self.make_result()
        record["blocks"][0]["metadata"].update(
            detection_review="Check recovered text", ocr_split_review=True)
        self.record_path.write_text(json.dumps(record), encoding="utf-8")
        session = self.open_session()
        neighbor = deepcopy(session.record["blocks"][1])
        result = session.rerender("left", "Corrected words")
        self.assertEqual(result["status"], "partial")
        self.assertIn("detection_review", result["blocks"][0]["metadata"])
        def translated(blocks, **kwargs):
            self.assertEqual(kwargs["target_ids"], {"left"})
            blocks[0].translated_text, blocks[0].status = "Reviewed words", "translated"
            return {"generation_calls": 1}
        with patch("pipeline.editing.translate", side_effect=translated):
            result = session.retranslate("left", "校正した原文")
        self.assertEqual(result["status"], "done")
        self.assertNotIn("detection_review", result["blocks"][0]["metadata"])
        self.assertNotIn("ocr_split_review", result["blocks"][0]["metadata"])
        self.assertEqual(result["blocks"][1], neighbor)
        self.pool.acquire.assert_not_called()

    def test_repeated_long_short_long_edits_do_not_accumulate_old_letters(self):
        self.make_result()
        session = self.open_session()
        text = "A longer sentence wraps across several lines."
        session.rerender("left", text)
        first_long_pixels = self.output_pixels()
        session.rerender("left", "Hi")
        short_pixels = self.output_pixels()
        self.assertFalse(np.array_equal(short_pixels[:, :320], first_long_pixels[:, :320]))
        session.rerender("left", text)
        np.testing.assert_array_equal(self.output_pixels(), first_long_pixels)
        np.testing.assert_array_equal(self.output_pixels()[:, 320:], np.asarray(self.fixture_output)[:, 320:])
        self.assertEqual(session.record["edit_revision"], 3)
        reopened = self.open_session()
        self.assertEqual(reopened.record["blocks"][0]["translated_text"], text)

    def test_retranslation_targets_one_id_and_retains_neighbor_context_and_original_ocr(self):
        self.make_result()
        session = self.open_session()
        neighbor = deepcopy(session.record["blocks"][1])
        before_pixels = self.output_pixels()
        seen_sources = []
        def fake_translate(blocks, **kwargs):
            self.assertEqual(kwargs["target_ids"], {"left"})
            self.assertFalse(kwargs["record_page"])
            self.assertFalse(kwargs["config"].extract_terms)
            self.assertEqual(kwargs["config"].translation_mode, "individual")
            self.assertEqual(len(blocks), 2)
            self.assertEqual(blocks[1].source_text, "neighbor OCR source")
            self.assertEqual(blocks[1].translated_text, "Neighbor stays")
            target = next(block for block in blocks if block.block_id == "left")
            seen_sources.append(target.source_text)
            target.translated_text = "Corrected meaning"
            target.status, target.error = "translated", ""
            return {"generation_calls": 1, "candidate_calls": 0}
        with patch("pipeline.editing.translate", side_effect=fake_translate) as translate, \
             patch("pipeline.editing.inpaint", side_effect=AssertionError("No cleanup needed")), \
             patch("vision.detection.detect", side_effect=AssertionError("No detection")), \
             patch("vision.detection.run_ocr", side_effect=AssertionError("No OCR")):
            session.retranslate("left", "  corrected OCR source  ")
            result = session.retranslate("left", "second correction")
        self.assertEqual(translate.call_count, 2)
        self.assertEqual(seen_sources, ["corrected OCR source", "second correction"])
        target = result["blocks"][0]
        self.assertEqual(target["source_text"], "second correction")
        self.assertEqual(target["metadata"]["original_source_text"], "first OCR source")
        self.assertEqual(target["metadata"]["original_translated_text"], "Old words here")
        self.assertEqual(result["blocks"][1], neighbor)
        np.testing.assert_array_equal(self.output_pixels()[:, 320:], before_pixels[:, 320:])

    def test_legacy_result_migrates_from_original_without_detection_ocr_or_translation(self):
        self.make_result(with_clean=False)
        session = self.open_session()
        self.assertIn("editor_notice", session.record)
        before_neighbor = self.output_pixels()[:, 320:].copy()
        actual_inpaint = graphics.inpaint
        calls = []
        def inspect_inpaint(image, blocks, **kwargs):
            np.testing.assert_array_equal(image, self.original)
            calls.append({block.block_id for block in blocks})
            return actual_inpaint(image, blocks, **kwargs)
        with patch("pipeline.editing.inpaint", side_effect=inspect_inpaint), \
             patch("pipeline.editing.translate", side_effect=AssertionError("Migration must not translate")), \
             patch("vision.detection.detect", side_effect=AssertionError("Migration must not detect")), \
             patch("vision.detection.run_ocr", side_effect=AssertionError("Migration must not OCR")):
            result = session.rerender("left", "Migrated correction")
        self.assertEqual(calls, [{"left", "right"}])
        self.assertEqual(set(result["clean_image"]["cleaned_block_ids"]), {"left", "right"})
        self.assertTrue(session.clean_path.is_file())
        self.assertNotIn("editor_notice", result)
        np.testing.assert_array_equal(self.output_pixels()[:, 320:], before_neighbor)
        self.pool.acquire.assert_not_called()

    def test_failed_bubble_recovery_only_cleans_its_mask(self):
        self.make_result(left_failed=True)
        session = self.open_session()
        before_neighbor = self.output_pixels()[:, 320:].copy()
        before_clean_neighbor = np.asarray(session.clean)[:, 320:].copy()
        calls = []
        actual_inpaint = graphics.inpaint
        def inspect_inpaint(image, blocks, **kwargs):
            np.testing.assert_array_equal(image, self.original)
            calls.append([block.block_id for block in blocks])
            return actual_inpaint(image, blocks, **kwargs)
        with patch("pipeline.editing.inpaint", side_effect=inspect_inpaint):
            result = session.rerender("left", "Recovered")
        self.assertEqual(calls, [["left"]])
        self.assertEqual(set(result["clean_image"]["cleaned_block_ids"]), {"left", "right"})
        self.assertEqual(result["blocks"][0]["status"], "translated")
        np.testing.assert_array_equal(np.asarray(session.clean)[:, 320:], before_clean_neighbor)
        np.testing.assert_array_equal(self.output_pixels()[:, 320:], before_neighbor)
        self.assertFalse(np.array_equal(np.asarray(session.clean)[:, :320], np.asarray(self.original)[:, :320]))

    def test_empty_translation_intentionally_blanks_even_a_previously_failed_bubble(self):
        self.make_result(left_failed=True)
        session = self.open_session()
        before_neighbor = self.output_pixels()[:, 320:].copy()
        result = session.rerender("left", " \n ")
        target = result["blocks"][0]
        self.assertEqual(target["translated_text"], "")
        self.assertEqual(target["status"], "translated")
        self.assertTrue(target["metadata"]["manual_blank"])
        self.assertIn("left", result["clean_image"]["cleaned_block_ids"])
        np.testing.assert_array_equal(self.output_pixels()[:, :320], np.asarray(session.clean)[:, :320])
        np.testing.assert_array_equal(self.output_pixels()[:, 320:], before_neighbor)

    def test_overflow_missing_glyph_and_empty_ocr_leave_disk_and_session_unchanged(self):
        self.make_result()
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        operations = [lambda: session.rerender("left", "X" * 400),
                      lambda: session.rerender("left", "Missing \U0010ffff"),
                      lambda: session.retranslate("left", " \n ")]
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    operation()
                self.assert_uncommitted(session, before_disk, before_record)

    def test_cancellation_after_render_does_not_commit(self):
        self.make_result()
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        cancelled = threading.Event()
        def cancel():
            if cancelled.is_set():
                raise Cancelled("cancel after render")
        def render_then_cancel(*args, **kwargs):
            image = graphics.render(*args, **kwargs)
            cancelled.set()
            return image
        with patch("pipeline.editing.render", side_effect=render_then_cancel):
            with self.assertRaises(Cancelled):
                session.rerender("left", "Computed but not saved", cancel=cancel)
        self.assert_uncommitted(session, before_disk, before_record)

    def test_retranslation_oom_does_not_restart_budget_or_commit_partial_state(self):
        self.make_result()
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        before_blocks = deepcopy(session.blocks)
        def out_of_memory(blocks, **kwargs):
            target = next(block for block in blocks if block.block_id == "left")
            target.metadata["translation_attempts"] = 1
            target.translated_text = "incomplete generation"
            raise RuntimeError("CUDA out of memory during translation")
        with patch("pipeline.editing.translate", side_effect=out_of_memory) as translate:
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                session.retranslate("left", "修正した日本語")
        translate.assert_called_once()
        self.pool.offload.assert_not_called()
        self.assertEqual(session.blocks, before_blocks)
        self.assert_uncommitted(session, before_disk, before_record)

    def test_translation_failure_does_not_commit_corrected_ocr_or_translation(self):
        self.make_result()
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        def failed_translation(blocks, **kwargs):
            target = next(block for block in blocks if block.block_id == "left")
            target.translated_text = "incomplete generation"
            target.status, target.error = "failed", "injected generation failure"
            return {"generation_calls": 1}
        with patch("pipeline.editing.translate", side_effect=failed_translation):
            with self.assertRaisesRegex(ValueError, "injected generation failure"):
                session.retranslate("left", "Corrected OCR that must not commit")
        self.assert_uncommitted(session, before_disk, before_record)

    def test_commit_io_failure_rolls_back_image_clean_base_and_json(self):
        self.make_result(left_failed=True)
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        real_replace = os.replace
        failed_once = False
        def fail_sidecar_install(source, destination):
            nonlocal failed_once
            if Path(destination).resolve() == self.record_path.resolve() and not failed_once:
                failed_once = True
                raise OSError("injected JSON replacement failure")
            return real_replace(source, destination)
        with patch("pipeline.editing.os.replace", side_effect=fail_sidecar_install):
            with self.assertRaisesRegex(OSError, "injected JSON"):
                session.rerender("left", "Recovered but commit fails")
        self.assertTrue(failed_once)
        self.assert_uncommitted(session, before_disk, before_record)

    def test_external_input_change_is_rejected_on_open_and_save(self):
        self.make_result()
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        self.input_path.write_bytes(self.input_path.read_bytes() + b"external-change")
        with self.assertRaisesRegex(ValueError, "Original image changed"):
            PageEditorSession(self.record_path, pool=self.pool)
        with self.assertRaisesRegex(ValueError, "Original image changed"):
            session.rerender("left", "Should not save")
        self.assert_uncommitted(session, before_disk, before_record)

    def test_corrupt_clean_base_is_rejected_on_open_and_save(self):
        self.make_result()
        session = self.open_session()
        self.clean_path.write_bytes(self.clean_path.read_bytes() + b"external-change")
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        with self.assertRaisesRegex(ValueError, "Clean base checksum mismatch"):
            PageEditorSession(self.record_path, pool=self.pool)
        with self.assertRaisesRegex(ValueError, "Clean base changed"):
            session.rerender("left", "Should not save")
        self.assert_uncommitted(session, before_disk, before_record)

    def test_external_final_image_change_is_not_overwritten(self):
        self.make_result()
        session = self.open_session()
        self.output_path.write_bytes(self.output_path.read_bytes() + b"external-change")
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        with self.assertRaisesRegex(ValueError, "Page changed outside"):
            session.rerender("left", "Should not save")
        self.assert_uncommitted(session, before_disk, before_record)

    def test_valid_clean_base_allows_reflow_when_original_is_missing(self):
        self.make_result()
        self.input_path.unlink()
        session = self.open_session()
        self.assertIsNone(session.original)
        before_neighbor = self.output_pixels()[:, 320:].copy()
        with patch("pipeline.editing.inpaint", side_effect=AssertionError("Never clean final image")), \
             patch("pipeline.editing.translate", side_effect=AssertionError("No translation required")):
            result = session.rerender("left", "Original missing, base valid")
        self.assertEqual(result["blocks"][0]["translated_text"], "Original missing, base valid")
        np.testing.assert_array_equal(self.output_pixels()[:, 320:], before_neighbor)

    def test_missing_original_prevents_recovery_of_an_uncleaned_bubble(self):
        self.make_result(left_failed=True)
        self.input_path.unlink()
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        with self.assertRaisesRegex(ValueError, "never cleaned"):
            session.rerender("left", "Cannot recover without original")
        self.assert_uncommitted(session, before_disk, before_record)

    def test_reading_edits_are_json_only_and_export_needs_no_detection_or_translation(self):
        original = self.make_reading_result()
        session = self.open_session()
        with patch("pipeline.editing.inpaint", side_effect=AssertionError("Reading edits do not clean")), \
             patch("pipeline.editing.prepare_layout", side_effect=AssertionError("Reading edits do not lay out")), \
             patch("pipeline.editing.translate", side_effect=AssertionError("Manual correction does not translate")):
            result = session.rerender("left", "Corrected reading text")
        self.assertFalse(self.output_path.exists())
        self.assertFalse(self.clean_path.exists())
        self.assertFalse(result["exported"])
        self.assertEqual(result["blocks"][1], original["blocks"][1])
        self.assertEqual(result["config_fingerprint"], original["config_fingerprint"])
        self.assertEqual(result["terms_fingerprint"], original["terms_fingerprint"])
        with patch("vision.detection.detect", side_effect=AssertionError("No detection")), \
             patch("vision.detection.run_ocr", side_effect=AssertionError("No OCR")), \
             patch("pipeline.editing.translate", side_effect=AssertionError("Export cannot translate")):
            exported = session.export_image()
        self.assertTrue(exported["exported"])
        self.assertTrue(self.output_path.is_file())
        self.assertTrue(self.clean_path.is_file())
        self.assertEqual(exported["output_sha256"], file_hash(self.output_path))
        self.assertEqual(exported["blocks"][0]["translated_text"], "Corrected reading text")
        reopened = self.open_session()
        with patch("pipeline.editing.inpaint", side_effect=AssertionError("Reuse verified clean image")):
            repeated = reopened.export_image()
        self.assertEqual(repeated["output_sha256"], exported["output_sha256"])

    def test_reading_accepts_long_text_but_failed_export_preserves_json(self):
        self.make_reading_result()
        session = self.open_session()
        session.rerender("left", "W" * 300)
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        with self.assertRaisesRegex(ValueError, "Export layout failed"):
            session.export_image()
        self.assert_uncommitted(session, before_disk, before_record)

    def test_reopened_neighbor_records_match_without_tuple_list_differences(self):
        self.make_result(inferred_regions=True)
        first = self.open_session()
        edited = first.rerender("left", "Human correction")
        second = self.open_session()
        result = second.rerender("right", "Another correction")
        self.assertEqual(edited["blocks"][0], result["blocks"][0])
        self.assertEqual(edited["config_fingerprint"], "fixture-fingerprint")
        self.assertEqual(result["terms_fingerprint"], "fixture-terms")

    def test_manual_cleanup_removes_only_selected_base_residual_and_persists(self):
        record = self.make_result()
        with Image.open(self.clean_path) as image:
            clean = image.convert("RGB")
        ImageDraw.Draw(clean).rectangle((80, 80, 92, 89), fill=(160, 160, 160))
        clean.save(self.clean_path)
        record["clean_image"]["sha256"] = file_hash(self.clean_path)
        self.record_path.write_text(json.dumps(record), encoding="utf-8")
        session = self.open_session()
        previous_neighbor = self.output_pixels()[:, 320:].copy()
        previous_blocks = deepcopy(session.record["blocks"])
        with patch("pipeline.editing.inpaint", side_effect=AssertionError("Already clean base")), \
             patch("pipeline.editing.translate", side_effect=AssertionError("Manual cleanup never translates")), \
             patch("vision.detection.detect", side_effect=AssertionError("No detection")), \
             patch("vision.detection.run_ocr", side_effect=AssertionError("No OCR")):
            result = session.reclean("left", [(76, 76, 98, 95)])
        self.assertTrue(np.all(np.asarray(session.clean)[80:90, 80:93] == 255))
        np.testing.assert_array_equal(self.output_pixels()[:, 320:], previous_neighbor)
        self.assertEqual(result["blocks"][1], previous_blocks[1])
        self.assertEqual(result["blocks"][0]["translated_text"], previous_blocks[0]["translated_text"])
        self.assertEqual(result["blocks"][0]["metadata"]["manual_cleanup_regions"], [[76, 76, 98, 95]])
        self.assertEqual(result["config_fingerprint"], "fixture-fingerprint")
        reopened = self.open_session()
        result = reopened.reclean("left", [(100, 76, 110, 95)])
        self.assertEqual(len(result["blocks"][0]["metadata"]["manual_cleanup_regions"]), 2)
        self.pool.acquire.assert_not_called()

    def test_manual_cleanup_rejects_outside_regions_and_honors_cancel_without_save(self):
        self.make_result()
        session = self.open_session()
        before_disk, before_record = self.disk_snapshot(), deepcopy(session.record)
        for region in [(0, 0, 100, 100), (25, 40, 25, 50), (10.5, 40, 100, 70)]:
            with self.subTest(region=region), self.assertRaises(ValueError):
                session.reclean("left", [region])
            self.assert_uncommitted(session, before_disk, before_record)
        def cancel():
            raise Cancelled("manual cleanup cancelled")
        with self.assertRaises(Cancelled):
            session.reclean("left", [(76, 76, 98, 95)], cancel=cancel)
        self.assert_uncommitted(session, before_disk, before_record)


if __name__ == "__main__":
    unittest.main()
