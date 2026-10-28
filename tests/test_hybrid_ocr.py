"""Deterministic routing tests; no downloads, model loading or GPU required."""
import json
import math
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
from core.runtime import Cancelled, Config, ModelPool, TextBlock
from vision import hybrid_ocr as hybrid
import main
from pipeline import batch as pipeline
from vision import detection as vision


def primary(text="主", score=.8):
    return {"text": text, "raw_text": text, "confidence": score}


def secondary(text="备用", error=""):
    return {"text": text, "raw_text": text, "confidence": None, "error": error}


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(use_hybrid_ocr=True)
        self.pool = Mock()
        self.crop = Image.new("RGB", (20, 30), "white")
        self.load = patch("vision.detection._acquire_ocr", return_value=object()).start()
        self.manga = patch("vision.hybrid_ocr.manga_result", return_value=primary()).start()
        self.acquire = patch("vision.hybrid_ocr.acquire_paddle", return_value=object()).start()
        self.paddle = patch("vision.hybrid_ocr.paddle_result", return_value=secondary()).start()
        self.addCleanup(patch.stopall)

    def run_crops(self, count=1, **kwargs):
        return hybrid.recognize_crops([self.crop] * count, pool=self.pool, config=self.config, **kwargs)

    def test_high_confidence_never_loads_secondary(self):
        rows = self.run_crops(40)
        self.assertEqual(self.manga.call_count, 40)
        self.load.assert_called_once()
        self.acquire.assert_not_called()
        self.paddle.assert_not_called()
        self.assertTrue(all(row["text"] == "主" and not row["reason"] for row in rows))

    def test_only_flagged_crops_fallback_in_one_pass_with_originals_retained(self):
        self.manga.side_effect = [primary("好", .8), primary("低", .05), primary("", .8), primary("边界", .1)]
        events = []
        self.acquire.side_effect = lambda *a: events.append(self.manga.call_count) or object()
        self.paddle.side_effect = [secondary("修一"), secondary("修二")]
        rows = self.run_crops(4)
        self.assertEqual(events, [4])
        self.acquire.assert_called_once()
        self.assertEqual(self.paddle.call_count, 2)
        self.assertEqual([r["text"] for r in rows], ["好", "修一", "修二", "边界"])
        self.assertEqual(rows[1]["primary"]["text"], "低")
        self.assertEqual(rows[1]["selected_engine"], "paddleocr-vl-1.6")
        self.assertIsNone(rows[1]["secondary"]["confidence"])
        json.dumps(rows, allow_nan=False)

    def test_secondary_failure_or_empty_retains_primary(self):
        self.manga.return_value = primary("保留", .01)
        for result in (secondary(""), secondary("错误输出", "repeated"), RuntimeError("offline")):
            with self.subTest(result=result):
                self.paddle.side_effect = [result]
                row = self.run_crops()[0]
                self.assertEqual(row["text"], "保留")
                self.assertEqual(row["selected_engine"], "manga-ocr")

    def test_load_failure_is_recorded_once_for_all_flagged_crops(self):
        self.manga.return_value = primary("保留", .01)
        self.acquire.side_effect = FileNotFoundError("missing local model")
        rows = self.run_crops(5)
        self.acquire.assert_called_once()
        self.paddle.assert_not_called()
        self.assertTrue(all(r["text"] == "保留" and "missing" in r["secondary"]["error"] for r in rows))

    def test_primary_error_can_be_rescued_and_unknown_score_routes(self):
        self.manga.side_effect = [ValueError("bad OCR"), primary("未知", None)]
        rows = self.run_crops(2)
        self.assertEqual([r["reason"] for r in rows], ["empty", "missing_confidence"])
        self.assertTrue(all(r["text"] == "备用" for r in rows))

    def test_cancellation_before_load_and_between_passes_propagates(self):
        with self.assertRaises(Cancelled):
            self.run_crops(cancel=Mock(side_effect=Cancelled()))
        self.load.assert_not_called()
        self.manga.return_value = primary("低", .01)
        with self.assertRaises(Cancelled):
            self.run_crops(cancel=Mock(side_effect=[None, None, Cancelled()]))
        self.acquire.assert_not_called()

    def test_oom_and_cancel_are_not_converted_to_text(self):
        for exception in (RuntimeError("CUDA out of memory"), Cancelled()):
            self.manga.side_effect = exception
            with self.assertRaises(type(exception)):
                self.run_crops()
        self.acquire.assert_not_called()

    def test_shared_entry_uses_same_segment_crops_and_preserves_partial_failure(self):
        block = TextBlock((0, 0, 20, 30), source_text="原记录", metadata={
            "ocr_segments": [[10, 0, 20, 30], [0, 0, 10, 30]]})
        self.manga.side_effect = [primary("右", .8), primary("", .1)]
        self.paddle.return_value = secondary("")
        observed = []
        vision.run_ocr(self.crop, [block], pool=self.pool, config=self.config,
                       observe_crop=lambda bid, index, crop: observed.append((index, crop.size)))
        self.assertEqual(observed, [(0, (10, 30)), (1, (10, 30))])
        self.assertEqual((block.source_text, block.status), ("原记录", "failed"))
        self.assertEqual(len(block.metadata["ocr_hybrid"]["segments"]), 2)
        self.assertIn("ocr_review", block.metadata)

    def test_segments_keep_order_after_successful_fallback(self):
        block = TextBlock((0, 0, 20, 30), metadata={"ocr_segments": [[10, 0, 20, 30], [0, 0, 10, 30]]})
        self.manga.side_effect = [primary("右", .8), primary("误", .01)]
        self.paddle.return_value = secondary("左")
        vision.run_ocr(self.crop, [block], pool=self.pool, config=self.config)
        self.assertEqual((block.source_text, block.status), ("右左", "pending"))

    def test_manual_edit_and_invalid_geometry_never_invoke_ocr(self):
        manual = TextBlock((0, 0, 20, 30), source_text="手改", metadata={"last_edit": "ocr_retranslation"})
        invalid = TextBlock((0, 0, 20, 30), metadata={"ocr_segments": [[0, 0, 50, 30]]})
        vision.run_ocr(self.crop, [manual, invalid], pool=self.pool, config=self.config)
        self.assertEqual(manual.source_text, "手改")
        self.assertEqual(invalid.status, "failed")
        self.load.assert_not_called()

    def test_preview_retains_source_and_both_results(self):
        block = TextBlock((0, 0, 20, 30), source_text="原文")
        self.manga.return_value = primary("低", .01)
        result = vision.preview_ocr_crop(self.crop, block, block.box, pool=self.pool, config=self.config)
        self.assertEqual((block.source_text, result["text"], result["engine"]),
                         ("原文", "备用", "paddleocr-vl-1.6"))
        self.assertEqual(result["hybrid"]["primary"]["text"], "低")
        self.assertIsNone(result["confidence"])
        self.assertTrue(self.load.call_args.args[1].local_files_only)


class ConfigurationTests(unittest.TestCase):
    def test_threshold_and_opt_in_validation(self):
        self.assertFalse(Config().use_hybrid_ocr)
        self.assertEqual(Config().ocr_prob_threshold, .1)
        for value in (-.01, 1.1, math.nan, math.inf, True, "0.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config(ocr_prob_threshold=value).validate()
        for value in (0, .1, 1):
            Config(ocr_prob_threshold=value).validate()
        with self.assertRaises(ValueError):
            Config(use_hybrid_ocr=1).validate()

    def test_cli_forwards_to_shared_reader(self):
        with patch("reader.launch_reader") as reader:
            main.main(["--read", "pages", "--hybrid-ocr", "--ocr-prob-threshold", "0.2",
                       "--paddle-ocr-model", "local-model"])
        config = reader.call_args.kwargs["config"]
        self.assertTrue(config.use_hybrid_ocr)
        self.assertEqual((config.ocr_prob_threshold, config.paddle_ocr_model_path), (.2, "local-model"))

    def test_missing_weights_never_download_or_allocate(self):
        with TemporaryDirectory() as directory, patch("vision.hybrid_ocr.require") as imports:
            pool = Mock()
            with self.assertRaises(FileNotFoundError):
                hybrid.acquire_paddle(pool, Config(paddle_ocr_model_path=directory))
            imports.assert_not_called()
            pool.acquire.assert_not_called()

    def test_fingerprint_tracks_active_weights_and_ignores_disabled_model(self):
        with TemporaryDirectory() as directory:
            weights = Path(directory) / "model.safetensors"
            weights.write_bytes(b"weights")
            config = Config(use_hybrid_ocr=True, paddle_ocr_model_path=directory,
                            detect_free_text=False, export_images=False)
            before = pipeline.configuration_fingerprint(config)
            weights.write_bytes(b"changed weights")
            self.assertNotEqual(before, pipeline.configuration_fingerprint(config))
            off = replace(config, use_hybrid_ocr=False)
            self.assertEqual(pipeline.configuration_fingerprint(off), pipeline.configuration_fingerprint(
                replace(off, paddle_ocr_model_path="unused", ocr_prob_threshold=.9)))

    def test_missing_score_boundary_and_repetition_guard(self):
        self.assertEqual(hybrid.fallback_reason("x", .1, .1), "")
        self.assertEqual(hybrid.fallback_reason("x", math.nan, .1), "missing_confidence")
        for text in ("。" * 30, "あいう" * 8, "a" * 25):
            self.assertTrue(hybrid.repeated_output(text))
        for text in ("", "……", "普通の日本語です。", "すーっ"):
            self.assertFalse(hybrid.repeated_output(text))


class DecodeTests(unittest.TestCase):
    def test_manga_beam_alignment_native_generation_and_special_token_exclusion(self):
        import torch
        ocr = Mock()
        ocr.model.device = "cpu"  # Only synthetic tensors, no CPU model inference.
        ocr.model.generation_config.eos_token_id = 2
        ocr.tokenizer.all_special_ids = [0, 2, 9]
        ocr._preprocess.return_value = torch.zeros(3, 2, 2)
        ocr.model.generate.return_value = SimpleNamespace(sequences=torch.tensor([[0, 5, 9, 6, 2]]),
            scores=(), beam_indices="beam-parent-indices")
        ocr.model.compute_transition_scores.return_value = torch.tensor([[-.1, -10., -.3, -20.]])
        ocr.tokenizer.decode.return_value = "あ い"
        def module(name):
            return torch if name == "torch" else SimpleNamespace(post_process=lambda s: s.replace(" ", ""))
        with patch("vision.hybrid_ocr.require", side_effect=module):
            row = hybrid.manga_result(ocr, Image.new("RGB", (2, 2)))
        self.assertAlmostEqual(row["confidence"], math.exp(-.2), places=6)
        self.assertEqual((row["text"], row["token_count"]), ("あい", 2))
        kwargs = ocr.model.generate.call_args.kwargs
        self.assertEqual(kwargs, dict(max_length=300, return_dict_in_generate=True, output_scores=True))
        self.assertEqual(ocr.model.compute_transition_scores.call_args.kwargs,
                         dict(beam_indices="beam-parent-indices", normalize_logits=True))

    def test_paddle_retries_once_for_repetition_and_never_invents_confidence(self):
        import torch
        processor, model = Mock(), Mock()
        model.device = "cpu"
        model.generation_config.eos_token_id = 2
        class Inputs(dict):
            def to(self, device):
                return self
        processor.apply_chat_template.return_value = Inputs(input_ids=torch.tensor([[1, 3]]))
        model.generate.return_value = torch.tensor([[1, 3, 5, 2]])
        processor.tokenizer.decode.side_effect = ["あ" * 30, "直った"]
        def module(name):
            return torch if name == "torch" else SimpleNamespace(post_process=lambda s: s)
        with patch("vision.hybrid_ocr.require", side_effect=module):
            row = hybrid.paddle_result((processor, model), Image.new("RGB", (2, 2)))
        self.assertEqual((row["text"], len(row["attempts"])), ("直った", 2))
        self.assertIsNone(row["confidence"])
        self.assertEqual(model.generate.call_count, 2)
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
