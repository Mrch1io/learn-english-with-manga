"""Real batch bookkeeping and SQLite, with a deterministic GPU-free page worker."""

import tempfile
import threading
import unittest
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.runtime import Config, ModelPool, TextBlock
from core.memory import GlossaryMemory
from pipeline import batch as pipeline


class FakePool:
    def __init__(self):
        self.load_seconds = {}
        self.loads = {}
        self.transfer_seconds = 0.0


class FakeProcessor:
    def __init__(self, *, failures=(), sources=None):
        self.calls = []
        self.failures = set(failures)
        self.sources = sources or {}

    def __call__(self, path, output, config, pool, memory, cancel, progress):
        cancel()
        self.calls.append(path.name)
        if path.name in self.failures:
            raise ValueError(f"broken page: {path.name}")
        text = self.sources.get(path.name, "カナが来た")
        memory.record_page(str(path), [text])
        output.write_bytes(b"fake translated image:" + path.read_bytes())
        return {
            "input_path": str(path), "output_path": str(output), "work": config.work,
            "target_language": config.target_language, "status": "done", "total_seconds": 0.01,
            "blocks": [{"block_id": "b0001", "source_text": text, "translated_text": "Kana arrived."}],
            "terms_fingerprint": pipeline.relevant_terms_fingerprint(memory, [text]),
            "speakers_fingerprint": pipeline.relevant_speakers_fingerprint(memory, str(path), ["b0001"], config),
            "output_sha256": pipeline.file_hash(output),
        }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.inputs = self.root / "chapter"
        self.inputs.mkdir()
        self.output = self.inputs / "translated_output"
        self.config = Config(work="test-work", translation_backend="qwen", memory_db=str(self.root / "memory.sqlite3"),
                             model_path=str(self.root / "fake.pt"), adapter_path=str(self.root / "adapter"))
        # Avoid reading large project weights or injecting the example glossary.
        self.root_patch = patch.object(pipeline, "ROOT", self.root)
        self.root_patch.start()
        self.pool = FakePool()
        self.processor = FakeProcessor()

    def tearDown(self):
        self.root_patch.stop()
        self.temporary.cleanup()

    def page(self, name, payload=b"original image"):
        path = self.inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def run_job(self, **kwargs):
        return pipeline.run_batch(self.inputs, self.output, kwargs.pop("config", self.config),
                                  pool=self.pool, page_processor=kwargs.pop("processor", self.processor), **kwargs)

    def memory(self):
        return GlossaryMemory(self.config.memory_db, self.config.work, self.config.target_language)

    def test_backend_fingerprint_tracks_selected_files_without_reading_gguf_or_unused_adapter(self):
        gguf, server = self.root / "selected.gguf", self.root / "llama-server.exe"
        gguf.write_bytes(b"selected GGUF")
        server.write_bytes(b"selected executable")
        adapter = self.root / "adapter"
        adapter.mkdir()
        adapter_weights = adapter / "adapter_model.safetensors"
        adapter_weights.write_bytes(b"previous adapter")
        vntl = replace(self.config, translation_backend="vntl", gguf_path=str(gguf),
                       llama_server_path=str(server), export_images=False)
        with patch.object(pipeline, "file_hash", wraps=pipeline.file_hash) as hashes:
            initial = pipeline.configuration_fingerprint(vntl)
        read_paths = {Path(call.args[0]) for call in hashes.call_args_list}
        self.assertNotIn(gguf, read_paths)
        self.assertNotIn(server, read_paths)
        self.assertNotIn(adapter_weights, read_paths)
        adapter_weights.write_bytes(b"changed old adapter")
        self.assertEqual(initial, pipeline.configuration_fingerprint(vntl))
        self.assertEqual(initial, pipeline.configuration_fingerprint(replace(
            vntl, model_name="unused", adapter_path=str(self.root / "unused-adapter"))))
        gguf.write_bytes(b"changed selected GGUF")
        self.assertNotEqual(initial, pipeline.configuration_fingerprint(vntl))
        qwen = replace(vntl, translation_backend="qwen")
        qwen_initial = pipeline.configuration_fingerprint(qwen)
        self.assertEqual(qwen_initial, pipeline.configuration_fingerprint(replace(
            qwen, gguf_path="unused.gguf", llama_server_path="unused.exe", llama_context_size=1024)))
        adapter_weights.write_bytes(b"changed selected adapter")
        self.assertNotEqual(qwen_initial, pipeline.configuration_fingerprint(qwen))

    def test_batch_report_keeps_llama_runtime_measurements(self):
        self.page("1.png")
        self.pool.vntl = SimpleNamespace(info={"offloaded_layers": 33, "total_layers": 33,
                                                "model_load_seconds": 2.25})
        result = self.run_job()
        self.assertEqual(result["translation_runtime"], self.pool.vntl.info)
        saved = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["translation_runtime"], self.pool.vntl.info)

    def test_page_text_model_and_setting_invalidate_cache_without_hashing_large_weights(self):
        magi = self.root / "magi"
        magi.mkdir()
        checkpoint = magi / "pytorch_model.bin"
        checkpoint.write_bytes(b"weights")
        config = replace(self.config, magi_model_path=str(magi), export_images=False)
        with patch.object(pipeline, "file_hash", wraps=pipeline.file_hash) as hashes:
            original = pipeline.configuration_fingerprint(config)
        self.assertNotIn(checkpoint, {Path(call.args[0]) for call in hashes.call_args_list})
        checkpoint.write_bytes(b"new weights")
        self.assertNotEqual(original, pipeline.configuration_fingerprint(config))
        disabled = replace(config, detect_free_text=False)
        self.assertNotEqual(original, pipeline.configuration_fingerprint(disabled))
        self.assertEqual(pipeline.configuration_fingerprint(disabled),
                         pipeline.configuration_fingerprint(replace(disabled, magi_model_path="unused")))

    def test_report_write_failure_still_closes_owned_llama_server_and_preserves_completed_page(self):
        from translation.vntl import VntlServer

        source = self.page("1.png")
        owned_pool = ModelPool()
        server = object.__new__(VntlServer)
        server.info, server.log = {"backend": "vntl"}, None
        server.process = Mock()
        server.process.poll.return_value = None
        owned_pool.vntl = server
        write_json = pipeline.atomic_json
        def fail_report(path, value):
            if Path(path).name.startswith("run-"):
                raise OSError("report disk is full")
            return write_json(path, value)
        with patch.object(pipeline, "ModelPool", return_value=owned_pool), \
                patch.object(pipeline, "atomic_json", side_effect=fail_report), \
                self.assertRaisesRegex(OSError, "report disk is full"):
            pipeline.run_batch(self.inputs, self.output, self.config, page_processor=self.processor)
        server.process.terminate.assert_called_once_with()
        server.process.wait.assert_called_once_with(timeout=10)
        self.assertIsNone(owned_pool.vntl)
        sidecar = pipeline.output_path_for(source, self.output, self.config).with_suffix(".json")
        self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8"))["status"], "done")

    def test_natural_order_and_recursive_output_exclusion(self):
        for name in ("10.png", "2.png", "1.png", "nested/3.JPG", ".cache/0.png", "notes.txt"):
            self.page(name)
        self.page("translated_output/9.png")
        self.page("translated_output/nested/8.png")
        result = pipeline.collect_images(self.inputs, self.output)
        self.assertEqual([p.relative_to(self.inputs).as_posix() for p in result],
                         ["1.png", "2.png", "10.png", "nested/3.JPG"])
        self.run_job()
        self.assertEqual(self.processor.calls, ["1.png", "2.png", "10.png", "3.JPG"])

    def test_same_stem_and_different_extensions_do_not_collide(self):
        first = self.page("1.png", b"png original")
        second = self.page("1.jpg", b"jpg original")
        report = self.run_job()
        outputs = [Path(page["output_path"]) for page in report["pages"]]
        self.assertEqual(len(set(outputs)), 2)
        self.assertTrue(all(path.exists() for path in outputs))
        self.assertEqual(first.read_bytes(), b"png original")
        self.assertEqual(second.read_bytes(), b"jpg original")
        self.assertEqual(report["completed"], 2)

    def test_failed_page_records_reason_and_continues(self):
        for name in ("1.png", "2.png", "3.png"):
            self.page(name)
        self.processor.failures.add("2.png")
        report = self.run_job()
        self.assertEqual(self.processor.calls, ["1.png", "2.png", "3.png"])
        self.assertEqual((report["completed"], report["failed"]), (2, 1))
        failed = report["pages"][1]
        self.assertEqual(failed["status"], "failed")
        self.assertIn("broken page: 2.png", failed["error"])
        self.assertTrue(Path(report["report_path"]).exists())
        retry = self.run_job()
        self.assertEqual((retry["skipped"], retry["failed"]), (2, 1))

    def test_cancel_preserves_completed_result_and_resumes(self):
        self.page("1.png")
        self.page("2.png")
        cancelled = threading.Event()

        def progress(event):
            if event.get("type") == "page_done":
                cancelled.set()

        report = self.run_job(cancel_event=cancelled, progress=progress)
        self.assertTrue(report["cancelled"])
        self.assertEqual(report["completed"], 1)
        finished = Path(report["pages"][0]["output_path"])
        self.assertTrue(finished.exists())
        self.assertEqual(self.processor.calls, ["1.png"])
        resume = self.run_job()
        self.assertEqual((resume["completed"], resume["skipped"]), (1, 1))
        self.assertTrue(finished.exists())

    def test_unchanged_result_skips_but_input_and_config_changes_invalidate(self):
        page = self.page("1.png")
        first = self.run_job()
        self.assertEqual(first["completed"], 1)
        self.assertEqual(self.run_job()["skipped"], 1)
        page.write_bytes(b"changed input")
        self.assertEqual(self.run_job()["completed"], 1)
        self.assertEqual(self.run_job(config=replace(self.config, min_font_size=13))["completed"], 1)
        self.assertEqual(self.run_job(config=replace(self.config, min_font_size=13))["skipped"], 1)
        self.assertEqual(len(self.processor.calls), 3)

    def test_relevant_terminology_changes_invalidate_unrelated_changes_do_not(self):
        self.page("1.png")
        with self.memory() as memory:
            memory.upsert_term("カナ", "Kana", locked=True)
        self.assertEqual(self.run_job()["completed"], 1)
        self.assertEqual(self.run_job()["skipped"], 1)
        with self.memory() as memory:
            memory.upsert_term("東京", "Tokyo")
        self.assertEqual(self.run_job()["skipped"], 1)
        with self.memory() as memory:
            memory.upsert_term("カナ", "Kanna", locked=True, user_edit=True)
        self.assertEqual(self.run_job()["completed"], 1)
        with self.memory() as memory:
            memory.delete_term("カナ")
        self.assertEqual(self.run_job()["completed"], 1)

    def test_new_term_and_new_alias_invalidate_matching_previous_page(self):
        self.page("1.png")
        self.assertEqual(self.run_job()["completed"], 1)
        with self.memory() as memory:
            memory.upsert_term("主人公", "Kana")
        self.assertEqual(self.run_job()["skipped"], 1)
        with self.memory() as memory:
            memory.upsert_term("主人公", "Kana", aliases=["カナ"], user_edit=True)
        self.assertEqual(self.run_job()["completed"], 1)
        with self.memory() as memory:
            memory.upsert_term("来た", "arrived")
        self.assertEqual(self.run_job()["completed"], 1)

    def test_confirmed_speaker_changes_invalidate_but_tentative_voice_and_unrelated_changes_do_not(self):
        source = self.page("1.png")
        with self.memory() as memory:
            memory.upsert_character("kana", "Kana", {"style_id": 1})
            memory.upsert_character("riku", "Riku")
            memory.bind_character(str(source), "b0001", "kana")
        self.assertEqual(self.run_job()["completed"], 1)
        self.assertEqual(self.run_job()["skipped"], 1)
        with self.memory() as memory:
            memory.upsert_character("kana", "Kana", {"style_id": 2})
            memory.bind_character(str(source), "unused-box", "riku", confirmed=True)
            memory.bind_character("another-page", "b0001", "riku", confirmed=True)
        self.assertEqual(self.run_job()["skipped"], 1)
        with self.memory() as memory:
            memory.bind_character(str(source), "b0001", "kana", confirmed=True, user_edit=True)
        self.assertEqual(self.run_job()["completed"], 1)
        self.assertEqual(self.run_job()["skipped"], 1)
        with self.memory() as memory:
            memory.upsert_character("kana", "Kana", {"style_id": 3})
            memory.upsert_character("riku", "Riku renamed")
        self.assertEqual(self.run_job()["skipped"], 1)
        with self.memory() as memory:
            memory.upsert_character("kana", "Kanna")
        self.assertEqual(self.run_job()["completed"], 1)
        with self.memory() as memory:
            memory.bind_character(str(source), "b0001", "riku", confirmed=True, user_edit=True)
        self.assertEqual(self.run_job()["completed"], 1)
        with self.memory() as memory:
            memory.bind_character(str(source), "b0001", "riku", confirmed=False, user_edit=True)
        self.assertEqual(self.run_job()["completed"], 1)
        self.assertEqual(self.run_job()["skipped"], 1)

    def test_missing_legacy_speaker_fingerprint_matches_only_empty_current_speakers(self):
        source = self.page("1.png")
        page = self.run_job()["pages"][0]
        page.pop("speakers_fingerprint")
        sidecar = Path(page["output_path"]).with_suffix(".json")
        pipeline.atomic_json(sidecar, page)
        self.assertEqual(self.run_job()["skipped"], 1)
        with self.memory() as memory:
            memory.upsert_character("kana", "Kana")
            memory.bind_character(str(source), "b0001", "kana", confirmed=True)
        self.assertEqual(self.run_job()["completed"], 1)

    def test_speaker_changes_preserve_manual_edits(self):
        source = self.page("1.png")
        page = self.run_job()["pages"][0]
        output = Path(page["output_path"])
        sidecar = output.with_suffix(".json")
        page["manual_edits"] = True
        pipeline.atomic_json(sidecar, page)
        before = {path: path.read_bytes() for path in (output, sidecar)}
        with self.memory() as memory:
            memory.upsert_character("kana", "Kana")
            memory.bind_character(str(source), "b0001", "kana", confirmed=True)
        report = self.run_job()
        self.assertEqual(report["pages"][0]["status"], "needs_review")
        self.assertEqual(report["partial"], 1)
        self.assertEqual(self.processor.calls, ["1.png"])
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_disabled_speaker_context_ignores_speaker_changes_without_querying_memory(self):
        for config in (replace(self.config, use_speaker_context=False), replace(self.config, context_chars=0)):
            with self.subTest(config=config), self.memory() as memory:
                with patch.object(memory, "translation_speakers", side_effect=AssertionError("unexpected speaker read")):
                    value = pipeline.relevant_speakers_fingerprint(memory, "page", ["b0001"], config)
                self.assertEqual(value, pipeline.digest_json({}))

    def test_edited_output_is_not_mistaken_for_a_valid_cache(self):
        self.page("1.png")
        first = self.run_job()
        Path(first["pages"][0]["output_path"]).write_bytes(b"externally edited result")
        self.assertEqual(self.run_job()["completed"], 1)

    def test_unowned_outputs_and_broken_sidecars_are_preserved(self):
        source = self.page("1.png")
        self.output.mkdir()
        reserved = pipeline.output_path_for(source, self.output, self.config)
        reserved.write_bytes(b"unrelated artwork")
        reserved.with_suffix(".json").write_text("broken metadata", encoding="utf-8")
        report = self.run_job()
        self.assertNotEqual(Path(report["pages"][0]["output_path"]), reserved)
        self.assertEqual(reserved.read_bytes(), b"unrelated artwork")
        self.assertEqual(reserved.with_suffix(".json").read_text(encoding="utf-8"), "broken metadata")

    def test_valid_json_with_wrong_sidecar_shape_is_preserved(self):
        source = self.page("1.png")
        self.output.mkdir()
        reserved = pipeline.output_path_for(source, self.output, self.config)
        reserved.write_bytes(b"unrelated artwork")
        reserved.with_suffix(".json").write_text("[]", encoding="utf-8")
        report = self.run_job()
        self.assertEqual(report["completed"], 1)
        self.assertNotEqual(Path(report["pages"][0]["output_path"]), reserved)
        self.assertEqual(reserved.read_bytes(), b"unrelated artwork")

    def test_unreadable_input_is_a_page_failure_and_does_not_stop_the_batch(self):
        unreadable = self.page("1.png")
        self.page("2.png")
        hash_file = pipeline.file_hash

        def fail_unreadable(path):
            if Path(path).resolve() == unreadable:
                raise PermissionError("simulated unreadable image")
            return hash_file(path)

        with patch.object(pipeline, "file_hash", side_effect=fail_unreadable):
            report = self.run_job()
        self.assertEqual((report["completed"], report["failed"]), (1, 1))
        self.assertEqual(report["pages"][0]["status"], "failed")
        self.assertIn("unreadable image", report["pages"][0]["error"])
        self.assertEqual(self.processor.calls, ["2.png"])
        sidecar = pipeline.output_path_for(unreadable, self.output, self.config).with_suffix(".json")
        self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8")), report["pages"][0])

    def test_unreadable_input_preserves_existing_result_record_even_with_force(self):
        unreadable = self.page("1.png")
        record = self.run_job()["pages"][0]
        output = Path(record["output_path"])
        sidecar = output.with_suffix(".json")
        image_bytes = output.read_bytes()
        hash_file = pipeline.file_hash

        def fail_unreadable(path):
            if Path(path).resolve() == unreadable:
                raise PermissionError("temporary source read failure")
            return hash_file(path)

        for manual in (False, True):
            for force in (False, True):
                with self.subTest(manual=manual, force=force):
                    record["manual_edits"] = manual
                    pipeline.atomic_json(sidecar, record)
                    saved_record_bytes = sidecar.read_bytes()
                    with patch.object(pipeline, "file_hash", side_effect=fail_unreadable):
                        report = self.run_job(force=force)
                    self.assertEqual(report["failed"], 1)
                    self.assertIn("temporary source read failure", report["pages"][0]["error"])
                    self.assertEqual(sidecar.read_bytes(), saved_record_bytes)
                    self.assertEqual(output.read_bytes(), image_bytes)
                    self.assertEqual(self.processor.calls, ["1.png"])
                    persisted_run = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
                    self.assertEqual(persisted_run["pages"][0]["status"], "failed")

    def test_crash_between_image_and_sidecar_does_not_overwrite_orphan_on_restart(self):
        self.page("1.png")
        save_json = pipeline.atomic_json

        def simulate_crash(path, value):
            if not Path(path).name.startswith("run-"):
                raise OSError("simulated interruption before sidecar commit")
            return save_json(path, value)

        with patch.object(pipeline, "atomic_json", side_effect=simulate_crash):
            failed = self.run_job()
        self.assertEqual(failed["failed"], 1)
        self.assertIn("simulated interruption", failed["pages"][0]["error"])
        orphan = next(self.output.glob("*.png"))
        original_bytes = orphan.read_bytes()
        self.assertFalse(orphan.with_suffix(".json").exists())
        report = self.run_job()
        self.assertEqual(report["completed"], 1)
        self.assertNotEqual(Path(report["pages"][0]["output_path"]), orphan)
        self.assertEqual(orphan.read_bytes(), original_bytes)

    def test_same_input_output_directory_is_rejected(self):
        self.page("1.png")
        with self.assertRaisesRegex(ValueError, "separate output directory"):
            pipeline.run_batch(self.inputs, self.inputs, self.config,
                               pool=self.pool, page_processor=self.processor)

    def test_manual_edits_survive_invalidated_batch_until_explicit_force(self):
        source = self.page("1.png")
        page = self.run_job()["pages"][0]
        output = Path(page["output_path"])
        before = output.read_bytes()
        page["manual_edits"] = True
        pipeline.atomic_json(output.with_suffix(".json"), page)
        source.write_bytes(b"changed original")
        report = self.run_job()
        self.assertEqual(report["pages"][0]["status"], "needs_review")
        self.assertEqual(report["partial"], 1)
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(self.processor.calls, ["1.png"])
        self.assertEqual(self.run_job(force=True)["completed"], 1)
        self.assertNotEqual(output.read_bytes(), before)

    def test_ocr_candidates_survive_invalidated_batch_until_explicit_force(self):
        source = self.page("1.png")
        page = self.run_job()["pages"][0]
        output = Path(page["output_path"])
        sidecar = output.with_suffix(".json")
        page["ocr_candidate_edits"] = True
        page["blocks"][0]["metadata"] = {"ocr_candidates": [{"candidate_id": "candidate-1", "text": "candidate"}],
                                           "ocr_review": "Candidate pending review"}
        pipeline.atomic_json(sidecar, page)
        before_record, before_image = sidecar.read_bytes(), output.read_bytes()
        source.write_bytes(b"changed original")
        report = self.run_job()
        self.assertEqual(report["pages"][0]["status"], "needs_review")
        self.assertEqual(report["partial"], 1)
        self.assertEqual(sidecar.read_bytes(), before_record)
        self.assertEqual(output.read_bytes(), before_image)
        self.assertEqual(self.processor.calls, ["1.png"])
        self.assertEqual(self.run_job(force=True)["completed"], 1)
        self.assertEqual(self.processor.calls, ["1.png", "1.png"])
        self.assertNotEqual(output.read_bytes(), before_image)

    def test_changed_clean_base_invalidates_completed_cache(self):
        self.page("1.png")
        page = self.run_job()["pages"][0]
        output = Path(page["output_path"])
        clean = output.with_name(output.stem + ".clean.png")
        clean.write_bytes(b"clean base")
        page["clean_image"] = {"file": clean.name, "sha256": pipeline.file_hash(clean)}
        pipeline.atomic_json(output.with_suffix(".json"), page)
        self.assertEqual(self.run_job()["skipped"], 1)
        clean.write_bytes(b"changed base")
        self.assertEqual(self.run_job()["completed"], 1)

    def test_process_page_persists_clean_base_before_drawing_and_full_config(self):
        from PIL import Image, ImageChops
        source = self.page("1.png")
        Image.new("RGB", (250, 160), "white").save(source)
        self.output.mkdir()
        output = self.output / "page.png"
        blocks = [TextBlock((20, 20, 230, 140), source_text="原文", block_id="bubble-1")]
        clean = Image.new("RGB", (250, 160), "white")
        def translated(items, **kwargs):
            items[0].translated_text, items[0].status = "Readable words", "translated"
            return {"generation_calls": 1}
        pool = SimpleNamespace(sync=lambda: None, is_oom=lambda exc: False)
        with self.memory() as memory:
            with patch("vision.detection.detect", return_value=blocks), patch("vision.detection.run_ocr"), \
                    patch("translation.translate", side_effect=translated), \
                    patch("vision.graphics.inpaint", return_value=clean):
                record = pipeline.process_page(source, output, self.config, pool, memory, lambda: None)
        self.assertEqual(record["config"], asdict(self.config))
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["clean_image"]["cleaned_block_ids"], ["bubble-1"])
        saved_clean = self.output / record["clean_image"]["file"]
        with Image.open(saved_clean) as actual:
            self.assertIsNone(ImageChops.difference(clean, actual).getbbox())
        with Image.open(output) as actual:
            self.assertIsNotNone(ImageChops.difference(clean, actual).getbbox())
        self.assertEqual(record["clean_image"]["sha256"], pipeline.file_hash(saved_clean))

    def test_process_page_fingerprints_the_exact_speaker_snapshot_sent_to_translation(self):
        from PIL import Image
        source = self.page("1.png")
        Image.new("RGB", (80, 60), "white").save(source)
        self.output.mkdir()
        blocks = [TextBlock((10, 10, 70, 50), source_text="待って", block_id="b0001")]
        pool = SimpleNamespace(sync=lambda: None, is_oom=lambda exc: False)
        config = replace(self.config, export_images=False)
        snapshot = {"b0001": {"character_id": "kana", "name": "Kana"}}
        with self.memory() as memory:
            memory.upsert_character("kana", "Kana")
            memory.bind_character(str(source), "b0001", "kana", confirmed=True)
            def translated(items, **kwargs):
                self.assertEqual(kwargs["speakers"], snapshot)
                # A simultaneous name edit must not make the old output look as
                # though it was generated with the new identity information.
                memory.upsert_character("kana", "Kanna")
                items[0].translated_text, items[0].status = "Wait!", "translated"
                return {"generation_calls": 1}
            with patch.object(memory, "translation_speakers", wraps=memory.translation_speakers) as read_speakers, \
                    patch("vision.detection.detect", return_value=blocks), patch("vision.detection.run_ocr"), \
                    patch("translation.translate", side_effect=translated):
                record = pipeline.process_page(source, self.output / "page.png", config, pool, memory, lambda: None)
            read_speakers.assert_called_once_with(str(source))
            self.assertEqual(record["speakers_fingerprint"], pipeline.digest_json(snapshot))
            self.assertNotEqual(record["speakers_fingerprint"], pipeline.relevant_speakers_fingerprint(
                memory, str(source), ["b0001"], config))

    def test_translation_oom_preserves_completed_bubbles_without_restarting_page(self):
        from PIL import Image
        from translation import translate

        for failure_stage in ("second_bubble", "term_extraction"):
            with self.subTest(failure_stage=failure_stage):
                source = self.page(failure_stage + ".png")
                Image.new("RGB", (80, 60), "white").save(source)
                self.output.mkdir(exist_ok=True)
                output = self.output / source.name
                blocks = [TextBlock((0, 0, 30, 30), source_text="行け", block_id="first"),
                          TextBlock((40, 0, 70, 30), source_text="待て", block_id="second")]
                pool = SimpleNamespace(sync=Mock(), is_oom=ModelPool.is_oom,
                                       policy="resident", offload=Mock())
                responses = ["Go!"] if failure_stage == "second_bubble" else ["Go!", "Wait!"]
                calls = []
                def generate(messages, limit):
                    calls.append((messages, limit))
                    if len(calls) <= len(responses):
                        return responses[len(calls) - 1]
                    raise RuntimeError("CUDA out of memory during " + failure_stage)
                config = replace(self.config, export_images=False, translation_mode="individual",
                                 use_continuations=False, extract_terms=failure_stage == "term_extraction")
                with self.memory() as memory, patch("vision.detection.detect", return_value=blocks), \
                        patch("vision.detection.run_ocr"), patch("translation.translate", wraps=translate) as translated, \
                        patch("translation._load_generator", return_value=generate), \
                        self.assertRaisesRegex(RuntimeError, "out of memory") as failure:
                    pipeline.process_page(source, output, config, pool, memory, lambda: None)
                translated.assert_called_once()
                self.assertEqual(len(calls), len(responses) + 1)
                self.assertEqual([block.metadata["translation_attempts"] for block in blocks], [1, 1])
                for block, response in zip(blocks, responses):
                    self.assertEqual((block.translated_text, block.status), (response, "translated"))
                record = failure.exception.page_record
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["blocks"], [asdict(block) for block in blocks])
                self.assertNotIn("translate", record.get("oom_retries", {}))
                pool.offload.assert_not_called()
                self.assertEqual(pool.policy, "resident")
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix(".json").exists())

    def test_image_stage_oom_offloads_once_and_stops_after_one_retry(self):
        from PIL import Image

        for recovered in (True, False):
            with self.subTest(recovered=recovered):
                source = self.page(f"detect-oom-{recovered}.png")
                Image.new("RGB", (80, 60), "white").save(source)
                self.output.mkdir(exist_ok=True)
                pool = SimpleNamespace(sync=Mock(), is_oom=ModelPool.is_oom,
                                       policy="resident", offload=Mock())
                responses = [RuntimeError("CUDA out of memory during detection"),
                             [] if recovered else RuntimeError("CUDA out of memory after offload")]
                with self.memory() as memory, patch("vision.detection.detect", side_effect=responses) as detect:
                    if recovered:
                        record = pipeline.process_page(source, self.output / source.name,
                            replace(self.config, export_images=False), pool, memory, lambda: None)
                        self.assertEqual(record["status"], "done")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "after offload") as failure:
                            pipeline.process_page(source, self.output / source.name,
                                replace(self.config, export_images=False), pool, memory, lambda: None)
                        record = failure.exception.page_record
                self.assertEqual(detect.call_count, 2)
                pool.offload.assert_called_once_with()
                self.assertEqual(pool.policy, "staged")
                self.assertEqual(record["oom_retries"], {"detect": 1})

    def test_untranslated_japanese_is_retried_before_layout_or_reported_as_translation_failure(self):
        from PIL import Image, ImageChops
        for retry_text in ("Reading pirated copies...", "海賊版は読んじゃダメですっ"):
            with self.subTest(retry_text=retry_text):
                source = self.page("language-check.png")
                original = Image.new("RGB", (600, 300), "white")
                original.save(source)
                self.output.mkdir(exist_ok=True)
                output = self.output / "language-check.png"
                blocks = [TextBlock((320, 20, 580, 280), source_text="海賊版は読んじゃ", block_id="right"),
                          TextBlock((20, 20, 280, 280), source_text="ダメですっ", block_id="left")]
                responses = iter(["海賊版は読んじゃダメですっ", retry_text, "Don't!"])
                pool = SimpleNamespace(sync=lambda: None, is_oom=lambda exc: False)
                with self.memory() as memory, patch("vision.detection.detect", return_value=blocks), \
                        patch("vision.detection.run_ocr"), patch("translation._load_generator", return_value=lambda *_: next(responses)):
                    record = pipeline.process_page(source, output, replace(self.config, extract_terms=False,
                                                                          use_continuations=False),
                                                   pool, memory, lambda: None)
                succeeded = retry_text.startswith("Reading")
                self.assertEqual(record["status"], "done" if succeeded else "partial")
                self.assertEqual(record["translation_stats"]["generation_calls"], 3)
                self.assertEqual(blocks[1].status, "translated")
                self.assertEqual(blocks[0].metadata["translation_attempts"], 2)
                if succeeded:
                    self.assertEqual(record["clean_image"]["cleaned_block_ids"], ["right", "left"])
                    with Image.open(output) as rendered:
                        for item in blocks:
                            self.assertIsNotNone(ImageChops.difference(original.crop(item.box), rendered.crop(item.box)).getbbox())
                else:
                    self.assertEqual(record["clean_image"]["cleaned_block_ids"], ["left"])
                    self.assertIn("Expected English", blocks[0].error)
                    self.assertNotIn("Font", blocks[0].error)
                    self.assertEqual(blocks[0].translated_text, "")

    def test_multiple_inputs_deduplicate_and_naturally_sort(self):
        files = [self.page(name) for name in ("10.png", "2.png", "1.png")]
        report = pipeline.run_batch([files[0], files[1], files[0], files[2]], self.output,
                                    self.config, pool=self.pool, page_processor=self.processor)
        self.assertEqual(self.processor.calls, ["1.png", "2.png", "10.png"])
        self.assertEqual(report["completed"], 3)

    def test_subdirectories_preserved_but_owned_legacy_flat_manual_result_reused(self):
        source = self.page("chapter2/3.png")
        fresh = self.run_job()["pages"][0]
        self.assertEqual(Path(fresh["output_path"]).parent, self.output / "chapter2")
        self.page("chapter1/2.png")
        legacy_source = self.inputs / "chapter1/2.png"
        legacy = pipeline.output_path_for(legacy_source, self.output, self.config)
        legacy.write_bytes(b"legacy hand corrected image")
        old = {"input_path": str(legacy_source), "work": self.config.work,
               "target_language": "en", "manual_edits": True, "blocks": []}
        pipeline.atomic_json(legacy.with_suffix(".json"), old)
        report = self.run_job()
        retained = next(row for row in report["pages"] if row["input_path"] == str(legacy_source))
        self.assertEqual(retained["status"], "needs_review")
        self.assertEqual(Path(retained["output_path"]), legacy)
        self.assertEqual(legacy.read_bytes(), b"legacy hand corrected image")
        self.assertFalse((self.output / "chapter1").exists())

    def test_forced_processing_failure_keeps_existing_images_and_record(self):
        self.page("1.png")
        first = self.run_job()["pages"][0]
        output = Path(first["output_path"])
        sidecar = output.with_suffix(".json")
        before = (output.read_bytes(), sidecar.read_bytes())
        self.processor.failures.add("1.png")
        report = self.run_job(force=True)
        self.assertEqual(report["failed"], 1)
        self.assertEqual((output.read_bytes(), sidecar.read_bytes()), before)

    def test_text_only_page_skips_graphics_and_caches_without_image(self):
        from PIL import Image
        source = self.page("1.png")
        Image.new("RGB", (250, 160), "white").save(source)
        config = replace(self.config, export_images=False)
        pool = SimpleNamespace(device="cpu", requested_device="cpu", sync=lambda: None,
                               is_oom=lambda exc: False, load_seconds={}, loads={}, transfer_seconds=0)
        def translated(items, **kwargs):
            items[0].translated_text, items[0].status = "Long reading text " * 100, "translated"
            return {"generation_calls": 1}
        with patch("vision.detection.detect", return_value=[TextBlock((20, 20, 230, 140), "原文", block_id="one")]), \
             patch("vision.detection.run_ocr"), patch("translation.translate", side_effect=translated), \
             patch("vision.graphics.prepare_layout", side_effect=AssertionError("No layout for reading")), \
             patch("vision.graphics.inpaint", side_effect=AssertionError("No cleanup for reading")), \
             patch("vision.graphics.render", side_effect=AssertionError("No rendering for reading")):
            report = pipeline.run_batch(source, self.output, config, pool=pool)
        record = report["pages"][0]
        self.assertEqual(record["status"], "done")
        self.assertFalse(record["exported"])
        self.assertFalse(Path(record["output_path"]).exists())
        self.assertNotIn("clean_image", record)
        with patch("vision.detection.detect", side_effect=AssertionError("Reading cache should skip")):
            report = pipeline.run_batch(source, self.output, config, pool=pool)
        self.assertEqual(report["skipped"], 1)

    def test_recovered_text_remains_reviewable_after_successful_translation(self):
        from PIL import Image
        source = self.page("review.png")
        Image.new("RGB", (250, 160), "white").save(source)
        self.output.mkdir()
        block = TextBlock((20, 20, 230, 140), "原文", block_id="free-text",
                          metadata={"detection_review": "Check recovered text"})
        def translated(items, **kwargs):
            items[0].translated_text, items[0].status = "Readable words", "translated"
            return {"generation_calls": 1}
        pool = SimpleNamespace(sync=lambda: None, is_oom=lambda exc: False)
        with self.memory() as memory, patch("vision.detection.detect", return_value=[block]), \
             patch("vision.detection.run_ocr"), patch("translation.translate", side_effect=translated):
            record = pipeline.process_page(source, self.output / "review.png",
                                           replace(self.config, export_images=False),
                                           pool, memory, lambda: None)
        self.assertEqual(record["status"], "partial")
        self.assertEqual(record["blocks"][0]["status"], "translated")
        self.assertEqual(record["blocks"][0]["metadata"]["detection_review"], "Check recovered text")
        self.assertIn("1 OCR regions need review", record["message"])

    def test_pipeline_three_file_commit_rolls_back_each_install_failure(self):
        from PIL import Image
        source = self.page("1.png")
        Image.new("RGB", (250, 160), "white").save(source)
        self.output.mkdir()
        output = self.output / "page.png"
        pool = SimpleNamespace(sync=lambda: None, is_oom=lambda exc: False)
        def translated(items, **kwargs):
            items[0].translated_text, items[0].status = "Readable words", "translated"
            return {"generation_calls": 1}
        with self.memory() as memory, patch("vision.detection.detect", side_effect=lambda *args, **kwargs: [
                TextBlock((20, 20, 230, 140), "原文", block_id="one")]), \
             patch("vision.detection.run_ocr"), patch("translation.translate", side_effect=translated), \
             patch("vision.graphics.inpaint", return_value=Image.new("RGB", (250, 160), "white")):
            record = pipeline.process_page(source, output, self.config, pool, memory, lambda: None)
            clean = output.parent / record["clean_image"]["file"]
            before = {path: path.read_bytes() for path in (clean, output, output.with_suffix(".json"))}
            real_replace = os.replace
            for failed_path in before:
                failed = False
                def fail_one(source_path, destination):
                    nonlocal failed
                    if Path(destination).resolve() == failed_path.resolve() and not failed:
                        failed = True
                        raise OSError("injected page commit failure")
                    return real_replace(source_path, destination)
                with self.subTest(path=failed_path), patch("pipeline.batch.os.replace", side_effect=fail_one):
                    with self.assertRaisesRegex(OSError, "injected page commit"):
                        pipeline.process_page(source, output, self.config, pool, memory, lambda: None)
                self.assertTrue(failed)
                self.assertEqual({path: path.read_bytes() for path in before}, before)
                self.assertFalse(list(self.output.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
