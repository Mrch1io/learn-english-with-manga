"""Magi mapping tests use fabricated detections/embeddings, never a model."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from vision.characters import _clip_character_box, _save_analysis, analyze_pages, suggest_characters
from core.runtime import Cancelled
from core.memory import GlossaryMemory
from pipeline.batch import file_hash


def vector(index=0):
    result = [0.0]*768
    result[index] = 1.0
    return result


class FakePool:
    closed = 0

    def __init__(self, *args):
        pass

    def close(self):
        type(self).closed += 1

    @staticmethod
    def is_oom(exc):
        return False


class CharacterTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.db = self.root/"memory.sqlite3"
        self.memory = GlossaryMemory(self.db, "work-a")
        self.record = {"work": "work-a", "target_language": "en", "input_path": "2.png", "input_sha256": "abc",
                       "config": {"memory_db": str(self.db)}, "image_size": [100, 100], "extra": {"keep": True},
                       "blocks": [
                           {"block_id": "b1", "box": [0, 0, 50, 50], "source_text": "凛", "translated_text": "Rin",
                            "status": "translated", "metadata": {"layout": {"keep": True}}},
                           {"block_id": "b2", "box": [50, 0, 100, 50], "source_text": "はい", "translated_text": "Yes",
                            "metadata": {}},
                           {"block_id": "b3", "box": [0, 50, 50, 100], "source_text": "...", "translated_text": "..."}]}
        self.result = {"texts": [[5, 5, 15, 15], [20, 5, 30, 15], [60, 5, 70, 15]],
                       "characters": [[0, 60, 20, 90], [60, 60, 80, 90]], "panels": [], "tails": [],
                       "text_character_associations": [[0, 0], [1, 1], [2, 1]],
                       "character_cluster_labels": [0, 0], "is_essential_text": [True, True, True]}
        self.embeddings = [vector(), vector()]

    def tearDown(self):
        self.memory.close()
        self.folder.cleanup()

    def suggest(self, result=None, embeddings=None):
        return suggest_characters(self.record, self.result if result is None else result,
                                  self.embeddings if embeddings is None else embeddings, self.memory)

    def page(self, name):
        original = self.root/(name+".png")
        Image.new("RGB", (100, 100), "white").save(original)
        record = deepcopy(self.record)
        record.update(input_path=str(original), input_sha256=file_hash(original))
        sidecar = self.root/(name+".json")
        sidecar.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return sidecar

    def test_same_cluster_gets_one_stable_candidate_and_preserves_other_fields(self):
        before = deepcopy(self.record)
        updated, characters, bindings, samples = self.suggest()
        first, second, third = [b["metadata"]["magi"] for b in updated["blocks"]]
        self.assertEqual(first["character_id"], second["character_id"])
        self.assertEqual(first["method"], "new_unconfirmed_cluster")
        self.assertEqual(third["character_id"], "unknown")
        self.assertEqual(len(characters), 1)
        self.assertEqual(characters[0]["voice"], {})
        self.assertEqual(samples, [])
        self.assertEqual(self.record, before)
        for original, changed in zip(before["blocks"], updated["blocks"]):
            changed["metadata"].pop("magi")
            if "metadata" not in original:
                changed.pop("metadata")
            self.assertEqual(original, changed)
        relabelled = {**self.result, "character_cluster_labels": [20, 20]}
        self.assertEqual(self.suggest(relabelled)[1], characters)
        self.assertEqual(len(self.memory.list_characters()), 2)

    def test_different_speakers_or_missing_association_in_one_bubble_stay_unknown(self):
        conflicting = {**self.result, "character_cluster_labels": [0, 1]}
        updated = self.suggest(conflicting)[0]
        self.assertEqual(updated["blocks"][0]["metadata"]["magi"]["method"], "conflicting_text_speakers")
        missing = {**self.result, "text_character_associations": [[0, 0], [2, 1]]}
        info = self.suggest(missing)[0]["blocks"][0]["metadata"]["magi"]
        self.assertEqual(info["character_id"], "unknown")
        self.assertNotIn("character_box", info)

    def test_coverage_keeps_unmatched_nonessential_text_without_mutating_bubbles(self):
        result = deepcopy(self.result)
        result["texts"].append([60, 60, 90, 80])
        result["is_essential_text"].append(False)
        before = deepcopy(self.record)
        updated = self.suggest(result)[0]
        self.assertEqual(updated["magi_coverage"], {
            "detected_text_count": 4, "covered_text_count": 3, "covered_bubble_count": 2,
            "unmatched_texts": [{"text_index": 3, "box": [60, 60, 90, 80]}],
        })
        self.assertEqual(self.record, before)
        for old, new in zip(before["blocks"], updated["blocks"]):
            for field in ("block_id", "box", "source_text", "translated_text"):
                self.assertEqual(old[field], new[field])

    def test_coverage_deduplicates_regions_shared_by_bubbles(self):
        self.record["blocks"][1]["box"] = self.record["blocks"][0]["box"]
        coverage = self.suggest()[0]["magi_coverage"]
        self.assertEqual(coverage["detected_text_count"], 3)
        self.assertEqual(coverage["covered_text_count"], 2)
        self.assertEqual(coverage["covered_bubble_count"], 2)
        self.assertEqual([row["text_index"] for row in coverage["unmatched_texts"]], [2])

    def test_coverage_with_no_bubbles_preserves_every_detected_region(self):
        self.record["blocks"] = []
        coverage = self.suggest()[0]["magi_coverage"]
        self.assertEqual(coverage["covered_text_count"], 0)
        self.assertEqual(coverage["covered_bubble_count"], 0)
        self.assertEqual([row["text_index"] for row in coverage["unmatched_texts"]], [0, 1, 2])

    def test_mapping_requires_center_and_area_and_does_not_use_character_position(self):
        result = {**self.result, "texts": [[-50, -50, 51, 51]], "text_character_associations": [[0, 0]]}
        # Center is inside b1 but overlap/min-area is too small for a safe match.
        result["texts"] = [[-99, 1, 101, 3]]
        info = self.suggest(result)[0]["blocks"][0]["metadata"]["magi"]
        self.assertEqual(info["character_id"], "unknown")
        self.record["blocks"][1]["box"] = self.record["blocks"][0]["box"]
        info = self.suggest()[0]["blocks"][0]["metadata"]["magi"]
        self.assertEqual(info["method"], "ambiguous_text_overlap")

    def test_manual_binding_wins_and_propagates_only_within_consistent_cluster(self):
        self.memory.upsert_character("rin", "凛")
        self.memory.bind_character("2.png", "b1", "rin", confirmed=True)
        updated, characters, bindings, samples = self.suggest()
        first, second = [b["metadata"]["magi"] for b in updated["blocks"][:2]]
        self.assertEqual(first["character_id"], "rin")
        self.assertTrue(first["confirmed"])
        self.assertEqual(first["method"], "manual")
        self.assertEqual(second["character_id"], "rin")
        self.assertFalse(second["confirmed"])
        self.assertEqual(characters, [])
        self.assertEqual(samples[0]["character_id"], "rin")
        self.assertNotIn(("b1", "rin"), bindings)

    def test_conflicting_confirmed_cluster_does_not_change_manual_bindings(self):
        self.memory.upsert_character("rin", "凛")
        self.memory.upsert_character("aya", "彩")
        self.memory.bind_character("2.png", "b1", "rin", confirmed=True)
        self.memory.bind_character("2.png", "b2", "aya", confirmed=True)
        result = deepcopy(self.result)
        result["texts"].append([5, 55, 15, 65])
        result["text_character_associations"].append([3, 0])
        updated = self.suggest(result)[0]
        self.assertEqual([b["metadata"]["magi"]["character_id"] for b in updated["blocks"]], ["rin", "aya", "unknown"])
        self.assertEqual(updated["blocks"][2]["metadata"]["magi"]["method"], "conflicting_manual_cluster")

    def test_cross_page_sample_distance_uses_different_character_margin(self):
        self.memory.upsert_character("rin", "凛")
        self.memory.save_character_sample("rin", "old.png", [0, 0, 20, 30], vector())
        self.memory.save_character_sample("rin", "old2.png", [0, 0, 20, 30], vector())
        info = self.suggest()[0]["blocks"][0]["metadata"]["magi"]
        self.assertEqual(info["character_id"], "rin")
        self.assertEqual(info["method"], "sample_cosine")
        self.assertAlmostEqual(info["distance"], 0)
        self.memory.upsert_character("aya", "彩")
        similar = vector()
        similar[1] = .1
        self.memory.save_character_sample("aya", "old.png", [0, 0, 20, 30], similar)
        info = self.suggest()[0]["blocks"][0]["metadata"]["magi"]
        self.assertEqual(info["character_id"], "unknown")
        self.assertEqual(info["method"], "ambiguous_sample_match")

    def test_work_scope_changes_candidate_identity(self):
        first = self.suggest()[1][0]["id"]
        with GlossaryMemory(self.db, "different-work") as other:
            second = suggest_characters(self.record, self.result, self.embeddings, other)[1][0]["id"]
        self.assertNotEqual(first, second)

    def test_character_crop_clips_minor_overhang_and_rejects_outside_boxes(self):
        self.assertEqual(_clip_character_box([-1.2, 3, 101.5, 102], (100, 100)), [0, 3, 100, 100])
        with self.assertRaises(ValueError):
            _clip_character_box([101, 1, 110, 10], (100, 100))

    def test_batch_natural_sort_failure_continue_and_model_reuse(self):
        ten, two = self.page("10"), self.page("2")
        before = json.loads(two.read_text(encoding="utf-8"))
        with (patch("vision.characters.ModelPool", FakePool), patch("vision.characters.load_magi", return_value=object()) as loader,
              patch("vision.characters._predict", side_effect=[ValueError("one page failed"), (self.result, self.embeddings)])):
            report = analyze_pages([ten.with_suffix(".png"), two])
        self.assertEqual([Path(page["path"]).stem for page in report["pages"]], ["2", "10"])
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["completed"], 1)
        loader.assert_called_once()
        self.assertEqual(json.loads(two.read_text(encoding="utf-8")), before)
        after = json.loads(ten.read_text(encoding="utf-8"))
        self.assertIn("magi", after["blocks"][0]["metadata"])
        self.assertEqual(len(self.memory.list_character_bindings(after["input_path"])), 3)

    def test_analysis_saves_coverage_and_reports_possible_misses(self):
        path = self.page("2")
        result = deepcopy(self.result)
        result["texts"].append([60, 60, 90, 80])
        result["is_essential_text"].append(False)
        events = []
        with (patch("vision.characters.ModelPool", FakePool), patch("vision.characters.load_magi", return_value=object()),
              patch("vision.characters._predict", return_value=(result, self.embeddings)) as predict):
            report = analyze_pages(path, progress=events.append)
        predict.assert_called_once()
        saved = json.loads(path.read_text(encoding="utf-8"))
        entry = report["pages"][0]
        self.assertEqual(report["completed"], 1)
        self.assertEqual(saved["magi_coverage"], entry["magi_coverage"])
        self.assertEqual(len(saved["blocks"]), 3)
        self.assertIn("文本区域覆盖 3/4", entry["message"])
        self.assertIn("1 个可能漏检待复核", events[-1]["message"])

    def test_external_page_change_during_model_run_is_preserved_without_db_rows(self):
        path = self.page("2")
        original = json.loads(path.read_text(encoding="utf-8"))
        edited = deepcopy(original)
        edited["blocks"][0]["translated_text"] = "User correction"
        def predict(*_):
            path.write_text(json.dumps(edited), encoding="utf-8")
            return self.result, self.embeddings
        with (patch("vision.characters.ModelPool", FakePool), patch("vision.characters.load_magi", return_value=object()),
              patch("vision.characters._predict", side_effect=predict)):
            report = analyze_pages(path)
        self.assertEqual(report["failed"], 1)
        self.assertIn("changed during Magi", report["pages"][0]["error"])
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), edited)
        self.assertEqual(len(self.memory.list_characters()), 2)
        self.assertEqual(self.memory.list_character_bindings(original["input_path"]), {})

    def test_original_hash_change_prevents_inference(self):
        path = self.page("2")
        path.with_suffix(".png").write_bytes(b"changed original")
        with patch("vision.characters.ModelPool", FakePool), patch("vision.characters.load_magi") as loader:
            report = analyze_pages(path)
        self.assertEqual(report["failed"], 1)
        loader.assert_not_called()

    def test_manual_binding_entered_during_inference_takes_precedence(self):
        path = self.page("2")
        record = json.loads(path.read_text(encoding="utf-8"))
        self.memory.upsert_character("rin", "凛")
        def predict(*_):
            self.memory.bind_character(record["input_path"], "b1", "rin", confirmed=True)
            return self.result, self.embeddings
        with (patch("vision.characters.ModelPool", FakePool), patch("vision.characters.load_magi", return_value=object()),
              patch("vision.characters._predict", side_effect=predict)):
            report = analyze_pages(path)
        self.assertEqual(report["completed"], 1)
        after = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(after["blocks"][0]["metadata"]["magi"]["character_id"], "rin")
        self.assertTrue(after["blocks"][0]["metadata"]["magi"]["confirmed"])
        self.assertEqual(len(self.memory.list_character_samples()), 1)

    def test_cancel_after_inference_keeps_page_and_db_unchanged(self):
        path = self.page("2")
        before = path.read_bytes()
        cancelled = False
        def predict(*_):
            nonlocal cancelled
            cancelled = True
            return self.result, self.embeddings
        def check():
            if cancelled:
                raise Cancelled()
        with (patch("vision.characters.ModelPool", FakePool), patch("vision.characters.load_magi", return_value=object()),
              patch("vision.characters._predict", side_effect=predict)):
            report = analyze_pages(path, cancel=check)
        self.assertTrue(report["cancelled"])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(self.memory.list_characters()), 2)

    def test_write_failure_rolls_back_new_characters_and_bindings(self):
        path = self.page("2")
        before = path.read_bytes()
        with (patch("vision.characters.ModelPool", FakePool), patch("vision.characters.load_magi", return_value=object()),
              patch("vision.characters._predict", return_value=(self.result, self.embeddings)),
              patch("vision.characters.os.replace", side_effect=OSError("disk error"))):
            report = analyze_pages(path)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(self.memory.list_characters()), 2)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_empty_folder_fails_before_loading_model(self):
        empty = self.root / "empty"
        empty.mkdir()
        with patch("vision.characters.load_magi", side_effect=AssertionError("No model needed")):
            with self.assertRaisesRegex(ValueError, "No page records"):
                analyze_pages(empty)

    def test_commit_failure_restores_page_or_retains_recovery_copy(self):
        for scenario in ("restore", "cancel", "external_edit", "restore_error"):
            with self.subTest(scenario=scenario):
                path = self.page(scenario)
                before = path.read_bytes()
                record = json.loads(before)
                external = b'{"manual_edit": true}'
                commit_attempted = False
                replace = os.replace

                def authorize(action, operation, *_):
                    nonlocal commit_attempted
                    if action == sqlite3.SQLITE_TRANSACTION and operation == "COMMIT":
                        commit_attempted = True
                        if scenario == "external_edit":
                            path.write_bytes(external)
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK

                def check():
                    if scenario == "cancel" and commit_attempted:
                        raise Cancelled()

                def replace_or_fail(source, destination):
                    if scenario == "restore_error" and commit_attempted:
                        raise OSError("restore denied")
                    return replace(source, destination)

                self.memory.connection.set_authorizer(authorize)
                try:
                    with patch("vision.characters.os.replace", side_effect=replace_or_fail):
                        with self.assertRaises(Exception) as caught:
                            _save_analysis(path, before, record, self.result, self.embeddings, self.memory, check)
                finally:
                    self.memory.connection.set_authorizer(None)
                self.assertTrue(commit_attempted)
                self.assertFalse(self.memory.connection.in_transaction)
                self.assertEqual(len(self.memory.list_characters()), 2)
                for table in ("character_bindings", "character_samples"):
                    self.assertEqual(self.memory.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)
                backups = list(self.root.glob(path.name + ".*.rollback"))
                if scenario in ("restore", "cancel"):
                    self.assertIsInstance(caught.exception, sqlite3.DatabaseError)
                    self.assertEqual(path.read_bytes(), before)
                    self.assertEqual(backups, [])
                else:
                    self.assertEqual(len(backups), 1)
                    self.assertEqual(backups[0].read_bytes(), before)
                    self.assertIn(str(backups[0]), str(caught.exception))
                    self.assertIsInstance(caught.exception.__cause__, sqlite3.DatabaseError)
                    if scenario == "external_edit":
                        self.assertEqual(path.read_bytes(), external)
                    else:
                        self.assertIn("magi", json.loads(path.read_bytes())["blocks"][0]["metadata"])
                self.assertEqual(list(self.root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
