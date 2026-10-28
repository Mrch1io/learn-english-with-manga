"""Offline checks for portable showcase assets and preservation of user edits."""
from dataclasses import fields
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from core.runtime import TextBlock
from pipeline.editing import PageEditorSession
from vision.graphics import prepare_layout
from core.memory import GlossaryMemory
from pipeline.batch import file_hash
import presentation
from reader import collect_pages
from reader.records import read_record as _read_record
from reader.images import translated_preview as _translated_preview


class PresentationDemoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_materialized_page_opens_reader_and_editor_without_models(self):
        with patch("core.runtime.ModelPool.acquire", side_effect=AssertionError("No models")), \
             patch("translation.translate", side_effect=AssertionError("No translation")), \
             patch("audio.speech.synthesize_cached", side_effect=AssertionError("No TTS")):
            path = presentation.ensure_demo(self.root / "session")
            pages, errors = collect_pages(path)
            self.assertFalse(errors)
            self.assertEqual(len(pages), 1)
            record = _read_record(path)
            self.assertTrue(record["presentation_fixture"])
            self.assertNotEqual(record["input_sha256"], record["output_sha256"])
            preview, full = _translated_preview(record, path, None, (480, 600))
            self.assertEqual(full.size, tuple(record["image_size"]))
            self.assertLessEqual(preview.height, 600)
            pool = SimpleNamespace(local_files_only=True, close=Mock())
            session = PageEditorSession(path, pool=pool)
            self.assertIsNotNone(session.clean)
            self.assertEqual(len(session.blocks), 13)
            self.assertEqual(Path(session.config.memory_db), path.parent / "memory.sqlite3")
            self.assertTrue(all(block.status == "translated" for block in session.blocks))
            session.close()

    def test_bundle_relocates_and_does_not_embed_host_paths(self):
        target = self.root / "moved checkout" / "presentation"
        shutil.copytree(presentation.BUNDLE, target, ignore=shutil.ignore_patterns("__pycache__"))
        with patch.object(presentation, "BUNDLE", target):
            path = presentation.ensure_demo(self.root / "new user data")
        record = _read_record(path)
        self.assertTrue(Path(record["input_path"]).is_file())
        self.assertEqual(Path(record["output_path"]), path.with_suffix(".png"))
        bundled = (target / "daily-life.json").read_text(encoding="utf-8")
        self.assertNotIn("C:\\", bundled)
        self.assertNotIn("reference_text", bundled)

    def test_three_consecutive_pages_share_memory_and_open_translated(self):
        with patch("core.runtime.ModelPool.acquire", side_effect=AssertionError("No models")), \
             patch("translation.translate", side_effect=AssertionError("No translation")), \
             patch("audio.speech.synthesize_cached", side_effect=AssertionError("No TTS")):
            paths = presentation.ensure_demo_pages(self.root / "session")
            pages, errors = collect_pages(paths)
            self.assertFalse(errors)
            self.assertEqual([page["path"] for page in pages], [str(path) for path in paths])
            self.assertEqual([page["label"] for page in pages], ["Page 4", "Page 5", "Page 6"])
            records = [_read_record(path) for path in paths]
            self.assertEqual(len({record["input_sha256"] for record in records}), 3)
            self.assertEqual({record["config"]["memory_db"] for record in records},
                             {str(paths[0].parent / "memory.sqlite3")})
            for path, record in zip(paths, records):
                with self.subTest(page=path.parent.name):
                    preview, full = _translated_preview(record, path, None, (480, 600))
                    self.assertEqual(full.size, tuple(record["image_size"]))
                    self.assertLessEqual(preview.height, 600)
                    self.assertEqual(file_hash(record["output_path"]), record["output_sha256"])
                    self.assertTrue(all(block["status"] == "translated" for block in record["blocks"]))
                    with GlossaryMemory(record["config"]["memory_db"], record["work"]) as memory:
                        self.assertTrue(memory.list_character_bindings(record["input_path"]))
                        self.assertEqual(len(memory.list_characters()), 4)

    def test_extending_legacy_sample_and_reopening_preserve_all_page_edits(self):
        first = presentation.ensure_demo(self.root / "session")
        record = _read_record(first)
        record["blocks"][0]["translated_text"] = "My original page correction"
        first.write_text(json.dumps(record), encoding="utf-8")
        before = file_hash(first)
        paths = presentation.ensure_demo_pages(first.parent)
        self.assertEqual(paths[0], first)
        self.assertEqual(file_hash(first), before)
        for path in paths[1:]:
            record = _read_record(path)
            record["blocks"][0]["translated_text"] = "My continuation correction"
            path.write_text(json.dumps(record), encoding="utf-8")
        hashes = {path: file_hash(path) for path in paths}
        with patch("shutil.copyfileobj", side_effect=AssertionError("Do not overwrite user assets")):
            self.assertEqual(presentation.ensure_demo_pages(first.parent), paths)
        self.assertEqual({path: file_hash(path) for path in paths}, hashes)

    def test_continuations_keep_artwork_outside_dialogue_and_fit_captions(self):
        allowed = {field.name for field in fields(TextBlock)}
        for name in ("page-005", "page-006"):
            bundle = presentation.BUNDLE / name
            record = json.loads((bundle / "daily-life.json").read_text(encoding="utf-8"))
            with self.subTest(page=name):
                blocks = [TextBlock(**{key: value for key, value in row.items() if key in allowed})
                          for row in record["blocks"]]
                with Image.open(bundle / "original.jpg") as source, Image.open(bundle / "daily-life.png") as rendered:
                    original = source.convert("RGB")
                    layouts = prepare_layout(original, blocks, min_font_size=12, max_font_size=26)
                    self.assertTrue(all(not layout.overflow for layout in layouts.values()))
                    outside = np.ones((source.height, source.width), dtype=bool)
                    for block in blocks:
                        x1, y1, x2, y2 = block.box
                        outside[y1:y2, x1:x2] = False
                    self.assertTrue(np.array_equal(np.asarray(original)[outside], np.asarray(rendered.convert("RGB"))[outside]))
                self.assertEqual(file_hash(bundle / "daily-life.png"), record["output_sha256"])
                self.assertEqual(file_hash(bundle / record["clean_image"]["file"]), record["clean_image"]["sha256"])

    def test_reopening_preserves_manual_edits_without_recopying(self):
        path = presentation.ensure_demo(self.root / "session")
        record = _read_record(path)
        record["blocks"][0]["translated_text"] = "My correction"
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        before = file_hash(path)
        with patch("shutil.copyfileobj", side_effect=AssertionError("Never recopy an existing session")):
            self.assertEqual(presentation.ensure_demo(path.parent), path)
        self.assertEqual(file_hash(path), before)

    def test_partial_destination_conflict_is_preserved(self):
        destination = self.root / "session"
        destination.mkdir()
        user_asset = destination / "original.jpg"
        user_asset.write_bytes(b"user-owned-file")
        with self.assertRaises(FileExistsError):
            presentation.ensure_demo(destination)
        self.assertEqual(user_asset.read_bytes(), b"user-owned-file")
        self.assertFalse((destination / "daily-life.json").exists())

    def test_curated_characters_and_only_clear_dialogue_bindings_are_seeded(self):
        path = presentation.ensure_demo(self.root / "session")
        record = _read_record(path)
        with GlossaryMemory(record["config"]["memory_db"], record["work"]) as memory:
            names = {role["id"]: role["name"] for role in memory.list_characters()}
            self.assertEqual(names["presentation-demo:shopkeeper"], "Shopkeeper")
            self.assertEqual(names["presentation-demo:meru"], "Meru")
            actual = memory.list_character_bindings(record["input_path"])
            self.assertEqual(len(actual), 11)
            for index, expected in enumerate((None, "shopkeeper", "meru", "shopkeeper", "meru", "meru",
                                               "shopkeeper", "shopkeeper", "meru", "meru", None,
                                               "shopkeeper", "shopkeeper")):
                key = record["blocks"][index]["block_id"]
                if expected is None:
                    self.assertNotIn(key, actual)
                else:
                    self.assertEqual(actual[key], {"character_id": f"presentation-demo:{expected}", "confirmed": True})
            self.assertTrue(all(not role["voice"] for role in memory.list_characters()))
            self.assertFalse(memory.list_character_samples())
        provenance = json.loads((path.parent / "character-seeds.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["source"], "curated_demo")
        self.assertTrue(provenance["not_model_output"])

    def test_legacy_session_adds_missing_roles_without_rewriting_page_assets(self):
        with patch.object(presentation, "_seed_characters"):
            path = presentation.ensure_demo(self.root / "session")
        record = _read_record(path)
        before = {name: file_hash(path.parent / name) for name in
                  ("daily-life.json", "original.jpg", "daily-life.png", "daily-life.clean.png")}
        with GlossaryMemory(record["config"]["memory_db"], record["work"]) as memory:
            self.assertEqual(len(memory.list_characters()), 2)
        presentation.ensure_demo(path.parent)
        with GlossaryMemory(record["config"]["memory_db"], record["work"]) as memory:
            self.assertEqual(len(memory.list_characters()), 4)
        self.assertEqual(before, {name: file_hash(path.parent / name) for name in before})

    def test_existing_names_voices_unknown_and_manual_bindings_always_win(self):
        path = presentation.ensure_demo(self.root / "session")
        record = _read_record(path)
        first, second, missing = [record["blocks"][index]["block_id"] for index in (1, 2, 3)]
        voice = {"engine_id": "chosen-engine", "style_id": 42}
        with GlossaryMemory(record["config"]["memory_db"], record["work"]) as memory:
            memory.upsert_character("presentation-demo:meru", "My renamed character", voice)
            memory.upsert_character("user-person", "My own person")
            memory.bind_character(record["input_path"], first, "unknown", confirmed=False, user_edit=True)
            memory.bind_character(record["input_path"], second, "user-person", confirmed=True, user_edit=True)
            with memory.connection:
                memory.connection.execute("DELETE FROM character_bindings WHERE work=? AND page_key=? AND block_id=?",
                                          (record["work"], record["input_path"], missing))
        presentation.ensure_demo(path.parent)
        with GlossaryMemory(record["config"]["memory_db"], record["work"]) as memory:
            role = next(role for role in memory.list_characters() if role["id"] == "presentation-demo:meru")
            self.assertEqual(role["name"], "My renamed character")
            self.assertEqual(role["voice"], voice)
            self.assertEqual(memory.get_character_binding(record["input_path"], first),
                             {"character_id": "unknown", "confirmed": False})
            self.assertEqual(memory.get_character_binding(record["input_path"], second),
                             {"character_id": "user-person", "confirmed": True})
            self.assertEqual(memory.get_character_binding(record["input_path"], missing)["character_id"],
                             "presentation-demo:shopkeeper")

    def test_unrelated_scopes_and_external_databases_are_not_opened(self):
        with patch.object(presentation, "_seed_characters"):
            path = presentation.ensure_demo(self.root / "session")
        base = path.read_text(encoding="utf-8")
        edits = (lambda record: record.update(work="other-work"),
                 lambda record: record["config"].update(work="other-work"),
                 lambda record: record["config"].update(memory_db=str(self.root / "user-memory.sqlite3")),
                 lambda record: record.update(input_path=str(self.root / "other.jpg")),
                 lambda record: record.update(input_sha256="changed"),
                 lambda record: record.update(presentation_fixture={}),
                 lambda record: record["presentation_fixture"].update(source_page_id="other-page"))
        for index, edit in enumerate(edits):
            record = json.loads(base)
            edit(record)
            path.write_text(json.dumps(record), encoding="utf-8")
            before = file_hash(path)
            with self.subTest(index=index), patch("core.memory.GlossaryMemory", side_effect=AssertionError("No foreign DB access")):
                self.assertEqual(presentation.ensure_demo(path.parent), path)
            self.assertEqual(file_hash(path), before)
        self.assertFalse((self.root / "user-memory.sqlite3").exists())
        self.assertFalse((path.parent / "memory.sqlite3").exists())

    def test_changed_source_or_geometry_gets_no_new_binding(self):
        with patch.object(presentation, "_seed_characters"):
            path = presentation.ensure_demo(self.root / "session")
        record = _read_record(path)
        record["blocks"][1]["source_text"] = "User source correction"
        record["blocks"][2]["box"][0] += 1
        path.write_text(json.dumps(record), encoding="utf-8")
        presentation.ensure_demo(path.parent)
        with GlossaryMemory(record["config"]["memory_db"], record["work"]) as memory:
            bindings = memory.list_character_bindings(record["input_path"])
            self.assertNotIn(record["blocks"][1]["block_id"], bindings)
            self.assertNotIn(record["blocks"][2]["block_id"], bindings)
            self.assertEqual(len(bindings), 9)

    def test_seed_transaction_rolls_back_every_role_and_binding_on_failure(self):
        with patch.object(presentation, "_seed_characters"):
            path = presentation.ensure_demo(self.root / "session")
        record = _read_record(path)
        with GlossaryMemory(record["config"]["memory_db"], record["work"]) as memory:
            with memory.connection:
                memory.connection.execute("""CREATE TRIGGER reject_demo_binding
                    BEFORE INSERT ON character_bindings
                    WHEN NEW.block_id = 'presentation:tencho_isoro:p0004:t005'
                    BEGIN SELECT RAISE(ABORT, 'test insertion failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            presentation.ensure_demo(path.parent)
        with GlossaryMemory(record["config"]["memory_db"], record["work"]) as memory:
            self.assertEqual({role["id"] for role in memory.list_characters()}, {"unknown", "narrator"})
            self.assertFalse(memory.list_character_bindings(record["input_path"]))

    def test_all_captions_fit_and_original_art_outside_regions_is_unchanged(self):
        bundle = presentation.BUNDLE
        record = json.loads((bundle / "daily-life.json").read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(TextBlock)}
        blocks = [TextBlock(**{key: value for key, value in row.items() if key in allowed})
                  for row in record["blocks"]]
        with Image.open(bundle / "original.jpg") as source, Image.open(bundle / "daily-life.png") as rendered:
            original = source.convert("RGB")
            layouts = prepare_layout(original, blocks, min_font_size=12, max_font_size=26)
            self.assertTrue(all(not layout.overflow for layout in layouts.values()))
            outside = np.ones((source.height, source.width), dtype=bool)
            for block in blocks:
                x1, y1, x2, y2 = block.box
                outside[y1:y2, x1:x2] = False
            self.assertTrue(np.array_equal(np.asarray(original)[outside], np.asarray(rendered.convert("RGB"))[outside]))
            self.assertFalse(np.array_equal(np.asarray(original), np.asarray(rendered.convert("RGB"))))
        self.assertEqual(file_hash(bundle / "daily-life.png"), record["output_sha256"])
        self.assertEqual(file_hash(bundle / record["clean_image"]["file"]), record["clean_image"]["sha256"])


if __name__ == "__main__":
    unittest.main()
