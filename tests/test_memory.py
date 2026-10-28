import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from core.memory import GlossaryMemory, contains_term


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "memory.sqlite3"
        self.memory = GlossaryMemory(self.path, "manga-a")

    def tearDown(self):
        self.memory.close()
        self.folder.cleanup()

    def test_reopen_persists_candidates_and_isolates_work_and_language(self):
        self.memory.upsert_term("ヘイズ", "Hayes", first_page="2.png")
        with GlossaryMemory(self.path, "manga-a") as reopened:
            term = reopened.match("ヘイズが来た")[0]
            self.assertEqual(term["target"], "Hayes")
            self.assertEqual(term["first_page"], "2.png")
            self.assertEqual(term["status"], "candidate")
        with GlossaryMemory(self.path, "manga-b") as other:
            self.assertEqual(other.list_terms(), [])
        with GlossaryMemory(self.path, "manga-a", "zh") as other:
            self.assertEqual(other.list_terms(), [])

    def test_locked_terms_and_provisional_spellings_are_not_auto_overwritten(self):
        self.memory.upsert_term("ヘイズ", "Hayes", locked=True, status="confirmed")
        self.assertFalse(self.memory.upsert_term("ヘイズ", "Haze"))
        self.assertEqual(self.memory.list_terms()[0]["target"], "Hayes")
        self.memory.upsert_term("ヘイズ", "Hays", locked=True, user_edit=True)
        self.assertEqual(self.memory.list_terms()[0]["target"], "Hays")
        self.memory.upsert_term("フィルフィ", "Philphy")
        self.assertFalse(self.memory.upsert_term("フィルフィ", "Filfy"))

    def test_alias_exact_match_and_ascii_word_boundaries(self):
        self.memory.upsert_term("ヘイズ", "Hayes", aliases=["ヘイス", "ヘイズ"])
        self.assertEqual(len(self.memory.match("ヘイスさん")), 1)
        self.assertEqual(self.memory.match("ヘイジ"), [])
        self.assertTrue(contains_term("Ann's book", "Ann"))
        self.assertFalse(contains_term("Hannah", "Ann", ignore_case=True))

    def test_fingerprint_changes_only_on_real_change_and_sees_other_connection(self):
        empty = self.memory.fingerprint()
        self.memory.upsert_term("ヘイズ", "Hayes")
        populated = self.memory.fingerprint()
        self.assertNotEqual(empty, populated)
        self.memory.upsert_term("ヘイズ", "Hayes")
        self.assertEqual(self.memory.fingerprint(), populated)
        with GlossaryMemory(self.path, "manga-a") as editor:
            editor.upsert_term("ヘイズ", "Hays", user_edit=True)
        self.assertNotEqual(self.memory.fingerprint(), populated)
        self.assertEqual(self.memory.match("ヘイズ")[0]["target"], "Hays")

    def test_affected_pages_include_aliases_and_new_terms(self):
        self.memory.record_page("2.png", ["ヘイスがいる"])
        self.memory.record_page("10.png", ["ヘイズだ"])
        self.memory.record_page("20.png", ["何ですか"])
        self.memory.upsert_term("ヘイズ", "Hayes", aliases=["ヘイス"])
        self.assertEqual(set(self.memory.affected_pages("ヘイズ")), {"2.png", "10.png"})
        self.memory.record_page("2.png", ["ヘイスがいる"])
        self.memory.delete_term("ヘイズ")
        self.assertIn("2.png", self.memory.affected_pages("ヘイズ"))

    def test_legacy_import_export_roundtrip_and_locked_import(self):
        legacy = Path(self.folder.name) / "glossary.json"
        legacy.write_text(json.dumps({"ヘイズ": "Hayes"}), encoding="utf-8")
        self.assertEqual(self.memory.import_json(legacy), 1)
        self.assertEqual(self.memory.import_json(legacy), 0)
        self.memory.upsert_term("ヘイズ", "Hays", aliases=["ヘイス"], locked=True,
                                status="confirmed", user_edit=True)
        self.assertEqual(self.memory.import_json(legacy), 0)
        exported = Path(self.folder.name) / "export.json"
        self.memory.export_json(exported)
        with GlossaryMemory(self.path, "new-work") as other:
            other.import_json(exported)
            self.assertEqual(other.list_terms(), self.memory.list_terms())
        with GlossaryMemory(self.path, "manga-a", "zh") as other:
            with self.assertRaises(ValueError):
                other.import_json(legacy)
            with self.assertRaises(ValueError):
                other.import_json(exported)

    def test_invalid_import_writes_nothing(self):
        file = Path(self.folder.name) / "bad.json"
        file.write_text(json.dumps({"terms": [
            {"source": "valid", "target": "Valid"}, {"source": "", "target": "Invalid"}
        ]}), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.memory.import_json(file)
        self.assertEqual(self.memory.list_terms(), [])

    def test_cached_rows_are_not_mutable_database_state(self):
        self.memory.upsert_term("ヘイズ", "Hayes", aliases=["ヘイス"])
        self.memory.list_terms()[0]["aliases"].append("bad")
        self.assertEqual(self.memory.list_terms()[0]["aliases"], ["ヘイス"])

    def test_imported_term_gets_first_observed_page_only_once(self):
        self.memory.upsert_term("ヘイズ", "Hayes", locked=True)
        self.memory.record_page("2.png", ["ヘイズだ"])
        self.memory.record_page("10.png", ["ヘイズがいる"])
        self.assertEqual(self.memory.list_terms()[0]["first_page"], "2.png")

    def test_legacy_seed_does_not_resurrect_deleted_term_after_restart(self):
        legacy = Path(self.folder.name) / "glossary.json"
        legacy.write_text(json.dumps({"ヘイズ": "Hayes"}), encoding="utf-8")
        self.assertEqual(self.memory.import_legacy_once(legacy), 1)
        self.memory.delete_term("ヘイズ")
        with GlossaryMemory(self.path, "manga-a") as reopened:
            self.assertEqual(reopened.import_legacy_once(legacy), 0)
            self.assertEqual(reopened.list_terms(), [])
            # Explicit user import retains its normal, repeatable behavior.
            self.assertEqual(reopened.import_json(legacy), 1)
            self.assertEqual(reopened.match("ヘイズ")[0]["target"], "Hayes")

    def test_legacy_seed_scope_isolation_and_existing_terms_preserved(self):
        legacy = Path(self.folder.name) / "glossary.json"
        legacy.write_text(json.dumps({"ヘイズ": "Hayes", "フィルフィ": "Philphy"}), encoding="utf-8")
        self.memory.upsert_term("ヘイズ", "Hays", locked=True)
        self.assertEqual(self.memory.import_legacy_once(legacy), 1)
        self.assertEqual(self.memory.match("ヘイズ")[0]["target"], "Hays")
        with GlossaryMemory(self.path, "manga-b") as other:
            self.assertEqual(other.import_legacy_once(legacy), 2)

    def test_failed_seed_does_not_mark_scope_as_imported(self):
        legacy = Path(self.folder.name) / "glossary.json"
        legacy.write_text('{"terms":[{"source":"", "target":"Bad"}]}', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.memory.import_legacy_once(legacy)
        legacy.write_text(json.dumps({"ヘイズ": "Hayes"}), encoding="utf-8")
        self.assertEqual(self.memory.import_legacy_once(legacy), 1)

    def test_old_schema_migration_keeps_terms_and_adds_empty_reading(self):
        old_path = Path(self.folder.name) / "old.sqlite3"
        with sqlite3.connect(old_path) as connection:
            connection.executescript("""
                CREATE TABLE terms(work TEXT,language TEXT,source TEXT,target TEXT,
                    aliases TEXT DEFAULT '[]',first_page TEXT DEFAULT '',locked INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'candidate',PRIMARY KEY(work,language,source));
                INSERT INTO terms(work,language,source,target) VALUES('manga-a','en','凛','Rin');
            """)
        connection.close()
        with GlossaryMemory(old_path, "manga-a") as migrated:
            self.assertEqual(migrated.list_terms()[0]["target"], "Rin")
            self.assertEqual(migrated.list_terms()[0]["reading"], "")
            migrated.upsert_term("凛", "Rin", reading="りん", user_edit=True)
        with GlossaryMemory(old_path, "manga-a") as reopened:
            self.assertEqual(reopened.list_terms()[0]["reading"], "りん")

    def test_reading_preservation_explicit_clear_and_roundtrip(self):
        self.memory.upsert_term("凛", "Rin", reading="りん", locked=True)
        self.assertFalse(self.memory.upsert_term("凛", "Rin", reading="リン"))
        self.memory.upsert_term("凛", "Rin", user_edit=True)
        self.assertEqual(self.memory.list_terms()[0]["reading"], "りん")
        exported = Path(self.folder.name) / "readings.json"
        self.memory.export_json(exported)
        with GlossaryMemory(self.path, "another") as other:
            other.import_json(exported)
            self.assertEqual(other.list_terms()[0]["reading"], "りん")
        self.memory.upsert_term("凛", "Rin", reading="", user_edit=True)
        self.assertEqual(self.memory.list_terms()[0]["reading"], "")

    def test_characters_persist_by_work_and_are_shared_between_languages(self):
        self.assertEqual([row["id"] for row in self.memory.list_characters()], ["unknown", "narrator"])
        voice = {"engine_uuid": "engine", "speaker_uuid": "speaker", "style_id": 77,
                 "params": {"speedScale": 1.1}}
        character = self.memory.upsert_character(None, "凛", voice)
        character_id = character["id"]
        self.assertEqual(self.memory.upsert_character(character_id, "Rin")["voice"], voice)
        with GlossaryMemory(self.path, "manga-a", "zh") as same_work:
            self.assertIn(character_id, [row["id"] for row in same_work.list_characters()])
            self.assertEqual(same_work.list_characters()[2]["name"], "Rin")
        with GlossaryMemory(self.path, "manga-b") as other:
            self.assertEqual([row["id"] for row in other.list_characters()], ["unknown", "narrator"])
        with GlossaryMemory(self.path, "manga-a") as reopened:
            self.assertEqual(reopened.list_characters()[2]["voice"], voice)

    def test_confirmed_binding_cannot_be_overwritten_by_suggestions(self):
        self.memory.upsert_character("rin", "凛")
        self.assertTrue(self.memory.bind_character("2.png", "bubble-1", "rin", confirmed=True))
        self.assertFalse(self.memory.bind_character("2.png", "bubble-1", "unknown"))
        self.assertEqual(self.memory.get_character_binding("2.png", "bubble-1"),
                         {"character_id": "rin", "confirmed": True})
        self.assertTrue(self.memory.bind_character("2.png", "bubble-1", "narrator", confirmed=True, user_edit=True))
        self.assertEqual(self.memory.list_character_bindings("2.png"),
                         {"bubble-1": {"character_id": "narrator", "confirmed": True}})
        with GlossaryMemory(self.path, "other") as other:
            self.assertIsNone(other.get_character_binding("2.png", "bubble-1"))
            with self.assertRaises(ValueError):
                other.bind_character("2.png", "bubble-1", "rin")

    def test_magi_samples_are_idempotent_persistent_and_work_scoped(self):
        self.memory.upsert_character("rin", "凛")
        vector = [0.0] * 768
        vector[0] = 1.0
        self.memory.save_character_sample("rin", "2.png", [1, 2, 30, 40], vector)
        vector[1] = 0.5
        self.memory.save_character_sample("rin", "2.png", [1.0, 2.0, 30.0, 40.0], vector)
        with GlossaryMemory(self.path, "manga-a") as reopened:
            samples = reopened.list_character_samples()
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["embedding"], vector)
            self.assertEqual(samples[0]["box"], [1, 2, 30, 40])
        with GlossaryMemory(self.path, "other") as other:
            self.assertEqual(other.list_character_samples(), [])
        with self.assertRaises(ValueError):
            self.memory.save_character_sample("rin", "2.png", [1, 2, 30, 40], [0.0] * 767)
        with self.assertRaises(ValueError):
            self.memory.save_character_sample("rin", "2.png", [1, 2, 30, 40], [float("nan")] * 768)
        with self.assertRaises(ValueError):
            self.memory.save_character_sample("rin", "2.png", [1, 2, 0, 40], vector)

    def test_task_history_is_compact_idempotent_and_scope_isolated(self):
        self.memory.record_run("run-1", {"started_at": "2000-01-01T01:00:00", "completed": 2,
                                         "failed": 1, "report_path": "report.json", "large_debug": [1, 2]})
        self.memory.record_run("run-2", {"started_at": "2000-01-01T02:00:00", "completed": 3})
        self.memory.record_run("run-2", {"started_at": "2000-01-01T02:00:00", "cancelled": True})
        runs = self.memory.list_runs()
        self.assertEqual([row["run_id"] for row in runs], ["run-2", "run-1"])
        self.assertEqual(runs[0]["status"], "cancelled")
        self.assertEqual(runs[1]["status"], "partial")
        self.assertNotIn("large_debug", runs[1])
        self.assertEqual(len(self.memory.list_runs(limit=1)), 1)
        with GlossaryMemory(self.path, "manga-a", "zh") as other:
            self.assertEqual(other.list_runs(), [])
        with GlossaryMemory(self.path, "other") as other:
            self.assertEqual(other.list_runs(), [])


if __name__ == "__main__":
    unittest.main()
