import importlib.util
from copy import deepcopy
from pathlib import Path
import re
import tempfile
import unittest

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"evidence_{name}", EVIDENCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvidenceClaimsTests(unittest.TestCase):
    def test_readme_numbers_match_saved_evidence(self):
        results, has_readme = load_script("verify_claims").check()
        self.assertTrue(has_readme)
        self.assertGreaterEqual(len(results), 20)
        self.assertEqual([(name, problems) for name, _, _, problems in results if problems], [])

    def test_changed_readme_number_is_reported(self):
        verify = load_script("verify_claims")
        readme = (EVIDENCE.parent / "README_CN.md").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder) / "README_CN.md"
            changed.write_text(readme.replace("2.497 [0.707, 4.503]", "2.6 [0.707, 4.503]"), encoding="utf-8")
            results, _ = verify.check(changed)
        failed = {name: problems for name, _, _, problems in results if problems}
        self.assertEqual(failed, {"tojime VNTL - Qwen chrF++": ["text no longer in README_CN.md"]})

    def test_english_readme_numbers_match_saved_evidence(self):
        verify = load_script("verify_claims")
        results, has_readme = verify.check(verify.README_EN, "en")
        self.assertTrue(has_readme)
        self.assertGreaterEqual(len(results), 20)
        self.assertEqual([(name, problems) for name, _, _, problems in results if problems], [])

    def test_changed_english_readme_number_is_reported(self):
        verify = load_script("verify_claims")
        readme = verify.README_EN.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder) / "README.md"
            changed.write_text(readme.replace("14.89 percentage points lower", "14.8 percentage points lower"),
                               encoding="utf-8")
            results, _ = verify.check(changed, "en")
        failed = {name: problems for name, _, _, problems in results if problems}
        self.assertEqual(failed, {"tojime human severe-error difference": ["text no longer in README.md"]})

    def test_human_statistics_match_original_ratings(self):
        verify = load_script("verify_claims")
        self.assertEqual(verify.check_human_summary(), [])
        summary = verify.rebuild_human_summary()
        counts = {system: item["completed"] for system, item in summary["systems"].items()}
        self.assertEqual(counts, {"qwen-gold-c0": 48, "vntl-gold-c0": 47,
                                  "vntl-gold-c1200": 46, "vntl-ocr-c0": 47})
        self.assertEqual(sum(counts.values()), summary["completed_judgments"])
        expected = [(47, -1 / 47, [-3 / 46, 0]),
                    (46, -5 / 46, [-9 / 46, -2 / 48]),
                    (47, -7 / 47, [-13 / 46, -1 / 48])]
        for comparison, (pairs, delta, ci) in zip(summary["comparisons"], expected):
            self.assertEqual(len(comparison["paired_rows"]), pairs)
            self.assertEqual(comparison["coverage"]["paired"], pairs)
            self.assertEqual(sum(comparison["coverage"].values()), 48)
            self.assertEqual(sum(comparison["severe_transitions"].values()), pairs)
            metric = comparison["metrics"]["severe_error"]
            self.assertAlmostEqual(metric["delta"], delta)
            for actual, wanted in zip(metric["ci95"], ci):
                self.assertAlmostEqual(actual, wanted)

    def test_correct_total_cannot_hide_stale_system_counts(self):
        verify = load_script("verify_claims")
        summary = verify.load("results/R1/analysis/summary.json")
        self.assertEqual(summary["completed_judgments"], 188)
        for system in ("vntl-gold-c0", "vntl-ocr-c0"):
            summary["systems"][system].update(completed=46, partial=2)
        problems = verify.check_human_summary(summary=summary)
        for system in ("vntl-gold-c0", "vntl-ocr-c0"):
            self.assertTrue(any(f"systems.{system}.completed" in p for p in problems), problems)

    def test_newly_completed_rating_invalidates_saved_statistics(self):
        verify = load_script("verify_claims")
        ratings = verify.load("results/R1/source/ratings_1.json")
        row = next(row for row in ratings["rows"] if row["item"] == "item035-4")
        self.assertIsNone(row["severe_error"])
        row["severe_error"] = False
        problems = verify.check_human_summary(ratings=ratings)
        self.assertTrue(any("completed_judgments" in p for p in problems), problems)
        self.assertTrue(any("comparisons" in p for p in problems), problems)

    def test_pair_identity_is_checked_even_when_count_is_unchanged(self):
        verify = load_script("verify_claims")
        summary = verify.load("results/R1/analysis/summary.json")
        pairs = summary["comparisons"][0]["paired_rows"]
        pairs[1] = deepcopy(pairs[0])
        problems = verify.check_human_summary(summary=summary)
        self.assertTrue(any("paired_rows[1]" in p for p in problems), problems)

    def test_human_confidence_interval_is_recomputed(self):
        verify = load_script("verify_claims")
        summary = verify.load("results/R1/analysis/summary.json")
        summary["comparisons"][0]["metrics"]["severe_error"]["ci95"][0] -= .01
        problems = verify.check_human_summary(summary=summary)
        self.assertTrue(any("severe_error.ci95[0]" in p for p in problems), problems)

    def test_zero_and_false_are_complete_ratings(self):
        verify = load_script("verify_claims")
        ratings = verify.load("results/R1/source/ratings_1.json")
        row = next(row for row in ratings["rows"] if row["item"] == "item002-2")
        row.update(faithfulness=0, severe_error=False)
        rebuilt = verify.rebuild_human_summary(ratings=ratings)
        self.assertEqual(rebuilt["completed_judgments"], 188)
        self.assertEqual(rebuilt["systems"]["vntl-gold-c0"]["completed"], 47)

    def test_invalid_or_duplicate_ratings_are_rejected(self):
        verify = load_script("verify_claims")
        original = verify.load("results/R1/source/ratings_1.json")
        for field, value in (("faithfulness", True), ("faithfulness", 5),
                             ("naturalness", 0), ("severe_error", 0),
                             ("candidate", "not the archived translation"),
                             ("item", original["rows"][1]["item"])):
            with self.subTest(field=field, value=value):
                ratings = deepcopy(original)
                ratings["rows"][0][field] = value
                problems = verify.check_human_summary(ratings=ratings)
                self.assertTrue(any("human source validation" in p for p in problems), problems)

    def test_human_rebuild_does_not_mutate_source_or_summary(self):
        verify = load_script("verify_claims")
        inputs = {"summary": verify.load("results/R1/analysis/summary.json"),
                  "ratings": verify.load("results/R1/source/ratings_1.json")}
        before = deepcopy(inputs)
        verify.rebuild_human_summary(**inputs)
        self.assertEqual(inputs, before)

    def test_saved_predictions_reproduce_scores_and_intervals(self):
        try:
            import sacrebleu  # noqa: F401
        except ImportError:
            self.skipTest("optional scorer missing; install with: python -m pip install sacrebleu==2.6.0")
        checks = load_script("recompute").run()
        self.assertGreaterEqual(len(checks), 16)
        self.assertEqual([name for name, ok, *_ in checks if not ok], [])

    def test_published_evidence_has_no_personal_paths(self):
        forbidden = re.compile(r"Users[\\/]|Arch Wizard|ARCHWI~1")
        files = sorted(p for p in EVIDENCE.rglob("*") if p.suffix in {".md", ".json", ".csv"})
        self.assertGreater(len(files), 30)
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(forbidden.search(text), path.relative_to(EVIDENCE).as_posix())


if __name__ == "__main__":
    unittest.main()
