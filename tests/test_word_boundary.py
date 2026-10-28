"""Local dictionary checks for a comma inside an already linked word.

Run: .venv\\Scripts\\python.exe -m unittest tests.test_word_boundary -v
No GPU, network, OCR, translation model, or reference English is used.
"""
from copy import deepcopy
from dataclasses import asdict
import unittest
from unittest.mock import patch

from core.runtime import TextBlock
from translation import vntl


def pair(first="はな、", second="せ..."):
    return (TextBlock((132, 447, 172, 535), source_text=first, block_id="right",
                      metadata={"manual_ocr": True}),
            TextBlock((93, 477, 132, 561), source_text=second, block_id="left"))


class WordBoundaryTests(unittest.TestCase):
    def normalize(self, first, second, *, terms=()):
        blocks = pair(first, second)
        before = deepcopy([asdict(block) for block in blocks])
        result = vntl.normalize_word_boundary(*blocks, terms)
        self.assertEqual([asdict(block) for block in blocks], before)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        return result

    def assert_preserved(self, first, second, *, terms=()):
        source, normalization = self.normalize(first, second, terms=terms)
        self.assertEqual(source, first.strip() + second.strip())
        self.assertIsNone(normalization)

    def test_known_word_crosses_boundary_for_each_comma_width(self):
        for comma in ("、", "，", ","):
            with self.subTest(comma=comma):
                source, normalization = self.normalize("はな" + comma, "せ...")
                self.assertEqual(source, "はなせ...")
                self.assertIsInstance(normalization, dict)
                self.assertTrue(normalization)
                self.assertEqual(normalization["removed"], comma)
                self.assertEqual(normalization["boundary_after"], 2)

    def test_dictionary_evidence_is_not_specific_to_the_observed_word(self):
        for first, second, expected in (("はな、", "せて！", "はなせて！"),
                                        ("あな、", "た？", "あなた？")):
            with self.subTest(first=first, second=second):
                source, normalization = self.normalize(first, second)
                self.assertEqual(source, expected)
                self.assertIsNotNone(normalization)

    def test_final_question_exclamation_and_ellipsis_are_preserved(self):
        for suffix in ("...", "…", "……", "！", "？", "！？"):
            with self.subTest(suffix=suffix):
                source, normalization = self.normalize("はな、", "せ" + suffix)
                self.assertEqual(source, "はなせ" + suffix)
                self.assertIsNotNone(normalization)

    def test_only_existing_outer_strip_behavior_is_applied(self):
        source, normalization = self.normalize(" \nはな、\t", "\nせ... ")
        self.assertEqual(source, "はなせ...")
        self.assertIsNotNone(normalization)

    def test_comma_after_the_complete_word_is_preserved(self):
        for ending in ("、", "，", ",", "、..."):
            with self.subTest(ending=ending):
                source, normalization = self.normalize("はな、", "せ" + ending)
                self.assertEqual(source, "はなせ" + ending)
                self.assertIsNotNone(normalization)

    def test_normal_word_and_clause_boundaries_keep_commas(self):
        for first, second in (("そう、", "だね"), ("きみ、", "たち"),
                              ("きみ、", "も"), ("ありがとう、", "ね"),
                              ("あの、", "ね"), ("いけ、", "ない"),
                              ("本当のことを、", "教えてください。")):
            with self.subTest(first=first, second=second):
                self.assert_preserved(first, second)

    def test_independent_right_word_blocks_accidental_dictionary_compound(self):
        # Both はなそう and あれこれ are dictionary words after concatenation.
        # That alone is not enough evidence to delete a normal address/pause.
        for first, second in (("はな、", "そう..."), ("あれ、", "これ"),
                              ("おま、", "え"), ("やめ、", "ろ！"),
                              ("たす、", "けて！")):
            with self.subTest(first=first, second=second):
                self.assert_preserved(first, second)

    def test_adverb_compound_does_not_swallow_a_possible_normal_pause(self):
        # The dictionary accepts 如何に, but いか + に can also be a noun + particle.
        self.assert_preserved("いか、", "に")

    def test_internal_or_repeated_commas_are_not_stripped(self):
        for first, second in (("は、な、", "せ..."), ("はな、、", "せ..."),
                              ("はな,,", "せ..."), ("はな、，", "せ..."),
                              ("はな、", "せ、て"), ("はな、", "、せ...")):
            with self.subTest(first=first, second=second):
                self.assert_preserved(first, second)

    def test_noncomma_boundary_punctuation_is_untouched(self):
        for suffix in ("", "。", "！", "？", "...", "…", "…、", "、...", "：", "；"):
            with self.subTest(suffix=suffix):
                self.assert_preserved("はな" + suffix, "せ...")

    def test_quotes_spaces_kanji_and_unsupported_scripts_are_untouched(self):
        for first, second in (("「はな、", "せ...」"), ("は な、", "せ..."),
                              ("放、", "せ..."), ("Hana,", "se..."),
                              ("はな、", "「せ...」"), ("はな、", "せ て")):
            with self.subTest(first=first, second=second):
                self.assert_preserved(first, second)

    def test_empty_or_long_candidates_and_unknown_words_are_untouched(self):
        for first, second in (("", "せ..."), ("、", "せ..."), ("はな、", ""),
                              ("はな、", "..."), ("はなはなはな、", "せせせ..."),
                              ("ヌヴェ、", "せ...")):
            with self.subTest(first=first, second=second):
                self.assert_preserved(first, second)

    def test_confirmed_or_locked_whole_fragment_terms_preserve_punctuation(self):
        for term in ({"source": "はな", "target": "Hana", "aliases": [],
                      "status": "confirmed", "locked": False},
                     {"source": "はな", "target": "Hana", "aliases": [],
                      "status": "candidate", "locked": True},
                     {"source": "花", "target": "Hana", "aliases": ["はな"],
                      "status": "confirmed", "locked": False},
                     {"source": "せ", "target": "Se", "aliases": [],
                      "status": "confirmed", "locked": False}):
            with self.subTest(term=term):
                self.assert_preserved("はな、", "せ...", terms=[term])

    def test_unconfirmed_candidate_and_partial_term_do_not_override_word_evidence(self):
        for term in ({"source": "はな", "target": "Hana", "aliases": [],
                      "status": "candidate", "locked": False},
                     {"source": "は", "target": "Ha", "aliases": [],
                      "status": "confirmed", "locked": True}):
            with self.subTest(term=term):
                source, normalization = self.normalize("はな、", "せ...", terms=[term])
                self.assertEqual(source, "はなせ...")
                self.assertIsNotNone(normalization)

    def test_missing_analyzer_preserves_source_instead_of_guessing(self):
        with patch("translation.vntl._word_tokens", return_value=None) as tokens:
            self.assert_preserved("はな、", "せ...")
        self.assertGreaterEqual(tokens.call_count, 1)


class TranslationSourceTests(unittest.TestCase):
    def block(self, metadata=None):
        return TextBlock((132, 447, 172, 535), source_text="はな、せ...", block_id="unit",
                         metadata={} if metadata is None else metadata)

    def test_ordinary_block_uses_exact_original_source(self):
        block = self.block({"manual_ocr": True})
        self.assertEqual(vntl.translation_source(block), block.source_text)

    def test_current_prepared_unit_is_used_without_mutating_original(self):
        block = self.block({"continuation": {
            "source_text": "はな、せ...", "translation_source_text": "はなせ..."}})
        before = deepcopy(asdict(block))
        self.assertEqual(vntl.translation_source(block), "はなせ...")
        self.assertEqual(asdict(block), before)

    def test_source_edit_invalidates_previous_prepared_input(self):
        block = self.block({"continuation": {
            "source_text": "はな、せ...", "translation_source_text": "はなせ..."}})
        block.source_text = "はな、そう..."
        before = deepcopy(asdict(block))
        self.assertEqual(vntl.translation_source(block), "はな、そう...")
        self.assertEqual(asdict(block), before)

    def test_empty_or_malformed_metadata_falls_back_to_source(self):
        for metadata in ({}, None, [], "", {"continuation": None}, {"continuation": []},
                         {"continuation": ""}, {"continuation": {}},
                         {"continuation": {"translation_source_text": "はなせ..."}}):
            with self.subTest(metadata=metadata):
                block = self.block()
                block.metadata = deepcopy(metadata)
                self.assertEqual(vntl.translation_source(block), block.source_text)

    def test_empty_or_nontext_prepared_value_falls_back_to_source(self):
        for prepared in (None, "", " \n\t", 0, [], {}):
            with self.subTest(prepared=prepared):
                block = self.block({"continuation": {
                    "source_text": "はな、せ...", "translation_source_text": prepared}})
                self.assertEqual(vntl.translation_source(block), block.source_text)

    def test_saved_reader_link_is_not_an_active_translation_unit(self):
        block = self.block({"translation_unit": {
            "source_text": "はな、せ...", "translation_source_text": "はなせ..."}})
        self.assertEqual(vntl.translation_source(block), block.source_text)


if __name__ == "__main__":
    unittest.main()
