"""Sentence continuity and word allocation; no model, HTTP, or GPU required."""
from copy import deepcopy
import re
import unittest

from core.runtime import TextBlock
from translation.vntl import continuation_pairs, split_sentence


def bubble(key, source, box=(0, 0, 100, 100)):
    return TextBlock(box, source_text=source, block_id=key)


class ContinuationTests(unittest.TestCase):
    def test_prohibition_grammar_covers_other_topics_and_conjugations(self):
        examples = [("海賊版は読んじゃ", "ダメですっ"),
                    ("この薬は飲んじゃ", "いけないんだ。"),
                    ("ここで食べちゃ", "駄目だよ！"),
                    ("秘密を話しては", "なりません。"),
                    ("勝手に読んでは", "ならない。"),
                    ("窓を開けては", "いけません。")]
        for first, second in examples:
            with self.subTest(first=first):
                self.assertEqual(continuation_pairs([bubble("a", first), bubble("b", second)]),
                                 [(0, 1, "prohibition")])

    def test_permission_is_not_classified_as_a_prohibition(self):
        for first, second in [("ここで休んでも", "いいよ。"), ("食べても", "良いんです。")]:
            self.assertEqual(continuation_pairs([bubble("a", first), bubble("b", second)]),
                             [(0, 1, "permission")])

    def test_complete_sentences_quotes_and_unrelated_dialogue_do_not_join(self):
        examples = [("もう読んじゃった", "ダメですっ"), ("読んじゃ。", "ダメですっ"),
                    ("読んじゃ．", "ダメですっ"), ("『読んじゃ』", "ダメですっ"),
                    ("読んじゃ", "『ダメですっ』"), ("読んじゃ", "ダメージを受けた。"),
                    ("食べても", "いい加減にしなさい。"), ("こんにちは。", "ダメですっ"),
                    ("読んじゃ", "先に帰るね。")]
        for first, second in examples:
            with self.subTest(first=first, second=second):
                self.assertEqual(continuation_pairs([bubble("a", first), bubble("b", second)]), [])

    def test_only_ellipsis_and_whitespace_are_ignored(self):
        for suffix in ("...", "．．．", "…", "……  ", "⋯"):
            with self.subTest(suffix=suffix):
                self.assertEqual(continuation_pairs([bubble("a", "読んじゃ" + suffix),
                                                     bubble("b", "  ダメですっ")]),
                                 [(0, 1, "prohibition")])

    def test_ocr_failure_does_not_become_a_joint_request(self):
        items = [bubble("a", "読んじゃ"), bubble("b", "ダメですっ")]
        items[1].status, items[1].error = "failed", "OCR: damaged crop"
        self.assertEqual(continuation_pairs(items), [])

    def test_two_different_confirmed_speakers_do_not_join(self):
        items = [bubble("a", "読んじゃ"), bubble("b", "ダメですっ")]
        bindings = {"a": {"character_id": "one", "confirmed": True},
                    "b": {"character_id": "two", "confirmed": True}}
        self.assertEqual(continuation_pairs(items, bindings), [])
        bindings["b"]["confirmed"] = False
        self.assertEqual(continuation_pairs(items, bindings), [(0, 1, "prohibition")])
        bindings["b"].update(character_id="one", confirmed=True)
        self.assertEqual(continuation_pairs(items, bindings), [(0, 1, "prohibition")])
        bindings["b"]["character_id"] = "unknown"
        self.assertEqual(continuation_pairs(items, bindings), [(0, 1, "prohibition")])

    def test_pairs_are_adjacent_disjoint_and_leave_input_untouched(self):
        items = [bubble("a", "読んじゃ"), bubble("b", "ダメですっ"), bubble("c", "食べても"),
                 bubble("d", "いいよ。"), bubble("e", "飲んじゃ"), bubble("f", "別の話。"),
                 bubble("g", "ダメです。")]
        before = deepcopy(items)
        self.assertEqual(continuation_pairs(items), [(0, 1, "prohibition"), (2, 3, "permission")])
        self.assertEqual(items, before)
        self.assertEqual(continuation_pairs([]), [])
        self.assertEqual(continuation_pairs(items[:1]), [])


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.left = bubble("a", "海賊版は読んじゃ")
        self.right = bubble("b", "ダメですっ")

    def test_generated_words_and_punctuation_appear_once_in_order(self):
        for text in ("Don't read pirated copies!", "You must never read pirated copies.",
                     "You shouldn't take that medicine!", "You can rest here if you're tired."):
            with self.subTest(text=text):
                parts = split_sentence(text, self.left, self.right, [])
                self.assertEqual(" ".join(parts), text)
                self.assertTrue(all(len(part.split()) >= 2 for part in parts))

    def test_inner_paragraph_whitespace_is_retained_without_creating_words(self):
        text = "Never read  pirated copies;\nplease support the author."
        parts = split_sentence(text, self.left, self.right, [])
        self.assertEqual(re.findall(r"\S+", " ".join(parts)), re.findall(r"\S+", text))
        self.assertIn("read  pirated", " ".join(parts))

    def test_prefers_existing_clause_boundary_over_cutting_a_phrase(self):
        parts = split_sentence("Never take that medicine, because it is dangerous.",
                               self.left, self.right, [])
        self.assertEqual(parts, ["Never take that medicine,", "because it is dangerous."])

    def test_larger_safe_area_can_take_more_words(self):
        left, right = bubble("a", "待って"), bubble("b", "待って")
        text = "Do not open the door or leave this room."
        left.safe_box, right.safe_box = (0, 0, 20, 20), (0, 0, 200, 200)
        small_first = split_sentence(text, left, right, [])
        left.safe_box, right.safe_box = right.safe_box, left.safe_box
        large_first = split_sentence(text, left, right, [])
        self.assertLess(len(small_first[0]), len(large_first[0]))

    def test_very_short_sentences_are_explicitly_handled(self):
        self.assertEqual(split_sentence("No smoking!", self.left, self.right, []), ["No", "smoking!"])
        for text in ("", "Stop!", "...", "Don't !"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "too short"):
                split_sentence(text, self.left, self.right, [])

    def test_confirmed_name_is_not_cut_even_without_a_lock(self):
        term = {"source": "美咲", "target": "Misaki Takanashi", "aliases": [], "locked": False}
        parts = split_sentence("You must not call Misaki Takanashi today.", self.left, self.right, [term])
        self.assertTrue(any("Misaki Takanashi" in part for part in parts))
        self.assertEqual(" ".join(parts), "You must not call Misaki Takanashi today.")

    def test_locked_name_stays_with_its_source_alias(self):
        term = {"source": "小鳥遊美咲", "target": "Misaki Takanashi", "aliases": ["みーちゃん"], "locked": True}
        self.left.source_text = "みーちゃんを呼んじゃ"
        text = "Do not call Misaki Takanashi to this room."
        parts = split_sentence(text, self.left, self.right, [term])
        self.assertIn("Misaki Takanashi", parts[0])
        self.assertNotIn("Misaki", parts[1])
        self.assertEqual(" ".join(parts), text)

    def test_lock_in_right_source_stays_on_the_right(self):
        term = {"source": "白銀門", "target": "Silver Gate", "aliases": [], "locked": True}
        self.right.source_text = "白銀門へ行ってもいいよ。"
        parts = split_sentence("You may go to Silver Gate now.", self.left, self.right, [term])
        self.assertIn("Silver Gate", parts[1])
        self.assertNotIn("Silver", parts[0])

    def test_missing_or_impossible_locked_term_raises_instead_of_moving_or_inserting_it(self):
        term = {"source": "美咲", "target": "Misaki Takanashi", "aliases": [], "locked": True}
        self.left.source_text = "美咲を呼んじゃ"
        for text in ("Do not call Misaki today.", "Do not call Misaki Takanashi."):
            with self.subTest(text=text), self.assertRaises(ValueError):
                split_sentence(text, self.left, self.right, [term])
        self.right.source_text = "美咲にはダメだよ。"
        with self.assertRaises(ValueError):
            split_sentence("Misaki Takanashi must not do that today.", self.left, self.right, [term])

    def test_repeated_name_can_satisfy_locks_in_both_sources_without_duplication(self):
        term = {"source": "美咲", "target": "Misaki Takanashi", "aliases": [], "locked": True}
        self.left.source_text, self.right.source_text = "美咲は", "美咲だよ。"
        text = "Misaki Takanashi must wait, so Misaki Takanashi cannot go."
        parts = split_sentence(text, self.left, self.right, [term])
        self.assertTrue(all("Misaki Takanashi" in part for part in parts))
        self.assertEqual(" ".join(parts), text)

    def test_fixed_neighbor_cut_overrides_new_safe_area_distribution(self):
        text = "You absolutely cannot read pirated copies!"
        self.left.safe_box, self.right.safe_box = (0, 0, 20, 20), (0, 0, 200, 200)
        for fixed in ({"a": "You absolutely cannot read"}, {"b": "pirated copies!"}):
            with self.subTest(fixed=fixed):
                self.assertEqual(split_sentence(text, self.left, self.right, [], fixed_parts=fixed),
                                 ["You absolutely cannot read", "pirated copies!"])

    def test_fixed_parts_keep_words_and_punctuation_but_tolerate_whitespace(self):
        text = "You absolutely cannot read pirated copies!"
        self.assertEqual(split_sentence(text, self.left, self.right, [],
                                       fixed_parts={"a": "You  absolutely\ncannot read"})[1], "pirated copies!")
        for fixed in ({"b": "pirated copies."}, {"a": "You should not read"}, {"b": ""}):
            with self.subTest(fixed=fixed), self.assertRaisesRegex(ValueError, "preserving the existing"):
                split_sentence(text, self.left, self.right, [], fixed_parts=fixed)

    def test_existing_one_word_neighbor_is_preserved(self):
        self.assertEqual(split_sentence("Do not enter here!", self.left, self.right, [],
                                       fixed_parts={"b": "here!"}), ["Do not enter", "here!"])

    def test_fixed_neighbor_does_not_override_locked_term_source_placement(self):
        term = {"source": "美咲", "target": "Misaki Takanashi", "aliases": [], "locked": True}
        self.left.source_text = "美咲を呼んじゃ"
        with self.assertRaisesRegex(ValueError, "preserving the existing"):
            split_sentence("You must not call Misaki Takanashi today.", self.left, self.right, [term],
                           fixed_parts={"b": "Misaki Takanashi today."})


if __name__ == "__main__":
    unittest.main()
