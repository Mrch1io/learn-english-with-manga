"""Conservative sentence-link regression tests; no GPU/model/API is loaded.

Run: .venv\\Scripts\\python.exe -m unittest tests.test_fragment_links -v

The first source/geometry pair is a user-observed, already-seen example. It is
an exploratory regression, not held-out translation-quality evidence. English
responses below are test doubles used only to verify routing and allocation.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import re
import unittest
from unittest.mock import Mock, patch

from core.runtime import Config, TextBlock
from core.memory import GlossaryMemory
from translation import translate
from translation.vntl import continuation_pairs


def observed_fragments():
    return [TextBlock((132, 447, 172, 535), source_text="はな、", block_id="t003"),
            TextBlock((93, 477, 132, 561), source_text="せ...", block_id="t004")]


class FragmentCandidateTests(unittest.TestCase):
    def test_seen_source_pair_is_linked_without_changing_source_or_geometry(self):
        blocks = observed_fragments()
        before = deepcopy([asdict(block) for block in blocks])
        self.assertEqual(continuation_pairs(blocks), [(0, 1, "fragment")])
        self.assertEqual([asdict(block) for block in blocks], before)

    def test_fragment_switch_keeps_existing_explicit_prohibition_rule(self):
        self.assertEqual(continuation_pairs(observed_fragments(), fragments=False), [])
        blocks = observed_fragments()
        blocks[0].source_text, blocks[1].source_text = "ここに入っちゃ", "だめです。"
        self.assertEqual(continuation_pairs(blocks, fragments=False), [(0, 1, "prohibition")])
        self.assertEqual(continuation_pairs(blocks), [(0, 1, "prohibition")])

    def test_nearby_unfinished_object_clause_is_linked(self):
        blocks = observed_fragments()
        blocks[0].source_text, blocks[1].source_text = "本当のことを", "教えてください。"
        self.assertEqual(continuation_pairs(blocks), [(0, 1, "fragment")])

    def test_proximity_and_missing_period_alone_are_insufficient(self):
        for first, second in (("ありがとう", "今日は晴れだね"),
                              ("こんにちは", "お元気ですか"),
                              ("なに", "うるさい"),
                              ("そうなの", "知らない"),
                              ("わたしも", "来ないで"),
                              ("ありがとう、", "どうも"),
                              ("わかった", "先に帰るね"),
                              ("はな。", "せ..."),
                              ("はな！", "せ..."),
                              ("はな？", "せ..."),
                              ("『はな、』", "せ..."),
                              ("はな、", "『せ...』"),
                              ("...", "せ..."),
                              ("はな、", "..."),
                              ("", "せ...")):
            with self.subTest(first=first, second=second):
                blocks = observed_fragments()
                blocks[0].source_text, blocks[1].source_text = first, second
                self.assertEqual(continuation_pairs(blocks), [])

    def test_failed_ocr_and_different_confirmed_speakers_stay_separate(self):
        for index in (0, 1):
            blocks = observed_fragments()
            blocks[index].status, blocks[index].error = "failed", "OCR: unreadable crop"
            self.assertEqual(continuation_pairs(blocks), [])
        bindings = {"t003": {"character_id": "speaker-one", "confirmed": True},
                    "t004": {"character_id": "speaker-two", "confirmed": True}}
        self.assertEqual(continuation_pairs(observed_fragments(), bindings), [])

    def test_far_overlapping_misaligned_or_differently_sized_boxes_stay_separate(self):
        for second_box in ((0, 477, 39, 561),
                           (120, 477, 159, 561),
                           (93, 550, 132, 634),
                           (118, 477, 132, 561)):
            with self.subTest(second_box=second_box):
                blocks = observed_fragments()
                blocks[1].box = second_box
                self.assertEqual(continuation_pairs(blocks), [])

    def test_vertical_columns_follow_requested_reading_direction(self):
        blocks = observed_fragments()
        self.assertEqual(continuation_pairs(blocks, direction="rtl"), [(0, 1, "fragment")])
        self.assertEqual(continuation_pairs(blocks, direction="ltr"), [])
        blocks[0].box, blocks[1].box = blocks[1].box, blocks[0].box
        self.assertEqual(continuation_pairs(blocks, direction="ltr"), [(0, 1, "fragment")])
        self.assertEqual(continuation_pairs(blocks, direction="rtl"), [])

    def test_horizontal_lines_follow_top_to_bottom_order(self):
        blocks = observed_fragments()
        blocks[0].box, blocks[1].box = (0, 0, 100, 25), (5, 25, 105, 50)
        for direction in ("rtl", "ltr"):
            self.assertEqual(continuation_pairs(blocks, direction=direction), [(0, 1, "fragment")])
        blocks[0].box, blocks[1].box = blocks[1].box, blocks[0].box
        self.assertEqual(continuation_pairs(blocks), [])

    def test_three_fragment_chain_is_not_arbitrarily_paired(self):
        blocks = [TextBlock((200, 0, 230, 100), source_text="やめ、", block_id="a"),
                  TextBlock((170, 0, 200, 100), source_text="ない、", block_id="b"),
                  TextBlock((140, 0, 170, 100), source_text="で...", block_id="c")]
        self.assertEqual(continuation_pairs(blocks[:2]), [(0, 1, "fragment")])
        self.assertEqual(continuation_pairs(blocks[1:]), [(0, 1, "fragment")])
        self.assertEqual(continuation_pairs(blocks), [])


class FragmentTranslationTests(unittest.TestCase):
    def setUp(self):
        self.memory = GlossaryMemory(":memory:", "fragment-tests", "en")
        self.addCleanup(self.memory.close)

    def run_translation(self, blocks, backend, answer="Let go...", *, target_ids=None,
                        overrides=None):
        config = Config(translation_backend=backend, extract_terms=False, group_size=1)
        if overrides:
            config = replace(config, **overrides)

        def generate(prompt, limit, **kwargs):
            self.assertGreater(limit, 0)
            ids = kwargs.get("ids")
            return f"[{ids[0]}]: {answer}" if ids else answer

        generator = Mock(side_effect=generate)
        stats = translate(blocks, pool=None, memory=self.memory, config=config,
                          page_key="fragment-page.png", generator=generator, target_ids=target_ids)
        return generator, stats

    def test_both_backends_translate_full_source_once_and_preserve_original_blocks(self):
        for backend in ("vntl", "qwen"):
            with self.subTest(backend=backend):
                blocks = observed_fragments()
                before = [(block.block_id, block.box, block.source_text) for block in blocks]
                generator, stats = self.run_translation(blocks, backend)
                self.assertEqual(generator.call_count, 1)
                self.assertEqual(stats["generation_calls"], 1)
                self.assertIn("はなせ...", str(generator.call_args.args[0]))
                self.assertNotIn("はな、", str(generator.call_args.args[0]))
                self.assertEqual([block.status for block in blocks], ["translated", "translated"])
                self.assertEqual(" ".join(block.translated_text for block in blocks), "Let go...")
                self.assertEqual([(block.block_id, block.box, block.source_text) for block in blocks], before)
                for block in blocks:
                    unit = block.metadata["translation_unit"]
                    self.assertEqual(unit["members"], ["t003", "t004"])
                    self.assertEqual(unit["source_text"], "はな、せ...")
                    self.assertEqual(unit["sources"], ["はな、", "せ..."])
                    self.assertEqual(unit["translation_source_text"], "はなせ...")
                    self.assertEqual(unit["source_normalization"]["removed"], "、")
                    self.assertEqual(unit["kind"], "fragment")

    def test_linked_phrase_boundary_keeps_ordinary_comma_in_both_backends(self):
        for backend in ("vntl", "qwen"):
            with self.subTest(backend=backend):
                blocks = observed_fragments()
                blocks[0].source_text, blocks[1].source_text = "そう、", "だね..."
                generator, _ = self.run_translation(blocks, backend, answer="That's right...")
                self.assertEqual(generator.call_count, 1)
                self.assertIn("そう、だね...", str(generator.call_args.args[0]))
                unit = blocks[0].metadata["translation_unit"]
                self.assertEqual(unit["translation_source_text"], unit["source_text"])
                self.assertNotIn("source_normalization", unit)

    def test_confirmed_name_keeps_boundary_comma(self):
        self.memory.upsert_term("花", "Hana", aliases=["はな"], locked=True, user_edit=True)
        for backend in ("vntl", "qwen"):
            with self.subTest(backend=backend):
                blocks = observed_fragments()
                generator, _ = self.run_translation(blocks, backend, answer="Hana, speak...")
                self.assertEqual(generator.call_count, 1)
                unit = blocks[0].metadata["translation_unit"]
                self.assertEqual(unit["translation_source_text"], "はな、せ...")
                self.assertNotIn("source_normalization", unit)

    def test_qwen_grouped_prompt_uses_prepared_input(self):
        blocks = observed_fragments()
        config = Config(translation_backend="qwen", translation_mode="grouped", extract_terms=False)

        def generate(prompt, limit):
            import json

            payload = json.loads(prompt[-1]["content"])
            target, = payload["targets"]
            self.assertEqual(target["source"], "はなせ...")
            return json.dumps({"translations": [{"id": target["id"], "text": "Let go..."}]})

        generator = Mock(side_effect=generate)
        translate(blocks, pool=None, memory=self.memory, config=config,
                  page_key="grouped-fragments", generator=generator)
        self.assertEqual(generator.call_count, 1)
        self.assertTrue(all(block.status == "translated" for block in blocks))
        self.assertEqual([block.source_text for block in blocks], ["はな、", "せ..."])

    def test_zero_context_or_disabled_links_keep_individual_requests(self):
        for backend in ("vntl", "qwen"):
            for overrides in ({"context_chars": 0}, {"use_continuations": False},
                              {"use_fragment_links": False}):
                with self.subTest(backend=backend, overrides=overrides):
                    blocks = observed_fragments()
                    generator, stats = self.run_translation(blocks, backend, overrides=overrides)
                    self.assertEqual(generator.call_count, 2)
                    self.assertEqual(stats["generation_calls"], 2)
                    self.assertTrue(all("translation_unit" not in block.metadata for block in blocks))
                    self.assertIn("はな、", str(generator.call_args_list[0].args[0]))

    def test_selected_retranslation_preserves_existing_neighbor(self):
        for backend in ("vntl", "qwen"):
            for selected in ("t003", "t004"):
                with self.subTest(backend=backend, selected=selected):
                    blocks = observed_fragments()
                    for block, english in zip(blocks, ("Let", "go...")):
                        block.translated_text, block.status = english, "translated"
                        block.metadata["last_edit"] = "manual_translation"
                    neighbor = next(block for block in blocks if block.block_id != selected)
                    before = deepcopy(asdict(neighbor))
                    generator, stats = self.run_translation(blocks, backend, target_ids={selected})
                    self.assertEqual(stats["generation_calls"], 1)
                    self.assertEqual(generator.call_count, 1)
                    self.assertEqual(asdict(neighbor), before)
                    self.assertEqual(" ".join(block.translated_text for block in blocks), "Let go...")
                    self.assertTrue(all(block.status == "translated" for block in blocks))

    def test_incompatible_fixed_neighbor_retries_once_without_overwriting_it(self):
        for backend in ("vntl", "qwen"):
            with self.subTest(backend=backend):
                blocks = observed_fragments()
                blocks[0].translated_text, blocks[0].status = "Release", "translated"
                before = deepcopy(asdict(blocks[0]))
                generator, stats = self.run_translation(blocks, backend, target_ids={"t004"})
                self.assertEqual(generator.call_count, 2)
                self.assertEqual(stats["generation_calls"], 2)
                for call in generator.call_args_list:
                    self.assertIn("はなせ...", str(call.args[0]))
                self.assertEqual(asdict(blocks[0]), before)
                self.assertEqual(blocks[1].status, "failed")
                self.assertFalse(blocks[1].translated_text)
                self.assertIn("preserving the existing neighboring translation", blocks[1].error)

    def test_grouping_is_not_reentered_for_virtual_sentence_units(self):
        for backend in ("vntl", "qwen"):
            with self.subTest(backend=backend):
                with patch("translation.vntl.continuation_pairs", wraps=continuation_pairs) as select:
                    generator, _ = self.run_translation(observed_fragments(), backend)
                self.assertEqual(select.call_count, 1)
                self.assertEqual(generator.call_count, 1)

    def test_single_english_word_is_rejected_instead_of_duplicated_across_boxes(self):
        for backend in ("vntl", "qwen"):
            with self.subTest(backend=backend):
                blocks = observed_fragments()
                generator, _ = self.run_translation(blocks, backend, answer="Stop!")
                self.assertEqual(generator.call_count, 2)
                self.assertTrue(all(block.status == "failed" for block in blocks))
                self.assertTrue(all(not block.translated_text for block in blocks))

    def test_longer_sentence_allocation_preserves_generated_words_exactly_once(self):
        sentence = "Please let go of my hand right now."
        for backend in ("vntl", "qwen"):
            with self.subTest(backend=backend):
                blocks = observed_fragments()
                generator, _ = self.run_translation(blocks, backend, answer=sentence)
                self.assertEqual(generator.call_count, 1)
                self.assertEqual(re.findall(r"\S+", " ".join(b.translated_text for b in blocks)),
                                 re.findall(r"\S+", sentence))
                self.assertTrue(all(block.translated_text.strip() for block in blocks))


if __name__ == "__main__":
    unittest.main()
