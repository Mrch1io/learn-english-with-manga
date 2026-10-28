"""Continuation/locked-term flow checks. Real model results are recorded separately."""
from copy import deepcopy
from dataclasses import asdict, replace
import unittest
from unittest.mock import Mock

from core.runtime import Config, TextBlock
from core.memory import GlossaryMemory
from translation import translate


class TranslationUnitTests(unittest.TestCase):
    def setUp(self):
        self.memory = GlossaryMemory(":memory:", "units", "en")
        self.addCleanup(self.memory.close)
        self.config = Config(extract_terms=False, group_size=1)

    def blocks(self):
        return [TextBlock((787, 94, 1037, 504), source_text="海賊版は読んじゃ", block_id="right"),
                TextBlock((69, 37, 267, 479), source_text="ダメですっ", block_id="left")]

    def run_translation(self, blocks, generate, **kwargs):
        return translate(blocks, memory=self.memory, pool=None, config=self.config, page_key="page.png",
                         generator=generate, **kwargs)

    def test_joined_once_even_when_group_size_one_and_split_never_duplicates_words(self):
        blocks = self.blocks()
        def generate(prompt, limit, *, ids):
            self.assertEqual(len(ids), 1)
            self.assertIn("海賊版は読んじゃダメですっ", prompt)
            return f"[{ids[0]}]: You must not read pirated copies."
        generator = Mock(side_effect=generate)
        stats = self.run_translation(blocks, generator)
        self.assertEqual(stats["generation_calls"], 1)
        self.assertEqual([b.block_id for b in blocks], ["right", "left"])
        self.assertEqual(" ".join(b.translated_text for b in blocks), "You must not read pirated copies.")
        self.assertTrue(all(b.status == "translated" for b in blocks))
        self.assertEqual(blocks[0].metadata["translation_unit"]["members"], ["right", "left"])
        self.assertEqual(blocks[1].source_text, "ダメですっ")

    def test_lost_prohibition_retries_once_then_explicitly_fails_without_echo(self):
        blocks = self.blocks()
        generate = Mock(side_effect=lambda prompt, limit, *, ids:
                        f"[{ids[0]}]: I already read pirated copies.")
        self.run_translation(blocks, generate)
        self.assertEqual(generate.call_count, 2)
        self.assertTrue(all(b.status == "failed" and not b.translated_text for b in blocks))
        self.assertIn("lost its negative meaning", blocks[0].error)
        self.assertEqual(blocks[1].source_text, "ダメですっ")

    def test_selected_retry_reads_whole_sentence_but_preserves_neighbor_exactly(self):
        blocks = self.blocks()
        blocks[0].translated_text, blocks[0].status = "Don't read", "translated"
        blocks[1].translated_text, blocks[1].status = "pirated copies.", "translated"
        blocks[0].metadata["manual_translation"] = True
        before = deepcopy(asdict(blocks[0]))
        def generate(prompt, limit, *, ids):
            self.assertIn("海賊版は読んじゃダメですっ", prompt)
            self.assertNotIn("Don't read", prompt)
            return f"[{ids[0]}]: Don't read pirated copies."
        self.run_translation(blocks, generate, target_ids={"left"})
        self.assertEqual(asdict(blocks[0]), before)
        self.assertEqual(blocks[1].translated_text, "pirated copies.")
        self.assertNotIn("context_review", blocks[1].metadata)

    def test_stale_positive_neighbor_cannot_turn_single_edit_into_positive_sentence(self):
        blocks = self.blocks()
        blocks[0].translated_text, blocks[0].status = "I've read", "translated"
        before = deepcopy(asdict(blocks[0]))
        self.run_translation(blocks, lambda prompt, limit, *, ids:
                             f"[{ids[0]}]: Don't read pirated copies.", target_ids={"left"})
        self.assertEqual(asdict(blocks[0]), before)
        self.assertEqual(blocks[1].status, "failed")
        self.assertIn("preserving the existing neighboring translation", blocks[1].error)

    def test_local_retry_cannot_move_read_across_an_unchanged_neighbor(self):
        sentence = "You absolutely cannot read pirated copies!"
        for target in ("right", "left"):
            with self.subTest(target=target):
                blocks = self.blocks()
                blocks[0].translated_text = "You absolutely cannot read"
                blocks[1].translated_text = "pirated copies!"
                for block in blocks:
                    block.status = "translated"
                # Safe areas can differ after export or manual region editing.
                # A new area-weighted cut must not duplicate/drop 'read'.
                blocks[0].safe_box = (787, 94, 807, 114)
                blocks[1].safe_box = blocks[1].box
                before = deepcopy([asdict(block) for block in blocks])
                generator = Mock(side_effect=lambda prompt, limit, *, ids: f"[{ids[0]}]: {sentence}")
                self.run_translation(blocks, generator, target_ids={target})
                self.assertEqual(generator.call_count, 1)
                self.assertTrue(all(block.status == "translated" for block in blocks))
                self.assertEqual(" ".join(block.translated_text for block in blocks), sentence)
                for index, block in enumerate(blocks):
                    if block.block_id != target:
                        self.assertEqual(asdict(block), before[index])
                    else:
                        self.assertNotIn("context_review", block.metadata)

    def test_changed_neighbor_wording_cannot_be_saved_as_a_mixed_sentence(self):
        blocks = self.blocks()
        blocks[0].translated_text, blocks[0].status = "You must not read", "translated"
        before = deepcopy(asdict(blocks[0]))
        generator = Mock(side_effect=lambda prompt, limit, *, ids:
                         f"[{ids[0]}]: You absolutely cannot read pirated copies!")
        self.run_translation(blocks, generator, target_ids={"left"})
        self.assertEqual(generator.call_count, 2)
        self.assertEqual(asdict(blocks[0]), before)
        self.assertEqual(blocks[1].status, "failed")
        self.assertFalse(blocks[1].translated_text)
        self.assertIn("preserving the existing neighboring translation", blocks[1].error)

    def test_local_retry_normalizes_neighbor_whitespace_without_changing_it(self):
        blocks = self.blocks()
        blocks[0].translated_text, blocks[0].status = "You  absolutely\ncannot read", "translated"
        before = deepcopy(asdict(blocks[0]))
        self.run_translation(blocks, lambda prompt, limit, *, ids:
                             f"[{ids[0]}]: You absolutely cannot read pirated copies!", target_ids={"left"})
        self.assertEqual(asdict(blocks[0]), before)
        self.assertEqual(blocks[1].translated_text, "pirated copies!")
        self.assertEqual(blocks[1].status, "translated")
        self.assertNotIn("context_review", blocks[1].metadata)

    def test_local_edit_uses_measured_safe_area_without_changing_neighbor(self):
        for storage in ("safe_box", "layout"):
            with self.subTest(storage=storage):
                blocks = [TextBlock((0, 0, 200, 200), source_text="窓を開けては", block_id="one"),
                          TextBlock((200, 0, 400, 200), source_text="いけません。", block_id="two",
                                    translated_text="the door or leave this room.", status="translated")]
                if storage == "safe_box":
                    blocks[0].safe_box = (0, 0, 20, 20)
                    # The explicit current safe box takes precedence over an
                    # earlier saved layout box.
                    blocks[0].metadata["layout"] = {"safe_box": [0, 0, 200, 200]}
                else:
                    blocks[0].metadata["layout"] = {"safe_box": [0, 0, 20, 20]}
                before = deepcopy(asdict(blocks[1]))
                self.run_translation(blocks, lambda prompt, limit, *, ids:
                                     f"[{ids[0]}]: Do not open the door or leave this room.", target_ids={"one"})
                self.assertEqual(blocks[0].status, "translated")
                self.assertEqual(blocks[0].translated_text, "Do not open")
                self.assertEqual(asdict(blocks[1]), before)
                self.assertNotIn("context_review", blocks[0].metadata)
                self.assertEqual(blocks[0].metadata["translation_unit"]["safe_boxes"],
                                 [[0, 0, 20, 20], [200, 0, 400, 200]])

    def test_permission_is_not_forced_to_be_negative(self):
        blocks = [TextBlock((0, 0, 100, 100), source_text="ここに座っても", block_id="one"),
                  TextBlock((100, 0, 200, 100), source_text="いいですか？", block_id="two")]
        self.run_translation(blocks, lambda prompt, limit, *, ids:
                             f"[{ids[0]}]: May I sit here?")
        self.assertEqual(" ".join(b.translated_text for b in blocks), "May I sit here?")
        self.assertTrue(all(b.status == "translated" for b in blocks))

    def test_permissive_negative_words_are_not_prohibition_evidence(self):
        for response in (
            "You can read pirated copies, no problem.",
            "It is not wrong to read pirated copies.",
            "Reading pirated copies is not forbidden.",
            "Reading pirated copies isn't prohibited.",
            "You don't have to read pirated copies.",
            "You can't help reading pirated copies.",
            "I didn't read pirated copies.",
        ):
            with self.subTest(response=response):
                blocks = self.blocks()
                generator = Mock(side_effect=lambda prompt, limit, *, ids:
                                 f"[{ids[0]}]: {response}")
                self.run_translation(blocks, generator)
                self.assertEqual(generator.call_count, 2)
                self.assertTrue(all(b.status == "failed" and not b.translated_text for b in blocks))

    def test_explicit_prohibitions_keep_full_sentence_once(self):
        for response in (
            "You shouldn't read pirated copies.",
            "Please do not read pirated copies.",
            "Reading pirated copies is not allowed.",
            "Avoid reading pirated copies.",
        ):
            with self.subTest(response=response):
                blocks = self.blocks()
                self.run_translation(blocks, lambda prompt, limit, *, ids: f"[{ids[0]}]: {response}")
                self.assertTrue(all(b.status == "translated" for b in blocks))
                self.assertEqual(" ".join(b.translated_text for b in blocks), response)

    def test_only_current_source_locks_enter_grammar_and_missing_lock_fails(self):
        self.memory.upsert_term("小鳥遊美咲", "Misaki Takanashi", aliases=["みーちゃん"], locked=True, user_edit=True, first_page="seed")
        self.memory.upsert_term("白銀門", "Silver Gate", locked=True, user_edit=True, first_page="seed")
        blocks = [TextBlock((0, 0, 100, 100), source_text="みーちゃん、鍵は？", block_id="one"),
                  TextBlock((100, 0, 200, 100), source_text="白銀門。", block_id="two")]
        before = self.memory.list_terms()
        def generate(prompt, limit, *, ids, required_terms):
            self.assertEqual(ids, ["one"])
            self.assertEqual(required_terms, {"one": ["Misaki Takanashi"]})
            return "[one]: Mii-chan, where's the key?"
        generator = Mock(side_effect=generate)
        self.run_translation(blocks, generator, target_ids={"one"})
        self.assertEqual(generator.call_count, 2)
        self.assertEqual(blocks[0].status, "failed")
        self.assertFalse(blocks[0].translated_text)
        self.assertEqual(self.memory.list_terms(), before)
        self.assertEqual(blocks[1].status, "pending")

    def test_locked_name_stays_intact_and_in_its_source_bubble(self):
        self.memory.upsert_term("美咲", "Misaki Takanashi", locked=True, user_edit=True)
        blocks = self.blocks()
        blocks[0].source_text = "美咲、ここに入っちゃ"
        def generate(prompt, limit, *, ids, required_terms):
            self.assertEqual(required_terms, {ids[0]: ["Misaki Takanashi"]})
            return f"[{ids[0]}]: Misaki Takanashi, do not enter this room."
        self.run_translation(blocks, generate)
        self.assertTrue(all(b.status == "translated" for b in blocks))
        self.assertIn("Misaki Takanashi", blocks[0].translated_text)
        self.assertNotIn("Takanashi", blocks[1].translated_text)
        self.assertEqual(" ".join(b.translated_text for b in blocks), "Misaki Takanashi, do not enter this room.")


if __name__ == "__main__":
    unittest.main()
