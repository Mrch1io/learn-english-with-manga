"""Source-range ambiguity must never force an extra translated character."""
import unittest
from unittest.mock import Mock

from core.runtime import Config, TextBlock
from core.memory import GlossaryMemory
from translation import translate


class LockedScopeTests(unittest.TestCase):
    def setUp(self):
        self.memory = GlossaryMemory(":memory:", "locked-scope", "en")
        self.addCleanup(self.memory.close)
        self.config = Config(extract_terms=False)

    def term(self, source, target, *aliases):
        self.memory.upsert_term(source, target, aliases=list(aliases), locked=True, user_edit=True)

    def run_bubble(self, source, output="Wait!", context=None):
        block = TextBlock((0, 0, 100, 100), source_text=source, block_id="only")
        blocks = [block]
        if context:
            blocks.append(TextBlock((100, 0, 200, 100), source_text=context, block_id="context"))
        generator = Mock(return_value="[only]: " + output)
        stats = translate(blocks, pool=None, memory=self.memory, config=self.config, page_key="p",
                          target_ids={"only"}, generator=generator)
        return block, generator, stats

    def assert_ambiguous(self, source):
        block, generator, stats = self.run_bubble(source)
        generator.assert_not_called()
        self.assertEqual(block.status, "failed")
        self.assertEqual(block.translated_text, "")
        self.assertIn("Ambiguous locked names", block.error)
        self.assertLessEqual(block.metadata["translation_attempts"], 2)

    def test_shared_alias_for_two_people_is_rejected_before_model_generation(self):
        self.term("小鳥遊美咲", "Misaki Takanashi", "みーちゃん")
        self.term("山田美月", "Mizuki Yamada", "みーちゃん")
        self.assert_ambiguous("みーちゃんが来た。")

    def test_alias_conflicting_with_another_exact_source_is_rejected(self):
        self.term("小鳥遊美咲", "Misaki Takanashi", "みーちゃん")
        self.term("みーちゃん", "Mia")
        self.assert_ambiguous("みーちゃんが来た。")

    def test_nested_sources_with_different_targets_are_explicitly_rejected(self):
        self.term("小鳥遊美咲", "Misaki Takanashi")
        self.term("美咲", "Mizuki")
        self.assert_ambiguous("小鳥遊美咲が来た。")

    def test_partial_source_overlap_is_rejected(self):
        self.term("甲乙", "AB")
        self.term("乙丙", "BC")
        self.assert_ambiguous("甲乙丙だ。")

    def test_same_canonical_for_nested_sources_and_repeated_mentions_is_allowed(self):
        self.term("小鳥遊美咲", "Misaki Takanashi", "みーちゃん")
        self.term("美咲", "Misaki Takanashi")
        block, generator, _ = self.run_bubble("小鳥遊美咲、みーちゃんだ。", "Misaki Takanashi is here.")
        self.assertEqual(block.status, "translated")
        generator.assert_called_once()
        self.assertEqual(generator.call_args.kwargs["required_terms"], {"only": ["Misaki Takanashi"]})

    def test_separate_mentions_require_two_people_without_ambiguity(self):
        self.term("美咲", "Misaki Takanashi")
        self.term("美月", "Mizuki Yamada")
        block, generator, _ = self.run_bubble("美咲と美月が来た。", "Misaki Takanashi and Mizuki Yamada arrived.")
        self.assertEqual(block.status, "translated")
        self.assertEqual(set(generator.call_args.kwargs["required_terms"]["only"]),
                         {"Misaki Takanashi", "Mizuki Yamada"})

    def test_locked_alias_is_normalized_only_in_model_source(self):
        self.term("小鳥遊美咲", "Misaki Takanashi", "美咲", "みーちゃん")
        source = "みーちゃん、その鍵はどこ？"
        block, generator, _ = self.run_bubble(source, "Misaki Takanashi, where's that key?")
        prompt = generator.call_args.args[0]
        japanese_turn = prompt.rsplit("<|start_header_id|>Japanese<|end_header_id|>", 1)[1]
        self.assertIn("[only]: Misaki Takanashi、その鍵はどこ？", japanese_turn)
        self.assertNotIn("みーちゃん", japanese_turn)
        self.assertEqual(block.source_text, source)
        self.assertEqual(block.translated_text, "Misaki Takanashi, where's that key?")

    def test_unlocked_names_stay_japanese_in_model_source(self):
        self.memory.upsert_term("美咲", "Misaki Takanashi", user_edit=True)
        block, generator, _ = self.run_bubble("美咲が来た。", "Misaki Takanashi arrived.")
        japanese_turn = generator.call_args.args[0].rsplit("<|start_header_id|>Japanese<|end_header_id|>", 1)[1]
        self.assertIn("[only]: 美咲が来た。", japanese_turn)
        self.assertEqual(block.source_text, "美咲が来た。")

    def test_latin_word_boundary_prevents_ann_matching_hannah(self):
        self.term("Ann", "Ann Smith")
        self.term("Hannah", "Hannah Jones")
        block, generator, _ = self.run_bubble("Hannahが来た。", "Hannah Jones has arrived.")
        self.assertEqual(block.status, "translated")
        self.assertEqual(generator.call_args.kwargs["required_terms"], {"only": ["Hannah Jones"]})

    def test_real_latin_name_overlap_is_not_hidden_by_boundaries(self):
        self.term("Captain Ann", "Commander Smith")
        self.term("Ann", "Ann Jones")
        self.assert_ambiguous("Captain Ann arrived.")

    def test_ambiguous_names_only_in_neighbor_context_do_not_constrain_target(self):
        self.term("小鳥遊美咲", "Misaki Takanashi", "みーちゃん")
        self.term("山田美月", "Mizuki Yamada", "みーちゃん")
        block, generator, _ = self.run_bubble("待って。", "Wait!", context="みーちゃんが来た。")
        self.assertEqual(block.status, "translated")
        self.assertNotIn("required_terms", generator.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
