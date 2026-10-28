import json
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core.runtime import Cancelled, Config, TextBlock
from core.memory import GlossaryMemory
from translation import ContextTooLong, _context, _groups, _load_generator, parse_group_response, translate


def block(key, text):
    return TextBlock((0, 0, 100, 100), source_text=text, block_id=key)


class FakeGenerator:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, messages, max_new_tokens):
        self.calls.append((messages, max_new_tokens))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.memory = GlossaryMemory(":memory:", "test-work")

    def tearDown(self):
        self.memory.close()

    def run_translation(self, blocks, responses, **overrides):
        config = Config(translation_backend="qwen", extract_terms=False, **overrides)
        generator = FakeGenerator(responses)
        stats = translate(blocks, pool=None, memory=self.memory, config=config,
                          page_key="page1.png", generator=generator)
        return generator, stats

    def test_group_mapping_is_by_id_not_output_order(self):
        blocks = [block("a", "行け"), block("b", "待て")]
        output = json.dumps({"translations": [{"id": "b", "text": "Wait!"},
                                               {"id": "a", "text": "Go!"}], "terms": []})
        fake, _ = self.run_translation(blocks, [output], translation_mode="grouped")
        self.assertEqual([b.translated_text for b in blocks], ["Go!", "Wait!"])
        self.assertEqual(len(fake.calls), 1)

    def test_group_missing_terms_runs_supplemental_extraction(self):
        blocks = [block("a", "ヘイズだ")]
        output = json.dumps({"translations": [{"id": "a", "text": "Hayes!"}]})
        extracted = json.dumps({"terms": [{"id": "a", "source": "ヘイズ", "target": "Hayes", "category": "person"}]})
        fake = FakeGenerator([output, extracted])
        stats = translate(blocks, pool=None, memory=self.memory,
                          config=Config(translation_backend="qwen", translation_mode="grouped"), page_key="p", generator=fake)
        self.assertEqual(stats["generation_calls"], 1)
        self.assertEqual(stats["candidate_calls"], 1)
        self.assertEqual(stats["candidates_added"], 1)
        self.assertEqual(blocks[0].status, "translated")

    def test_group_invalid_terms_type_runs_supplemental_extraction(self):
        for invalid in (None, {}, "none"):
            with self.subTest(terms=invalid):
                blocks = [block("a", "待て")]
                output = json.dumps({"translations": [{"id": "a", "text": "Wait!"}], "terms": invalid})
                fake = FakeGenerator([output, '{"terms":[]}'])
                stats = translate(blocks, pool=None, memory=self.memory,
                                  config=Config(translation_backend="qwen", translation_mode="grouped"), page_key="p", generator=fake)
                self.assertEqual(len(fake.calls), 2)
                self.assertEqual(stats["candidate_calls"], 1)
                self.assertEqual(blocks[0].translated_text, "Wait!")

    def test_group_valid_empty_terms_avoids_redundant_extraction(self):
        blocks = [block("a", "待て")]
        output = json.dumps({"translations": [{"id": "a", "text": "Wait!"}], "terms": []})
        fake = FakeGenerator([output])
        stats = translate(blocks, pool=None, memory=self.memory,
                          config=Config(translation_backend="qwen", translation_mode="grouped"), page_key="p", generator=fake)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(stats["candidate_calls"], 0)
        self.assertEqual(stats["warnings"], [])

    def test_disabled_extraction_ignores_unsolicited_candidates_in_both_modes(self):
        for mode in ("individual", "grouped"):
            with self.subTest(mode=mode):
                blocks = [block("a", "リナはパリへ行った。")]
                output = json.dumps({"translations": [{"id": "a", "text": "Rina went to Paris."}],
                                     "terms": [{"id": "a", "source": "リナ", "target": "Rina",
                                                "category": "person"}]})
                fake, stats = self.run_translation(blocks, [output], translation_mode=mode)
                self.assertEqual(blocks[0].status, "translated")
                self.assertEqual(self.memory.list_terms(), [])
                self.assertEqual(stats["candidates_added"], 0)
                self.assertEqual(stats["candidate_calls"], 0)
                if mode == "grouped":
                    self.assertNotIn('"terms"', fake.calls[0][0][0]["content"])
                    self.assertIn("translations only", fake.calls[0][0][0]["content"])

    def test_disabled_extraction_still_injects_and_checks_existing_terms(self):
        self.memory.upsert_term("ヘイズ", "Hayes", locked=True, status="confirmed")
        blocks = [block("a", "ヘイズだ")]
        response = json.dumps({"translations": [{"id": "a", "text": "Haze!"}]})
        fake, stats = self.run_translation(blocks, [response, "Hayes!"], translation_mode="grouped")
        payload = json.loads(fake.calls[0][0][1]["content"])
        self.assertEqual(payload["glossary"][0]["target"], "Hayes")
        self.assertEqual(blocks[0].translated_text, "Hayes!")
        self.assertEqual(blocks[0].metadata["translation_attempts"], 2)
        self.assertEqual(stats["candidates_added"], 0)
        self.assertEqual(stats["candidate_calls"], 0)

    def test_partial_failure_retries_only_bad_bubble(self):
        blocks = [block("a", "行け"), block("b", "待て")]
        output = json.dumps({"translations": [{"id": "a", "text": "Go!"},
                                               {"id": "b", "text": ""}]})
        fake, _ = self.run_translation(blocks, [output, "Wait!"], translation_mode="grouped")
        messages = fake.calls[1][0]
        self.assertEqual(messages[1]["content"], "待て")
        self.assertIn("CURRENT BUBBLE ID: b", messages[0]["content"])
        self.assertEqual([b.metadata["translation_attempts"] for b in blocks], [1, 2])
        self.assertEqual(blocks[0].translated_text, "Go!")

    def test_duplicate_missing_and_empty_are_explicit(self):
        output = '{"translations":[{"id":"a","text":"first"},{"id":"a","text":"second"},{"id":"b","text":""}]}'
        result, errors, _ = parse_group_response(output, ["a", "b", "c"])
        self.assertEqual(result, {})
        self.assertEqual(set(errors), {"a", "b", "c"})
        self.assertIn("Duplicate", errors["a"])
        self.assertIn("Missing", errors["c"])

    def test_unknown_ids_and_duplicate_json_keys_are_rejected(self):
        for output in ['{"translations":[{"id":"other","text":"x"}]}',
                       '{"translations":[],"translations":[{"id":"a","text":"x"}]}']:
            result, errors, _ = parse_group_response(output, ["a"])
            self.assertFalse(result)
            self.assertIn("a", errors)

    def test_no_original_text_fallback_on_failure(self):
        blocks = [block("a", "待って")]
        fake, _ = self.run_translation(blocks, ["", RuntimeError("model failed")])
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(blocks[0].status, "failed")
        self.assertEqual(blocks[0].translated_text, "")
        self.assertEqual(blocks[0].source_text, "待って")
        self.assertIn("model failed", blocks[0].error)

    def test_identical_japanese_is_not_translation_success(self):
        for source, output in [("待って", "待って"), ("待って！", "待って！"), ("待って！", "待って!")]:
            with self.subTest(source=source, output=output):
                blocks = [block("a", source)]
                self.run_translation(blocks, [output], retries=0)
                self.assertEqual(blocks[0].status, "failed")

    def test_extended_japanese_response_retries_only_target_without_neighbor(self):
        blocks = [block("right", "海賊版は読んじゃ"), block("left", "ダメですっ")]
        fake, stats = self.run_translation(blocks,
            ["海賊版は読んじゃダメですっ", "Don't read pirated copies...", "Don't!"], use_continuations=False)
        self.assertEqual(stats["generation_calls"], 3)
        self.assertEqual([b.metadata["translation_attempts"] for b in blocks], [2, 1])
        first, retry = fake.calls[0][0], fake.calls[1][0]
        self.assertIn("ダメですっ", first[0]["content"])
        self.assertNotIn("ダメですっ", retry[0]["content"])
        self.assertNotIn("REFERENCE ONLY", retry[0]["content"])
        self.assertEqual(retry[1]["content"], "海賊版は読んじゃ")
        self.assertEqual(blocks[0].translated_text, "Don't read pirated copies...")
        self.assertEqual(blocks[0].metadata["translation_rejections"][0]["response"], "海賊版は読んじゃダメですっ")
        self.assertEqual([b.status for b in blocks], ["translated", "translated"])

    def test_japanese_after_one_retry_is_failed_before_font_layout(self):
        for response in ("海賊版は読んじゃダメ", "Do not read 海賊版", "ｶｲｿﾞｸ", "海賊版"):
            with self.subTest(response=response):
                blocks = [block("right", "海賊版は読んじゃ")]
                fake, _ = self.run_translation(blocks, [response, response])
                self.assertEqual(len(fake.calls), 2)
                self.assertEqual(blocks[0].status, "failed")
                self.assertEqual(blocks[0].translated_text, "")
                self.assertIn("Expected English", blocks[0].error)

    def test_grouped_japanese_retry_preserves_successful_neighbor(self):
        blocks = [block("right", "海賊版は読んじゃ"), block("left", "ダメですっ")]
        output = json.dumps({"translations": [{"id": "right", "text": "海賊版は読んじゃダメですっ"},
                                              {"id": "left", "text": "Don't!"}]})
        fake, stats = self.run_translation(blocks, [output, "Reading pirated copies..."],
                                         translation_mode="grouped", use_continuations=False)
        self.assertEqual(stats["generation_calls"], 2)
        self.assertEqual([b.metadata["translation_attempts"] for b in blocks], [2, 1])
        self.assertEqual(blocks[1].translated_text, "Don't!")
        self.assertNotIn("ダメですっ", fake.calls[1][0][0]["content"])

    def test_english_accents_and_japanese_punctuation_are_not_wrong_language(self):
        blocks = [block("a", "カフェだ")]
        self.run_translation(blocks, ['「Meet José at the café!」'], retries=0)
        self.assertEqual(blocks[0].status, "translated")

    def test_empty_input_never_loads_or_calls_models(self):
        blocks = [block("a", "   ")]
        fake, _ = self.run_translation(blocks, [])
        self.assertEqual(fake.calls, [])
        self.assertEqual(blocks[0].status, "empty")

    def test_ocr_failure_reason_survives_translation(self):
        failed = block("a", "")
        failed.status, failed.error = "failed", "OCR: broken crop"
        fake, _ = self.run_translation([failed], [])
        self.assertEqual(fake.calls, [])
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.error, "OCR: broken crop")

    def test_punctuation_only_is_normalized_without_loading_or_calling_model(self):
        blocks = [block("a", "．．．"), block("b", "！？\n…")]
        pool = SimpleNamespace(acquire=Mock(side_effect=AssertionError("must not load model")))
        stats = translate(blocks, pool=pool, memory=self.memory, config=Config(translation_backend="qwen"), page_key="p")
        self.assertEqual([item.translated_text for item in blocks], ["...", "!?\n..."])
        self.assertEqual([item.source_text for item in blocks], ["．．．", "！？\n…"])
        self.assertTrue(all(item.status == "translated" for item in blocks))
        self.assertTrue(all(item.metadata["translation_attempts"] == 0 for item in blocks))
        self.assertEqual(stats["generation_calls"], 0)
        self.assertEqual(stats["candidate_calls"], 0)
        pool.acquire.assert_not_called()

    def test_punctuation_fast_path_does_not_skip_letters_ideographs_numbers_or_symbols(self):
        blocks = [block("a", "．．．"), block("b", "待って！"), block("c", "A!"),
                  block("d", "１？"), block("e", "+")]
        fake, stats = self.run_translation(blocks, ["Wait!", "A!", "1?", "+"], retries=0)
        self.assertEqual(len(fake.calls), 4)
        self.assertEqual(blocks[0].translated_text, "...")
        self.assertTrue(all(item.metadata["translation_attempts"] == 1 for item in blocks[1:]))
        self.assertEqual(stats["generation_calls"], 4)

    def test_english_punctuation_width_normalization_preserves_names_and_paragraphs(self):
        source = "ヘイズだ。待って！"
        blocks = [block("a", source)]
        output = "Ｈａｙｅｓ， wait．\n\nReady？！ Meet ﬁo at ２．"
        self.run_translation(blocks, [output], retries=0)
        self.assertEqual(blocks[0].translated_text,
                         "Ｈａｙｅｓ, wait.\n\nReady?! Meet ﬁo at ２.")
        self.assertEqual(blocks[0].source_text, source)
        self.assertTrue(blocks[0].metadata["punctuation_normalized"])
        self.assertEqual(blocks[0].status, "translated")

    def test_context_is_bounded_and_distinct_from_target(self):
        blocks = [block("a", "あ" * 50), block("b", "ヘイズ！"), block("c", "う" * 50)]
        context = _context(blocks, {"b"}, 60)
        self.assertLessEqual(sum(len(row["source"]) for row in context), 60)
        self.assertNotIn("b", [row["id"] for row in context])
        self.memory.upsert_term("ヘイズ", "Hayes", locked=True)
        fake, _ = self.run_translation(blocks, ["Hmm.", "Hayes!", "Uh."], retries=0)
        messages = fake.calls[1][0]
        self.assertEqual(messages[1]["content"], "ヘイズ！")
        self.assertIn("CURRENT BUBBLE ID: b", messages[0]["content"])
        self.assertIn("Translate 'ヘイズ' strictly as 'Hayes'.", messages[0]["content"])
        self.assertIn("REFERENCE ONLY, do not translate", messages[0]["content"])

    def test_individual_has_at_most_one_neighbor_each_side_and_explicit_alias_rules(self):
        self.memory.upsert_term("ヘイズ", "Hayes", aliases=["ヘイス"])
        blocks = [block("a", "far before"), block("b", "near before"),
                  block("c", "ヘイスだ"), block("d", "near after"), block("e", "far after")]
        fake, _ = self.run_translation(blocks,
                                      ["Far before.", "Near before.", "Hayes!", "Near after.", "Far after."], retries=0)
        system, user = fake.calls[2][0]
        self.assertEqual(user["content"], "ヘイスだ")
        self.assertIn("near before", system["content"])
        self.assertIn("near after", system["content"])
        self.assertNotIn("far before", system["content"])
        self.assertNotIn("far after", system["content"])
        self.assertIn("Translate 'ヘイス' strictly as 'Hayes' (alias of 'ヘイズ').", system["content"])
        context = _context(blocks, {"c"}, 15, neighbors_only=True)
        self.assertLessEqual(sum(len(row["source"]) for row in context), 15)
        self.assertLessEqual(len(context), 2)

    def test_term_conflict_retries_without_string_replacement(self):
        self.memory.upsert_term("ヘイズ", "Hayes", locked=True)
        blocks = [block("a", "ヘイズが来た")]
        fake, stats = self.run_translation(blocks, ["Haze is here.", "He is here."])
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(blocks[0].translated_text, "He is here.")
        self.assertIn("term_conflicts", blocks[0].metadata)
        self.assertTrue(stats["warnings"])

    def test_default_candidate_extraction_saves_only_evidenced_named_entities(self):
        blocks = [block("a", "ヘイズが来た")]
        terms = {"terms": [
            {"id": "a", "source": "ヘイズ", "target": "Hayes", "category": "person"},
            {"id": "a", "source": "ない", "target": "Hayes", "category": "person"},
            {"id": "a", "source": "来た", "target": "arrived", "category": "verb"},
            {"id": "a", "source": "ヘイズ", "target": "Haze", "category": "person"},
        ]}
        fake = FakeGenerator(["Hayes arrived.", json.dumps(terms)])
        stats = translate(blocks, pool=None, memory=self.memory, config=Config(translation_backend="qwen"), page_key="2.png", generator=fake)
        self.assertEqual(stats["candidate_calls"], 1)
        self.assertEqual(stats["candidates_added"], 1)
        self.assertGreaterEqual(stats["candidate_seconds"], 0)
        self.assertEqual(self.memory.list_terms()[0]["first_page"], "2.png")
        next_page = [block("a", "ヘイズだ")]
        next_fake, _ = self.run_translation(next_page, ["It's Hayes."])
        messages = next_fake.calls[0][0]
        self.assertEqual(messages[1]["content"], "ヘイズだ")
        self.assertIn("Translate 'ヘイズ' strictly as 'Hayes'.", messages[0]["content"])

    def test_extraction_failure_is_warning_not_translation_failure(self):
        blocks = [block("a", "待って")]
        fake = FakeGenerator(["Wait!", "not JSON"])
        stats = translate(blocks, pool=None, memory=self.memory, config=Config(translation_backend="qwen"), page_key="p", generator=fake)
        self.assertEqual(blocks[0].status, "translated")
        self.assertTrue(stats["warnings"])

    def test_extraction_dispatches_to_base_generator_when_available(self):
        blocks = [block("a", "ヘイズだ")]
        fake = FakeGenerator(["Hayes!"])
        fake.extract_terms = FakeGenerator([json.dumps({"terms": [
            {"id": "a", "source": "ヘイズ", "target": "Hayes", "category": "person"}
        ]})])
        stats = translate(blocks, pool=None, memory=self.memory, config=Config(translation_backend="qwen"), page_key="p", generator=fake)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len(fake.extract_terms.calls), 1)
        self.assertEqual(stats["candidates_added"], 1)

    def test_direct_candidate_array_is_accepted_with_strict_row_validation(self):
        blocks = [block("a", "リナはパリへ行った。")]
        raw = json.dumps([
            {"id": "a", "source": "リナ", "target": "Rina", "category": "person"},
            {"id": "wrong", "source": "パリ", "target": "Paris", "category": "place"},
            {"id": "a", "source": "パリ", "target": "London", "category": "place"},
            {"id": "a", "source": "パリ", "target": "Paris", "category": "verb"},
        ])
        fake = FakeGenerator(["Rina went to Paris.", raw])
        stats = translate(blocks, pool=None, memory=self.memory, config=Config(translation_backend="qwen"), page_key="p", generator=fake)
        self.assertEqual(stats["candidates_added"], 1)
        self.assertEqual(self.memory.list_terms()[0]["source"], "リナ")
        self.assertEqual(stats["warnings"], [])

    def test_direct_empty_array_is_valid_extraction_result(self):
        fake = FakeGenerator(["Wait!", "[]"])
        stats = translate([block("a", "待って")], pool=None, memory=self.memory,
                          config=Config(translation_backend="qwen"), page_key="p", generator=fake)
        self.assertEqual(stats["candidate_calls"], 1)
        self.assertEqual(stats["candidates_added"], 0)
        self.assertEqual(stats["warnings"], [])

    def test_individual_prompt_preserves_chapter_numbers(self):
        fake, _ = self.run_translation([block("a", "第３４話、裏切りの夜")],
                                      ["Chapter 34: Night of Betrayal"])
        self.assertIn("Never omit a chapter number", fake.calls[0][0][0]["content"])
        self.assertEqual(fake.calls[0][0][1]["content"], "第３４話、裏切りの夜")

    def test_base_extraction_reuses_model_and_restores_adapter(self):
        class Batch(dict):
            def to(self, _device):
                return self

        class Tokenizer:
            model_max_length = 8192
            pad_token_id, eos_token_id = 0, 2

            def apply_chat_template(self, _messages, **_):
                return "prompt"

            def __call__(self, _prompt, **_):
                return Batch(input_ids=SimpleNamespace(shape=(1, 5)))

            def decode(self, _output, **_):
                return '{"terms":[]}'

        class Output:
            def __getitem__(self, _slice):
                return [2]

        class Model:
            config = SimpleNamespace(max_position_embeddings=8192)
            device = "cuda:0"
            adapter_enabled = True

            def __init__(self):
                self.calls = []

            @contextmanager
            def disable_adapter(self):
                self.adapter_enabled = False
                try:
                    yield
                finally:
                    self.adapter_enabled = True

            def generate(self, **_):
                self.calls.append(self.adapter_enabled)
                return Output()

        model = Model()
        pool = SimpleNamespace(acquire=Mock(return_value=(Tokenizer(), model)))
        with patch.dict("sys.modules", {"torch": SimpleNamespace(inference_mode=nullcontext)}):
            generator = _load_generator(pool, Config(translation_backend="qwen"))
        generator([{"role": "user", "content": "source"}], 32)
        generator.extract_terms([{"role": "user", "content": "pairs"}], 32)
        self.assertEqual(model.calls, [True, False])
        self.assertTrue(model.adapter_enabled)
        self.assertEqual(pool.acquire.call_count, 1)

    def test_invalid_candidate_field_types_cannot_break_page(self):
        blocks = [block("a", "ヘイズだ")]
        output = json.dumps({"translations": [{"id": "a", "text": "It's Hayes."}], "terms": [
            {"id": [], "source": "ヘイズ", "target": "Hayes", "category": "person"},
            {"id": "a", "source": "ヘイズ", "target": "Hayes", "category": []},
        ]})
        _, stats = self.run_translation(blocks, [output], translation_mode="grouped")
        self.assertEqual(blocks[0].status, "translated")
        self.assertEqual(stats["candidates_added"], 0)

    def test_name_learned_on_page_checks_later_bubbles_with_same_retry_budget(self):
        blocks = [block("a", "ヘイズだ"), block("b", "ヘイズがいる")]
        terms = json.dumps({"terms": [{"id": "a", "source": "ヘイズ", "target": "Hayes", "category": "person"}]})
        fake = FakeGenerator(["Hayes!", "Haze is here.", terms, "Hayes is here."])
        stats = translate(blocks, pool=None, memory=self.memory, config=Config(translation_backend="qwen"), page_key="p", generator=fake)
        self.assertEqual(blocks[1].translated_text, "Hayes is here.")
        self.assertEqual(blocks[1].metadata["translation_attempts"], 2)
        self.assertEqual(stats["warnings"], [])

    def test_groups_bound_page_size(self):
        blocks = [block(str(i), "あ" * 200) for i in range(9)]
        groups = list(_groups(blocks, 4, 512))
        self.assertEqual([len(group) for group in groups], [2, 2, 2, 2, 1])

    def test_cancellation_is_not_swallowed(self):
        blocks = [block("a", "待って")]

        def cancel():
            raise Cancelled("cancel")

        with self.assertRaises(Cancelled):
            translate(blocks, pool=None, memory=self.memory, config=Config(translation_backend="qwen"), page_key="p", cancel=cancel)
        with self.assertRaises(Cancelled):
            self.run_translation(blocks, [Cancelled("cancel")])

    def test_unvalidated_language_rejected(self):
        with self.assertRaises(ValueError):
            self.run_translation([block("a", "待って")], [], target_language="zh")

    def test_cuda_oom_escapes_individual_grouped_and_extraction(self):
        for mode in ("individual", "grouped"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                    self.run_translation([block("a", "待って")],
                                         [RuntimeError("CUDA out of memory")], translation_mode=mode)
        fake = FakeGenerator(["Wait!", RuntimeError("CUDA out of memory")])
        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            translate([block("a", "待って")], pool=None, memory=self.memory,
                      config=Config(translation_backend="qwen"), page_key="p", generator=fake)

    def test_oversized_target_is_explicit_failure_and_is_never_truncated(self):
        source = "待って。" * 10000
        item = block("a", source)
        fake, _ = self.run_translation([item], [ContextTooLong("Prompt exceeds token budget")])
        self.assertEqual(fake.calls[0][0][1]["content"], source)
        self.assertEqual(item.source_text, source)
        self.assertEqual(item.status, "failed")
        self.assertEqual(item.translated_text, "")
        self.assertIn("ContextTooLong", item.error)
        self.assertEqual(len(fake.calls), 1)

    def test_token_budget_checked_before_device_transfer_or_generation(self):
        class TinyTokenizer:
            model_max_length = 128

            def apply_chat_template(self, messages, **_):
                return messages[0]["content"]

            def __call__(self, prompt, **_):
                return {"input_ids": SimpleNamespace(shape=(1, len(prompt)))}

        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128), generate=Mock())
        pool = SimpleNamespace(acquire=Mock(return_value=(TinyTokenizer(), model)))
        # Exercise the real guard without model dependencies, allocations or downloads.
        with patch.dict("sys.modules", {"torch": SimpleNamespace()}):
            generator = _load_generator(pool, Config(translation_backend="qwen"))
        with self.assertRaisesRegex(ContextTooLong, "exceeds token budget"):
            generator([{"role": "user", "content": "あ" * 120}], 32)
        model.generate.assert_not_called()

    def test_target_ids_preserve_other_objects_and_use_full_page_context(self):
        blocks = [block("a", "前のセリフ"), block("b", "待って"), block("c", "後のセリフ")]
        blocks[0].status, blocks[0].translated_text = "pending", "Manual earlier text"
        blocks[0].metadata = {"term_conflicts": ["untouched"], "nested": {"value": [1, 2]}}
        blocks[2].status, blocks[2].error = "overflow", "keep layout warning"
        blocks[2].translated_text = "Manual later text"
        blocks[2].safe_box = (1, 2, 90, 99)
        before, metadata_objects = deepcopy(blocks), [b.metadata for b in blocks]
        fake = FakeGenerator(["Wait!", '{"terms":[]}'])
        stats = translate(blocks, pool=None, memory=self.memory, config=Config(translation_backend="qwen"), page_key="edited",
                          generator=fake, target_ids={"b"})
        self.assertEqual(blocks[0], before[0])
        self.assertEqual(blocks[2], before[2])
        self.assertIs(blocks[0].metadata, metadata_objects[0])
        self.assertIs(blocks[2].metadata, metadata_objects[2])
        self.assertEqual(blocks[1].translated_text, "Wait!")
        self.assertEqual(fake.calls[0][0][1]["content"], "待って")
        self.assertIn("前のセリフ", fake.calls[0][0][0]["content"])
        self.assertIn("後のセリフ", fake.calls[0][0][0]["content"])
        extraction_pairs = json.loads(fake.calls[1][0][-1]["content"])
        self.assertEqual([pair["id"] for pair in extraction_pairs], ["b"])
        self.assertEqual(stats["generation_calls"], 1)
        self.memory.upsert_term("後のセリフ", "Later dialogue")
        self.assertIn("edited", self.memory.affected_pages("後のセリフ"))

    def test_target_ids_grouped_candidates_and_conflicts_cannot_modify_other_bubbles(self):
        blocks = [block("a", "ヘイズだ"), block("b", "リナだ")]
        blocks[0].status, blocks[0].translated_text = "translated", "Haze!"
        blocks[0].metadata = {"translation_attempts": 0, "custom": "keep"}
        self.memory.upsert_term("ヘイズ", "Hayes", locked=True)
        before = deepcopy(blocks[0])
        output = json.dumps({"translations": [{"id": "b", "text": "Rina!"}], "terms": [
            {"id": "a", "source": "ヘイズ", "target": "Haze", "category": "person"},
            {"id": "b", "source": "リナ", "target": "Rina", "category": "person"},
        ]})
        fake = FakeGenerator([output])
        stats = translate(blocks, pool=None, memory=self.memory, config=Config(translation_backend="qwen", translation_mode="grouped"),
                          page_key="p", generator=fake, target_ids={"b"})
        self.assertEqual(blocks[0], before)
        self.assertEqual(blocks[1].translated_text, "Rina!")
        self.assertEqual(stats["candidates_added"], 1)
        self.assertEqual(stats["warnings"], [])
        payload = json.loads(fake.calls[0][0][1]["content"])
        self.assertEqual([row["id"] for row in payload["targets"]], ["b"])

    def test_unknown_or_duplicate_ids_fail_before_any_mutation(self):
        for ids, targets in [(["a", "b"], {"unknown"}), (["a", "a"], {"a"}),
                             (["", "b0001"], None), (["a", "a"], set())]:
            with self.subTest(ids=ids, targets=targets):
                blocks = [block(key, "待って") for key in ids]
                blocks[0].translated_text = "Existing text"
                blocks[0].metadata = {"nested": [1, 2]}
                before = deepcopy(blocks)
                fake = FakeGenerator([])
                with self.assertRaises(ValueError):
                    translate(blocks, pool=None, memory=self.memory, config=Config(translation_backend="qwen"), page_key="p",
                              generator=fake, target_ids=targets)
                self.assertEqual(blocks, before)
                self.assertEqual(fake.calls, [])

    def test_empty_target_set_is_noop_without_model_or_memory_writes(self):
        blocks = [block("a", "待って"), block("", "．．．")]
        blocks[0].translated_text, blocks[0].status = "Keep me", "translated"
        before = deepcopy(blocks)
        with patch.object(self.memory, "record_page") as record:
            stats = translate(blocks, pool=None, memory=self.memory, config=Config(translation_backend="qwen"), page_key="p", target_ids=set())
        self.assertEqual(blocks, before)
        self.assertEqual(stats["generation_calls"], 0)
        self.assertEqual(stats["candidate_calls"], 0)
        record.assert_not_called()

    def test_selected_failure_does_not_clear_other_translations(self):
        blocks = [block("a", "行け"), block("b", "待って")]
        blocks[0].translated_text, blocks[0].status = "Go!", "translated"
        before = deepcopy(blocks[0])
        fake = FakeGenerator(["", RuntimeError("broken")])
        translate(blocks, pool=None, memory=self.memory, config=Config(translation_backend="qwen", extract_terms=False),
                  page_key="p", generator=fake, target_ids={"b"})
        self.assertEqual(blocks[0], before)
        self.assertEqual(blocks[1].status, "failed")
        self.assertEqual(blocks[1].translated_text, "")
        self.assertEqual(len(fake.calls), 2)

    def test_record_page_false_defers_dependencies_for_translation_and_fast_path(self):
        self.memory.record_page("p", ["元の原文"])
        for source, replies in [("変更した原文", ["Changed text."]), ("．．．", [])]:
            with self.subTest(source=source):
                item = block("a", source)
                with patch.object(self.memory, "record_page") as record:
                    translate([item], pool=None, memory=self.memory, config=Config(translation_backend="qwen", extract_terms=False),
                              page_key="p", generator=FakeGenerator(replies), target_ids={"a"}, record_page=False)
                record.assert_not_called()
                self.assertEqual(item.status, "translated")
        self.assertEqual(self.memory.affected_pages("元の原文"), ["p"])
        self.assertEqual(self.memory.affected_pages("変更した原文"), [])


if __name__ == "__main__":
    unittest.main()
