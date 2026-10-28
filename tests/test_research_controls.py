"""Controlled prompt ablations; all generators/loaders are CPU-only test doubles.

Run: .venv\\Scripts\\python.exe -m unittest discover -s tests -p test_research_controls.py -v
"""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core.runtime import Config, TextBlock
from core.memory import GlossaryMemory
from translation import _load_generator, _messages, translate
from translation.vntl import build_prompt


def bubble(key, source, translated=""):
    return TextBlock((0, 0, 100, 100), source_text=source, translated_text=translated,
                     block_id=key, status="translated" if translated else "pending")


class ResearchPromptTests(unittest.TestCase):
    def setUp(self):
        self.memory = GlossaryMemory(":memory:", "research", "en")
        self.addCleanup(self.memory.close)

    def test_zero_context_is_valid_but_negative_and_small_budgets_are_rejected(self):
        Config(context_chars=0).validate()
        Config(context_chars=128).validate()
        for budget in (-1, 1, 127):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "context_chars"):
                Config(context_chars=budget).validate()

    def test_production_context_and_continuation_defaults_are_preserved(self):
        config = Config()
        self.assertEqual(config.context_chars, 1200)
        self.assertTrue(config.use_continuations)
        self.assertTrue(config.use_translation_history)

    def test_qwen_zero_context_excludes_neighbor_text_and_neighbor_only_terms(self):
        self.memory.upsert_term("綾", "Aya", locked=True, user_edit=True)
        items = [bubble("prior", "綾は来た？", "PRIVATE_PREVIOUS_ENGLISH"),
                 bubble("current", "まだ。"), bubble("next", "待とう。")]
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                messages, terms = _messages(items, [items[1]], self.memory,
                    Config(translation_backend="qwen", context_chars=0), grouped=grouped)
                prompt = str(messages)
                self.assertIn("まだ。", prompt)
                for hidden in ("綾は来た？", "待とう。", "Aya", "PRIVATE_PREVIOUS_ENGLISH"):
                    self.assertNotIn(hidden, prompt)
                self.assertEqual(terms, [])

    def test_vntl_zero_context_excludes_neighbor_japanese_english_and_terms(self):
        self.memory.upsert_term("綾", "Aya", locked=True, user_edit=True)
        items = [bubble("prior", "綾は来た？", "PRIVATE_PREVIOUS_ENGLISH"),
                 bubble("current", "まだ。"), bubble("next", "待とう。")]
        prompt, terms = build_prompt(items, [items[1]], self.memory, Config(context_chars=0))
        self.assertIn("[current]: まだ。", prompt)
        for hidden in ("綾は来た？", "待とう。", "Aya", "PRIVATE_PREVIOUS_ENGLISH"):
            self.assertNotIn(hidden, prompt)
        self.assertEqual(terms, [])

    def test_history_off_keeps_japanese_context_without_english_or_history_cost(self):
        items = [bubble("prior", "先に行く。", "PRIVATE_PREVIOUS_ENGLISH" * 30),
                 bubble("current", "待って。"), bubble("next", "一緒に。")]
        config = Config(context_chars=128, use_translation_history=False)
        prompt, _ = build_prompt(items, [items[1]], self.memory, config)
        self.assertIn("[context before] [prior]: 先に行く。", prompt)
        self.assertIn("[context after] [next]: 一緒に。", prompt)
        self.assertNotIn("PRIVATE_PREVIOUS_ENGLISH", prompt)
        self.assertEqual(prompt.count("<|start_header_id|>Japanese"), 1)
        self.assertEqual(prompt.count("<|start_header_id|>English"), 1)

    def test_zero_context_single_retry_cannot_join_or_read_other_bubble(self):
        items = [bubble("right", "海賊版は読んじゃ", "PRIVATE_PREVIOUS_ENGLISH"),
                 bubble("left", "ダメですっ")]
        before = deepcopy(asdict(items[0]))
        generator = Mock(side_effect=["", "[left]: No!"])
        config = Config(context_chars=0, translation_mode="individual", extract_terms=False)
        with patch("translation.vntl.continuation_pairs", side_effect=AssertionError("No neighbor analysis allowed")):
            stats = translate(items, pool=None, memory=self.memory, config=config,
                page_key="p", generator=generator, target_ids={"left"})
        self.assertEqual(stats["generation_calls"], 2)
        self.assertEqual(items[1].translated_text, "No!")
        self.assertEqual(asdict(items[0]), before)
        for call in generator.call_args_list:
            prompt = call.args[0]
            self.assertNotIn("海賊版は読んじゃ", prompt)
            self.assertNotIn("PRIVATE_PREVIOUS_ENGLISH", prompt)
            self.assertIn("[left]: ダメですっ", prompt)
            self.assertEqual(call.kwargs["ids"], ["left"])
        self.assertNotIn("translation_unit", items[1].metadata)

    def test_continuation_off_retains_context_but_keeps_one_target_per_call(self):
        items = [bubble("right", "海賊版は読んじゃ"), bubble("left", "ダメですっ")]
        generator = Mock(side_effect=lambda prompt, limit, *, ids: f"[{ids[0]}]: Do not.")
        config = Config(translation_mode="individual", use_continuations=False,
                        use_translation_history=False, extract_terms=False)
        with patch("translation.vntl.continuation_pairs", side_effect=AssertionError("Continuation rule disabled")):
            translate(items, pool=None, memory=self.memory, config=config, page_key="p", generator=generator)
        self.assertEqual(generator.call_count, 2)
        self.assertEqual([call.kwargs["ids"] for call in generator.call_args_list], [["right"], ["left"]])
        self.assertIn("[context after] [left]: ダメですっ", generator.call_args_list[0].args[0])
        self.assertIn("[context before] [right]: 海賊版は読んじゃ", generator.call_args_list[1].args[0])
        self.assertTrue(all("translation_unit" not in item.metadata for item in items))

    def test_request_log_keeps_exact_group_and_retry_prompts_without_references(self):
        for backend in ("qwen", "vntl"):
            with self.subTest(backend=backend):
                items = [bubble("a", "先に行く。"), bubble("b", "待って。")]
                for item in items:
                    item.metadata["reference_en"] = "PRIVATE_EVALUATION_REFERENCE"
                responses = (["[a]: I'll go ahead.", "[b]: Wait!"] if backend == "vntl" else
                    ['{"translations":[{"id":"a","text":"I will go ahead."}]}', "Wait!"])
                generator, request_log = Mock(side_effect=responses), []
                config = Config(translation_backend=backend, translation_mode="grouped", context_chars=0,
                    use_continuations=False, use_translation_history=False, extract_terms=False)
                stats = translate(items, pool=None, memory=self.memory, config=config,
                    page_key="p", generator=generator, request_log=request_log)
                self.assertEqual(stats["generation_calls"], 2)
                self.assertEqual([row["ids"] for row in request_log], [["a", "b"], ["b"]])
                self.assertEqual([row["attempt"] for row in request_log], [1, 2])
                self.assertEqual([row["max_new_tokens"] for row in request_log], [512, 256])
                self.assertEqual([row["response"] for row in request_log], responses)
                for record, call in zip(request_log, generator.call_args_list):
                    self.assertEqual(record["prompt"], call.args[0])
                    self.assertEqual(record["max_new_tokens"], call.args[1])
                    self.assertNotIn("PRIVATE_EVALUATION_REFERENCE", str(record))
                self.assertTrue(all(item.status == "translated" for item in items))

    def test_request_log_continuation_records_actual_unit_id_and_joined_prompt(self):
        items = [bubble("right", "海賊版は読んじゃ"), bubble("left", "ダメですっ")]
        request_log = []
        generator = Mock(side_effect=lambda prompt, limit, *, ids:
                         f"[{ids[0]}]: You must not read pirated copies.")
        translate(items, pool=None, memory=self.memory, config=Config(extract_terms=False),
                  page_key="p", generator=generator, request_log=request_log)
        self.assertEqual(len(request_log), 1)
        self.assertEqual(request_log[0]["ids"], generator.call_args.kwargs["ids"])
        self.assertTrue(request_log[0]["ids"][0].startswith("u-"))
        self.assertEqual(request_log[0]["prompt"], generator.call_args.args[0])
        self.assertIn("海賊版は読んじゃダメですっ", request_log[0]["prompt"])

    def test_request_log_records_failed_calls_and_stops_after_one_retry(self):
        requests = []
        generator = Mock(side_effect=RuntimeError("Model request failed"))
        item = bubble("current", "待って。")
        stats = translate([item], pool=None, memory=self.memory,
            config=Config(context_chars=0, extract_terms=False, translation_mode="individual"),
            page_key="p", generator=generator, request_log=requests)
        self.assertEqual(stats["generation_calls"], 2)
        self.assertEqual([record["attempt"] for record in requests], [1, 2])
        self.assertTrue(all(record["error"] == "RuntimeError: Model request failed" for record in requests))
        self.assertTrue(all("response" not in record for record in requests))
        self.assertEqual(item.status, "failed")
        self.assertIn("Model request failed", requests[1]["prompt"])

    def test_request_log_retains_malformed_response_before_retry(self):
        for backend in ("qwen", "vntl"):
            with self.subTest(backend=backend):
                requests = []
                responses = ["{not valid JSON", "[current]: Wait!" if backend == "vntl" else "Wait!"]
                item = bubble("current", "待って。")
                stats = translate([item], pool=None, memory=self.memory,
                    config=Config(translation_backend=backend, context_chars=0, extract_terms=False,
                                  translation_mode="individual"), page_key="p",
                    generator=Mock(side_effect=responses), request_log=requests)
                self.assertEqual(stats["generation_calls"], 2)
                self.assertEqual([record["response"] for record in requests], responses)
                self.assertEqual(item.translated_text, "Wait!")

    def test_request_log_oom_is_recorded_and_still_propagates(self):
        for mode in ("individual", "grouped"):
            with self.subTest(mode=mode):
                requests = []
                error = RuntimeError("CUDA out of memory")
                with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                    translate([bubble("current", "待って。")], pool=None, memory=self.memory,
                        config=Config(context_chars=0, extract_terms=False, translation_mode=mode),
                        page_key="p", generator=Mock(side_effect=error), request_log=requests)
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0]["error"], "RuntimeError: CUDA out of memory")


class ResearchModelIsolationTests(unittest.TestCase):
    def test_qwen_base_and_adapter_have_distinct_cache_keys_and_only_adapter_loads_peft(self):
        # Exercise the real loader/cache boundary without importing torch or
        # allocating model tensors. Repeated base calls reuse the same entry.
        tokenizer = SimpleNamespace(model_max_length=8192)
        base = Mock(config=SimpleNamespace(max_position_embeddings=8192))
        adapted = Mock(config=SimpleNamespace(max_position_embeddings=8192))
        model_loader = Mock(return_value=base)
        adapter_loader = Mock(return_value=adapted)
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader))
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda: True),
                                     bfloat16="bf16", float16="f16", float32="f32")
        entries = {}

        def acquire(name, loader, _mover):
            if name not in entries:
                entries[name] = loader()
            return entries[name]

        pool = SimpleNamespace(acquire=acquire, device="cuda:0")
        with tempfile.TemporaryDirectory() as directory:
            config = Config(translation_backend="qwen", adapter_path=directory, local_files_only=True)
            modules = {"torch": fake_torch, "transformers": fake_transformers,
                       "peft": SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=adapter_loader))}
            with patch.dict("sys.modules", modules), patch("translation._windows_streaming_weights", nullcontext):
                _load_generator(pool, config)
                _load_generator(pool, replace(config, adapter_path=""))
                _load_generator(pool, replace(config, adapter_path=""))
            self.assertEqual(set(entries), {f"translation:{config.model_name}:{directory}",
                                            f"translation:{config.model_name}:"})
            self.assertIs(entries[f"translation:{config.model_name}:"][1], base)
            self.assertIs(entries[f"translation:{config.model_name}:{directory}"][1], adapted)
            adapter_loader.assert_called_once_with(base, str(Path(directory)), is_trainable=False,
                                                    local_files_only=True)
            self.assertEqual(model_loader.call_count, 2)
            self.assertTrue(all(call.kwargs["device_map"] == {"": "cuda:0"}
                                for call in model_loader.call_args_list))


if __name__ == "__main__":
    unittest.main()
