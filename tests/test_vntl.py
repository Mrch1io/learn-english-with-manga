"""VNTL prompt, ID and CUDA server boundaries; no model or HTTP service loads.

Run: .venv\\Scripts\\python.exe -m unittest discover -s tests -p test_vntl.py -v
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict, replace
import json
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core.runtime import Config, TextBlock
from core.memory import GlossaryMemory
from translation import ContextTooLong, translate
from translation import vntl


def bubble(key, source, translated=""):
    return TextBlock((0, 0, 100, 100), source_text=source, translated_text=translated,
                     block_id=key, status="translated" if translated else "pending")


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.memory = GlossaryMemory(str(Path(self.temp.name) / "memory.sqlite3"), "sample", "en")
        self.addCleanup(self.memory.close)
        self.config = Config(context_chars=128)

    def test_native_role_format_and_selected_text_keep_stable_ids(self):
        items = [bubble("right", "海賊版は読んじゃ"), bubble("left", "ダメですっ")]
        prompt, terms = vntl.build_prompt(items, items, self.memory, self.config)
        self.assertTrue(prompt.startswith("<|begin_of_text|><|start_header_id|>Metadata<|end_header_id|>\n\n"))
        self.assertIn("<|start_header_id|>Japanese<|end_header_id|>\n\n[right]: 海賊版は読んじゃ\n[left]: ダメですっ<|eot_id|>", prompt)
        self.assertTrue(prompt.endswith("<|start_header_id|>English<|end_header_id|>\n\n"))
        self.assertNotIn("<|start_header_id|>system", prompt)
        self.assertNotIn("<|start_header_id|>user", prompt)
        self.assertEqual(terms, [])

    def test_single_fragment_retry_keeps_ending_and_does_not_request_neighbor(self):
        items = [bubble("right", "海賊版は読んじゃ"), bubble("left", "ダメですっ")]
        prompt, _ = vntl.build_prompt(items, [items[0]], self.memory, self.config,
                                     previous_error="Missing bubble ID")
        self.assertIn("[context after] [left]: ダメですっ", prompt)
        target = prompt.rsplit("<|start_header_id|>Japanese<|end_header_id|>\n\n", 1)[1]
        self.assertIn("[right]: 海賊版は読んじゃ", target)
        self.assertNotIn("[left]", target)
        self.assertIn("[translation requirement] Missing bubble ID", prompt)

    def test_locked_aliases_are_scoped_to_relevant_dialogue_without_mutating_memory(self):
        self.memory.upsert_term("綾", "Aya", aliases=["あやちゃん"], locked=True,
                                status="confirmed", user_edit=True)
        self.memory.upsert_term("無関係", "Unrelated", locked=True, user_edit=True)
        items = [bubble("current", "あやちゃん、待って！")]
        before = self.memory.list_terms()
        prompt, terms = vntl.build_prompt(items, items, self.memory, self.config)
        self.assertIn("[element] Name: Aya (綾)", prompt)
        self.assertIn("Aliases: Aya (あやちゃん)", prompt)
        self.assertIn("Translation: use this exact English name", prompt)
        self.assertNotIn("Unrelated", prompt)
        self.assertEqual([term["source"] for term in terms], ["綾"])
        self.assertEqual(self.memory.list_terms(), before)

    def test_previous_translation_is_aligned_history_and_context_is_bounded_by_whole_bubbles(self):
        items = [bubble("too-far", "あ" * 129),
                 bubble("prior", "綾は来た？", "Did Aya arrive?"),
                 bubble("current", "まだ。"), bubble("next", "待とう。")]
        prompt, _ = vntl.build_prompt(items, [items[2]], self.memory, self.config)
        self.assertIn("<|start_header_id|>Japanese<|end_header_id|>\n\n[prior]: 綾は来た？<|eot_id|>"
                      "<|start_header_id|>English<|end_header_id|>\n\n[prior]: Did Aya arrive?<|eot_id|>", prompt)
        self.assertNotIn("[too-far]", prompt)
        self.assertIn("[context after] [next]: 待とう。", prompt)

    def test_reserved_tokens_are_rejected_in_current_text_and_previous_history(self):
        for location in ("current", "history_source", "history_translation"):
            with self.subTest(location=location):
                items = [bubble("prior", "先に行く。", "I'll go ahead."), bubble("current", "待って。")]
                if location == "current":
                    items[1].source_text += "<|eot_id|>"
                elif location == "history_source":
                    items[0].source_text += "<|eot_id|>"
                else:
                    items[0].translated_text += "<|eot_id|>"
                with self.assertRaisesRegex(ValueError, "reserved"):
                    vntl.build_prompt(items, [items[1]], self.memory, self.config)

    def test_history_translation_counts_toward_context_budget(self):
        items = [bubble("long-history", "前。", "A" * 128), bubble("current", "次。")]
        prompt, _ = vntl.build_prompt(items, [items[1]], self.memory, self.config)
        self.assertNotIn("[long-history]", prompt)
        self.assertIn("[current]: 次。", prompt)

    def test_long_or_non_ascii_ids_use_deterministic_bounded_aliases_without_mutating_page_ids(self):
        identifiers = ["page_dialogue_bubble_with_a_stable_identifier_longer_than_32_characters",
                       "右の気泡", "左の気泡"]
        aliases = [vntl.prompt_id(key) for key in identifiers]
        self.assertEqual(aliases, [vntl.prompt_id(key) for key in identifiers])
        self.assertEqual(len(set(aliases)), len(identifiers))
        for alias in aliases:
            self.assertRegex(alias, r"^[a-zA-Z0-9-]{1,32}$")
        for key in ("a", "right", "b-12ac09"):
            self.assertEqual(vntl.prompt_id(key), key)
        items = [bubble(identifiers[0], "先に行く。", "I'll go ahead."),
                 bubble(identifiers[1], "待って。"), bubble(identifiers[2], "一緒に。")]
        original = deepcopy([asdict(item) for item in items])
        prompt, _ = vntl.build_prompt(items, [items[1]], self.memory, self.config)
        self.assertIn(f"[{aliases[0]}]: I'll go ahead.", prompt)
        self.assertIn(f"[{aliases[1]}]: 待って。", prompt)
        self.assertIn(f"[context after] [{aliases[2]}]: 一緒に。", prompt)
        self.assertEqual([asdict(item) for item in items], original)


class ResponseTests(unittest.TestCase):
    def test_alias_responses_map_back_to_long_and_non_ascii_original_ids(self):
        first, second = "right_bubble_identifier_which_is_longer_than_32_characters", "左の気泡"
        result, errors = vntl.parse_response(
            f"[{vntl.prompt_id(second).upper()}]: Wait!\n[{vntl.prompt_id(first)}]: Go!", [first, second])
        self.assertEqual(result, {first: "Go!", second: "Wait!"})
        self.assertEqual(errors, {})

    def test_alias_collision_rejects_all_ambiguous_targets(self):
        expected = ["first_stable_original_id", "second_stable_original_id"]
        with patch.object(vntl, "prompt_id", return_value="b-collision"):
            result, errors = vntl.parse_response("[b-collision]: Guessing is forbidden.", expected)
        self.assertEqual(result, {})
        self.assertEqual(set(errors), set(expected))
        self.assertTrue(all("Ambiguous" in error for error in errors.values()))

    def test_model_uppercased_ids_map_back_to_original_stable_ids_without_position_guessing(self):
        result, errors = vntl.parse_response("[B-0FA2]: Wait!\n[B-91dC]: Go!", ["b-91dc", "b-0fa2"])
        self.assertEqual(result, {"b-91dc": "Go!", "b-0fa2": "Wait!"})
        self.assertEqual(errors, {})

    def test_duplicate_id_with_mixed_case_is_still_duplicate(self):
        result, errors = vntl.parse_response("[B-0FA2]: First\n[b-91dc]: Okay\n[b-0fa2]: Second",
                                             ["b-0fa2", "b-91dc"])
        self.assertEqual(result, {"b-91dc": "Okay"})
        self.assertEqual(errors, {"b-0fa2": "Duplicate bubble ID"})

    def test_expected_ids_that_differ_only_in_case_are_rejected_as_ambiguous(self):
        result, errors = vntl.parse_response("[a]: First\n[A]: Second", ["a", "A"])
        self.assertEqual(result, {})
        self.assertEqual(errors, {"a": "Ambiguous case-insensitive bubble IDs",
                                  "A": "Ambiguous case-insensitive bubble IDs"})

    def test_order_is_resolved_by_id_and_multiline_text_is_preserved(self):
        result, errors = vntl.parse_response("[left]: is forbidden!\nSecond line.\n[right]: Reading pirated copies...",
                                             ["right", "left"])
        self.assertEqual(result, {"right": "Reading pirated copies...", "left": "is forbidden!\nSecond line."})
        self.assertEqual(errors, {})

    def test_missing_empty_and_repeated_ids_only_invalidate_their_own_bubbles(self):
        cases = [
            ("[right]: Reading...", {"right": "Reading..."}, {"left": "Missing bubble ID"}),
            ("[right]: Reading...\n[left]:", {"right": "Reading..."}, {"left": "Empty translation"}),
            ("[right]: First\n[left]: Okay\n[right]: Second", {"left": "Okay"}, {"right": "Duplicate bubble ID"}),
        ]
        for raw, expected, failures in cases:
            with self.subTest(raw=raw):
                self.assertEqual(vntl.parse_response(raw, ["right", "left"]), (expected, failures))

    def test_unknown_id_or_preamble_rejects_untrusted_mapping(self):
        for raw in ("[right]: One\n[unknown]: Two", "Here is the translation:\n[right]: One\n[left]: Two"):
            with self.subTest(raw=raw):
                result, errors = vntl.parse_response(raw, ["right", "left"])
                self.assertEqual(result, {})
                self.assertEqual(set(errors), {"right", "left"})
                self.assertTrue(all("Unexpected" in error for error in errors.values()))

    def test_unlabelled_text_only_allowed_for_a_single_requested_id(self):
        self.assertEqual(vntl.parse_response("Wait!", ["only"]), ({"only": "Wait!"}, {}))
        result, errors = vntl.parse_response("One\nTwo", ["right", "left"])
        self.assertEqual(result, {})
        self.assertEqual(set(errors), {"right", "left"})
        for raw in (None, "", " \n "):
            self.assertEqual(vntl.parse_response(raw, ["only"]), ({}, {"only": "Empty translation"}))


class LabelGrammarTests(unittest.TestCase):
    def test_grammar_requires_all_requested_labels_once_and_only_literals_are_labels_or_newlines(self):
        ids = ["right", "左の気泡", "long_identifier_for_a_bubble_that_needs_an_alias"]
        grammar = vntl.label_grammar(ids)
        root, text = grammar.strip().splitlines()
        literals = [json.loads(literal) for literal in re.findall(r'"(?:\\.|[^"\\])*"', root)]
        self.assertEqual(literals, [f"[{vntl.prompt_id(ids[0])}]: ",
                                    f"\n[{vntl.prompt_id(ids[1])}]: ",
                                    f"\n[{vntl.prompt_id(ids[2])}]: "])
        # Every label has its own generated text. No source-specific answer,
        # word list or vocabulary restriction is embedded in the grammar.
        self.assertEqual(root.count(" text"), len(ids))
        self.assertEqual(re.findall(r'"(?:\\.|[^"\\])*"', text), ['"\\n"'])
        self.assertIn(r"[^\n]+", text)

    def test_grammar_rejects_case_or_alias_collisions_before_completion(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            vntl.label_grammar(["a", "A"])
        with patch.object(vntl, "prompt_id", return_value="b-same"), self.assertRaisesRegex(ValueError, "Ambiguous"):
            vntl.label_grammar(["first", "second"])


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.gguf, self.executable = root / "model.gguf", root / "llama-server.exe"
        self.gguf.write_bytes(b"test model file; not actual weights")
        self.executable.write_bytes(b"test executable; never launched")
        self.config = Config(gguf_path=str(self.gguf), llama_server_path=str(self.executable), llama_context_size=2048)
        self.root_patch = patch.object(vntl, "ROOT", root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.process = Mock(pid=2468)
        self.process.poll.return_value = None
        self.process.terminate.side_effect = lambda: setattr(self.process.poll, "return_value", 0)
        self.process.kill.side_effect = lambda: setattr(self.process.poll, "return_value", 0)

    def start(self, log):
        def launch(_command, **kwargs):
            kwargs["stdout"].write(log.encode("utf-8"))
            kwargs["stdout"].flush()
            return self.process
        with patch.object(vntl.subprocess, "Popen", side_effect=launch) as spawn, \
                patch.object(vntl.VntlServer, "_request", return_value={"status": "ok"}) as request:
            server = vntl.VntlServer(self.config)
        self.addCleanup(server.close)
        return server, spawn, request

    def test_full_gpu_offload_is_required_and_launch_is_loopback_cuda(self):
        server, spawn, request = self.start("load_tensors: offloaded 33/33 layers to GPU\n"
                                            "load_tensors: CUDA0 model buffer size = 4695.00 MiB\n")
        command = spawn.call_args.args[0]
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--device") + 1], "CUDA0")
        self.assertEqual(command[command.index("--gpu-layers") + 1], "99")
        self.assertEqual(command[command.index("--ctx-size") + 1], "2048")
        self.assertEqual(command[command.index("--parallel") + 1], "1")
        self.assertNotIn("--lora", command)
        self.assertEqual(server.info["offloaded_layers"], 33)
        self.assertEqual(server.info["total_layers"], 33)
        self.assertEqual(server.info["pid"], 2468)
        self.assertGreaterEqual(server.info["load_seconds"], 0)
        request.assert_called_once_with("/health", timeout=1)
        server.close()
        server.close()
        self.process.terminate.assert_called_once_with()
        self.assertTrue(server.log.closed)

    def test_partial_offload_missing_evidence_and_cpu_buffers_fail_without_fallback(self):
        logs = ["offloaded 32/33 layers to GPU\nCUDA0 model buffer size = 4500 MiB",
                "CUDA0 model buffer size = 4500 MiB",
                "offloaded 0/33 layers to GPU\nCPU model buffer size = 4500 MiB",
                "offloaded 33/33 layers to GPU\nCPU model buffer size = 4500 MiB"]
        for log in logs:
            with self.subTest(log=log):
                self.process.reset_mock()
                self.process.poll.return_value = None
                with self.assertRaisesRegex(RuntimeError, "CUDA loading failed"):
                    self.start(log)
                self.process.terminate.assert_called_once_with()

    def test_missing_model_never_launches_another_backend(self):
        self.gguf.unlink()
        with patch.object(vntl.subprocess, "Popen") as spawn, self.assertRaises(FileNotFoundError):
            vntl.VntlServer(self.config)
        spawn.assert_not_called()

    def test_generation_rejects_context_overflow_before_completion(self):
        server = object.__new__(vntl.VntlServer)
        server.context_size = 512
        server._request = Mock(return_value={"tokens": list(range(500))})
        with self.assertRaises(ContextTooLong):
            server.generate("oversized prompt", 32)
        server._request.assert_called_once_with("/tokenize", {
            "content": "oversized prompt", "add_special": False, "parse_special": True})

    def test_generation_uses_author_baseline_and_rejects_truncation(self):
        server = object.__new__(vntl.VntlServer)
        server.context_size = 512
        for limit_field in ({}, {"truncated": True}, {"stopped_limit": True}, {"stop_type": "limit"}):
            with self.subTest(limit_field=limit_field):
                server.info = {"generation_calls": 0, "generation_seconds": 0.0}
                server._request = Mock(side_effect=[{"tokens": [1, 2, 3]}, {
                    "content": "[only]: Wait!", "timings": {"predicted_n": 5}, **limit_field}])
                if limit_field:
                    with self.assertRaisesRegex(ValueError, "token limit"):
                        server.generate("prompt", 32)
                else:
                    self.assertEqual(server.generate("prompt", 32), "[only]: Wait!")
                body = server._request.call_args_list[1].args[1]
                self.assertEqual(body["temperature"], 0)
                self.assertEqual(body["repeat_penalty"], 1.0)
                self.assertEqual(body["repeat_last_n"], 0)
                self.assertEqual(body["n_predict"], 32)
                self.assertTrue(body["cache_prompt"])
                self.assertEqual(server.info["generation_calls"], 1)

    def test_generation_sends_optional_label_grammar_without_rewriting_model_words(self):
        server = object.__new__(vntl.VntlServer)
        server.context_size = 512
        # This is deliberately not a correctness verdict: an unrelated but
        # structurally valid sentence must remain available for human review.
        content = "[only]: A structurally valid but unrelated sentence."
        for ids in (None, ["only"]):
            with self.subTest(ids=ids):
                server.info = {"generation_calls": 0, "generation_seconds": 0.0}
                server._request = Mock(side_effect=[{"tokens": [1, 2, 3]}, {"content": content}])
                self.assertEqual(server.generate("Japanese prompt", 32, ids=ids), content)
                body = server._request.call_args_list[1].args[1]
                if ids:
                    self.assertEqual(body["grammar"], vntl.label_grammar(ids))
                else:
                    self.assertNotIn("grammar", body)


class TranslationFlowTests(unittest.TestCase):
    def setUp(self):
        self.memory = GlossaryMemory(":memory:", "sample", "en")
        self.addCleanup(self.memory.close)

    def run_translation(self, items, responses, *, target_ids=None, mode="grouped", extract_terms=False):
        generate = Mock(side_effect=responses)
        with patch("translation._load_generator", side_effect=AssertionError("VNTL must not load Qwen")), \
                patch("translation._extract_candidates", side_effect=AssertionError("VNTL must not use chat candidate extraction")):
            stats = translate(items, pool=None, memory=self.memory, page_key="page.png",
                config=Config(translation_backend="vntl", translation_mode=mode, extract_terms=extract_terms),
                generator=generate, target_ids=target_ids)
        return generate, stats

    def test_auto_mode_groups_vntl_and_keeps_qwen_individual(self):
        for backend, responses, calls in (
                ("vntl", ["[left]: That's not okay!\n[right]: It's a pirated copy."], 1),
                ("qwen", ["It's a pirated copy.", "That's not okay!"], 2)):
            with self.subTest(backend=backend):
                items = [bubble("right", "海賊版だ。"), bubble("left", "それはダメです。")]
                generator = Mock(side_effect=responses)
                stats = translate(items, pool=None, memory=self.memory, page_key="auto.png",
                    config=Config(translation_backend=backend, translation_mode="auto", extract_terms=False),
                    generator=generator)
                self.assertEqual(stats["generation_calls"], calls)
                self.assertEqual([item.translated_text for item in items],
                                 ["It's a pirated copy.", "That's not okay!"])

    def test_native_generator_receives_group_ids_then_only_the_failed_retry_id(self):
        calls = []
        responses = iter(["[left]: That's not okay!", "[right]: It's a pirated copy."])
        def generate(prompt, max_new_tokens, *, ids):
            self.assertIsInstance(prompt, str)
            self.assertGreater(max_new_tokens, 0)
            calls.append(ids)
            return next(responses)
        items = [bubble("right", "海賊版だ。"), bubble("left", "それはダメです。")]
        stats = translate(items, pool=None, memory=self.memory, page_key="keyword-ids.png",
            config=Config(translation_backend="vntl", translation_mode="auto", extract_terms=False),
            generator=generate)
        self.assertEqual(calls, [["right", "left"], ["right"]])
        self.assertEqual(stats["generation_calls"], 2)
        self.assertTrue(all(item.status == "translated" for item in items))

    def test_group_reordered_ids_map_to_original_bubbles_with_raw_native_prompt(self):
        items = [bubble("right", "海賊版だ。"), bubble("left", "それはダメです。")]
        generate, stats = self.run_translation(items, ["[left]: That's not okay!\n[right]: It's a pirated copy."])
        self.assertEqual([item.translated_text for item in items], ["It's a pirated copy.", "That's not okay!"])
        self.assertTrue(all(item.status == "translated" for item in items))
        self.assertEqual(stats["generation_calls"], 1)
        self.assertIsInstance(generate.call_args.args[0], str)
        self.assertTrue(generate.call_args.args[0].startswith("<|begin_of_text|>"))

    def test_group_missing_id_retries_only_that_fragment_and_retains_future_dialogue(self):
        items = [bubble("right", "海賊版だ。"), bubble("left", "それはダメです。")]
        generate, stats = self.run_translation(items, ["[left]: That's not okay!", "[right]: It's a pirated copy."])
        self.assertEqual(stats["generation_calls"], 2)
        self.assertEqual([item.metadata["translation_attempts"] for item in items], [2, 1])
        retry = generate.call_args_list[1].args[0]
        self.assertIn("[context after] [left]: それはダメです。", retry)
        target = retry.rsplit("<|start_header_id|>Japanese<|end_header_id|>\n\n", 1)[1]
        self.assertIn("[right]: 海賊版だ。", target)
        self.assertNotIn("[left]", target)
        self.assertTrue(all(item.status == "translated" for item in items))

    def test_japanese_output_retry_keeps_context_and_failure_cannot_masquerade_as_success(self):
        for retried in ("[right]: It's a pirated copy.", "[right]: 海賊版だ。"):
            with self.subTest(retried=retried):
                items = [bubble("right", "海賊版だ。"), bubble("left", "それはダメです。")]
                generate, stats = self.run_translation(items, ["[right]: 海賊版だ。それはダメです。", retried],
                                                        target_ids={"right"}, mode="individual")
                self.assertEqual(stats["generation_calls"], 2)
                self.assertNotIn("translation_retry_without_context", items[0].metadata)
                self.assertEqual(items[1].status, "pending")
                self.assertEqual(items[1].translated_text, "")
                for call in generate.call_args_list:
                    self.assertIn("[context after] [left]: それはダメです。", call.args[0])
                self.assertEqual(items[0].source_text, "海賊版だ。")
                if "pirated copy" in retried:
                    self.assertEqual(items[0].status, "translated")
                else:
                    self.assertEqual(items[0].status, "failed")
                    self.assertEqual(items[0].translated_text, "")
                    self.assertIn("Expected English", items[0].error)

    def test_local_retranslation_keeps_neighbor_manual_edits_and_original_ids(self):
        items = [bubble("prior", "先に行く。", "I'll go ahead."), bubble("current", "待って。", "Wait."),
                 bubble("next", "一緒に。", "Together.")]
        items[0].metadata["manual_translation"] = True
        before = deepcopy([asdict(item) for item in items])
        generate, stats = self.run_translation(items, ["[current]: Hold on!"], target_ids={"current"})
        self.assertEqual(asdict(items[0]), before[0])
        self.assertEqual(asdict(items[2]), before[2])
        self.assertEqual(items[1].block_id, "current")
        self.assertEqual(items[1].translated_text, "Hold on!")
        self.assertIn("[prior]: I'll go ahead.", generate.call_args.args[0])
        self.assertIn("[context after] [next]: 一緒に。", generate.call_args.args[0])
        self.assertEqual(stats["generation_calls"], 1)

    def test_candidate_warning_never_mutates_locked_terms_or_loads_qwen(self):
        self.memory.upsert_term("綾", "Aya", aliases=["あやちゃん"], locked=True, user_edit=True,
                                first_page="earlier.png")
        before = self.memory.list_terms()
        items = [bubble("only", "あやちゃんが来た。")]
        _, stats = self.run_translation(items, ["[only]: Aya has arrived."], extract_terms=True)
        self.assertEqual(self.memory.list_terms(), before)
        self.assertEqual(stats["candidate_calls"], 0)
        self.assertEqual(stats["candidates_added"], 0)
        self.assertTrue(any("confirmation/import" in warning for warning in stats["warnings"]))
        self.assertEqual(items[0].status, "translated")

    def test_empty_and_punctuation_bubbles_do_not_load_or_call_any_model(self):
        items = [bubble("empty", "  "), bubble("pause", "……！？")]
        generate, stats = self.run_translation(items, [])
        generate.assert_not_called()
        self.assertEqual(stats["generation_calls"], 0)
        self.assertEqual(items[0].status, "empty")
        self.assertEqual(items[0].translated_text, "")
        self.assertEqual(items[1].status, "translated")
        self.assertEqual(items[1].translated_text, "......!?")

    def test_production_dispatch_reuses_one_server_and_never_invokes_qwen_loader(self):
        server = Mock(info={"load_seconds": 0.25}, generate=Mock(return_value="[only]: Wait!"))
        pool = SimpleNamespace(vntl=None, offload=Mock(), load_seconds={}, loads={})
        with patch("translation.vntl.VntlServer", return_value=server) as factory, \
                patch("translation._load_generator", side_effect=AssertionError("No Qwen or LoRA")), \
                patch("translation._extract_candidates", side_effect=AssertionError("No Qwen candidate prompt")):
            for page in ("one.png", "two.png"):
                items = [bubble("only", "待って！")]
                translate(items, pool=pool, memory=self.memory, page_key=page,
                          config=Config(translation_backend="vntl", extract_terms=False))
                self.assertEqual(items[0].translated_text, "Wait!")
        factory.assert_called_once()
        self.assertEqual(server.generate.call_count, 2)
        self.assertEqual(pool.offload.call_count, 2)
        self.assertEqual(list(pool.loads.values()), [1])

    def test_model_server_context_or_device_changes_close_previous_server_before_reloading(self):
        baseline = Config(translation_backend="vntl", extract_terms=False)
        for change in ({"gguf_path": "different-model.gguf"},
                       {"llama_server_path": "different-server.exe"},
                       {"llama_context_size": 1024}, {"device": "cuda:1"}):
            with self.subTest(change=change):
                events = []
                first = Mock(info={"load_seconds": 0.25}, generate=Mock(return_value="[only]: Wait!"))
                second = Mock(info={"load_seconds": 0.5}, generate=Mock(return_value="[only]: Hold on!"))
                first.close.side_effect = lambda: events.append("close-first")
                servers = iter([first, second])
                def create(config):
                    events.append("load")
                    return next(servers)
                pool = SimpleNamespace(vntl=None, offload=Mock(), load_seconds={}, loads={})
                with patch("translation.vntl.VntlServer", side_effect=create) as factory, \
                        patch("translation._load_generator", side_effect=AssertionError("No Qwen")):
                    for config in (baseline, replace(baseline, **change)):
                        items = [bubble("only", "待って！")]
                        translate(items, pool=pool, memory=self.memory, page_key="page.png", config=config)
                self.assertEqual(events, ["load", "close-first", "load"])
                first.close.assert_called_once_with()
                second.close.assert_not_called()
                self.assertEqual(factory.call_count, 2)
                self.assertIs(pool.vntl, second)
                self.assertEqual(sum(pool.loads.values()), 2)
                self.assertAlmostEqual(sum(pool.load_seconds.values()), 0.75)
                self.assertEqual(items[0].translated_text, "Hold on!")

    def test_switching_to_qwen_releases_vntl_before_loading_qwen(self):
        events = []
        server = Mock()
        server.close.side_effect = lambda: events.append("close-vntl")
        pool = SimpleNamespace(vntl=server)
        generator = Mock(return_value="Wait!")
        def load_qwen(_pool, _config):
            events.append("load-qwen")
            self.assertIsNone(_pool.vntl)
            return generator
        items = [bubble("only", "待って！")]
        with patch("translation._load_generator", side_effect=load_qwen) as loader, \
                patch("translation.vntl.VntlServer", side_effect=AssertionError("Do not reload VNTL")):
            translate(items, pool=pool, memory=self.memory, page_key="qwen.png",
                      config=Config(translation_backend="qwen", extract_terms=False))
        self.assertEqual(events, ["close-vntl", "load-qwen"])
        loader.assert_called_once()
        server.close.assert_called_once_with()
        self.assertIsNone(pool.vntl)
        self.assertEqual(items[0].translated_text, "Wait!")


if __name__ == "__main__":
    unittest.main()
