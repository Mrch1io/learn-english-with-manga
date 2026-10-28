"""Permanent VNTL input failures must not spend model calls or poison neighbors.

Run: .venv\\Scripts\\python.exe -m unittest tests.test_translation_input_validation -v
All HTTP, model loading and generated responses are mocked; SQLite is in memory.
"""
from copy import deepcopy
from dataclasses import replace
import re
import unittest
from unittest.mock import Mock, patch

from core.runtime import Config, TextBlock
from core.memory import GlossaryMemory
from translation import translate
from translation import vntl


def bubble(key, source):
    return TextBlock((0, 0, 100, 100), source_text=source, block_id=key)


class TranslationInputValidationTests(unittest.TestCase):
    def setUp(self):
        self.memory = GlossaryMemory(":memory:", "input-validation")
        self.addCleanup(self.memory.close)
        self.config = Config(translation_backend="vntl", translation_mode="grouped",
                             extract_terms=False, use_continuations=False)

    def invalid_source(self, kind, memory=None):
        memory = self.memory if memory is None else memory
        if kind == "reserved_source":
            return "待て<|eot_id|>"
        if kind == "reserved_term":
            memory.upsert_term("甲", "Name<|eot_id|>", locked=True, user_edit=True)
            return "甲だ。"
        if kind == "alias_conflict":
            memory.upsert_term("甲", "Alpha", aliases=["あだな"], locked=True, user_edit=True)
            memory.upsert_term("乙", "Beta", aliases=["あだな"], locked=True, user_edit=True)
            return "あだなが来た。"
        if kind == "locked_overlap":
            memory.upsert_term("甲", "Mary Jane", locked=True, user_edit=True)
            memory.upsert_term("乙", "Jane Smith", locked=True, user_edit=True)
            return "甲と乙だ。"
        for source, target in zip("甲乙丙丁戊", ("Alpha", "Bravo", "Charlie", "Delta", "Echo")):
            memory.upsert_term(source, target, locked=True, user_edit=True)
        return "甲乙丙丁戊だ。"

    def server(self):
        server = object.__new__(vntl.VntlServer)
        server.context_size = 2048
        server.info = {"generation_calls": 0, "generation_seconds": 0.0}
        def request(route, body):
            if route == "/tokenize":
                return {"tokens": [1, 2, 3]}
            self.assertEqual(route, "/completion")
            target = body["prompt"].rsplit(vntl._header("Japanese"), 1)[1]
            ids = re.findall(r"^\[([^\]]+)\]:", target, re.MULTILINE)
            return {"content": "\n".join(f"[{key}]: Wait!" for key in ids)}
        server._request = Mock(side_effect=request)
        return server

    def test_invalid_bubble_is_rejected_before_loading_building_or_tokenizing(self):
        for kind in ("reserved_source", "reserved_term", "alias_conflict", "locked_overlap", "locked_limit"):
            for mode in ("individual", "grouped"):
                with self.subTest(kind=kind, mode=mode), GlossaryMemory(":memory:", "invalid") as memory:
                    source = self.invalid_source(kind, memory)
                    item = bubble("bad", source)
                    server = self.server()
                    with patch("translation.vntl.build_prompt", wraps=vntl.build_prompt) as build:
                        stats = translate([item], pool=None, memory=memory,
                            config=replace(self.config, translation_mode=mode), page_key="p",
                            generator=server.generate)
                    self.assertEqual(item.source_text, source)
                    self.assertEqual(item.status, "failed")
                    self.assertEqual(item.metadata["translation_attempts"], 0)
                    self.assertEqual(stats["generation_calls"], 0)
                    server._request.assert_not_called()
                    build.assert_not_called()
                    pool = Mock()
                    with patch("translation.vntl.VntlServer") as factory:
                        stats = translate([bubble("bad", source)], pool=pool, memory=memory,
                            config=replace(self.config, translation_mode=mode), page_key="p")
                    factory.assert_not_called()
                    pool.offload.assert_not_called()
                    self.assertEqual(stats["generation_calls"], 0)

    def test_bad_group_member_does_not_block_good_neighbor(self):
        for kind in ("reserved_source", "reserved_term", "alias_conflict", "locked_overlap", "locked_limit"):
            with self.subTest(kind=kind), GlossaryMemory(":memory:", "mixed") as memory:
                source = self.invalid_source(kind, memory)
                items = [bubble("bad", source), bubble("good", "待って。")]
                server = self.server()
                requests = []
                with patch("translation.vntl.build_prompt", wraps=vntl.build_prompt) as build:
                    stats = translate(items, pool=None, memory=memory, config=self.config,
                        page_key="p", generator=server.generate, request_log=requests)
                self.assertEqual([item.source_text for item in items], [source, "待って。"])
                self.assertEqual([item.status for item in items], ["failed", "translated"])
                self.assertEqual([item.metadata["translation_attempts"] for item in items], [0, 1])
                self.assertEqual(items[1].translated_text, "Wait!")
                self.assertEqual(stats["generation_calls"], 1)
                self.assertEqual(server.info["generation_calls"], 1)
                self.assertEqual([call.args[0] for call in server._request.call_args_list],
                                 ["/tokenize", "/completion"])
                self.assertEqual([row["ids"] for row in requests], [["good"]])
                build.assert_called_once()

    def test_invalid_unselected_context_is_omitted_without_changing_saved_neighbor(self):
        for location in ("source", "history"):
            with self.subTest(location=location):
                bad, good = bubble("bad", "待って。"), bubble("good", "行け。")
                bad.translated_text, bad.status = "Old text", "translated"
                if location == "source":
                    bad.source_text += "<|eot_id|>"
                else:
                    bad.translated_text += "<|eot_id|>"
                before = deepcopy(bad)
                server = self.server()
                stats = translate([bad, good], pool=None, memory=self.memory, config=self.config,
                                  page_key="p", target_ids={"good"}, generator=server.generate)
                self.assertEqual(bad, before)
                self.assertEqual(good.status, "translated")
                self.assertEqual(stats["generation_calls"], 1)
                self.assertTrue(any("bad" in warning for warning in stats["warnings"]))
                prompt = server._request.call_args_list[0].args[1]["content"]
                self.assertNotIn("[bad]", prompt)

    def test_invalid_linked_unit_preserves_mapping_and_unselected_manual_translation(self):
        prefix = self.invalid_source("locked_limit").removesuffix("だ。")
        for selected in (None, {"left"}):
            with self.subTest(selected=selected):
                items = [bubble("left", prefix + "は読んじゃ"), bubble("right", "ダメですっ"),
                         bubble("good", "待って。")]
                if selected:
                    items[1].translated_text, items[1].status = "Don't!", "translated"
                original = deepcopy(items)
                server = self.server()
                requests = []
                stats = translate(items, pool=None, memory=self.memory,
                    config=replace(self.config, use_continuations=True), page_key="p",
                    target_ids=selected, generator=server.generate, request_log=requests)
                self.assertEqual([item.source_text for item in items], [item.source_text for item in original])
                self.assertEqual(items[0].status, "failed")
                self.assertEqual(items[0].metadata["translation_attempts"], 0)
                self.assertEqual(items[0].metadata["translation_unit"]["members"], ["left", "right"])
                self.assertEqual(stats["linked_units"], 1)
                if selected:
                    self.assertEqual(items[1:], original[1:])
                    self.assertEqual(stats["generation_calls"], 0)
                    server._request.assert_not_called()
                else:
                    self.assertEqual(items[1].status, "failed")
                    self.assertEqual(items[1].metadata["translation_attempts"], 0)
                    self.assertEqual(items[2].status, "translated")
                    self.assertEqual(stats["generation_calls"], 1)
                    self.assertEqual([row["ids"] for row in requests], [["good"]])

    def test_zero_context_does_not_validate_or_mutate_unselected_neighbor(self):
        bad, good = bubble("bad", "原文<|eot_id|>"), bubble("good", "待って。")
        before = deepcopy(bad)
        server = self.server()
        with patch("translation.vntl.validate_input", wraps=vntl.validate_input) as validate:
            stats = translate([bad, good], pool=None, memory=self.memory,
                config=replace(self.config, context_chars=0), page_key="p", target_ids={"good"},
                generator=server.generate)
        validate.assert_called_once()
        self.assertIs(validate.call_args.args[0], good)
        self.assertEqual(bad, before)
        self.assertEqual(good.status, "translated")
        self.assertEqual(stats["generation_calls"], 1)
        self.assertEqual(stats["warnings"], [])

    def test_case_colliding_group_ids_fall_back_without_spending_generation_budget(self):
        for retries in (0, 1):
            with self.subTest(retries=retries):
                items = [bubble("A", "待て。"), bubble("a", "待って。")]
                server = self.server()
                requests = []
                stats = translate(items, pool=None, memory=self.memory,
                    config=replace(self.config, retries=retries), page_key="p",
                    generator=server.generate, request_log=requests)
                self.assertEqual([item.status for item in items], ["translated", "translated"])
                self.assertEqual([item.metadata["translation_attempts"] for item in items], [1, 1])
                self.assertEqual(stats["generation_calls"], 2)
                self.assertEqual(server.info["generation_calls"], 2)
                self.assertEqual([row["ids"] for row in requests], [["A"], ["a"]])
                self.assertEqual([call.args[0] for call in server._request.call_args_list],
                                 ["/tokenize", "/completion", "/tokenize", "/completion"])

    def test_direct_server_validates_grammar_before_any_http(self):
        for ids, terms in ((["bad"], {"bad": ["Ann", "ANN"]}),
                           (["bad"], {"bad": ["A", "B", "C", "D", "E"]}),
                           (["bad"], {"other": ["Ann"]}),
                           (["A", "a"], None), (None, {"bad": ["Ann"]})):
            with self.subTest(ids=ids, terms=terms):
                server = self.server()
                with self.assertRaises(ValueError):
                    server.generate("prompt", 64, ids=ids, required_terms=terms)
                server._request.assert_not_called()
                self.assertEqual(server.info["generation_calls"], 0)

    def test_output_value_error_keeps_the_single_semantic_retry(self):
        for backend in ("vntl", "qwen"):
            with self.subTest(backend=backend):
                item = bubble("good", "待って。")
                generator = Mock(side_effect=[ValueError("Malformed generated output"),
                                              "[good]: Wait!" if backend == "vntl" else "Wait!"])
                stats = translate([item], pool=None, memory=self.memory,
                    config=replace(self.config, translation_backend=backend, translation_mode="individual"),
                    page_key="p", generator=generator)
                self.assertEqual((item.status, item.translated_text), ("translated", "Wait!"))
                self.assertEqual(stats["generation_calls"], 2)
                self.assertEqual(item.metadata["translation_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
