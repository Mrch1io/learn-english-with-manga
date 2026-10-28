"""Lexical decoding constraints: acceptance tests without GPU/model loading."""
import json
import re
import unittest
from unittest.mock import Mock

from translation.vntl import VntlServer, label_grammar


def grammar_pattern(grammar):
    """Expand this acyclic GBNF subset to regex to test its accepted language.

    Literal, class, grouping and repetition semantics are shared by GBNF and
    regex here. Testing completed strings catches boundary/scoping bugs that
    merely checking for canonical names in a prompt would miss.
    """
    rules = dict(line.split(" ::= ", 1) for line in grammar.splitlines())
    token = re.compile(r'"(?:\\.|[^"\\])*"|\[(?:\\.|[^\]\\])*\]|[A-Za-z][A-Za-z0-9-]*|[()|*+?]')
    cache = {}

    def expand(name):
        if name not in cache:
            pieces = []
            for match in token.finditer(rules[name]):
                part = match[0]
                if part.startswith('"'):
                    pieces.append(re.escape(json.loads(part)))
                elif part[0].isalpha():
                    pieces.append(expand(part))
                else:
                    pieces.append(part)
            cache[name] = "(?:" + "".join(pieces) + ")"
        return cache[name]

    return re.compile(expand("root"))


class LockedGrammarTests(unittest.TestCase):
    def test_full_canonical_name_is_required_in_its_own_bubble(self):
        pattern = grammar_pattern(label_grammar(["one", "two"], {"one": ["Misaki Takanashi"]}))
        self.assertIsNotNone(pattern.fullmatch("[one]: Where is Misaki Takanashi?\n[two]: She's here."))
        self.assertIsNone(pattern.fullmatch("[one]: Where is Misaki?\n[two]: She's here."))
        self.assertIsNone(pattern.fullmatch("[one]: Where is she?\n[two]: Misaki Takanashi is here."))
        # Context-only names must not be required in an unrelated line.
        self.assertIsNotNone(pattern.fullmatch("[one]: Misaki Takanashi!\n[two]: Wait!"))

    def test_multiple_terms_accept_both_english_orders_and_punctuation(self):
        pattern = grammar_pattern(label_grammar(["one"], {"one": ["Misaki Takanashi", "Silver Gate"]}))
        for text in ("Misaki Takanashi is at the Silver Gate.",
                     "At the Silver Gate, I saw Misaki Takanashi.",
                     "Misaki Takanashi:Silver Gate", "Silver Gate (Misaki Takanashi)"):
            with self.subTest(text=text):
                self.assertIsNotNone(pattern.fullmatch("[one]: " + text))
        for text in ("Misaki is at the Silver Gate.", "Misaki Takanashi is here.",
                     "Silver GateMisaki Takanashi", "Misaki TakanashiSilver Gate"):
            with self.subTest(text=text):
                self.assertIsNone(pattern.fullmatch("[one]: " + text))

    def test_term_matching_respects_latin_word_boundaries(self):
        pattern = grammar_pattern(label_grammar(["one"], {"one": ["Ann"]}))
        for text in ("Ann", "Hello, Ann!", "Ann's here.", "(Ann)", "Ann-Marie"):
            self.assertIsNotNone(pattern.fullmatch("[one]: " + text), text)
        for text in ("Hannah", "Annie", "_Ann", "Ann2", "XAnnY", "Ann_"):
            self.assertIsNone(pattern.fullmatch("[one]: " + text), text)

    def test_duplicate_and_contained_names_do_not_force_repetition(self):
        pattern = grammar_pattern(label_grammar(["one"], {"one": ["Ann", "Ann Marie", "Ann Marie"]}))
        self.assertIsNotNone(pattern.fullmatch("[one]: Ann Marie arrived."))
        self.assertIsNone(pattern.fullmatch("[one]: Ann arrived."))

    def test_four_terms_have_bounded_grammar_and_no_fixed_name_order(self):
        terms = ["Alice", "Betty", "Clara", "Diana"]
        grammar = label_grammar(["one"], {"one": terms})
        self.assertLess(len(grammar), 8000)
        self.assertLessEqual(sum(line.startswith("locked-0-") for line in grammar.splitlines()), 32)
        pattern = grammar_pattern(grammar)
        self.assertIsNotNone(pattern.fullmatch("[one]: Diana, Clara, Betty and Alice are here."))
        self.assertIsNone(pattern.fullmatch("[one]: Diana, Clara and Betty are here."))

    def test_ambiguous_oversized_or_malformed_targets_fail_explicitly(self):
        cases = [(["Ann", "ANN"], "capitalization"),
                 (["Mary Jane", "Jane Smith"], "Overlapping"),
                 (["Alice", "Betty", "Clara", "Diana", "Emma"], "At most 4"),
                 (["A" * 257], "256"), (["Ann\n[next]: bad"], "single-line"),
                 ([""], "nonempty"), ("Ann", "list")]
        for terms, message in cases:
            with self.subTest(terms=terms), self.assertRaisesRegex(ValueError, message):
                label_grammar(["one"], {"one": terms})
        with self.assertRaisesRegex(ValueError, "unrequested"):
            label_grammar(["one"], {"two": ["Ann"]})

    def test_punctuation_in_locked_target_is_literal_not_grammar_code(self):
        term = '[Boss] "Silver Gate"'
        pattern = grammar_pattern(label_grammar(["one"], {"one": [term]}))
        self.assertIsNotNone(pattern.fullmatch("[one]: Hello, " + term + "!"))
        self.assertIsNone(pattern.fullmatch("[one]: Hello, Boss Silver Gate!"))

    def test_generate_passes_constraints_and_does_not_rewrite_response(self):
        server = object.__new__(VntlServer)
        server.context_size = 512
        server.info = {"generation_calls": 0, "generation_seconds": 0}
        content = "[one]: Misaki Takanashi is here."
        server._request = Mock(side_effect=[{"tokens": [1, 2]}, {"content": content}])
        terms = {"one": ["Misaki Takanashi"]}
        self.assertEqual(server.generate("prompt", 64, ids=["one"], required_terms=terms), content)
        body = server._request.call_args_list[-1].args[1]
        self.assertEqual(body["grammar"], label_grammar(["one"], terms))
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["repeat_penalty"], 1)


if __name__ == "__main__":
    unittest.main()
