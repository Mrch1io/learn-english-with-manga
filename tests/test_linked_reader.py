"""Current linked-sentence display; no Tk window, model, HTTP, or playback.

Run: .venv\\Scripts\\python.exe -m unittest tests.test_linked_reader -v
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from reader import ReaderWindow
from reader.records import linked_translation as _linked_translation


def linked_blocks():
    unit = {"kind": "fragment", "members": ["right", "left"],
            "sources": ["はな、", "せ..."], "translated_text": "Stale stored sentence.",
            "parts": {"right": "Stale stored", "left": "sentence."}}
    return {
        "right": {"block_id": "right", "source_text": "はな、", "translated_text": "Let",
                  "status": "translated", "metadata": {"translation_unit": deepcopy(unit)}},
        "left": {"block_id": "left", "source_text": "せ...", "translated_text": "go...",
                 "status": "translated", "metadata": {"translation_unit": deepcopy(unit)}},
    }


class LinkedReaderTests(unittest.TestCase):
    def test_uses_current_parts_in_member_order_without_mutating_records(self):
        blocks = linked_blocks()
        # The dictionary/page order is not the linked sentence's order.
        blocks = {"left": blocks["left"], "right": blocks["right"]}
        before = deepcopy(blocks)
        for selected in blocks:
            self.assertEqual(_linked_translation(selected, blocks), "Let go...")
        self.assertEqual(blocks, before)

    def test_manual_english_edit_updates_sentence_for_either_selected_member(self):
        blocks = linked_blocks()
        blocks["left"]["translated_text"] = "go of me..."
        for selected in blocks:
            self.assertEqual(_linked_translation(selected, blocks), "Let go of me...")
        self.assertEqual(blocks["right"]["metadata"]["translation_unit"]["translated_text"],
                         "Stale stored sentence.")

    def test_changed_japanese_invalidates_old_link_for_both_members(self):
        blocks = linked_blocks()
        blocks["left"]["source_text"] = "放せ。"
        for selected in blocks:
            self.assertEqual(_linked_translation(selected, blocks), "")

    def test_missing_failed_or_blank_member_hides_incomplete_sentence(self):
        changes = [{"status": "failed"}, {"status": "pending"}, {"status": "overflow"},
                   {"status": None}, {"translated_text": " \n "}, {"translated_text": None},
                   {"block_id": "different"}]
        for change in changes:
            with self.subTest(change=change):
                blocks = linked_blocks()
                blocks["left"].update(change)
                self.assertEqual(_linked_translation("right", blocks), "")
        blocks = linked_blocks()
        del blocks["left"]
        self.assertEqual(_linked_translation("right", blocks), "")

    def test_incomplete_legacy_metadata_never_displays_unverifiable_saved_sentence(self):
        blocks = linked_blocks()
        blocks["right"]["metadata"]["translation_unit"] = {"translated_text": "Old complete sentence."}
        self.assertEqual(_linked_translation("right", blocks), "")

    def test_invalid_member_identifiers_and_source_shapes_are_rejected(self):
        for changes in (
                {"members": ["right", "right"]}, {"members": ["right", ""]},
                {"members": ["right", " "]}, {"members": ["right", {}]},
                {"members": "right,left"}, {"members": ["left"]},
                {"members": ["left", "other"]}, {"sources": ["はな、"]},
                {"sources": "はな、せ..."}, {"sources": ["はな、", None]}):
            with self.subTest(changes=changes):
                blocks = linked_blocks()
                blocks["right"]["metadata"]["translation_unit"].update(changes)
                self.assertEqual(_linked_translation("right", blocks), "")

    def test_unlinked_and_missing_selection_have_no_linked_sentence(self):
        blocks = linked_blocks()
        for metadata in ({}, None, {"translation_unit": None}, {"translation_unit": []}):
            blocks["right"]["metadata"] = metadata
            self.assertEqual(_linked_translation("right", blocks), "")
        self.assertEqual(_linked_translation("absent", blocks), "")

    def test_show_bubble_uses_current_sentence_and_clears_invalid_link(self):
        blocks = linked_blocks()
        window = SimpleNamespace(block_id="right", blocks=blocks, bubble_context=Mock(),
            bubble_notice=Mock(), japanese=Mock(), english=Mock(), bindings={},
            _set_roles=Mock(), _draw_boxes=Mock())
        ReaderWindow._show_bubble(window)
        window.bubble_context.set.assert_called_with("关联整句：Let go...")
        blocks["left"]["translated_text"] = "go of me..."
        ReaderWindow._show_bubble(window)
        window.bubble_context.set.assert_called_with("关联整句：Let go of me...")
        blocks["left"]["source_text"] = "放せ。"
        ReaderWindow._show_bubble(window)
        window.bubble_context.set.assert_called_with("")


if __name__ == "__main__":
    unittest.main()
