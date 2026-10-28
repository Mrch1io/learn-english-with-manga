import unittest
from unittest.mock import patch

from PIL import Image

from core.runtime import Config, TextBlock
from vision.detection import add_text_regions, detect_page_text, order_blocks


class FrontendThresholdTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (800, 600), "white")
        self.config = Config()

    def test_lower_threshold_recovers_vertical_candidates_at_the_boundary(self):
        proposals = [{"box": [x, 20, x + 20, 100], "score": score}
                     for x, score in zip((20, 60, 100, 140), (.19, .2, .25, .3))]
        with patch("vision.characters.load_magi", return_value=object()), \
                patch("vision.detection._text_proposals", return_value=proposals) as infer, \
                patch("vision.detection.locate_wide_line", return_value=None):
            regions = detect_page_text(self.image, object(), self.config, min_text_score=.2)
        self.assertEqual(regions, proposals[1:])
        infer.assert_called_once()
        self.assertEqual(infer.call_args.kwargs["threshold"], .1)

    def test_default_is_still_point_three(self):
        proposals = [{"box": [20, 20, 40, 100], "score": .25},
                     {"box": [60, 20, 80, 100], "score": .3}]
        with patch("vision.characters.load_magi", return_value=object()), \
                patch("vision.detection._text_proposals", return_value=proposals), \
                patch("vision.detection.locate_wide_line", return_value=None):
            default = detect_page_text(self.image, object(), self.config)
            explicit = detect_page_text(self.image, object(), self.config, min_text_score=.3)
        self.assertEqual(default, explicit)
        self.assertEqual(default, [proposals[1]])

    def test_low_horizontal_recovery_still_checks_at_most_four_candidates(self):
        high = {"box": [700, 10, 720, 90], "score": .3}
        lows = [{"box": [20, y, 620, y + 30], "score": score}
                for y, score in zip((120, 200, 280, 360, 440), (.19, .18, .17, .16, .15))]
        def locate(image, box):
            return {"box": box, "ocr_segments": [box]} if box[0] == 20 else None
        outputs = []
        for threshold in (.3, .2):
            with self.subTest(threshold=threshold), \
                    patch("vision.characters.load_magi", return_value=object()), \
                    patch("vision.detection._text_proposals", return_value=[high] + lows), \
                    patch("vision.detection.locate_wide_line", side_effect=locate) as localize:
                regions = detect_page_text(self.image, object(), self.config,
                                           min_text_score=threshold)
            self.assertEqual(localize.call_count, 5)
            self.assertEqual([region["box"] for region in regions],
                             [high["box"]] + [region["box"] for region in lows[:4]])
            self.assertTrue(all(region["detection_review"] for region in regions[1:]))
            outputs.append(regions)
        self.assertEqual(outputs[0], outputs[1])

    def test_adding_nearby_text_preserves_existing_ids_and_manual_text(self):
        bubble = TextBlock((300, 20, 390, 120), block_id="manual-bubble",
                           source_text="原文", translated_text="Reviewed text",
                           metadata={"manual_translation": True})
        proposals = [{"box": [180, 20, 200, 100], "score": .3},
                     {"box": [205, 20, 225, 100], "score": .2}]
        with patch("vision.characters.load_magi", return_value=object()), \
                patch("vision.detection._text_proposals", return_value=proposals), \
                patch("vision.detection.locate_wide_line", return_value=None):
            old_regions = detect_page_text(self.image, object(), self.config)
            new_regions = detect_page_text(self.image, object(), self.config, min_text_score=.2)
        old = order_blocks(add_text_regions([bubble], old_regions, self.image.size))
        new = order_blocks(add_text_regions([bubble], new_regions, self.image.size))
        self.assertEqual([block.box for block in new],
                         [bubble.box, (205, 20, 225, 100), (180, 20, 200, 100)])
        self.assertEqual([block.reading_order for block in new], [0, 1, 2])
        self.assertEqual(new[2].block_id, old[1].block_id)
        self.assertIs(new[0], bubble)
        self.assertEqual((bubble.block_id, bubble.source_text, bubble.translated_text),
                         ("manual-bubble", "原文", "Reviewed text"))
        self.assertTrue(bubble.metadata["manual_translation"])

    def test_invalid_threshold_fails_before_loading_model(self):
        for score in (-1, 0, .099, 1.01, float("inf"), float("nan")):
            with self.subTest(score=score), patch("vision.characters.load_magi") as load:
                with self.assertRaisesRegex(ValueError, "score"):
                    detect_page_text(self.image, object(), self.config, min_text_score=score)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
