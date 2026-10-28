from contextlib import ExitStack
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from vision import graphics
from pipeline.batch import file_hash
from tests.editing_fixture import EditingFixture


class EditingFixtureTests(EditingFixture, unittest.TestCase):
    def test_saved_page_needs_no_image_processing(self):
        with ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {"cv2": None}))
            for name in ("prepare_layout", "inpaint", "render"):
                stack.enter_context(patch.object(graphics, name, side_effect=AssertionError(name)))
            for failed in (False, True):
                with self.subTest(left_failed=failed):
                    record = self.make_result(left_failed=failed)
                    session = self.open_session()
                    self.assertEqual(session.cleaned_ids, {"right"} if failed else {"left", "right"})
                    self.assertEqual(record["output_sha256"], file_hash(self.output_path))
                    self.assertEqual(record["clean_image"]["sha256"], file_hash(self.clean_path))
                    original, clean = np.asarray(self.original), np.asarray(session.clean)
                    if failed:
                        np.testing.assert_array_equal(clean[:, :320], original[:, :320])
                    else:
                        self.assertTrue(np.all(clean[65:195, 60:260] == 255))
                    self.assertTrue(np.any(original[65:195, 380:580] != 255))
                    self.assertTrue(np.all(clean[65:195, 380:580] == 255))
                    self.assertTrue(np.any(self.output_pixels()[65:195, 380:580] != 255))
                    self.pool.acquire.assert_not_called()

    def test_prepared_layout_reproduces_saved_pixels(self):
        for inferred in (False, True):
            with self.subTest(inferred_regions=inferred):
                self.make_result(inferred_regions=inferred)
                session = self.open_session()
                before = deepcopy(session.blocks)
                with patch.dict("sys.modules", {"cv2": None}):
                    output = graphics.render(session.clean, session.blocks, self.font_path, "en", 12, 24)
                np.testing.assert_array_equal(output, self.fixture_output)
                self.assertEqual(session.blocks, before)

    def test_reading_edit_is_independent_of_image_processing(self):
        with ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {"cv2": None}))
            for module in ("vision.graphics", "pipeline.editing"):
                for name in ("prepare_layout", "inpaint", "render"):
                    target = f"{module}.{name}"
                    stack.enter_context(patch(target, side_effect=AssertionError(target)))
            self.make_reading_result()
            session = self.open_session()
            text = "Long reading text " * 100
            session.rerender("left", text)
            saved = json.loads(self.record_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["blocks"][0]["translated_text"], text.strip())
            self.assertFalse(saved["exported"])
            self.assertFalse(self.output_path.exists())
            self.assertFalse(self.clean_path.exists())
            self.pool.acquire.assert_not_called()


if __name__ == "__main__":
    unittest.main()
