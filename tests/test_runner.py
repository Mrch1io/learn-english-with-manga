import io
import unittest

from tools import test as runner


def skipped_case(test_id, reason):
    def skip():
        raise unittest.SkipTest(reason)
    skip.__name__ = test_id
    return unittest.FunctionTestCase(skip)


class RunnerTests(unittest.TestCase):
    def run_cases(self, *cases):
        output = io.StringIO()
        code = runner.run(unittest.TestSuite(cases), stream=output)
        return code, output.getvalue()

    def test_passing_suite_succeeds(self):
        code, output = self.run_cases(unittest.FunctionTestCase(lambda: None))
        self.assertEqual(code, 0)
        self.assertIn("Offline suite: PASS; run=1, skipped=0, failures=0, errors=0", output)

    def test_empty_suite_fails(self):
        code, output = self.run_cases()
        self.assertEqual(code, 1)
        self.assertIn("No tests executed.", output)

    def test_failures_and_errors_fail(self):
        for action, summary in ((lambda: self.fail("broken"), "failures=1, errors=0"),
                                (lambda: 1 / 0, "failures=0, errors=1")):
            with self.subTest(summary=summary):
                code, output = self.run_cases(unittest.FunctionTestCase(action))
                self.assertEqual(code, 1)
                self.assertIn(summary, output)

    def test_only_declared_font_skip_is_allowed(self):
        test_id = "tests.test_graphics.LayoutTests.test_chinese_punctuation_and_paragraphs"
        reason = "A CJK font is not installed; no CJK model support is implied"
        for identity, message, expected in ((test_id, reason, 0),
                                            ("another_test", reason, 1),
                                            (test_id, "OpenCV is missing", 1)):
            with self.subTest(identity=identity, message=message):
                code, output = self.run_cases(unittest.FunctionTestCase(lambda: None),
                                              skipped_case(identity, message))
                self.assertEqual(code, expected)
                self.assertIn(f"{'Allowed' if expected == 0 else 'Unexpected'} skip: "
                              f"{identity}: {message}", output)

    def test_all_skipped_suite_fails_even_with_allowed_reason(self):
        code, output = self.run_cases(skipped_case(
            "tests.test_graphics.LayoutTests.test_chinese_punctuation_and_paragraphs",
            "A CJK font is not installed; no CJK model support is implied"))
        self.assertEqual(code, 1)
        self.assertIn("No tests executed.", output)

    def test_class_setup_error_fails(self):
        class BrokenFixture(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise RuntimeError("fixture failed")

            def runTest(self):
                pass

        code, output = self.run_cases(BrokenFixture())
        self.assertEqual(code, 1)
        self.assertIn("fixture failed", output)
        self.assertIn("errors=1", output)


if __name__ == "__main__":
    unittest.main()
