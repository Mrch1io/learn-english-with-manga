"""Run the offline suite: python -m tools.test"""
import sys
import unittest


ALLOWED_SKIPS = frozenset({
    ("tests.test_graphics.LayoutTests.test_chinese_punctuation_and_paragraphs",
     "A CJK font is not installed; no CJK model support is implied"),
})


def run(suite, *, stream=None):
    if stream is None:
        stream = sys.stderr
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    unexpected = False
    for test, reason in result.skipped:
        allowed = (test.id(), reason) in ALLOWED_SKIPS
        print(f"{'Allowed' if allowed else 'Unexpected'} skip: {test.id()}: {reason}",
              file=stream)
        if not allowed:
            unexpected = True
    executed = result.testsRun - len(result.skipped)
    if executed <= 0:
        print("No tests executed.", file=stream)
    passed = result.wasSuccessful() and executed > 0 and not unexpected
    print(f"Offline suite: {'PASS' if passed else 'FAIL'}; "
          f"run={result.testsRun}, skipped={len(result.skipped)}, "
          f"failures={len(result.failures)}, errors={len(result.errors)}", file=stream)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run(unittest.defaultTestLoader.discover("tests", top_level_dir=".")))
