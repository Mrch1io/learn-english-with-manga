"""Exercise the training CLI without importing PyTorch or loading real models."""

import builtins
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools import train


class TrainToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = self.root / "dataset" / "data.yaml"
        self.data.parent.mkdir()
        self.data.write_text("path: .\ntrain: images\nval: images\nnames: [bubble]\n", encoding="utf-8")
        self.yolo = mock.Mock()
        self.module = types.ModuleType("ultralytics")
        self.module.YOLO = self.yolo
        self.root_patch = mock.patch.object(train, "PROJECT_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def run_main(self, args):
        with mock.patch.dict(sys.modules, {"ultralytics": self.module}):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return train.main(args)

    def reject_without_import(self, args, error):
        original_import = builtins.__import__
        imported = []

        def tracked_import(name, *args, **kwargs):
            if name == "ultralytics":
                imported.append(name)
                raise AssertionError("Ultralytics must not be imported")
            return original_import(name, *args, **kwargs)

        stderr = io.StringIO()
        with mock.patch("builtins.__import__", side_effect=tracked_import):
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                train.main(args)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(error, stderr.getvalue())
        self.assertEqual(imported, [])

    def test_defaults_and_model_cache_are_forwarded(self):
        self.assertEqual(self.run_main(["--data", "dataset/data.yaml"]), 0)
        self.yolo.assert_called_once_with(str(self.root / "output/models/yolov8m.pt"))
        self.yolo.return_value.train.assert_called_once_with(
            data=str(self.data.resolve()), epochs=30, imgsz=640, batch=8, device="0",
            workers=2, project=str(self.root / "output/training"), name="bubble-detector",
        )
        self.assertTrue((self.root / "output/models").is_dir())

    def test_existing_default_weights_are_reused(self):
        weights = self.root / "yolov8m.pt"
        weights.touch()
        self.assertEqual(self.run_main(["--data", str(self.data)]), 0)
        self.yolo.assert_called_once_with(str(weights))
        self.assertFalse((self.root / "output/models").exists())

    def test_custom_checkpoint_and_all_options_are_forwarded(self):
        weights = self.root / "best.pt"
        weights.touch()
        self.assertEqual(self.run_main([
            "--data", "dataset/data.yaml", "--weights", "best.pt", "--epochs", "100",
            "--imgsz", "512", "--batch", "4", "--device", "cpu", "--workers", "0",
            "--project", "output/custom", "--name", "fine-tuned",
        ]), 0)
        self.yolo.assert_called_once_with(str(weights))
        self.yolo.return_value.train.assert_called_once_with(
            data=str(self.data.resolve()), epochs=100, imgsz=512, batch=4, device="cpu",
            workers=0, project=str(self.root / "output/custom"), name="fine-tuned",
        )

    def test_missing_data_is_rejected_before_import(self):
        self.reject_without_import([], "--data")
        self.reject_without_import(["--data", "absent.yaml"], "dataset YAML not found")

    def test_missing_custom_checkpoint_is_not_downloaded(self):
        self.reject_without_import(
            ["--data", str(self.data), "--weights", "previous/best.pt"], "weights file not found",
        )

    def test_invalid_values_are_rejected_before_import(self):
        for flag, value in [("--epochs", "0"), ("--imgsz", "-1"), ("--batch", "0"),
                            ("--workers", "-1"), ("--epochs", "no"), ("--device", ""),
                            ("--name", "../outside")]:
            with self.subTest(flag=flag, value=value):
                self.reject_without_import(["--data", str(self.data), flag, value], flag)
        self.reject_without_import(
            ["--data", str(self.data), "--project", str(self.data)], "output path is not a directory",
        )

    def test_model_creation_and_training_failures_return_one(self):
        for stage in (self.yolo, self.yolo.return_value.train):
            with self.subTest(stage=stage):
                stage.side_effect = RuntimeError("fake training failure")
                self.assertEqual(self.run_main(["--data", str(self.data)]), 1)
                stage.side_effect = None

    def test_interruption_returns_130(self):
        self.yolo.return_value.train.side_effect = KeyboardInterrupt
        self.assertEqual(self.run_main(["--data", str(self.data)]), 130)

    def subprocess_cli(self, args, fake_module):
        tools_dir = self.root / "tools"
        tools_dir.mkdir(exist_ok=True)
        script = tools_dir / "train.py"
        shutil.copyfile(train.__file__, script)
        (tools_dir / "ultralytics.py").write_text(fake_module, encoding="utf-8")
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir(exist_ok=True)
        return subprocess.run(
            [sys.executable, str(script), *args], cwd=elsewhere, capture_output=True,
            text=True, timeout=15, check=False,
        )

    def test_help_is_lightweight_in_a_fresh_process(self):
        result = self.subprocess_cli(["--help"], "raise AssertionError('must not import Ultralytics')\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--weights", result.stdout)
        self.assertFalse((self.root / "output").exists())

    def test_cli_from_different_cwd_uses_script_repository(self):
        result = self.subprocess_cli(["--data", "dataset/data.yaml"], (
            "import json\n"
            "class YOLO:\n"
            "    def __init__(self, weights):\n"
            "        self.weights = weights\n"
            "    def train(self, **kwargs):\n"
            "        print(json.dumps({'weights': self.weights, **kwargs}))\n"
        ))
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = json.loads(result.stdout.splitlines()[0])
        self.assertEqual(arguments["data"], str(self.data.resolve()))
        self.assertEqual(arguments["weights"], str(self.root / "output/models/yolov8m.pt"))
        self.assertEqual(arguments["project"], str(self.root / "output/training"))
        self.assertFalse((self.root / "elsewhere/output").exists())

    def test_cli_failure_has_nonzero_exit_code(self):
        result = self.subprocess_cli(["--data", "dataset/data.yaml"], (
            "class YOLO:\n"
            "    def __init__(self, weights):\n"
            "        raise RuntimeError('checkpoint cannot load')\n"
        ))
        self.assertEqual(result.returncode, 1)
        self.assertIn("checkpoint cannot load", result.stderr)


if __name__ == "__main__":
    unittest.main()
