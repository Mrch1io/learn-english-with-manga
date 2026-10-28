"""CPU tensor fixtures: compatibility must never leave random active weights."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from vision.characters import load_magi


class MagiLoadingTests(unittest.TestCase):
    OLD = "crop_embedding_model.encoder.layer.0.attention.attention.query.weight"
    NEW = "crop_embedding_model.layers.0.attention.q_proj.weight"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / "pytorch_model.bin").write_bytes(b"tensor loader mocked")
        self.model = torch.nn.Module()
        self.model.crop_embedding_model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.attention = torch.nn.Module()
        layer.attention.q_proj = torch.nn.Linear(2, 2, bias=False)
        self.model.crop_embedding_model.layers = torch.nn.ModuleList([layer])
        self.config = SimpleNamespace(detection_model_config=SimpleNamespace(to_dict=lambda: {}),
                                      crop_embedding_model_config=SimpleNamespace(to_dict=lambda: {}))
        self.backbone = SimpleNamespace(use_pretrained_backbone=True)
        self.transformers = SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=self.config)),
            AutoModel=SimpleNamespace(from_config=Mock(return_value=self.model)),
            ConditionalDetrConfig=SimpleNamespace(from_dict=lambda _: SimpleNamespace(backbone_config=self.backbone, to_dict=lambda: {})),
            ViTMAEConfig=SimpleNamespace(from_dict=lambda _: SimpleNamespace(to_dict=lambda: {})))
        self.pool = SimpleNamespace(acquire=lambda key, load, move: move(load(), "cpu"))

    def load(self, state):
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"transformers": self.transformers}))
            checkpoint = stack.enter_context(patch("torch.load", return_value=state))
            result = load_magi(self.pool, self.path)
            self.assertTrue(checkpoint.call_args.kwargs["weights_only"])
            self.assertTrue(checkpoint.call_args.kwargs["mmap"])
            return result

    def test_legacy_key_loads_exact_trained_tensor_and_only_ocr_is_ignored(self):
        weights = torch.tensor([[1., 2.], [3., 4.]])
        model = self.load({self.OLD: weights, "ocr_model.unused.weight": torch.zeros(1)})
        self.assertTrue(torch.equal(model.state_dict()[self.NEW], weights))
        self.assertEqual(model._manga_loading_info["ignored_ocr_keys"], 1)
        self.assertEqual(model._manga_loading_info["renamed_keys"], 1)
        self.assertTrue(self.config.disable_ocr)
        self.assertFalse(self.backbone.use_pretrained_backbone)
        self.assertTrue(self.transformers.AutoConfig.from_pretrained.call_args.kwargs["local_files_only"])

    def test_missing_mismatched_or_unknown_active_weights_fail(self):
        examples = [{}, {self.OLD: torch.zeros(3, 2)},
                    {self.OLD: torch.zeros(2, 2), "unrecognised.weight": torch.zeros(1)}]
        for state in examples:
            with self.subTest(keys=list(state)), self.assertRaisesRegex(RuntimeError, "did not load completely"):
                self.load(state)

    def test_ambiguous_mapping_cannot_silently_replace_a_weight(self):
        with self.assertRaisesRegex(RuntimeError, "mapping collision"):
            self.load({self.OLD: torch.ones(2, 2), self.NEW: torch.zeros(2, 2)})

    def test_missing_local_checkpoint_is_an_explicit_error(self):
        (self.path / "pytorch_model.bin").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Local Magi checkpoint missing"):
            load_magi(self.pool, self.path)


if __name__ == "__main__":
    unittest.main()
