"""GPU lifecycle contract tests use a fake CUDA API; no weights are loaded."""

import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.runtime import Config, ModelPool, VNTL_STAGE_RESERVE_BYTES


class ModelPoolTests(unittest.TestCase):
    def setUp(self):
        self.cuda = Mock()
        self.cuda.is_available.return_value = True
        self.cuda.mem_get_info.return_value = (4*1024**3, 8*1024**3)
        self.torch = types.SimpleNamespace(cuda=self.cuda)
        self.require_patch = patch("core.runtime.require", return_value=self.torch)
        self.require_patch.start()
        self.gc_patch = patch("core.runtime.gc.collect")
        self.gc_patch.start()
        self.moves = []

    def tearDown(self):
        self.gc_patch.stop()
        self.require_patch.stop()

    def move(self, value, device):
        self.moves.append((value, device))

    def test_weights_load_once_and_repeated_acquisition_reuses_instance(self):
        pool = ModelPool("cuda:0", "staged")
        model = object()
        loader = Mock(return_value=model)
        self.assertIs(pool.acquire("ocr", loader, self.move), model)
        self.assertIs(pool.acquire("ocr", loader, self.move), model)
        loader.assert_called_once_with()
        self.assertEqual(pool.loads, {"ocr": 1})
        self.assertEqual(self.moves, [(model, "cuda:0")])
        self.assertGreaterEqual(pool.load_seconds["ocr"], 0)

    def test_staged_offloads_previous_model_but_keeps_cached_weights(self):
        pool = ModelPool("cuda:0", "staged")
        first, second = object(), object()
        first_loader, second_loader = Mock(return_value=first), Mock(return_value=second)
        pool.acquire("ocr", first_loader, self.move)
        pool.acquire("translate", second_loader, self.move)
        self.assertEqual(pool.active, {"translate"})
        pool.acquire("ocr", first_loader, self.move)
        self.assertEqual(pool.active, {"ocr"})
        self.assertEqual(self.moves, [(first, "cuda:0"), (first, "cpu"), (second, "cuda:0"),
                                      (second, "cpu"), (first, "cuda:0")])
        first_loader.assert_called_once_with()
        second_loader.assert_called_once_with()

    def test_resident_oom_falls_back_once_to_staged(self):
        pool = ModelPool("cuda:0", "resident")
        first, second = object(), object()
        pool.acquire("ocr", lambda: first, self.move)
        attempts = 0

        def move_second(value, device):
            nonlocal attempts
            self.move(value, device)
            if device.startswith("cuda"):
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("CUDA out of memory")

        loader = Mock(return_value=second)
        self.assertIs(pool.acquire("translate", loader, move_second), second)
        self.assertEqual(attempts, 2)
        self.assertEqual(pool.policy, "staged")
        self.assertEqual(pool.active, {"translate"})
        self.assertIn((first, "cpu"), self.moves)
        self.assertIn((second, "cpu"), self.moves)
        loader.assert_called_once_with()

    def test_persistent_oom_stops_after_one_retry(self):
        pool = ModelPool("cuda:0", "resident")
        attempts = 0

        def cannot_fit(_value, device):
            nonlocal attempts
            if device.startswith("cuda"):
                attempts += 1
                raise RuntimeError("CUDA out of memory")

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            pool.acquire("translate", object, cannot_fit)
        self.assertEqual(attempts, 2)
        self.assertEqual(pool.policy, "staged")
        self.assertEqual(pool.active, set())

    def test_non_memory_failure_is_not_retried(self):
        pool = ModelPool("cuda:0", "resident")
        mover = Mock(side_effect=ValueError("invalid tensor"))
        with self.assertRaisesRegex(ValueError, "invalid tensor"):
            pool.acquire("translate", object, mover)
        mover.assert_called_once()

    def test_close_releases_cache_without_allocating_a_cpu_copy(self):
        pool = ModelPool("cuda:0", "resident")
        model = pool.acquire("ocr", object, self.move)
        pool.close()
        self.assertEqual(pool.models, {})
        self.assertEqual(pool.active, set())
        self.assertNotIn((model, "cpu"), self.moves)

    def test_llama_server_survives_stage_changes_and_closes_once_with_pool(self):
        pool = ModelPool("cuda:0")
        server = Mock()
        pool.vntl = server
        pool.acquire("ocr", object, self.move)
        pool.acquire("detector", object, self.move)
        server.close.assert_not_called()
        self.assertNotIn("vntl", pool.models)
        pool.close()
        pool.close()
        server.close.assert_called_once_with()
        self.assertIsNone(pool.vntl)

    def test_ocr_releases_owned_vntl_before_loader_when_stage_headroom_is_low(self):
        pool = ModelPool("cuda:0")
        server = Mock(info={"model_file": "sample.gguf", "pid": 2345})
        pool.vntl = server
        self.cuda.mem_get_info.side_effect = [(1100*1024**2, 8*1024**3), (5*1024**3, 8*1024**3)]

        def load():
            server.close.assert_called_once_with()
            self.assertIsNone(pool.vntl)
            return object()

        with self.assertWarnsRegex(RuntimeWarning, "reload"):
            pool.acquire("ocr", load, self.move)
        self.assertEqual(pool.active, {"ocr"})
        self.assertEqual(self.moves[-1][1], "cuda:0")
        entry = pool.evictions[0]
        self.assertEqual(entry["reason"], "insufficient_cuda_stage_headroom")
        self.assertEqual(entry["stage"], "ocr")
        self.assertEqual(entry["server_pid"], 2345)
        self.assertEqual(entry["free_before_bytes"], 1100*1024**2)
        self.assertEqual(entry["free_after_bytes"], 5*1024**3)
        self.assertEqual(entry["stage_reserve_bytes"], VNTL_STAGE_RESERVE_BYTES["ocr"])
        self.assertGreaterEqual(entry["close_seconds"], 0)
        pool.close()
        server.close.assert_called_once_with()

    def test_cached_ocr_and_its_load_count_survive_vntl_release(self):
        pool = ModelPool("cuda:0")
        loader = Mock(return_value=object())
        original = pool.acquire("ocr", loader, self.move)
        pool.offload()
        pool.vntl = Mock(info={"pid": 99})
        self.cuda.mem_get_info.side_effect = [(1024**3, 8*1024**3), (5*1024**3, 8*1024**3)]
        with self.assertWarns(RuntimeWarning):
            self.assertIs(pool.acquire("ocr", loader, self.move), original)
        loader.assert_called_once_with()
        self.assertEqual(pool.loads, {"ocr": 1})

    def test_insufficient_memory_after_owned_server_close_stops_before_loading(self):
        pool = ModelPool("cuda:0")
        pool.vntl = Mock(info={})
        self.cuda.mem_get_info.return_value = (200*1024**2, 8*1024**3)
        loader = Mock()
        with self.assertWarns(RuntimeWarning), self.assertRaisesRegex(RuntimeError, "CPU inference is disabled"):
            pool.acquire("ocr", loader, self.move)
        loader.assert_not_called()
        self.assertIsNone(pool.vntl)
        self.assertEqual(pool.models, {})
        self.assertEqual(len(pool.evictions), 1)

    def test_stage_reserve_applies_to_magi_named_cache_and_resident_policy(self):
        pool = ModelPool("cuda:0", "resident")
        server = Mock(info={})
        pool.vntl = server
        self.cuda.mem_get_info.side_effect = [(1536*1024**2, 8*1024**3), (5*1024**3, 8*1024**3)]
        with self.assertWarns(RuntimeWarning):
            pool.acquire("magi:local-weights:revision", object, self.move)
        self.assertEqual(pool.evictions[0]["stage_reserve_bytes"], VNTL_STAGE_RESERVE_BYTES["magi"])
        server.close.assert_called_once_with()

    def test_translator_defaults_and_context_bounds(self):
        config = Config()
        self.assertEqual(config.translation_backend, "vntl")
        self.assertEqual(config.translation_mode, "auto")
        self.assertEqual(config.llama_context_size, 2048)
        self.assertTrue(config.gguf_path.endswith("vntl-llama3-8b-v2-hf.Q4_K_M.gguf"))
        self.assertEqual(config.llama_server_path, str(Path(__file__).resolve().parents[1]
            / "runtime" / "llama.cpp" / "b11706" / "llama-server.exe"))
        for backend, context in (("cloud", 2048), ("vntl", 511), ("vntl", 8193)):
            with self.subTest(backend=backend, context=context), self.assertRaises(ValueError):
                Config(translation_backend=backend, llama_context_size=context).validate()
        for mode in ("auto", "individual", "grouped"):
            Config(translation_mode=mode).validate()
        with self.assertRaisesRegex(ValueError, "translation_mode"):
            Config(translation_mode="unknown").validate()

    def test_low_windows_commit_evicts_large_translation_without_cpu_copy(self):
        tensor = types.SimpleNamespace(numel=lambda: 3_000_000_000, element_size=lambda: 2)
        weights = types.SimpleNamespace(parameters=lambda: iter([tensor]), buffers=lambda: iter([]))
        loaded = (object(), weights)
        loader = Mock(return_value=loaded)
        pool = ModelPool("cuda:0", "staged")
        pool.acquire("translation:test", loader, self.move)
        with patch("core.runtime.available_commit_bytes", return_value=3_000_000_000), self.assertWarnsRegex(RuntimeWarning, "reload"):
            pool.acquire("detector", object, self.move)
        self.assertNotIn("translation:test", pool.models)
        self.assertNotIn((loaded, "cpu"), self.moves)
        self.assertEqual(pool.evictions[0]["model_bytes"], 6_000_000_000)
        pool.acquire("translation:test", loader, self.move)
        self.assertEqual(pool.loads["translation:test"], 2)

    def test_sufficient_windows_commit_keeps_translation_weights_reusable(self):
        tensor = types.SimpleNamespace(numel=lambda: 3_000_000_000, element_size=lambda: 2)
        loaded = (object(), types.SimpleNamespace(parameters=lambda: iter([tensor]), buffers=lambda: iter([])))
        pool = ModelPool("cuda:0", "staged")
        pool.acquire("translation:test", lambda: loaded, self.move)
        with patch("core.runtime.available_commit_bytes", return_value=12_000_000_000):
            pool.acquire("detector", object, self.move)
        self.assertIn("translation:test", pool.models)
        self.assertIn((loaded, "cpu"), self.moves)
        self.assertEqual(pool.evictions, [])

    def test_no_cuda_is_an_error_and_never_silently_uses_cpu(self):
        self.cuda.is_available.return_value = False
        for requested in ("cuda:0", "auto"):
            with self.subTest(device=requested):
                with self.assertRaises((RuntimeError, ValueError)):
                    _ = ModelPool(requested).device

    def test_explicit_cpu_is_rejected_in_config_and_pool(self):
        with self.assertRaises((RuntimeError, ValueError)):
            Config(device="cpu").validate()
        with self.assertRaises((RuntimeError, ValueError)):
            _ = ModelPool("cpu").device

    def test_failed_response_retry_limit_is_at_most_one(self):
        Config(retries=1).validate()
        with self.assertRaisesRegex(ValueError, "retries 0–1"):
            Config(retries=2).validate()


if __name__ == "__main__":
    unittest.main()
