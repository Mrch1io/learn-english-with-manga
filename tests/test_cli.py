"""Reject incomplete editing commands before opening a GUI or loading models."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main
from core.memory import GlossaryMemory


class TranslationInputTests(unittest.TestCase):
    def test_missing_input_does_not_start_batch_or_load_models(self):
        self.assertEqual(main.parser().parse_args([]).input, [])
        for args in ([], ["--offline"], ["--translator", "qwen"]):
            with self.subTest(args=args), patch("pipeline.batch.run_batch") as batch, \
                    patch("core.runtime.require") as dependency, \
                    contextlib.redirect_stderr(io.StringIO()) as stderr, \
                    self.assertRaises(SystemExit) as error:
                main.main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertIn("at least one image or folder", stderr.getvalue())
            batch.assert_not_called()
            dependency.assert_not_called()

    def test_explicit_images_and_folders_share_one_batch_call(self):
        for inputs in (["page.png"], ["comic-folder"], ["first.png", "second.png", "comic-folder"]):
            with self.subTest(inputs=inputs), patch("pipeline.batch.run_batch", return_value={
                    "completed": 1, "partial": 0, "failed": 0, "skipped": 0,
                    "cancelled": False, "report_path": "result.json"}) as batch, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(inputs + ["--offline"]), 0)
            batch.assert_called_once()
            self.assertEqual(batch.call_args.args[0], inputs[0] if len(inputs) == 1 else inputs)
            self.assertTrue(batch.call_args.args[2].local_files_only)

    def test_gui_launch_does_not_require_translation_inputs(self):
        with patch("ui.app.launch_gui") as launch, patch("pipeline.batch.run_batch") as batch:
            self.assertEqual(main.main(["--gui"]), 0)
        launch.assert_called_once_with()
        batch.assert_not_called()


class EditorArgumentTests(unittest.TestCase):
    def test_local_ocr_default_off_and_explicit_flag_reaches_config(self):
        self.assertFalse(main.parser().parse_args([]).ocr_candidate_review)
        with patch("reader.launch_reader") as launch:
            self.assertEqual(main.main(["--read", "folder", "--ocr-candidate-review"]), 0)
        self.assertTrue(launch.call_args.kwargs["config"].ocr_candidate_review)

    def test_local_ocr_commands_require_page_and_explicit_opt_in(self):
        for action in (["--preview-ocr", "0", "0", "20", "30"], ["--apply-ocr-candidate", "candidate"]):
            for prefix in ([], ["--edit-page", "page.json", "--bubble", "left"]):
                with self.subTest(action=action, prefix=prefix), contextlib.redirect_stderr(io.StringIO()), \
                     patch("pipeline.editing.PageEditorSession") as session, self.assertRaises(SystemExit):
                    main.main(prefix + action)
                session.assert_not_called()

    def test_local_ocr_cli_uses_shared_backend_and_prints_candidate(self):
        prefix = ["--edit-page", "page.json", "--bubble", "left", "--ocr-candidate-review"]
        with patch("pipeline.editing.PageEditorSession") as factory, contextlib.redirect_stdout(io.StringIO()) as stdout:
            session = factory.return_value
            session.preview_ocr.return_value = {"candidate_id": "abc", "status": "ok"}
            self.assertEqual(main.main(prefix + ["--preview-ocr", "0", "1", "30", "40"]), 0)
            self.assertEqual(json.loads(stdout.getvalue())["candidate_id"], "abc")
            self.assertEqual(session.preview_ocr.call_args.args, ("left", [0, 1, 30, 40]))
            self.assertTrue(session.preview_ocr.call_args.kwargs["enabled"])
            session.retranslate.assert_not_called()
            session.close.assert_called_once()
        with patch("pipeline.editing.PageEditorSession") as factory, contextlib.redirect_stdout(io.StringIO()):
            session = factory.return_value
            session.apply_ocr_candidate.return_value = {"output_path": "page.png", "message": "saved"}
            self.assertEqual(main.main(prefix + ["--apply-ocr-candidate", "abc"]), 0)
            self.assertEqual(session.apply_ocr_candidate.call_args.args, ("left", "abc"))
            self.assertTrue(session.apply_ocr_candidate.call_args.kwargs["enabled"])
            session.preview_ocr.assert_not_called()

    def test_raw_reader_receives_cli_work_model_output_and_offline_settings(self):
        with patch("reader.launch_reader") as launch:
            self.assertEqual(main.main(["--read", "comic-folder", "--work", "Book One", "--db", "book.sqlite3",
                                        "--translator", "vntl", "--offline", "-o", "reader-results"]), 0)
        config = launch.call_args.kwargs["config"]
        self.assertEqual((config.work, config.memory_db, config.translation_backend),
                         ("Book One", "book.sqlite3", "vntl"))
        self.assertTrue(config.local_files_only)
        self.assertTrue(config.detect_free_text)
        self.assertEqual(launch.call_args.kwargs["output_dir"], "reader-results")

    def test_bubbles_only_and_local_magi_path_reach_reader_config(self):
        with patch("reader.launch_reader") as launch:
            self.assertEqual(main.main(["--read", "comic-folder", "--bubbles-only", "--magi-model", "local-magi"]), 0)
        config = launch.call_args.kwargs["config"]
        self.assertFalse(config.detect_free_text)
        self.assertEqual(config.magi_model_path, "local-magi")

    def test_translation_cli_defaults_and_explicit_overrides_reach_shared_pipeline(self):
        defaults = main.parser().parse_args([])
        self.assertEqual((defaults.translator, defaults.llama_context), ("vntl", 2048))
        self.assertEqual(defaults.mode, "auto")
        self.assertFalse(defaults.no_fragment_links)
        self.assertTrue(main.parser().parse_args(["--no-fragment-links"]).no_fragment_links)
        for mode in ("auto", "individual", "grouped"):
            self.assertEqual(main.parser().parse_args(["--mode", mode]).mode, mode)
        for backend in ("vntl", "qwen"):
            with self.subTest(backend=backend), patch("pipeline.batch.run_batch", return_value={
                    "completed": 1, "partial": 0, "failed": 0, "skipped": 0,
                    "cancelled": False, "report_path": "result.json"}) as batch, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(["image.png", "--translator", backend,
                    "--gguf", "selected.gguf", "--llama-server", "selected-server.exe",
                    "--llama-context", "1024", "--text-only", "--no-fragment-links"]), 0)
                config = batch.call_args.args[2]
                self.assertEqual(config.translation_backend, backend)
                self.assertEqual(config.gguf_path, "selected.gguf")
                self.assertEqual(config.llama_server_path, "selected-server.exe")
                self.assertEqual(config.llama_context_size, 1024)
                self.assertEqual(config.translation_mode, "auto")
                self.assertFalse(config.use_fragment_links)

    def test_edit_options_require_page(self):
        for args in (["--bubble", "b-1"], ["--list-bubbles"],
                     ["--set-translation", ""], ["--set-ocr-file", "source.txt"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with patch("pipeline.batch.run_batch", side_effect=AssertionError("Must not start a job")):
                    with self.assertRaises(SystemExit) as error:
                        main.main(args)
                    self.assertEqual(error.exception.code, 2)

    def test_gui_and_cli_editor_are_not_silently_mixed(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main.main(["--gui", "--edit-page", "page.json"])
        self.assertEqual(error.exception.code, 2)

    def test_speech_uses_confirmed_default_voice_and_japanese(self):
        with tempfile.TemporaryDirectory() as directory:
            db = str(Path(directory) / "terms.sqlite3")
            page = Path(directory) / "page.json"
            voice = {"style_id": 27, "params": {"pitchScale": 0.1}}
            with GlossaryMemory(db, "Book", "en") as memory:
                memory.upsert_character("unknown", "Unknown", voice=voice)
                role = memory.upsert_character(None, "Role A")
                memory.bind_character("original.png", "b-1", role["id"], confirmed=True, user_edit=True)
            page.write_text(json.dumps({"work": "Book", "target_language": "en", "input_path": "original.png",
                "config": {"memory_db": db}, "blocks": [{"block_id": "b-1", "source_text": "こんにちは。",
                                                         "translated_text": "Hello."}]}), encoding="utf-8")
            args = main.parser().parse_args(["--speak-page", str(page), "--bubble", "b-1"])
            with patch("audio.speech.VoicevoxClient"), patch("audio.speech.list_voices", return_value=[]), \
                    patch("audio.speech.synthesize_cached", return_value={"cached": True}) as synthesize, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.speech_command(args), 0)
            self.assertEqual(synthesize.call_args.args[1], "こんにちは。")
            self.assertEqual(synthesize.call_args.kwargs["voice"]["style_id"], 27)
            self.assertEqual(synthesize.call_args.kwargs["voice"]["params"]["pitchScale"], 0.1)


if __name__ == "__main__":
    unittest.main()
