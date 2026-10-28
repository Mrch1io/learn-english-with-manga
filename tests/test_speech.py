"""Protocol/cache tests use mocked HTTP responses, not a real TTS engine."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit
import wave
from audio import speech

from audio.speech import (SpeechError, VoicevoxClient, apply_readings,
                    list_voices, play_wav, start_local_voicevox, stop_playback, synthesize_cached, voice_from_style)


def pcm_wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"\0\0" * 240)
    return output.getvalue()


class MockHTTP:
    def __init__(self):
        self.calls = []
        self.version = "0.25.2"
        self.manifest = {"uuid": "engine-a"}
        self.cores = ["0.9.0", "0.16.0"]
        self.speakers = [{"name": "Voice A", "speaker_uuid": "speaker-a", "version": "1.2.3",
                          "styles": [{"name": "Normal", "id": 77, "type": "talk"},
                                     {"name": "Singing", "id": 88, "type": "sing"}]}]
        self.wav = pcm_wav()

    def open(self, request, timeout):
        parsed = urlsplit(request.full_url)
        params = parse_qs(parsed.query)
        body = json.loads(request.data) if request.data else None
        self.calls.append((request.method, parsed.path, params, body))
        value = {"/version": self.version, "/engine_manifest": self.manifest,
                 "/core_versions": self.cores, "/speakers": self.speakers,
                 "/audio_query": {"accent_phrases": [], "speedScale": 1.0, "outputSamplingRate": 24000}}
        if parsed.path == "/synthesis":
            return io.BytesIO(self.wav)
        return io.BytesIO(json.dumps(value[parsed.path], ensure_ascii=False).encode("utf-8"))

    def posts(self):
        return [call for call in self.calls if call[0] == "POST"]


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.client = VoicevoxClient()
        self.http = MockHTTP()
        self.client.opener = self.http
        self.voice = self.client.list_voices()[0]
        self.terms = [{"source": "凛", "target": "Rin", "aliases": ["リン"], "reading": "りん"}]

    def tearDown(self):
        self.folder.cleanup()

    def synthesize(self, source="凛は来た。", voice=None, terms=None):
        return synthesize_cached(self.client, source, reading_terms=self.terms if terms is None else terms,
                                 voice=self.voice if voice is None else voice, cache_dir=self.folder.name)

    def test_only_loopback_http_is_allowed(self):
        for address in ("http://example.com:50021", "http://192.168.0.1:50021", "https://127.0.0.1",
                        "http://user@127.0.0.1", "http://127.0.0.1/path", "http://127.0.0.1?x=1",
                        "http://127.0.0.1:0", "http://[::ffff:192.168.0.1]:50021"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                VoicevoxClient(address)
        self.assertEqual(VoicevoxClient("http://localhost").base_url, "http://127.0.0.1:50021")
        self.assertEqual(VoicevoxClient("http://[::1]:50022/").base_url, "http://[::1]:50022")

    def test_catalog_reads_actual_style_and_speaker_versions(self):
        catalog = self.client.get_catalog()
        rows = list_voices(catalog)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["voice"]["style_id"], 77)
        self.assertEqual(rows[0]["voice"]["core_version"], "0.16.0")
        self.assertEqual(rows[0]["voice"]["speaker_version"], "1.2.3")
        self.assertEqual(voice_from_style(catalog, 77, speaker_uuid="speaker-a"), rows[0]["voice"])
        with self.assertRaises(SpeechError):
            voice_from_style(catalog, True)
        with self.assertRaises(SpeechError):
            voice_from_style(catalog, 88)

    def test_longest_reading_aliases_boundaries_and_no_recursive_replacement(self):
        source = "東京大学と東京、リンとHannahとAnn"
        terms = self.terms + [
            {"source": "東京", "reading": "とうきょう"},
            {"source": "東京大学", "reading": "とうだい"},
            {"source": "Ann", "reading": "アン"},
            {"source": "とうだい", "reading": "wrong"}]
        spoken, used = apply_readings(source, terms)
        self.assertEqual(spoken, "とうだいととうきょう、りんとHannahとアン")
        self.assertEqual(source, "東京大学と東京、リンとHannahとAnn")
        self.assertEqual({row["matched"] for row in used}, {"東京大学", "東京", "リン", "Ann"})

    def test_http_protocol_and_query_parameters_apply_only_to_spoken_source(self):
        voice = copy.deepcopy(self.voice)
        voice["params"] = {"speedScale": 1.2, "enable_interrogative_upspeak": False}
        result = self.synthesize(voice=voice)
        query, synthesis = self.http.posts()
        self.assertEqual(query[1], "/audio_query")
        self.assertEqual(query[2]["text"], ["りんは来た。"])
        self.assertEqual(query[2]["speaker"], ["77"])
        self.assertEqual(query[2]["core_version"], ["0.16.0"])
        self.assertEqual(synthesis[1], "/synthesis")
        self.assertEqual(synthesis[2]["enable_interrogative_upspeak"], ["false"])
        self.assertEqual(synthesis[3]["speedScale"], 1.2)
        self.assertNotIn("enable_interrogative_upspeak", synthesis[3])
        self.assertAlmostEqual(result["duration_seconds"], .01)
        self.assertEqual(Path(result["wav_path"]).read_bytes(), pcm_wav())

    def test_cache_reuses_wav_when_only_english_translation_changes(self):
        first = self.synthesize()
        terms = copy.deepcopy(self.terms)
        terms[0]["target"] = "A different English spelling"
        second = self.synthesize(terms=terms)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["cache_key"], second["cache_key"])
        self.assertEqual(len(self.http.posts()), 2)
        metadata = json.loads(Path(first["metadata_path"]).read_text(encoding="utf-8"))
        self.assertNotIn("target", json.dumps(metadata))
        self.assertEqual(metadata["identity"]["source_text"], "凛は来た。")

    def test_cache_invalidates_on_reading_source_voice_parameter_and_actual_versions(self):
        keys = {self.synthesize()["cache_key"]}
        changed_reading = [{**self.terms[0], "reading": "リン"}]
        keys.add(self.synthesize(terms=changed_reading)["cache_key"])
        keys.add(self.synthesize(source="凛は去った。")["cache_key"])
        voice = copy.deepcopy(self.voice)
        voice["params"] = {"speedScale": 1.3}
        keys.add(self.synthesize(voice=voice)["cache_key"])
        self.http.version = "0.25.3"
        keys.add(self.synthesize()["cache_key"])
        self.http.speakers[0]["version"] = "1.2.4"
        keys.add(self.synthesize()["cache_key"])
        self.http.cores.append("0.17.0")
        updated_voice = voice_from_style(self.client.get_catalog(), 77)
        keys.add(self.synthesize(voice=updated_voice)["cache_key"])
        self.http.speakers[0]["styles"].append({"name": "Soft", "id": 123, "type": "talk"})
        changed_style = voice_from_style(self.client.get_catalog(), 123)
        keys.add(self.synthesize(voice=changed_style)["cache_key"])
        self.assertEqual(len(keys), 8)
        self.assertEqual(len(self.http.posts()), 16)

    def test_unrelated_reading_does_not_invalidate_cache(self):
        first = self.synthesize()
        terms = self.terms + [{"source": "無関係", "reading": "むかんけい"}]
        second = self.synthesize(terms=terms)
        self.assertTrue(second["cached"])
        self.assertEqual(first["cache_key"], second["cache_key"])

    def test_corrupt_or_incomplete_cache_is_rebuilt(self):
        first = self.synthesize()
        Path(first["wav_path"]).write_bytes(b"corrupt")
        self.assertFalse(self.synthesize()["cached"])
        Path(first["metadata_path"]).unlink()
        self.assertFalse(self.synthesize()["cached"])
        self.assertEqual(len(self.http.posts()), 6)
        self.assertEqual(list(Path(self.folder.name).glob("*.tmp")), [])

    def test_invalid_wav_does_not_create_completed_cache(self):
        self.http.wav = b"not wav"
        with self.assertRaises(SpeechError):
            self.synthesize()
        self.assertEqual(list(Path(self.folder.name).iterdir()), [])

    def test_missing_core_or_wrong_engine_does_not_silently_change_voice(self):
        self.http.cores = ["1.0.0"]
        with self.assertRaises(SpeechError):
            self.synthesize()
        self.http.cores = ["0.16.0"]
        self.http.manifest["uuid"] = "different-engine"
        with self.assertRaises(SpeechError):
            self.synthesize()
        self.assertEqual(self.http.posts(), [])

    def test_engine_connection_errors_are_explicit(self):
        with patch.object(self.http, "open", side_effect=URLError("not running")):
            with self.assertRaisesRegex(SpeechError, "Cannot reach local VOICEVOX"):
                self.synthesize()

    def test_invalid_parameters_fail_before_audio_requests(self):
        for params in ({"speedScale": 0}, {"speedScale": float("nan")}, {"outputStereo": "yes"},
                       {"unsupported": 1}, {"outputSamplingRate": 24000.5}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.synthesize(voice={**self.voice, "params": params})
        self.assertEqual(self.http.posts(), [])

    @unittest.skipUnless(__import__("os").name == "nt", "Windows playback")
    def test_windows_playback_is_async_and_can_stop_without_playing_audio(self):
        import winsound
        path = Path(self.folder.name) / "test.wav"
        path.write_bytes(pcm_wav())
        with patch("winsound.PlaySound") as play:
            self.assertIsNone(play_wav(path))
            self.assertTrue(play.call_args.args[1] & winsound.SND_ASYNC)
            stop_playback()
            play.assert_called_with(None, 0)


class EngineStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.launcher = self.root / "start_voicevox.ps1"
        self.launcher.touch()
        engine = self.root / "runtime/voicevox/windows-directml/run.exe"
        engine.parent.mkdir(parents=True)
        engine.touch()
        self.client = Mock(base_url="http://127.0.0.1:50021")
        self.voices = [{"name": "Actual voice", "style_id": 71}]
        self.client.list_voices.return_value = self.voices
        self.client_patch = patch.object(speech, "VoicevoxClient", return_value=self.client)
        self.client_patch.start()
        self.platform_patch = patch.object(speech.sys, "platform", "win32")
        self.platform_patch.start()
        self.real_run = speech.subprocess.run
        self.run_patch = patch.object(speech.subprocess, "run", return_value=Mock(returncode=0))
        self.run = self.run_patch.start()

    def tearDown(self):
        self.run_patch.stop()
        self.platform_patch.stop()
        self.client_patch.stop()
        self.temp.cleanup()

    def start(self, **kwargs):
        return start_local_voicevox(self.client.base_url, project_root=self.root, **kwargs)

    def test_running_engine_is_reused_without_starting_a_process(self):
        self.assertEqual(self.start(), self.voices)
        self.run.assert_not_called()

    def test_cold_engine_launches_hidden_once_then_reads_actual_voices(self):
        self.client.list_voices.side_effect = [SpeechError("offline"), SpeechError("warming"), self.voices]
        self.assertEqual(self.start(), self.voices)
        self.run.assert_called_once()
        args, kwargs = self.run.call_args
        self.assertEqual(args[0][-2:], ["-File", str(self.launcher)])
        self.assertEqual(kwargs["creationflags"], speech.subprocess.CREATE_NO_WINDOW)
        self.assertFalse(kwargs.get("shell", False))
        self.assertFalse(kwargs.get("capture_output", False))
        self.assertTrue(kwargs["stdout"].closed)
        self.assertTrue(kwargs["stderr"].closed)
        self.assertEqual(self.client.list_voices.call_count, 3)

    def test_other_address_only_connects_and_never_launches_default_engine(self):
        self.client.base_url = "http://127.0.0.1:50022"
        self.client.list_voices.side_effect = SpeechError("offline")
        with self.assertRaisesRegex(SpeechError, "connection-only"):
            self.start()
        self.run.assert_not_called()

    def test_missing_engine_and_non_windows_have_actionable_errors(self):
        self.client.list_voices.side_effect = SpeechError("offline")
        with patch.object(speech.sys, "platform", "linux"):
            with self.assertRaisesRegex(SpeechError, "Automatic startup supports Windows only"):
                self.start()
        self.launcher.unlink()
        with self.assertRaisesRegex(SpeechError, "local GPU voice engine or launcher is missing"):
            self.start()
        self.run.assert_not_called()

    def test_failed_launcher_is_reported_without_retrying(self):
        self.client.list_voices.side_effect = SpeechError("offline")

        def failed_launcher(*args, **kwargs):
            kwargs["stdout"].write(b"launcher stdout must not replace its error")
            kwargs["stderr"].write(b"old details\n" + b"x" * 8192 + b"\nport occupied")
            return Mock(returncode=1)

        self.run.side_effect = failed_launcher
        with self.assertRaisesRegex(SpeechError, "port occupied") as caught:
            self.start()
        self.assertNotIn("old details", str(caught.exception))
        self.assertNotIn("launcher stdout", str(caught.exception))
        self.assertLess(len(str(caught.exception)), 1100)
        self.run.assert_called_once()
        self.assertEqual(self.client.list_voices.call_count, 1)

    def test_launcher_return_does_not_wait_for_child_inheriting_output_handles(self):
        # A harmless Python child models the background engine's inherited
        # handles. It exits on our stop file or after five seconds, whichever
        # happens first; no real engine or local port is used.
        child = self.root / "background.py"
        child.write_text("""
from pathlib import Path
import sys, time
folder = Path(sys.argv[1])
(folder / 'child-started').touch()
deadline = time.monotonic() + 5
while not (folder / 'child-stop').exists() and time.monotonic() < deadline:
    time.sleep(.02)
(folder / 'child-finished').touch()
""", encoding="utf-8")
        launcher = self.root / "launcher.py"
        launcher.write_text("""
from pathlib import Path
import subprocess, sys, time
folder = Path(__file__).parent
subprocess.Popen([sys.executable, str(folder / 'background.py'), str(folder)],
    stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr,
    creationflags=subprocess.CREATE_NO_WINDOW)
deadline = time.monotonic() + 3
while not (folder / 'child-started').exists() and time.monotonic() < deadline:
    time.sleep(.02)
print('launcher returned; child still owns the output handles', flush=True)
""", encoding="utf-8")
        self.client.list_voices.side_effect = [SpeechError("offline"), self.voices]
        self.run.side_effect = lambda _args, **kwargs: self.real_run([sys.executable, str(launcher)], **kwargs)
        try:
            self.assertEqual(self.start(), self.voices)
            self.assertTrue((self.root / "child-started").exists())
            self.assertFalse((self.root / "child-finished").exists(), "Startup waited for the background child's output handles")
        finally:
            (self.root / "child-stop").touch()
            deadline = time.monotonic() + 5
            while not (self.root / "child-finished").exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue((self.root / "child-finished").exists())

    def test_launcher_timeout_closes_temporary_output_files(self):
        self.client.list_voices.side_effect = SpeechError("offline")
        self.run.side_effect = speech.subprocess.TimeoutExpired("powershell.exe", 12)
        with self.assertRaisesRegex(SpeechError, "launcher did not finish"):
            self.start()
        self.assertTrue(self.run.call_args.kwargs["stdout"].closed)
        self.assertTrue(self.run.call_args.kwargs["stderr"].closed)

    def test_close_cancels_readiness_wait_without_terminating_engine(self):
        cancelled = threading.Event()
        calls = []

        def unavailable():
            calls.append(True)
            if len(calls) == 2:
                cancelled.set()
            raise SpeechError("warming")

        self.client.list_voices.side_effect = unavailable
        with self.assertRaisesRegex(SpeechError, "any started engine remains running"):
            self.start(cancel_event=cancelled)
        self.run.assert_called_once()
        self.assertEqual(len(calls), 2)

    def test_startup_wait_has_a_deadline_and_does_not_restart(self):
        self.client.list_voices.side_effect = SpeechError("warming")
        with patch.object(speech.time, "monotonic", side_effect=[0, 31]):
            with self.assertRaisesRegex(SpeechError, "VOICEVOX is not ready yet"):
                self.start(timeout=30)
        self.run.assert_called_once()
        self.assertEqual(self.client.list_voices.call_count, 1)


if __name__ == "__main__":
    unittest.main()
