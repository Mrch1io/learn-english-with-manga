"""Run the Windows launcher with fake process creation, never a GPU engine.

Each test uses a temporary local port and leaves existing engines alone.
Run: python -m tools.test
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest


def ps_string(value):
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(sys.platform == "win32", "The bundled launcher uses Windows PowerShell")
class VoicevoxLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.launcher = self.root / "start_voicevox.ps1"
        shutil.copyfile(Path(__file__).resolve().parents[1] / "start_voicevox.ps1", self.launcher)
        self.engine = self.root / "fake-engine.exe"
        self.engine.touch()
        self.marker = self.root / "launched.json"
        self.wrapper = self.root / "test-launcher.ps1"

    def unused_port(self):
        while True:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            if port != 50021:
                return port

    def run_launcher(self, port, *, recorded_engine=False):
        if recorded_engine:
            logs = self.root / "data" / "voicevox"
            if port != 50021:
                logs /= str(port)
            logs.mkdir(parents=True)
            (logs / "process.json").write_text(json.dumps({"pid": 12345}), encoding="utf-8")
        self.wrapper.write_text("""
$ErrorActionPreference = 'Stop'
function Start-Process {
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle,
          [switch]$PassThru, $RedirectStandardOutput, $RedirectStandardError)
    @{ executable=$FilePath; arguments=$ArgumentList; window=$WindowStyle;
       stdout=$RedirectStandardOutput; stderr=$RedirectStandardError } |
        ConvertTo-Json | Set-Content -LiteralPath MARKER -Encoding utf8
    [pscustomobject]@{ Id=12345 }
}
function Get-CimInstance {
    param($ClassName, $Filter)
    [pscustomobject]@{ ExecutablePath=ENGINE }
}
& LAUNCHER -EnginePath ENGINE -Port PORT
""".replace("MARKER", ps_string(self.marker)).replace("PORT", str(port))
            .replace("LAUNCHER", ps_string(self.launcher)).replace("ENGINE", ps_string(self.engine)),
            encoding="utf-8")
        return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                               "-File", str(self.wrapper)], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=15,
                              creationflags=subprocess.CREATE_NO_WINDOW)

    @contextmanager
    def endpoint(self, payload, *, delay=False):
        release = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if delay:
                    release.wait(5)
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def log_message(self, *_args):
                pass

        class ExclusiveServer(ThreadingHTTPServer):
            allow_reuse_address = False

            def server_bind(self):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                super().server_bind()

        server = ExclusiveServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            yield server.server_port
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())

    def test_unused_port_launches_once_even_when_windows_refusal_exceeds_probe_timeout(self):
        port = self.unused_port()
        result = self.run_launcher(port)
        self.assertEqual(result.returncode, 0, result.stderr)
        launch = json.loads(self.marker.read_text(encoding="utf-8-sig"))
        self.assertEqual(launch["executable"], str(self.engine))
        self.assertEqual(launch["window"], "Hidden")
        self.assertIn("--use_gpu", launch["arguments"])
        self.assertIn("--disable_mutable_api", launch["arguments"])
        self.assertEqual(launch["arguments"][launch["arguments"].index("--port") + 1], str(port))
        logs = self.root / "data" / "voicevox" / str(port)
        self.assertEqual(launch["stdout"], str(logs / "stdout.log"))
        self.assertEqual(launch["stderr"], str(logs / "stderr.log"))
        self.assertTrue((logs / "process.json").is_file())
        self.assertFalse((logs.parent / "process.json").exists())

    def test_ready_engine_is_reused_without_process_creation(self):
        with self.endpoint("0.25.2") as port:
            result = self.run_launcher(port)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already available", result.stdout)
        self.assertIn(f"http://127.0.0.1:{port}", result.stdout)
        self.assertFalse(self.marker.exists())

    def test_unverified_listener_never_starts_another_engine(self):
        with self.endpoint({"service": "unrelated"}) as port:
            result = self.run_launcher(port)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"Port {port} is unavailable", result.stderr)
        self.assertFalse(self.marker.exists())

    def test_slow_unverified_listener_never_starts_another_engine(self):
        with self.endpoint("0.25.2", delay=True) as port:
            result = self.run_launcher(port)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"Port {port} is unavailable", result.stderr)
        self.assertFalse(self.marker.exists())

    def test_recorded_engine_is_reused_while_its_endpoint_is_initializing(self):
        with self.endpoint("0.25.2", delay=True) as port:
            result = self.run_launcher(port, recorded_engine=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("still initializing", result.stdout)
        self.assertFalse(self.marker.exists())

    def test_other_ports_process_records_are_ignored_and_preserved(self):
        port = self.unused_port()
        logs = self.root / "data" / "voicevox"
        other_port = port - 1 if port > 1 else port + 1
        old_record = json.dumps({"pid": 12345})
        for directory in (logs, logs / str(other_port)):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "process.json").write_text(old_record, encoding="utf-8")
        result = self.run_launcher(port)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.marker.exists())
        self.assertTrue((logs / str(port) / "process.json").is_file())
        for directory in (logs, logs / str(other_port)):
            self.assertEqual((directory / "process.json").read_text(encoding="utf-8"), old_record)

    def test_invalid_ports_fail_before_process_creation(self):
        for port in (-1, 0, 65536):
            with self.subTest(port=port):
                result = self.run_launcher(port)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ParameterArgumentValidationError", result.stderr)
                self.assertFalse(self.marker.exists())
                self.assertFalse((self.root / "data").exists())


if __name__ == "__main__":
    unittest.main()
