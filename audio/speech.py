"""本地日语合成、读音替换与 WAV 缓存。

协议：https://voicevox.github.io/voicevox_engine/api/
HTTP 请求在单工作线程执行，播放及请求是否过期的检查在主线程执行。
"""
from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import wave

from core.runtime import ROOT
from core.memory import term_pattern


class SpeechError(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise SpeechError("VOICEVOX redirects are disabled; use the local engine URL directly")


def _local_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("Use a local HTTP engine URL such as http://127.0.0.1:50021")
    host = parsed.hostname or ""
    if host.lower() == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("VOICEVOX host must be localhost or a loopback IP address") from exc
    if not address.is_loopback:
        raise ValueError("Only local loopback VOICEVOX engines are allowed")
    host = f"[{address}]" if address.version == 6 else str(address)
    port = 50021 if parsed.port is None else parsed.port
    if port == 0:
        raise ValueError("VOICEVOX port must be between 1 and 65535")
    return f"http://{host}:{port}"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _version_key(version):
    return [(0, int(part)) if part.isdigit() else (1, part)
            for part in re.split(r"(\d+)", version)]


def list_voices(catalog):
    """从引擎目录生成可用声线，style ID 取实际返回值。"""
    versions = catalog.get("core_versions", [])
    if not versions or not all(isinstance(version, str) for version in versions):
        raise SpeechError("VOICEVOX returned no usable core versions")
    core = catalog.get("selected_core") or max(versions, key=_version_key)
    manifest = catalog.get("manifest", {})
    if not isinstance(manifest.get("uuid"), str) or not manifest["uuid"]:
        raise SpeechError("VOICEVOX engine manifest has no UUID")
    result = []
    for speaker in catalog.get("speakers", []):
        if not isinstance(speaker, dict) or not isinstance(speaker.get("speaker_uuid"), str):
            raise SpeechError("Invalid VOICEVOX speaker metadata")
        for style in speaker.get("styles", []):
            if not isinstance(style, dict):
                raise SpeechError("Invalid VOICEVOX style metadata")
            if style.get("type", "talk") not in {"talk", "streaming_talk"}:
                continue
            style_id = style.get("id")
            if isinstance(style_id, bool) or not isinstance(style_id, int):
                raise SpeechError("Invalid VOICEVOX style ID")
            label = f"{speaker.get('name', speaker['speaker_uuid'])} / {style.get('name', str(style_id))}"
            voice = {"name": label, "engine_uuid": manifest["uuid"], "engine_version": catalog["version"],
                     "core_version": core, "speaker_uuid": speaker["speaker_uuid"],
                     "speaker_version": str(speaker.get("version", "")), "style_id": style_id, "params": {}}
            result.append({"label": label, "voice": voice})
    return result


def voice_from_style(catalog, style_id, *, speaker_uuid=None):
    if isinstance(style_id, bool) or not isinstance(style_id, int):
        raise SpeechError("Choose an actual numeric style ID from the local engine")
    for row in list_voices(catalog):
        voice = row["voice"]
        if voice["style_id"] == style_id and (speaker_uuid is None or voice["speaker_uuid"] == speaker_uuid):
            return voice
    raise SpeechError("The selected speaker/style is unavailable; choose a voice from the current engine")


class VoicevoxClient:
    def __init__(self, base_url="http://127.0.0.1:50021", timeout=30):
        self.base_url = _local_url(base_url)
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("HTTP timeout must be positive")
        self.timeout = timeout
        # 禁用环境代理和重定向，让文本只发往本地引擎。
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _request(self, path, *, method="GET", params=None, body=None, binary=False):
        url = self.base_url + path + (("?" + urlencode(params)) if params else "")
        payload = _canonical(body).encode("utf-8") if body is not None else (b"" if method == "POST" else None)
        request = Request(url, data=payload, method=method, headers={"Content-Type": "application/json"})
        limit = 64 * 1024 * 1024 if binary else 8 * 1024 * 1024
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(limit + 1)
            if len(raw) > limit:
                raise SpeechError("VOICEVOX response exceeds the local response size limit")
        except HTTPError as exc:
            detail = exc.read(1024).decode("utf-8", errors="replace")
            raise SpeechError(f"VOICEVOX HTTP {exc.code} on {path}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise SpeechError(f"Cannot reach local VOICEVOX at {self.base_url}: {exc}") from exc
        if binary:
            return raw
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise SpeechError(f"VOICEVOX returned invalid JSON from {path}") from exc

    def get_catalog(self, core_version=None):
        version = self._request("/version")
        manifest = self._request("/engine_manifest")
        versions = self._request("/core_versions")
        if not isinstance(version, str) or not isinstance(manifest, dict) or not isinstance(versions, list):
            raise SpeechError("Invalid VOICEVOX engine metadata")
        if not versions or not all(isinstance(value, str) for value in versions):
            raise SpeechError("VOICEVOX returned no core versions")
        core = core_version or max(versions, key=_version_key)
        if core not in versions:
            raise SpeechError("The saved VOICEVOX core version is unavailable; choose a current voice")
        speakers = self._request("/speakers", params={"core_version": core})
        if not isinstance(speakers, list):
            raise SpeechError("Invalid VOICEVOX speaker list")
        return {"version": version, "manifest": manifest, "core_versions": versions,
                "selected_core": core, "speakers": speakers}

    def list_voices(self):
        return [row["voice"] for row in list_voices(self.get_catalog())]

    def audio_query(self, text, style_id, core_version):
        query = self._request("/audio_query", method="POST",
                              params={"text": text, "speaker": style_id, "core_version": core_version})
        if not isinstance(query, dict) or not isinstance(query.get("accent_phrases"), list):
            raise SpeechError("VOICEVOX returned an invalid audio query")
        return query

    def synthesis(self, query, style_id, core_version, *, enable_interrogative_upspeak=True):
        return self._request("/synthesis", method="POST", params={"speaker": style_id,
            "core_version": core_version, "enable_interrogative_upspeak": str(enable_interrogative_upspeak).lower()},
            body=query, binary=True)


def _launcher_output_tail(stream):
    stream.seek(0, os.SEEK_END)
    stream.seek(max(0, stream.tell() - 4096))
    return stream.read(4096).decode("utf-8", errors="replace").strip()[-1000:]


def start_local_voicevox(base_url, *, cancel_event=None, timeout=30, project_root=None):
    """在工作线程连接或启动 Windows GPU 引擎；取消只结束等待。

    仅默认地址可自动启动，其他本地地址只尝试连接。
    """
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Startup timeout must be positive")
    cancelled = cancel_event if cancel_event is not None else threading.Event()
    client = VoicevoxClient(base_url, timeout=min(2, timeout))
    if cancelled.is_set():
        raise SpeechError("Waiting for VOICEVOX was cancelled.")
    try:
        return client.list_voices()
    except SpeechError as exc:
        connection_error = exc
    if client.base_url != "http://127.0.0.1:50021":
        raise SpeechError(f"Cannot connect to {client.base_url}. This address is connection-only; start its engine manually.") from connection_error
    if sys.platform != "win32":
        raise SpeechError("Automatic startup supports Windows only. Start VOICEVOX manually, then select Refresh Voices.") from connection_error
    root = Path(project_root) if project_root is not None else ROOT
    launcher = root / "start_voicevox.ps1"
    engine = root / "runtime" / "voicevox" / "windows-directml" / "run.exe"
    if not launcher.is_file() or not engine.is_file():
        raise SpeechError("The local GPU voice engine or launcher is missing. Configure VOICEVOX before using speech; reading remains available.")
    if cancelled.is_set():
        raise SpeechError("Waiting for VOICEVOX was cancelled.")
    try:
        # 引擎会继承输出句柄；用文件避免管道读线程等到引擎退出。
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            started = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(launcher)],
                cwd=root, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=12, check=False)
            detail = (_launcher_output_tail(stderr) or _launcher_output_tail(stdout)) if started.returncode else ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SpeechError(f"VOICEVOX launcher did not finish: {exc}. Check the logs in data/voicevox.") from exc
    if started.returncode:
        raise SpeechError(f"VOICEVOX startup failed: {detail or 'the launcher returned an error'}. Check the logs in data/voicevox.")
    deadline = time.monotonic() + timeout
    while not cancelled.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        client.timeout = min(2, remaining)
        try:
            return client.list_voices()
        except SpeechError:
            # 冷启动时端口可能先就绪，声线目录尚未加载完。
            cancelled.wait(min(.25, max(0, deadline-time.monotonic())))
    if cancelled.is_set():
        raise SpeechError("Waiting for VOICEVOX was cancelled; any started engine remains running.")
    raise SpeechError("VOICEVOX is not ready yet. Check the logs in data/voicevox, then select Refresh Voices; reading remains available.")


def apply_readings(source_text, reading_terms):
    """在朗读副本中按最长精确匹配替换读音，不递归替换。"""
    choices = {}
    # 原词优先于别名；同级匹配优先使用锁定术语。
    terms = sorted(reading_terms, key=lambda term: (not bool(term.get("locked")), str(term.get("source", ""))))
    for alias_pass in (False, True):
        for term in terms:
            reading = term.get("reading", "")
            if not isinstance(reading, str) or not reading.strip():
                continue
            spellings = term.get("aliases", []) if alias_pass else [term.get("source", "")]
            for spelling in spellings:
                if isinstance(spelling, str) and spelling and spelling not in choices:
                    choices[spelling] = {"source": term["source"], "matched": spelling, "reading": reading.strip()}
    if not choices:
        return source_text, []
    patterns = [term_pattern(spelling) for spelling in sorted(choices, key=lambda value: (-len(value), value))]
    used = {}
    def replace(match):
        term = choices[match.group(0)]
        used[match.group(0)] = term
        return term["reading"]
    spoken = re.sub("|".join(patterns), replace, source_text)
    return spoken, [used[key] for key in sorted(used)]


def _params(values):
    if not isinstance(values, dict):
        raise ValueError("Voice parameters must be an object")
    limits = {"speedScale": (0.1, 4.0), "pitchScale": (-1.0, 1.0), "intonationScale": (0.0, 2.0),
              "volumeScale": (0.0, 2.0), "prePhonemeLength": (0.0, 2.0), "postPhonemeLength": (0.0, 2.0),
              "pauseLengthScale": (0.0, 4.0), "outputSamplingRate": (8000, 192000)}
    result = {}
    for name, value in values.items():
        if name in {"outputStereo", "enable_interrogative_upspeak"}:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        elif name in limits:
            low, high = limits[name]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
            if name == "outputSamplingRate" and not isinstance(value, int):
                raise ValueError("outputSamplingRate must be an integer")
        else:
            raise ValueError(f"Unsupported voice parameter: {name}")
        result[name] = value
    return result


def _wave_info(raw):
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            frames, rate = wav.getnframes(), wav.getframerate()
            if rate <= 0 or frames <= 0 or wav.getcomptype() != "NONE":
                raise SpeechError("VOICEVOX returned an empty or unsupported WAV")
            expected = frames * wav.getnchannels() * wav.getsampwidth()
            if len(wav.readframes(frames)) != expected:
                raise SpeechError("VOICEVOX returned a truncated WAV")
            return {"duration_seconds": frames / rate, "sample_rate": rate,
                    "channels": wav.getnchannels(), "sample_width": wav.getsampwidth()}
    except (wave.Error, EOFError) as exc:
        raise SpeechError("VOICEVOX response is not a readable PCM WAV") from exc


def _atomic_bytes(path, data):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def synthesize_cached(client, source_text, *, reading_terms, voice, cache_dir):
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("Japanese source text is empty")
    if len(source_text) > 10000:
        raise ValueError("Speech text exceeds the 10,000-character per-request limit")
    if not isinstance(voice, dict) or not voice.get("engine_uuid") or not voice.get("speaker_uuid"):
        raise ValueError("Choose a voice from the local VOICEVOX engine first")
    catalog = client.get_catalog(core_version=voice.get("core_version"))
    current = voice_from_style(catalog, voice.get("style_id"), speaker_uuid=voice["speaker_uuid"])
    if current["engine_uuid"] != voice["engine_uuid"]:
        raise SpeechError("The selected voice belongs to a different VOICEVOX engine")
    current["params"] = _params(voice.get("params", {}))
    spoken, readings = apply_readings(source_text, reading_terms)
    identity = {"schema": 1, "source_text": source_text, "spoken_text": spoken, "readings": readings,
                "voice": {key: value for key, value in current.items() if key != "name"}}
    key = hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()
    directory = Path(cache_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    wav_path, metadata_path = directory / (key + ".wav"), directory / (key + ".json")
    result = {"wav_path": str(wav_path), "metadata_path": str(metadata_path), "cache_key": key,
              "spoken_text": spoken, "voice": current}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw = wav_path.read_bytes()
        if metadata.get("identity") == identity and metadata.get("wav_sha256") == hashlib.sha256(raw).hexdigest():
            info = _wave_info(raw)
            return {**result, **info, "cached": True}
    except (OSError, ValueError, TypeError, AttributeError, SpeechError):
        pass
    query = client.audio_query(spoken, current["style_id"], current["core_version"])
    query.update({key: value for key, value in current["params"].items() if key != "enable_interrogative_upspeak"})
    raw = client.synthesis(query, current["style_id"], current["core_version"],
                           enable_interrogative_upspeak=current["params"].get("enable_interrogative_upspeak", True))
    info = _wave_info(raw)
    metadata = {"identity": identity, "wav_sha256": hashlib.sha256(raw).hexdigest(), **info}
    _atomic_bytes(wav_path, raw)
    # JSON 最后写入，作为缓存完成标记；只有 WAV 时会重新合成。
    _atomic_bytes(metadata_path, _canonical(metadata).encode("utf-8"))
    return {**result, **info, "cached": False}


def play_wav(path):
    if os.name != "nt":
        raise SpeechError("WAV playback currently supports Windows only")
    import winsound
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)


def stop_playback():
    if os.name != "nt":
        raise SpeechError("WAV playback currently supports Windows only")
    import winsound
    winsound.PlaySound(None, 0)


