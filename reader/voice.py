from __future__ import annotations

import copy

from .messages import CatalogRequest, Command, SpeechRequest


class VoiceMixin:
    def _choose_temporary_voice(self):
        if self.current_voice is None and self.voices and self.voice_connected:
            self.current_voice = copy.deepcopy(self.voices[0])
            self.temporary_voice = True

    def _voice_status_ready(self):
        if self.voice_connected:
            detail = f"Connected · {len(self.voices)} voices available."
            if self.temporary_voice:
                detail += " Temporary voice; use Save Name & Default Voice to keep it."
            self.voice_status.set("VOICEVOX · " + detail)

    def _show_voice_error(self, summary, detail, *, notify=True):
        """Keep diagnostics in Settings, rather than expanding the reading footer."""
        self.voice_status.set(f"{summary}\n\n{detail}")
        if notify:
            self.status.set(f"{summary} See Settings for details.")

    def _show_voice(self):
        voice = self.current_voice or {}
        index = next((i for i, candidate in enumerate(self.voices)
                      if all(candidate.get(key) == voice.get(key) for key in ("engine_uuid", "speaker_uuid", "style_id"))), -1)
        if index >= 0:
            self.voice_box.current(index)
        else:
            self.voice.set(voice.get("name", "Connect VOICEVOX to choose a voice"))

    def _voice_changed(self, _event=None):
        self._invalidate()
        index = self.voice_box.current()
        if index >= 0:
            self.current_voice = copy.deepcopy(self.voices[index])
            self.temporary_voice = True
            self._voice_status_ready()

    def _url_changed(self):
        self._invalidate()
        self.catalog_request = -1
        self.voice_connected = False
        self.voices = []
        self.voice_box.configure(values=[])
        if self.temporary_voice:
            self.current_voice = None
            self.temporary_voice = False
            self._show_voice()
        self._set_voice_connecting(False)
        self.voice_status.set("VOICEVOX · Address changed. Select Start / Connect.")

    def _settings_changed(self):
        self._invalidate()

    def _settings(self):
        speed = float(self.speed.get())
        if not 0.5 <= speed <= 2.0:
            raise ValueError("Choose a speed between 0.5 and 2.0.")
        return {"url": self.url.get().strip(), "speed": speed}

    def _set_voice_connecting(self, value):
        self.voice_connecting = value
        state = "disabled" if value or self.magi_busy or self.translation_busy else "normal"
        self.connect_button.configure(state=state)
        self.refresh_button.configure(state=state)

    def refresh_voices(self, *, start_local=False):
        token = self._invalidate()
        self.catalog_request = token
        self.catalog_start_local = start_local
        self.voice_connected = False
        self._set_voice_connecting(True)
        self.voice_status.set("VOICEVOX · Starting / connecting to the local engine…" if start_local else "VOICEVOX · Connecting and loading voices…")
        self.commands.put(Command("catalog", CatalogRequest(self.url.get().strip(), start_local), token))

    def start_voice_engine(self):
        self.refresh_voices(start_local=True)

    def _save_settings(self):
        try:
            settings = self._settings()
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self._invalidate()
        self.commands.put(Command("save_settings", settings, self.token))

    def _request_audio(self, sample=None):
        if self.loading or self.magi_busy or self.translation_busy:
            self.status.set("Wait for the current page, translation, or character analysis to finish.")
            return
        try:
            settings = self._settings()
            if sample is None and (self.page_index < 0 or self.block_id is None):
                raise ValueError("Select a speech bubble first.")
            if not self.voice_connected:
                if self.voice_connecting:
                    self._invalidate()
                else:
                    self.refresh_voices(start_local=True)
                # 只续播最后一次请求；停止、切页和换声线都会作废它。
                self.pending_audio = (self.token, sample)
                self.status.set("Connecting Japanese speech. The selected bubble will play when ready…")
                return
            if not self.current_voice:
                raise ValueError("No voices are available. Check the VOICEVOX settings.")
            voice = copy.deepcopy(self.current_voice)
            voice.setdefault("params", {})["speedScale"] = settings["speed"]
        except ValueError as exc:
            self.status.set(str(exc))
            return
        token = self._invalidate()
        payload = SpeechRequest(
            settings["url"], voice, sample=sample or "",
            path=self.pages[self.page_index]["path"] if self.page_index >= 0 else None,
            block_id=self.block_id if sample is None else None)
        self.status.set("Preparing Japanese audio…")
        self.commands.put(Command("speak", payload, token))

    def play(self):
        self._request_audio()

    def audition(self):
        self._request_audio("こんにちは。よろしくお願いします。")

    def _auto_changed(self):
        self._invalidate()
        self.status.set("Play on bubble click enabled." if self.autoplay.get() else "Play on bubble click disabled.")

    def _audio_finished(self, token):
        self.after_audio = None
        if token != self.token or self.closing.is_set():
            return
        self.status.set("Playback finished. Select another bubble or replay this one.")
