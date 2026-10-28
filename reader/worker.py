from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from queue import Queue
from threading import Event as ThreadEvent
from typing import TYPE_CHECKING, cast

from core.runtime import ROOT, Cancelled, Config
from . import images, records
from .messages import (
    BindingRequest, BindingResult, CatalogRequest, CatalogResult, CharacterRequest,
    CharacterResult, Command, ErrorResult, Event, ExportRequest, ExportResult,
    JsonObject, MagiRequest, OpenRequest, OpenResult, PreviewRequest, PreviewResult,
    Request, ResizeRequest, ResizeResult, Result, SpeechRequest, ThumbnailRequest,
    ThumbnailResult, Token, TranslationRequest, TranslationResult,
)

if TYPE_CHECKING:
    from core.runtime import ModelPool

SETTINGS_PATH = ROOT / "data" / "reader-settings.json"


class ReaderWorker:
    """Own blocking I/O and the model pool on one worker thread."""

    def __init__(self, commands: Queue[Command], events: Queue[Event],
                 closing: ThreadEvent, operation_cancel: ThreadEvent) -> None:
        self.commands = commands
        self.events = events
        self.closing = closing
        self.operation_cancel = operation_cancel
        self.pool: ModelPool | None = None

    def run(self) -> None:
        from audio.speech import VoicevoxClient, synthesize_cached

        self.voice_client = VoicevoxClient
        self.synthesize = synthesize_cached
        while True:
            action, payload, token = self.commands.get()
            if action == "close":
                self._close(token)
                return
            if self.closing.is_set():
                continue
            try:
                self.events.put(self._execute(action, payload, token))
            except Exception as exc:
                self.events.put(Event("error", ErrorResult(
                    action, f"{type(exc).__name__}: {exc}", isinstance(exc, Cancelled)), token))

    def _execute(self, action: str, payload: Request, token: Token) -> Event:
        result: Result
        match action, payload:
            case "settings", _:
                result = self._settings()
            case "save_settings", dict():
                result = self._save_settings(payload)
            case "open", OpenRequest():
                result = self._open(payload)
            case "catalog", CatalogRequest():
                result = self._catalog(payload)
            case "thumbnail", ThumbnailRequest():
                return self._thumbnail(payload, token)
            case ("page" | "zoom"), PreviewRequest():
                result = self._preview(payload)
            case "resize", ResizeRequest():
                result = ResizeResult(images.resize_original(
                    payload.original, payload.scale, payload.fit, fit_width=payload.fit_width),
                    payload.original.size)
            case "translate", TranslationRequest():
                result = self._translate(payload, token)
            case "export", ExportRequest():
                result = self._export(payload, token)
            case "character", CharacterRequest():
                result = self._character(payload)
            case "bind", BindingRequest():
                result = self._bind(payload)
            case "magi", MagiRequest():
                result = self._magi(payload, token)
            case "speak", SpeechRequest():
                result = self._speak(payload)
            case _:
                raise ValueError(f"Unknown reader command: {action}")
        return Event(action, result, token)

    def _close(self, token: Token) -> None:
        if self.pool is not None:
            try:
                self.pool.close()
            except Exception as exc:
                self.events.put(Event("error", ErrorResult(
                    "close", f"Could not release model resources: {exc}"), token))
        self.events.put(Event("closed", None, token))

    def _settings(self) -> JsonObject:
        try:
            result = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            return result if isinstance(result, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_settings(self, settings: JsonObject) -> JsonObject:
        from pipeline.batch import atomic_json

        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(SETTINGS_PATH, settings)
        return settings

    def _open(self, request: OpenRequest) -> OpenResult:
        pages, errors = records.collect_pages(request.paths, request.output_dir, config=request.config)
        if not pages:
            if errors:
                raise ValueError(f"{len(errors)} page records could not be read. First error: {errors[0]}")
            raise ValueError("No images or page records found. Choose images, JSON files, or a manga folder.")
        return OpenResult(pages, errors)

    def _catalog(self, request: CatalogRequest) -> CatalogResult:
        if request.start_local:
            from audio.speech import start_local_voicevox

            voices = start_local_voicevox(request.url, cancel_event=self.closing)
        else:
            voices = self.voice_client(request.url, timeout=2).list_voices()
        return CatalogResult(request.url, voices)

    def _thumbnail(self, request: ThumbnailRequest, token: Token) -> Event:
        identity = (request.path, request.input)
        try:
            image, view = images.page_thumbnail(request)
        except Exception as exc:
            return Event("thumbnail_error", ThumbnailResult(
                request.index, identity, request.version, message=str(exc)), token)
        return Event("thumbnail", ThumbnailResult(
            request.index, identity, request.version, image=image, view=view), token)

    def _preview(self, request: PreviewRequest) -> PreviewResult:
        record = (records.read_record(request.path) if request.path else
                  {"input_path": request.input, "blocks": [], "untranslated": True})
        view, notice = request.view, request.notice
        if view == "translated" and not record.get("exported", True):
            view = "original"
            notice = notice or "Text-only page. Export its saved translation in Processing."
        if view == "translated":
            preview, original = images.translated_preview(
                record, cast(str, request.path), request.scale, request.fit, fit_width=request.fit_width)
        else:
            preview, original = images.original_preview(
                record, request.scale, request.fit, untranslated=not request.path,
                fit_width=request.fit_width)
        # The canvas can still be 1x1 at startup; its preview is not a thumbnail.
        thumbnail = images.resize_original(original, 0, images.THUMBNAIL_SIZE)
        characters, bindings = [], {}
        if not record.get("untranslated"):
            with records.open_memory(record) as memory:
                characters = memory.list_characters()
                bindings = memory.list_character_bindings(record["input_path"])
        return PreviewResult(record, preview, original, view, notice, thumbnail, characters, bindings)

    def _model_pool(self, config: Config) -> ModelPool:
        from core.runtime import ModelPool

        if self.pool is None:
            self.pool = ModelPool(config.device, config.memory_policy)
        return self.pool

    def _translation_target(self, request: TranslationRequest) -> tuple[Config, Path]:
        config = request.config
        destination = Path(request.output_dir)
        if not request.path:
            return config, destination / Path(request.input).parent.relative_to(Path(request.root))
        old = records.read_record(request.path)
        if old.get("manual_edits") or old.get("ocr_candidate_edits"):
            raise ValueError("This page has manual corrections or OCR candidates. They were preserved; retranslate selected bubbles in Workbench.")
        saved = {key: value for key, value in old.get("config", {}).items()
                 if key in {field.name for field in fields(Config)}}
        saved.update(work=old.get("work", config.work), target_language=old.get("target_language", "en"),
                     translation_backend=config.translation_backend, translation_mode="auto", export_images=False)
        config = Config(**saved)
        destination = Path(request.path).parent
        if old.get("exported", True):
            # Keep reviewed text and its existing exported pixels together.
            destination /= "reader_text"
        return config, destination

    def _translate(self, request: TranslationRequest, token: Token) -> TranslationResult:
        from pipeline.batch import run_batch

        config, destination = self._translation_target(request)
        result = run_batch(request.input, destination, config, pool=self._model_pool(config),
                           cancel_event=self.operation_cancel,
                           progress=lambda event: self.events.put(Event("progress", event, token)))
        return TranslationResult(request.index, result)

    def _check_cancel(self, message: str) -> None:
        if self.operation_cancel.is_set() or self.closing.is_set():
            raise Cancelled(message)

    def _export(self, request: ExportRequest, token: Token) -> ExportResult:
        from pipeline.editing import PageEditorSession

        def check_cancel() -> None:
            self._check_cancel("Image export cancelled. Saved translations were preserved.")

        check_cancel()
        pool = self._model_pool(request.config)
        self.events.put(Event("progress", {
            "message": "Text saved. Cleaning the page and typesetting the English image…"}, token))
        editor = PageEditorSession(request.path, pool=pool)
        try:
            record = editor.export_image(cancel=check_cancel)
        finally:
            editor.close()
        return ExportResult(request.index, request.path, record)

    def _character(self, request: CharacterRequest) -> CharacterResult:
        record = records.read_record(request.path)
        with records.open_memory(record) as memory:
            saved = memory.upsert_character(request.character_id, request.name, request.voice)
            return CharacterResult(memory.list_characters(), saved["id"])

    def _bind(self, request: BindingRequest) -> BindingResult:
        record = records.read_record(request.path)
        with records.open_memory(record) as memory:
            memory.bind_character(record["input_path"], request.block_id, request.character_id,
                                  confirmed=True, user_edit=True)
            bindings = memory.list_character_bindings(record["input_path"])
            block: JsonObject = next((b for b in record["blocks"] if b["block_id"] == request.block_id), {})
            magi = block.get("metadata", {}).get("magi", {})
            warning = ""
            if magi.get("character_box") and magi.get("embedding"):
                try:
                    memory.save_character_sample(request.character_id, record["input_path"],
                                                 magi["character_box"], magi["embedding"])
                except ValueError as exc:
                    warning = f"Binding saved, but the character reference could not be saved: {exc}"
            return BindingResult(bindings, warning)

    def _magi(self, request: MagiRequest, token: Token) -> JsonObject:
        from vision.characters import analyze_pages

        def check_cancel() -> None:
            self._check_cancel("Magi character suggestions cancelled.")

        if self.pool is not None:
            self.pool.close()
            self.pool = None
        return analyze_pages(request.paths, model_path=request.model_path, device=request.device,
                             progress=lambda event: self.events.put(Event("progress", event, token)),
                             cancel=check_cancel)

    def _speak(self, request: SpeechRequest) -> JsonObject:
        terms = []
        text = request.sample
        if request.path:
            record = records.read_record(request.path)
            if request.block_id:
                block = next(block for block in record["blocks"] if block["block_id"] == request.block_id)
                text = block.get("source_text", "")
            with records.open_memory(record) as memory:
                terms = memory.list_terms()
        if not text.strip():
            raise ValueError("The selected bubble has no Japanese text to read aloud.")
        return self.synthesize(self.voice_client(request.url), text, reading_terms=terms,
                               voice=request.voice, cache_dir=ROOT / "data" / "speech-cache")
