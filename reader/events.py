from __future__ import annotations

import queue
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox

from core.runtime import Config
from .messages import (
    BindingResult, CatalogResult, CharacterResult, ErrorResult, Event, ExportResult,
    JsonObject, OpenResult, PreviewResult, ResizeResult, Result, Token, TranslationResult,
)


class EventsMixin:
    def _drain(self) -> None:
        for _ in range(40):
            try:
                action, result, token = self.events.get_nowait()
            except queue.Empty:
                break
            if action == "closed":
                if self.worker.is_alive():
                    self.events.put(Event(action, result, token))
                    break
                self._finish_close()
                return
            if action == "error" and result.action == "close":
                messagebox.showerror("Resource cleanup failed", result.message, parent=self.window)
                continue
            if self.closing.is_set():
                continue
            if action in ("thumbnail", "thumbnail_error"):
                self._receive_thumbnail(action, result, token)
                continue
            if action == "settings":
                self.url.set(result.get("url", self.url.get()))
                self.speed.set(str(result.get("speed", 1.0)))
                self.refresh_voices()
                continue
            # 调整语速只作废音频，不能丢弃尚未返回的翻页结果。
            event_action = result.action if action == "error" else action
            expected = (self.preview_token if event_action == "resize"
                        else self.page_token if event_action in ("open", "page", "zoom", "character", "bind")
                        else self.catalog_request if event_action == "catalog" else self.token)
            if token == expected:
                self._receive_result(action, result, token)
        self._poll = self.window.after(80, self._drain)

    def _receive_result(self, action: str, result: Result, token: Token) -> None:
        match action, result:
            case "error", ErrorResult():
                self._receive_error(result)
            case "progress", _:
                self.status.set(str(result.get("message", result)) if isinstance(result, dict) else str(result))
            case "magi", dict():
                self._receive_magi(result)
            case "translate", TranslationResult():
                self._receive_translation(result)
            case "export", ExportResult():
                self._receive_export(result)
            case "open", OpenResult():
                self._receive_open(result)
            case "resize", ResizeResult():
                self.resizing = False
                self._display_preview(result.preview, result.original_size)
                self._canvas_resized()
            case ("page" | "zoom"), PreviewResult():
                self._receive_page(action, result)
            case "catalog", CatalogResult():
                self._receive_catalog(result)
            case "character", CharacterResult():
                self.characters = result.characters
                self._set_roles(result.selected_id)
                self._refresh_rows()
                self.status.set("Character and default voice saved.")
            case "bind", BindingResult():
                self.bindings = result.bindings
                self._refresh_rows()
                self.status.set(result.warning or "Bubble character binding saved.")
            case "save_settings", _:
                self.status.set("VOICEVOX address and speed saved.")
            case "speak", dict():
                self._play_result(result, token)

    def _receive_error(self, result: ErrorResult) -> None:
        if result.action == "open":
            self.loading = self.opening = False
            self._schedule_thumbnails()
            self._update_page_navigation()
        if result.action == "catalog":
            self.voice_connected = False
            self._set_voice_connecting(False)
            pending = self.pending_audio
            self.pending_audio = None
            if pending and pending[0] == self.token and not self.catalog_start_local:
                # 连接失败时，为等待播放的点击补做一次引擎启动。
                self._request_audio(pending[1])
                return
            self._show_voice_error("Voice unavailable.", result.message,
                                   notify=bool(pending) or self.catalog_start_local)
            return
        if result.action == "speak":
            self._show_voice_error("Voice playback failed.", result.message)
            return
        if result.action in ("page", "zoom"):
            self.loading = False
            self._clear_page()
            self._schedule_thumbnails()
            self._update_page_navigation()
        if result.action == "resize":
            self.resizing = False
        if result.action == "magi":
            self._set_magi_busy(False)
        if result.action == "translate":
            self._translation_finished()
            self.pages[self.page_index]["state"] = "failed"
            self._refresh_page_rows()
        if result.action == "export":
            self._translation_finished()
            self._invalidate_thumbnail(self.page_index)
            self.view.set("原图")
            notice = ("Image export cancelled. Saved text remains available; showing the original image." if result.cancelled else
                      "Text translated, but English image export failed. Showing the original image. " + result.message)
            self._load_page(notice=notice)
            return
        self.status.set(result.message)

    def _receive_magi(self, result: JsonObject) -> None:
        self._set_magi_busy(False)
        completed, failed = result.get("completed", 0), result.get("failed", 0)
        outcome = ("cancelled" if result.get("cancelled") else
                   "partially failed" if failed and completed else "failed" if failed else "completed")
        notice = f"Magi analysis {outcome}: {completed} pages completed, {failed} failed."
        error = next((page["error"] for page in result.get("pages", []) if page.get("error")), "")
        if error:
            notice += f" First error: {error}"
        self._load_page(notice=notice)

    def _receive_translation(self, result: TranslationResult) -> None:
        self._translation_finished()
        index = result.index
        self._invalidate_thumbnail(index)
        page = self.pages[index]
        attempt = next(iter(result.report.get("pages", [])), {})
        page["state"] = attempt.get("status", "cancelled" if result.report.get("cancelled") else "failed")
        output = attempt.get("output_path")
        if output and page["state"] in {"done", "partial", "skipped", "needs_review"}:
            page["path"] = str(Path(output).with_suffix(".json"))
            if result.report.get("cancelled") or self.operation_cancel.is_set():
                self.view.set("原图")
                self._load_page(notice="Translated text saved. Task cancelled before the English image was exported.")
            else:
                self._export_page()
        else:
            self.status.set(attempt.get("error", "This page was cancelled. Translation can be restarted." if result.report.get("cancelled") else "Page translation failed. Retry when ready."))
        self._refresh_page_rows()

    def _receive_export(self, result: ExportResult) -> None:
        self._translation_finished()
        index = result.index
        self._invalidate_thumbnail(index)
        page = self.pages[index]
        page["path"] = result.path
        page["state"] = result.record.get("status", "done")
        self._refresh_page_rows()
        cancelled = self.operation_cancel.is_set()
        self.view.set("原图" if cancelled else "英文译图")
        self._load_page(notice="English image saved. Task cancelled; keeping the original image visible." if cancelled else "")

    def _receive_open(self, result: OpenResult) -> None:
        self.opening = False
        self.pages = result.pages
        self.page_index = next((index for index, page in enumerate(self.pages)
            if self.initial_path and self.initial_path in (page["path"], page["input"])), 0)
        self.initial_path = None
        self.block_id = None
        self.view.set("英文译图" if self.pages[self.page_index]["path"] else "原图")
        self._refresh_page_rows()
        self._see_thumbnail(self.page_index)
        errors = result.errors
        self._load_page(notice=f"{len(errors)} page records could not be read. First error: {errors[0]}" if errors else "")

    def _receive_page(self, action: str, result: PreviewResult) -> None:
        self.record = result.record
        self.view.set("英文译图" if result.view == "translated" else "原图")
        if action == "page" and not self.record.get("untranslated"):
            self.work.set(self.record.get("work", "default"))
            self.config = replace(self.config, work=self.work.get(),
                target_language=self.record.get("target_language", "en"),
                memory_db=self.record.get("config", {}).get("memory_db", Config().memory_db))
        self._update_display_title()
        self.loading = False
        self.blocks = {b["block_id"]: b for b in self.record["blocks"]}
        self.characters, self.bindings = result.characters, result.bindings
        self.original = result.original
        if result.thumbnail is not None:
            from PIL import ImageTk

            self.thumbnail = ImageTk.PhotoImage(result.thumbnail, master=self.window)
            self.thumbnail_label.configure(image=self.thumbnail)
            if (result.view == "translated" or self.record.get("untranslated")
                    or not self.record.get("exported", True)):
                self._remember_thumbnail(self.page_index, self.thumbnail)
        self._schedule_thumbnails()
        self._display_preview(result.preview, self.original.size)
        if action == "page":
            self.canvas.yview_moveto(0)
        self._canvas_resized()
        self.bubble_list.delete(*self.bubble_list.get_children())
        for key in self.blocks:
            self.bubble_list.insert("", "end", iid=key)
        self._refresh_rows()
        if self.block_id not in self.blocks:
            self.block_id = next(iter(self.blocks), None)
        if self.block_id:
            self.bubble_list.selection_set(self.block_id)
            self._show_bubble()
        self._update_page_navigation()
        view_name = "English translation" if self.view.get() == "英文译图" else "Original"
        status = f"{self.pages[self.page_index]['label']} · {view_name} · {len(self.blocks)} bubbles"
        if self.record.get("untranslated"):
            status = f"{self.pages[self.page_index]['label']} · Original only · Translate from Processing."
        elif self.record.get("status") == "partial":
            status += " · Details available in Workbench."
        uncovered = self.record.get("magi_coverage", {}).get("unmatched_texts", [])
        if uncovered:
            status += f" · Magi detected {len(uncovered)} additional text regions."
        if result.notice:
            status = result.notice + " · " + status
        self.status.set(status)

    def _receive_catalog(self, result: CatalogResult) -> None:
        if result.url != self.url.get().strip():
            return
        self.voices = result.voices
        self.voice_connected = bool(self.voices)
        self._set_voice_connecting(False)
        self.voice_box.configure(values=[f"{v['name']} · {v['style_id']}" for v in self.voices])
        self._choose_temporary_voice()
        self._show_voice()
        if self.voice_connected:
            self._voice_status_ready()
        else:
            self.voice_status.set("VOICEVOX · No voices are available; reading remains available.")
        pending = self.pending_audio
        self.pending_audio = None
        if pending and pending[0] == self.token:
            if self.voice_connected:
                self._request_audio(pending[1])
            else:
                self.status.set("No speech voices are available. Check the VOICEVOX settings.")

    def _play_result(self, result: JsonObject, token: Token) -> None:
        from audio.speech import play_wav

        try:
            play_wav(result["wav_path"])
            self.status.set("Playing Japanese audio." + (" (cached)" if result.get("cached") else ""))
            self.after_audio = self.window.after(max(1, round(result["duration_seconds"]*1000)+80),
                                                 lambda token=token: self._audio_finished(token))
        except Exception as exc:
            self._show_voice_error("Playback failed.", str(exc))

    def _finish_close(self) -> None:
        for value, trace in self._variable_traces:
            value.trace_remove("write", trace)
        self._variable_traces.clear()
        self.photo = self.thumbnail = None
        self.thumbnail_cache.clear()
        self.thumbnail_cards.clear()
        self.thumbnail_errors.clear()
        self.thumbnail_pending = None
        callback = self.on_closed
        self.on_closed = self.on_workbench = self.on_tasks = None
        self.on_terms = self.on_tools = None
        self.window.destroy()
        callback()
