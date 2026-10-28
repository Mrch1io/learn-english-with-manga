from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from core.runtime import Config
from ui.theme import apply_theme
from .build import BuildMixin
from .canvas import CanvasMixin
from .events import EventsMixin
from .images import preview_size
from .layout import LayoutMixin
from .messages import Command, Event, OpenRequest, PreviewRequest, ResizeRequest, TranslationRequest, Paths
from .pages import PagesMixin
from .records import linked_translation, same_scope
from .roles import RolesMixin
from .thumbnails import THUMBNAIL_ROW_HEIGHT, ThumbnailsMixin
from .voice import VoiceMixin
from .worker import ReaderWorker

FIT_MODES = ("Fit to Window", "适合窗口", "Fit to Width")


class ReaderWindow(BuildMixin, EventsMixin, ThumbnailsMixin, VoiceMixin, RolesMixin, CanvasMixin, PagesMixin, LayoutMixin):
    def __init__(self, parent: tk.Misc, paths: Paths | None = None, *, config: Config | None = None,
                 output_dir: str | Path | None = None, on_closed: Callable[[], None] | None = None,
                 on_workbench: Callable[[str | None], None] | None = None,
                 on_tasks: Callable[[], None] | None = None, on_terms: Callable[[], None] | None = None,
                 on_tools: Callable[[str, str | None], None] | None = None,
                 initial_path: str | Path | None = None) -> None:
        apply_theme(parent)
        self.window = tk.Toplevel(parent)
        self.window.title("Manga Translator · Reading")
        width = min(1250, max(1, self.window.winfo_screenwidth()-60))
        height = min(850, max(1, self.window.winfo_screenheight()-100))
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(min(900, width), min(650, height))
        # Windows 隐藏根窗口时，transient 子窗口也会无法显示。
        if parent.state() != "withdrawn":
            self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.on_closed = on_closed or (lambda: None)
        self.on_workbench = on_workbench
        self.on_tasks = on_tasks
        self.on_terms = on_terms
        self.on_tools = on_tools
        self.initial_path = str(Path(initial_path).resolve()) if initial_path else None
        self.commands: queue.Queue[Command] = queue.Queue()
        self.events: queue.Queue[Event] = queue.Queue()
        self.closing = threading.Event()
        self.operation_cancel = threading.Event()
        self.token = 0
        self.page_token = 0
        self.catalog_request = -1
        self.catalog_start_local = False
        self.pending_audio = None
        self.loading = False
        self.opening = False
        self.magi_busy = False
        self.translation_busy = False
        self.exporting = False
        self.config = replace(config or Config(), export_images=False)
        self.work = tk.StringVar(value=self.config.work)
        self.output_dir = tk.StringVar(value=str(output_dir or ""))
        self._disabled_widgets = []
        self.after_audio = None
        self.pages, self.record, self.blocks = [], {}, {}
        self.characters, self.bindings, self.voices = [], {}, []
        self.page_index, self.block_id = -1, None
        self.photo, self.scale = None, (1.0, 1.0)
        self.thumbnail = None
        self.thumbnail_cache = OrderedDict()
        self.thumbnail_errors = {}
        self.thumbnail_cards = {}
        self.thumbnail_visible = ()
        self.thumbnail_pending = None
        self.thumbnail_generation = 0
        self._thumbnail_request = 0
        self._thumbnail_versions = {}
        self.after_thumbnails = None
        self.original = None
        self.preview_token = 0
        self.after_resize = None
        self.resizing = False
        self._layout_resize = False
        self._pages_restore_zoom = None
        self.current_voice = None
        self.temporary_voice = False
        self.voice_connected = False
        self.voice_connecting = False
        self.url = tk.StringVar(value="http://127.0.0.1:50021")
        self.speed = tk.StringVar(value="1.0")
        self.zoom = tk.StringVar(value="Fit to Window")
        self.view = tk.StringVar(value="英文译图")
        self.role = tk.StringVar()
        self.voice = tk.StringVar()
        self.role_name = tk.StringVar()
        self.autoplay = tk.BooleanVar(value=True)
        self.show_regions = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Open a manga page. Click a speech bubble to hear its Japanese dialogue.")
        self.voice_status = tk.StringVar(value="VOICEVOX · Checking local service…")
        self.page_position = tk.StringVar(value="No pages")
        self.selected_bubble = tk.StringVar(value="Select a bubble on the page")
        self.display_title = tk.StringVar(value=self.work.get())
        self.bubble_notice = tk.StringVar()
        self.bubble_context = tk.StringVar()
        self._build()
        self._variable_traces = [
            (value, value.trace_add("write", callback)) for value, callback in (
                (self.speed, lambda *_: self._settings_changed()),
                (self.url, lambda *_: self._url_changed()),
                (self.work, lambda *_: self._update_display_title()),
                (self.view, lambda *_: self._update_view_button()),
            )
        ]
        # This is the main reading window; a grab blocks native Windows minimize commands.
        worker = ReaderWorker(self.commands, self.events, self.closing, self.operation_cancel)
        self.worker = threading.Thread(target=worker.run, name="manga-reader", daemon=False)
        self.worker.start()
        self.commands.put(Command("settings", None, self.token))
        self._poll = self.window.after(80, self._drain)
        if paths:
            self.open_paths(paths)

    def _invalidate(self):
        self.token += 1
        self.pending_audio = None
        if self.after_audio is not None:
            self.window.after_cancel(self.after_audio)
            self.after_audio = None
        from audio.speech import stop_playback

        try:
            stop_playback()
        except Exception as exc:
            self._show_voice_error("Could not stop audio.", str(exc))
        return self.token

    def stop(self):
        if self.magi_busy or self.translation_busy:
            self.operation_cancel.set()
            self.status.set("Cancelling image export. Saved translations will be preserved…" if self.exporting else
                            "Cancelling translation after the current safe step…" if self.translation_busy else
                            "Cancelling Magi after the current safe step…")
            return
        self._invalidate()
        self.status.set("Stopped. In-flight audio may finish caching but will not play.")

    def open_paths(self, paths):
        if self.magi_busy or self.translation_busy or self.closing.is_set():
            return
        first = Path(paths if isinstance(paths, (str, Path)) else paths[0]).resolve()
        if not self.output_dir.get():
            self.output_dir.set(str((first if first.is_dir() else first.parent) / "reader_output"))
        if self.work.get() == "default":
            self.work.set(first.name if first.is_dir() else first.parent.name)
        self._invalidate()
        self.page_token += 1
        self.thumbnail_generation += 1
        self.thumbnail_cache.clear()
        self.thumbnail_errors.clear()
        self._thumbnail_versions.clear()
        self.loading = True
        self.opening = True
        self._schedule_thumbnails()
        self._update_page_navigation()
        self.status.set("Loading pages…")
        self.commands.put(Command("open", OpenRequest(
            paths, self.output_dir.get(), replace(self.config, work=self.work.get().strip())), self.page_token))

    def _translate_page(self):
        if self.page_index < 0 or self.loading or self.magi_busy or self.translation_busy:
            return
        config = replace(self.config, work=self.work.get().strip(), export_images=False)
        matching_scope = bool(self.pages[self.page_index]["path"]) and same_scope(self.record, config)
        if matching_scope and (self.record.get("manual_edits") or self.record.get("ocr_candidate_edits")):
            self._export_page()
            return
        try:
            config.validate()
        except ValueError as exc:
            self.status.set(str(exc))
            return
        token = self._invalidate()
        self.operation_cancel.clear()
        self.translation_busy = True
        self._set_controls_busy(True)
        self.cancel_button.configure(state="normal")
        self.progress.start(15)
        self.status.set("Recognizing and translating this page. Loaded models will be reused.")
        page = self.pages[self.page_index]
        page["state"] = "translating"
        self._refresh_page_rows()
        self.commands.put(Command("translate", TranslationRequest(
            page["path"] if matching_scope else None, page["input"], page["root"],
            self.page_index, config, self.output_dir.get()), token))

    def _refresh_page_rows(self):
        labels = {"done": "", "partial": "", "skipped": "", "needs_review": "",
                  "failed": "Failed", "cancelled": "Cancelled", "untranslated": "Original only", "translating": "Translating"}
        self.page_list.delete(0, "end")
        for page in self.pages:
            state = labels.get(page["state"], page["state"])
            self.page_list.insert("end", page["label"] + (f" · {state}" if state else ""))
        if 0 <= self.page_index < len(self.pages):
            self.page_list.selection_set(self.page_index)
        self.page_thumbnails.configure(scrollregion=(0, 0, 148, len(self.pages) * THUMBNAIL_ROW_HEIGHT))
        self._schedule_thumbnails()
        self._update_page_navigation()

    def _load_page(self, zoom_only=False, *, notice=""):
        if self.opening or self.closing.is_set() or self.page_index < 0 or not self.pages:
            return
        self._invalidate()
        self.page_token += 1
        self.preview_token += 1
        self.resizing = False
        self.loading = True
        self._update_page_navigation()
        if not zoom_only:
            self._clear_page()
        if not self.pages[self.page_index]["path"]:
            self.view.set("原图")
        scale = 0 if self.zoom.get() in FIT_MODES else float(self.zoom.get().rstrip("%"))/100
        self.status.set("Loading English image…" if self.view.get() == "英文译图" else "Loading original image…")
        page = self.pages[self.page_index]
        self.commands.put(Command("zoom" if zoom_only else "page", PreviewRequest(
            page["path"], page["input"], scale, self._canvas_size(),
            view="translated" if self.view.get() == "英文译图" else "original",
            fit_width=self.zoom.get() == "Fit to Width", notice=notice), self.page_token))

    def _canvas_resized(self, _event=None):
        if self.closing.is_set() or (self.zoom.get() not in FIT_MODES and not self._layout_resize):
            return
        if self.after_resize is not None:
            self.window.after_cancel(self.after_resize)
        self.after_resize = self.window.after(150, self._resize_to_canvas)

    def _resize_to_canvas(self):
        self.after_resize = None
        if (self.closing.is_set() or self.loading or self.resizing
                or self.original is None or (self.zoom.get() not in FIT_MODES and not self._layout_resize)):
            return
        fit = self._canvas_size()
        scale = 0 if self.zoom.get() in FIT_MODES else float(self.zoom.get().rstrip("%"))/100
        fit_width = self.zoom.get() == "Fit to Width"
        self._layout_resize = False
        if self.photo is not None and (self.photo.width(), self.photo.height()) == preview_size(
                self.original.size, scale, fit, fit_width=fit_width):
            return
        # 缩放复用已校验的像素，不重读文件或中断语音。
        self.resizing = True
        self.commands.put(Command("resize", ResizeRequest(
            self.original, fit, scale, fit_width), self.preview_token))

    def _show_bubble(self):
        if self.block_id not in self.blocks:
            return
        block = self.blocks[self.block_id]
        if hasattr(self, "selected_bubble"):
            self.selected_bubble.set("Dialogue selected")
        metadata = block.get("metadata", {})
        unit_text = linked_translation(self.block_id, self.blocks)
        self.bubble_context.set("关联整句：" + unit_text if unit_text else "")
        notices = [block.get("error", ""), metadata.get("context_review", ""),
                   metadata.get("detection_review", ""), metadata.get("ocr_review", "")]
        conflicts = metadata.get("term_conflicts", [])
        if conflicts:
            notices.append("Terminology conflicts: " + "; ".join(conflicts))
        self.bubble_notice.set("\n".join(notice for notice in notices if notice))
        for widget, text in ((self.japanese, block.get("source_text", "")), (self.english, block.get("translated_text", ""))):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.configure(state="disabled")
        binding = self.bindings.get(self.block_id) or {}
        character_id = binding.get("character_id", "unknown")
        self._set_roles(character_id)
        self._draw_boxes()

    def close(self):
        if self.closing.is_set():
            return
        self._invalidate()
        if self.after_resize is not None:
            self.window.after_cancel(self.after_resize)
            self.after_resize = None
        if self.after_thumbnails is not None:
            self.window.after_cancel(self.after_thumbnails)
            self.after_thumbnails = None
        self.operation_cancel.set()
        self.closing.set()
        self.status.set("Closing after the current network operation finishes…")
        self.commands.put(Command("close", None, self.token))


def launch_reader(paths: Paths | None = None, *, config: Config | None = None,
                  output_dir: str | Path | None = None) -> None:
    root = tk.Tk()
    root.withdraw()
    ReaderWindow(root, paths, config=config, output_dir=output_dir, on_closed=root.destroy)
    root.mainloop()
