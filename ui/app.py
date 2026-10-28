"""Small Tk interface; model inference runs exclusively on one worker thread."""

from __future__ import annotations

import copy
from dataclasses import replace
import queue
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from core.runtime import ROOT
from .theme import PALETTE, apply_theme, style_text


IMAGE_TYPES = [("Manga images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All files", "*.*")]


def default_output(input_path: str) -> str:
    source = Path(input_path).expanduser()
    return str((source if source.is_dir() else source.parent) / "translated_output")


def parse_aliases(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(part.strip() for part in re.split(r"[,，\n]", value) if part.strip()))


def _run_worker(input_path: str, output_dir: str, config, events: queue.Queue,
                cancel_event: threading.Event) -> None:
    """No Tk objects cross this boundary, including progress callback closures."""
    try:
        from pipeline.batch import run_batch

        result = run_batch(input_path, output_dir, config,
                           progress=lambda event: events.put(("progress", dict(event))),
                           cancel_event=cancel_event)
        events.put(("finished", result))
    except (Exception, SystemExit) as exc:
        events.put(("failed", f"{type(exc).__name__}: {exc}"))


def _editor_snapshot(session, preview_size=(720, 700)) -> dict:
    """Read/resize pixels on the worker; only PhotoImage construction needs Tk."""
    from PIL import Image, ImageOps

    result = {"record": copy.deepcopy(session.record), "preview": None, "image_size": None,
              "original_preview": None, "original_size": None, "warning": "", "ocr_candidates": {}}
    list_candidates = getattr(session, "list_ocr_candidates", None)
    if list_candidates is not None:
        result["ocr_candidates"] = {
            block["block_id"]: list_candidates(block["block_id"])
            for block in result["record"].get("blocks", [])
            if block.get("metadata", {}).get("ocr_candidates")}
    try:
        with Image.open(session.record["output_path"]) as opened:
            result["image_size"] = opened.size
            preview = opened.convert("RGB")
        preview.thumbnail(preview_size, Image.Resampling.LANCZOS)
        result["preview"] = preview
    except (OSError, KeyError) as exc:
        if session.record.get("exported", True):
            result["warning"] = f"Cannot load the translated preview: {exc}"
    try:
        with Image.open(session.record["input_path"]) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
        result["original_size"] = original.size
        original.thumbnail(preview_size, Image.Resampling.LANCZOS)
        result["original_preview"] = original
    except (OSError, KeyError) as exc:
        result["warning"] += f" Original image unavailable: {exc}"
    return result


def _page_editor_worker(result_path: str, commands: queue.Queue, events: queue.Queue,
                        cancel_event: threading.Event, preview_size=(720, 700)) -> None:
    """One session and one worker own all page I/O, database and model access."""
    session = None
    try:
        from core.runtime import Cancelled
        from pipeline.editing import PageEditorSession

        def check_cancel():
            if cancel_event.is_set():
                raise Cancelled("Editing operation cancelled")

        session = PageEditorSession(result_path)
        events.put(("opened", _editor_snapshot(session, preview_size)))
        while True:
            action, block_id, text = commands.get()
            if action == "close":
                break
            try:
                check_cancel()
                if action in ("rerender", "replace_names"):
                    session.rerender(block_id, text, cancel=check_cancel)
                elif action == "retranslate":
                    session.retranslate(block_id, text, cancel=check_cancel)
                elif action == "reclean":
                    session.reclean(block_id, [text], cancel=check_cancel)
                elif action == "export":
                    session.export_image(cancel=check_cancel)
                elif action == "preview_ocr":
                    candidate = session.preview_ocr(block_id, text["crop_box"],
                                                    enabled=text["enabled"], cancel=check_cancel)
                elif action == "apply_ocr_candidate":
                    session.apply_ocr_candidate(block_id, text["candidate_id"],
                                                enabled=text["enabled"], cancel=check_cancel)
                elif action == "preview_region":
                    comparison = session.preview_region(block_id, text, cancel=check_cancel)
                    check_cancel()
                    events.put(("region_preview", comparison))
                    continue
                else:
                    raise ValueError(f"Unknown editor action: {action}")
                snapshot = _editor_snapshot(session, preview_size)
                snapshot.update(action=action, block_id=block_id)
                if action == "preview_ocr":
                    snapshot["ocr_candidate"] = candidate
                events.put(("ocr_preview" if action == "preview_ocr" else "saved", snapshot))
            except (Exception, SystemExit) as exc:
                events.put(("error", f"{type(exc).__name__}: {exc}"))
    except (Exception, SystemExit) as exc:
        events.put(("fatal", f"{type(exc).__name__}: {exc}"))
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                events.put(("close_error", f"Failed to release resources: {exc}"))
        events.put(("closed", None))


class TranslatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.running = False
        self.closing = False
        self.term_window: tk.Toplevel | None = None
        self.page_editor: PageEditor | None = None
        self.reader_window = None
        self.tool_window = None
        self.reader_paths: list[str] = []
        self.reader_config = None
        self.task_config = None
        self.reader_resume_path = None
        self.pending_view = None
        self.pending_term_config = None
        self.input_paths: list[str] | None = None
        self.last_result_paths: list[str] = []
        self.result_paths: dict[str, str] = {}
        self.inputs: list[ttk.Widget] = []

        apply_theme(root)
        root.title("Manga Translator · Tasks")
        root.geometry("1140x850")
        root.minsize(980, 700)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self.input_path = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value=str(ROOT / "translated_output"))
        self.work = tk.StringVar(value="default")
        self.language = tk.StringVar(value="English (en)")
        self.translator = tk.StringVar(value="VNTL 8B Q4_K_M")
        self.debug_mask = tk.BooleanVar(value=False)
        self.text_only = tk.BooleanVar(value=False)
        self.auto_read = tk.BooleanVar(value=True)
        self.ocr_candidate_review = tk.BooleanVar(value=False)
        self.use_hybrid_ocr = tk.BooleanVar(value=False)
        self.use_fragment_links = tk.BooleanVar(value=True)
        self.ocr_prob_threshold = tk.StringVar(value="0.1")
        self.status = tk.StringVar(value="Choose pages to translate, or open the daily-life sample. Originals are preserved.")
        self._build()
        self._input_path_trace = self.input_path.trace_add("write", lambda *_: setattr(self, "input_paths", None))
        self._poll_after = root.after(100, self._drain_events)

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)
        header = ttk.Frame(frame)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Translation Tasks", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Import pages, translate locally, then read or edit the results.", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        back = ttk.Button(header, text="Reading", style="Nav.TButton", command=self._resume_reading)
        back.grid(row=0, column=1, rowspan=2, padx=(16, 0))
        self.inputs.append(back)
        demo = ttk.Button(header, text="Open Sample", command=self._open_demo)
        demo.grid(row=0, column=2, rowspan=2, padx=(8, 0))
        self.inputs.append(demo)
        tools_button = ttk.Menubutton(header, text="Tools")
        tools_menu = tk.Menu(tools_button, tearoff=False, background=PALETTE["panel"],
                             foreground=PALETTE["text"], activebackground=PALETTE["selection"])
        tools_menu.add_command(label="Image Cleanup", command=lambda: self._open_tool("cleanup"))
        tools_menu.add_command(label="Revision Manager", command=lambda: self._open_tool("revisions"))
        tools_button.configure(menu=tools_menu)
        tools_button.grid(row=0, column=3, rowspan=2, padx=(8, 0))
        self.inputs.append(tools_button)

        source = ttk.LabelFrame(frame, text="Source & Output", padding=10)
        source.grid(row=1, column=0, sticky="ew")
        source.columnconfigure(1, weight=1)
        ttk.Label(source, text="Images / folder").grid(row=0, column=0, sticky="w", pady=6)
        self._entry(source, self.input_path, 0)
        choices = ttk.Frame(source)
        choices.grid(row=0, column=2, sticky="e", padx=(8, 0))
        for label, callback in (("Choose Images", self._choose_image), ("Choose Folder", self._choose_folder)):
            button = ttk.Button(choices, text=label, command=callback)
            button.pack(side="left", padx=2)
            self.inputs.append(button)
        ttk.Label(source, text="Output folder").grid(row=1, column=0, sticky="w", pady=6)
        self._entry(source, self.output_dir, 1)
        button = ttk.Button(source, text="Browse", command=self._choose_output)
        button.grid(row=1, column=2, sticky="ew", padx=(8, 0))
        self.inputs.append(button)
        ttk.Label(source, text="Work title").grid(row=2, column=0, sticky="w", pady=6)
        self._entry(source, self.work, 2)
        button = ttk.Button(source, text="Terms & Readings", command=self._open_terms)
        button.grid(row=2, column=2, sticky="ew", padx=(8, 0))
        self.inputs.append(button)
        ttk.Label(source, text="Target language").grid(row=3, column=0, sticky="w", pady=6)
        language_model = ttk.Frame(source)
        language_model.grid(row=3, column=1, sticky="w", padx=(10, 0))
        self.language_box = ttk.Combobox(language_model, textvariable=self.language,
                                         values=["English (en)"], state="readonly", width=12)
        self.language_box.pack(side="left")
        self.inputs.append(self.language_box)
        ttk.Label(language_model, text="Model").pack(side="left", padx=(12, 6))
        self.translator_box = ttk.Combobox(language_model, textvariable=self.translator,
            values=["VNTL 8B Q4_K_M", "Qwen 3B + LoRA (legacy)"], state="readonly", width=23)
        self.translator_box.pack(side="left")
        self.inputs.append(self.translator_box)
        check = ttk.Checkbutton(source, text="Save Debug Masks", variable=self.debug_mask)
        check.grid(row=3, column=2, sticky="w", padx=(8, 0))
        self.inputs.append(check)
        options = ttk.LabelFrame(frame, text="Processing Options", padding=8)
        options.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        text_check = ttk.Checkbutton(options, text="Text Only — export translated images later in Workbench", variable=self.text_only)
        text_check.pack(anchor="w")
        self.inputs.append(text_check)
        auto_check = ttk.Checkbutton(options, text="Open Reader When Finished", variable=self.auto_read)
        auto_check.pack(anchor="w", pady=(5, 0))
        self.inputs.append(auto_check)
        fragment_check = ttk.Checkbutton(options, text="Link Adjacent Sentence Fragments (keep original regions)",
                                         variable=self.use_fragment_links)
        fragment_check.pack(anchor="w", pady=(5, 0))
        self.inputs.append(fragment_check)
        audit_check = ttk.Checkbutton(options, text="Record Region Audit for Local OCR",
                                      variable=self.ocr_candidate_review)
        audit_check.pack(anchor="w", pady=(5, 0))
        self.inputs.append(audit_check)
        hybrid = ttk.Frame(options)
        hybrid.pack(anchor="w", pady=(5, 0))
        self.hybrid_check = ttk.Checkbutton(hybrid, text="Hybrid OCR — use Paddle for low-confidence text",
                                            variable=self.use_hybrid_ocr)
        self.hybrid_check.pack(side="left")
        ttk.Label(hybrid, text="Threshold").pack(side="left", padx=(8, 4))
        self.hybrid_threshold_entry = ttk.Entry(hybrid, textvariable=self.ocr_prob_threshold, width=5)
        self.hybrid_threshold_entry.pack(side="left")
        self.inputs.extend([self.hybrid_check, self.hybrid_threshold_entry])
        ttk.Label(options, text="Use a consistent work title to retain terminology, pronunciations and voices.", style="Muted.TLabel").pack(anchor="w", pady=(7, 0))
        action = ttk.Frame(frame)
        action.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.start_button = ttk.Button(action, text="Start Translation", style="Accent.TButton", command=self._start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(action, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        self.edit_button = ttk.Button(action, text="Open Workbench", command=self._open_page_editor)
        self.edit_button.pack(side="left")
        self.inputs.append(self.edit_button)
        self.read_button = ttk.Button(action, text="Open in Reader", command=self._open_reader)
        self.read_button.pack(side="left", padx=(6, 0))
        self.inputs.append(self.read_button)
        self.progress = ttk.Progressbar(action, mode="determinate", maximum=1)
        self.progress.pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Label(frame, textvariable=self.status, wraplength=840).grid(row=4, column=0, sticky="w", pady=8)
        tabs = ttk.Notebook(frame)
        tabs.grid(row=5, column=0, sticky="nsew")
        results = ttk.Frame(tabs, padding=8)
        log_frame = ttk.Frame(tabs, padding=8)
        tabs.add(results, text="Page Results")
        tabs.add(log_frame, text="Activity Log")
        toolbar = ttk.Frame(results)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Label(toolbar, text="Double-click a result to read it. Processing errors stay visible.").pack(side="left")
        existing = ttk.Button(toolbar, text="Open Saved Results…", command=self._choose_results)
        existing.pack(side="right")
        self.inputs.append(existing)
        self.result_tree = ttk.Treeview(results, columns=("page", "status", "message"), show="headings", height=7)
        for name, title, width in (("page", "Page", 190), ("status", "State", 85), ("message", "Details", 500)):
            self.result_tree.heading(name, text=title)
            self.result_tree.column(name, width=width, stretch=name != "status")
        result_scroll = ttk.Scrollbar(results, orient="vertical", command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=result_scroll.set)
        result_scroll.pack(side="right", fill="y")
        self.result_tree.pack(fill="both", expand=True)
        self.result_tree.bind("<Double-1>", self._open_selected_result)
        self.result_tree.bind("<Return>", self._open_selected_result)
        self.log = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        style_text(self.log)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

    def _entry(self, parent, variable, row: int) -> None:
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=6)
        self.inputs.append(entry)

    def _choose_image(self) -> None:
        paths = filedialog.askopenfilenames(parent=self.root, title="Choose manga images", filetypes=IMAGE_TYPES)
        if paths:
            self.input_path.set(paths[0] if len(paths) == 1 else f"{len(paths)} images selected: {Path(paths[0]).name} …")
            self.input_paths = list(paths)
            self.output_dir.set(default_output(paths[0]))

    def _choose_folder(self) -> None:
        path = filedialog.askdirectory(parent=self.root, title="Choose a manga folder")
        if path:
            self.input_path.set(path)
            self.output_dir.set(default_output(path))
            self.work.set(Path(path).name)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(parent=self.root, title="Choose an output folder", mustexist=False)
        if path:
            self.output_dir.set(path)

    def _config(self):
        from core.runtime import Config

        return replace(self.task_config or Config(), work=self.work.get().strip() or "default", target_language="en",
                      translation_backend="vntl" if self.translator.get() == "VNTL 8B Q4_K_M" else "qwen",
                      debug_mask=self.debug_mask.get(), export_images=not self.text_only.get(),
                      ocr_candidate_review=self.ocr_candidate_review.get(),
                      use_fragment_links=self.use_fragment_links.get(),
                      use_hybrid_ocr=self.use_hybrid_ocr.get(),
                      ocr_prob_threshold=float(self.ocr_prob_threshold.get()))

    def _set_running(self, value: bool) -> None:
        self.running = value
        locked = (value or self.page_editor is not None or self.reader_window is not None
                  or self.tool_window is not None)
        for widget in self.inputs:
            widget.configure(state="disabled" if locked else "normal")
        if not locked:
            self.language_box.configure(state="readonly")
            self.translator_box.configure(state="readonly")
        self.start_button.configure(state="disabled" if locked else "normal")
        self.cancel_button.configure(state="normal" if value else "disabled")

    def _start(self) -> None:
        if (self.running or self.page_editor is not None or self.reader_window is not None
                or self.tool_window is not None):
            return
        source_text = self.input_paths or self.input_path.get().strip()
        destination = self.output_dir.get().strip()
        selected = source_text if isinstance(source_text, list) else [source_text]
        if not source_text or not all(Path(path).expanduser().exists() for path in selected):
            messagebox.showerror("Cannot start", "Choose an existing image or folder.", parent=self.root)
            return
        if not destination:
            messagebox.showerror("Cannot start", "Choose an output folder.", parent=self.root)
            return
        try:
            config = self._config()
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc), parent=self.root)
            return
        if self.term_window is not None and self.term_window.winfo_exists():
            self.term_window.destroy()
        self.cancel_event.clear()
        self.last_result_paths = []
        self.result_paths.clear()
        self.result_tree.delete(*self.result_tree.get_children())
        self.progress.configure(value=0, maximum=1)
        self.status.set("Preparing models and tasks. The first load may take a moment.")
        self._append_log(f"Source: {source_text}\nOutput: {destination}")
        self._set_running(True)
        self.worker = threading.Thread(
            target=_run_worker, args=(source_text, destination, config, self.events, self.cancel_event),
            name="manga-translation", daemon=False)
        self.worker.start()

    def _cancel(self) -> None:
        if self.running:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self.status.set("Cancelling after the current safe step. Completed results will be preserved.")

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 1000:
            self.log.delete("1.0", "201.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain_events(self) -> None:
        # A bounded drain keeps Tk responsive if a producer emits many events.
        for _ in range(100):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                current, total = payload.get("current", 0), payload.get("total", 0)
                event_type = payload.get("type", "")
                message = payload.get("message") or {
                    "page_start": "Processing", "page_done": "Completed", "error": "Page failed",
                    "skipped": "Existing result reused", "stage": "Processing", "complete": "Batch finished",
                }.get(event_type, event_type)
                path = payload.get("path", "")
                line = f"[{current}/{total}] {message}"
                if path:
                    line += f" · {Path(path).name}"
                if not self.cancel_event.is_set():
                    self.status.set(line)
                if total:
                    completed = max(0, current - 1) if event_type in ("page_start", "stage") else current
                    self.progress.configure(maximum=total, value=min(completed, total))
                self._append_log(line)
            elif kind in ("finished", "failed"):
                self._set_running(False)
                if kind == "failed":
                    self.status.set("Task failed. Details are available in the log.")
                    self._append_log(f"Task failed: {payload}")
                    if not self.closing:
                        messagebox.showerror("Task failed", payload, parent=self.root)
                else:
                    self.status.set("Cancelled. Completed results have been preserved." if self.cancel_event.is_set()
                                    else "Finished. Double-click a result to read it, or open Workbench to edit.")
                    if isinstance(payload, dict):
                        self._show_results(payload)
                        counts = [(key, payload[key]) for key in ("completed", "partial", "failed", "skipped", "cancelled")
                                  if key in payload and isinstance(payload[key], (int, bool))]
                        if counts:
                            labels = {"completed": "Completed", "partial": "Partial", "failed": "Failed", "skipped": "Skipped", "cancelled": "Cancelled"}
                            self._append_log(", ".join(f"{labels[key]}: {value}" for key, value in counts))
                if self.closing:
                    self._destroy()
                    return
                if (kind == "finished" and self.auto_read.get() and self.last_result_paths
                        and not self.cancel_event.is_set() and not payload.get("cancelled")):
                    self._open_reader(self.last_result_paths)
        self._poll_after = self.root.after(100, self._drain_events)

    def _show_results(self, report: dict) -> None:
        self.result_tree.delete(*self.result_tree.get_children())
        self.result_paths.clear()
        self.last_result_paths = []
        labels = {"done": "Done", "partial": "Partial", "skipped": "Reused", "needs_review": "Edits preserved",
                  "failed": "Failed", "cancelled": "Cancelled"}
        for index, page in enumerate(report.get("pages", [])):
            key = f"page-{index}"
            status = page.get("status", "")
            self.result_tree.insert("", "end", iid=key, values=(Path(page.get("input_path", "")).name,
                labels.get(status, status), page.get("error", page.get("message", ""))))
            if page.get("output_path") and status in {"done", "partial", "skipped", "needs_review"}:
                path = str(Path(page["output_path"]).with_suffix(".json"))
                self.result_paths[key] = path
                self.last_result_paths.append(path)

    def _open_selected_result(self, _event=None) -> None:
        selected = self.result_tree.selection()
        if selected and selected[0] in self.result_paths:
            self._open_reader([self.result_paths[selected[0]]])

    def _choose_results(self) -> None:
        paths = filedialog.askopenfilenames(parent=self.root, title="Open saved images or page records",
            initialdir=self.output_dir.get(), filetypes=[("Translated image / page record", "*.png *.json")])
        if paths:
            self._open_reader(list(paths))

    def _open_demo(self) -> None:
        from presentation import ensure_demo_pages

        try:
            paths = ensure_demo_pages()
        except (OSError, ValueError) as exc:
            self._show_tasks()
            messagebox.showerror("Sample unavailable", str(exc), parent=self.root)
        else:
            self._open_reader([str(path) for path in paths])

    def start_reading(self) -> None:
        self._open_demo()

    def _show_tasks(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self._set_running(self.running)

    def _resume_reading(self) -> None:
        if self.reader_paths:
            self._open_reader(self.reader_paths, config=self.reader_config,
                              initial_path=self.reader_resume_path)
        else:
            self.start_reading()

    def _switch_from_reader(self, destination: str, path=None) -> None:
        """Queue navigation; the next session must wait for the worker's close event."""
        reader = self.reader_window
        if reader is None or self.pending_view is not None:
            return
        pages = reader.pages
        self.reader_paths = [page.get("path") or page["input"] for page in pages]
        self.reader_config = reader.config
        self.task_config = reader.config
        self.work.set(reader.work.get())
        if reader.output_dir.get().strip():
            self.output_dir.set(reader.output_dir.get().strip())
        self.translator.set("VNTL 8B Q4_K_M" if reader.config.translation_backend == "vntl"
                            else "Qwen 3B + LoRA (legacy)")
        self.debug_mask.set(reader.config.debug_mask)
        self.ocr_candidate_review.set(reader.config.ocr_candidate_review)
        self.use_hybrid_ocr.set(reader.config.use_hybrid_ocr)
        self.use_fragment_links.set(reader.config.use_fragment_links)
        self.ocr_prob_threshold.set(str(reader.config.ocr_prob_threshold))
        if 0 <= reader.page_index < len(pages):
            current = pages[reader.page_index]
            self.reader_resume_path = current.get("path") or current["input"]
            self.input_path.set(current["input"])
            self.input_paths = [page["input"] for page in pages]
        if destination == "terms":
            self.pending_term_config = reader.config
        self.pending_view = ("tasks" if destination == "workbench" and not path else destination, path)
        reader.close()

    def _switch_from_editor(self, destination: str) -> None:
        if self.page_editor is None or self.pending_view is not None:
            return
        path = self.page_editor.result_path if destination in ("cleanup", "revisions") else None
        if self.page_editor.request_close():
            self.pending_view = (destination, path)

    def _finish_navigation(self) -> None:
        destination, path = self.pending_view or ("tasks", None)
        self.pending_view = None
        if destination == "workbench":
            self._open_page_editor(path)
        elif destination == "reading":
            self._resume_reading()
        elif destination in ("cleanup", "revisions"):
            self._open_tool(destination, path)
        else:
            self._show_tasks()
            if destination == "terms":
                config, self.pending_term_config = self.pending_term_config, None
                self._open_terms(config=config)

    def _destroy(self) -> None:
        self.root.after_cancel(self._poll_after)
        if self._input_path_trace is not None:
            self.input_path.trace_remove("write", self._input_path_trace)
            self._input_path_trace = None
        self.root.destroy()

    def _close(self) -> None:
        if self.reader_window is not None:
            self.closing = True
            self.reader_window.close()
        elif self.page_editor is not None:
            if self.page_editor.request_close():
                self.closing = True
                self.status.set("Closing the editing session safely…")
        elif self.tool_window is not None:
            if self.tool_window.request_close():
                self.closing = True
                self.status.set("Closing the tool safely…")
        elif self.running:
            self.closing = True
            self._cancel()
            self.status.set("Finishing the current safe step before closing. Completed results will be preserved.")
        else:
            self._destroy()

    def _open_terms(self, *, config=None) -> None:
        if (self.running or self.page_editor is not None or self.reader_window is not None
                or self.tool_window is not None):
            return
        if self.term_window is not None and self.term_window.winfo_exists():
            self.term_window.lift()
            return
        try:
            config = config or self._config()
            editor = TermEditor(self.root, config)
            self.term_window = editor.window
        except Exception as exc:
            messagebox.showerror("Terminology memory", str(exc), parent=self.root)

    def _open_page_editor(self, path=None) -> None:
        if (self.running or self.page_editor is not None or self.reader_window is not None
                or self.tool_window is not None):
            return
        if path is None:
            path = filedialog.askopenfilename(
                parent=self.root, title="Open a translated page or its JSON record",
                initialdir=self.output_dir.get(),
                filetypes=[("Translated image / page record", "*.png *.json")])
        if not path:
            return
        if self.term_window is not None and self.term_window.winfo_exists():
            self.term_window.destroy()
        if not any(Path(saved).with_suffix(".json").resolve() == Path(path).with_suffix(".json").resolve()
                   for saved in self.reader_paths):
            self.reader_paths = [str(path)]
            self.reader_config = None
        self.reader_resume_path = str(path)
        self.root.withdraw()
        try:
            self.page_editor = PageEditor(self.root, path, on_closed=self._editor_closed,
                on_read=lambda: self._switch_from_editor("reading"),
                on_tasks=lambda: self._switch_from_editor("tasks"),
                on_tools=self._switch_from_editor)
        except Exception:
            self._show_tasks()
            raise
        self.page_editor.window.protocol("WM_DELETE_WINDOW", self._close)
        self._set_running(False)

    def _session_closed(self) -> None:
        if self.closing:
            self._destroy()
        else:
            self._set_running(False)
            self._finish_navigation()

    def _editor_closed(self) -> None:
        self.page_editor = None
        self._session_closed()

    def _open_reader(self, paths=None, *, config=None, initial_path=None) -> None:
        if (self.running or self.page_editor is not None or self.reader_window is not None
                or self.tool_window is not None):
            return
        from reader import ReaderWindow

        if self.term_window is not None and self.term_window.winfo_exists():
            self.term_window.destroy()
        if paths is None:
            paths = self.input_paths or self.input_path.get().strip() or self.last_result_paths
            if not paths:
                paths = filedialog.askdirectory(parent=self.root, title="Choose a manga or results folder")
                if not paths:
                    return
        self.root.withdraw()
        try:
            self.reader_window = ReaderWindow(self.root, paths=paths, config=config or self._config(),
                output_dir=self.output_dir.get().strip(), on_closed=self._reader_closed,
                on_workbench=lambda path: self._switch_from_reader("workbench", path),
                on_tasks=lambda: self._switch_from_reader("tasks"),
                on_terms=lambda: self._switch_from_reader("terms"),
                on_tools=self._switch_from_reader, initial_path=initial_path)
        except Exception:
            self._show_tasks()
            raise
        self.reader_window.window.protocol("WM_DELETE_WINDOW", self._close)
        self._set_running(False)

    def _reader_closed(self) -> None:
        self.reader_window = None
        self._session_closed()

    def _open_tool(self, destination: str, path=None) -> None:
        if (self.running or self.page_editor is not None or self.reader_window is not None
                or self.tool_window is not None):
            return
        from .tools import CleanupWindow, RevisionManagerWindow

        if destination == "cleanup":
            path = path or filedialog.askopenfilename(
                parent=self.root, title="Open a page for image cleanup",
                initialdir=self.output_dir.get(), filetypes=[("Page record / translated image", "*.json *.png")])
            if not path:
                self._show_tasks()
                return
            factory = CleanupWindow
            if not any(Path(saved).with_suffix(".json").resolve() == Path(path).with_suffix(".json").resolve()
                       for saved in self.reader_paths):
                self.reader_paths = [str(path)]
                self.reader_config = None
            self.reader_resume_path = str(path)
        elif destination == "revisions":
            path = Path(path).parent if path and Path(path).suffix.lower() in (".json", ".png") else path
            path = path or self.output_dir.get()
            factory = RevisionManagerWindow
        else:
            raise ValueError(f"Unknown desktop tool: {destination}")
        if self.term_window is not None and self.term_window.winfo_exists():
            self.term_window.destroy()
        self.root.withdraw()
        callbacks = {"on_closed": self._tool_closed,
                     "on_read": lambda: self._switch_from_tool("reading")}
        if destination == "revisions":
            callbacks["on_edit"] = lambda record: self._switch_from_tool("workbench", record)
        try:
            self.tool_window = factory(self.root, path, **callbacks)
        except Exception:
            self._show_tasks()
            raise
        self.tool_window.window.protocol("WM_DELETE_WINDOW", self._close)
        self._set_running(False)

    def _switch_from_tool(self, destination: str, path=None) -> None:
        if self.tool_window is None or self.pending_view is not None:
            return
        if self.tool_window.request_close():
            self.pending_view = (destination, path)

    def _tool_closed(self) -> None:
        self.tool_window = None
        self._session_closed()


class PageEditor:
    """草稿仅在用户保存后写入页面。"""
    def __init__(self, parent: tk.Tk, result_path: str, *, on_closed,
                 on_read=None, on_tasks=None, on_tools=None) -> None:
        apply_theme(parent)
        self.window = tk.Toplevel(parent)
        self.window.title(f"Manga Translator · Workbench · {Path(result_path).stem}")
        self.window.geometry("1250x850")
        self.window.minsize(1000, 700)
        if parent.state() != "withdrawn":
            self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.request_close)
        self.on_closed = on_closed
        self.on_read = on_read
        self.on_tasks = on_tasks
        self.on_tools = on_tools
        self.result_path = str(result_path)
        self.commands: queue.Queue = queue.Queue()
        self.events: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.record: dict = {}
        self.blocks: dict[str, dict] = {}
        self.drafts: dict[str, tuple[str, str]] = {}
        self.selected_id: str | None = None
        self.busy = True
        self.closing = False
        self.photo = None
        self.thumbnail_photo = None
        self.image_scale = (1.0, 1.0)
        self.snapshot = {}
        self.show_original = tk.BooleanVar(value=False)
        self.ocr_enabled = tk.BooleanVar(value=False)
        self.ocr_coordinates = [tk.StringVar() for _ in range(4)]
        self.ocr_audit = tk.StringVar(value="No box-relation audit is saved. Enter local OCR crop coordinates manually.")
        self.ocr_crop_choice = tk.StringVar()
        self.ocr_crop_choices: dict[str, list[int]] = {}
        self.ocr_candidate: dict | None = None
        self.saved_ocr_choice = tk.StringVar()
        self.saved_ocr_candidates: dict[str, dict] = {}
        self.ocr_candidate_status = tk.StringVar()
        self.ocr_selected_id: str | None = None
        self.ocr_updating = False
        self.comparison_window = None
        self.comparison_photos = []
        self.linked_sentence = tk.StringVar(value="No linked sentence for this region.")
        self.notice = tk.StringVar(value="Loading the page and its saved text…")
        self._build()
        self._ocr_coordinate_traces = [
            (value, value.trace_add("write", self._ocr_crop_changed))
            for value in self.ocr_coordinates
        ]
        self._set_busy(True)
        self.window.grab_set()
        self.worker = None
        self.window.after_idle(self._start_worker)
        self.window.after(80, self._drain_events)

    def _start_worker(self):
        # A new Toplevel can still report a 1x1 canvas before its first Map event.
        if self.closing:
            self.events.put(("closed", None))
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            self.window.after(20, self._start_worker)
            return
        preview_size = (max(120, width-28), max(120, height-28))
        self.worker = threading.Thread(target=_page_editor_worker,
                                       args=(self.result_path, self.commands, self.events, self.cancel_event, preview_size),
                                       name="manga-page-editor", daemon=False)
        self.worker.start()

    def _build(self) -> None:
        frame = ttk.Frame(self.window, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, minsize=165)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, minsize=375)
        frame.rowconfigure(2, weight=1)
        header = ttk.Frame(frame)
        header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Manga Translator", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        modes = ttk.Frame(header)
        modes.grid(row=0, column=1)
        ttk.Button(modes, text="Reading", style="Nav.TButton", command=self.on_read or self.request_close).pack(side="left")
        active = ttk.Button(modes, text="Workbench", style="Nav.TButton")
        active.state(["selected"])
        active.pack(side="left", padx=6)
        ttk.Button(header, text="Tasks", command=self.on_tasks or self.request_close).grid(row=0, column=2, padx=(16, 0))
        if self.on_tools is not None:
            tools = ttk.Menubutton(header, text="Tools")
            tools.grid(row=0, column=3, padx=(8, 0))
            menu = tk.Menu(tools, tearoff=False, background=PALETTE["panel"],
                           foreground=PALETTE["text"], activebackground=PALETTE["selection"])
            menu.add_command(label="Image Cleanup", command=lambda: self.on_tools("cleanup"))
            menu.add_command(label="Revision Manager", command=lambda: self.on_tools("revisions"))
            tools.configure(menu=menu)

        toolbar = ttk.Frame(frame)
        toolbar.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        ttk.Label(toolbar, text=Path(self.result_path).stem, style="Muted.TLabel").pack(side="left")
        self.export_button = ttk.Button(toolbar, text="Export Image", style="Accent.TButton", command=lambda: self._save("export"))
        self.export_button.pack(side="right")
        self.compare_button = ttk.Button(toolbar, text="Compare Region", command=self._preview_region)
        self.compare_button.pack(side="right", padx=8)
        ttk.Checkbutton(toolbar, text="Show Original", variable=self.show_original,
                        command=self._display_preview).pack(side="right", padx=12)

        pages = ttk.Frame(frame, padding=(0, 4, 12, 0))
        pages.grid(row=2, column=0, sticky="nsew")
        pages.columnconfigure(0, weight=1)
        pages.rowconfigure(3, weight=1)
        ttk.Label(pages, text="CURRENT PAGE", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.thumbnail_label = ttk.Label(pages, anchor="center")
        self.thumbnail_label.grid(row=1, column=0, sticky="ew", pady=(10, 18))
        ttk.Label(pages, text="Text Regions", style="Heading.TLabel").grid(row=2, column=0, sticky="w", pady=(0, 8))
        self.table = ttk.Treeview(pages, columns=("id", "status"), show="headings", height=9,
                                  selectmode="browse")
        self.table.heading("id", text="Region / draft*")
        self.table.heading("status", text="State")
        self.table.column("id", width=105, minwidth=70)
        self.table.column("status", width=60, minwidth=45)
        self.table.grid(row=3, column=0, sticky="nsew")
        region_scroll = ttk.Scrollbar(pages, orient="vertical", command=self.table.yview)
        region_scroll.grid(row=3, column=1, sticky="ns")
        self.table.configure(yscrollcommand=region_scroll.set)
        self.table.bind("<<TreeviewSelect>>", self._select)

        preview = ttk.Frame(frame)
        preview.grid(row=2, column=1, sticky="nsew", padx=(0, 16))
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(preview, background=PALETTE["canvas"], highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(preview, orient="vertical", command=self.canvas.yview)
        horizontal = ttk.Scrollbar(preview, orient="horizontal", command=self.canvas.xview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.canvas.bind("<Button-1>", self._click_preview)
        self.canvas.bind("<Configure>", self._center_preview)
        panel = ttk.Frame(frame)
        panel.grid(row=2, column=2, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        ttk.Label(panel, text="Text Inspector", style="Heading.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 12))
        self.tabs = tabs = ttk.Notebook(panel)
        tabs.grid(row=1, column=0, sticky="nsew")
        edit = ttk.Frame(tabs, padding=10)
        edit.columnconfigure(0, weight=1)
        candidate_panel = ttk.Frame(tabs, padding=8)
        candidate_panel.columnconfigure(0, weight=1)
        context = ttk.Frame(tabs, padding=12)
        tabs.add(edit, text="Text")
        tabs.add(candidate_panel, text="Local OCR")
        tabs.add(context, text="Context")
        ttk.Label(edit, text="Japanese source").grid(row=1, column=0, sticky="w", pady=(8, 6))
        self.source_text = tk.Text(edit, height=5, width=30, wrap="word", undo=True)
        style_text(self.source_text)
        self.source_text.grid(row=2, column=0, sticky="ew")
        ttk.Label(edit, text="English translation").grid(row=3, column=0, sticky="w", pady=(18, 6))
        self.translated_text = tk.Text(edit, height=6, width=30, wrap="word", undo=True)
        style_text(self.translated_text)
        self.translated_text.grid(row=4, column=0, sticky="ew")
        self.render_button = ttk.Button(edit, text="Save Translation & Redraw", style="Accent.TButton", command=lambda: self._save("rerender"))
        self.render_button.grid(row=5, column=0, sticky="ew", pady=(18, 4))
        self.translate_button = ttk.Button(edit, text="Save Source & Retranslate", command=lambda: self._save("retranslate"))
        self.translate_button.grid(row=6, column=0, sticky="ew", pady=4)
        self.replace_button = ttk.Button(edit, text="Replace Name on Page", command=self._replace_names)
        self.replace_button.grid(row=7, column=0, sticky="ew", pady=4)
        ttk.Label(edit, text="Drafts stay when you switch regions.\nAn empty translation clears the original text.",
                  style="Muted.TLabel", wraplength=310).grid(row=8, column=0, sticky="w", pady=10)
        self._build_ocr_panel(candidate_panel)
        ttk.Label(context, text="Linked Sentence", style="Heading.TLabel").pack(anchor="w", pady=(0, 12))
        ttk.Label(context, textvariable=self.linked_sentence, wraplength=320,
                  justify="left").pack(anchor="w", fill="x")
        ttk.Label(context, text="Built from the current saved text of linked regions.\nUnsaved drafts are not included.",
                  style="Muted.TLabel", wraplength=320).pack(anchor="w", pady=16)
        footer = ttk.Frame(frame)
        footer.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.notice, wraplength=930, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.cancel_button = ttk.Button(footer, text="Cancel Task", command=self._cancel)
        self.cancel_button.grid(row=0, column=1, padx=(12, 0))

    def _build_ocr_panel(self, panel) -> None:
        self.ocr_toggle = ttk.Checkbutton(panel, text="Enable Local OCR", variable=self.ocr_enabled,
                                          command=self._toggle_ocr)
        self.ocr_toggle.grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(panel, text="The green region is used for OCR only. Image Cleanup is a separate tool.",
                  wraplength=340).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(panel, textvariable=self.ocr_audit, wraplength=340).grid(row=2, column=0, sticky="w", pady=2)
        self.ocr_crop_box = ttk.Combobox(panel, textvariable=self.ocr_crop_choice, state="disabled")
        self.ocr_crop_box.grid(row=3, column=0, sticky="ew", pady=2)
        self.ocr_crop_box.bind("<<ComboboxSelected>>", self._choose_ocr_crop)
        coordinates = ttk.Frame(panel)
        coordinates.grid(row=4, column=0, sticky="ew", pady=2)
        self.ocr_entries = []
        for column, (name, variable) in enumerate(zip(("x1", "y1", "x2", "y2"), self.ocr_coordinates)):
            ttk.Label(coordinates, text=name).grid(row=0, column=column)
            entry = ttk.Entry(coordinates, width=9, textvariable=variable)
            entry.grid(row=1, column=column, padx=(0, 4))
            self.ocr_entries.append(entry)
        self.ocr_preview_button = ttk.Button(panel, text="Preview OCR Candidate", command=self._preview_ocr)
        self.ocr_preview_button.grid(row=5, column=0, sticky="ew", pady=2)
        saved = ttk.Frame(panel)
        saved.grid(row=6, column=0, sticky="ew", pady=2)
        saved.columnconfigure(1, weight=1)
        ttk.Label(saved, text="Saved Candidates").grid(row=0, column=0, padx=(0, 4))
        self.saved_ocr_box = ttk.Combobox(saved, textvariable=self.saved_ocr_choice, state="disabled", width=23)
        self.saved_ocr_box.grid(row=0, column=1, sticky="ew")
        self.saved_ocr_box.bind("<<ComboboxSelected>>", self._choose_saved_ocr)
        self.ocr_candidate_text = tk.Text(panel, height=4, width=30, wrap="word", state="disabled")
        style_text(self.ocr_candidate_text)
        self.ocr_candidate_text.grid(row=7, column=0, sticky="ew")
        ttk.Label(panel, textvariable=self.ocr_candidate_status, wraplength=340).grid(row=8, column=0, sticky="w", pady=2)
        self.ocr_apply_button = ttk.Button(panel, text="Apply Candidate & Retranslate", command=self._apply_ocr_candidate)
        self.ocr_apply_button.grid(row=9, column=0, sticky="ew", pady=2)

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        disabled = busy or self.closing or self.selected_id is None
        for widget in (self.source_text, self.translated_text, self.render_button, self.translate_button, self.replace_button):
            widget.configure(state="disabled" if disabled else "normal")
        self.table.state(["disabled" if busy or self.closing else "!disabled"])
        self.cancel_button.configure(state="normal" if busy and not self.closing else "disabled")
        self.export_button.configure(state="disabled" if busy or self.closing or not self.record else "normal")
        self.compare_button.configure(state="disabled" if disabled else "normal")
        self.ocr_toggle.configure(state="disabled" if busy or self.closing else "normal")
        ocr_disabled = disabled or not self.ocr_enabled.get()
        for widget in (*self.ocr_entries, self.ocr_preview_button):
            widget.configure(state="disabled" if ocr_disabled else "normal")
        self.ocr_crop_box.configure(state="disabled" if ocr_disabled else "readonly")
        self.saved_ocr_box.configure(state="disabled" if ocr_disabled or not self.saved_ocr_candidates else "readonly")
        self.ocr_apply_button.configure(state="normal" if not ocr_disabled and self._can_apply_ocr() else "disabled")

    def _remember_draft(self) -> None:
        if self.selected_id is not None:
            self.drafts[self.selected_id] = (self.source_text.get("1.0", "end-1c"),
                                             self.translated_text.get("1.0", "end-1c"))
            self._update_row(self.selected_id)

    def _is_dirty(self, block_id: str) -> bool:
        block = self.blocks[block_id]
        return self.drafts.get(block_id) != (block.get("source_text", ""), block.get("translated_text", ""))

    def _update_row(self, block_id: str) -> None:
        if self.table.exists(block_id):
            label = block_id.rsplit(":", 1)[-1] + (" *" if self._is_dirty(block_id) else "")
            status = self.blocks[block_id].get("status", "")
            self.table.item(block_id, values=(label, {"translated": "Saved", "failed": "Error"}.get(status, status)))

    def _select(self, _event=None) -> None:
        selected = self.table.selection()
        if self.busy or self.closing or not selected or selected[0] not in self.blocks:
            return
        if selected[0] == self.selected_id:
            return
        self._remember_draft()
        self.selected_id = selected[0]
        self._show_selected()

    def _show_selected(self) -> None:
        if self.selected_id is None:
            return
        from reader.records import linked_translation

        self.linked_sentence.set(linked_translation(self.selected_id, self.blocks)
                                 or "No linked sentence for this region.")
        source, translated = self.drafts[self.selected_id]
        for widget, value in ((self.source_text, source), (self.translated_text, translated)):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", value)
            widget.edit_reset()
        if self.ocr_selected_id != self.selected_id:
            self._load_ocr_block()
        self._set_busy(self.busy)
        self._draw_boxes()

    def _load_ocr_block(self) -> None:
        self.ocr_selected_id = self.selected_id
        self._clear_ocr_candidate()
        block = self.blocks[self.selected_id]
        box = list(block["box"])
        self.ocr_crop_choices = {"Current bubble": box}
        labels = {"covered": "Covered", "new": "Uncovered", "expansion_candidate": "Expansion candidate", "conflict": "Overlap conflict"}
        audit = block.get("metadata", {}).get("ocr_box_audit", [])
        summary: dict[str, int] = {}
        for index, item in enumerate(audit, 1):
            label = labels.get(item.get("classification"), item.get("classification", "Unclassified"))
            crop = item.get("crop_box")
            if crop and len(crop) == 4:
                choice = f"{index}. {label} {','.join(map(str, crop))}"
                self.ocr_crop_choices[choice] = list(crop)
                summary[label] = summary.get(label, 0) + 1
        self.ocr_audit.set("Box relations: " + ", ".join(f"{label} {count}" for label, count in summary.items()) if summary else
                           "No box-relation audit is saved. Enter local OCR crop coordinates manually.")
        self.ocr_crop_box.configure(values=list(self.ocr_crop_choices))
        self.ocr_crop_choice.set("Current bubble")
        self._set_ocr_crop(box)
        self._refresh_saved_ocr()

    def _refresh_saved_ocr(self) -> None:
        candidates = self.snapshot.get("ocr_candidates", {}).get(self.selected_id, [])
        self.saved_ocr_candidates = {}
        for index, candidate in enumerate(candidates, 1):
            state = "Available" if candidate.get("usable") is True else "Unavailable"
            label = f"{index}. {state} {candidate.get('created_at', '')}"
            self.saved_ocr_candidates[label] = candidate
        self.saved_ocr_box.configure(values=list(self.saved_ocr_candidates))
        self.saved_ocr_choice.set("")

    def _show_saved_ocr(self, candidate_id) -> None:
        match = next(((label, candidate) for label, candidate in self.saved_ocr_candidates.items()
                      if candidate.get("candidate_id") == candidate_id), None)
        if match is None:
            self._clear_ocr_candidate()
            return
        label, candidate = match
        crop = candidate.get("crop_box")
        if isinstance(crop, (list, tuple)) and len(crop) == 4 and all(type(value) is int for value in crop):
            self._set_ocr_crop(crop)
        else:
            self._clear_ocr_candidate()
        self.saved_ocr_choice.set(label)
        self.ocr_candidate = copy.deepcopy(candidate)
        self.ocr_candidate_text.configure(state="normal")
        self.ocr_candidate_text.insert("1.0", candidate.get("text", ""))
        self.ocr_candidate_text.configure(state="disabled")
        self.ocr_candidate_status.set("Saved candidate; check the text before applying." if candidate.get("usable") is True else
                                      candidate.get("unavailable_reason") or "This candidate cannot be applied.")
        self._set_busy(self.busy)

    def _choose_saved_ocr(self, _event=None) -> None:
        if self.busy or self.closing or not self.ocr_enabled.get():
            return
        candidate = self.saved_ocr_candidates.get(self.saved_ocr_choice.get())
        if candidate is not None:
            self._show_saved_ocr(candidate.get("candidate_id"))

    def _set_ocr_crop(self, box) -> None:
        self.ocr_updating = True
        try:
            for variable, value in zip(self.ocr_coordinates, box):
                variable.set(str(value))
        finally:
            self.ocr_updating = False
        self._ocr_crop_changed()

    def _choose_ocr_crop(self, _event=None) -> None:
        if not self.busy and not self.closing and self.ocr_enabled.get():
            crop = self.ocr_crop_choices.get(self.ocr_crop_choice.get())
            if crop:
                self._set_ocr_crop(crop)

    def _read_ocr_crop(self) -> list[int]:
        try:
            box = [int(value.get()) for value in self.ocr_coordinates]
        except ValueError as exc:
            raise ValueError("OCR crop coordinates must be integers.") from exc
        size = self.snapshot.get("original_size")
        if not size:
            raise ValueError("Local OCR requires the original image.")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
            raise ValueError("The OCR crop must be a nonempty rectangle inside the original image.")
        return box

    def _clear_ocr_candidate(self) -> None:
        self.ocr_candidate = None
        self.saved_ocr_choice.set("")
        self.ocr_candidate_status.set("")
        self.ocr_candidate_text.configure(state="normal")
        self.ocr_candidate_text.delete("1.0", "end")
        self.ocr_candidate_text.configure(state="disabled")

    def _ocr_crop_changed(self, *_args) -> None:
        if self.ocr_updating:
            return
        self._clear_ocr_candidate()
        self._draw_ocr_crop()
        self._set_busy(self.busy)

    def _toggle_ocr(self) -> None:
        self._clear_ocr_candidate()
        if self.ocr_enabled.get():
            self.show_original.set(True)
            self._display_preview()
        else:
            self._draw_ocr_crop()
        self._set_busy(self.busy)

    def _draw_ocr_crop(self) -> None:
        self.canvas.delete("ocr_crop")
        if not self.ocr_enabled.get() or not self.show_original.get() or self.photo is None:
            return
        try:
            x1, y1, x2, y2 = self._read_ocr_crop()
        except ValueError:
            return
        sx, sy = self.image_scale
        self.canvas.create_rectangle(x1*sx, y1*sy, x2*sx, y2*sy, outline="#50e69d", width=3,
                                     dash=(6, 3), tags="ocr_crop")

    def _can_apply_ocr(self) -> bool:
        candidate = self.ocr_candidate
        if (not candidate or candidate.get("usable") is not True or candidate.get("accepted")
                or candidate.get("status") != "ok" or not candidate.get("text", "").strip()):
            return False
        try:
            return (self.ocr_selected_id == self.selected_id and candidate.get("crop_box") == self._read_ocr_crop()
                    and candidate.get("base_source_text") == self.blocks[self.selected_id].get("source_text", ""))
        except (ValueError, KeyError):
            return False

    def _preview_ocr(self) -> None:
        if self.busy or self.closing or self.selected_id is None or not self.ocr_enabled.get():
            return
        try:
            crop = self._read_ocr_crop()
        except ValueError as exc:
            self.notice.set(str(exc))
            return
        self._remember_draft()
        self._clear_ocr_candidate()
        self.cancel_event.clear()
        self._set_busy(True)
        self.notice.set("Running local OCR on the selected crop; source text and drafts are retained…")
        self.commands.put(("preview_ocr", self.selected_id, {"crop_box": crop, "enabled": True}))

    def _apply_ocr_candidate(self) -> None:
        if self.busy or self.closing or not self.ocr_enabled.get() or not self._can_apply_ocr():
            return
        self._remember_draft()
        if self._is_dirty(self.selected_id):
            self.notice.set("This bubble has unsaved drafts. Save or restore them before applying an OCR candidate.")
            return
        self.cancel_event.clear()
        self._set_busy(True)
        self.notice.set("Applying the OCR candidate and retranslating this bubble only; cleanup bounds are unchanged…")
        self.commands.put(("apply_ocr_candidate", self.selected_id,
                           {"candidate_id": self.ocr_candidate["candidate_id"], "enabled": True}))

    def _preview_region(self) -> None:
        if self.busy or self.closing or self.selected_id is None:
            return
        try:
            crop = self._read_ocr_crop() if self.ocr_enabled.get() else None
        except ValueError as exc:
            self.notice.set(str(exc))
            return
        self._remember_draft()
        self.cancel_event.clear()
        self._set_busy(True)
        self.notice.set("Reading matching crops from the original and saved translation…")
        self.commands.put(("preview_region", self.selected_id, crop))

    def _close_region_preview(self) -> None:
        if self.comparison_window is not None and self.comparison_window.winfo_exists():
            self.comparison_window.destroy()
        self.comparison_window = None
        self.comparison_photos = []
        if not self.closing and self.window.winfo_exists():
            self.window.grab_set()

    def _show_region_preview(self, payload: dict) -> bool:
        if self.closing or payload.get("block_id") != self.selected_id or self.cancel_event.is_set():
            return False
        from PIL import Image, ImageTk

        self._close_region_preview()
        window = tk.Toplevel(self.window)
        self.comparison_window = window
        window.title(f"Region Comparison · {payload['block_id']} · {payload['crop_box']}")
        window.transient(self.window)
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", self._close_region_preview)
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)
        width, height = payload["original"].size
        scale = min(1.0, max(1, window.winfo_screenwidth() - 100) / (2 * width),
                    max(1, window.winfo_screenheight() - 180) / height)
        size = max(1, int(width * scale)), max(1, int(height * scale))
        self.comparison_captions = ("Original (read-only)", "Saved translation (excluding drafts)" if payload["translated"] is not None
                                    else "No exported translation; right side is empty")
        for column, (caption, image) in enumerate(zip(self.comparison_captions,
                                                     (payload["original"], payload["translated"]))):
            ttk.Label(frame, text=caption).grid(row=0, column=column, sticky="w", padx=5, pady=(0, 5))
            canvas = tk.Canvas(frame, width=size[0], height=size[1], background="#252525", highlightthickness=0)
            canvas.grid(row=1, column=column, padx=5, sticky="nw")
            if image is not None:
                if image.size != size:
                    image = image.resize(size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image, master=window)
                self.comparison_photos.append(photo)
                canvas.create_image(0, 0, image=photo, anchor="nw")
        ttk.Label(frame, text="Matching crop and scale. Read-only comparison; source text, candidates, and images are unchanged.",
                  wraplength=max(220, size[0] * 2)).grid(row=2, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Button(frame, text="Close Comparison", command=self._close_region_preview).grid(row=3, column=0, columnspan=2)
        window.grab_set()
        return True

    def _draw_boxes(self) -> None:
        self.canvas.delete("bubble")
        if self.photo is None:
            return
        sx, sy = self.image_scale
        for block_id, block in self.blocks.items():
            x1, y1, x2, y2 = block["box"]
            selected = block_id == self.selected_id
            color = PALETTE["accent_hover"] if selected else PALETTE["accent"]
            self.canvas.create_rectangle(x1*sx, y1*sy, x2*sx, y2*sy, outline=color,
                                         width=3 if selected else 1, tags="bubble")
        self._draw_ocr_crop()

    def _click_preview(self, event) -> None:
        if self.busy or self.closing or self.photo is None:
            return
        sx, sy = self.image_scale
        x, y = self.canvas.canvasx(event.x)/sx, self.canvas.canvasy(event.y)/sy
        matches = [(block_id, block["box"]) for block_id, block in self.blocks.items()
                   if block["box"][0] <= x <= block["box"][2] and block["box"][1] <= y <= block["box"][3]]
        if matches:
            block_id = min(matches, key=lambda item: (item[1][2]-item[1][0])*(item[1][3]-item[1][1]))[0]
            self.table.selection_set(block_id)
            self.table.see(block_id)
            self._select()

    def _replace_names(self) -> None:
        if self.busy or self.closing:
            return
        self._remember_draft()
        if any(self._is_dirty(key) for key in self.blocks):
            self.notice.set("Save your drafts before replacing a name on this page.")
            return
        old = (simpledialog.askstring("Replace Name on Page", "Old name:", parent=self.window) or "").strip()
        if not old:
            return
        new = (simpledialog.askstring("Replace Name on Page", "New name:", parent=self.window) or "").strip()
        if not new or old == new:
            return
        pattern = re.compile(r"(?<!\w)" + re.escape(old) + r"(?!\w)")
        updates = {key: pattern.sub(lambda _: new, block["translated_text"])
                   for key, block in self.blocks.items() if block.get("status") == "translated"
                   and pattern.search(block.get("translated_text", ""))}
        if not updates:
            self.notice.set("No matching name on this page.")
            return
        self.cancel_event.clear()
        self._set_busy(True)
        self.notice.set("Replacing the name on this page…")
        self.commands.put(("replace_names", next(iter(updates)), updates))

    def _save(self, action: str) -> None:
        if self.busy or self.closing or (self.selected_id is None and action != "export"):
            return
        self._remember_draft()
        source, translated = self.drafts.get(self.selected_id, ("", ""))
        if action == "retranslate" and not source.strip():
            self.notice.set("The source text cannot be empty. Add source text before retranslating; drafts are retained.")
            return
        self.cancel_event.clear()
        self._set_busy(True)
        self.notice.set({"rerender": "Saving the translation…", "export": "Exporting saved text; unsaved drafts are excluded…",
                         "retranslate": "Saving the source and retranslating this bubble; cancellation waits for a safe step…"}[action])
        self.commands.put((action, self.selected_id, translated if action == "rerender" else source))

    def _cancel(self) -> None:
        if self.busy:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self.notice.set("Cancelling after the current safe step. Unsaved drafts are retained.")

    def _apply_snapshot(self, payload: dict) -> None:
        previous_candidate_id = self.ocr_candidate.get("candidate_id") if self.ocr_candidate else None
        self.record = payload["record"]
        self.blocks = {block["block_id"]: block for block in self.record.get("blocks", [])}
        if payload.get("action") == "replace_names":
            self.drafts.clear()
        for block_id, block in self.blocks.items():
            self.drafts.setdefault(block_id, (block.get("source_text", ""), block.get("translated_text", "")))
        saved_id = payload.get("block_id")
        if saved_id in self.blocks and payload.get("action") in ("rerender", "retranslate", "apply_ocr_candidate"):
            block = self.blocks[saved_id]
            source = self.drafts[saved_id][0] if payload["action"] == "rerender" else block.get("source_text", "")
            self.drafts[saved_id] = (source, block.get("translated_text", ""))
        self.table.delete(*self.table.get_children())
        for block_id in self.blocks:
            self.table.insert("", "end", iid=block_id)
            self._update_row(block_id)
        if self.selected_id not in self.blocks:
            self.selected_id = next(iter(self.blocks), None)
        if self.selected_id is not None:
            self.table.selection_set(self.selected_id)
            self.table.see(self.selected_id)
        self.snapshot = payload
        if payload.get("preview") is None and payload.get("original_preview") is not None:
            self.show_original.set(True)
        self._display_preview()
        self._set_busy(False)
        self._show_selected()
        self._refresh_saved_ocr()
        if payload.get("action") == "apply_ocr_candidate":
            self._clear_ocr_candidate()
        elif payload.get("action") == "preview_ocr" and saved_id == self.selected_id:
            self._show_saved_ocr(payload["ocr_candidate"]["candidate_id"])
        elif previous_candidate_id is not None:
            self._show_saved_ocr(previous_candidate_id)
        self._set_busy(False)
        message = ("Saved. Source drafts and drafts in other bubbles still need to be saved separately." if payload.get("action")
                   else "Select a bubble to edit. Text-only results can be exported when ready.")
        if payload.get("action") == "preview_ocr":
            candidate = payload["ocr_candidate"]
            message = ("OCR candidate saved. Source text and drafts are unchanged." if candidate.get("status") == "ok"
                       else "OCR returned no usable candidate: " + (candidate.get("error") or candidate.get("status", "failed")))
        self.notice.set(" ".join(filter(None, (message, self.record.get("editor_notice"), payload.get("warning")))))

    def _display_preview(self):
        original = self.show_original.get()
        preview = self.snapshot.get("original_preview" if original else "preview")
        self.canvas.delete("all")
        self.photo = None
        if preview is not None:
            from PIL import ImageTk

            self.photo = ImageTk.PhotoImage(preview, master=self.window)
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
            width, height = self.snapshot["original_size" if original else "image_size"]
            self.image_scale = (preview.width/width, preview.height/height)
            self._center_preview()
            thumbnail = preview.copy()
            thumbnail.thumbnail((122, 155))
            self.thumbnail_photo = ImageTk.PhotoImage(thumbnail, master=self.window)
            self.thumbnail_label.configure(image=self.thumbnail_photo)
        self._draw_boxes()
        self._set_busy(self.busy)

    def _center_preview(self, _event=None):
        """Center with canvas coordinates, keeping hit testing and cleanup boxes exact."""
        if self.photo is None:
            return
        width, height = self.photo.width(), self.photo.height()
        padding_x = max(0, (self.canvas.winfo_width()-width)/2)
        padding_y = max(0, (self.canvas.winfo_height()-height)/2)
        self.canvas.configure(scrollregion=(-padding_x, -padding_y, width+padding_x, height+padding_y))
        if padding_x:
            self.canvas.xview_moveto(0)
        if padding_y:
            self.canvas.yview_moveto(0)

    def _drain_events(self) -> None:
        for _ in range(30):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind in ("opened", "saved", "ocr_preview"):
                if not self.closing:
                    self._apply_snapshot(payload)
            elif kind == "region_preview":
                if not self.closing:
                    self._set_busy(False)
                    if self._show_region_preview(payload):
                        self.notice.set("Comparison opened with the original and saved translation. Drafts are unchanged.")
                    else:
                        self.notice.set("Ignored a cancelled or outdated region comparison.")
            elif kind == "error":
                if not self.closing:
                    self._set_busy(False)
                    self.notice.set(f"Operation failed; drafts are retained. {payload}")
            elif kind in ("fatal", "close_error"):
                messagebox.showerror("Page Workbench", payload, parent=self.window)
            elif kind == "closed":
                if self.worker is not None and self.worker.is_alive():
                    self.events.put(("closed", None))
                    break
                if self.window.grab_current() == self.window:
                    self.window.grab_release()
                # 在 Tk 主线程释放回调和图片，再销毁窗口。
                for value, trace in self._ocr_coordinate_traces:
                    value.trace_remove("write", trace)
                self._ocr_coordinate_traces.clear()
                self.photo = None
                self.thumbnail_photo = None
                self.comparison_photos.clear()
                self.window.destroy()
                self.on_closed()
                return
        self.window.after(80, self._drain_events)

    def request_close(self) -> bool:
        if self.closing:
            return True
        self._remember_draft()
        if any(self._is_dirty(block_id) for block_id in self.blocks):
            if not messagebox.askyesno("Discard unsaved drafts?", "This page has unsaved corrections. Close and discard those drafts?",
                                       parent=self.window):
                return False
        self.closing = True
        self._close_region_preview()
        self.cancel_event.set()
        self._set_busy(self.busy)
        self.notice.set("Closing safely after the current step and releasing models…")
        self.commands.put(("close", None, None))
        return True


class TermEditor:
    """Short-lived SQLite connections keep edits scoped and thread-local."""
    def __init__(self, parent: tk.Tk, config) -> None:
        self.config = config
        apply_theme(parent)
        self.window = tk.Toplevel(parent)
        self.window.title(f"Terms & Readings · {config.work} · English")
        self.window.geometry("860x590")
        self.window.minsize(680, 460)
        self.source = tk.StringVar()
        self.target = tk.StringVar()
        self.aliases = tk.StringVar()
        self.reading = tk.StringVar()
        self.first_page = tk.StringVar()
        self.locked = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="candidate")
        self.notice = tk.StringVar(value="Candidate terms can be reused. Locked terms are protected from automatic updates.")
        self.terms: dict[str, dict] = {}
        self._build()
        try:
            self._refresh()
        except Exception:
            self.window.destroy()
            raise

    def _memory(self):
        from core.memory import GlossaryMemory

        return GlossaryMemory(self.config.memory_db, self.config.work, self.config.target_language)

    def _build(self) -> None:
        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)
        table_frame = ttk.Frame(frame)
        table_frame.grid(row=0, column=0, columnspan=4, sticky="nsew", pady=(0, 10))
        columns = ("source", "target", "aliases", "reading", "locked", "status", "first_page")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse", height=10)
        for column, heading, width in zip(columns,
                ("Source", "Translation", "Aliases", "Reading", "Locked", "State", "First Page"), (100, 120, 110, 110, 55, 70, 120)):
            self.table.heading(column, text=heading)
            self.table.column(column, width=width, minwidth=45)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.table.pack(side="left", fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self._selected)
        for row, label, variable in ((1, "Source", self.source), (2, "Translation", self.target),
                                     (3, "Aliases (comma-separated)", self.aliases), (4, "First Page", self.first_page),
                                     (5, "Japanese Reading (optional)", self.reading)):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(frame, textvariable=variable)
            entry.grid(row=row, column=1, columnspan=3, sticky="ew", padx=(10, 0), pady=4)
            if row == 1:
                self.source_entry = entry
        options = ttk.Frame(frame)
        options.grid(row=6, column=0, columnspan=4, sticky="ew", pady=6)
        ttk.Checkbutton(options, text="Lock Translation", variable=self.locked).pack(side="left")
        ttk.Label(options, text="State").pack(side="left", padx=(20, 6))
        ttk.Combobox(options, textvariable=self.status, values=["candidate", "confirmed"],
                     state="readonly", width=15).pack(side="left")
        actions = ttk.Frame(frame)
        actions.grid(row=7, column=0, columnspan=4, sticky="w", pady=6)
        for label, command in (("New / Clear", self._clear), ("Save", self._save),
                               ("Delete", self._delete), ("Import JSON", self._import), ("Export JSON", self._export)):
            ttk.Button(actions, text=label, command=command).pack(side="left", padx=(0, 6))
        ttk.Label(frame, textvariable=self.notice, wraplength=800).grid(
            row=8, column=0, columnspan=4, sticky="w", pady=8)

    def _refresh(self) -> None:
        with self._memory() as memory:
            if self.config.target_language == "en" and (ROOT / "glossary.json").exists():
                memory.import_legacy_once(ROOT / "glossary.json")
            terms = memory.list_terms()
        self.table.delete(*self.table.get_children())
        self.terms.clear()
        for index, term in enumerate(terms):
            item_id = str(index)
            self.terms[item_id] = term
            aliases = term.get("aliases") or []
            self.table.insert("", "end", iid=item_id, values=(
                term["source"], term["target"], ", ".join(aliases), term.get("reading") or "",
                "Yes" if term.get("locked") else "No", term.get("status", "candidate"), term.get("first_page", "")))

    def _selected(self, _event=None) -> None:
        selected = self.table.selection()
        if not selected or selected[0] not in self.terms:
            return
        term = self.terms[selected[0]]
        self.source.set(term["source"])
        self.source_entry.configure(state="readonly")
        self.target.set(term["target"])
        self.aliases.set(", ".join(term.get("aliases") or []))
        self.reading.set(term.get("reading") or "")
        self.first_page.set(term.get("first_page", ""))
        self.locked.set(bool(term.get("locked")))
        self.status.set(term.get("status", "candidate"))

    def _clear(self) -> None:
        self.table.selection_remove(*self.table.selection())
        for variable in (self.source, self.target, self.aliases, self.first_page, self.reading):
            variable.set("")
        self.source_entry.configure(state="normal")
        self.locked.set(False)
        self.status.set("candidate")

    def _save(self) -> None:
        source, target = self.source.get().strip(), self.target.get().strip()
        if not source or not target:
            messagebox.showerror("Cannot save", "Source and preferred translation cannot be empty.", parent=self.window)
            return
        try:
            with self._memory() as memory:
                affected = set(memory.affected_pages(source))
                memory.upsert_term(source, target, aliases=parse_aliases(self.aliases.get()),
                                   first_page=self.first_page.get().strip(), locked=self.locked.get(),
                                   status=self.status.get(), user_edit=True, reading=self.reading.get().strip())
                affected.update(memory.affected_pages(source))
            self._refresh()
            self._clear()
            self.notice.set(f"Saved. {len(affected)} recorded pages use this term; the next run checks affected results.")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.window)

    def _delete(self) -> None:
        source = self.source.get().strip()
        if not source:
            return
        try:
            with self._memory() as memory:
                affected = len(memory.affected_pages(source))
                memory.delete_term(source, user_edit=True)
            self._refresh()
            self._clear()
            self.notice.set(f"Removed. {affected} recorded pages are affected; the next run checks their results.")
        except Exception as exc:
            messagebox.showerror("Removal failed", str(exc), parent=self.window)

    def _import(self) -> None:
        path = filedialog.askopenfilename(parent=self.window, title="Import terminology",
                                          filetypes=[("Terminology JSON", "*.json")])
        if not path:
            return
        try:
            with self._memory() as memory:
                count = memory.import_json(path)
            self._refresh()
            self._clear()
            self.notice.set(f"Processed {count} terms. The next run checks affected results; locked terms stay protected.")
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc), parent=self.window)

    def _export(self) -> None:
        path = filedialog.asksaveasfilename(parent=self.window, title="Export terminology", defaultextension=".json",
                                            initialfile="glossary-export.json", filetypes=[("Terminology JSON", "*.json")])
        if not path:
            return
        try:
            with self._memory() as memory:
                memory.export_json(path)
            self.notice.set(f"Exported: {path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self.window)


def launch_gui() -> None:
    root = tk.Tk()
    root.withdraw()
    app = TranslatorApp(root)
    root.after_idle(app.start_reading)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
