r"""Separate Tk utilities for saved-page cleanup and protected revision inspection.

Run with the project environment:
    .venv\Scripts\python.exe -m ui.tools cleanup [page.json]
    .venv\Scripts\python.exe -m ui.tools revisions [output_folder]
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .theme import PALETTE, apply_theme


def cleanup_rectangle(start, end, scale, bubble):
    """Convert a canvas drag into a nonempty, bubble-contained source rectangle."""
    sx, sy = scale
    if not (math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0):
        raise ValueError("Preview scale must be positive and finite.")
    region = (math.floor(min(start[0], end[0]) / sx), math.floor(min(start[1], end[1]) / sy),
              math.ceil(max(start[0], end[0]) / sx), math.ceil(max(start[1], end[1]) / sy))
    if not (bubble[0] <= region[0] < region[2] <= bubble[2]
            and bubble[1] <= region[1] < region[3] <= bubble[3]):
        raise ValueError("Select leftover text inside the selected bubble, away from borders and artwork.")
    return region


def _record_time(value, fallback):
    """Prefer recorded UTC timestamps; file times are a legacy fallback only."""
    try:
        parsed = datetime.fromisoformat(value)
        return int(parsed.timestamp() * 1_000_000_000) if parsed.tzinfo else fallback
    except (TypeError, ValueError, OverflowError, OSError):
        return fallback


def _record_path(value, directory):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else directory / path).resolve())


def collect_revisions(folder, cancel=None):
    """只读扫描修订；批次备注是历史记录，不能当作当前校验结果。"""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("Choose an existing output folder.")
    records, notes, errors = [], {}, []
    for path in folder.rglob("*.json"):
        if cancel is not None and cancel.is_set():
            return [], []
        if path.is_symlink():
            continue
        try:
            stamp = path.stat().st_mtime_ns
            record = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if not isinstance(record, dict):
            continue
        if path.name.startswith("run-") and isinstance(record.get("pages"), list):
            stamp = _record_time(record.get("started_at"), stamp)
            for page in record["pages"]:
                if not isinstance(page, dict) or not isinstance(page.get("output_path"), str) or not page["output_path"]:
                    continue
                key = _record_path(page["output_path"], path.parent)
                if stamp >= notes.get(key, (0, None))[0]:
                    notes[key] = (stamp, page)
            continue
        if not (isinstance(record.get("input_path"), str) and record["input_path"]
                and isinstance(record.get("output_path"), str) and record["output_path"]
                and isinstance(record.get("blocks"), list)
                and (record.get("manual_edits") or record.get("ocr_candidate_edits"))):
            continue
        protection = []
        if record.get("manual_edits"):
            protection.append("Manual edits")
        if record.get("ocr_candidate_edits"):
            protection.append("OCR candidates")
        last_edit = record.get("last_edit")
        last_edit = last_edit if isinstance(last_edit, dict) else {}
        records.append({"path": str(path), "output": _record_path(record["output_path"], path.parent),
                        "page": Path(record["input_path"]).name, "work": str(record.get("work", "")),
                        "protection": " + ".join(protection), "saved_ns": _record_time(last_edit.get("at"), stamp),
                        "last_edit": last_edit, "note": "No matching batch note."})
    for record in records:
        batch_stamp, page = notes.get(record["output"], (0, None))
        if page is None:
            continue
        if page.get("status") == "needs_review":
            note = str(page.get("message") or "The last batch retained this page after an input, settings, or terminology change.")
            if record["saved_ns"] > batch_stamp:
                note = "Saved after the last batch note. Previous note: " + note
        else:
            note = "Last batch: " + str(page.get("status", "unknown")) + ". " + str(page.get("message", ""))
        record["note"] = note
    records.sort(key=lambda record: (record["work"].casefold(), record["page"].casefold(), record["path"]))
    return records, errors


class CleanupWindow:
    """One saved-page session, with all image/model work on the existing worker."""
    def __init__(self, parent, path, *, on_closed, on_read=None):
        apply_theme(parent)
        self.window = tk.Toplevel(parent)
        self.window.title("Manga Translator · Image Cleanup")
        self.window.geometry("1050x800")
        self.window.minsize(760, 560)
        if parent.state() != "withdrawn":
            self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.request_close)
        self.on_closed, self.on_read = on_closed, on_read
        self.result_path = str(path)
        self.commands, self.events = queue.Queue(), queue.Queue()
        self.cancel_event = threading.Event()
        self.busy, self.closing = True, False
        self.worker = None
        self.blocks, self.record, self.snapshot = {}, {}, {}
        self.selected_id, self.region_start, self.region, self.photo = None, None, None, None
        self.image_scale = (1.0, 1.0)
        self.show_original = tk.BooleanVar(master=self.window, value=True)
        self.notice = tk.StringVar(master=self.window, value="Loading the saved page…")
        self._build()
        self._set_busy(True)
        self.window.grab_set()
        self.window.after_idle(self._start_worker)
        self.window.after(60, self._drain_events)

    def _build(self):
        frame = ttk.Frame(self.window, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)
        header = ttk.Frame(frame)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Image Cleanup", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Reading" if self.on_read else "Close", command=self.on_read or self.request_close).grid(row=0, column=1)
        ttk.Label(frame, text="Select a bubble, then drag over leftover text inside it. Saved translations are retained.",
                  style="Muted.TLabel", wraplength=900).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))
        sidebar = ttk.Frame(frame, padding=(0, 0, 14, 0))
        sidebar.grid(row=2, column=0, sticky="ns")
        ttk.Label(sidebar, text="Text Regions", style="Heading.TLabel").pack(anchor="w", pady=(0, 8))
        self.table = ttk.Treeview(sidebar, columns=("id",), show="headings", selectmode="browse", height=16)
        self.table.heading("id", text="Region")
        self.table.column("id", width=155, minwidth=110)
        self.table.pack(fill="y", expand=True)
        self.table.bind("<<TreeviewSelect>>", self._select)
        self.original_toggle = ttk.Checkbutton(sidebar, text="Show Original", variable=self.show_original, command=self._display_preview)
        self.original_toggle.pack(anchor="w", pady=12)
        self.reclean_button = ttk.Button(sidebar, text="Clean Selection & Export", style="Accent.TButton", command=self._reclean)
        self.reclean_button.pack(fill="x")
        preview = ttk.Frame(frame)
        preview.grid(row=2, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(preview, background=PALETTE["canvas"], highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(preview, orient="vertical", command=self.canvas.yview)
        horizontal = ttk.Scrollbar(preview, orient="horizontal", command=self.canvas.xview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.canvas.bind("<ButtonPress-1>", self._start_region)
        self.canvas.bind("<B1-Motion>", self._drag_region)
        self.canvas.bind("<ButtonRelease-1>", self._finish_region)
        self.canvas.bind("<Configure>", self._center_preview)
        footer = ttk.Frame(frame)
        footer.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.notice, style="Muted.TLabel", wraplength=760).grid(row=0, column=0, sticky="w")
        self.cancel_button = ttk.Button(footer, text="Cancel Task", command=self._cancel)
        self.cancel_button.grid(row=0, column=1, padx=(12, 0))

    def _start_worker(self):
        if self.closing:
            self.events.put(("closed", None))
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            self.window.after(20, self._start_worker)
            return
        from .app import _page_editor_worker

        self.worker = threading.Thread(target=_page_editor_worker,
            args=(self.result_path, self.commands, self.events, self.cancel_event, (max(120, width-28), max(120, height-28))),
            name="manga-image-cleanup", daemon=False)
        self.worker.start()

    def _set_busy(self, busy):
        self.busy = busy
        disabled = busy or self.closing
        self.table.state(["disabled" if disabled else "!disabled"])
        self.original_toggle.configure(state="disabled" if disabled else "normal")
        self.reclean_button.configure(state="normal" if not disabled and self.region is not None else "disabled")
        self.cancel_button.configure(state="normal" if busy and not self.closing else "disabled")

    def _select(self, _event=None):
        selected = self.table.selection()
        if self.busy or self.closing or not selected or selected[0] == self.selected_id:
            return
        self.selected_id = selected[0]
        self.region = None
        self.canvas.delete("selection")
        self._draw_boxes()
        self._set_busy(False)

    def _display_preview(self):
        original = self.show_original.get()
        image = self.snapshot.get("original_preview" if original else "preview")
        self.canvas.delete("all")
        self.region, self.photo = None, None
        if image is not None:
            from PIL import ImageTk

            self.photo = ImageTk.PhotoImage(image, master=self.window)
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
            width, height = self.snapshot["original_size" if original else "image_size"]
            self.image_scale = (image.width / width, image.height / height)
            self._center_preview()
            self._draw_boxes()
        self._set_busy(self.busy)

    def _center_preview(self, _event=None):
        if self.photo is None:
            return
        width, height = self.photo.width(), self.photo.height()
        px, py = max(0, (self.canvas.winfo_width()-width)/2), max(0, (self.canvas.winfo_height()-height)/2)
        self.canvas.configure(scrollregion=(-px, -py, width+px, height+py))
        if px:
            self.canvas.xview_moveto(0)
        if py:
            self.canvas.yview_moveto(0)

    def _draw_boxes(self):
        self.canvas.delete("bubble")
        if self.photo is None:
            return
        sx, sy = self.image_scale
        for key, block in self.blocks.items():
            x1, y1, x2, y2 = block["box"]
            self.canvas.create_rectangle(x1*sx, y1*sy, x2*sx, y2*sy,
                outline=PALETTE["accent"], width=3 if key == self.selected_id else 1, tags="bubble")

    def _start_region(self, event):
        if self.busy or self.closing or self.photo is None:
            return
        start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        x, y = start[0]/self.image_scale[0], start[1]/self.image_scale[1]
        matches = [(key, block["box"]) for key, block in self.blocks.items()
                   if block["box"][0] <= x <= block["box"][2] and block["box"][1] <= y <= block["box"][3]]
        if matches:
            key = min(matches, key=lambda item: (item[1][2]-item[1][0])*(item[1][3]-item[1][1]))[0]
            self.table.selection_set(key)
            self._select()
        self.region_start, self.region = start, None
        self.canvas.delete("selection")
        self._set_busy(False)

    def _drag_region(self, event):
        if self.region_start is None or self.busy or self.closing:
            return
        self.canvas.delete("selection")
        self.canvas.create_rectangle(*self.region_start, self.canvas.canvasx(event.x), self.canvas.canvasy(event.y),
                                     outline=PALETTE["accent_hover"], width=2, dash=(4, 2), tags="selection")

    def _finish_region(self, event):
        start, self.region_start = self.region_start, None
        if start is None or self.busy or self.closing or self.selected_id is None:
            return
        end = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if max(abs(end[0]-start[0]), abs(end[1]-start[1])) < 4:
            return
        try:
            self.region = cleanup_rectangle(start, end, self.image_scale, self.blocks[self.selected_id]["box"])
        except ValueError as exc:
            self.region = None
            self.canvas.delete("selection")
            self.notice.set(str(exc))
        else:
            self.notice.set(f"Selected source region {self.region}. Cleanup will retain the saved translation.")
        self._set_busy(False)

    def _reclean(self):
        if self.busy or self.closing or self.region is None or self.selected_id is None:
            return
        self.cancel_event.clear()
        self._set_busy(True)
        self.notice.set("Cleaning the selection and exporting saved text…")
        self.commands.put(("reclean", self.selected_id, self.region))

    def _cancel(self):
        if self.busy:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self.notice.set("Cancelling after the current safe step…")

    def _drain_events(self):
        for _ in range(30):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind in {"opened", "saved"} and not self.closing:
                self.snapshot, self.record = payload, payload["record"]
                self.blocks = {block["block_id"]: block for block in self.record.get("blocks", [])}
                self.table.delete(*self.table.get_children())
                for key in self.blocks:
                    self.table.insert("", "end", iid=key, values=(key.rsplit(":", 1)[-1],))
                if self.selected_id not in self.blocks:
                    self.selected_id = next(iter(self.blocks), None)
                if self.selected_id is not None:
                    self.table.selection_set(self.selected_id)
                if payload.get("original_preview") is None:
                    self.show_original.set(False)
                self._display_preview()
                self._set_busy(False)
                self.notice.set("Cleanup saved. The translation text is unchanged." if kind == "saved" else
                                "Select a bubble and drag over leftover text. No translation is run.")
            elif kind in {"error", "fatal", "close_error"}:
                self._set_busy(False)
                self.notice.set(f"Operation failed; existing results are retained. {payload}")
                if kind in {"fatal", "close_error"}:
                    messagebox.showerror("Image Cleanup", payload, parent=self.window)
            elif kind == "closed":
                if self.worker is not None and self.worker.is_alive():
                    self.events.put(("closed", None))
                    break
                if self.window.grab_current() == self.window:
                    self.window.grab_release()
                self.window.destroy()
                self.on_closed()
                return
        self.window.after(60, self._drain_events)

    def request_close(self):
        if self.closing:
            return True
        self.closing = True
        self.cancel_event.set()
        self._set_busy(self.busy)
        self.notice.set("Closing the saved-page session safely…")
        self.commands.put(("close", None, None))
        return True


def _revision_worker(folder, events, cancel_event):
    """Only queues, strings and synchronization events cross the Tk boundary."""
    try:
        events.put(("loaded", collect_revisions(folder, cancel_event)))
    except Exception as exc:
        events.put(("error", f"{type(exc).__name__}: {exc}"))


class RevisionManagerWindow:
    """Read-only inventory; mutation stays in the existing Workbench workflow."""
    def __init__(self, parent, path, *, on_closed, on_read=None, on_edit=None):
        apply_theme(parent)
        self.window = tk.Toplevel(parent)
        self.window.title("Manga Translator · Revision Manager")
        self.window.geometry("1000x620")
        self.window.minsize(760, 500)
        if parent.state() != "withdrawn":
            self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.request_close)
        self.on_closed, self.on_read, self.on_edit = on_closed, on_read, on_edit
        self.folder = tk.StringVar(master=self.window, value=str(path))
        self.notice = tk.StringVar(master=self.window, value="")
        self.details = tk.StringVar(master=self.window, value="Select a protected page to see its last batch note.")
        self.events, self.cancel_event = queue.Queue(), threading.Event()
        self.worker, self.records = None, {}
        self.closing = False
        self._build()
        self.window.grab_set()
        self.refresh()
        self.window.after(60, self._drain_events)

    def _build(self):
        frame = ttk.Frame(self.window, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)
        header = ttk.Frame(frame)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Revision Manager", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Reading" if self.on_read else "Close", command=self.on_read or self.request_close).grid(row=0, column=1)
        ttk.Label(frame, text="Batch processing always preserves manual corrections and OCR candidates. Protection remains automatic.",
                  style="Muted.TLabel", wraplength=900).grid(row=1, column=0, sticky="w", pady=(0, 12))
        folder = ttk.Frame(frame)
        folder.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        folder.columnconfigure(0, weight=1)
        ttk.Entry(folder, textvariable=self.folder, state="readonly").grid(row=0, column=0, sticky="ew")
        self.choose_button = ttk.Button(folder, text="Choose Folder", command=self._choose_folder)
        self.choose_button.grid(row=0, column=1, padx=8)
        self.refresh_button = ttk.Button(folder, text="Refresh", command=self.refresh)
        self.refresh_button.grid(row=0, column=2)
        self.table = ttk.Treeview(frame, columns=("page", "work", "protection"), show="headings", selectmode="browse")
        for key, label, width in (("page", "Page", 260), ("work", "Work", 220), ("protection", "Preserved Content", 240)):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, minwidth=100)
        self.table.grid(row=3, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        scroll.grid(row=3, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        self.table.bind("<<TreeviewSelect>>", self._select)
        ttk.Label(frame, textvariable=self.details, wraplength=900, justify="left").grid(row=4, column=0, sticky="ew", pady=16)
        footer = ttk.Frame(frame)
        footer.grid(row=5, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.notice, style="Muted.TLabel", wraplength=700).grid(row=0, column=0, sticky="w")
        self.edit_button = ttk.Button(footer, text="Open Workbench", command=self._edit, state="disabled")
        self.edit_button.grid(row=0, column=1, padx=(12, 0))

    def _choose_folder(self):
        path = filedialog.askdirectory(parent=self.window, initialdir=self.folder.get(), title="Choose a translation output folder")
        if path:
            self.folder.set(path)
            self.refresh()

    def refresh(self):
        if self.closing or self.worker is not None and self.worker.is_alive():
            return
        self.cancel_event.clear()
        self.choose_button.configure(state="disabled")
        self.refresh_button.configure(state="disabled")
        self.edit_button.configure(state="disabled")
        self.notice.set("Reading saved records and batch notes…")
        folder = self.folder.get()

        self.worker = threading.Thread(target=_revision_worker,
            args=(folder, self.events, self.cancel_event), name="manga-revision-scan", daemon=False)
        self.worker.start()

    def _select(self, _event=None):
        selected = self.table.selection()
        record = self.records.get(selected[0]) if selected else None
        self.details.set(record["note"] if record else "Select a protected page to see its last batch note.")
        enabled = record is not None and self.on_edit is not None and not self.closing
        self.edit_button.configure(state="normal" if enabled else "disabled")

    def _edit(self):
        selected = self.table.selection()
        if not self.closing and selected and self.on_edit is not None:
            self.on_edit(self.records[selected[0]]["path"])

    def _drain_events(self):
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            if self.window.grab_current() == self.window:
                self.window.grab_release()
            self.window.destroy()
            self.on_closed()
            return
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if self.closing:
                continue
            if kind == "loaded":
                records, errors = payload
                self.records = {str(index): record for index, record in enumerate(records)}
                self.table.delete(*self.table.get_children())
                for key, record in self.records.items():
                    self.table.insert("", "end", iid=key, values=(record["page"], record["work"], record["protection"]))
                self.details.set("Select a protected page to see its last batch note.")
                self.notice.set(f"{len(records)} protected pages. Read-only scan; saved files are unchanged."
                                + (f" {len(errors)} unreadable JSON files: {errors[0]}" if errors else ""))
            else:
                self.notice.set("Scan failed. " + payload)
            self.choose_button.configure(state="normal")
            self.refresh_button.configure(state="normal")
        self.window.after(60, self._drain_events)

    def request_close(self):
        self.closing = True
        self.cancel_event.set()
        self.choose_button.configure(state="disabled")
        self.refresh_button.configure(state="disabled")
        self.edit_button.configure(state="disabled")
        self.notice.set("Closing after the read-only scan stops…")
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Saved-page desktop utilities")
    parser.add_argument("tool", choices=("cleanup", "revisions"))
    parser.add_argument("path", nargs="?")
    args = parser.parse_args(argv)
    root = tk.Tk()
    root.withdraw()
    apply_theme(root)
    path = args.path
    if not path:
        path = (filedialog.askopenfilename(parent=root, title="Choose a saved page", filetypes=[("Saved page", "*.json *.png")])
                if args.tool == "cleanup" else filedialog.askdirectory(parent=root, title="Choose a translation output folder"))
    if not path:
        root.destroy()
        return 0
    if args.tool == "cleanup":
        CleanupWindow(root, path, on_closed=root.destroy)
    else:
        pending = []

        def edit(record_path):
            if not pending:
                pending.append(record_path)
                manager.request_close()

        def closed():
            if pending:
                from .app import PageEditor
                PageEditor(root, pending.pop(), on_closed=root.destroy)
            else:
                root.destroy()
        manager = RevisionManagerWindow(root, path, on_closed=closed, on_edit=edit)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
