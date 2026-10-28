from __future__ import annotations

from tkinter import filedialog

from .messages import Command, ExportRequest


class PagesMixin:
    def _update_page_navigation(self):
        valid = 0 <= self.page_index < len(self.pages)
        self.page_position.set(f"{self.page_index + 1} / {len(self.pages)}" if valid else "No pages")
        busy = self.loading or self.magi_busy or self.translation_busy
        self.previous_button.configure(state="normal" if valid and self.page_index > 0 and not busy else "disabled")
        self.next_button.configure(state="normal" if valid and self.page_index < len(self.pages)-1 and not busy else "disabled")
        self._highlight_thumbnail()

    def _step_page(self, offset):
        if self.loading or self.magi_busy or self.translation_busy:
            return
        index = self.page_index + offset
        if 0 <= index < len(self.pages):
            self.page_list.selection_clear(0, "end")
            self.page_list.selection_set(index)
            self.page_list.see(index)
            self._select_page()

    def _choose_folder(self):
        path = filedialog.askdirectory(parent=self.window, title="Open manga folder or saved results")
        if path:
            self.open_paths(path)

    def _choose_pages(self):
        paths = filedialog.askopenfilenames(parent=self.window, title="Import manga images or saved pages",
            filetypes=[("Images / page records", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.json")])
        if paths:
            self.open_paths(paths)

    def _choose_output(self):
        path = filedialog.askdirectory(parent=self.window, title="Choose output folder", mustexist=False)
        if path:
            self.output_dir.set(path)

    def _view_changed(self):
        if self.page_index < 0 or self.loading or self.magi_busy or self.translation_busy:
            return
        if self.view.get() == "英文译图":
            if not self.pages[self.page_index]["path"]:
                self.view.set("原图")
                self.status.set("Original only. Open Processing to translate this page.")
                return
            if not self.record.get("exported", True):
                self._export_page()
                return
        self._load_page(zoom_only=True)

    def _export_page(self):
        """Use saved translations; exporting never reruns detection, OCR or TTS."""
        if self.page_index < 0 or self.loading or self.magi_busy or self.translation_busy:
            return
        page = self.pages[self.page_index]
        if not page["path"]:
            return
        token = self._invalidate()
        self.operation_cancel.clear()
        self.translation_busy = self.exporting = True
        self._set_controls_busy(True)
        self.cancel_button.configure(state="normal")
        self.progress.start(15)
        self.status.set("Exporting saved translations with cleanup and typesetting. Manual corrections are preserved.")
        self.commands.put(Command("export", ExportRequest(page["path"], self.page_index, self.config), token))

    def _select_page(self, _event=None):
        if self.magi_busy or self.translation_busy or self.closing.is_set() or self.opening:
            return
        selected = self.page_list.curselection()
        if selected and selected[0] != self.page_index:
            self.page_index = selected[0]
            self.block_id = None
            self.view.set("英文译图" if self.pages[self.page_index]["path"] else "原图")
            self._see_thumbnail(self.page_index)
            self._load_page()

    def current_result_path(self):
        if 0 <= self.page_index < len(self.pages):
            return self.pages[self.page_index]["path"]
        return None

    def _clear_page(self):
        self.record, self.blocks, self.bindings = {}, {}, {}
        self.characters = []
        self.block_id = self.photo = self.current_voice = None
        self.thumbnail = None
        self.thumbnail_label.configure(image="")
        self._update_display_title()
        self.original = None
        self.temporary_voice = False
        self.canvas.delete("all")
        self.bubble_list.delete(*self.bubble_list.get_children())
        self.role_box.configure(values=[])
        self.role.set("")
        self.role_name.set("")
        self.voice.set("")
        self.bubble_notice.set("")
        self.bubble_context.set("")
        self.selected_bubble.set("Select a bubble on the page")
        for widget in (self.japanese, self.english):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.configure(state="disabled")

    def _translation_finished(self):
        self.translation_busy = False
        self.exporting = False
        self.progress.stop()
        self._set_controls_busy(False)
        self.cancel_button.configure(state="disabled")
        self._schedule_thumbnails()
