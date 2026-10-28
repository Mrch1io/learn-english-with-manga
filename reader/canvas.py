from __future__ import annotations

from ui.theme import PALETTE


class CanvasMixin:
    def _canvas_size(self):
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _scroll_page(self, event):
        if event.delta and self.canvas.yview() != (0.0, 1.0):
            steps = -int(event.delta / 120) or (-1 if event.delta > 0 else 1)
            self.canvas.yview_scroll(steps, "units")
            return "break"

    def _display_preview(self, image, original_size):
        from PIL import ImageTk

        self.photo = ImageTk.PhotoImage(image, master=self.window)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        padding_x = max(0, (self.canvas.winfo_width()-image.width)/2)
        padding_y = max(0, (self.canvas.winfo_height()-image.height)/2)
        self.canvas.configure(scrollregion=(-padding_x, -padding_y,
                                             image.width+padding_x, image.height+padding_y))
        self.scale = image.width/original_size[0], image.height/original_size[1]
        if self.zoom.get() in ("Fit to Window", "适合窗口"):
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)
        elif self.zoom.get() == "Fit to Width":
            self.canvas.xview_moveto(0)
        self._draw_boxes()

    def _select_bubble(self, _event=None):
        if self.loading or self.magi_busy or self.translation_busy:
            return
        selected = self.bubble_list.selection()
        if selected and selected[0] in self.blocks and selected[0] != self.block_id:
            self._invalidate()
            self.block_id = selected[0]
            self._show_bubble()
            if self.autoplay.get():
                self.play()

    def _draw_boxes(self):
        self.canvas.delete("bubble")
        if self.photo is None:
            return
        sx, sy = self.scale
        for block_id, block in self.blocks.items():
            if block_id != self.block_id and not self.show_regions.get():
                continue
            x1, y1, x2, y2 = block["box"]
            self.canvas.create_rectangle(x1*sx, y1*sy, x2*sx, y2*sy,
                outline=PALETTE["accent"] if block_id == self.block_id else PALETTE["muted"],
                width=2 if block_id == self.block_id else 1, tags="bubble")
        for candidate in (self.record.get("magi_coverage", {}).get("unmatched_texts", [])
                          if self.show_regions.get() else []):
            x1, y1, x2, y2 = candidate["box"]
            self.canvas.create_rectangle(x1*sx, y1*sy, x2*sx, y2*sy,
                outline="#d13c86", width=2, dash=(5, 3), tags=("bubble", "unmatched-text"))
            self.canvas.create_text(x1*sx+3, y1*sy+3, anchor="nw", text="Detected text",
                fill="#a71c65", tags=("bubble", "unmatched-text"))
        character_box = self.blocks.get(self.block_id, {}).get("metadata", {}).get("magi", {}).get("character_box")
        if isinstance(character_box, (list, tuple)) and len(character_box) == 4:
            x1, y1, x2, y2 = character_box
            self.canvas.create_rectangle(x1*sx, y1*sy, x2*sx, y2*sy, outline="#72e29c", width=2, tags="bubble")
            self.canvas.create_text(x1*sx+3, y1*sy+3, anchor="nw", text="Magi suggestion", fill="#16854b", tags="bubble")

    def _click(self, event):
        if self.photo is None or self.loading or self.magi_busy or self.translation_busy:
            return
        x, y = self.canvas.canvasx(event.x)/self.scale[0], self.canvas.canvasy(event.y)/self.scale[1]
        matches = [(key, b["box"]) for key, b in self.blocks.items()
                   if b["box"][0] <= x < b["box"][2] and b["box"][1] <= y < b["box"][3]]
        if matches:
            key = min(matches, key=lambda item: (item[1][2]-item[1][0])*(item[1][3]-item[1][1]))[0]
            self.bubble_list.selection_set(key)
            self.bubble_list.see(key)
            if key == self.block_id and self.autoplay.get():
                self.play()
            else:
                self._select_bubble()
