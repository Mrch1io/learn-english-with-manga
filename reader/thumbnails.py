from __future__ import annotations

import math

from ui.theme import PALETTE
from .messages import Command, ThumbnailRequest, ThumbnailResult

THUMBNAIL_ROW_HEIGHT = 200
THUMBNAIL_CACHE_LIMIT = 24


class ThumbnailsMixin:
    def _thumbnail_scrolled(self, first, last):
        self.thumbnail_scroll.set(first, last)
        self._schedule_thumbnails()

    def _schedule_thumbnails(self, _event=None):
        if self.after_thumbnails is None and not self.closing.is_set() and not self.pages_collapsed:
            self.after_thumbnails = self.window.after_idle(self._render_thumbnails)

    def _highlight_thumbnail(self):
        for index, card in self.thumbnail_cards.items():
            selected = index == self.page_index
            self.page_thumbnails.itemconfigure(card["background"],
                fill=PALETTE["selection"] if selected else PALETTE["panel"],
                outline=PALETTE["accent"] if selected else PALETTE["border"],
                width=2 if selected else 1)

    def _request_thumbnail(self):
        # 缩略图最多排队一张，避免翻页和模型任务等待整章解码。
        if (self.thumbnail_pending is not None or self.loading or self.magi_busy
                or self.translation_busy or self.closing.is_set() or self.pages_collapsed):
            return
        candidates = dict.fromkeys((*self.thumbnail_visible, *self.thumbnail_cards))
        for index in candidates:
            if index in self.thumbnail_cache or index in self.thumbnail_errors:
                continue
            page = self.pages[index]
            self._thumbnail_request += 1
            token = (self.thumbnail_generation, self._thumbnail_request)
            self.thumbnail_pending = (index, token)
            self.commands.put(Command("thumbnail", ThumbnailRequest(
                page["path"], page["input"], index, self._thumbnail_versions.get(index, 0)), token))
            break

    def _receive_thumbnail(self, action: str, result: ThumbnailResult, token: tuple[int, int]) -> None:
        index = result.index
        if self.thumbnail_pending != (index, token):
            return
        self.thumbnail_pending = None
        if (token[0] == self.thumbnail_generation and 0 <= index < len(self.pages)
                and result.identity == (self.pages[index]["path"], self.pages[index]["input"])
                and result.version == self._thumbnail_versions.get(index, 0)):
            if action == "thumbnail":
                from PIL import ImageTk

                self._remember_thumbnail(index, ImageTk.PhotoImage(result.image, master=self.window))
            else:
                self.thumbnail_errors[index] = result.message
        self._schedule_thumbnails()

    def _invalidate_thumbnail(self, index):
        self._thumbnail_versions[index] = self._thumbnail_versions.get(index, 0) + 1
        self.thumbnail_cache.pop(index, None)
        self.thumbnail_errors.pop(index, None)
        self._schedule_thumbnails()

    def _select_thumbnail(self, index):
        if (self.closing.is_set() or self.opening or self.magi_busy or self.translation_busy
                or not 0 <= index < len(self.pages)):
            return
        self.page_thumbnails.focus_set()
        self.page_list.selection_clear(0, "end")
        self.page_list.selection_set(index)
        self._select_page()

    def _thumbnail_wheel_step(self, units):
        self.page_thumbnails.yview_scroll(units, "units")
        return "break"

    def _thumbnail_wheel(self, event):
        if event.delta:
            return self._thumbnail_wheel_step(-int(event.delta / 120) or (-1 if event.delta > 0 else 1))
        return "break"

    def _render_thumbnails(self):
        """Virtualize the rail: Canvas items and decoding track the viewport."""
        self.after_thumbnails = None
        if self.closing.is_set() or self.pages_collapsed:
            return
        canvas = self.page_thumbnails
        width = max(148, canvas.winfo_width())
        top = max(0, canvas.canvasy(0))
        height = max(1, canvas.winfo_height())
        first = min(len(self.pages), int(top // THUMBNAIL_ROW_HEIGHT))
        last = min(len(self.pages), math.ceil((top + height) / THUMBNAIL_ROW_HEIGHT))
        self.thumbnail_visible = tuple(range(first, last))
        wanted = range(max(0, first - 1), min(len(self.pages), last + 1))
        for index in set(self.thumbnail_cards).difference(wanted):
            for item in self.thumbnail_cards.pop(index).values():
                canvas.delete(item)
        for index in wanted:
            y = index * THUMBNAIL_ROW_HEIGHT
            if index not in self.thumbnail_cards:
                self.thumbnail_cards[index] = {
                    "background": canvas.create_rectangle(3, y + 3, width - 3, y + THUMBNAIL_ROW_HEIGHT - 3),
                    "image": canvas.create_image(width / 2, y + 100, anchor="center"),
                    "placeholder": canvas.create_text(width / 2, y + 100, fill=PALETTE["muted"],
                                                       font=("Segoe UI", 9), width=124),
                    "label": canvas.create_text(width / 2, y + 190, text="",
                                                 fill=PALETTE["muted"], font=("Segoe UI", 9)),
                }
            card = self.thumbnail_cards[index]
            canvas.coords(card["background"], 3, y + 3, width - 3, y + THUMBNAIL_ROW_HEIGHT - 3)
            for key, offset in (("image", 100), ("placeholder", 100), ("label", 190)):
                canvas.coords(card[key], width / 2, y + offset)
            photo = self.thumbnail_cache.get(index)
            canvas.itemconfigure(card["image"], image=photo or "")
            canvas.itemconfigure(card["placeholder"], text="" if photo else
                                 "Unavailable" if index in self.thumbnail_errors else "Loading…")
            if photo is not None:
                self.thumbnail_cache.move_to_end(index)
        self._highlight_thumbnail()
        self._request_thumbnail()

    def _remember_thumbnail(self, index, photo):
        self.thumbnail_cache[index] = photo
        self.thumbnail_cache.move_to_end(index)
        self.thumbnail_errors.pop(index, None)
        while len(self.thumbnail_cache) > THUMBNAIL_CACHE_LIMIT:
            self.thumbnail_cache.popitem(last=False)
        self._schedule_thumbnails()

    def _see_thumbnail(self, index):
        if self.pages_collapsed or not 0 <= index < len(self.pages):
            return
        canvas = self.page_thumbnails
        height = max(1, canvas.winfo_height())
        top = max(0, canvas.canvasy(0))
        row_top = index * THUMBNAIL_ROW_HEIGHT
        row_bottom = row_top + THUMBNAIL_ROW_HEIGHT
        if row_top < top or row_bottom > top + height:
            # 按滚动步长向外取整，避免选中缩略图被裁掉。
            target = (math.floor(row_top / 24) * 24 if row_top < top
                      else math.ceil((row_bottom - height) / 24) * 24)
            canvas.yview_moveto(max(0, target) / max(1, len(self.pages) * THUMBNAIL_ROW_HEIGHT))
        self._schedule_thumbnails()

    def _thumbnail_clicked(self, event):
        self._select_thumbnail(int(self.page_thumbnails.canvasy(event.y) // THUMBNAIL_ROW_HEIGHT))
        return "break"

    def _thumbnail_key(self, event):
        step = max(1, self.page_thumbnails.winfo_height() // THUMBNAIL_ROW_HEIGHT)
        offsets = {"Up": -1, "Left": -1, "Down": 1, "Right": 1, "Prior": -step, "Next": step}
        if event.keysym in offsets:
            self._select_thumbnail(min(max(0, self.page_index + offsets[event.keysym]), len(self.pages) - 1))
        elif event.keysym in ("Home", "End"):
            self._select_thumbnail(0 if event.keysym == "Home" else len(self.pages) - 1)
        else:
            return None
        return "break"
