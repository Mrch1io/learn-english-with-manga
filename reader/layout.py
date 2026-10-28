from __future__ import annotations


class LayoutMixin:
    def _update_display_title(self):
        fixture = self.record.get("presentation_fixture")
        title = fixture.get("title") if isinstance(fixture, dict) else None
        self.display_title.set(title.strip() if isinstance(title, str) and title.strip()
                               else self.work.get().strip() or "Manga")

    def _scroll_sidebar(self, event):
        if self.side_canvas.yview() != (0.0, 1.0):
            self.side_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

    def _toggle_pages(self):
        """Reflow the rail and resize cached pixels, preserving page and audio."""
        if self.closing.is_set():
            return
        self.pages_collapsed = not self.pages_collapsed
        if self.pages_collapsed:
            self._pages_restore_zoom = self.zoom.get()
            self.zoom.set("Fit to Width")
            self.pages_panel.grid_remove()
            self.pages_rail.grid()
            if self.after_thumbnails is not None:
                self.window.after_cancel(self.after_thumbnails)
                self.after_thumbnails = None
            self.expand_pages_button.focus_set()
        else:
            self.pages_rail.grid_remove()
            self.pages_panel.grid()
            self.zoom.set(self._pages_restore_zoom or "Fit to Window")
            self._pages_restore_zoom = None
            self._see_thumbnail(self.page_index)
            self.page_thumbnails.focus_set()
        # 页面加载或缩放尚未结束时，也要保留恢复手动倍率的请求。
        self._layout_resize = True
        self._canvas_resized()

    def _toggle_sidebar(self):
        self.sidebar_collapsed = not self.sidebar_collapsed
        if self.sidebar_collapsed:
            self.sidebar.grid_remove()
            self.sidebar_rail.grid()
            self.expand_sidebar_button.focus_set()
        else:
            self.sidebar_rail.grid_remove()
            self.sidebar.grid()
            self.collapse_sidebar_button.focus_set()

    def _toggle_character_details(self):
        if self.character_details.winfo_manager():
            self.character_details.grid_remove()
            self.character_details_button.configure(text="Manage Character")
        else:
            self.character_details.grid()
            self.character_details_button.configure(text="Hide Character Details")

    def _open_workbench(self):
        if self.loading or self.magi_busy or self.translation_busy:
            return
        if self.on_workbench is not None:
            self.on_workbench(self.current_result_path())
        else:
            self._show_tools("Processing")
            self.status.set("Translate this page to make its text available in Workbench.")

    def _open_tasks(self):
        if self.loading or self.magi_busy or self.translation_busy:
            return
        if self.on_tasks is not None:
            self.on_tasks()
        else:
            self._show_tools("Processing")

    def _open_terms(self):
        if self.loading or self.magi_busy or self.translation_busy or self.closing.is_set():
            return
        if self.on_terms is not None:
            self.on_terms()

    def _open_tool(self, destination):
        """Hand navigation to the application, which owns the close/wait sequence."""
        if self.loading or self.magi_busy or self.translation_busy or self.closing.is_set():
            return
        if self.on_tools is not None:
            self.on_tools(destination, self.current_result_path())

    def _show_tools(self, name):
        if self.magi_busy or self.translation_busy or self.closing.is_set():
            return
        if name in ("Settings", "Voice Settings"):
            self.settings_window.deiconify()
            self.settings_window.lift()
            self.autoplay_button.focus_set()
            return
        self.tools.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        for tab in self.tools.tabs():
            if self.tools.tab(tab, "text") == name:
                self.tools.select(tab)
                break

    def _hide_tools(self):
        self.tools.grid_remove()

    def _hide_settings(self):
        self.settings_window.withdraw()
        self.settings_button.focus_set()

    def _focus_reading(self):
        self._hide_tools()
        self._hide_settings()
        self.canvas.focus_set()

    def _update_view_button(self):
        self.view_button.configure(text="Show Original" if self.view.get() == "英文译图" else "Show Translation")

    def _toggle_view(self):
        if self.page_index < 0 or self.loading or self.magi_busy or self.translation_busy:
            return
        if self.view.get() == "原图" and not self.record.get("exported", True):
            self._show_tools("Processing")
            self.status.set("No translated image yet. Export Saved Translation to generate it.")
            return
        self.view.set("原图" if self.view.get() == "英文译图" else "英文译图")
        self._view_changed()

    def _set_controls_busy(self, value):
        if value:
            pending = list(self.window.winfo_children())
            while pending:
                widget = pending.pop()
                pending.extend(widget.winfo_children())
                if widget not in (self.stop_button, self.cancel_button) and widget.winfo_class() in (
                        "TButton", "TMenubutton", "TCombobox", "TEntry", "TSpinbox", "TCheckbutton", "Listbox"):
                    self._disabled_widgets.append((widget, str(widget.cget("state"))))
                    widget.configure(state="disabled")
        else:
            for widget, state in self._disabled_widgets:
                widget.configure(state=state)
            self._disabled_widgets.clear()
            self._set_voice_connecting(self.voice_connecting)
            self._update_page_navigation()
