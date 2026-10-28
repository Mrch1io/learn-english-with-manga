from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ui.theme import PALETTE, style_listbox, style_text


class BuildMixin:
    def _build(self):
        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)
        self._build_chrome(frame)
        self._build_toolbar(frame)
        self._build_pages(frame)
        self._build_preview(frame)
        self._build_sidebar(frame)
        self._build_processing(frame)
        self._build_settings()
        self._build_footer(frame)

    def _build_chrome(self, frame):
        chrome = ttk.Frame(frame, style="Panel.TFrame", padding=(14, 10))
        chrome.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Label(chrome, text="Manga Translator", style="Title.TLabel").pack(side="left")
        ttk.Label(chrome, text="  /  Your reading space", style="Muted.TLabel").pack(side="left")
        ttk.Button(chrome, text="Workbench", command=self._open_workbench,
                   style="Nav.TButton").pack(side="right")
        ttk.Button(chrome, text="Reading", style="Accent.TButton",
                   command=self._focus_reading).pack(side="right", padx=(0, 6))

    def _build_toolbar(self, frame):
        self.toolbar = toolbar = ttk.Frame(frame, padding=(0, 9))
        toolbar.grid(row=1, column=0, columnspan=3, sticky="ew")
        ttk.Label(toolbar, textvariable=self.display_title, style="Heading.TLabel", width=20,
                  wraplength=170).pack(side="left", padx=(2, 14))
        self.import_button = ttk.Menubutton(toolbar, text="Import", width=8)
        self.import_menu = tk.Menu(self.import_button, tearoff=False, background=PALETTE["panel"],
                                  foreground=PALETTE["text"], activebackground=PALETTE["selection"])
        self.import_menu.add_command(label="Images / JSON...", command=self._choose_pages)
        self.import_menu.add_command(label="Folder...", command=self._choose_folder)
        self.import_button.configure(menu=self.import_menu)
        self.import_button.pack(side="left", padx=(0, 5))
        self.tools_button = ttk.Menubutton(toolbar, text="Tools", width=8)
        self.tools_menu = tk.Menu(self.tools_button, tearoff=False, background=PALETTE["panel"],
                                 foreground=PALETTE["text"], activebackground=PALETTE["selection"])
        self.tools_menu.add_command(label="Tasks", command=self._open_tasks)
        self.tools_menu.add_command(label="Terms", command=self._open_terms,
                                    state="normal" if self.on_terms is not None else "disabled")
        self.tools_menu.add_command(label="Processing", command=lambda: self._show_tools("Processing"))
        self.tools_menu.add_command(label="Settings", command=lambda: self._show_tools("Settings"))
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label="Translate Current Page", command=self._translate_page)
        self.tools_menu.add_command(label="Export Saved Translation", command=self._export_page)
        self.tools_menu.add_separator()
        for label, destination in (("Image Cleanup", "cleanup"), ("Revision Manager", "revisions")):
            self.tools_menu.add_command(label=label,
                command=lambda destination=destination: self._open_tool(destination),
                state="normal" if self.on_tools is not None else "disabled")
        self.tools_button.configure(menu=self.tools_menu)
        self.tools_button.pack(side="left")
        self.view_button = ttk.Button(toolbar, text="Show Original", style="Accent.TButton",
                                      command=self._toggle_view)
        self.view_button.pack(side="right")
        self.settings_button = ttk.Button(toolbar, text="Settings", command=lambda: self._show_tools("Settings"))
        self.settings_button.pack(side="right", padx=(0, 6))
        self.view_box = ttk.Combobox(toolbar, textvariable=self.view,
                                    values=["原图", "英文译图"], state="readonly")

    def _build_pages(self, frame):
        self.pages_collapsed = False
        self.pages_panel = pages = ttk.Frame(frame, style="Panel.TFrame", padding=10)
        pages.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
        pages.rowconfigure(1, weight=1)
        pages.columnconfigure(0, weight=1)
        pages_heading = ttk.Frame(pages, style="Panel.TFrame")
        pages_heading.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 9))
        ttk.Label(pages_heading, text="PAGES", style="Muted.TLabel").pack(side="left")
        self.collapse_pages_button = ttk.Button(pages_heading, text="<", width=3,
                                               style="ThumbnailToggle.TButton", command=self._toggle_pages)
        self.collapse_pages_button.pack(side="right")
        self.pages_rail = ttk.Frame(frame, style="Panel.TFrame", padding=5)
        self.pages_rail.grid(row=2, column=0, sticky="ns", padx=(0, 10))
        self.pages_rail.grid_remove()
        self.expand_pages_button = ttk.Button(self.pages_rail, text=">", width=3, command=self._toggle_pages)
        self.expand_pages_button.pack(pady=(5, 8))
        ttk.Label(self.pages_rail, text="Pages", style="Muted.TLabel").pack()
        # 保留现有选择事件，页面列表改由缩略图显示。
        self.thumbnail_label = ttk.Label(pages, style="Panel.TLabel", anchor="center")
        self.page_list = tk.Listbox(pages, width=19, exportselection=False, activestyle="none")
        style_listbox(self.page_list)
        self.page_list.bind("<<ListboxSelect>>", self._select_page)
        self.page_thumbnails = tk.Canvas(pages, width=148, height=1, background=PALETTE["panel"],
            highlightthickness=1, highlightbackground=PALETTE["border"],
            highlightcolor=PALETTE["accent"], takefocus=True, yscrollincrement=24)
        self.page_thumbnails.grid(row=1, column=0, sticky="nsew")
        self.thumbnail_scroll = ttk.Scrollbar(pages, orient="vertical", command=self.page_thumbnails.yview)
        self.thumbnail_scroll.grid(row=1, column=1, sticky="ns")
        self.page_thumbnails.configure(yscrollcommand=self._thumbnail_scrolled)
        self.page_thumbnails.bind("<Configure>", self._schedule_thumbnails)
        self.page_thumbnails.bind("<Button-1>", self._thumbnail_clicked)
        self.page_thumbnails.bind("<MouseWheel>", self._thumbnail_wheel)
        self.page_thumbnails.bind("<Button-4>", lambda _: self._thumbnail_wheel_step(-1))
        self.page_thumbnails.bind("<Button-5>", lambda _: self._thumbnail_wheel_step(1))
        self.page_thumbnails.bind("<KeyPress>", self._thumbnail_key)

    def _build_preview(self, frame):
        preview = ttk.Frame(frame)
        preview.grid(row=2, column=1, sticky="nsew", padx=(0, 10))
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(preview, width=1, height=1, background=PALETTE["canvas"], highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        xs = ttk.Scrollbar(preview, orient="horizontal", command=self.canvas.xview)
        ys = ttk.Scrollbar(preview, orient="vertical", command=self.canvas.yview)
        xs.grid(row=1, column=0, sticky="ew")
        ys.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<MouseWheel>", self._scroll_page)
        self.canvas.bind("<Configure>", self._canvas_resized)
        navigation = ttk.Frame(preview, padding=(4, 8))
        navigation.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.previous_button = ttk.Button(navigation, text="Previous", command=lambda: self._step_page(-1), width=9)
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(navigation, text="Next", command=lambda: self._step_page(1), width=7)
        self.next_button.pack(side="right")
        ttk.Label(navigation, textvariable=self.page_position, anchor="center").pack(side="left", expand=True)
        self.zoom_box = ttk.Combobox(navigation, textvariable=self.zoom,
            values=["Fit to Window", "Fit to Width", "25%", "50%", "75%", "100%", "150%", "200%"], state="readonly", width=13)
        self.zoom_box.pack(side="right", padx=7)
        self.zoom_box.bind("<<ComboboxSelected>>", lambda _: self._load_page(zoom_only=True))

    def _build_sidebar(self, frame):
        self.sidebar = ttk.Frame(frame, style="Panel.TFrame")
        self.sidebar.grid(row=2, column=2, sticky="nsew")
        self.sidebar.rowconfigure(0, weight=1)
        self.sidebar.columnconfigure(0, weight=1)
        self.sidebar_collapsed = False
        self.sidebar_rail = ttk.Frame(frame, style="Panel.TFrame", padding=5)
        self.sidebar_rail.grid(row=2, column=2, sticky="ns")
        self.sidebar_rail.grid_remove()
        self.expand_sidebar_button = ttk.Button(self.sidebar_rail, text="<", width=3, command=self._toggle_sidebar)
        self.expand_sidebar_button.pack(pady=(5, 8))
        ttk.Label(self.sidebar_rail, text="Voice", style="Muted.TLabel").pack()
        self.side_canvas = tk.Canvas(self.sidebar, width=296, height=1,
                                     background=PALETTE["panel"], highlightthickness=0)
        self.side_canvas.grid(row=0, column=0, sticky="nsew")
        side_scroll = ttk.Scrollbar(self.sidebar, orient="vertical", command=self.side_canvas.yview)
        side_scroll.grid(row=0, column=1, sticky="ns")
        self.side_canvas.configure(yscrollcommand=side_scroll.set)
        side = ttk.Frame(self.side_canvas, style="Panel.TFrame", padding=14)
        side_window = self.side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", lambda _: self.side_canvas.configure(scrollregion=self.side_canvas.bbox("all")))
        self.side_canvas.bind("<Configure>", lambda event: self.side_canvas.itemconfigure(side_window, width=event.width))
        side.columnconfigure(0, weight=1)
        side_heading = ttk.Frame(side, style="Panel.TFrame")
        side_heading.grid(row=0, column=0, sticky="ew")
        ttk.Label(side_heading, text="Character & Voice", style="Title.TLabel").pack(side="left")
        self.collapse_sidebar_button = ttk.Button(side_heading, text=">", width=3, command=self._toggle_sidebar)
        self.collapse_sidebar_button.pack(side="right")
        # 气泡选择仍沿用 Treeview 事件，侧栏不再显示列表。
        self.bubble_list = ttk.Treeview(side, columns=("id", "role"), show="headings", height=3, selectmode="browse")
        self.bubble_list.heading("id", text="Bubble")
        self.bubble_list.heading("role", text="Character")
        self.bubble_list.column("id", width=85, stretch=False)
        self.bubble_list.column("role", width=170)
        self.bubble_list.bind("<<TreeviewSelect>>", self._select_bubble)
        self.japanese = tk.Text(side, height=1, width=1, state="disabled")
        self.english = tk.Text(side, height=1, width=1, state="disabled")
        for widget in (self.japanese, self.english):
            style_text(widget)
        ttk.Label(side, text="Character", style="Panel.TLabel").grid(row=3, column=0, sticky="w", pady=(14, 5))
        self.role_box = ttk.Combobox(side, textvariable=self.role, state="readonly")
        self.role_box.grid(row=4, column=0, sticky="ew")
        self.role_box.bind("<<ComboboxSelected>>", self._role_changed)
        ttk.Button(side, text="Bind Selected Bubble", command=self._bind_role).grid(
            row=5, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(side, text="Voice", style="Panel.TLabel").grid(row=7, column=0, sticky="w", pady=(14, 5))
        self.voice_box = ttk.Combobox(side, textvariable=self.voice, state="readonly")
        self.voice_box.grid(row=8, column=0, sticky="ew")
        self.voice_box.bind("<<ComboboxSelected>>", self._voice_changed)
        buttons = ttk.Frame(side, style="Panel.TFrame")
        buttons.grid(row=10, column=0, sticky="ew", pady=(16, 7))
        for index, (label, callback) in enumerate((("Play", self.play), ("Stop", self.stop), ("Replay", self.play))):
            buttons.columnconfigure(index, weight=1)
            button = ttk.Button(buttons, text=label, command=callback, width=7,
                                style="Accent.TButton" if index == 0 else "TButton")
            button.grid(row=0, column=index, sticky="ew", padx=(0, 4 if index < 2 else 0))
            if label == "Stop":
                self.stop_button = button
        pending = [side]
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            if widget.winfo_class() not in ("TCombobox", "Treeview"):
                widget.bind("<MouseWheel>", self._scroll_sidebar, add="+")
        self.side_canvas.bind("<MouseWheel>", self._scroll_sidebar)

    def _build_processing(self, frame):
        self.tools = ttk.Notebook(frame)
        processing = ttk.Frame(self.tools, padding=10)
        self.tools.add(processing, text="Processing")
        task = ttk.Frame(processing)
        task.pack(fill="x")
        ttk.Label(task, text="Work").pack(side="left")
        ttk.Entry(task, textvariable=self.work, width=18).pack(side="left", padx=6)
        ttk.Button(task, text="Output Folder", command=self._choose_output).pack(side="left", padx=4)
        self.translate_button = ttk.Button(task, text="Translate Current Page", command=self._translate_page)
        self.translate_button.pack(side="left", padx=4)
        ttk.Button(task, text="Export Saved Translation", command=self._export_page).pack(side="left", padx=4)
        ttk.Button(task, text="Close Panel", command=self._hide_tools).pack(side="right")
        ttk.Checkbutton(processing, text="Show detected regions", variable=self.show_regions,
                        command=self._draw_boxes).pack(anchor="w", pady=(8, 0))
        ttk.Label(processing, textvariable=self.bubble_notice, wraplength=1080).pack(anchor="w", pady=(5, 0))

    def _build_settings(self):
        self.settings_window = tk.Toplevel(self.window)
        self.settings_window.withdraw()
        self.settings_window.title("Manga Translator · Settings")
        self.settings_window.transient(self.window)
        self.settings_window.geometry("720x520")
        self.settings_window.minsize(680, 500)
        self.settings_window.protocol("WM_DELETE_WINDOW", self._hide_settings)
        settings = ttk.Frame(self.settings_window, padding=20)
        settings.pack(fill="both", expand=True)
        settings.columnconfigure(0, weight=1, uniform="settings")
        settings.columnconfigure(1, weight=1, uniform="settings")
        settings.rowconfigure(1, weight=1)
        ttk.Label(settings, text="Settings", style="Title.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 18))
        ttk.Button(settings, text="Close", command=self._hide_settings).grid(row=0, column=1, sticky="e", pady=(0, 18))
        self._build_playback_settings(settings)
        self._build_character_settings(settings)

    def _build_playback_settings(self, settings):
        playback = ttk.Frame(settings)
        playback.grid(row=1, column=0, sticky="nsew", padx=(0, 20))
        ttk.Label(playback, text="Playback & Voice", style="Heading.TLabel").pack(anchor="w")
        self.autoplay_button = ttk.Checkbutton(playback, text="Play on bubble click", variable=self.autoplay,
                                              command=self._auto_changed)
        self.autoplay_button.pack(anchor="w", pady=(12, 8))
        self.preview_voice_button = ttk.Button(playback, text="Preview Voice", command=self.audition)
        self.preview_voice_button.pack(anchor="w", pady=(0, 16))
        connection = ttk.Frame(playback)
        connection.pack(fill="x")
        connection.columnconfigure(1, weight=1)
        ttk.Label(connection, text="VOICEVOX URL").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(connection, textvariable=self.url, width=29).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 12))
        ttk.Label(connection, text="Speed").grid(row=2, column=0, sticky="w")
        speed = ttk.Spinbox(connection, textvariable=self.speed, from_=0.5, to=2.0, increment=0.1, width=5,
                            command=self._settings_changed)
        speed.grid(row=2, column=1, sticky="w", padx=8)
        speed.bind("<FocusOut>", lambda _: self._settings_changed())
        voice_actions = ttk.Frame(playback)
        voice_actions.pack(fill="x", pady=(12, 0))
        self.connect_button = ttk.Button(voice_actions, text="Start / Connect", command=self.start_voice_engine)
        self.connect_button.pack(side="left", padx=(0, 6))
        self.refresh_button = ttk.Button(voice_actions, text="Refresh Voices", command=self.refresh_voices)
        self.refresh_button.pack(side="left")
        ttk.Button(playback, text="Save Settings", command=self._save_settings).pack(anchor="w", pady=(8, 12))
        status_area = ttk.Frame(playback)
        status_area.pack(fill="x")
        status_area.columnconfigure(0, weight=1)
        self.voice_status_canvas = tk.Canvas(status_area, height=70, width=1,
                                            background=PALETTE["bg"], highlightthickness=0)
        self.voice_status_canvas.grid(row=0, column=0, sticky="ew")
        status_scroll = ttk.Scrollbar(status_area, command=self.voice_status_canvas.yview)
        status_scroll.grid(row=0, column=1, sticky="ns")
        self.voice_status_canvas.configure(yscrollcommand=status_scroll.set)
        status_label = ttk.Label(self.voice_status_canvas, textvariable=self.voice_status, style="Muted.TLabel")
        status_item = self.voice_status_canvas.create_window((0, 0), window=status_label, anchor="nw")
        self.voice_status_canvas.bind("<Configure>", lambda event: (
            self.voice_status_canvas.itemconfigure(status_item, width=event.width),
            status_label.configure(wraplength=max(1, event.width))))
        status_label.bind("<Configure>", lambda _: self.voice_status_canvas.configure(
            scrollregion=self.voice_status_canvas.bbox("all")))
        status_label.bind("<MouseWheel>", lambda event: self.voice_status_canvas.yview_scroll(
            -1 if event.delta > 0 else 1, "units"))

    def _build_character_settings(self, settings):
        characters = ttk.Frame(settings)
        characters.grid(row=1, column=1, sticky="nsew")
        characters.columnconfigure(0, weight=1)
        ttk.Label(characters, text="Character", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(characters, textvariable=self.role, style="Muted.TLabel", wraplength=300).grid(
            row=1, column=0, sticky="w", pady=(12, 8))
        self.character_details_button = ttk.Button(characters, text="Manage Character", command=self._toggle_character_details)
        self.character_details_button.grid(row=2, column=0, sticky="ew")
        self.character_details = ttk.Frame(characters)
        self.character_details.columnconfigure(0, weight=1)
        self.character_details.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self.character_details.grid_remove()
        names = ttk.Frame(self.character_details)
        names.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        names.columnconfigure(0, weight=1)
        ttk.Entry(names, textvariable=self.role_name, width=16).grid(row=0, column=0, sticky="ew")
        ttk.Button(names, text="Add", command=lambda: self._save_role(new=True), width=6).grid(
            row=0, column=1, padx=(5, 0))
        ttk.Button(self.character_details, text="Save Name & Default Voice", command=self._save_role).grid(
            row=1, column=0, sticky="ew", pady=(7, 0))
        ttk.Button(self.character_details, text="Suggest Characters (Magi)", command=self._suggest_characters).grid(
            row=2, column=0, sticky="ew", pady=(9, 0))

    def _build_footer(self, frame):
        footer = ttk.Frame(frame, padding=(2, 7))
        footer.grid(row=4, column=0, columnspan=3, sticky="ew")
        self.cancel_button = ttk.Button(footer, text="Cancel Task", command=self.stop, state="disabled")
        self.cancel_button.pack(side="right")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=120)
        self.progress.pack(side="right", padx=10)
        ttk.Label(footer, textvariable=self.status, wraplength=760).pack(side="left", fill="x", expand=True)
        self._update_page_navigation()
