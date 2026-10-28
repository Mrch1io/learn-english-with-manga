"""Shared, dependency-free ttk styling for the local desktop application."""

from pathlib import Path
from tkinter import PhotoImage, TclError, ttk


PALETTE = {
    "bg": "#f8f8fa", "panel": "#ffffff", "soft": "#f1f2f5",
    "canvas": "#e9e9ef", "text": "#272631", "muted": "#686775",
    "border": "#e4e3eb", "accent": "#a74677", "accent_hover": "#963e6b",
    "selection": "#f8eaf2",
}


def _apply_app_icon(widget):
    """Set an interpreter-wide default once, including future Toplevels."""
    root = widget._root()
    if getattr(root, "_manga_app_icon_applied", False):
        return
    assets = Path(__file__).resolve().parents[1] / "assets"
    icon = assets / "app-icon.ico"
    if root.tk.call("tk", "windowingsystem") == "win32" and icon.is_file():
        try:
            root.iconbitmap(default=str(icon))
        except TclError:
            pass
        else:
            root._manga_app_icon_applied = True
            return
    image_path = assets / "app-icon.png"
    if not image_path.is_file():
        return
    try:
        photo = PhotoImage(master=root, file=str(image_path))
        root.iconphoto(True, photo)
    except TclError:
        return
    # Keep the fallback image with its owning interpreter, never a global Tk ref.
    root._manga_app_icon_photo = photo
    root._manga_app_icon_applied = True


def apply_theme(root):
    """Use stock Tk widgets; configuring the interpreter's theme is idempotent."""
    _apply_app_icon(root)
    p = PALETTE
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=("Segoe UI", 10), background=p["bg"], foreground=p["text"],
                    bordercolor=p["border"], lightcolor=p["panel"], darkcolor=p["border"])
    style.configure("TFrame", background=p["bg"])
    style.configure("Panel.TFrame", background=p["panel"])
    style.configure("TLabel", background=p["bg"], foreground=p["text"])
    style.configure("Panel.TLabel", background=p["panel"])
    style.configure("Muted.TLabel", foreground=p["muted"])
    style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
    style.configure("Heading.TLabel", font=("Segoe UI", 11, "bold"))
    style.configure("TLabelframe", background=p["bg"], bordercolor=p["border"])
    style.configure("TLabelframe.Label", foreground=p["muted"], background=p["bg"])
    style.configure("TButton", padding=(11, 7), background=p["panel"],
                    bordercolor=p["border"], focuscolor=p["accent"])
    style.map("TButton", background=[("pressed", p["selection"]), ("active", p["soft"])],
              foreground=[("disabled", p["muted"])])
    style.configure("Accent.TButton", background=p["accent"], foreground="#ffffff",
                    bordercolor=p["accent"], font=("Segoe UI", 10, "bold"))
    style.map("Accent.TButton", background=[("pressed", p["accent_hover"]), ("active", p["accent_hover"])],
              foreground=[("disabled", p["muted"]), ("!disabled", "#ffffff")])
    style.configure("Nav.TButton", padding=(16, 8), borderwidth=0, background=p["bg"])
    style.configure("ThumbnailToggle.TButton", padding=(4, 0))
    style.map("Nav.TButton", background=[("selected", p["selection"]), ("active", p["soft"])])
    style.configure("TCheckbutton", padding=(0, 3), background=p["bg"])
    style.map("TCheckbutton", background=[("active", p["soft"])])
    style.configure("TEntry", fieldbackground=p["panel"], padding=5)
    style.configure("TCombobox", fieldbackground=p["panel"], padding=5,
                    selectbackground=p["selection"], selectforeground=p["text"])
    style.map("TCombobox", fieldbackground=[("readonly", p["panel"])],
              selectbackground=[("readonly", p["panel"])],
              selectforeground=[("readonly", p["text"])])
    style.configure("Treeview", background=p["panel"], fieldbackground=p["panel"],
                    foreground=p["text"], rowheight=30, borderwidth=0)
    style.configure("Treeview.Heading", background=p["soft"], padding=(7, 7),
                    font=("Segoe UI", 10, "bold"))
    style.map("Treeview", background=[("selected", p["selection"])],
              foreground=[("selected", p["text"])])
    style.configure("TNotebook", background=p["bg"], borderwidth=0)
    style.configure("TNotebook.Tab", padding=(12, 8), background=p["soft"])
    style.map("TNotebook.Tab", background=[("selected", p["panel"])],
              foreground=[("selected", p["text"])])
    style.configure("TSeparator", background=p["border"])
    style.configure("TProgressbar", background=p["accent"], troughcolor=p["soft"], borderwidth=0)
    style.configure("TScrollbar", background=p["soft"], troughcolor=p["bg"],
                    arrowcolor=p["muted"], borderwidth=0)
    root.configure(background=p["bg"])
    root.option_add("*TCombobox*Listbox.background", p["panel"])
    root.option_add("*TCombobox*Listbox.foreground", p["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", p["selection"])
    root.option_add("*TCombobox*Listbox.selectForeground", p["text"])
    return style


def style_text(widget):
    widget.configure(background=PALETTE["panel"], foreground=PALETTE["text"],
                     insertbackground=PALETTE["text"], selectbackground=PALETTE["selection"],
                     selectforeground=PALETTE["text"], relief="flat", borderwidth=0,
                     highlightthickness=1, highlightbackground=PALETTE["border"],
                     highlightcolor=PALETTE["accent"], padx=9, pady=8,
                     font=("Segoe UI", 11))


def style_listbox(widget):
    widget.configure(background=PALETTE["panel"], foreground=PALETTE["text"],
                     selectbackground=PALETTE["selection"], selectforeground=PALETTE["text"],
                     relief="flat", borderwidth=0, highlightthickness=0,
                     activestyle="none", font=("Segoe UI", 10))
