"""Real Tk icon inheritance checks; run on the private test desktop on Windows.

Run: .venv\\Scripts\\python.exe -m unittest tests.test_app_icon -v
No application, model, voice engine, or network calls are started.
"""
import ctypes
import gc
import os
from pathlib import Path
import struct
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

from ui import theme as ui_theme


class AppIconAssetTests(unittest.TestCase):
    def test_assets_keep_all_sizes_transparency_and_small_lens_contrast(self):
        from PIL import Image

        assets = Path(ui_theme.__file__).resolve().parents[1] / "assets"
        sizes = {16, 20, 24, 32, 40, 48, 64, 128, 256}
        raw_ico = (assets / "app-icon.ico").read_bytes()
        self.assertEqual(struct.unpack_from("<HHH", raw_ico), (0, 1, len(sizes)))
        for index in range(len(sizes)):
            offset = struct.unpack_from("<I", raw_ico, 6 + 16 * index + 12)[0]
            # Tk 8.6 selects native 16/32px frames correctly with DIB entries;
            # PNG-encoded entries can be silently rescaled to a blurry 32px icon.
            self.assertEqual(struct.unpack_from("<I", raw_ico, offset)[0], 40)
        with Image.open(assets / "app-icon.png") as png, Image.open(assets / "app-icon.ico") as ico:
            png.load()
            self.assertEqual((png.size, png.mode), ((256, 256), "RGBA"))
            self.assertEqual(ico.ico.sizes(), {(size, size) for size in sizes})
            frames = {}
            for size in sorted(sizes):
                frame = ico.ico.getimage((size, size)).convert("RGBA")
                frame.load()
                frames[size] = frame
                with self.subTest(size=size):
                    self.assertEqual(frame.size, (size, size))
                    alpha = frame.getchannel("A")
                    self.assertIsNotNone(alpha.getbbox())
                    for edge in ((0, 0, size, 1), (0, size - 1, size, size),
                                 (0, 0, 1, size), (size - 1, 0, size, size)):
                        self.assertEqual(alpha.crop(edge).getextrema(), (0, 0))
            self.assertEqual(frames[256].tobytes(), png.tobytes())
            # The space beneath the central hair tips is transparent, not a
            # dark background baked into the avatar.
            self.assertEqual(png.getchannel("A").crop((112, 216, 144, 240)).getextrema(), (0, 0))
            small = frames[16]
            lenses = [pixel for pixel in small.crop((4, 9, 12, 11)).getdata() if pixel[3] > 200]
            hair = [pixel for pixel in small.crop((4, 4, 12, 8)).getdata() if pixel[3] > 200]
            self.assertGreaterEqual(len(lenses), 12)
            self.assertGreaterEqual(len(hair), 24)
            luminance = lambda pixels: sum(.299 * r + .587 * g + .114 * b for r, g, b, _ in pixels) / len(pixels)
            self.assertLess(luminance(lenses), 110)
            self.assertGreater(luminance(hair) - luminance(lenses), 80)


def native_icons(window):
    """Read the actual small/large HWND icons rather than mocking Tk commands."""
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetParent.argtypes = [wintypes.HWND]
    user32.GetParent.restype = wintypes.HWND
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.SendMessageW.restype = wintypes.LPARAM
    get_class_icon = getattr(user32, "GetClassLongPtrW", user32.GetClassLongW)
    get_class_icon.argtypes = [wintypes.HWND, ctypes.c_int]
    get_class_icon.restype = ctypes.c_size_t
    window.update_idletasks()
    hwnd = user32.GetParent(window.winfo_id()) or window.winfo_id()
    if hwnd != int(window.wm_frame(), 0):
        raise AssertionError("Icon inspection must target the Tk window-manager wrapper")
    # Windows falls back to the registered class icon when WM_GETICON is zero;
    # Tk uses that path for interpreter-wide defaults.
    return tuple(user32.SendMessageW(hwnd, 0x007F, size, 0) or get_class_icon(hwnd, slot)
                 for size, slot in ((0, -34), (1, -14)))  # WM_GETICON; GCLP_HICONSM/HICON


class AppIconTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.root = None
        self.addCleanup(self._cleanup)
        self.assets = Path(ui_theme.__file__).resolve().parents[1] / "assets"
        self.assertTrue((self.assets / "app-icon.ico").is_file(), "The selected ICO must be installed")
        self.assertTrue((self.assets / "app-icon.png").is_file(), "The PNG fallback must be installed")
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.root.withdraw()

    def _cleanup(self):
        if self.root is not None:
            self.root.destroy()
            self.root = None
        gc.collect()

    @unittest.skipUnless(sys.platform == "win32", "Native multi-size ICO is Windows-specific")
    def test_native_default_replaces_tk_icon_and_is_inherited_by_new_toplevel(self):
        self.root.deiconify()
        self.root.update()
        before = native_icons(self.root)
        ui_theme.apply_theme(self.root)
        self.root.update()
        expected = native_icons(self.root)
        self.assertTrue(all(expected), "Both small and large native icons must be available")
        self.assertNotEqual(expected, before, "The Tk feather must be replaced")
        child = tk.Toplevel(self.root)
        child.update()
        self.assertEqual(native_icons(child), expected)

    @unittest.skipUnless(sys.platform == "win32", "Native ICO default is Windows-specific")
    def test_other_cwd_and_repeated_theme_calls_set_default_once_per_interpreter(self):
        is_file = Path.is_file
        with patch.object(self.root, "iconbitmap", wraps=self.root.iconbitmap) as iconbitmap, \
             patch.object(ui_theme, "PhotoImage", wraps=ui_theme.PhotoImage) as image, \
             patch.object(Path, "is_file", autospec=True, side_effect=is_file) as file_check:
            previous = Path.cwd()
            with tempfile.TemporaryDirectory() as directory:
                try:
                    os.chdir(directory)
                    ui_theme.apply_theme(self.root)
                    child = tk.Toplevel(self.root)
                    child.withdraw()
                    ui_theme.apply_theme(child)
                    ui_theme.apply_theme(self.root)
                finally:
                    os.chdir(previous)
            iconbitmap.assert_called_once_with(default=str(self.assets / "app-icon.ico"))
            file_check.assert_called_once_with(self.assets / "app-icon.ico")
            image.assert_not_called()
            self.assertTrue(self.root._manga_app_icon_applied)

        other = tk.Tk()
        other.withdraw()
        try:
            with patch.object(other, "iconbitmap", wraps=other.iconbitmap) as iconbitmap:
                ui_theme.apply_theme(other)
                iconbitmap.assert_called_once_with(default=str(self.assets / "app-icon.ico"))
        finally:
            other.destroy()

    def test_child_entry_point_keeps_png_fallback_on_root_and_loads_only_once(self):
        child = tk.Toplevel(self.root)
        child.withdraw()
        # Force the portable fallback even on Windows, without changing the
        # interpreter or replacing any real Tk image/icon command.
        with patch.object(Path, "is_file", autospec=True,
                          side_effect=lambda path: path.suffix != ".ico" and path.exists()), \
             patch.object(ui_theme, "PhotoImage", wraps=ui_theme.PhotoImage) as image, \
             patch.object(self.root, "iconphoto", wraps=self.root.iconphoto) as iconphoto:
            ui_theme.apply_theme(child)
            photo = self.root._manga_app_icon_photo
            self.assertIs(photo.tk, self.root.tk)
            self.assertIn(str(photo), self.root.tk.call("image", "names"))
            child.destroy()
            replacement = tk.Toplevel(self.root)
            replacement.withdraw()
            ui_theme.apply_theme(replacement)
            ui_theme.apply_theme(self.root)
            image.assert_called_once_with(master=self.root, file=str(self.assets / "app-icon.png"))
            iconphoto.assert_called_once_with(True, photo)
            self.assertIs(self.root._manga_app_icon_photo, photo)


if __name__ == "__main__":
    unittest.main()
