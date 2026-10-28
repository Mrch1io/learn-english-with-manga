from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .messages import JsonObject, ThumbnailRequest
from .records import read_record

if TYPE_CHECKING:
    from PIL.Image import Image

THUMBNAIL_SIZE = (132, 184)


def preview_size(size: tuple[int, int], scale: float, fit: tuple[int, int], *,
                 fit_width: bool = False) -> tuple[int, int]:
    width, height = size
    if scale:
        factor = float(scale)
    elif fit_width:
        factor = max(1, fit[0])/width
    else:
        factor = min(max(1, fit[0])/width, max(1, fit[1])/height)
    # 宽高同时缩放，像素数随倍率的平方增长。
    factor = min(factor, math.sqrt(16_000_000/(width*height)))
    return max(1, round(width*factor)), max(1, round(height*factor))


def resize_original(original: Image, scale: float, fit: tuple[int, int], *,
                    fit_width: bool = False) -> Image:
    from PIL import Image

    size = preview_size(original.size, scale, fit, fit_width=fit_width)
    return original.resize(size, Image.Resampling.LANCZOS) if size != original.size else original


def original_preview(record: JsonObject, scale: float, fit: tuple[int, int], *,
                     untranslated: bool = False, fit_width: bool = False) -> tuple[Image, Image]:
    from PIL import Image, ImageOps
    from pipeline.batch import file_hash

    expected = record.get("input_sha256")
    if not expected and not untranslated:
        raise ValueError("The page record is missing the original-image checksum. Regenerate the page text before opening it.")
    if expected and file_hash(record["input_path"]) != expected:
        raise ValueError("The original image has changed. Regenerate the page text before reusing its bubble coordinates.")
    with Image.open(record["input_path"]) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    return resize_original(original, scale, fit, fit_width=fit_width), original


def translated_preview(record: JsonObject, path: str | Path, scale: float, fit: tuple[int, int], *,
                       fit_width: bool = False) -> tuple[Image, Image]:
    """Read only a verified exported image; never label source pixels a translation."""
    from PIL import Image
    from pipeline.batch import file_hash

    if not record.get("exported", True):
        raise ValueError("This page contains translated text only. Export the English image first.")
    output = Path(path).with_suffix(".png")
    if not record.get("output_sha256") or file_hash(output) != record["output_sha256"]:
        raise ValueError("The English image is missing or failed verification. Check the saved output; no original image was substituted.")
    _, original = original_preview(record, 1.0, fit)
    with Image.open(output) as opened:
        translated = opened.convert("RGB")
    if translated.size != original.size:
        raise ValueError("The English image dimensions differ from the original. Existing text regions cannot be used safely.")
    return resize_original(translated, scale, fit, fit_width=fit_width), translated


def page_thumbnail(page: ThumbnailRequest) -> tuple[Image, str]:
    from PIL import Image, ImageOps
    from pipeline.batch import file_hash

    record = read_record(page.path) if page.path else None
    translated = record is not None and record.get("exported", True)
    path = Path(cast(str, page.path)).with_suffix(".png") if translated else Path(page.input)
    if record is not None:
        expected = record.get("output_sha256" if translated else "input_sha256")
        if not expected or file_hash(path) != expected:
            raise ValueError("Thumbnail image is missing or failed verification.")
    with Image.open(path) as opened:
        # JPEG 先降采样解码，避免生成整页 RGB 像素。
        opened.draft("RGB", THUMBNAIL_SIZE)
        image = ImageOps.exif_transpose(opened)
        image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        image = resize_original(image, 0, THUMBNAIL_SIZE).convert("RGB")
    return image, "translated" if translated else "original"
