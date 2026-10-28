"""先确认译文能放下，再清字；只处理已翻译的气泡。"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.runtime import TextBlock

import bisect
import os
import re
import struct
import unicodedata
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class FontError(ValueError):
    """字体缺失或不支持译文中的字符。"""


@dataclass
class TextLayout:
    font_size: int
    lines: list[str]
    line_height: int
    line_gap: int
    width: float
    height: int
    stroke_width: int
    overflow: bool = False
    reason: str = ""
    measurements: int = 0


@dataclass(frozen=True)
class CleanupPlan:
    """单次清字的掩膜缓存，使用期间保持原图不变。

    持有原图引用以防对象 ID 复用，不复制像素。
    """
    image_size: tuple[int, int]
    image_mode: str
    targets: tuple
    patches: tuple
    _source_image: Image.Image = field(repr=False, compare=False)


def _metadata(block):
    if getattr(block, "metadata", None) is None:
        block.metadata = {}
    return block.metadata


def _eligible(block):
    return (getattr(block, "status", "") == "translated"
            and bool(getattr(block, "translated_text", "").strip()))


def _clip_box(box, size):
    width, height = size
    x1, y1, x2, y2 = map(int, box)
    return (max(0, min(width, x1)), max(0, min(height, y1)),
            max(0, min(width, x2)), max(0, min(height, y2)))


def _uniform_interior(gray):
    """填满近白或近黑背景内的字形孔洞；无法确定气泡内部时返回 None。"""
    import cv2
    if gray.size == 0:
        return None
    light = np.median(gray) >= 127
    level = float(np.percentile(gray, 85 if light else 15))
    if 30 < level < 225:
        return None
    background = (gray >= max(180, level - 18)) if light else (gray <= min(75, level + 18))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background.astype(np.uint8), 8)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < gray.size * 0.30:
        return None
    region = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    interior = np.zeros_like(region)
    cv2.drawContours(interior, contours, -1, 1, thickness=cv2.FILLED)
    return interior


def _interior_rectangles(interior, pad):
    import cv2
    inner = cv2.erode(interior, np.ones((3, 3), np.uint8), iterations=pad,
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if not inner.any():
        return []
    distance = cv2.distanceTransform(inner, cv2.DIST_L2, 3)
    cy, cx = np.unravel_index(int(distance.argmax()), distance.shape)
    column = inner[:, cx] > 0
    above = np.flatnonzero(~column[:cy])
    below = np.flatnonzero(~column[cy:])
    top = int(above[-1] + 1) if len(above) else 0
    bottom = int(cy + below[0]) if len(below) else inner.shape[0]
    rectangles = []
    # 试几种高度，适应椭圆和锯齿状气泡。
    for fraction in (0.40, 0.55, 0.70, 0.85, 1.0):
        height = max(1, round((bottom - top) * fraction))
        y1 = max(top, min(bottom - height, int(cy - height // 2)))
        y2 = y1 + height
        rows = inner[y1:y2] > 0
        left = np.max(np.where(~rows[:, :cx + 1], np.arange(cx + 1), -1), axis=1) + 1
        right_part = rows[:, cx:]
        right = np.min(np.where(~right_part, np.arange(right_part.shape[1]), right_part.shape[1]), axis=1) + cx
        x1, x2 = int(left.max()), int(right.min())
        if x2 > x1:
            rectangles.append((x1, y1, x2, y2))
    return rectangles


def safe_layout_box(image, block):
    """优先使用指定安全区，其次估算背景内矩形，最后内缩检测框。"""
    _metadata(block).pop("safe_box_candidates", None)
    bx1, by1, bx2, by2 = _clip_box(block.box, image.size)
    supplied = getattr(block, "safe_box", None)
    if supplied:
        sx1, sy1, sx2, sy2 = _clip_box(supplied, image.size)
        box = (max(bx1, sx1), max(by1, sy1), min(bx2, sx2), min(by2, sy2))
        method = "supplied_interior"
        _metadata(block)["safe_box_candidates"] = [list(box)]
    else:
        pad = max(3, round(min(bx2 - bx1, by2 - by1) * 0.04))
        box = (bx1 + pad, by1 + pad, bx2 - pad, by2 - pad)
        method = "inset_detection_box"
        try:
            interior = _uniform_interior(np.asarray(image.crop((bx1, by1, bx2, by2)).convert("L")))
            rectangles = _interior_rectangles(interior, pad) if interior is not None else []
            if rectangles:
                rectangle = max(rectangles, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
                rx1, ry1, rx2, ry2 = rectangle
                box = (bx1 + rx1, by1 + ry1, bx1 + rx2, by1 + ry2)
                method = "rectangle_inside_uniform_background"
                _metadata(block)["safe_box_candidates"] = [
                    [bx1 + r[0], by1 + r[1], bx1 + r[2], by1 + r[3]] for r in rectangles]
        except ImportError:
            pass  # 未安装 OpenCV 时仍可按内缩矩形排版。
    _metadata(block)["safe_region_method"] = method
    return box


def resolve_font(font_path=None):
    """要求真实字体文件，避免退回过小的位图字体。"""
    candidates = [str(font_path)] if font_path else [
        str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            font = ImageFont.truetype(candidate, 12)
            path = Path(font.path).resolve()
            if path.is_file():
                return str(path)
        except (OSError, TypeError):
            pass
    raise FontError(f"Font not found: {font_path or 'Arial / DejaVu Sans'}. "
                    "Choose an existing .ttf/.otf/.ttc file with --font.")


@lru_cache(maxsize=16)
def _font_cmaps(path, mtime_ns):
    """检查 cmap 4/12/13，避免 Pillow 把缺字画成方框。

    TTC 使用第一个字体，与 Pillow 默认索引一致。
    """
    data = Path(path).read_bytes()
    offset = struct.unpack_from(">I", data, 12)[0] if data[:4] == b"ttcf" else 0
    count = struct.unpack_from(">H", data, offset + 4)[0]
    cmap_offset = None
    for i in range(count):
        tag, _, table_offset, _ = struct.unpack_from(">4sIII", data, offset + 12 + 16 * i)
        if tag == b"cmap":
            cmap_offset = table_offset
            break
    if cmap_offset is None:
        raise FontError(f"Font has no Unicode cmap: {path}")
    maps = []
    table_count = struct.unpack_from(">H", data, cmap_offset + 2)[0]
    for i in range(table_count):
        platform, encoding, suboffset = struct.unpack_from(">HHI", data, cmap_offset + 4 + 8 * i)
        if platform != 0 and not (platform == 3 and encoding in (1, 10)):
            continue
        start = cmap_offset + suboffset
        fmt = struct.unpack_from(">H", data, start)[0]
        if fmt == 4:
            length = struct.unpack_from(">H", data, start + 2)[0]
            table = data[start:start + length]
            n = struct.unpack_from(">H", table, 6)[0] // 2
            ends = struct.unpack_from(f">{n}H", table, 14)
            starts = struct.unpack_from(f">{n}H", table, 16 + 2 * n)
            deltas = struct.unpack_from(f">{n}h", table, 16 + 4 * n)
            ranges = struct.unpack_from(f">{n}H", table, 16 + 6 * n)
            maps.append((4, table, ends, starts, deltas, ranges))
        elif fmt in (12, 13):
            n = struct.unpack_from(">I", data, start + 12)[0]
            groups = tuple(struct.unpack_from(">III", data, start + 16 + 12 * j) for j in range(n))
            maps.append((fmt, tuple(g[0] for g in groups), groups))
    if not maps:
        raise FontError(f"Unsupported or absent Unicode cmap in font: {path}")
    return maps


def _has_glyph(maps, codepoint):
    for table in maps:
        if table[0] == 4:
            if codepoint > 0xFFFF:
                continue
            _, data, ends, starts, deltas, ranges = table
            i = bisect.bisect_left(ends, codepoint)
            if i == len(ends) or codepoint < starts[i]:
                continue
            if ranges[i] == 0:
                glyph = (codepoint + deltas[i]) & 0xFFFF
            else:
                address = 16 + 6 * len(ends) + 2 * i + ranges[i] + 2 * (codepoint - starts[i])
                glyph = struct.unpack_from(">H", data, address)[0] if address + 2 <= len(data) else 0
                if glyph:
                    glyph = (glyph + deltas[i]) & 0xFFFF
            if glyph:
                return True
        else:
            fmt, starts, groups = table
            i = bisect.bisect_right(starts, codepoint) - 1
            if i >= 0:
                first, last, glyph = groups[i]
                if first <= codepoint <= last and (glyph if fmt == 13 else glyph + codepoint - first):
                    return True
    return False


def validate_font_text(font_path, text):
    try:
        maps = _font_cmaps(font_path, Path(font_path).stat().st_mtime_ns)
        missing = [char for char in sorted(set(text))
                   if not char.isspace() and unicodedata.category(char) != "Cf"
                   and not _has_glyph(maps, ord(char))]
    except (OSError, ValueError, struct.error) as exc:
        raise FontError(f"Cannot inspect font {font_path}: {exc}") from exc
    if missing:
        display = " ".join(f"{char} (U+{ord(char):04X})" for char in missing[:8])
        raise FontError(f"Font {Path(font_path).name} lacks: {display}. Choose a font supporting the target text.")


@lru_cache(maxsize=128)
def _load_font(font_path, size):
    return ImageFont.truetype(font_path, size)


_OPEN_PUNCT = frozenset("([{（【《「『“‘〈〔〖〘〚")
_CLOSE_PUNCT = frozenset(")]},.!?;:，。！？；：、】）》」』”’〉〕〗〙〛…")
_CJK_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|\s+|.")


def _cjk_tokens(paragraph):
    tokens = []
    pending = ""
    for token in _CJK_TOKEN.findall(paragraph):
        if token in _OPEN_PUNCT:
            pending += token
        elif token in _CLOSE_PUNCT and tokens and not pending:
            tokens[-1] += token
        elif token.isspace():
            if tokens and tokens[-1] != " ":
                tokens.append(" ")
        else:
            tokens.append(pending + token)
            pending = ""
    if pending:
        if tokens:
            tokens[-1] += pending
        else:
            tokens.append(pending)
    return tokens


def _wrap(text, width, draw, font, stroke_width, language):
    lines = []
    cjk = language.lower().startswith(("zh", "ja", "ko"))
    def fits(value):
        bounds = draw.textbbox((0, 0), value, font=font, stroke_width=stroke_width)
        return bounds[2] - bounds[0] <= width
    for paragraph in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        tokens = _cjk_tokens(paragraph) if cjk else paragraph.split()
        current = ""
        for token in tokens:
            if not current and token.isspace():
                continue
            candidate = current + ((" " if current else "") if not cjk else "") + token
            if fits(candidate):
                current = candidate
            else:
                if current.strip():
                    lines.append(current.rstrip())
                current = token.lstrip()
                if current and not fits(current):
                    return lines + [current], f"Unbreakable token is wider than the safe area: {current[:48]!r}"
        if current.strip():
            lines.append(current.rstrip())
    return lines, ""


def layout_text(text, width, height, font_path=None, target_language="en",
                min_font_size=12, max_font_size=40, stroke_width=1):
    """英文单词不拆分；最小字号仍放不下时报告溢出，保留完整译文。"""
    if min_font_size <= 0 or max_font_size < min_font_size or stroke_width < 0:
        raise ValueError("Invalid font-size or stroke-width bounds")
    if not text.strip():
        return TextLayout(min_font_size, [], 0, 0, 0, 0, stroke_width)
    font_path = resolve_font(font_path)
    validate_font_text(font_path, text)
    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    measurements = 0
    def candidate(size):
        nonlocal measurements
        measurements += 1
        font = _load_font(font_path, size)
        lines, reason = _wrap(text, width, draw, font, stroke_width, target_language)
        ascent, descent = font.getmetrics()
        line_height = ascent + descent + 2 * stroke_width
        gap = max(2, round(size * 0.15))
        bounds = [draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width) for line in lines]
        line_height = max([line_height] + [b[3] - b[1] for b in bounds])
        actual_width = max([0] + [b[2] - b[0] for b in bounds])
        total_height = line_height * len(lines) + gap * max(0, len(lines) - 1)
        overflow = bool(reason) or actual_width > width or total_height > height or width <= 0 or height <= 0
        if overflow and not reason:
            reason = f"Text needs {actual_width}x{total_height}px; safe area is {width}x{height}px at {size}px"
        return TextLayout(size, lines, line_height, gap, actual_width, total_height,
                          stroke_width, overflow, reason)
    best = candidate(min_font_size)
    if not best.overflow:
        # 用二分查找能放下的最大字号。
        low, high = min_font_size + 1, max_font_size
        while low <= high:
            mid = (low + high) // 2
            trial = candidate(mid)
            if trial.overflow:
                high = mid - 1
            else:
                best = trial
                low = mid + 1
    best.measurements = measurements
    return best


def prepare_layout(image, blocks, font_path=None, target_language="en",
                   min_font_size=12, max_font_size=40):
    """先检查字体和排版，译文放不下时保留原图。"""
    active = [block for block in blocks if _eligible(block)]
    if not active:
        return {}
    font_path = resolve_font(font_path)
    layouts = {}
    for i, block in enumerate(active):
        signature = [block.translated_text, list(block.box),
                     list(block.safe_box) if getattr(block, "safe_box", None) else None,
                     list(image.size), font_path, target_language, min_font_size, max_font_size]
        old = _metadata(block).get("layout")
        if old and old.get("signature") == signature:
            layouts[getattr(block, "block_id", str(i))] = TextLayout(**{
                name: old[name] for name in TextLayout.__dataclass_fields__})
            continue
        box = safe_layout_box(image, block)
        choices = []
        try:
            for candidate_box in _metadata(block).get("safe_box_candidates", [box]):
                candidate_layout = layout_text(block.translated_text.strip(),
                                               candidate_box[2] - candidate_box[0], candidate_box[3] - candidate_box[1],
                                               font_path, target_language, min_font_size, max_font_size)
                choices.append((candidate_layout, candidate_box))
        except FontError as exc:
            block.status = "failed"
            block.error = f"Font unsupported for bubble {getattr(block, 'block_id', i)}: {exc}"
            _metadata(block).pop("layout", None)
            continue
        layout, box = max(choices, key=lambda pair: (
            not pair[0].overflow, pair[0].font_size,
            (pair[1][2] - pair[1][0]) * (pair[1][3] - pair[1][1])))
        _metadata(block)["layout"] = dict(asdict(layout), safe_box=list(box), font_path=font_path,
                                          signature=signature)
        layouts[getattr(block, "block_id", str(i))] = layout
        if layout.overflow:
            block.status = "overflow"
            block.error = layout.reason
    return layouts


def _halo(patch, contrast, interior):
    """清除浅色字形残影，同时保护线稿和纸张纹理。

    Otsu 可能漏掉浅色线条；扩展掩膜前还需与周围纸张比较。
    """
    import cv2
    kernel = np.ones((3, 3), np.uint8)
    band = cv2.dilate(patch.astype(np.uint8), kernel, iterations=4) > 0
    guard = cv2.dilate(((contrast >= 20) & ~patch).astype(np.uint8), kernel, iterations=5) > 0
    allowed = (interior > 0) & ~guard
    paper = (cv2.dilate(patch.astype(np.uint8), kernel, iterations=10) > 0) & ~band & allowed
    if paper.sum() < 32:
        return np.zeros_like(patch)
    floor = max(2.0, float(np.percentile(contrast[paper], 95)))
    return band & ~patch & allowed & (contrast > floor) & (contrast < 20)


def _mask_components(strong: np.ndarray) -> tuple[np.ndarray, int]:
    import cv2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strong, 8)
    height, width = strong.shape
    selected = np.zeros(count, dtype=np.uint8)
    rejected = 0
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        touches_edge = x == 0 or y == 0 or x + w >= width or y + h >= height
        edge_art = touches_edge and (w > width * 0.45 or h > height * 0.45 or area > strong.size * 0.08)
        long_line = (w > width * 0.80 and h < max(3, height * 0.035)) or (h > height * 0.80 and w < max(3, width * 0.035))
        huge = area > strong.size * 0.28
        if edge_art or long_line or huge:
            rejected += 1
        else:
            selected[label] = 255
    core = selected[labels]
    return core, rejected


def _mask_patches(image: Image.Image, blocks: Sequence[TextBlock]) -> list[tuple[tuple[int, int, int, int], np.ndarray, bool, np.ndarray]]:
    import cv2
    rgb = np.asarray(image.convert("RGB"))
    gray = np.asarray(image.convert("L"))
    patches = []
    kernel = np.ones((3, 3), np.uint8)
    for block in blocks:
        if not _eligible(block):
            continue
        box = _clip_box(block.box, image.size)
        x1, y1, x2, y2 = box
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        light = np.median(roi) >= 127
        threshold, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        polarity_source = "bubble_background"
        if _metadata(block).get("kind") == "free_text":
            # 用边缘对比判断文字明暗，避免彩色背景误导绝对亮度阈值。
            rim = min(2, max(1, min(roi.shape) // 8))
            border = np.concatenate((roi[:rim].ravel(), roi[-rim:].ravel(),
                                     roi[rim:-rim, :rim].ravel(),
                                     roi[rim:-rim, -rim:].ravel()))
            bright_fraction = np.mean(border > threshold)
            if bright_fraction <= 0.30 or bright_fraction >= 0.70:
                light = bright_fraction >= 0.70
                polarity_source = "text_border_otsu"
        level = float(np.percentile(roi, 85 if light else 15))
        contrast = np.maximum(0, level - roi.astype(np.float32) if light else roi.astype(np.float32) - level)
        strong = 255 - binary if light else binary
        strong[contrast < 20] = 0
        core, rejected = _mask_components(strong)
        # 一并填掉字形内部的灰色残留，避免留下旧字轮廓。
        contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(core, contours, -1, 255, thickness=cv2.FILLED)
        # 先扩一像素覆盖抗锯齿，再只向残留笔画扩展。
        expanded = cv2.dilate(core, kernel, iterations=1)
        fringe = (cv2.dilate(expanded, kernel, iterations=1) > 0) & (contrast > 8)
        patch = (expanded > 0) | fringe
        interior = _uniform_interior(roi)
        if interior is not None:
            patch &= interior > 0
        rejected_pixels = (strong > 0) & (core == 0)
        protected = cv2.dilate(rejected_pixels.astype(np.uint8), kernel, iterations=1) > 0
        patch &= ~protected
        background = ~patch
        residual = np.abs(roi.astype(np.float32) - level)
        # 在文字附近判断背景是否均匀，排除保留的边框和线稿。
        ring = (cv2.dilate(patch.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & background & ~protected
        if interior is not None:
            ring &= interior > 0
        flat = bool(np.any(patch) and background.mean() >= 0.50 and ring.sum() >= max(8, patch.sum() * 0.05)
                    and np.percentile(residual[ring], 95) <= 10
                    and (level >= 225 or level <= 30))
        bg_candidates = residual <= 8
        color = np.median(rgb[y1:y2, x1:x2][bg_candidates], axis=0).astype(np.uint8) if np.any(bg_candidates) else np.array([level] * 3, dtype=np.uint8)
        halo = _halo(patch, contrast, interior) if flat and interior is not None else np.zeros_like(patch)
        patch = patch | halo
        _metadata(block)["cleanup"] = {"background": "light" if light else "dark",
                                        "polarity_source": polarity_source,
                                        "method": "solid_fill" if flat else "lama",
                                        "masked_pixels": int(patch.sum()),
                                        "halo_pixels": int(halo.sum()),
                                        "rejected_components": rejected,
                                        "uniform_interior_constraint": interior is not None,
                                        "max_dilation_px": 6 if halo.any() else 2}
        patches.append((box, patch, flat, color))
    return patches


def _cleanup_targets(blocks):
    return tuple((getattr(block, "block_id", None), tuple(block.box),
                  getattr(block, "status", ""), getattr(block, "translated_text", ""),
                  (getattr(block, "metadata", None) or {}).get("kind")) for block in blocks)


def prepare_cleanup(image, blocks):
    """复用同一原图和目标的清字分析；目标标识、边界、状态、译文或类型变化后失效。"""
    blocks = tuple(blocks)
    targets = _cleanup_targets(blocks)
    patches = tuple(_mask_patches(image, blocks))
    for _, mask, _, color in patches:
        mask.setflags(write=False)
        color.setflags(write=False)
    return CleanupPlan(image.size, image.mode, targets, patches, image)


def _cleanup_patches(image, blocks, plan):
    if plan is None:
        return _mask_patches(image, blocks)
    if (not isinstance(plan, CleanupPlan) or plan._source_image is not image
            or plan.image_size != image.size or plan.image_mode != image.mode
            or plan.targets != _cleanup_targets(blocks)):
        raise ValueError("Cleanup plan no longer matches its original image and text regions")
    return plan.patches


def build_text_mask(image, blocks, *, plan=None):
    """返回原图坐标下的 uint8 掩膜，重叠区域取并集。"""
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for (x1, y1, x2, y2), patch, _, _ in _cleanup_patches(image, blocks, plan):
        mask[y1:y2, x1:x2] |= patch.astype(np.uint8) * 255
    return mask


def inpaint(image, blocks, pool=None, debug_mask_path=None, cancel=None, *, plan=None):
    """纯色背景直接填充，其余区域使用缓存的 LaMa。"""
    return _inpaint_patches(image, _cleanup_patches(image, blocks, plan), pool, debug_mask_path, cancel)


def clean_regions(image, block, regions, pool=None, cancel=None):
    """在原图或清字底图上应用半开区间矩形，保留掩膜外像素。

    气泡边界内缩两像素；背景不明确时使用安全排版区。
    """
    import cv2
    box = _clip_box(block.box, image.size)
    x1, y1, x2, y2 = box
    normalized = []
    if not isinstance(regions, (list, tuple)) or not regions:
        raise ValueError("Select at least one cleanup rectangle inside the bubble")
    for region in regions:
        if (not isinstance(region, (list, tuple)) or len(region) != 4
                or any(not isinstance(value, int) or isinstance(value, bool) for value in region)):
            raise ValueError("Cleanup rectangles must contain four integer coordinates")
        a, b, c, d = region
        if not (x1 <= a < c <= x2 and y1 <= b < d <= y2):
            raise ValueError("Cleanup rectangle falls outside the selected bubble")
        normalized.append(list(region))
    rgb = np.asarray(image.convert("RGB"))[y1:y2, x1:x2]
    gray = np.asarray(image.convert("L"))[y1:y2, x1:x2]
    allowed = _uniform_interior(gray)
    if allowed is None:
        allowed = np.zeros(gray.shape, dtype=np.uint8)
        a, b, c, d = safe_layout_box(image, block)
        allowed[b-y1:d-y1, a-x1:c-x1] = 255
    allowed = cv2.erode(allowed.astype(np.uint8), np.ones((5, 5), np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    mask = np.zeros(gray.shape, dtype=bool)
    for a, b, c, d in normalized:
        mask[b-y1:d-y1, a-x1:c-x1] = True
    mask &= allowed
    if not mask.any():
        raise ValueError("Selected area contains no safely editable bubble interior")
    background = allowed & ~mask
    if background.sum() < 8:
        raise ValueError("Leave some bubble background unselected to determine its color")
    color = np.median(rgb[background], axis=0).astype(np.uint8)
    residual = np.max(np.abs(rgb[background].astype(np.int16)-color), axis=1)
    flat = bool(np.percentile(residual, 95) <= 10 and (np.median(color) >= 225 or np.median(color) <= 30))
    result = _inpaint_patches(image, [(box, mask, flat, color)], pool, cancel=cancel)
    metadata = _metadata(block)
    previous = metadata.get("manual_cleanup_regions", [])
    metadata["manual_cleanup_regions"] = previous + [region for region in normalized if region not in previous]
    metadata["manual_cleanup"] = {"mode": "additive", "masked_pixels": int(mask.sum()),
                                   "method": "solid_fill" if flat else "lama",
                                   "boundary_inset_px": 2}
    return result


def _inpaint_patches(image, patches, pool=None, debug_mask_path=None, cancel=None):
    if cancel:
        cancel()
    result = np.array(image.convert("RGB"))
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    complex_mask = np.zeros_like(mask)
    complex_boxes = []
    for (x1, y1, x2, y2), patch, flat, color in patches:
        mask[y1:y2, x1:x2] |= patch.astype(np.uint8) * 255
        if flat:
            result[y1:y2, x1:x2][patch] = color
        else:
            complex_mask[y1:y2, x1:x2] |= patch.astype(np.uint8) * 255
            if np.any(patch):
                py, px = np.nonzero(patch)
                complex_boxes.append((max(0, x1 + int(px.min()) - 64),
                                      max(0, y1 + int(py.min()) - 64),
                                      min(image.width, x1 + int(px.max()) + 65),
                                      min(image.height, y1 + int(py.max()) + 65)))
    if debug_mask_path:
        path = Path(debug_mask_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(path)
    if not np.any(complex_mask):
        return Image.fromarray(result)
    if cancel:
        cancel()
    import torch
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise RuntimeError("Complex-background cleanup requires simple-lama-inpainting; install requirements.txt") from exc
    def load():
        if getattr(pool, "local_files_only", False):
            explicit = os.environ.get("LAMA_MODEL")
            if explicit:
                checkpoint = Path(explicit)
            else:
                url = os.environ.get("LAMA_MODEL_URL", "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt")
                checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(url).path).name
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Offline LaMa checkpoint missing: {checkpoint}. Download it first or set LAMA_MODEL to a local file.")
        return SimpleLama(device=torch.device("cpu"))
    def move(model, device):
        model.device = torch.device(device)
        model.model.to(model.device)
        return model
    if pool is not None:
        lama = pool.acquire("lama", load, move)
    else:
        lama = move(load(), "cuda" if torch.cuda.is_available() else "cpu")
    pending = complex_mask.copy()
    for x1, y1, x2, y2 in complex_boxes:
        if cancel:
            cancel()
        local_mask = pending[y1:y2, x1:x2]
        if not np.any(local_mask):
            continue
        # 用局部裁图限制 GPU 激活显存。
        crop = Image.fromarray(result[y1:y2, x1:x2])
        repaired = lama(crop, Image.fromarray(local_mask))
        if repaired.width < crop.width or repaired.height < crop.height:
            raise RuntimeError(f"LaMa returned {repaired.size}, smaller than input crop {crop.size}")
        repaired_array = np.asarray(repaired.convert("RGB"))[:crop.height, :crop.width]
        # 只修改掩膜内的像素，保留外面的线稿。
        result[y1:y2, x1:x2][local_mask > 0] = repaired_array[local_mask > 0]
        local_mask[:] = 0  # 重叠裁图不重复修复同一像素。
    return Image.fromarray(result)


def render(image, blocks, font_path=None, target_language="en", min_font_size=12, max_font_size=40):
    prepare_layout(image, blocks, font_path, target_language, min_font_size, max_font_size)
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    gray = np.asarray(image.convert("L"))
    for block in blocks:
        if not _eligible(block):
            continue
        layout = _metadata(block)["layout"]
        x1, y1, x2, y2 = layout["safe_box"]
        roi = gray[y1:y2, x1:x2]
        dark = bool(roi.size and np.median(roi) < 127)
        fill, stroke = ((255, 255, 255), (0, 0, 0)) if dark else ((0, 0, 0), (255, 255, 255))
        font = _load_font(layout["font_path"], layout["font_size"])
        y = y1 + (y2 - y1 - layout["height"]) / 2
        for line in layout["lines"]:
            if line:
                bbox = draw.textbbox((0, 0), line, font=font, stroke_width=layout["stroke_width"])
                x = x1 + (x2 - x1 - (bbox[2] - bbox[0])) / 2 - bbox[0]
                line_y = y + (layout["line_height"] - (bbox[3] - bbox[1])) / 2 - bbox[1]
                draw.text((round(x), round(line_y)), line, font=font, fill=fill,
                          stroke_width=layout["stroke_width"], stroke_fill=stroke)
            y += layout["line_height"] + layout["line_gap"]
    return result
