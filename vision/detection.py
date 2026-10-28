"""YOLO / Magi 检测与日文 OCR。"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import replace
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING
from core.runtime import Cancelled, Config, ModelPool, TextBlock, require

if TYPE_CHECKING:
    from numpy import ndarray
    from PIL import Image


def order_blocks(blocks, direction="rtl"):
    """按行重叠排序，不合并独立检测框；复杂分镜仍需人工复核。"""
    rows = []
    for block in sorted(blocks, key=lambda b: (b.box[1], b.box[0])):
        y1, y2 = block.box[1], block.box[3]
        row = next((r for r in rows if
                    min(r[1], y2) - max(r[0], y1) >= 0.5 * min(r[1]-r[0], y2-y1)), None)
        if row is None:
            rows.append([y1, y2, [block]])
        else:
            row[2].append(block)
            row[0], row[1] = min(row[0], y1), max(row[1], y2)
    ordered = [b for row in sorted(rows, key=lambda r: r[0])
               for b in sorted(row[2], key=lambda b: b.box[0], reverse=direction == "rtl")]
    for index, block in enumerate(ordered):
        block.reading_order = index
    return ordered


def detect(image, conf=0.5, device="cuda:0", *, pool=None, config=None):
    config = config or Config(device=device, detection_conf=conf)
    pool = pool or ModelPool(config.device, config.memory_policy)
    path = Path(config.model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Detection model missing: {path}")
    model = pool.acquire("detector", lambda: require("ultralytics").YOLO(str(path)),
                         lambda model, target: model.to(target))
    # 直接传入 PIL 图像，由 Ultralytics 处理 RGB/BGR 转换。
    results = model.predict(image, conf=config.detection_conf, device=pool.device, verbose=False)
    blocks, seen = [], set()
    for result in results:
        for coords in result.boxes.xyxy.detach().cpu().tolist():
            x1, y1, x2, y2 = (round(v) for v in coords)
            box = (max(0, min(image.width, x1)), max(0, min(image.height, y1)),
                   max(0, min(image.width, x2)), max(0, min(image.height, y2)))
            if box in seen or box[2] <= box[0] or box[3] <= box[1]:
                continue
            seen.add(box)
            block_id = "b-" + hashlib.sha256(str(box).encode()).hexdigest()[:10]
            blocks.append(TextBlock(box=box, block_id=block_id))
    if config.detect_free_text:
        regions = detect_page_text(image, pool, config)
        audit = audit_text_regions(blocks, regions, image.size) if config.ocr_candidate_review else None
        blocks = add_text_regions(blocks, regions, image.size)
        if audit is not None:
            for block in blocks:
                block.metadata["ocr_box_audit"] = [entry for entry in audit
                    if block.block_id in entry["related_ids"] or
                    _intersection(block.box, entry["crop_box"]) /
                    ((entry["crop_box"][2] - entry["crop_box"][0]) *
                     (entry["crop_box"][3] - entry["crop_box"][1])) >= .8]
    return order_blocks(blocks, config.reading_direction)


def _text_proposals(model, image, threshold=0.3):
    """仅做一次 Magi 检测，不计算角色特征或关联。"""
    np, torch = require("numpy"), require("torch")
    with torch.inference_mode():
        inputs = model.move_to_device(model.processor.preprocess_inputs_for_detection([np.array(image)]))
        output = model._get_detection_transformer_output(**inputs)
        logits, boxes = model._get_predicted_bboxes_and_classes(output)
        scores, labels = logits.max(-1)
        scores = scores.sigmoid()
        centers, sizes = boxes[..., :2], boxes[..., 2:]
        boxes = torch.cat((centers-sizes/2, centers+sizes/2), -1)
        boxes *= boxes.new_tensor([image.width, image.height, image.width, image.height])
        indices = model.processor._get_indices_of_texts_to_keep(scores, labels, boxes, threshold)[0]
        # 一次传回 CPU，避免逐框读取引发 CUDA 同步。
        values = torch.cat((boxes[0, indices], scores[0, indices, None]), -1).cpu().tolist()
    return [{"box": row[:4], "score": row[4]} for row in values]


def detect_page_text(image, pool, config, *, min_text_score=.3):
    from .characters import load_magi

    if not .1 <= min_text_score <= 1:
        raise ValueError("Magi text score must be between 0.1 and 1")
    model = load_magi(pool, config.magi_model_path)
    proposals = _text_proposals(model, image, threshold=.1)
    high = [region for region in proposals if region["score"] >= min_text_score]
    regions = []
    for region in high:
        line = locate_wide_line(image, region["box"])
        regions.append(dict(region, **line) if line else region)
    # 低分候选需有独立成行的文字轮廓，避免把画稿当成对白。
    checked = 0
    for region in sorted(proposals, key=lambda item: -item["score"]):
        if region["score"] >= min_text_score or not _wide_box(region["box"]):
            continue
        if any(_overlap_small(region["box"], other["box"]) >= .5 for other in high):
            continue
        if checked == 4:
            break
        checked += 1
        line = locate_wide_line(image, region["box"])
        if line:
            regions.append({**region, **line, "detection_review":
                "框外横排文字由低分候选恢复，请核对 OCR 原文。" + line.get("detection_review", "")})
    return regions


def _wide_box(box):
    return box[3] > box[1] and box[2] - box[0] >= 6 * (box[3] - box[1])


def _overlap_small(first, second):
    area = min((first[2] - first[0]) * (first[3] - first[1]),
               (second[2] - second[0]) * (second[3] - second[1]))
    return _intersection(first, second) / area if area > 0 else 0


def _line_components(stats: ndarray, width: int, height: int) -> list[tuple[int, ...]]:
    return sorted((tuple(map(int, value)) for value in stats[1:]
        if value[0] > 0 and value[1] > 0 and value[0] + value[2] < width
        and value[1] + value[3] < height and max(6, height * .1) <= value[3] <= height * .6
        and value[4] >= 8 and value[2] <= width * .6), key=lambda value: value[0])


def _group_line_components(active: list[tuple[int, ...]], gap: float) -> list[list[int]]:
    groups = []
    for x, y, w, h, _ in active:
        if not groups or x - groups[-1][2] > gap:
            groups.append([x, y, x + w, y + h, 1])
        else:
            group = groups[-1]
            group[1], group[2], group[3] = min(group[1], y), max(group[2], x + w), max(group[3], y + h)
            group[4] += 1
    return groups


def _best_text_row(gray: ndarray, thresholds: tuple[float, float], cv2: ModuleType, np: ModuleType):
    """只扫描连通域边缘行；同分时保留先遇到的候选。"""
    height, width = gray.shape
    best, candidate_rows = None, set()
    for light, threshold in zip((True, False), thresholds):
        ink = ((gray > threshold) if light else (gray < threshold)).astype("uint8")
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        components = _line_components(stats, width, height)
        rows = sorted({y for _, y, _, _, _ in components} |
                      {y + h - 1 for _, y, _, h, _ in components})
        for row in rows:
            active = [value for value in components if value[1] <= row < value[1] + value[3]]
            if len(active) < 4:
                continue
            gap = float(np.median([value[3] for value in active])) * 1.6
            for x1, y1, x2, y2, count in _group_line_components(active, gap):
                if count < 4 or x2 - x1 < 4 * (y2 - y1):
                    continue
                candidate_rows.add((light, x1, y1, x2, y2))
                score = count * (x2 - x1) / (y2 - y1)
                if best is None or score > best[0]:
                    best = (score, light, float(threshold), (x1, y1, x2, y2), stats)
    return best, candidate_rows


def _include_line_punctuation(bounds: tuple[int, int, int, int], stats: ndarray,
                              width: int, height: int) -> tuple[int, int, int, int]:
    """补回同行两端小标点，保持原行高作为可达范围。"""
    x1, y1, x2, y2 = bounds
    row_left, row_right, row_top, row_bottom = x1, x2, y1, y2
    glyph_height = row_bottom - row_top
    for x, y, w, h, area in stats[1:]:
        if (x <= 0 or y <= 0 or x + w >= width or y + h >= height or area < 2
                or w > glyph_height * .55 or h > glyph_height
                or area > glyph_height * glyph_height * .35):
            continue
        near_end = (row_left - glyph_height <= x + w and x <= row_left + glyph_height * .5
                    or row_right - glyph_height * .5 <= x + w and x <= row_right + glyph_height)
        within_reach = row_left - glyph_height <= x and x + w <= row_right + glyph_height
        same_row = y < row_bottom + glyph_height * .2 and y + h > row_top - glyph_height * .15
        if near_end and within_reach and same_row:
            x1, y1, x2, y2 = min(x1, int(x)), min(y1, int(y)), max(x2, int(x + w)), max(y2, int(y + h))
    return x1, y1, x2, y2


def _line_cuts(ink: ndarray) -> tuple[list[int], bool]:
    line_height, line_width = ink.shape
    parts = min(8, max(1, math.ceil(line_width / (5 * line_height))))
    profile = ink.sum(axis=0)
    cuts, uncertain = [0], False
    for part in range(1, parts):
        target, radius = line_width * part / parts, line_width / parts * .18
        positions = range(max(cuts[-1] + 1, round(target - radius)),
                          min(line_width - 1, round(target + radius)) + 1)
        cut = min(positions, key=lambda x: (int(profile[max(0, x - 1):x + 1].sum()), abs(x - target)))
        uncertain |= bool(profile[max(0, cut - 1):cut + 1].sum())
        cuts.append(cut)
    cuts.append(line_width)
    return cuts, uncertain


def locate_wide_line(image: Image.Image, box: Sequence[float]) -> dict | None:
    """按图像对比定位横排文字并在空隙切分；几何证据不能代替人工复核。"""
    if not _wide_box(box):
        return None
    cv2, np = require("cv2", "opencv-python"), require("numpy")
    left, top = max(0, math.floor(box[0]) - 4), max(0, math.floor(box[1]) - 4)
    right, bottom = min(image.width, math.ceil(box[2]) + 4), min(image.height, math.ceil(box[3]) + 4)
    if right <= left or bottom <= top:
        return None
    gray = np.asarray(image.crop((left, top, right, bottom)).convert("L"))
    height, width = gray.shape
    low, middle, high = np.percentile(gray, [10, 50, 90])
    # 细笔画占比可能不足 10%，分位数会漏掉它们。
    if low == middle:
        low = float(gray.min())
    if high == middle:
        high = float(gray.max())
    if high - low < 24:
        return None
    best, candidate_rows = _best_text_row(gray, ((middle + high) / 2, (low + middle) / 2), cv2, np)
    if best is None:
        return None
    _, light, threshold, (x1, y1, x2, y2), stats = best
    # 多行候选保留完整裁图，避免只取最强一行而漏字。
    if any(polarity == light and min(y2, end) - max(y1, start) <= 0
           and end - start >= .6 * (y2 - y1)
           and min(x2, right_edge) - max(x1, left_edge) >= .5 * (x2 - x1)
           for polarity, left_edge, start, right_edge, end in candidate_rows):
        return {"box": [max(0, math.floor(box[0])), max(0, math.floor(box[1])),
                        min(image.width, math.ceil(box[2])), min(image.height, math.ceil(box[3]))],
                "detection_review": "宽文字框内有多行候选，已保留完整区域，请核对 OCR 是否漏行。"}
    x1, y1, x2, y2 = _include_line_punctuation((x1, y1, x2, y2), stats, width, height)
    # 留出抗锯齿边缘。
    x1, y1, x2, y2 = max(0, x1 - 4), max(0, y1 - 4), min(width, x2 + 4), min(height, y2 + 4)
    local = gray[y1:y2, x1:x2]
    ink = (local > threshold) if light else (local < threshold)
    cuts, uncertain = _line_cuts(ink)
    page_box = [left + x1, top + y1, left + x2, top + y2]
    # 相邻裁片共用边界，接缝不丢像素也不重复识字。
    segments = [[page_box[0] + start, page_box[1], page_box[0] + end, page_box[3]]
                for start, end in zip(cuts, cuts[1:])]
    return {"box": page_box, "ocr_segments": segments,
            "ocr_split_review": uncertain, "original_text_box": list(box)}


def _intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def validate_ocr_crop(box, size):
    """仅接受图内非空整数矩形，不自动裁剪或取整。"""
    if (not isinstance(size, (tuple, list)) or len(size) != 2 or
            any(type(value) is not int or value <= 0 for value in size)):
        raise ValueError("OCR image size must contain two positive integers")
    if (not isinstance(box, (tuple, list)) or len(box) != 4 or
            any(type(value) is not int for value in box)):
        raise ValueError("OCR crop coordinates must be four integers")
    if not 0 <= box[0] < box[2] <= size[0] or not 0 <= box[1] < box[3] <= size[1]:
        raise ValueError("OCR crop must be nonempty and inside the image")
    return tuple(box)


def audit_text_regions(blocks, regions, size):
    """以原始检测框为基准生成几何复核建议，不修改或合并区域。

    只比较矩形交集，复杂度 O(B * R)，不调用模型。
    """
    validate_ocr_crop((0, 0, 1, 1), size)
    originals = sorted(((block.block_id, validate_ocr_crop(block.box, size))
                        for block in blocks), key=lambda item: (item[0], item[1]))
    entries = []
    for region in regions:
        raw, score = region["box"], region["score"]
        if (not isinstance(raw, (tuple, list)) or len(raw) != 4 or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in raw) or
                raw[2] <= raw[0] or raw[3] <= raw[1]):
            raise ValueError("Magi returned invalid text coordinates")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Magi returned invalid text score")
        box = (max(0, math.floor(raw[0])), max(0, math.floor(raw[1])),
               min(size[0], math.ceil(raw[2])), min(size[1], math.ceil(raw[3])))
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        overlaps = []
        for block_id, original in originals:
            intersection = _intersection(original, box)
            if intersection:
                overlaps.append({"block_id": block_id, "box": list(original),
                    "intersection_area": intersection, "proposal_coverage": intersection / area,
                    "block_coverage": intersection / ((original[2] - original[0]) *
                                                       (original[3] - original[1]))})
        if not overlaps:
            classification, reason = "new", "No overlap with an original detection."
        elif len(overlaps) > 1:
            classification, reason = "conflict", "Overlaps multiple original detections; do not merge."
        elif overlaps[0]["proposal_coverage"] >= .8:
            classification, reason = "covered", "At least 80% of the proposal is covered by one original detection."
        elif overlaps[0]["block_coverage"] == 1:
            classification, reason = "expansion_candidate", "Contains one original detection; extra pixels require manual review."
        else:
            classification, reason = "conflict", "Partial overlap; the target region is ambiguous."
        entries.append({"classification": classification, "crop_box": list(box), "raw_box": list(raw),
                        "related_ids": [overlap["block_id"] for overlap in overlaps],
                        "overlaps": overlaps, "score": score, "reason": reason})
    return sorted(entries, key=lambda entry: (tuple(entry["crop_box"]), -entry["score"],
                                             tuple(entry["raw_box"])))


def add_text_regions(blocks, regions, size):
    """保留气泡 ID，仅补充未覆盖的文字区。"""
    width, height = size
    added = list(blocks)
    for region in sorted(regions, key=lambda item: -item["score"]):
        raw = region["box"]
        if len(raw) != 4 or not all(math.isfinite(value) for value in raw):
            raise ValueError("Magi returned invalid text coordinates")
        # 向外取整，保留字形边缘的抗锯齿像素。
        box = (max(0, math.floor(raw[0])), max(0, math.floor(raw[1])),
               min(width, math.ceil(raw[2])), min(height, math.ceil(raw[3])))
        area = (box[2]-box[0])*(box[3]-box[1])
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if any(_intersection(block.box, box) / area >= .8 for block in added):
            continue
        block_id = "t-" + hashlib.sha256(str(box).encode()).hexdigest()[:10]
        metadata = {"kind": "free_text", "detector": "magi", "detection_score": region["score"]}
        for key in ("ocr_segments", "ocr_split_review", "original_text_box"):
            if key in region:
                metadata[key] = region[key]
        review = region.get("detection_review")
        messages = ([review if isinstance(review, str) else "框外文字检测需要复核，请核对 OCR 原文。"]
                    if review else [])
        if region.get("ocr_split_review"):
            messages.append("横行切分点未找到完全空白，请检查分段接缝是否漏字或重复。")
        if messages:
            metadata["detection_review"] = " ".join(messages)
        added.append(TextBlock(box=box, block_id=block_id, metadata=metadata))
    return added


def shrink_crop_to_text(pil_crop):
    """忽略连接边缘的线稿来收紧裁图；没有可信字形时保留原区域。"""
    cv2, np = require("cv2", "opencv-python"), require("numpy")
    gray = np.asarray(pil_crop.convert("L"))
    if min(gray.shape) < 6:
        return pil_crop
    inverse = float(np.median(gray)) >= 127
    _, binary = cv2.threshold(gray, 0, 255,
                              (cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY) | cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    selected = []
    h, w = gray.shape
    for x, y, cw, ch, area in stats[1:]:
        if x <= 0 or y <= 0 or x + cw >= w or y + ch >= h:
            continue
        if area >= 2 and area < w*h*0.3:
            selected.append((x, y, x+cw, y+ch))
    if not selected:
        return pil_crop
    x1 = max(0, min(v[0] for v in selected)-3)
    y1 = max(0, min(v[1] for v in selected)-3)
    x2 = min(w, max(v[2] for v in selected)+3)
    y2 = min(h, max(v[3] for v in selected)+3)
    return pil_crop.crop((x1, y1, x2, y2))


def _acquire_ocr(pool, config):
    def load():
        model_path = "kha-white/manga-ocr-base"
        if config.local_files_only:
            hub = require("huggingface_hub")
            model_path = hub.snapshot_download(model_path, local_files_only=True,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "pytorch_model.bin"])
        # Manga-OCR 构造时会预热，直接在推理 GPU 上运行以免 CPU 生成过慢。
        with require("torch").cuda.device(pool.device):
            return require("manga_ocr", "manga-ocr").MangaOcr(model_path, force_cpu=False)
    def move(model, target):
        model.model.to(target)
        # 兼容通过参数设备或 device 属性判断设备的 Manga-OCR 版本。
        if hasattr(model, "device"):
            model.device = require("torch").device(target)
    return pool.acquire("ocr", load, move)


def preview_ocr_crop(image, block, crop_box, *, pool, config, cancel=None):
    """识别指定裁图，保持原文不变。

    摘要覆盖 UTF-8 ``mode\0width\0height\0`` 和像素字节。
    Manga-OCR 原生接口不提供置信度。
    """
    box = validate_ocr_crop(crop_box, image.size)
    if cancel:
        cancel()
    crop = image.crop(box)
    digest = hashlib.sha256(f"{crop.mode}\0{crop.width}\0{crop.height}\0".encode("utf-8"))
    digest.update(crop.tobytes())
    result = {"engine": "manga-ocr", "model": "kha-white/manga-ocr-base",
              "block_id": block.block_id, "crop_box": list(box), "crop_sha256": digest.hexdigest(),
              "raw_text": "", "text": "", "status": "failed", "error": "", "confidence": None}
    started = time.perf_counter()
    try:
        if config.use_hybrid_ocr:
            from .hybrid_ocr import recognize_crops
            row = recognize_crops([crop], pool=pool, config=replace(config, local_files_only=True), cancel=cancel)[0]
            selected = row["secondary"] if row["selected_engine"] == "paddleocr-vl-1.6" else row["primary"]
            result.update(engine=row["selected_engine"],
                          model="PaddlePaddle/PaddleOCR-VL-1.6" if row["selected_engine"] == "paddleocr-vl-1.6"
                          else "kha-white/manga-ocr-base",
                          raw_text=selected.get("raw_text", ""), text=row["text"],
                          confidence=selected.get("confidence"), hybrid=row,
                          status="ok" if row["text"] else "empty",
                          error="" if row["text"] else "Both OCR engines returned no usable text; original retained.")
            result["seconds"] = time.perf_counter() - started
            return result
        mocr = _acquire_ocr(pool, replace(config, local_files_only=True))
        if cancel:
            cancel()
        raw_text = mocr(crop)
        if cancel:
            cancel()
        if not isinstance(raw_text, str):
            raise TypeError("OCR returned non-text output")
        result["raw_text"], result["text"] = raw_text, raw_text.strip()
        result["status"] = "ok" if result["text"] else "empty"
        if not result["text"]:
            result["error"] = "OCR returned no text; original source retained."
    except Cancelled:
        raise
    except Exception as exc:
        if ModelPool.is_oom(exc):
            raise
        result["error"] = f"OCR: {type(exc).__name__}: {exc}"
    result["seconds"] = time.perf_counter() - started
    return result


def ocr_crops(image, block, crop_padding):
    """原生与混合 OCR 共用同一裁图。"""
    segments = block.metadata.get("ocr_segments")
    if segments:
        for segment in segments:
            if (len(segment) != 4 or not 0 <= segment[0] < segment[2] <= image.width
                    or not 0 <= segment[1] < segment[3] <= image.height):
                raise ValueError("OCR segment is outside the image")
            yield image.crop(segment)
    else:
        x1, y1, x2, y2 = block.box
        crop = image.crop((max(0, x1-crop_padding), max(0, y1-crop_padding),
                           min(image.width, x2+crop_padding), min(image.height, y2+crop_padding)))
        yield shrink_crop_to_text(crop) if block.metadata.get("kind") != "free_text" else crop


def run_ocr(image, blocks, device="cuda:0", *, pool=None, config=None, cancel=None,
            crop_padding=4, observe_crop=None):
    if type(crop_padding) is not int or not 0 <= crop_padding <= 16:
        raise ValueError("OCR crop_padding must be an integer from 0 to 16")
    config = config or Config(device=device)
    pool = pool or ModelPool(config.device, config.memory_policy)
    if config.use_hybrid_ocr:
        from .hybrid_ocr import run_hybrid_ocr
        return run_hybrid_ocr(image, blocks, pool=pool, config=config, cancel=cancel,
                              crop_padding=crop_padding, observe_crop=observe_crop)
    mocr = _acquire_ocr(pool, config)
    for block in blocks:
        if cancel:
            cancel()
        try:
            texts = []
            for index, crop in enumerate(ocr_crops(image, block, crop_padding)):
                if cancel:
                    cancel()
                if observe_crop:
                    observe_crop(block.block_id, index, crop)
                text = mocr(crop).strip()
                if not text and block.metadata.get("ocr_segments"):
                    raise ValueError("OCR returned an empty segment; review the complete text line")
                texts.append(text)
            block.source_text = "".join(texts)
            block.status = "pending" if block.source_text else "empty"
            if not block.source_text:
                block.error = "OCR returned no text; original pixels retained."
        except Cancelled:
            raise
        except Exception as exc:
            if ModelPool.is_oom(exc):
                raise
            block.status, block.error = "failed", f"OCR: {type(exc).__name__}: {exc}"
    return blocks
