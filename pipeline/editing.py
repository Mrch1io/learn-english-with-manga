"""界面与 CLI 共用的页面编辑后端。

修改原文或采纳 OCR 候选才重新翻译；渲染使用已校验的底图，暂存后再替换结果。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from PIL import Image, ImageOps

from core.runtime import Config, ModelPool, TextBlock
from vision.graphics import build_text_mask, clean_regions, inpaint, prepare_cleanup, prepare_layout, render
from core.memory import GlossaryMemory
from .batch import (clean_path_for, configuration_fingerprint,
                      file_hash, relevant_terms_fingerprint, _png_bytes, _hash_bytes, _commit_files)
from translation import terminology_conflicts, translate
from vision.detection import preview_ocr_crop, validate_ocr_crop


def _check(cancel):
    if cancel:
        cancel()


def _ocr_source_signature(block):
    """文字或目标框变化后，旧候选不再适用。"""
    values = {"source_text": block.source_text, "translated_text": block.translated_text,
              "box": list(block.box), "last_edit": block.metadata.get("last_edit")}
    return _hash_bytes(json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _candidate_unavailable_reason(candidate, signature, size):
    if not isinstance(candidate, dict) or not candidate.get("candidate_id"):
        return "候选记录不完整；请重新预览。"
    if candidate.get("accepted"):
        return "候选已采用，保留用于查看。"
    if (candidate.get("status") != "ok" or not isinstance(candidate.get("text"), str)
            or not candidate["text"].strip()):
        return "候选为空或识别失败，不能采用。"
    if candidate.get("base_source_signature") != signature:
        return "原文或译文已修改，旧候选已过期；请重新预览以保留人工修订。"
    try:
        validate_ocr_crop(candidate.get("crop_box"), size)
    except ValueError:
        return "候选裁剪坐标无效；请重新预览。"
    return ""




class PageEditorSession:
    """会话由一个工作线程使用，模型复用到 close() 为止。"""

    def __init__(self, result_path, *, pool=None):
        path = Path(result_path).expanduser().resolve()
        self.sidecar_path = path if path.suffix.lower() == ".json" else path.with_suffix(".json")
        if not self.sidecar_path.is_file():
            raise ValueError("No page JSON found. Select a generated translated PNG or its page JSON.")
        self.record = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(self.record, dict) or not isinstance(self.record.get("blocks"), list):
            raise ValueError("Select a single-page JSON, not a batch report or glossary.")
        self.output_path = self.sidecar_path.with_suffix(".png")
        self.input_path = Path(self.record.get("input_path", "")).expanduser()
        if not self.record.get("input_path") or self.input_path.resolve() == self.output_path.resolve():
            raise ValueError("Invalid input/output paths; originals cannot be overwritten.")
        self._sidecar_hash = file_hash(self.sidecar_path)
        self.exported = self.record.get("exported", True)
        if self.exported and not self.output_path.is_file():
            raise FileNotFoundError(f"Translated image missing: {self.output_path}")
        self._output_hash = file_hash(self.output_path) if self.output_path.is_file() else None
        if self.exported and self.record.get("output_sha256") != self._output_hash:
            raise ValueError("Translated image does not match its page record; do not overwrite external changes.")
        if self.exported:
            with Image.open(self.output_path) as image:
                self.size = image.size
        else:
            self.size = tuple(self.record.get("image_size", ()))
            if len(self.size) != 2 or any(not isinstance(value, int) or value <= 0 for value in self.size):
                raise ValueError("Point-reading record needs valid original image dimensions")
        self.blocks = self._blocks(self.record)
        self.config = self._config()
        self.config.validate()
        self.pool = pool or ModelPool(self.config.device, self.config.memory_policy)
        self._own_pool = pool is None
        self.pool.local_files_only = self.config.local_files_only
        self._load_original()
        self._load_clean_base()
        # 按当前 JSON 所在目录找图片，方便整体搬迁。
        self.record["output_path"] = str(self.output_path)
        if self.clean is None and self.exported:
            self.record["editor_notice"] = "旧结果尚无清字底图；首次保存仅补做清字，不重复检测、OCR 或整页翻译。"

    def _load_original(self) -> None:
        self.original = None
        if self.input_path.is_file():
            if self.record.get("input_sha256") != file_hash(self.input_path):
                raise ValueError("Original image changed since translation. Reprocess it before editing this result.")
            with Image.open(self.input_path) as image:
                self.original = ImageOps.exif_transpose(image).convert("RGB")
            if self.original.size != self.size:
                raise ValueError("Original/output dimensions differ; this result cannot be safely edited.")

    def _load_clean_base(self) -> None:
        self.clean = None
        self.cleaned_ids = set()
        self.clean_path = None
        entry = self.record.get("clean_image")
        if entry is not None:
            if not isinstance(entry, dict):
                raise ValueError("Invalid clean image record")
            name = entry.get("file", "")
            if not name or Path(name).name != name or not name.startswith(self.output_path.stem + ".clean"):
                raise ValueError("Clean base must be an adjacent owned image")
            self.clean_path = self.output_path.parent / name
            if self.clean_path.is_file():
                if file_hash(self.clean_path) != entry.get("sha256"):
                    raise ValueError("Clean base checksum mismatch. Restore the base or reprocess the original.")
                with Image.open(self.clean_path) as image:
                    self.clean = image.convert("RGB")
                if self.clean.size != self.size:
                    raise ValueError("Clean base dimensions do not match the page")
                self.cleaned_ids = set(entry.get("cleaned_block_ids", []))
        if self.clean is None and self.original is None:
            raise ValueError("Both the clean base and original are missing; rebuilding from a translated image is unsafe.")

    def _blocks(self, record):
        allowed = {field.name for field in fields(TextBlock)}
        blocks, ids = [], set()
        for row in record["blocks"]:
            if not isinstance(row, dict):
                raise ValueError("Invalid text block")
            block = TextBlock(**{key: deepcopy(value) for key, value in row.items() if key in allowed})
            if not block.block_id or block.block_id in ids:
                raise ValueError("Missing or duplicate bubble ID in page record")
            ids.add(block.block_id)
            if (not isinstance(block.box, (list, tuple)) or len(block.box) != 4
                    or any(not isinstance(v, int) for v in block.box)):
                raise ValueError("Invalid bubble coordinates")
            x1, y1, x2, y2 = block.box
            if not (0 <= x1 < x2 <= self.size[0] and 0 <= y1 < y2 <= self.size[1]):
                raise ValueError("Bubble coordinates fall outside the image")
            if not isinstance(block.metadata, dict):
                raise ValueError("Invalid bubble metadata")
            blocks.append(block)
        return blocks

    def _config(self):
        values = self.record.get("config")
        if not isinstance(values, dict):
            # 旧版本只在批次报告中保存完整配置。
            for path in sorted(self.sidecar_path.parent.glob("run-*.json"), reverse=True):
                try:
                    run = json.loads(path.read_text(encoding="utf-8"))
                    if any(row.get("input_path") == self.record["input_path"]
                           and row.get("config_fingerprint") == self.record.get("config_fingerprint")
                           for row in run.get("pages", [])):
                        values = run.get("config")
                        if isinstance(values, dict):
                            break
                except (OSError, ValueError, AttributeError, TypeError):
                    continue
        allowed = {field.name for field in fields(Config)}
        values = {key: value for key, value in (values or {}).items() if key in allowed}
        # 旧页面沿用原来的翻译模型，局部修改不受新默认值影响。
        values.setdefault("translation_backend", "qwen")
        values.setdefault("translation_mode", "individual")
        values.update(work=self.record.get("work", "default"),
                      target_language=self.record.get("target_language", "en"))
        # 沿用原来测量排版时的字体，避免跨平台后布局变化。
        layout = next((block.metadata["layout"] for block in self.blocks
                       if block.metadata.get("layout", {}).get("font_path")), None)
        if layout:
            values["font_path"] = layout["font_path"]
            signature = layout.get("signature", [])
            if len(signature) >= 8:
                values.setdefault("min_font_size", signature[-2])
                values.setdefault("max_font_size", signature[-1])
        return Config(**values)

    def close(self):
        if self._own_pool:
            self.pool.close()

    def _unchanged(self):
        current_output = file_hash(self.output_path) if self.output_path.is_file() else None
        if file_hash(self.sidecar_path) != self._sidecar_hash or current_output != self._output_hash:
            raise ValueError("Page changed outside this editor. Reopen it before saving.")
        if self.input_path.is_file() and file_hash(self.input_path) != self.record.get("input_sha256"):
            raise ValueError("Original image changed; existing coordinates are no longer trusted.")
        if self.clean is not None and file_hash(self.clean_path) != self.record["clean_image"]["sha256"]:
            raise ValueError("Clean base changed outside this editor. Reopen the page.")

    def _stage(self, operation, cancel):
        _check(cancel)
        with ModelPool.defer_oom_retry():
            try:
                result = operation()
            except Exception as exc:
                if not ModelPool.is_oom(exc):
                    raise
                ModelPool.release_exception_frames(exc)
                self.pool.policy = "staged"
                self.pool.offload()
                _check(cancel)
                result = operation()
        _check(cancel)
        return result

    def rerender(self, block_id, translated_text, *, cancel=None):
        return self._edit(block_id, translated_text, retranslate=False, cancel=cancel)

    def list_ocr_candidates(self, block_id):
        """从当前记录读取候选，不重新推理或改写来源记录。"""
        target = next((block for block in self.blocks if block.block_id == block_id), None)
        if target is None:
            raise ValueError(f"Unknown bubble ID: {block_id}")
        signature = _ocr_source_signature(target)
        rows = []
        for saved in target.metadata.get("ocr_candidates", []):
            reason = _candidate_unavailable_reason(saved, signature, self.size)
            row = deepcopy(saved) if isinstance(saved, dict) else {"text": ""}
            row.update(usable=not bool(reason), unavailable_reason=reason)
            rows.append(row)
        return rows

    def preview_region(self, block_id, crop_box=None, *, cancel=None):
        """在工作线程生成大小受限的对齐预览，不保存或调用模型。"""
        self._unchanged()
        _check(cancel)
        target = next((block for block in self.blocks if block.block_id == block_id), None)
        if target is None:
            raise ValueError(f"Unknown bubble ID: {block_id}")
        if self.original is None:
            raise ValueError("局部对照需要原图。")
        box = validate_ocr_crop(target.box if crop_box is None else crop_box, self.size)
        width, height = box[2] - box[0], box[3] - box[1]
        scale = min(3.0, 768 / width, 768 / height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        def crop(image):
            # 先裁剪再缩放，避免重采样混入框外像素。
            return image.crop(box).resize(size, Image.Resampling.LANCZOS)
        original, translated = crop(self.original), None
        if self.exported:
            with Image.open(self.output_path) as image:
                if image.size != self.size:
                    raise ValueError("Original/output dimensions differ; local comparison unavailable")
                translated = crop(image.convert("RGB"))
        _check(cancel)
        self._unchanged()
        return {"block_id": block_id, "crop_box": list(box),
                "original": original, "translated": translated}

    def preview_ocr(self, block_id, crop_box, *, enabled=False, cancel=None):
        """单独保存 OCR 候选，保留当前文字、底图和译图。"""
        if enabled is not True:
            raise ValueError("请先启用局部 OCR 候选。")
        self._unchanged()
        _check(cancel)
        if self.original is None:
            raise ValueError("局部 OCR 需要原图，不能从译图或清字底图识别。")
        blocks = deepcopy(self.blocks)
        target = next((block for block in blocks if block.block_id == block_id), None)
        if target is None:
            raise ValueError(f"Unknown bubble ID: {block_id}")
        crop_box = validate_ocr_crop(crop_box, self.size)
        # 预览只发起一次 OCR 请求，不走阶段级 OOM 重试。
        candidate = preview_ocr_crop(self.original, target, crop_box, pool=self.pool,
                                     config=self.config, cancel=cancel)
        _check(cancel)
        candidate.update(candidate_id=uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(),
                         base_source_text=target.source_text,
                         base_source_signature=_ocr_source_signature(target), accepted=False)
        target.metadata.setdefault("ocr_candidates", []).append(candidate)
        target.metadata["ocr_review"] = "局部 OCR 候选已保存，原文未变；请在校对窗口比较后决定是否采用。"
        record = deepcopy(self.record)
        record.update(blocks=[asdict(block) for block in blocks], ocr_candidate_edits=True,
                      status="partial", message="Local OCR candidate saved; original texts retained")
        self._save(record, blocks, self.clean, set(self.cleaned_ids), False, None, cancel)
        return deepcopy(candidate)

    def apply_ocr_candidate(self, block_id, candidate_id, *, enabled=False, cancel=None):
        """采纳未过期的候选，重新翻译成功后才保存。"""
        if enabled is not True:
            raise ValueError("请先启用局部 OCR 候选。")
        self._unchanged()
        target = next((block for block in self.blocks if block.block_id == block_id), None)
        if target is None:
            raise ValueError(f"Unknown bubble ID: {block_id}")
        candidate = next((row for row in target.metadata.get("ocr_candidates", [])
                          if isinstance(row, dict) and row.get("candidate_id") == candidate_id), None)
        reason = _candidate_unavailable_reason(candidate, _ocr_source_signature(target), self.size)
        if reason:
            raise ValueError(reason)
        return self._edit(block_id, candidate["text"], retranslate=True, cancel=cancel,
                          ocr_candidate_id=candidate_id)

    def retranslate(self, block_id, source_text, *, cancel=None):
        if not isinstance(source_text, str) or not source_text.strip():
            raise ValueError("OCR 原文不能为空。若要留空气泡，请清空译文并重新排版。")
        return self._edit(block_id, source_text, retranslate=True, cancel=cancel)

    def _set_target_text(self, target: TextBlock, blocks: list[TextBlock],
                         updates: dict[str, str], *,
                         retranslate: bool, memory: GlossaryMemory, cancel,
                         ocr_candidate_id: str | None) -> dict:
        """只改工作副本；翻译失败时不触碰已保存页面。"""
        target.metadata.setdefault("original_source_text", target.source_text)
        target.metadata.setdefault("original_translated_text", target.translated_text)
        previous_layout = target.metadata.get("layout", {})
        if previous_layout.get("safe_box"):
            target.safe_box = tuple(previous_layout["safe_box"])
        target.metadata.pop("layout", None)
        target.metadata.pop("term_conflicts", None)
        target.status, target.error = "translated", ""
        stats = {"generation_calls": 0, "candidate_calls": 0}
        if retranslate:
            target.source_text = updates[target.block_id].strip()
            # 翻译自行管理重试，阶段级重跑会丢失已完成的结果。
            stats = translate(
                blocks, pool=self.pool, memory=memory,
                config=replace(self.config, translation_mode="individual", extract_terms=False),
                page_key=str(self.input_path), target_ids={target.block_id}, record_page=False,
                cancel=cancel)
            if target.status != "translated":
                raise ValueError(f"当前气泡翻译失败，旧结果未改动：{target.error}")
            target.metadata.pop("detection_review", None)
            target.metadata.pop("ocr_split_review", None)
            target.metadata.pop("ocr_review", None)
            if ocr_candidate_id is not None:
                candidate = next(row for row in target.metadata["ocr_candidates"]
                                 if row["candidate_id"] == ocr_candidate_id)
                candidate.update(accepted=True, accepted_at=datetime.now(timezone.utc).isoformat())
                # OCR 框只记录识别范围，清字和排版仍使用 box/safe_box。
                target.metadata["ocr_crop_box"] = list(candidate["crop_box"])
        else:
            target.translated_text = updates[target.block_id].strip()
            conflicts = terminology_conflicts(target.source_text, target.translated_text,
                                               memory.match(target.source_text))
            if conflicts and target.translated_text:
                target.metadata["term_conflicts"] = conflicts
        return stats

    def _layout_edit_target(self, target: TextBlock, blocks: list[TextBlock]) -> None:
        """验证当前目标的排版，拒绝侵入其他已排版区域。"""
        target.metadata["manual_blank"] = not bool(target.translated_text)
        reference = self.original if self.original is not None else self.clean
        if self.exported and target.translated_text:
            prepare_layout(reference, [target], self.config.font_path, self.config.target_language,
                           self.config.min_font_size, self.config.max_font_size)
            if target.status != "translated":
                raise ValueError(f"无法保存排版，旧结果未改动：{target.error}")
            area = target.metadata["layout"]["safe_box"]
            for other in blocks:
                other_area = other.metadata.get("layout", {}).get("safe_box")
                if other is not target and other.status == "translated" and other_area:
                    if (min(area[2], other_area[2]) > max(area[0], other_area[0])
                            and min(area[3], other_area[3]) > max(area[1], other_area[1])):
                        raise ValueError(f"当前排版区域与 {other.block_id} 重叠，无法保证独立校对；旧结果未改动。")

    def _edit_record(self, block_id: str, updates: dict[str, str], blocks: list[TextBlock],
                     targets: list[TextBlock], stats: dict, memory: GlossaryMemory) -> dict:
        target = targets[-1]
        record = deepcopy(self.record)
        record.update(schema_version=2, config=asdict(self.config), blocks=[asdict(block) for block in blocks],
                      output_path=str(self.output_path), manual_edits=True,
                      edit_revision=int(record.get("edit_revision", 0)) + 1)
        record.pop("editor_notice", None)
        if target.metadata.get("context_review"):
            record["editor_notice"] = target.metadata["context_review"]
        record.pop("error", None)
        issues = [block for block in blocks if block.status != "translated" or block.metadata.get("term_conflicts")
                  or block.metadata.get("context_review") or block.metadata.get("detection_review")
                  or block.metadata.get("ocr_review")]
        record["status"] = "partial" if issues else "done"
        record["message"] = f"Manual edit saved; {len(issues)} bubbles need review"
        record["last_edit"] = {"block_id": block_id, "block_ids": list(updates), "operation": target.metadata["last_edit"],
            "at": datetime.now(timezone.utc).isoformat(), "translation_stats": stats,
            "config_fingerprint": configuration_fingerprint(self.config),
            "terms_fingerprint": relevant_terms_fingerprint(memory, [block.source_text for block in targets])}
        return record

    def _edit(self, block_id: str, text: str | dict[str, str], *, retranslate: bool,
              cancel, ocr_candidate_id: str | None = None) -> dict:
        updates = text if not retranslate and isinstance(text, dict) else {block_id: text}
        if not updates or any(not isinstance(value, str) for value in updates.values()):
            raise ValueError("Text must be a string")
        self._unchanged()
        _check(cancel)
        started = time.perf_counter()
        blocks = deepcopy(self.blocks)
        targets = [block for block in blocks if block.block_id in updates]
        if block_id not in updates or len(targets) != len(updates):
            raise ValueError(f"Unknown bubble ID: {block_id}")
        stats = {"generation_calls": 0, "candidate_calls": 0}
        with GlossaryMemory(self.config.memory_db, self.config.work, self.config.target_language) as memory:
            for target in targets:
                stats = self._set_target_text(
                    target, blocks, updates,
                    retranslate=retranslate, memory=memory, cancel=cancel,
                    ocr_candidate_id=ocr_candidate_id)
                self._layout_edit_target(target, blocks)
                target.metadata["last_edit"] = "ocr_retranslation" if retranslate else "manual_translation"
                if ocr_candidate_id is not None:
                    target.metadata["last_edit"] = "ocr_candidate_retranslation"
            clean, cleaned_ids, dirty, output = self.clean, set(self.cleaned_ids), False, None
            if self.exported:
                clean, cleaned_ids, dirty = self._base(blocks, cancel)
                output = render(clean, blocks, self.config.font_path, self.config.target_language,
                                self.config.min_font_size, self.config.max_font_size)
                if any(block.status != "translated" for block in targets):
                    raise ValueError("无法保存排版；旧结果未改动。")
            _check(cancel)
            record = self._edit_record(block_id, updates, blocks, targets, stats, memory)
            # 只改一个气泡不更新整页指纹，其他气泡仍对应上次整页处理。
            record["last_edit"]["seconds_before_commit"] = time.perf_counter() - started
            self._save(record, blocks, clean, cleaned_ids, dirty, output, cancel)
            try:
                memory.record_page(str(self.input_path), [block.source_text for block in blocks])
            except Exception as exc:
                self.record["editor_notice"] = f"译图已保存，但术语页面索引更新失败：{exc}"
        return deepcopy(self.record)

    def _save(self, record, blocks, clean, cleaned_ids, dirty, output, cancel):
        files = []
        clean_path = self.clean_path
        if output is not None:
            clean_path = clean_path or clean_path_for(self.output_path, self.record)
            if dirty:
                clean_bytes = _png_bytes(clean)
                files.append((clean_path, clean_bytes))
                clean_digest = _hash_bytes(clean_bytes)
            else:
                clean_digest = record["clean_image"]["sha256"]
            record["clean_image"] = {"file": clean_path.name, "sha256": clean_digest,
                                     "cleaned_block_ids": sorted(cleaned_ids)}
            output_bytes = _png_bytes(output)
            record["output_sha256"] = _hash_bytes(output_bytes)
            record["exported"] = True
            files.append((self.output_path, output_bytes))
        serialized = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
        files.append((self.sidecar_path, serialized))
        self._unchanged()
        _check(cancel)
        _commit_files(files)
        # 与保存后的 JSON 类型保持一致，例如坐标元组会变成列表。
        self.record, self.blocks = json.loads(serialized), blocks
        self.exported = self.record.get("exported", True)
        self.clean, self.cleaned_ids, self.clean_path = clean, cleaned_ids, clean_path
        self._sidecar_hash = _hash_bytes(serialized)
        if output is not None:
            self._output_hash = record["output_sha256"]

    def export_image(self, *, cancel=None):
        """用已保存的文字导出译图，按需清字，不做 OCR 或翻译。"""
        self._unchanged()
        blocks = deepcopy(self.blocks)
        reference = self.original if self.original is not None else self.clean
        previously_translated = {block.block_id for block in blocks if block.status == "translated"}
        prepare_layout(reference, blocks, self.config.font_path, self.config.target_language,
                       self.config.min_font_size, self.config.max_font_size)
        invalid = [block for block in blocks if block.block_id in previously_translated and block.status != "translated"]
        if invalid:
            raise ValueError("Export layout failed; previous files retained: " + "; ".join(block.error for block in invalid))
        clean, cleaned_ids, dirty = self._base(blocks, cancel)
        output = render(clean, blocks, self.config.font_path, self.config.target_language,
                        self.config.min_font_size, self.config.max_font_size)
        record = deepcopy(self.record)
        record.update(blocks=[asdict(block) for block in blocks], exported=True,
                      last_export_at=datetime.now(timezone.utc).isoformat())
        self._save(record, blocks, clean, cleaned_ids, dirty, output, cancel)
        return deepcopy(self.record)

    def reclean(self, block_id, regions, *, cancel=None):
        """按新增清字框修补底图，再绘制现有译文。"""
        self._unchanged()
        _check(cancel)
        blocks = deepcopy(self.blocks)
        target = next((block for block in blocks if block.block_id == block_id), None)
        if target is None or target.status != "translated":
            raise ValueError("Select a successfully translated bubble before manual cleanup")
        if not isinstance(regions, (list, tuple)) or not regions:
            raise ValueError("Select at least one cleanup rectangle")
        for region in regions:
            if (not isinstance(region, (list, tuple)) or len(region) != 4
                    or any(not isinstance(value, int) or isinstance(value, bool) for value in region)):
                raise ValueError("Cleanup rectangles must contain four integer coordinates")
            x1, y1, x2, y2 = region
            for other in blocks:
                if other is not target and (min(x2, other.box[2]) > max(x1, other.box[0])
                                           and min(y2, other.box[3]) > max(y1, other.box[1])):
                    raise ValueError("Cleanup rectangle overlaps another bubble; select a smaller region")
        reference = self.original if self.original is not None else self.clean
        prepare_layout(reference, blocks, self.config.font_path, self.config.target_language,
                       self.config.min_font_size, self.config.max_font_size)
        if target.status != "translated":
            raise ValueError(f"Cannot save cleanup with invalid layout: {target.error}")
        clean, cleaned_ids, _ = self._base(blocks, cancel)
        clean = self._stage(lambda: clean_regions(clean, target, regions, pool=self.pool, cancel=cancel), cancel)
        output = render(clean, blocks, self.config.font_path, self.config.target_language,
                        self.config.min_font_size, self.config.max_font_size)
        record = deepcopy(self.record)
        record.update(blocks=[asdict(block) for block in blocks], manual_edits=True,
                      edit_revision=int(record.get("edit_revision", 0)) + 1,
                      last_edit={"block_id": block_id, "operation": "manual_cleanup",
                                 "at": datetime.now(timezone.utc).isoformat()})
        self._save(record, blocks, clean, cleaned_ids, True, output, cancel)
        return deepcopy(self.record)

    def _base(self, blocks, cancel):
        required = [block for block in blocks if block.status == "translated"]
        missing = [block for block in required if block.block_id not in self.cleaned_ids]
        if self.clean is not None and not missing:
            return self.clean, set(self.cleaned_ids), False
        if self.original is None:
            raise ValueError("This bubble was never cleaned; restore the original image to repair its background.")
        base = self.clean.copy() if self.clean is not None else self.original.copy()
        cleanup = deepcopy(missing)
        for block in cleanup:
            # 人工清空译文表示留白。
            block.translated_text = block.translated_text or "."
        if cleanup:
            _check(cancel)
            plan = prepare_cleanup(self.original, cleanup)
            repaired = self._stage(lambda: inpaint(self.original, cleanup, pool=self.pool, cancel=cancel,
                                                    plan=plan), cancel)
            mask = Image.fromarray(build_text_mask(self.original, cleanup, plan=plan))
            base = Image.composite(repaired, base, mask)
            by_id = {block.block_id: block for block in blocks}
            for block in cleanup:
                by_id[block.block_id].metadata["cleanup"] = block.metadata.get("cleanup", {})
                regions = block.metadata.get("manual_cleanup_regions")
                if regions:
                    base = self._stage(lambda: clean_regions(base, by_id[block.block_id], regions,
                                                              pool=self.pool, cancel=cancel), cancel)
        ids = self.cleaned_ids | {block.block_id for block in cleanup}
        return base, ids, True
