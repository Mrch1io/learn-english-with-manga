from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
from io import BytesIO
import threading
import time
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable

from core.runtime import Cancelled, Config, ModelPool, ROOT, TextBlock
from core.memory import GlossaryMemory

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def natural_key(value):
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold())
                 for part in re.split(r"(\d+)", str(value)))


def collect_images(input_path, output_dir=None):
    if isinstance(input_path, (list, tuple)):
        pages = {page for item in input_path for page in collect_images(item, output_dir)}
        return sorted(pages, key=lambda page: (natural_key(page), str(page)))
    source = Path(input_path).expanduser().resolve()
    excluded = Path(output_dir).expanduser().resolve() if output_dir else None
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    return sorted((p.resolve() for p in source.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                   and not any(x.startswith(".") for x in p.relative_to(source).parts)
                   and not (excluded and p.resolve().is_relative_to(excluded))),
                  key=lambda p: (natural_key(p.relative_to(source)), str(p)))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def configuration_fingerprint(config):
    from vision.graphics import resolve_font
    values = asdict(config)
    # 缓存取决于相关术语和身份内容，数据库搬家不影响结果。
    values.pop("memory_db", None)
    paths = [ROOT / name for name in ("core/runtime.py", "core/memory.py", "pipeline/batch.py",
                                      "pipeline/editing.py", "vision/detection.py", "vision/graphics.py")]
    paths.extend(sorted((ROOT / "translation").rglob("*.py")))
    paths.append(Path(config.model_path))
    if config.detect_free_text:
        paths.append(ROOT / "vision/characters.py")
        magi = Path(config.magi_model_path).expanduser().resolve()
        paths.extend(magi / name for name in ("config.json", "configuration_magiv2.py",
                                              "modelling_magiv2.py", "processing_magiv2.py", "utils.py"))
        checkpoint = magi / "pytorch_model.bin"
        stat = checkpoint.stat() if checkpoint.is_file() else None
        values["magi_checkpoint"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} if stat else None
    else:
        values.pop("magi_model_path", None)
    if config.use_hybrid_ocr:
        paths.append(ROOT / "vision/hybrid_ocr.py")
        paddle = Path(config.paddle_ocr_model_path).expanduser().resolve()
        paths.extend(paddle / name for name in ("config.json", "generation_config.json", "tokenizer_config.json",
            "processor_config.json", "preprocessor_config.json", "special_tokens_map.json", "chat_template.jinja"))
        values["paddle_files"] = {}
        for name in ("model.safetensors", "tokenizer.json", "tokenizer.model", "added_tokens.json"):
            path = paddle / name
            stat = path.stat() if path.is_file() else None
            values["paddle_files"][name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} if stat else None
    else:
        values.pop("paddle_ocr_model_path", None)
        values.pop("ocr_prob_threshold", None)
    if config.translation_backend == "vntl":
        values.pop("model_name", None)
        values.pop("adapter_path", None)
        # 用路径、大小和修改时间识别模型文件，避免每次编辑都读取约 5 GB 权重。
        values["translation_files"] = {}
        for filename in (config.gguf_path, config.llama_server_path):
            path = Path(filename).expanduser().resolve()
            stat = path.stat() if path.is_file() else None
            values["translation_files"][str(path)] = (
                {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} if stat else None)
    else:
        for key in ("gguf_path", "llama_server_path", "llama_context_size"):
            values.pop(key, None)
        paths.extend([Path(config.adapter_path)/"adapter_config.json",
                      Path(config.adapter_path)/"adapter_model.safetensors"])
    if config.export_images:
        resolved_font = Path(resolve_font(config.font_path)).resolve()
        values["font_path"] = str(resolved_font)
        paths.append(resolved_font)
    values["files"] = {str(path): file_hash(path) if path.is_file() else None for path in paths}
    return digest_json(values)


def relevant_terms_fingerprint(memory, sources):
    terms = {}
    for source in sources:
        for term in memory.match(source):
            terms[term["source"]] = {k: term.get(k) for k in ("source", "target", "aliases", "locked", "status")}
    return digest_json(terms)


def relevant_speakers_fingerprint(memory, page_key, block_ids, config, *, speakers=None):
    """只计算本页已确认的姓名和身份；传入快照时与生成使用同一份数据。"""
    if not config.use_speaker_context or config.context_chars <= 0:
        return digest_json({})
    if speakers is None:
        speakers = memory.translation_speakers(str(page_key))
    return digest_json({key: {field: speakers[key][field] for field in ("character_id", "name")}
                        for key in block_ids if key in speakers})


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _hash_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _commit_files(files):
    """先暂存文件，最后替换 JSON；替换失败时回滚。

    多个文件无法一起原子替换，崩溃后靠 JSON 中的哈希检查是否保存完整。
    """
    staged, backups, installed, recovery = {}, {}, [], set()
    try:
        for path, data in files:
            if path.is_symlink():
                raise ValueError(f"Refusing to replace a symbolic link: {path}")
            descriptor, name = tempfile.mkstemp(prefix=".manga-edit-", suffix=".tmp", dir=path.parent)
            staged[path] = Path(name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                descriptor, name = tempfile.mkstemp(prefix=".manga-backup-", suffix=".tmp", dir=path.parent)
                os.close(descriptor)
                backups[path] = Path(name)
                shutil.copy2(path, name)
        for path, _ in files:
            os.replace(staged[path], path)
            installed.append(path)
    except Exception as failure:
        for path in reversed(installed):
            try:
                if path in backups:
                    os.replace(backups[path], path)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                if path in backups:
                    recovery.add(backups[path])
        if recovery:
            raise OSError("Save and rollback failed; original backups retained: "
                          + ", ".join(map(str, recovery))) from failure
        raise
    finally:
        for path in [*staged.values(), *backups.values()]:
            if path not in recovery:
                path.unlink(missing_ok=True)


def clean_path_for(output, old=None):
    """复用本页底图，避开其他同名文件。"""
    output = Path(output)
    entry = (old or {}).get("clean_image", {})
    name = entry.get("file", "")
    if name and Path(name).name == name and name.startswith(output.stem + ".clean") and name.endswith(".png"):
        return output.parent / name
    for number in range(10000):
        suffix = "" if number == 0 else f"-{number}"
        candidate = output.with_name(f"{output.stem}.clean{suffix}.png")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Cannot allocate a clean base filename")


def clean_image_valid(record, output):
    entry = record.get("clean_image")
    if entry is None:
        return True  # 旧结果由页面编辑器迁移。
    name = entry.get("file", "")
    return (bool(name) and Path(name).name == name
            and file_hash(Path(output).parent / name) == entry.get("sha256"))


def output_path_for(source, output_dir, config):
    identity = digest_json([str(Path(source).resolve()), config.work, config.target_language])[:12]
    stem = re.sub(r"[^\w.-]+", "_", Path(source).stem)[:90]
    base = Path(output_dir) / f"{stem}__{identity}.png"
    # 用配套 JSON 确认图片归属，避免覆盖其他文件。
    for suffix in range(10000):
        candidate = base if suffix == 0 else base.with_stem(f"{base.stem}_{suffix}")
        sidecar = candidate.with_suffix(".json")
        if not candidate.exists() and not sidecar.exists() and not candidate.with_name(candidate.stem + ".clean.png").exists():
            return candidate
        try:
            owner = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(owner, dict) and owner.get("input_path") == str(Path(source).resolve()) and owner.get("work") == config.work and owner.get("target_language") == config.target_language:
                return candidate
        except (OSError, ValueError):
            pass
    raise RuntimeError("Could not allocate a unique output filename")


def _read_page_record(path: Path) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return record if isinstance(record, dict) else {}
    except (OSError, ValueError):
        return {}


def _page_files(output_path: Path, config: Config, record: dict, blocks: list[TextBlock],
                canvas, output) -> list[tuple[Path, bytes]]:
    """组装提交内容，保留 JSON 最后提交的顺序。"""
    old = _read_page_record(output_path.with_suffix(".json"))
    files = []
    if config.export_images:
        clean_path = clean_path_for(output_path, old)
        clean_bytes, output_bytes = _png_bytes(canvas), _png_bytes(output)
        record["clean_image"] = {"file": clean_path.name, "sha256": _hash_bytes(clean_bytes),
            "cleaned_block_ids": [b.block_id for b in blocks
                if not config.skip_inpaint and b.status == "translated" and b.translated_text.strip()]}
        record["output_sha256"] = _hash_bytes(output_bytes)
        files.extend([(clean_path, clean_bytes), (output_path, output_bytes)])
    issues = [b for b in blocks if b.status != "translated"]
    term_conflicts = [b for b in blocks if b.metadata.get("term_conflicts")]
    text_reviews = [b for b in blocks if b.metadata.get("detection_review") or b.metadata.get("ocr_review")]
    record["status"] = "partial" if issues or term_conflicts or text_reviews else "done"
    record["message"] = (f"{len(blocks)-len(issues)}/{len(blocks)} text regions translated; "
                         f"{len(term_conflicts)} terminology warnings; {len(text_reviews)} OCR regions need review")
    record["blocks"] = [asdict(b) for b in blocks]
    return files


def process_page(path, output_path, config, pool, memory, cancel, progress=None,
                 *, input_digest=None, config_digest=None):
    from PIL import Image, ImageOps
    from vision.detection import detect, run_ocr
    from translation import translate
    from vision.graphics import prepare_layout, inpaint, render
    timings, blocks = {}, []
    started = time.perf_counter()
    record = {"input_path": str(path), "work": config.work,
              "target_language": config.target_language, "output_path": str(output_path),
              "schema_version": 2, "config": asdict(config),
              "exported": config.export_images,
              "input_sha256": input_digest or file_hash(path),
              "config_fingerprint": config_digest or configuration_fingerprint(config)}
    def stage(name, operation, *, retry_oom=True):
        cancel()
        if progress:
            progress({"type": "stage", "path": str(path), "message": name})
        start = time.perf_counter()
        try:
            with ModelPool.defer_oom_retry(retry_oom):
                try:
                    value = operation()
                except Exception as exc:
                    if not retry_oom or not pool.is_oom(exc):
                        raise
                    ModelPool.release_exception_frames(exc)
                    pool.policy = "staged"
                    pool.offload()
                    cancel()
                    record.setdefault("oom_retries", {})[name] = 1
                    value = operation()
            pool.sync()
            return value
        finally:
            timings[name] = time.perf_counter() - start
    try:
        cancel()
        with Image.open(path) as opened:
            if getattr(opened, "n_frames", 1) > 1:
                raise ValueError("Multi-frame images are not supported in v1; export individual frames first.")
            image = ImageOps.exif_transpose(opened).convert("RGB")
        record["image_size"] = list(image.size)
        blocks = stage("detect", lambda: detect(image, pool=pool, config=config))
        if blocks:
            stage("ocr", lambda: run_ocr(image, blocks, pool=pool, config=config, cancel=cancel))
        sources = [b.source_text for b in blocks]
        # 记录生成时的术语，后续新增候选不应算作本页已用过的术语。
        record["terms_fingerprint"] = relevant_terms_fingerprint(memory, sources)
        speakers = (memory.translation_speakers(str(path))
                    if config.use_speaker_context and config.context_chars > 0 else {})
        record["speakers_fingerprint"] = relevant_speakers_fingerprint(
            memory, str(path), [block.block_id or f"b{index:04d}" for index, block in enumerate(blocks, 1)],
            config, speakers=speakers)
        memory.record_page(str(path), sources)
        if blocks:
            # 翻译按气泡重试；整页重跑会丢失已完成结果并重置重试次数。
            record["translation_stats"] = stage("translate", lambda: translate(
                blocks, pool=pool, memory=memory, config=config, page_key=str(path), cancel=cancel,
                speakers=speakers), retry_oom=False)
        if blocks and config.export_images:
            stage("layout", lambda: prepare_layout(image, blocks, font_path=config.font_path,
                target_language=config.target_language, min_font_size=config.min_font_size,
                max_font_size=config.max_font_size))
            mask_path = output_path.with_name(output_path.stem+".mask.png") if config.debug_mask else None
            canvas = image.copy() if config.skip_inpaint else stage("inpaint", lambda: inpaint(
                image, blocks, pool=pool, debug_mask_path=mask_path, cancel=cancel))
            output = stage("render", lambda: render(canvas, blocks, font_path=config.font_path,
                target_language=config.target_language, min_font_size=config.min_font_size,
                max_font_size=config.max_font_size))
        else:
            canvas = output = image
        cancel()
        files = _page_files(output_path, config, record, blocks, canvas, output)
        record["timings"] = timings
        record["total_seconds"] = time.perf_counter() - started
        files.append((output_path.with_suffix(".json"), json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")))
        cancel()
        _commit_files(files)
        return record
    except Exception as exc:
        record.update(status="cancelled" if isinstance(exc, Cancelled) else "failed",
                      error=f"{type(exc).__name__}: {exc}", blocks=[asdict(b) for b in blocks])
        exc.page_record = record
        raise
    finally:
        record["timings"] = timings
        record.setdefault("total_seconds", time.perf_counter()-started)


def _hardware(pool):
    result = {"platform": platform.platform(), "python": platform.python_version(), "device": pool.requested_device}
    try:
        import torch
        result["torch"] = torch.__version__
        if torch.cuda.is_available() and pool.device.startswith("cuda"):
            result.update(device=pool.device, gpu=torch.cuda.get_device_name(pool.device),
                          vram_bytes=torch.cuda.get_device_properties(pool.device).total_memory,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(pool.device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(pool.device))
    except (ImportError, RuntimeError):
        pass
    return result


def _valid_cached_page(old: dict, path: Path, output: Path, input_digest: str,
                       fingerprint: str, config: Config, memory: GlossaryMemory) -> bool:
    try:
        images_valid = (not old.get("exported", True) or (output.is_file()
            and old.get("output_sha256") == file_hash(output)
            and clean_image_valid(old, output)))
        return (old.get("status") == "done" and images_valid
            and old.get("input_sha256") == input_digest
            and old.get("config_fingerprint") == fingerprint
            and old.get("terms_fingerprint") == relevant_terms_fingerprint(memory,
                [b.get("source_text", "") for b in old.get("blocks", [])])
            and old.get("speakers_fingerprint", digest_json({})) == relevant_speakers_fingerprint(
                memory, str(path), [b.get("block_id", "") for b in old.get("blocks", [])], config))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _batch_output_path(path: Path, sources: list[Path], destination: Path, config: Config) -> Path:
    output = output_path_for(path, destination, config)
    root = next((source for source in sources if source.is_dir() and path.is_relative_to(source)), path.parent)
    nested = destination / path.parent.relative_to(root)
    if nested != destination and not output.with_suffix(".json").is_file():
        nested.mkdir(parents=True, exist_ok=True)
        output = output_path_for(path, nested, config)
    return output


@dataclass
class _BatchRun:
    config: Config
    pool: ModelPool
    processor: Callable[..., dict]
    custom_processor: bool
    fingerprint: str
    force: bool
    report: dict
    cancel_event: threading.Event
    progress: Callable[[dict], None] | None

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled("Cancelled after the last safe step")

    def emit(self, event: dict) -> None:
        if self.progress:
            self.progress(event)

    def process_page(self, path: Path, output: Path, index: int, total: int,
                     memory: GlossaryMemory) -> None:
        sidecar = output.with_suffix(".json")
        try:
            input_digest = file_hash(path)
        except OSError as exc:
            record = {"input_path": str(path), "work": self.config.work,
                      "target_language": self.config.target_language, "status": "failed",
                      "error": f"Input read failed: {type(exc).__name__}: {exc}"}
            self.report["failed"] += 1
            self.report["pages"].append(record)
            # 原图暂时读不到时保留旧页面记录，本次错误另记在批次报告中。
            if not sidecar.exists():
                atomic_json(sidecar, record)
            self.emit({"type": "error", "current": index, "total": total,
                  "path": str(path), "message": record["error"]})
            return
        old = _read_page_record(sidecar)
        valid_cache = not self.force and _valid_cached_page(
            old, path, output, input_digest, self.fingerprint, self.config, memory)
        if valid_cache:
            self.report["skipped"] += 1
            self.report["pages"].append({"input_path": str(path), "status": "skipped", "output_path": str(output)})
            self.emit({"type": "skipped", "current": index, "total": total, "path": str(path), "message": "Unchanged result"})
            return
        if (old.get("manual_edits") or old.get("ocr_candidate_edits")) and not self.force:
            # 配置或术语变化时保留人工修订，只有 --force 才允许覆盖。
            self.report["partial"] += 1
            self.report["pages"].append({"input_path": str(path), "output_path": str(output),
                "status": "needs_review", "message": "Manual edits/OCR candidates retained; input/config/terms changed or the page needs review. Open the page editor, or use --force to replace these records."})
            self.emit({"type": "page_done", "current": index, "total": total,
                  "path": str(path), "message": self.report["pages"][-1]["message"]})
            return
        self.emit({"type": "page_start", "current": index, "total": total, "path": str(path), "message": path.name})
        self._execute(path, output, input_digest, index, total, memory)

    def _execute(self, path: Path, output: Path, input_digest: str, index: int,
                 total: int, memory: GlossaryMemory) -> None:
        sidecar = output.with_suffix(".json")
        try:
            extras = {"input_digest": input_digest, "config_digest": self.fingerprint} if not self.custom_processor else {}
            record = self.processor(path, output, self.config, self.pool, memory, self.check_cancel,
                lambda event: self.emit({**event, "current": index, "total": total}), **extras)
            record.update(input_sha256=input_digest, config_fingerprint=self.fingerprint)
            if self.custom_processor:
                atomic_json(sidecar, record)
            key = "completed" if record["status"] == "done" else "partial"
            self.report[key] += 1
        except Exception as exc:
            record = getattr(exc, "page_record", {"input_path": str(path), "work": self.config.work,
                "target_language": self.config.target_language, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            if isinstance(exc, Cancelled):
                self.report["cancelled"] = True
            else:
                self.report["failed"] += 1
            record.update(input_sha256=input_digest, config_fingerprint=self.fingerprint)
            # 本次处理失败时保留可用的旧结果，包括人工修订和底图哈希。
            if not sidecar.exists():
                try:
                    atomic_json(sidecar, record)
                except OSError as save_error:
                    record["record_save_error"] = str(save_error)
        self.report["pages"].append(record)
        self.emit({"type": "error" if record["status"] == "failed" else "page_done", "current": index,
              "total": total, "path": str(path), "message": record.get("error", record.get("message", record["status"]))})


def _finish_batch(report: dict, pool: ModelPool, config: Config, destination: Path,
                  started: float, page_processor) -> Path:
    report["total_seconds"] = time.perf_counter()-started
    report["model_load_seconds"] = dict(pool.load_seconds)
    report["model_load_counts"] = dict(pool.loads)
    report["model_evictions"] = list(getattr(pool, "evictions", []))
    report["model_transfer_seconds"] = pool.transfer_seconds
    if getattr(pool, "vntl", None) is not None:
        report["translation_runtime"] = dict(getattr(pool.vntl, "info", {}))
    durations = [p["total_seconds"] for p in report["pages"] if p.get("status") in {"done", "partial"} and "total_seconds" in p]
    report["batch_mean_page_seconds"] = sum(durations)/len(durations) if durations else None
    report["warm_mean_page_seconds"] = sum(durations[1:])/len(durations[1:]) if len(durations)>1 else None
    report["hardware"] = _hardware(pool) if page_processor is None else {"note": "test processor; no model benchmark"}
    report_path = destination / ("run-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
    report["report_path"] = str(report_path)
    atomic_json(report_path, report)
    try:
        with GlossaryMemory(config.memory_db, config.work, config.target_language) as memory:
            memory.record_run(report_path.stem, report)
    except Exception as exc:
        report["history_error"] = str(exc)
    return report_path


def run_batch(input_path, output_dir, config=None, *, progress=None, cancel_event=None,
              force=False, pool=None, page_processor=None) -> dict:
    config = config or Config()
    config.validate()
    selected = input_path if isinstance(input_path, (list, tuple)) else [input_path]
    sources = [Path(item).expanduser().resolve() for item in selected]
    destination = Path(output_dir).expanduser().resolve()
    if any(destination == (source if source.is_dir() else source.parent) for source in sources):
        raise ValueError("Choose a separate output directory to preserve original images.")
    pages = collect_images(sources, destination)
    if not pages:
        raise ValueError("No supported images were found.")
    destination.mkdir(parents=True, exist_ok=True)
    cancel_event = cancel_event or threading.Event()
    own_pool = pool is None
    pool = pool or ModelPool(config.device, config.memory_policy)
    pool.local_files_only = config.local_files_only
    processor = page_processor or process_page
    fingerprint = configuration_fingerprint(config)
    started = time.perf_counter()
    report = {"started_at": datetime.now(timezone.utc).isoformat(),
              "input": str(sources[0]) if len(sources) == 1 else [str(source) for source in sources],
              "output_dir": str(destination), "config": asdict(config), "pages": [],
              "completed": 0, "failed": 0, "partial": 0, "skipped": 0, "cancelled": False}
    batch = _BatchRun(config, pool, processor, page_processor is not None, fingerprint,
                      force, report, cancel_event, progress)
    try:
        # 只统计本次任务的显存峰值。
        if page_processor is None:
            import torch
            if pool.device.startswith("cuda"):
                torch.cuda.set_device(pool.device)
                torch.cuda.reset_peak_memory_stats(pool.device)
        with GlossaryMemory(config.memory_db, config.work, config.target_language) as memory:
            if config.target_language == "en" and (ROOT/"glossary.json").exists():
                memory.import_legacy_once(ROOT/"glossary.json")
            for index, path in enumerate(pages, 1):
                if cancel_event.is_set():
                    report["cancelled"] = True
                    break
                output = _batch_output_path(path, sources, destination, config)
                batch.process_page(path, output, index, len(pages), memory)
                if report["cancelled"]:
                    break
    finally:
        try:
            report_path = _finish_batch(report, pool, config, destination, started, page_processor)
        finally:
            # 写报告失败时也必须关闭外部模型进程。
            if own_pool:
                pool.close()
    batch.emit({"type": "complete", "message": str(report_path)})
    return report
