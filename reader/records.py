from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from core.runtime import Config
from .messages import JsonObject, PageEntry, Paths

if TYPE_CHECKING:
    from core.memory import GlossaryMemory


def same_scope(record: JsonObject, config: Config) -> bool:
    saved = record.get("config", {})
    return (record.get("work", "default") == config.work
            and record.get("target_language", "en") == config.target_language
            and Path(saved.get("memory_db", Config().memory_db)).resolve() == Path(config.memory_db).resolve())


def linked_translation(block_id: str, blocks: dict[str, JsonObject]) -> str:
    """用当前译文重建关联句；原文变化或成员不完整时不展示。"""
    block = blocks.get(block_id)
    if not isinstance(block, dict):
        return ""
    metadata = block.get("metadata", {})
    unit = metadata.get("translation_unit") if isinstance(metadata, dict) else None
    if not isinstance(unit, dict):
        return ""
    members, sources = unit.get("members"), unit.get("sources")
    if (not isinstance(members, (list, tuple)) or not isinstance(sources, (list, tuple))
            or len(members) < 2 or len(members) != len(sources)
            or any(not isinstance(key, str) or not key.strip() for key in members)
            or len(set(members)) != len(members) or block_id not in members):
        return ""
    parts = []
    for key, source in zip(members, sources):
        member = blocks.get(key)
        if (not isinstance(source, str) or not isinstance(member, dict)
                or member.get("block_id") != key or member.get("source_text") != source
                or member.get("status") != "translated"):
            return ""
        text = member.get("translated_text")
        if not isinstance(text, str) or not text.strip():
            return ""
        parts.append(text.strip())
    return " ".join(parts)


def _input_sources(paths: Paths) -> tuple[set[Path], set[Path], dict[Path, Path]]:
    from pipeline.batch import IMAGE_SUFFIXES

    if isinstance(paths, (str, Path)):
        paths = [paths]
    candidates: set[Path] = set()
    explicit_records: set[Path] = set()
    images: dict[Path, Path] = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            candidates.update(path.rglob("*.json"))
            images.update((p.resolve(), path) for p in path.rglob("*")
                          if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                          and not any(part.startswith(".") for part in p.relative_to(path).parts))
        else:
            sidecar = path if path.suffix.lower() == ".json" else path.with_suffix(".json")
            if sidecar.is_file() or path.suffix.lower() == ".json":
                candidates.add(sidecar)
                explicit_records.add(sidecar)
            if path.suffix.lower() in IMAGE_SUFFIXES:
                images[path] = path.parent
    return candidates, explicit_records, images


def _page_candidate(path: Path, original: Path, record: JsonObject,
                    images: dict[Path, Path]) -> tuple[PageEntry, tuple[int, int, str]]:
    fixture = record.get("presentation_fixture")
    number = str(fixture.get("source_page_id", "")).rpartition(":p")[2] if isinstance(fixture, dict) else ""
    label = f"Page {int(number)}" if number.isascii() and number.isdigit() else original.name
    candidate: PageEntry = {"path": str(path), "input": str(original), "label": label,
                 "root": str(images.get(original, original.parent)), "state": record.get("status", "done")}
    # Prefer manually reviewed records when a folder contains several results for one input.
    tier = 2 if record.get("manual_edits") else 1 if record.get("ocr_candidate_edits") else 0
    priority = (tier, path.stat().st_mtime_ns, str(path))
    return candidate, priority


def collect_pages(paths: Paths, output_dir: str | Path | None = None, *,
                  config: Config | None = None) -> tuple[list[PageEntry], list[str]]:
    """Read page records and raw images once; generated assets are not new pages."""
    from pipeline.batch import natural_key

    candidates, explicit_records, images = _input_sources(paths)
    errors: list[str] = []
    selected_records = candidates.copy()
    if output_dir and Path(output_dir).is_dir():
        candidates.update(Path(output_dir).resolve().rglob("*.json"))
    pages: dict[Path, PageEntry] = {}
    priorities: dict[Path, tuple[int, int, str]] = {}
    generated: set[Path] = set()
    for path in sorted(candidates):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or not isinstance(record.get("blocks"), list) or not record.get("input_path"):
                continue
            original = Path(record["input_path"]).resolve()
            output = Path(record.get("output_path", path.with_suffix(".png"))).resolve()
            generated.update((output, path.with_suffix(".png").resolve(), path.with_suffix(".mask.png").resolve()))
            if isinstance(record.get("clean_image"), dict) and record["clean_image"].get("file"):
                generated.add((path.parent / record["clean_image"]["file"]).resolve())
            if path in selected_records or original in images:
                if (config is not None and original in images and path not in explicit_records
                        and not same_scope(record, config)):
                    continue
                candidate, priority = _page_candidate(path, original, record, images)
                if original not in pages or priority > priorities[original]:
                    pages[original] = candidate
                    priorities[original] = priority
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{path.name}: {exc}")
    for path, root in images.items():
        if path not in generated and path not in pages:
            pages[path] = {"path": None, "input": str(path), "label": path.name,
                           "root": str(root), "state": "untranslated"}
    return sorted(pages.values(), key=lambda page: (natural_key(page["input"]), page["input"])), errors


def open_memory(record: JsonObject) -> GlossaryMemory:
    from core.memory import GlossaryMemory

    return GlossaryMemory(record.get("config", {}).get("memory_db", Config().memory_db),
                          record.get("work", "default"), record.get("target_language", "en"))


def read_record(path: str | Path) -> JsonObject:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("blocks"), list):
        raise ValueError("Choose a page JSON file or its matching PNG.")
    ids = set()
    for block in record["blocks"]:
        if not isinstance(block, dict) or not isinstance(block.get("block_id"), str) or not block["block_id"]:
            raise ValueError("A speech bubble is missing its stable ID.")
        if block["block_id"] in ids:
            raise ValueError("Speech bubble IDs must be unique.")
        ids.add(block["block_id"])
        box = block.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(isinstance(n, (int, float)) and math.isfinite(n) for n in box):
            raise ValueError("Invalid speech bubble coordinates.")
    return record
