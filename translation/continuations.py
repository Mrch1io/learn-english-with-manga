from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field, replace

from core.runtime import Config, ModelPool, TextBlock
from core.memory import GlossaryMemory
from .responses import _has_prohibition, terminology_conflicts
from .state import CancelCheck, Generator, RequestLog, Speakers, StatsResult


@dataclass(slots=True)
class _ContinuationUnits:
    blocks: list[TextBlock] = field(default_factory=list)
    members: dict[str, list[TextBlock]] = field(default_factory=dict)
    requested: set[str] = field(default_factory=set)


def _make_continuation(originals: list[TextBlock], kind: str, requested: set[str],
                       memory: GlossaryMemory, page_ids: set[str],
                       existing_units: dict[str, list[TextBlock]]) -> TextBlock:
    from .vntl import normalize_word_boundary

    ids = [item.block_id for item in originals]
    unit_id = "u-" + hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    if unit_id in page_ids or unit_id in existing_units:
        raise ValueError("Continuation ID collides with a page ID")
    original_source = "".join(item.source_text.strip() for item in originals)
    model_source, normalization = original_source, None
    if kind == "fragment":
        model_source, normalization = normalize_word_boundary(
            originals[0], originals[1], memory.match(original_source))
    unit = TextBlock(originals[0].box, source_text=original_source,
        block_id=unit_id, metadata={"continuation": {"kind": kind, "members": ids,
            "link_method": "adjacent_geometry_and_fragment_text_v1" if kind == "fragment" else "explicit_grammar",
            "source_text": original_source, "translation_source_text": model_source,
            "sources": [item.source_text for item in originals],
            "boxes": [list(item.box) for item in originals],
            "safe_boxes": [list(item.safe_box or item.metadata.get("layout", {}).get("safe_box") or item.box)
                           for item in originals],
            "retained_parts": {item.block_id: item.translated_text for item in originals
                               if item.block_id not in requested}}})
    if normalization:
        unit.metadata["continuation"]["source_normalization"] = normalization
    return unit


def _collect_units(blocks: list[TextBlock], pairs: list[tuple[int, int, str]], requested: set[str],
                   memory: GlossaryMemory) -> _ContinuationUnits:
    pair_by_start = {left: (right, kind) for left, right, kind in pairs}
    page_ids = {block.block_id for block in blocks}
    units = _ContinuationUnits()
    index = 0
    while index < len(blocks):
        if index in pair_by_start:
            right, kind = pair_by_start[index]
            originals = [blocks[index], blocks[right]]
            unit = _make_continuation(originals, kind, requested, memory, page_ids, units.members)
            units.members[unit.block_id] = originals
            index = right + 1
        else:
            unit = deepcopy(blocks[index])
            units.members[unit.block_id] = [blocks[index]]
            index += 1
        units.blocks.append(unit)
        if requested.intersection(item.block_id for item in units.members[unit.block_id]):
            units.requested.add(unit.block_id)
    return units


def _write_linked_unit(unit: TextBlock, originals: list[TextBlock], requested: set[str],
                       memory: GlossaryMemory, warnings: list[str]) -> None:
    parts = unit.metadata.get("continuation_parts", ["", ""])
    changed_neighbors = [item.block_id for item, part in zip(originals, parts)
                         if item.block_id not in requested
                         and " ".join(item.translated_text.split()) != " ".join(part.split())]
    shared = {**unit.metadata["continuation"], "unit_id": unit.block_id,
              "translated_text": unit.translated_text,
              "parts": dict(zip(unit.metadata["continuation"]["members"], parts))}
    mixed = " ".join(part if item.block_id in requested else item.translated_text
                     for item, part in zip(originals, parts))
    unsafe_edit = (changed_neighbors and shared["kind"] == "prohibition"
                   and unit.status == "translated" and not _has_prohibition(mixed))
    for item, part in zip(originals, parts):
        if item.block_id not in requested:
            continue
        item.translated_text = part if unit.status == "translated" else ""
        item.status, item.error = unit.status, unit.error
        for key in ("term_conflicts", "translation_kind", "translation_rejections", "translation_responses",
                    "translation_model_id", "translation_retry_without_context", "vntl_response", "context_review"):
            item.metadata.pop(key, None)
        item.metadata.update({key: deepcopy(value) for key, value in unit.metadata.items()
                              if key not in {"continuation", "continuation_parts", "term_conflicts"}})
        item.metadata["translation_unit"] = deepcopy(shared)
        if unsafe_edit:
            item.status, item.translated_text = "failed", ""
            item.error = "当前分句与旧关联气泡组合会丢失禁止语气；请同时校对关联两泡，旧结果未自动覆盖。"
            continue
        if item.status == "translated":
            conflicts = terminology_conflicts(item.source_text, part, memory.match(item.source_text))
            if conflicts:
                item.metadata["term_conflicts"] = conflicts
        if changed_neighbors and unit.status == "translated":
            notice = ("关联气泡的新分句与现有译文不同；只保存了当前气泡，请校对关联气泡："
                      + ", ".join(changed_neighbors))
            item.metadata["context_review"] = notice
            warnings.append(notice)


def _translate_continuations(blocks: list[TextBlock], pairs: list[tuple[int, int, str]], requested: set[str], *,
                             pool: ModelPool | None, memory: GlossaryMemory, config: Config, page_key: str,
                             cancel: CancelCheck, generator: Generator | None, record_page: bool,
                             request_log: RequestLog | None, speakers: Speakers,
                             translate: Callable[..., StatsResult]) -> StatsResult:
    """关联原文仅合并一次，回写范围限于选中气泡。"""
    units = _collect_units(blocks, pairs, requested, memory)
    stats = translate(units.blocks, pool=pool, memory=memory, config=replace(config, use_continuations=False),
                      page_key=page_key, cancel=cancel, generator=generator, target_ids=units.requested,
                      record_page=False, request_log=request_log, speakers=speakers)
    stats["linked_units"] = sum(len(units.members[key]) > 1 for key in units.requested)
    for unit in units.blocks:
        if unit.block_id not in units.requested:
            continue
        originals = units.members[unit.block_id]
        if len(originals) == 1:
            target = originals[0]
            target.translated_text, target.status, target.error = unit.translated_text, unit.status, unit.error
            target.metadata = deepcopy(unit.metadata)
        else:
            _write_linked_unit(unit, originals, requested, memory, stats["warnings"])
    if record_page:
        memory.record_page(page_key, [block.source_text for block in blocks])
    return stats
