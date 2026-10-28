from __future__ import annotations

from collections.abc import Callable, Collection
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, cast
import unicodedata

from core.runtime import Cancelled, Config, ModelPool, TextBlock
from core.memory import GlossaryMemory, contains_term
from .backends import TranslationBackend
from .continuations import _translate_continuations
from .prompts import _groups
from .responses import _has_japanese_text, _has_prohibition, _normalize_punctuation, terminology_conflicts
from .shared import ContextTooLong, InvalidTranslationInput, _check_cancel
from .state import (
    CancelCheck, ExtractCandidates, Generator, LoadGenerator, Prompt, RequestLog,
    RunState, Speakers, StatsResult, Term, TranslationResponse,
)


def _translation_mode(config: Config) -> str:
    if config.target_language != "en":
        raise ValueError("Only Japanese → English is validated in this release")
    mode = config.translation_mode
    if mode == "auto":
        mode = "grouped" if config.translation_backend == "vntl" else "individual"
    if mode not in {"individual", "grouped"}:
        raise ValueError("translation_mode must be auto, individual or grouped")
    if not 0 <= config.retries <= 1 or config.max_new_tokens < 1 or config.group_size < 1:
        raise ValueError("Invalid translation retry/token/group configuration")
    return mode


def _resolve_selection(blocks: list[TextBlock], target_ids: Collection[str] | None) -> tuple[list[str], set[str]]:
    # 整页校验通过后才能清空任何气泡的旧译文。
    seen, resolved_ids = set(), []
    for index, block in enumerate(blocks, 1):
        if not isinstance(block.block_id, str):
            raise ValueError("Bubble IDs must be strings")
        key = block.block_id or f"b{index:04d}"
        if key in seen:
            raise ValueError(f"Duplicate input bubble ID: {key}")
        seen.add(key)
        resolved_ids.append(key)
    if isinstance(target_ids, str):
        raise ValueError("target_ids must be a collection of IDs, not a string")
    requested = seen if target_ids is None else set(target_ids)
    if any(not isinstance(key, str) for key in requested):
        raise ValueError("target_ids must contain strings")
    unknown = requested - seen
    if unknown:
        raise ValueError("Unknown target bubble ID(s): " + ", ".join(sorted(unknown)))
    return resolved_ids, requested


def _reset_translation(block: TextBlock, key: str) -> None:
    block.block_id, block.translated_text = key, ""
    for field in ("term_conflicts", "translation_kind", "punctuation_normalized",
                  "translation_rejections", "translation_responses", "vntl_response",
                  "translation_model_id", "translation_retry_without_context",
                  "translation_unit", "context_review"):
        block.metadata.pop(field, None)
    block.metadata["translation_attempts"] = 0
    if block.status == "failed" and block.error.startswith("OCR:"):
        return
    if not block.source_text.strip():
        block.status = "empty"
        return
    if all(char.isspace() or unicodedata.category(char).startswith("P") for char in block.source_text):
        block.translated_text = unicodedata.normalize("NFKC", block.source_text).strip()
        block.status, block.error = "translated", ""
        block.metadata["translation_kind"] = "punctuation"
        return
    block.error, block.status = "", "pending"


def _prepare_targets(blocks: list[TextBlock], resolved_ids: list[str], requested: set[str]
                     ) -> tuple[list[TextBlock], list[TextBlock]]:
    selected = []
    for block, key in zip(blocks, resolved_ids):
        if key in requested:
            selected.append(block)
            _reset_translation(block, key)
    return selected, [block for block in selected if block.status == "pending"]


def _translate_linked(blocks: list[TextBlock], resolved_ids: list[str], requested: set[str], *,
                      memory: GlossaryMemory, config: Config, page_key: str, speakers: Speakers,
                      translate: Callable[..., StatsResult], **options: Any) -> StatsResult | None:
    from .vntl import continuation_pairs

    speaker_enabled = config.use_speaker_context and config.context_chars > 0
    bindings = ({key: {**role, "confirmed": True} for key, role in speakers.items()} if speaker_enabled
                else memory.list_character_bindings(str(page_key)))
    pairs = continuation_pairs(blocks, bindings,
                               fragments=config.use_fragment_links, direction=config.reading_direction)
    if not any(requested.intersection({resolved_ids[left], resolved_ids[right]})
               for left, right, _ in pairs):
        return None
    identified = deepcopy(blocks)
    for block, key in zip(identified, resolved_ids):
        block.block_id = key
    result = _translate_continuations(identified, pairs, requested, memory=memory, config=config,
                                     page_key=page_key, speakers=speakers, translate=translate, **options)
    for original, current in zip(blocks, identified):
        if current.block_id in requested:
            original.block_id, original.translated_text = current.block_id, current.translated_text
            original.status, original.error, original.metadata = current.status, current.error, current.metadata
    return result



def _reject(block: TextBlock, reason: str, text: str, *, clear_translation: bool = False) -> None:
    block.error = reason
    if clear_translation:
        block.status, block.translated_text = "pending", ""
    block.metadata.setdefault("translation_rejections", []).append({
        "attempt": block.metadata["translation_attempts"], "reason": block.error, "response": text[:1000]})


def _split_continuation(block: TextBlock, text: str, terms: list[Term]) -> None:
    from .vntl import split_sentence

    continuation = block.metadata["continuation"]
    block.metadata.pop("continuation_parts", None)
    if continuation["kind"] == "prohibition" and not _has_prohibition(text):
        raise ValueError("Complete Japanese prohibition lost its negative meaning; review required")
    pair = cast(tuple[TextBlock, TextBlock], tuple(
        TextBlock(box, source_text=source, block_id=key, safe_box=tuple(safe_box))
        for box, source, key, safe_box in zip(continuation["boxes"], continuation["sources"],
                                             continuation["members"], continuation["safe_boxes"])))
    block.metadata["continuation_parts"] = split_sentence(
        text, *pair, terms, fixed_parts=continuation["retained_parts"])


@dataclass(slots=True)
class _TranslationSession:
    prompt_blocks: list[TextBlock]
    backend: TranslationBackend
    config: Config
    memory: GlossaryMemory
    page_key: str
    cancel: CancelCheck
    generator: Generator
    request_log: RequestLog | None
    state: RunState

    def _request(self, targets: list[TextBlock], prompt: Prompt, limit: int) -> str:
        self.state.used_terms.update(term["source"] for term in prompt.terms)
        self.state.stats.generation_calls += 1
        for block in targets:
            block.metadata["translation_attempts"] += 1
        if self.request_log is not None:
            self.request_log.append({"ids": [block.block_id for block in targets],
                                     "prompt": deepcopy(prompt.content), "max_new_tokens": limit,
                                     "attempt": targets[0].metadata["translation_attempts"]})
        try:
            raw = self.generator(prompt.content, limit, **prompt.options)
        except Exception as exc:
            if self.request_log is not None:
                self.request_log[-1]["error"] = f"{type(exc).__name__}: {exc}"
            raise
        if self.request_log is not None:
            self.request_log[-1]["response"] = raw
        return raw

    def _accept(self, block: TextBlock, text: str, terms: list[Term]) -> bool:
        protected = [term["target"] for term in terms if term["locked"]
                     and any(contains_term(block.source_text, spelling)
                             for spelling in (term["source"], *term["aliases"]))]
        normalized = _normalize_punctuation(text, protected)
        if normalized != text:
            block.metadata["punctuation_normalized"] = True
        text = normalized
        if _has_japanese_text(text):
            _reject(block, "Expected English output; model returned untranslated Japanese text", text)
            self.backend.reject_japanese(block)
            return False
        conflicts = terminology_conflicts(block.source_text, text, terms)
        if conflicts:
            block.metadata["term_conflicts"] = conflicts
        else:
            block.metadata.pop("term_conflicts", None)
        reason = self.backend.locked_term_error(block.source_text, text, terms)
        if reason:
            _reject(block, reason, text, clear_translation=True)
            return False
        if block.metadata.get("continuation"):
            try:
                _split_continuation(block, text, terms)
            except ValueError as exc:
                _reject(block, str(exc), text, clear_translation=True)
                return False
        block.translated_text, block.status, block.error = text, "translated", ""
        return not conflicts

    def individual(self, block: TextBlock, attempts: int) -> None:
        for _ in range(attempts):
            _check_cancel(self.cancel)
            previous = block.error or "; ".join(block.metadata.get("term_conflicts", []))
            try:
                prompt = self.backend.prompt(
                    self.prompt_blocks, [block], grouped=False, previous_error=previous,
                    include_context=not block.metadata.get("translation_retry_without_context", False))
                raw = self._request([block], prompt, self.config.max_new_tokens)
                response = self.backend.parse(raw, [block], grouped=False)
                error = response.errors.get(block.block_id, "")
                if error:
                    block.error = error
                elif self._accept(block, response.translations.get(block.block_id, ""), prompt.terms):
                    self.state.candidate_rows.extend(response.candidates)
                    if response.terms_complete:
                        self.state.structured_ids.add(block.block_id)
                    break
            except Cancelled:
                raise
            except (ContextTooLong, InvalidTranslationInput) as exc:
                block.error = f"{type(exc).__name__}: {exc}"
                break
            except Exception as exc:
                if ModelPool.is_oom(exc):
                    raise
                block.error = f"{type(exc).__name__}: {exc}"
            _check_cancel(self.cancel)
        if block.status != "translated":
            block.status = "failed"
            block.translated_text = ""
        elif block.error:
            self.state.stats.warnings.append(f"{block.block_id} retry failed: {block.error}")
            block.error = ""

    def grouped(self, group: list[TextBlock]) -> None:
        _check_cancel(self.cancel)
        try:
            prompt = self.backend.prompt(self.prompt_blocks, group, grouped=True)
            limit = min(2048, self.config.max_new_tokens * len(group))
            raw = self._request(group, prompt, limit)
            response = self.backend.parse(raw, group, grouped=True)
        except Cancelled:
            raise
        except InvalidTranslationInput as exc:
            for block in group:
                block.status, block.error = "failed", f"InvalidTranslationInput: {exc}"
            return
        except Exception as exc:
            if ModelPool.is_oom(exc):
                raise
            response = TranslationResponse({}, {block.block_id: f"{type(exc).__name__}: {exc}" for block in group})
        self.state.candidate_rows.extend(response.candidates)
        for block in group:
            if block.block_id in response.translations:
                if self._accept(block, response.translations[block.block_id], prompt.terms):
                    if response.terms_complete:
                        self.state.structured_ids.add(block.block_id)
                    continue
            else:
                block.error = response.errors.get(block.block_id, "Missing translation")
            remaining = self.config.retries + 1 - block.metadata["translation_attempts"]
            self.individual(block, remaining)
        _check_cancel(self.cancel)

    def finish(self, selected: list[TextBlock], active: list[TextBlock], extract: ExtractCandidates) -> None:
        self.backend.finish(selected, active, self.state, self.generator, extract, self.page_key, self.cancel)
        # 新术语可能暴露本页冲突，复查仍使用原有重试预算。
        for block in selected:
            if block.status == "translated" and block.metadata.get("translation_kind") != "punctuation":
                conflicts = terminology_conflicts(block.source_text, block.translated_text,
                                                   self.memory.match(block.source_text))
                if conflicts:
                    block.metadata["term_conflicts"] = conflicts
                    remaining = self.config.retries + 1 - block.metadata["translation_attempts"]
                    if remaining > 0:
                        self.individual(block, remaining)
            if block.metadata.get("term_conflicts"):
                self.state.stats.warnings.append(f"{block.block_id}: " + "; ".join(block.metadata["term_conflicts"]))


def translate(blocks: list[TextBlock], *, pool: ModelPool | None, memory: GlossaryMemory,
              config: Config, page_key: str, load_generator: LoadGenerator, extract_candidates: ExtractCandidates,
              backend_type: Callable[[Config, GlossaryMemory, Speakers], TranslationBackend],
              cancel: CancelCheck = None, generator: Generator | None = None,
              target_ids: Collection[str] | None = None, record_page: bool = True,
              request_log: RequestLog | None = None, speakers: Speakers | None = None) -> StatsResult:
    mode = _translation_mode(config)
    resolved_ids, requested = _resolve_selection(blocks, target_ids)
    state = RunState()
    if target_ids is not None and not requested:
        return state.stats.as_result()
    speaker_enabled = config.use_speaker_context and config.context_chars > 0
    speakers = ((memory.translation_speakers(str(page_key)) if speakers is None else speakers)
                if speaker_enabled else {})
    if config.use_continuations and config.context_chars > 0:
        result = _translate_linked(
            blocks, resolved_ids, requested, memory=memory, config=config, page_key=page_key,
            speakers=speakers, pool=pool, cancel=cancel, generator=generator,
            record_page=record_page, request_log=request_log,
            translate=partial(translate, load_generator=load_generator, extract_candidates=extract_candidates,
                              backend_type=backend_type))
        if result is not None:
            return result
    backend = backend_type(config, memory, speakers)
    selected, active = _prepare_targets(blocks, resolved_ids, requested)
    prompt_blocks, active = backend.prepare(blocks, active, cancel, state.stats.warnings)
    if not active:
        if record_page:
            memory.record_page(page_key, [block.source_text for block in blocks])
        return state.stats.as_result()
    _check_cancel(cancel)
    generator = backend.start(active, pool, generator, load_generator)
    session = _TranslationSession(prompt_blocks, backend, config, memory, page_key,
                                  cancel, generator, request_log, state)
    if mode == "individual":
        for block in active:
            session.individual(block, config.retries + 1)
    else:
        for group in _groups(active, config.group_size, max(512, config.context_chars * 2)):
            session.grouped(group)
    session.finish(selected, active, extract_candidates)
    if record_page:
        memory.record_page(page_key, [block.source_text for block in blocks], state.used_terms)
    return state.stats.as_result()
