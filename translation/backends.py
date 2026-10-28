from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from core.runtime import Config, ModelPool, TextBlock
from core.memory import GlossaryMemory
from .prompts import _messages
from .responses import _json_response, _parse_individual, parse_group_response, terminology_conflicts
from .shared import InvalidTranslationInput, _check_cancel
from .state import (
    CancelCheck, ExtractCandidates, Generator, LoadGenerator, Prompt, RunState,
    Speakers, Term, TranslationResponse,
)
from .terminology import _save_candidates


class TranslationBackend(Protocol):
    """后端负责模型协议与验收策略；共用流程管理重试、取消和气泡写回。"""

    def prepare(self, blocks: list[TextBlock], active: list[TextBlock], cancel: CancelCheck,
                warnings: list[str]) -> tuple[list[TextBlock], list[TextBlock]]: ...

    def start(self, active: list[TextBlock], pool: ModelPool | None,
              generator: Generator | None, load_generator: LoadGenerator) -> Generator: ...

    def prompt(self, blocks: list[TextBlock], targets: list[TextBlock], *, grouped: bool,
               previous_error: str = "", include_context: bool = True) -> Prompt: ...

    def parse(self, raw: str, targets: list[TextBlock], *, grouped: bool) -> TranslationResponse: ...

    def reject_japanese(self, block: TextBlock) -> None: ...

    def locked_term_error(self, source: str, text: str, terms: list[Term]) -> str | None: ...

    def finish(self, selected: list[TextBlock], active: list[TextBlock], state: RunState,
               generator: Generator, extract: ExtractCandidates, page_key: str, cancel: CancelCheck) -> None: ...


@dataclass(slots=True)
class QwenBackend:
    config: Config
    memory: GlossaryMemory
    speakers: Speakers

    def prepare(self, blocks: list[TextBlock], active: list[TextBlock], cancel: CancelCheck,
                warnings: list[str]) -> tuple[list[TextBlock], list[TextBlock]]:
        return blocks, active

    def start(self, active: list[TextBlock], pool: ModelPool | None,
              generator: Generator | None, load_generator: LoadGenerator) -> Generator:
        if generator is not None:
            return generator
        pool = cast(ModelPool, pool)
        if getattr(pool, "vntl", None) is not None:
            pool.vntl.close()
            pool.vntl = None
        return load_generator(pool, self.config)

    def prompt(self, blocks: list[TextBlock], targets: list[TextBlock], *, grouped: bool,
               previous_error: str = "", include_context: bool = True) -> Prompt:
        messages, terms = _messages(blocks, targets, self.memory, self.config, grouped=grouped,
                                    previous_error=previous_error, include_context=include_context,
                                    speakers=self.speakers)
        return Prompt(messages, terms)

    def parse(self, raw: str, targets: list[TextBlock], *, grouped: bool) -> TranslationResponse:
        if not grouped:
            key = targets[0].block_id
            text, error, candidates = _parse_individual(raw, key)
            return TranslationResponse({key: text}, {key: error}, candidates, bool(candidates))
        result, errors, candidates = parse_group_response(raw, [block.block_id for block in targets])
        complete = bool(result) and isinstance(_json_response(raw).get("terms"), list)
        return TranslationResponse(result, errors, candidates, complete)

    def reject_japanese(self, block: TextBlock) -> None:
        block.metadata["translation_retry_without_context"] = True

    def locked_term_error(self, source: str, text: str, terms: list[Term]) -> str | None:
        return None

    def finish(self, selected: list[TextBlock], active: list[TextBlock], state: RunState,
               generator: Generator, extract: ExtractCandidates, page_key: str, cancel: CancelCheck) -> None:
        if not getattr(self.config, "extract_terms", True):
            return
        state.stats.candidates_added += _save_candidates(state.candidate_rows, selected, self.memory, page_key)
        missing = [block for block in active if block.block_id not in state.structured_ids]
        added, warnings, calls, elapsed = extract(missing, generator, self.memory, self.config, page_key, cancel)
        state.stats.candidates_added += added
        state.stats.warnings.extend(warnings)
        state.stats.candidate_calls, state.stats.candidate_seconds = calls, elapsed


@dataclass(slots=True)
class VntlBackend:
    config: Config
    memory: GlossaryMemory
    speakers: Speakers

    def prepare(self, blocks: list[TextBlock], active: list[TextBlock], cancel: CancelCheck,
                warnings: list[str]) -> tuple[list[TextBlock], list[TextBlock]]:
        if not active:
            return blocks, active
        from .vntl import validate_input

        active_ids = {block.block_id for block in active}
        prompt_blocks = []
        for block in blocks:
            _check_cancel(cancel)
            target = block.block_id in active_ids
            try:
                if target or self.config.context_chars > 0:
                    validate_input(block, self.memory.match(block.source_text), target=target,
                                   history=self.config.use_translation_history and block.status == "translated")
            except InvalidTranslationInput as exc:
                if target:
                    block.status, block.error = "failed", f"InvalidTranslationInput: {exc}"
                else:
                    warnings.append(f"{block.block_id} context omitted: {exc}")
                continue
            prompt_blocks.append(block)
        return prompt_blocks, [block for block in active if block.status == "pending"]

    def start(self, active: list[TextBlock], pool: ModelPool | None,
              generator: Generator | None, load_generator: LoadGenerator) -> Generator:
        from .vntl import prompt_id

        for block in active:
            block.metadata["translation_model_id"] = prompt_id(block.block_id)
        if generator is not None:
            return generator
        from .vntl import VntlServer

        pool = cast(ModelPool, pool)
        # 外部 llama.cpp 启动前，先释放 PyTorch 模型占用的显存。
        pool.offload()
        key = (str(Path(self.config.gguf_path).resolve()), str(Path(self.config.llama_server_path).resolve()),
               self.config.llama_context_size, self.config.device)
        if pool.vntl is not None and pool.vntl.config_key != key:
            pool.vntl.close()
            pool.vntl = None
        if pool.vntl is None:
            pool.vntl = VntlServer(self.config)
            pool.vntl.config_key = key
            name = "translation:vntl:" + str(Path(self.config.gguf_path).resolve())
            pool.load_seconds[name] = pool.load_seconds.get(name, 0) + pool.vntl.info["load_seconds"]
            pool.loads[name] = pool.loads.get(name, 0) + 1
        return pool.vntl.generate

    def prompt(self, blocks: list[TextBlock], targets: list[TextBlock], *, grouped: bool,
               previous_error: str = "", include_context: bool = True) -> Prompt:
        from .vntl import build_prompt, locked_mentions, validate_request

        messages, terms = build_prompt(blocks, targets, self.memory, self.config,
                                      previous_error=previous_error, speakers=self.speakers)
        required = {block.block_id: list(dict.fromkeys(name for _, _, name in
                    locked_mentions(block.source_text, self.memory.match(block.source_text))))
                    for block in targets}
        required = {key: names for key, names in required.items() if names}
        options: dict[str, Any] = {"ids": [block.block_id for block in targets]}
        if required:
            options["required_terms"] = required
        validate_request(options["ids"], required)
        return Prompt(messages, terms, options)

    @staticmethod
    def _record_response(raw: str, targets: list[TextBlock]) -> None:
        for block in targets:
            block.metadata["vntl_response"] = raw[:8192]
            block.metadata.setdefault("translation_responses", []).append({
                "attempt": block.metadata["translation_attempts"], "response": raw[:8192]})

    def parse(self, raw: str, targets: list[TextBlock], *, grouped: bool) -> TranslationResponse:
        from .vntl import parse_response

        # 单泡解析失败时也保留响应；分组沿用先解析、后记录的顺序。
        if not grouped:
            self._record_response(raw, targets)
        result, errors = parse_response(raw, [block.block_id for block in targets])
        if grouped:
            self._record_response(raw, targets)
        return TranslationResponse(result, errors)

    def reject_japanese(self, block: TextBlock) -> None:
        pass

    def locked_term_error(self, source: str, text: str, terms: list[Term]) -> str | None:
        if terminology_conflicts(source, text, [term for term in terms if term["locked"]]):
            return "Locked translation missing; output was not accepted"
        return None

    def finish(self, selected: list[TextBlock], active: list[TextBlock], state: RunState,
               generator: Generator, extract: ExtractCandidates, page_key: str, cancel: CancelCheck) -> None:
        if self.config.extract_terms:
            state.stats.warnings.append(
                "VNTL does not use chat/JSON candidate extraction; saved terminology is applied, new terms need confirmation/import.")
