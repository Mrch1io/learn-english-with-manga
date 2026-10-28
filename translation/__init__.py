"""按气泡 ID 翻译；保留模型加载与术语提取的可替换入口。"""
from __future__ import annotations

from collections.abc import Collection

from core.runtime import Config, ModelPool, TextBlock
from core.memory import GlossaryMemory
from . import qwen, workflow
from .backends import QwenBackend, VntlBackend
from .prompts import _context, _groups, _messages
from .qwen import _windows_streaming_weights
from .responses import (
    _has_japanese_text, _has_prohibition, _json_response, _normalize_punctuation,
    _parse_individual, _strict_object, parse_group_response, terminology_conflicts,
)
from .shared import ContextTooLong, InvalidTranslationInput, _check_cancel
from .terminology import _candidate_messages, _candidate_rows, _extract_candidates, _save_candidates
from .state import CancelCheck, Generator, RequestLog, Speakers, StatsResult


def _load_generator(pool: ModelPool, config: Config) -> Generator:
    return qwen.load_generator(pool, config, streaming_weights=_windows_streaming_weights)


def translate(blocks: list[TextBlock], *, pool: ModelPool | None, memory: GlossaryMemory,
              config: Config, page_key: str, cancel: CancelCheck = None, generator: Generator | None = None,
              target_ids: Collection[str] | None = None, record_page: bool = True,
              request_log: RequestLog | None = None, speakers: Speakers | None = None) -> StatsResult:
    """原地翻译选中气泡，保留全页上下文；未知或重复 ID 在修改前拒绝。

    target_ids=None 处理全页，空集合不处理。record_page=False 将依赖记录交给调用者。
    request_log 保存翻译请求和响应，不含术语提取；speakers 可传入已确认角色快照。
    """
    return workflow.translate(
        blocks, pool=pool, memory=memory, config=config, page_key=page_key,
        load_generator=_load_generator, extract_candidates=_extract_candidates,
        backend_type=VntlBackend if config.translation_backend == "vntl" else QwenBackend,
        cancel=cancel, generator=generator, target_ids=target_ids,
        record_page=record_page, request_log=request_log, speakers=speakers)
