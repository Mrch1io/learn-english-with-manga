from __future__ import annotations

from collections.abc import Callable


class ContextTooLong(ValueError):
    pass


class InvalidTranslationInput(ValueError):
    """需要先修正原文或术语，重试生成无效。"""


def _check_cancel(cancel: Callable[[], None] | None) -> None:
    if cancel is not None:
        cancel()

