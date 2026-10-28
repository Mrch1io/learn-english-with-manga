from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict

from core.runtime import Config, ModelPool, TextBlock
from core.memory import GlossaryMemory

CancelCheck = Callable[[], None] | None
Generator = Callable[..., str]
LoadGenerator = Callable[[ModelPool, Config], Generator]
ExtractCandidates = Callable[
    [list[TextBlock], Generator, GlossaryMemory, Config, str, CancelCheck],
    tuple[int, list[str], int, float],
]
Speakers = dict[str, dict[str, Any]]
Term = dict[str, Any]
RequestLog = list[dict[str, Any]]


class StatsResult(TypedDict):
    generation_calls: int
    candidate_calls: int
    candidates_added: int
    candidate_seconds: float
    warnings: list[str]
    linked_units: NotRequired[int]


@dataclass(slots=True)
class TranslationStats:
    generation_calls: int = 0
    candidate_calls: int = 0
    candidates_added: int = 0
    candidate_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_result(self) -> StatsResult:
        return {
            "generation_calls": self.generation_calls,
            "candidate_calls": self.candidate_calls,
            "candidates_added": self.candidates_added,
            "candidate_seconds": self.candidate_seconds,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class RunState:
    stats: TranslationStats = field(default_factory=TranslationStats)
    used_terms: set[str] = field(default_factory=set)
    candidate_rows: list[Any] = field(default_factory=list)
    structured_ids: set[str] = field(default_factory=set)


@dataclass(slots=True)
class Prompt:
    content: str | list[dict[str, str]]
    terms: list[Term]
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationResponse:
    translations: dict[str, str]
    errors: dict[str, str]
    candidates: list[Any] = field(default_factory=list)
    terms_complete: bool = False
