from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING, TypedDict

from core.runtime import Config

if TYPE_CHECKING:
    from PIL.Image import Image

JsonObject = dict[str, Any]
Paths = str | Path | Sequence[str | Path]
Token = int | tuple[int, int]


class PageEntry(TypedDict):
    path: str | None
    input: str
    label: str
    root: str
    state: str


@dataclass(frozen=True, slots=True)
class OpenRequest:
    paths: Paths
    output_dir: str | None = None
    config: Config | None = None


@dataclass(frozen=True, slots=True)
class PreviewRequest:
    path: str | None
    input: str
    scale: float
    fit: tuple[int, int]
    view: str = "original"
    fit_width: bool = False
    notice: str = ""


@dataclass(frozen=True, slots=True)
class ResizeRequest:
    original: Image
    fit: tuple[int, int]
    scale: float = 0
    fit_width: bool = False


@dataclass(frozen=True, slots=True)
class ThumbnailRequest:
    path: str | None
    input: str
    index: int
    version: int = 0


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    path: str | None
    input: str
    root: str
    index: int
    config: Config
    output_dir: str


@dataclass(frozen=True, slots=True)
class ExportRequest:
    path: str
    index: int
    config: Config


@dataclass(frozen=True, slots=True)
class CharacterRequest:
    path: str
    character_id: str | None
    name: str
    voice: JsonObject | None


@dataclass(frozen=True, slots=True)
class BindingRequest:
    path: str
    block_id: str
    character_id: str


@dataclass(frozen=True, slots=True)
class CatalogRequest:
    url: str
    start_local: bool = False


@dataclass(frozen=True, slots=True)
class SpeechRequest:
    url: str
    voice: JsonObject
    path: str | None = None
    block_id: str | None = None
    sample: str = ""


@dataclass(frozen=True, slots=True)
class MagiRequest:
    paths: list[str]
    model_path: str
    device: str


@dataclass(frozen=True, slots=True)
class OpenResult:
    pages: list[PageEntry]
    errors: list[str]


@dataclass(frozen=True, slots=True)
class PreviewResult:
    record: JsonObject
    preview: Image
    original: Image
    view: str
    notice: str
    thumbnail: Image
    characters: list[JsonObject] = field(default_factory=list)
    bindings: dict[str, JsonObject] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResizeResult:
    preview: Image
    original_size: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ThumbnailResult:
    index: int
    identity: tuple[str | None, str]
    version: int = 0
    image: Image | None = None
    view: str = ""
    message: str = ""


@dataclass(frozen=True, slots=True)
class TranslationResult:
    index: int
    report: JsonObject


@dataclass(frozen=True, slots=True)
class ExportResult:
    index: int
    path: str
    record: JsonObject


@dataclass(frozen=True, slots=True)
class CharacterResult:
    characters: list[JsonObject]
    selected_id: str


@dataclass(frozen=True, slots=True)
class BindingResult:
    bindings: dict[str, JsonObject]
    warning: str = ""


@dataclass(frozen=True, slots=True)
class CatalogResult:
    url: str
    voices: list[JsonObject]


@dataclass(frozen=True, slots=True)
class ErrorResult:
    action: str
    message: str
    cancelled: bool = False


Request = (OpenRequest | PreviewRequest | ResizeRequest | ThumbnailRequest | TranslationRequest
           | ExportRequest | CharacterRequest | BindingRequest | CatalogRequest | SpeechRequest
           | MagiRequest | JsonObject | None)
Result = (OpenResult | PreviewResult | ResizeResult | ThumbnailResult | TranslationResult
          | ExportResult | CharacterResult | BindingResult | CatalogResult | ErrorResult
          | JsonObject | str | None)


class Command(NamedTuple):
    action: str
    payload: Request
    token: Token


class Event(NamedTuple):
    action: str
    result: Result
    token: Token
