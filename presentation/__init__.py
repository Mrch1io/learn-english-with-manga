"""Materialize editable presentation pages with isolated shared character memory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from core.runtime import ROOT


BUNDLE = Path(__file__).resolve().parent
DEMO_TITLE = "Tencho Isoro · A new beginning"
DEMO_WORK = "presentation:tencho_isoro"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_characters(destination: Path, record: dict, *, bundle: Path | None = None,
                     database: Path | None = None) -> None:
    """原图、原文和坐标匹配时才补齐演示角色与绑定，保留已有值。"""
    from core.memory import GlossaryMemory

    bundle = BUNDLE if bundle is None else bundle
    database = destination / "memory.sqlite3" if database is None else database
    canonical = json.loads((bundle / "daily-life.json").read_text(encoding="utf-8"))
    config = record.get("config", {})
    fixture = record.get("presentation_fixture", {})
    if (not isinstance(config, dict) or not isinstance(fixture, dict)
            or record.get("work") != DEMO_WORK
            or config.get("work") != DEMO_WORK
            or record.get("target_language") != "en"
            or config.get("target_language") != "en"
            or fixture.get("source_page_id") != canonical["presentation_fixture"]["source_page_id"]):
        return
    original = destination / "original.jpg"
    try:
        # Reject external/symlinked paths rather than opening another work's DB.
        if (Path(config.get("memory_db", "")).resolve() != database
                or Path(record.get("input_path", "")).resolve() != original
                or not original.is_file()):
            return
    except (TypeError, ValueError, OSError):
        return
    if (record.get("input_sha256") != canonical["input_sha256"]
            or _sha256(original) != canonical["input_sha256"]):
        return
    seeds = json.loads((bundle / "character-seeds.json").read_text(encoding="utf-8"))
    expected = {block["block_id"]: block for block in canonical["blocks"]}
    eligible = set()
    blocks = record.get("blocks")
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict) or not isinstance(block.get("block_id"), str):
            continue
        source = expected.get(block.get("block_id"))
        if source and block.get("source_text") == source["source_text"] and block.get("box") == source["box"]:
            eligible.add(block["block_id"])
    bindings = [(DEMO_WORK, str(original), item["block_id"], item["character_id"], 1)
                for item in seeds["bindings"]
                if item["character_id"] is not None and item["block_id"] in eligible]
    if not bindings:
        return
    with GlossaryMemory(database, DEMO_WORK, "en") as memory:
        with memory.connection:
            memory.connection.execute("BEGIN IMMEDIATE")
            memory.connection.executemany(
                "INSERT OR IGNORE INTO characters(work,character_id,name,voice) VALUES(?,?,?,'{}')",
                [(DEMO_WORK, character["id"], character["name"]) for character in seeds["characters"]])
            memory.connection.executemany(
                "INSERT OR IGNORE INTO character_bindings(work,page_key,block_id,character_id,confirmed) "
                "VALUES(?,?,?,?,?)", bindings)
    provenance = destination / "character-seeds.json"
    if not provenance.exists():
        with (bundle / provenance.name).open("rb") as source, provenance.open("xb") as output:
            shutil.copyfileobj(source, output)


def ensure_demo(directory: str | Path | None = None) -> Path:
    """复制演示页到独立会话；保留已有修改，无需模型或网络。"""
    destination = (Path(directory) if directory is not None else ROOT / "data/presentation-demo").expanduser().resolve()
    return _ensure_page(destination, BUNDLE, destination / "memory.sqlite3")


def ensure_demo_pages(directory: str | Path | None = None) -> list[Path]:
    first = ensure_demo(directory)
    destination = first.parent
    database = destination / "memory.sqlite3"
    return [first, *(_ensure_page(destination / name, BUNDLE / name, database)
                     for name in ("page-005", "page-006"))]


def _ensure_page(destination: Path, bundle: Path, database: Path) -> Path:
    result = destination / "daily-life.json"
    if result.exists():
        record = json.loads(result.read_text(encoding="utf-8"))
        if isinstance(record, dict):
            _seed_characters(destination, record, bundle=bundle, database=database)
        return result
    record = json.loads((bundle / "daily-life.json").read_text(encoding="utf-8"))
    assets = {
        "original.jpg": record["input_sha256"],
        "daily-life.png": record["output_sha256"],
        "daily-life.clean.png": record["clean_image"]["sha256"],
    }
    for name, expected in assets.items():
        if _sha256(bundle / name) != expected:
            raise ValueError(f"Bundled demonstration asset failed its checksum: {name}")
        existing = destination / name
        if existing.exists() and _sha256(existing) != expected:
            raise FileExistsError(f"Preserving an existing demonstration asset: {existing}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in (*assets, "LICENSE.txt", "NOTICE.txt", "character-seeds.json"):
        if name in {"LICENSE.txt", "NOTICE.txt"} and not (bundle / name).is_file():
            continue  # Continuation pages inherit the bundle's top-level notice.
        target = destination / name
        if not target.exists():
            # Exclusive creation prevents overwriting a concurrent edit or import.
            with (bundle / name).open("rb") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    record["input_path"] = str(destination / "original.jpg")
    record["output_path"] = str(destination / "daily-life.png")
    record["config"]["memory_db"] = str(database)
    with result.open("x", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    _seed_characters(destination, record, bundle=bundle, database=database)
    return result
