"""Magi 角色建议以人工绑定为准；距离阈值未经校准，不代表概率。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from core.runtime import ROOT, Cancelled, Config, ModelPool
from core.memory import GlossaryMemory
from pipeline.batch import file_hash, natural_key

DISTANCE_LIMIT = 0.23
DISTANCE_MARGIN = 0.05


def load_magi(pool, model_path):
    """按 Transformers 5 的 conversion_mapping.py 映射并完整校验本地权重。

    仅跳过停用的 OCR 分支；CUDA 搬运交给 ModelPool。
    """
    import re
    path = Path(model_path).expanduser().resolve()
    checkpoint = path / "pytorch_model.bin"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Local Magi checkpoint missing: {checkpoint}")
    def configured_model(path):
        from transformers import AutoConfig, ConditionalDetrConfig, ViTMAEConfig
        path = Path(path).resolve()
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        config.disable_ocr = True
        config.detection_model_config = ConditionalDetrConfig.from_dict(config.detection_model_config.to_dict())
        config.detection_model_config.backbone_config.use_pretrained_backbone = False
        config.crop_embedding_model_config = ViTMAEConfig.from_dict(config.crop_embedding_model_config.to_dict())
        return config

    def current_key(key):
        if key.startswith("crop_embedding_model."):
            for old, new in [(".encoder.layer.", ".layers."),
                             (".attention.attention.query.", ".attention.q_proj."),
                             (".attention.attention.key.", ".attention.k_proj."),
                             (".attention.attention.value.", ".attention.v_proj."),
                             (".attention.output.dense.", ".attention.o_proj."),
                             (".intermediate.dense.", ".mlp.fc1."), (".output.dense.", ".mlp.fc2.")]:
                key = key.replace(old, new)
        elif key.startswith("detection_transformer."):
            key = key.replace(".backbone.conv_encoder.", ".backbone.")
            key = key.replace(".self_attn.out_proj.", ".self_attn.o_proj.").replace(".encoder_attn.out_proj.", ".encoder_attn.o_proj.")
            key = re.sub(r"(\.layers\.\d+)\.(fc[12])\.", r"\1.mlp.\2.", key)
            for old, new in [("qcontent", "q_content"), ("qpos", "q_pos"), ("kcontent", "k_content"),
                             ("kpos", "k_pos"), ("v", "v"), ("qpos_sine", "q_pos_sine")]:
                key = key.replace(f".sa_{old}_proj.", f".self_attn.{new}_proj.")
                key = key.replace(f".ca_{old}_proj.", f".encoder_attn.{new}_proj.")
        return key

    def load_strict(path):
        import torch
        from transformers import AutoModel
        path = Path(path).resolve()
        config = configured_model(path)
        started = time.perf_counter()
        model = AutoModel.from_config(config, trust_remote_code=True)
        checkpoint = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
        expected = model.state_dict()
        state, renamed, ignored = {}, {}, []
        for original, tensor in checkpoint.items():
            if original.startswith("ocr_model."):
                ignored.append(original)
                continue
            key = original if original in expected else current_key(original)
            if key in state:
                raise RuntimeError(f"Magi weight mapping collision: {original} -> {key}")
            state[key] = tensor
            if key != original:
                renamed[original] = key
        missing, unexpected = sorted(set(expected)-set(state)), sorted(set(state)-set(expected))
        mismatched = [(key, list(state[key].shape), list(expected[key].shape)) for key in set(state)&set(expected)
                      if state[key].shape != expected[key].shape]
        if missing or unexpected or mismatched:
            raise RuntimeError("Magi active weights did not load completely: " + json.dumps(
                {"missing": missing, "unexpected": unexpected, "mismatched": mismatched}))
        model.load_state_dict(state, strict=True)
        # 键名和形状匹配后，仍逐张量核对权重值。
        unequal = [key for key, value in model.state_dict().items() if not torch.equal(value, state[key])]
        if unequal:
            raise RuntimeError("Loaded Magi tensor differs from checkpoint: " + ", ".join(unequal))
        model.eval()
        metadata = {"validated_revision": "fbc890fec52977142e8ee00bfe26e9458b65517c", "checkpoint_bytes": (path / "pytorch_model.bin").stat().st_size,
                    "load_seconds": time.perf_counter()-started, "parameters": sum(p.numel() for p in model.parameters()),
                    "active_state_keys": len(model.state_dict()), "ignored_ocr_keys": len(ignored),
                    "missing_keys": missing, "mismatched_keys": mismatched, "unexpected_keys": unexpected,
                    "all_loaded_tensors_equal_checkpoint": True, "renamed_keys": len(renamed),
                    "conversion": "Explicit scoped renames from Transformers official ViTMAE/ConditionalDetr conversion_mapping.py; strict=True",
                    "dtype": str(next(model.parameters()).dtype), "torch": torch.__version__}
        model._manga_loading_info = metadata
        return model, metadata

    stat = checkpoint.stat()
    key = f"magi:{path}:{stat.st_size}:{stat.st_mtime_ns}"
    return pool.acquire(key, lambda: load_strict(path)[0], lambda model, device: model.to(device).eval())


def _predict(model, image):
    import numpy as np
    import torch

    array = np.array(image, copy=True)
    with torch.inference_mode():
        result = model.predict_detections_and_associations([array])[0]
        # DETR 框可能略微越界；裁图和保存的框必须使用同一有效区域。
        result["characters"] = [_clip_character_box(box, image.size) for box in result["characters"]]
        embeddings = model.predict_crop_embeddings([array], [result["characters"]], batch_size=16)[0]
        return result, embeddings.detach().cpu().tolist()


def _box(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)
            or value[2] <= value[0] or value[3] <= value[1]):
        raise ValueError("Invalid Magi or bubble box")
    return list(value)


def _clip_character_box(value, size):
    box = _box(value)
    width, height = size
    return _box([min(width, max(0, box[0])), min(height, max(0, box[1])),
                 min(width, max(0, box[2])), min(height, max(0, box[3]))])


def _unit(vector):
    if (not isinstance(vector, (list, tuple)) or len(vector) != 768
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector)):
        raise ValueError("Magi requires finite 768-dimensional character embeddings")
    norm = math.sqrt(math.fsum(x*x for x in vector))
    return [x/norm for x in vector] if norm > 1e-12 else None


def _matched_texts(blocks, texts):
    matches = [[] for _ in blocks]
    ambiguous = set()
    for text_index, text in enumerate(texts):
        cx, cy = (text[0]+text[2])/2, (text[1]+text[3])/2
        text_area = (text[2]-text[0])*(text[3]-text[1])
        owners = []
        for block_index, block in enumerate(blocks):
            box = block["box"]
            if not (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]):
                continue
            intersection = max(0, min(box[2], text[2])-max(box[0], text[0])) * max(0, min(box[3], text[3])-max(box[1], text[1]))
            if intersection/min(text_area, (box[2]-box[0])*(box[3]-box[1])) >= .65:
                owners.append(block_index)
        for owner in owners:
            matches[owner].append(text_index)
        if len(owners) > 1:
            ambiguous.update(owners)
    return matches, ambiguous


def _nearest(vectors, samples):
    usable = [vector for vector in vectors if vector is not None]
    if not usable:
        return None, None, None
    centroid = _unit([math.fsum(values)/len(usable) for values in zip(*usable)])
    if centroid is None:
        return None, None, None
    by_character = {}
    for character_id, reference in samples:
        if reference is None:
            continue
        distance = max(0.0, min(2.0, 1-math.fsum(a*b for a, b in zip(centroid, reference))))
        by_character[character_id] = min(distance, by_character.get(character_id, 2.0))
    ranked = sorted((distance, character_id) for character_id, distance in by_character.items())
    if not ranked:
        return None, None, None
    distance, character_id = ranked[0]
    margin = ranked[1][0]-distance if len(ranked) > 1 else None
    accepted = distance <= DISTANCE_LIMIT and (margin is None or margin >= DISTANCE_MARGIN)
    return character_id if accepted else None, distance, margin


def _validate_bubbles(blocks: list[dict[str, Any]]) -> None:
    ids = [block.get("block_id") for block in blocks]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Page record has missing or duplicate bubble IDs")
    for block in blocks:
        _box(block["box"])
        if not isinstance(block.get("metadata", {}), dict):
            raise ValueError("Invalid bubble metadata")


def _associations(pairs: list, text_count: int, character_count: int) -> dict[int, set[int]]:
    associations = {}
    for pair in pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(isinstance(index, bool) or not isinstance(index, int) for index in pair)
                or not 0 <= pair[0] < text_count or not 0 <= pair[1] < character_count):
            raise ValueError("Invalid Magi text/character association")
        associations.setdefault(pair[0], set()).add(pair[1])
    return associations


def _map_speakers(matches: list[list[int]], ambiguous: set[int],
                  associations: dict[int, set[int]], labels: list[str | int],
                  characters: list) -> list[tuple[str | int | None, int | None, str]]:
    mapped = []
    for index, indices in enumerate(matches):
        assigned = [associations.get(text_index, set()) for text_index in indices]
        if index in ambiguous:
            mapped.append((None, None, "ambiguous_text_overlap"))
        elif not indices:
            mapped.append((None, None, "no_matching_text"))
        elif any(len(owners) != 1 for owners in assigned):
            mapped.append((None, None, "missing_or_ambiguous_association"))
        else:
            character_indices = {next(iter(owners)) for owners in assigned}
            clusters = {labels[character_index] for character_index in character_indices}
            representative = min(character_indices, key=lambda i: tuple(characters[i]))
            mapped.append((next(iter(clusters)), representative, "associated") if len(clusters) == 1
                          else (None, None, "conflicting_text_speakers"))
    return mapped


def _manual_clusters(blocks: list[dict[str, Any]], mapped: list[tuple[str | int | None, int | None, str]],
                     bindings: dict[str, dict[str, Any]]) -> dict[str | int, set[str]]:
    manual_clusters = {}
    for block, (cluster, _, _) in zip(blocks, mapped):
        binding = bindings.get(block["block_id"], {})
        if cluster is not None and binding.get("confirmed"):
            manual_clusters.setdefault(cluster, set()).add(binding["character_id"])
    return manual_clusters


def _character_vectors(labels: list[str | int], characters: list,
                       embeddings: list[list[float]]) -> list[list[float] | None]:
    if (not isinstance(labels, list) or len(labels) != len(characters) or len(embeddings) != len(characters)
            or any(isinstance(label, bool) or not isinstance(label, (str, int)) for label in labels)):
        raise ValueError("Magi character clusters/embeddings do not match detections")
    return [_unit(vector) for vector in embeddings]


def suggest_characters(record: dict[str, Any], result: dict[str, Any],
                       embeddings: list[list[float]], memory: GlossaryMemory) -> tuple[dict, list, list, list]:
    """返回新记录和待写入数据，保留原记录不变。"""
    blocks = record["blocks"]
    _validate_bubbles(blocks)
    texts = [_box(value) for value in result["texts"]]
    characters = [_box(value) for value in result["characters"]]
    labels = result["character_cluster_labels"]
    vectors = _character_vectors(labels, characters, embeddings)
    associations = _associations(result["text_character_associations"], len(texts), len(characters))
    matches, ambiguous = _matched_texts(blocks, texts)
    mapped = _map_speakers(matches, ambiguous, associations, labels, characters)
    bindings = memory.list_character_bindings(record["input_path"])
    manual_clusters = _manual_clusters(blocks, mapped, bindings)
    # 样本每页归一化一次，供所有角色簇复用。
    samples = [(sample["character_id"], _unit(sample["embedding"])) for sample in memory.list_character_samples()
               if sample["character_id"] not in {"unknown", "narrator"}]
    decisions, new_characters = {}, {}
    for cluster in dict.fromkeys(cluster for cluster, _, _ in mapped if cluster is not None):
        manual = manual_clusters.get(cluster, set())
        if manual:
            decisions[cluster] = {"character_id": next(iter(manual)) if len(manual) == 1 else "unknown",
                                  "method": "same_page_manual_cluster" if len(manual) == 1 else "conflicting_manual_cluster"}
            continue
        indices = [index for index, label in enumerate(labels) if label == cluster]
        character_id, distance, margin = _nearest([vectors[index] for index in indices], samples)
        decision = {"distance": distance, "margin": margin,
                    "distance_limit": DISTANCE_LIMIT, "margin_minimum": DISTANCE_MARGIN}
        if character_id:
            decision.update(character_id=character_id, method="sample_cosine")
        elif distance is not None and distance <= DISTANCE_LIMIT:
            decision.update(character_id="unknown", method="ambiguous_sample_match")
        else:
            representative = min((characters[index] for index in indices), key=tuple)
            identity = [memory.work, record["input_sha256"], [round(value, 3) for value in representative]]
            suffix = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:20]
            character_id = "magi-" + suffix
            new_characters[character_id] = {"id": character_id, "name": "候选角色 " + suffix[-6:], "voice": {}}
            decision.update(character_id=character_id, method="new_unconfirmed_cluster")
        decisions[cluster] = decision
    updated = deepcopy(record)
    # 所有文本区都计入覆盖率；漏检仅供复核，不改变气泡或像素。
    covered_texts = {index for indices in matches for index in indices}
    updated["magi_coverage"] = {
        "detected_text_count": len(texts),
        "covered_text_count": len(covered_texts),
        "covered_bubble_count": sum(bool(indices) for indices in matches),
        "unmatched_texts": [{"text_index": index, "box": box}
                            for index, box in enumerate(texts) if index not in covered_texts],
    }
    pending_bindings, pending_samples = [], []
    for block, (cluster, char_index, reason), indices in zip(updated["blocks"], mapped, matches):
        binding = bindings.get(block["block_id"], {})
        info = {"character_id": "unknown", "confirmed": False, "method": reason, "text_indices": indices}
        if cluster is not None:
            info.update(decisions[cluster])
            info.update(character_box=characters[char_index], embedding=list(embeddings[char_index]), cluster=cluster)
        if binding.get("confirmed"):
            info.update(character_id=binding["character_id"], confirmed=True, method="manual")
            if char_index is not None and binding["character_id"] not in {"unknown", "narrator"}:
                pending_samples.append({"character_id": binding["character_id"], "box": characters[char_index],
                                        "embedding": list(embeddings[char_index])})
        else:
            pending_bindings.append((block["block_id"], info["character_id"]))
        block.setdefault("metadata", {})["magi"] = info
    return updated, list(new_characters.values()), pending_bindings, pending_samples


def _check(cancel):
    if cancel:
        cancel()


def _write_guarded(path, data, guard):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        guard()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_analysis(path, original_bytes, record, result, embeddings, memory, cancel):
    expected = hashlib.sha256(original_bytes).hexdigest()
    def guard():
        _check(cancel)
        if file_hash(path) != expected:
            raise ValueError("Page JSON changed during Magi analysis; reopen it before retrying")
        if file_hash(record["input_path"]) != record["input_sha256"]:
            raise ValueError("Original image changed during Magi analysis; existing coordinates are unsafe")

    # 推理后再加写锁并重读绑定，保留推理期间的人工修改。
    with memory.connection:
        memory.connection.execute("BEGIN IMMEDIATE")
        updated, characters, bindings, samples = suggest_characters(record, result, embeddings, memory)
        guard()
        for character in characters:
            memory.connection.execute("INSERT OR IGNORE INTO characters(work,character_id,name,voice) VALUES(?,?,?,'{}')",
                                      (memory.work, character["id"], character["name"]))
        for block_id, character_id in bindings:
            memory.connection.execute(
                "INSERT INTO character_bindings(work,page_key,block_id,character_id,confirmed) VALUES(?,?,?,?,0) "
                "ON CONFLICT(work,page_key,block_id) DO UPDATE SET character_id=excluded.character_id,confirmed=0 "
                "WHERE character_bindings.confirmed=0", (memory.work, record["input_path"], block_id, character_id))
        for sample in samples:
            memory.connection.execute(
                "INSERT INTO character_samples(work,character_id,page_key,box,embedding) VALUES(?,?,?,?,?) "
                "ON CONFLICT(work,character_id,page_key,box) DO UPDATE SET embedding=excluded.embedding",
                (memory.work, sample["character_id"], record["input_path"],
                 json.dumps([float(x) for x in sample["box"]], separators=(",", ":")),
                 json.dumps([float(x) for x in sample["embedding"]], separators=(",", ":"))))
        data = json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        with tempfile.NamedTemporaryFile(prefix=path.name+".", suffix=".rollback", dir=path.parent, delete=False) as backup:
            backup.write(original_bytes)
            backup.flush()
            os.fsync(backup.fileno())
        rollback = Path(backup.name)
        try:
            _write_guarded(path, data, guard)
        except BaseException:
            rollback.unlink()
            raise
        try:
            memory.connection.commit()
        except BaseException as error:
            try:
                if path.read_bytes() != data:
                    raise ValueError("Page JSON changed after save")
                os.replace(rollback, path)
            except (OSError, ValueError) as restore_error:
                raise RuntimeError(f"Database commit failed ({error}); page recovery failed ({restore_error}); "
                                   f"original retained: {rollback}") from error
            raise
        else:
            rollback.unlink()
    return updated


def _page_paths(values):
    if isinstance(values, (str, Path)):
        values = [values]
    candidates, explicit = set(), set()
    for value in values:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            candidates.update(path.rglob("*.json"))
        else:
            path = path if path.suffix.lower() == ".json" else path.with_suffix(".json")
            candidates.add(path)
            explicit.add(path)
    pages = []
    for path in candidates:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and isinstance(record.get("blocks"), list) and record.get("input_path"):
                pages.append((path, record["input_path"]))
            elif path in explicit:
                pages.append((path, str(path)))
        except (OSError, ValueError):
            pages.append((path, str(path)))
    return [path for path, _ in sorted(pages, key=lambda pair: (natural_key(pair[1]), str(pair[0])))]


def analyze_pages(page_paths, *, model_path=ROOT/"runtime"/"magiv2", device="cuda:0", progress=None, cancel=None):
    """分析已有页面记录，不重跑 OCR 和翻译。"""
    from PIL import Image, ImageOps

    paths = _page_paths(page_paths)
    if not paths:
        raise ValueError("No page records found. Translate or generate point-reading texts first.")
    report = {"pages": [], "completed": 0, "failed": 0, "cancelled": False}
    pool, model = ModelPool(device, "staged"), None
    started = time.perf_counter()
    try:
        for index, path in enumerate(paths, 1):
            try:
                _check(cancel)
                if progress:
                    progress({"type": "stage", "current": index, "total": len(paths), "path": str(path),
                              "message": "Magi 正在分析人物与对白关联"})
                original_bytes = path.read_bytes()
                record = json.loads(original_bytes)
                if not isinstance(record, dict) or not isinstance(record.get("blocks"), list) or not record.get("input_path"):
                    raise ValueError("Select an existing single-page JSON or its PNG")
                if not record.get("input_sha256") or file_hash(record["input_path"]) != record["input_sha256"]:
                    raise ValueError("Original image checksum differs from the page record; reprocess it first")
                with Image.open(record["input_path"]) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                if record.get("image_size") and list(image.size) != record["image_size"]:
                    raise ValueError("Original image dimensions differ from the page record")
                if model is None:
                    model = load_magi(pool, Path(model_path))
                _check(cancel)
                result, embeddings = _predict(model, image)
                _check(cancel)
                with GlossaryMemory(record.get("config", {}).get("memory_db", Config().memory_db),
                                    record.get("work", "default"), record.get("target_language", "en")) as memory:
                    updated = _save_analysis(path, original_bytes, record, result, embeddings, memory, cancel)
                unknown = sum(block["metadata"]["magi"]["character_id"] == "unknown" for block in updated["blocks"])
                coverage = updated["magi_coverage"]
                uncovered = len(coverage["unmatched_texts"])
                message = ("Magi 角色建议已保存；文本区域覆盖 "
                           f"{coverage['covered_text_count']}/{coverage['detected_text_count']}")
                if uncovered:
                    message += f"；{uncovered} 个可能漏检待复核"
                entry = {"path": str(path), "input_path": record["input_path"], "status": "done",
                         "assigned": len(updated["blocks"])-unknown, "unknown": unknown,
                         "magi_coverage": coverage, "message": message}
                report["completed"] += 1
            except Cancelled:
                report["cancelled"] = True
                break
            except Exception as exc:
                if ModelPool.is_oom(exc):
                    ModelPool.release_exception_frames(exc)
                entry = {"path": str(path), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                report["failed"] += 1
            report["pages"].append(entry)
            if progress:
                progress({"type": "page_done" if entry["status"] == "done" else "error", "current": index,
                          "total": len(paths), "path": str(path), "message": entry.get("error", entry.get("message", ""))})
    finally:
        model = None
        pool.close()
    report["total_seconds"] = time.perf_counter()-started
    return report
