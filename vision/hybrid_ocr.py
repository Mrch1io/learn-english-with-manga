"""本地 Manga-OCR 识别后，按需使用 PaddleOCR-VL。

流程参考 hgmzhn/manga-translator-ui 的 use_hybrid_ocr。
似然分数仅用于分流，未经校准，不代表正确率。
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path

from core.runtime import Cancelled, ModelPool, require


def fallback_reason(text, confidence, threshold):
    if not text.strip():
        return "empty"
    if confidence is None or not math.isfinite(confidence):
        return "missing_confidence"
    return "low_confidence" if confidence < threshold else ""


def manga_result(ocr, crop):
    """保留原生预处理与解码，收集与 beam 对齐的 token 似然。"""
    torch = require("torch")
    post_process = require("manga_ocr.ocr").post_process
    with torch.inference_mode():
        pixels = ocr._preprocess(crop.convert("L").convert("RGB"))
        output = ocr.model.generate(pixels[None].to(ocr.model.device), max_length=300,
                                    return_dict_in_generate=True, output_scores=True)
        scores = ocr.model.compute_transition_scores(output.sequences, output.scores,
            beam_indices=getattr(output, "beam_indices", None), normalize_logits=True)[0]
        ids, logs = output.sequences[0].cpu().tolist(), scores.cpu().tolist()
    eos = ocr.model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    special = set(ocr.tokenizer.all_special_ids)
    values = []
    for token, value in zip(ids[1:], logs):
        if token in eos:
            break
        if token not in special:
            values.append(value)
    score = None
    truncated = len(ids) >= 300 and not any(t in eos for t in ids[1:])
    if not truncated and values and all(math.isfinite(v) and v <= 0 for v in values):
        score = math.exp(math.fsum(values) / len(values))
    raw = ocr.tokenizer.decode(ids, skip_special_tokens=True)
    return {"engine": "manga-ocr", "raw_text": raw, "text": post_process(raw).strip(),
            "confidence": score, "confidence_kind": "geometric_mean_token_probability",
            "token_count": len(values), "truncated": truncated}


def acquire_paddle(pool, config):
    path = Path(config.paddle_ocr_model_path).expanduser().resolve()
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Local PaddleOCR-VL-1.6 weights missing: {path}")
    # resident 策略下也只让一个 OCR 模型占用 GPU。
    name = "paddle_ocr:" + str(path)
    pool.offload(except_name=name)

    def load():
        torch, transformers = require("torch"), require("transformers")
        processor = transformers.PaddleOCRVLProcessor.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False)
        model, info = transformers.PaddleOCRVLForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("PaddleOCR-VL weights did not load exactly")
        return processor, model.eval()

    return pool.acquire(name, load, lambda value, device: value[1].to(device))


def repeated_output(text):
    """仅检查重复输出，不用参考文本选择结果。"""
    text = "".join(text.split())
    if len(text) < 24:
        return False
    counts = Counter(text)
    return bool(re.search(r"(.)\1{19,}|(.{2,8})\2{7,}", text) or
                (len(text) >= 30 and sum(n for c, n in counts.items() if not c.isalnum())
                 / len(text) >= .65 and max(counts.values()) / len(text) >= .45))


def paddle_result(engine, crop, *, retries=1, cancel=None):
    if type(retries) is not int or retries not in (0, 1):
        raise ValueError("PaddleOCR retries must be 0 or 1")
    processor, model = engine
    torch = require("torch")
    attempts = []
    for attempt in range(retries + 1):
        if cancel:
            cancel()
        prompt = "OCR: Extract all Japanese text."
        generation = {"max_new_tokens": 128, "do_sample": False, "num_beams": 1}
        if attempt:
            prompt += "\nReturn only the visible OCR text once. Do not repeat characters or punctuation."
            generation.update(max_new_tokens=64, repetition_penalty=1.15, no_repeat_ngram_size=4)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": crop.convert("RGB")}, {"type": "text", "text": prompt}]}]
        with torch.inference_mode():
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, **generation)
            ids = output[0, inputs["input_ids"].shape[1]:].cpu().tolist()
        raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
        eos = model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        truncated = len(ids) >= generation["max_new_tokens"] and not any(t in eos for t in ids)
        repeated = repeated_output(raw)
        attempts.append({"raw_text": raw, "truncated": truncated, "repeated": repeated})
        if cancel:
            cancel()
        if not repeated:
            break
    error = ("PaddleOCR returned repeated text" if repeated else
             "PaddleOCR output reached token limit" if truncated else
             "PaddleOCR returned no text" if not raw.strip() else "")
    text = require("manga_ocr.ocr").post_process(raw).strip() if not error else ""
    return {"engine": "paddleocr-vl-1.6", "raw_text": raw, "text": text,
            "confidence": None, "error": error, "attempts": attempts}


def recognize_crops(crops, *, pool, config, cancel=None):
    """先整批主识别，再整批备用识别，避免逐裁片搬运模型。

    保留两份结果；备用失败时保留主结果，生成最多重试一次。
    """
    from .detection import _acquire_ocr

    config.validate()
    if not crops:
        return []
    if cancel:
        cancel()
    pool.offload(except_name="ocr")
    primary = _acquire_ocr(pool, config)
    rows, pending = [], []
    for crop in crops:
        if cancel:
            cancel()
        started = time.perf_counter()
        try:
            result = manga_result(primary, crop)
        except Cancelled:
            raise
        except Exception as exc:
            if ModelPool.is_oom(exc):
                raise
            result = {"engine": "manga-ocr", "text": "", "confidence": None,
                      "error": f"{type(exc).__name__}: {exc}"}
        result["seconds"] = time.perf_counter() - started
        reason = fallback_reason(result["text"], result["confidence"], config.ocr_prob_threshold)
        row = {"primary": result, "secondary": None, "reason": reason,
               "text": result["text"], "selected_engine": "manga-ocr"}
        rows.append(row)
        if reason:
            pending.append((crop, row))
    if cancel:
        cancel()
    if pending:
        try:
            secondary = acquire_paddle(pool, config)
        except Cancelled:
            raise
        except Exception as exc:
            if ModelPool.is_oom(exc):
                ModelPool.release_exception_frames(exc)
                pool.offload()
            for _, row in pending:
                row["secondary"] = {"text": "", "confidence": None, "error": f"{type(exc).__name__}: {exc}"}
            return rows
        for index, (crop, row) in enumerate(pending):
            if cancel:
                cancel()
            started = time.perf_counter()
            try:
                result = paddle_result(secondary, crop, retries=config.retries, cancel=cancel)
            except Cancelled:
                raise
            except Exception as exc:
                if ModelPool.is_oom(exc):
                    message = f"{type(exc).__name__}: {exc}"
                    ModelPool.release_exception_frames(exc)
                    pool.offload()
                    for _, unprocessed in pending[index:]:
                        unprocessed["secondary"] = {"text": "", "confidence": None, "error": message}
                    return rows
                result = {"text": "", "confidence": None, "error": f"{type(exc).__name__}: {exc}"}
            result["seconds"] = time.perf_counter() - started
            row["secondary"] = result
            if result["text"].strip() and not result.get("error"):
                row.update(text=result["text"], selected_engine="paddleocr-vl-1.6")
    return rows


def run_hybrid_ocr(image, blocks, *, pool, config, cancel=None, crop_padding=4, observe_crop=None):
    from .detection import ocr_crops

    crops, groups = [], []
    for block in blocks:
        if cancel:
            cancel()
        # 保留人工校正和已采纳的 OCR 候选。
        if block.metadata.get("last_edit"):
            continue
        try:
            parts = list(ocr_crops(image, block, crop_padding))
            for index, crop in enumerate(parts):
                if observe_crop:
                    observe_crop(block.block_id, index, crop)
            groups.append((block, len(crops), len(parts)))
            crops.extend(parts)
        except Cancelled:
            raise
        except Exception as exc:
            block.status, block.error = "failed", f"OCR: {type(exc).__name__}: {exc}"
    rows = recognize_crops(crops, pool=pool, config=config, cancel=cancel)
    for block, offset, count in groups:
        records = rows[offset:offset + count]
        block.metadata["ocr_hybrid"] = {"threshold": config.ocr_prob_threshold, "segments": records}
        if any(row["reason"] for row in records):
            adopted = sum(row["selected_engine"] == "paddleocr-vl-1.6" for row in records)
            retained = sum(bool(row["reason"]) and row["selected_engine"] == "manga-ocr" for row in records)
            block.metadata["ocr_review"] = (f"混合 OCR：采用备用 {adopted} 段；备用未成功 {retained} 段（保留主结果）。"
                                             "请核对原图；两次识别内容均已记录。")
        if all(row["text"].strip() for row in records):
            block.source_text = "".join(row["text"] for row in records)
            block.status, block.error = "pending", ""
        else:
            block.status = "failed" if count > 1 else "empty"
            block.error = "OCR returned incomplete text; original text/pixels retained."
    return blocks
