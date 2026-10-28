from __future__ import annotations

import gc
import importlib
import os
import time
import traceback
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_IMAGE = str(ROOT / "translated_output")
LOCAL_MODEL_PATH = str(ROOT / "best.pt")
DEVICE = "cuda:0"
DETECTION_CONF = 0.50

# llama.cpp 常驻时，为图像阶段预留显存，包括权重、激活、CUDA 工作区和 WDDM 余量。
# 这是保守预算，不是实测峰值；只按权重预留可能用到 Windows 共享显存。
VNTL_STAGE_RESERVE_BYTES = {name: mib * 1024**2 for name, mib in
                          (("detector", 768), ("ocr", 1536), ("paddle_ocr", 3072), ("lama", 2048), ("magi", 2048))}


@dataclass
class TextBlock:
    box: tuple[int, int, int, int]
    source_text: str = ""
    translated_text: str = ""
    block_id: str = ""
    status: str = "pending"
    error: str = ""
    safe_box: tuple[int, int, int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    reading_order: int = 0


@dataclass
class Config:
    work: str = "default"
    target_language: str = "en"
    device: str = DEVICE
    memory_policy: str = "staged"
    memory_db: str = str(ROOT / "data" / "memory.sqlite3")
    model_path: str = LOCAL_MODEL_PATH
    translation_backend: str = "vntl"
    gguf_path: str = str(ROOT / "runtime" / "vntl" / "vntl-llama3-8b-v2-hf.Q4_K_M.gguf")
    llama_server_path: str = str(ROOT / "runtime" / "llama.cpp" / "b11706" / "llama-server.exe")
    llama_context_size: int = 2048
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    adapter_path: str = str(ROOT / "manga-lora-adapter")
    detection_conf: float = DETECTION_CONF
    detect_free_text: bool = True
    ocr_candidate_review: bool = False
    use_hybrid_ocr: bool = False
    ocr_prob_threshold: float = 0.1
    paddle_ocr_model_path: str = str(ROOT / "runtime" / "ocr-comparison" / "PaddleOCR-VL-1.6")
    magi_model_path: str = str(ROOT / "runtime" / "magiv2")
    reading_direction: str = "rtl"
    translation_mode: str = "auto"
    group_size: int = 4
    context_chars: int = 1200
    use_continuations: bool = True
    use_fragment_links: bool = True
    use_translation_history: bool = True
    use_speaker_context: bool = True
    max_new_tokens: int = 256
    retries: int = 1
    extract_terms: bool = True
    local_files_only: bool = False
    font_path: str | None = None
    min_font_size: int = 12
    max_font_size: int = 40
    skip_inpaint: bool = False
    debug_mask: bool = False
    export_images: bool = True

    def validate(self) -> None:
        if type(self.use_speaker_context) is not bool:
            raise ValueError("use_speaker_context must be a boolean")
        if type(self.use_fragment_links) is not bool:
            raise ValueError("use_fragment_links must be a boolean")
        if type(self.use_hybrid_ocr) is not bool:
            raise ValueError("use_hybrid_ocr must be a boolean")
        if (type(self.ocr_prob_threshold) not in (int, float)
                or not 0 <= self.ocr_prob_threshold <= 1):
            raise ValueError("ocr_prob_threshold must be a finite number in [0, 1]")
        if type(self.ocr_candidate_review) is not bool:
            raise ValueError("ocr_candidate_review must be a boolean")
        if self.device != "auto" and not (self.device == "cuda" or self.device.startswith("cuda:")):
            raise ValueError("This project is configured for GPU PyTorch; select a CUDA device.")
        if self.target_language != "en":
            raise ValueError("Version 1 enables Japanese → English only; other languages need model/font validation.")
        if not self.work.strip():
            raise ValueError("A work name is required to isolate terminology.")
        if self.memory_policy not in {"staged", "resident"}:
            raise ValueError("memory_policy must be staged or resident")
        if self.translation_backend not in {"vntl", "qwen"}:
            raise ValueError("translation_backend must be vntl or qwen")
        if not 512 <= self.llama_context_size <= 8192:
            raise ValueError("llama_context_size must be 512–8192 tokens")
        if self.translation_mode not in {"auto", "individual", "grouped"}:
            raise ValueError("translation_mode must be auto, individual or grouped")
        if self.reading_direction not in {"rtl", "ltr"}:
            raise ValueError("reading_direction must be rtl or ltr")
        if not 0 < self.detection_conf <= 1:
            raise ValueError("detection_conf must be in (0, 1]")
        if not 1 <= self.group_size <= 16 or not 0 <= self.retries <= 1:
            raise ValueError("group_size must be 1–16 and retries 0–1")
        if self.context_chars != 0 and self.context_chars < 128 or self.max_new_tokens < 32:
            raise ValueError("context_chars must be 0 (no context) or >=128 and max_new_tokens >=32")
        if not 8 <= self.min_font_size <= self.max_font_size <= 128:
            raise ValueError("font sizes must satisfy 8 <= min <= max <= 128")


class Cancelled(Exception):
    """在安全步骤之间取消任务，保留已完成的文件。"""


def require(pkg: str, install_name: str | None = None):
    try:
        return importlib.import_module(pkg)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency {install_name or pkg}. Install requirements.txt in the project environment."
        ) from exc


def available_commit_bytes():
    """Windows 还能分配的提交内存，包含分页文件的余量。"""
    if os.name != "nt":
        return None
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical",
                "total_pagefile", "available_pagefile", "total_virtual", "available_virtual", "extended")]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.available_pagefile)


class ModelPool:
    """在内存中缓存权重；分阶段模式只让一个池内模型占用 GPU。"""

    _stage_retry = ContextVar("stage_oom_retry", default=False)

    def __init__(self, device=DEVICE, policy="staged"):
        self.requested_device = device
        self._device = None
        self.policy = policy
        self.models: dict[str, tuple[Any, Callable]] = {}
        self.active: set[str] = set()
        self.load_seconds: dict[str, float] = {}
        self.transfer_seconds = 0.0
        self.loads: dict[str, int] = {}
        self.evictions: list[dict] = []
        # llama.cpp 是跨页复用的独立进程，不能用 .to(device) 搬移。
        self.vntl = None

    @property
    def device(self):
        if self._device is None:
            torch = require("torch")
            available = torch.cuda.is_available()
            requested = self.requested_device
            if requested == "auto":
                requested = "cuda:0"
            if not (requested == "cuda" or requested.startswith("cuda:")):
                raise RuntimeError("CPU inference is disabled; use the configured GPU Python.")
            if not available:
                raise RuntimeError("CUDA was requested but this PyTorch installation cannot use CUDA. Use launch.bat / the configured GPU Python.")
            self._device = requested
        return self._device

    def sync(self):
        if self._device and self._device.startswith("cuda"):
            require("torch").cuda.synchronize(self._device)

    def offload(self, except_name=None):
        for name in list(self.active):
            if name != except_name:
                value, mover = self.models[name]
                if name.startswith("translation:") and isinstance(value, tuple) and len(value) == 2:
                    weights = value[1]
                    size = sum(t.numel() * t.element_size() for t in weights.parameters())
                    size += sum(t.numel() * t.element_size() for t in weights.buffers())
                    del weights
                    available = available_commit_bytes()
                    margin = max(512 * 1024**2, size // 4)
                    if available is not None and available < size + margin:
                        # 提交内存不足时直接释放权重，下次重载，避免复制到内存时耗尽额度。
                        self.sync()
                        self.models.pop(name)
                        self.active.remove(name)
                        self.evictions.append({"model": name, "reason": "insufficient_windows_commit",
                                               "model_bytes": size, "available_commit_bytes": available,
                                               "safety_margin_bytes": margin})
                        del value, mover
                        gc.collect()
                        warnings.warn("Low Windows commit memory: translation weights released instead of copied to RAM; next page will reload them.",
                                      RuntimeWarning, stacklevel=2)
                        continue
                start = time.perf_counter()
                mover(value, "cpu")
                self.sync()
                self.transfer_seconds += time.perf_counter() - start
                self.active.remove(name)
        if self._device and self._device.startswith("cuda"):
            require("torch").cuda.empty_cache()

    def acquire(self, name, loader, mover):
        if self.policy == "staged":
            self.offload(except_name=name)
        self._reserve_image_stage(name)
        if name not in self.models:
            start = time.perf_counter()
            value = loader()
            self.models[name] = (value, mover)
            self.load_seconds[name] = self.load_seconds.get(name, 0) + time.perf_counter() - start
            self.loads[name] = self.loads.get(name, 0) + 1
        value, mover = self.models[name]
        if name not in self.active:
            start = time.perf_counter()
            try:
                mover(value, self.device)
            except Exception as exc:
                if not self.is_oom(exc):
                    raise
                try:
                    if self._stage_retry.get():
                        raise
                    self.release_exception_frames(exc)
                    self.policy = "staged"
                    mover(value, "cpu")
                    self.offload()
                    gc.collect()
                    require("torch").cuda.empty_cache()
                    mover(value, self.device)
                except Exception:
                    # 搬到 GPU 失败时，部分权重可能已占用显存。
                    try:
                        mover(value, "cpu")
                        require("torch").cuda.empty_cache()
                    except Exception:
                        pass
                    raise
            self.sync()
            self.transfer_seconds += time.perf_counter() - start
            self.active.add(name)
        return value

    def _reserve_image_stage(self, name):
        reserve = VNTL_STAGE_RESERVE_BYTES.get(name.partition(":")[0])
        if self.vntl is None or reserve is None:
            return
        cuda = require("torch").cuda
        cuda.empty_cache()
        free, total = cuda.mem_get_info(self.device)
        if free >= reserve:
            return
        server = self.vntl
        entry = {"model": "translation:vntl:" + str(server.info.get("model_file", "unknown")),
                 "reason": "insufficient_cuda_stage_headroom", "stage": name,
                 "free_before_bytes": int(free), "stage_reserve_bytes": reserve,
                 "total_cuda_bytes": int(total), "server_pid": server.info.get("pid")}
        started = time.perf_counter()
        server.close()  # 只关闭当前模型池启动的子进程。
        self.vntl = None
        entry["close_seconds"] = time.perf_counter()-started
        free_after, _ = cuda.mem_get_info(self.device)
        entry["free_after_bytes"] = int(free_after)
        self.evictions.append(entry)
        warnings.warn(f"CUDA memory before {name}: {free/1024**2:.0f} MiB free, "
                      f"{reserve/1024**2:.0f} MiB reserved for the stage; released this task's VNTL server. "
                      "Translation will reload it when needed.", RuntimeWarning, stacklevel=3)
        if free_after < reserve:
            raise RuntimeError(f"Insufficient CUDA memory for {name} after releasing this task's VNTL: "
                               f"{free_after/1024**2:.0f} MiB free; {reserve/1024**2:.0f} MiB required. "
                               "Free GPU memory used by other tasks before retrying; CPU inference is disabled.")

    @staticmethod
    @contextmanager
    def defer_oom_retry(enabled=True):
        token = ModelPool._stage_retry.set(enabled)
        try:
            yield
        finally:
            ModelPool._stage_retry.reset(token)

    @staticmethod
    def release_exception_frames(exc):
        # 异常栈还可能引用 GPU 输入和 KV 缓存，重试前先释放。
        traceback.clear_frames(exc.__traceback__)
        exc.__traceback__ = exc.__context__ = exc.__cause__ = None
        gc.collect()

    @staticmethod
    def is_oom(exc):
        return "out of memory" in str(exc).lower() and (
            "cuda" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"
        )

    def close(self):
        if self.vntl is not None:
            self.vntl.close()
            self.vntl = None
        # 直接释放显存中的权重，避免先搬到 CPU 再销毁带来的临时内存占用。
        self.models.clear()
        self.active.clear()
        gc.collect()
        if self._device and self._device.startswith("cuda"):
            require("torch").cuda.empty_cache()
