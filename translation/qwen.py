from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from pathlib import Path

from .shared import ContextTooLong

@contextmanager
def _windows_streaming_weights():
    """Windows 下临时用 pread 读权重，避免 mmap 为整个分片预占提交内存。

    模型由单工作线程串行加载；替换仅覆盖本次加载，退出时恢复原 loader。
    """
    if os.name != "nt":
        yield
        return
    import transformers.modeling_utils as modeling
    original = modeling.safe_open
    def streaming_open(*args, **kwargs):
        if kwargs.get("backend", "mmap") == "mmap":
            kwargs["backend"] = "pread"
        return original(*args, **kwargs)
    modeling.safe_open = streaming_open
    try:
        yield
    finally:
        modeling.safe_open = original


def load_generator(pool, config, *, streaming_weights):
    if config.translation_backend != "qwen":
        raise ValueError("The Qwen/LoRA loader requires translation_backend='qwen'; VNTL uses CUDA GGUF.")
    import torch

    def loader():
        from transformers import AutoModelForCausalLM, AutoTokenizer

        gpu = str(pool.device).startswith("cuda")
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if gpu else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=config.local_files_only)
        # 权重直接载入 GPU，避免 CPU 副本额外占用 Windows 提交内存。
        with streaming_weights():
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name, dtype=dtype, device_map={"": pool.device},
                low_cpu_mem_usage=True, local_files_only=config.local_files_only
            )
        if config.adapter_path:
            from peft import PeftModel

            adapter = Path(config.adapter_path)
            if not adapter.is_dir():
                raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter}")
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False,
                                              local_files_only=config.local_files_only)
        model.eval()
        return tokenizer, model

    def mover(loaded, device):
        loaded[1].to(device)
        return loaded

    tokenizer, model = pool.acquire(f"translation:{config.model_name}:{config.adapter_path}", loader, mover)
    limits = [8192, getattr(model.config, "max_position_embeddings", 8192),
              getattr(tokenizer, "model_max_length", 8192)]
    context_limit = min(int(value) for value in limits if isinstance(value, (int, float)) and value > 0)

    def generate(messages, max_new_tokens):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_length = inputs["input_ids"].shape[-1]
        if input_length + max_new_tokens > context_limit:
            raise ContextTooLong(
                f"Prompt ({input_length}) + output ({max_new_tokens}) exceeds token budget ({context_limit}); "
                "reduce context/group size or inspect unusually long OCR text"
            )
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=pad_id)
        generated = output[0, input_length:]
        eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", tokenizer.eos_token_id)
        eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        if len(generated) >= max_new_tokens and int(generated[-1]) not in eos_ids:
            raise ValueError("Generation reached max_new_tokens without EOS; increase the output token limit")
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def extract_terms(messages, max_new_tokens):
        # LoRA 只训练过纯文本翻译；提取 JSON 时暂时停用，复用基座权重。
        adapter_context = model.disable_adapter() if config.adapter_path else nullcontext()
        with adapter_context:
            return generate(messages, max_new_tokens)

    generate.extract_terms = extract_terms
    return generate

