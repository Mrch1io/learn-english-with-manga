"""Reproducible real-model benchmark; never substitutes fake timings."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.runtime import Config, ModelPool, TextBlock
from core.memory import GlossaryMemory
from pipeline.batch import atomic_json, file_hash, run_batch


def compare_modes(blocks, config, pool, output_dir):
    from translation import translate, _load_generator
    comparisons = {}
    generator = _load_generator(pool, config)
    for mode in ("legacy_prompt_exact_terms", "individual", "grouped"):
        current = deepcopy(blocks)
        for block in current:
            block.translated_text, block.status, block.error = "", "pending", ""
            block.metadata = {}
        with GlossaryMemory(":memory:", "comparison", "en") as memory:
            memory.import_json(ROOT/"glossary.json")
            pool.sync()
            start = time.perf_counter()
            if mode == "legacy_prompt_exact_terms":
                # Reproduce the original plain prompt on the same OCR, with
                # exact glossary rules only. The removed BERT is NOT benchmarked.
                calls = 0
                for block in current:
                    if not block.source_text.strip():
                        block.status = "empty"
                        continue
                    rules = [f"Translate '{term['source']}' strictly as '{term['target']}'"
                             for term in memory.match(block.source_text)]
                    prompt = "You are an expert manga translator. Translate the following Japanese dialogue into natural, conversational English. Output ONLY the English translation, with no explanations, notes, or quotes."
                    if rules:
                        prompt += " GLOSSARY RULES: " + ", ".join(rules) + "."
                    calls += 1
                    try:
                        block.translated_text = generator([{"role":"system","content":prompt},
                            {"role":"user","content":block.source_text}],100)
                        block.status = "translated" if block.translated_text.strip() else "failed"
                    except Exception as exc:
                        block.status, block.error = "failed", str(exc)
                stats = {"generation_calls": calls, "note": "Original prompt, exact terms only; no original BERT retrieval"}
            else:
                stats = translate(current,pool=pool,memory=memory,
                    config=replace(config,translation_mode=mode,extract_terms=False),page_key="comparison",
                    generator=generator)
            pool.sync()
            comparisons[mode] = {"seconds": time.perf_counter()-start,"stats":stats,
                "blocks":[asdict(b) for b in current]}
    atomic_json(output_dir/"translation-modes.json",comparisons)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",nargs="?",default=str(ROOT/"testimg.png"))
    parser.add_argument("--output",default=str(ROOT/"output"/"benchmarks"))
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--compare-modes",action="store_true")
    parser.add_argument("--translator", choices=["vntl", "qwen"], default="vntl")
    parser.add_argument("--mode", choices=["auto", "individual", "grouped"], default="auto")
    parser.add_argument("--offline",action="store_true")
    args=parser.parse_args()
    if args.compare_modes and args.translator != "qwen":
        parser.error("--compare-modes compares Qwen prompts; use --translator qwen")
    if args.rounds < 2:
        parser.error("--rounds must be >=2 to measure a warm page")
    source=Path(args.input).resolve()
    if not source.is_file():
        parser.error("benchmark input must be one image")
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    # Unique session prevents an earlier measurement/database from contaminating results.
    session=root/time.strftime("%Y%m%d-%H%M%S")
    session.mkdir(exist_ok=False)
    inputs=session/"inputs"
    inputs.mkdir()
    for i in range(args.rounds):
        shutil.copy2(source,inputs/f"{i+1}{source.suffix}")
    config=Config(work="benchmark",memory_db=str(session/"terms.sqlite3"),
                  translation_backend=args.translator, translation_mode=args.mode,
                  local_files_only=args.offline,debug_mask=True)
    pool=ModelPool(config.device,config.memory_policy)
    try:
        report=run_batch(inputs,session/"outputs",config,pool=pool,force=True,
            progress=lambda e:print(e.get("message",e["type"]),flush=True))
        report["sample"]={"path":str(source),"sha256":file_hash(source),"repeats":args.rounds,
            "conditions":"Same image repeated sequentially, staged GPU residency, real OCR/translation/cleanup; first page includes cold model loading. Later pages reuse cached models when host/GPU memory permits. Windows low-commit translation evictions and reloads are reported in model_evictions/model_load_counts and included in timings. Page totals include PNG encoding but exclude final file commit; batch total includes all file commits."}
        usable=next((p for p in report["pages"] if p.get("blocks") and p.get("status") in {"done","partial"}),None)
        if args.compare_modes and usable:
            blocks=[TextBlock(**row) for row in usable["blocks"]]
            report["translation_comparison"]=compare_modes(blocks,config,pool,session)
        atomic_json(session/"benchmark.json",report)
        print("Benchmark:",session/"benchmark.json")
        return 1 if report["failed"] else 0
    finally:
        pool.close()


if __name__ == "__main__":
    raise SystemExit(main())
