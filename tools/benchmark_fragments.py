"""Check and benchmark fragment selection on the CPU without loading models.

Run from the project root:
  py -3.12 -B tools/benchmark_fragments.py

This reproduces the saved fragment-bench.json methodology: 20, 200 and 2,000
blocks, one warmup and five measured scans. Fixture construction and correctness
checks are excluded from the timings. Results do not measure translation quality
or end-to-end inference performance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.runtime import TextBlock
from translation.vntl import continuation_pairs


def source_blocks() -> list[TextBlock]:
    rows = [("ーーー！？\n", (103, 48, 182, 236)),
            ("や、やめろ!", (685, 421, 755, 540)),
            ("はっ...", (402, 453, 433, 533)),
            ("はな、", (132, 447, 172, 535)),
            ("せ...", (93, 477, 132, 561)),
            ("だめだ...", (121, 763, 185, 896))]
    return [TextBlock(box, source_text=text, block_id=f"rasetugari:p0037:t{i:03d}",
                      reading_order=i) for i, (text, box) in enumerate(rows)]


def benchmark() -> dict:
    results = []
    for count in (10, 100, 1000):
        blocks = []
        for index in range(count):
            for member in source_blocks()[3:5]:
                x1, y1, x2, y2 = member.box
                member.box = (x1, y1 + index * 500, x2, y2 + index * 500)
                member.block_id = f"pair-{index}-{member.block_id}"
                blocks.append(member)
        before = [asdict(block) for block in blocks]
        expected = [(i, i + 1, "fragment") for i in range(0, len(blocks), 2)]
        if continuation_pairs(blocks) != expected:
            raise AssertionError("Warmup did not select exactly the isolated pairs")
        if [asdict(block) for block in blocks] != before:
            raise AssertionError("Warmup modified source fields")
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            selected = continuation_pairs(blocks)
            samples.append(time.perf_counter() - start)
            if selected != expected or [asdict(block) for block in blocks] != before:
                raise AssertionError("Pair mapping changed or source fields were modified")
        results.append({"pairs": count, "blocks": len(blocks), "repeats": 5,
                        "median_seconds": statistics.median(samples), "seconds": samples})
    return {"scope": "CPU fragment selection only; no model or quality measurement",
            "source_fields_preserved": True, "results": results}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(benchmark(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
