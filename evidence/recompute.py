"""Recompute chrF++, paired page bootstrap and OCR CER from the saved predictions.

Install the optional scorer with: python -m pip install sacrebleu==2.6.0.
Recomputation itself needs no model, GPU or network. The
scoring mirrors the project's research scorer: corpus chrF++ per work, work-macro mean,
failed translations kept as empty hypotheses, stratified paired page bootstrap.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import random
from statistics import fmean
import sys
import unicodedata

EVIDENCE = Path(__file__).resolve().parent
RUNS = {
    "tojime": ("results/test24-systems/cases.csv", "results/test24-systems/comparison.json"),
    "rasetugari": ("results/riskflag/automatic/cases.csv",
                   "results/riskflag/automatic/comparison.json"),
}
OCR = ("results/test24-label/cases.csv", "results/test24-label/scores.json")
FULLWIDTH_ASCII = str.maketrans({code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)})
TOLERANCE = 1e-9


def chrf():
    from sacrebleu.metrics import CHRF
    return CHRF(char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False,
                eps_smoothing=False)


def read_rows(relative):
    with open(EVIDENCE / relative, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def page_statistics(rows, metric):
    """{system: {page_id: summed chrF++ statistics}} over eligible blocks only."""
    blocks = defaultdict(lambda: defaultdict(list))
    for row in rows:
        page = blocks[row["system"]][row["page_id"]]
        if row["mt_eligible"] == "True":
            page.append(row)
    stats = {}
    for system, pages in blocks.items():
        stats[system] = {}
        for page_id, items in pages.items():
            if items:
                counts = metric._extract_corpus_statistics([r["hypothesis"] for r in items],
                                                           [[r["reference_text"] for r in items]])
                stats[system][page_id] = [sum(column) for column in zip(*counts)]
    return stats


def score(metric, counts, page_ids):
    total = [sum(column) for column in zip(*(counts[p] for p in page_ids))]
    return metric._compute_score_from_stats(total).score


def percentile(values, probability):
    index = (len(values) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def by_work(page_ids):
    works = defaultdict(list)
    for page_id in sorted(page_ids):
        works[page_id.split(":")[0]].append(page_id)
    return dict(sorted(works.items()))


def work_macro(metric, counts):
    return fmean(score(metric, counts, ids) for ids in by_work(counts).values())


def paired_bootstrap(metric, baseline, candidate, *, samples, seed):
    works = by_work(baseline)
    delta = fmean(score(metric, candidate, ids) - score(metric, baseline, ids) for ids in works.values())
    generator, differences = random.Random(seed), []
    for _ in range(samples):
        deltas = []
        for ids in works.values():
            chosen = [ids[generator.randrange(len(ids))] for _ in ids]
            deltas.append(score(metric, candidate, chosen) - score(metric, baseline, chosen))
        differences.append(fmean(deltas))
    differences.sort()
    return delta, [percentile(differences, 0.025), percentile(differences, 0.975)]


def levenshtein(reference, hypothesis):
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def cer(rows, *, fold):
    errors = characters = format_only = 0
    for row in rows:
        reference, hypothesis = (unicodedata.normalize("NFC", row[k]) for k in ("source_text", "ocr_text"))
        if fold:
            reference, hypothesis = ("".join(t.translate(FULLWIDTH_ASCII).split())
                                     for t in (reference, hypothesis))
        format_only += row["source_text"] != row["ocr_text"] and reference == hypothesis
        if reference:
            characters += len(reference)
            errors += levenshtein(reference, hypothesis)
    return errors, characters, format_only


def run():
    checks = []

    def compare(name, recomputed, saved):
        ok = all(abs(a - b) <= TOLERANCE for a, b in zip(recomputed, saved))
        checks.append((name, ok, recomputed, saved))

    metric = chrf()
    for work, (cases, comparison) in RUNS.items():
        saved = json.loads((EVIDENCE / comparison).read_text(encoding="utf-8"))
        stats = page_statistics(read_rows(cases), metric)
        for system, run in saved["runs"].items():
            compare(f"{work} {system} chrF++", [work_macro(metric, stats[system])],
                    [run["score"]["work_macro_chrf_pp"]])
        for item in saved["contrasts"]:
            delta, ci95 = paired_bootstrap(metric, stats[item["baseline"]], stats[item["candidate"]],
                                           samples=item["samples"], seed=item["seed"])
            compare(f"{work} {item['candidate']} - {item['baseline']} delta, 95% CI",
                    [delta, *ci95], [item["delta"], *item["ci95"]])

    rows = read_rows(OCR[0])
    saved = json.loads((EVIDENCE / OCR[1]).read_text(encoding="utf-8"))
    errors, characters, _ = cer(rows, fold=False)
    compare("tojime OCR strict CER errors/chars", [errors, characters],
            [saved["cer_nfc"]["edit_errors"], saved["cer_nfc"]["reference_characters"]])
    errors, characters, format_only = cer(rows, fold=True)
    compare("tojime OCR width/space CER errors/chars/format-only", [errors, characters, format_only],
            [saved["cer_width_space"]["edit_errors"], saved["cer_width_space"]["reference_characters"],
             saved["format_only_regions"]])
    return checks


def main():
    checks = run()
    for name, ok, recomputed, saved in checks:
        shown = ", ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in recomputed)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {shown}")
        if not ok:
            print(f"      saved: {saved}")
    failed = sum(not ok for _, ok, *_ in checks)
    print(f"\n{len(checks) - failed}/{len(checks)} recomputed values match the saved results.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
