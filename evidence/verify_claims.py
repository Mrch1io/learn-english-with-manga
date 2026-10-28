"""Check README numbers against the saved evidence; reads saved files only, no model or GPU.

Each claim is the exact text shown in the Chinese README_CN.md and in the English README.md.
The text is rebuilt from the evidence files and must match; when a README is present it must
also still contain that text.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
import random
from statistics import fmean
import sys

EVIDENCE = Path(__file__).resolve().parent
README_CN = EVIDENCE.parent / "README_CN.md"
README_EN = EVIDENCE.parent / "README.md"
READMES = {"zh": README_CN, "en": README_EN}
MINUS = "−"
HUMAN_SYSTEMS = ("vntl-gold-c0", "vntl-ocr-c0", "vntl-gold-c1200", "qwen-gold-c0")
HUMAN_METRICS = ("faithfulness", "severe_error", "naturalness")
HUMAN_TAGS = ("omission", "negation", "reference", "name_term", "alignment", "ocr_propagation",
              "repetition", "addition", "tone", "order", "annotation", "other")
HUMAN_COMPARISONS = (("vntl-gold-c0", "vntl-ocr-c0"), ("vntl-gold-c0", "vntl-gold-c1200"),
                     ("qwen-gold-c0", "vntl-gold-c0"))


def load(relative):
    return json.loads((EVIDENCE / relative).read_text(encoding="utf-8"))


def rows(relative):
    with open(EVIDENCE / relative, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def num(value, digits, sign=False):
    return (f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}").replace("-", MINUS)


def interval(low, high, digits):
    return f"[{num(low, digits)}, {num(high, digits)}]"


def ordinal(value):
    suffix = "th" if 10 <= value % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def contrast(comparison, baseline, candidate):
    return next(c for c in comparison["contrasts"]
                if c["baseline"] == baseline and c["candidate"] == candidate)


def crosses_zero(item):
    return item["ci95"][0] < 0 < item["ci95"][1]


def _human_rows(ratings, freeze, predictions):
    """Recover R1's blind order, verified against every archived translation."""
    protocol = freeze["human_review"]
    blocks = protocol["block_ids"]
    if len(set(blocks)) != len(blocks) or protocol["actual_cases"] != len(blocks):
        raise ValueError("human_review.block_ids: duplicate IDs or incorrect actual_cases")
    source = {row["item"]: row for row in ratings["rows"]}
    expected = {f"item{i:03}-{j}" for i in range(1, len(blocks) + 1) for j in range(1, 5)}
    if len(source) != len(ratings["rows"]) or source.keys() != expected:
        raise ValueError("ratings.rows: duplicate, missing or unexpected item IDs")
    cases = {(row["system"], row["block_id"]): row for row in predictions}
    if len(cases) != len(predictions):
        raise ValueError("predictions: duplicate system/block IDs")
    grouped = {system: {} for system in HUMAN_SYSTEMS}
    generator = random.Random(protocol["seed"])
    for i, block in enumerate(blocks, 1):
        order = list(HUMAN_SYSTEMS)
        generator.shuffle(order)
        for j, system in enumerate(order, 1):
            row = source[f"item{i:03}-{j}"]
            for key, low, high in (("faithfulness", 0, 4), ("naturalness", 1, 3)):
                value = row[key]
                if value is not None and (type(value) is not int or not low <= value <= high):
                    raise ValueError(f"{row['item']}.{key}: invalid rating")
            if row["severe_error"] is not None and type(row["severe_error"]) is not bool:
                raise ValueError(f"{row['item']}.severe_error: expected boolean or null")
            if (not isinstance(row["tags"], list) or any(tag not in HUMAN_TAGS for tag in row["tags"])
                    or len(set(row["tags"])) != len(row["tags"])):
                raise ValueError(f"{row['item']}.tags: invalid or duplicate tags")
            case = cases[(system, block)]
            if row["candidate"] != case["hypothesis"] or row["source_text"] != case["source_text"]:
                raise ValueError(f"{row['item']}: blind mapping does not match saved translation")
            grouped[system][block] = dict(row, page_id=block.rsplit(":", 1)[0], block_id=block)
    return grouped


def _human_complete(row):
    return all(row[key] is not None for key in HUMAN_METRICS)


def _human_system(rows):
    complete = [row for row in rows if _human_complete(row)]
    unscored = sum(all(row[key] is None for key in HUMAN_METRICS) for row in rows)
    return {
        "expected_judgments": len(rows), "completed": len(complete),
        "partial": len(rows) - len(complete) - unscored, "unscored": unscored,
        "empty_candidates": sum(not row["candidate"].strip() for row in rows),
        "faithfulness_mean": fmean(row["faithfulness"] for row in complete) if complete else None,
        "naturalness_mean": fmean(row["naturalness"] for row in complete) if complete else None,
        "naturalness_denominator": len(complete),
        "severe_error_rate": fmean(row["severe_error"] for row in complete) if complete else None,
        "tags": dict(Counter(tag for row in complete for tag in row["tags"])),
        "weighting": "completed judgments; use paired metrics for system comparison",
    }


def _human_metrics(pairs):
    """Page bootstrap; aggregating sums/counts avoids copying rows on each draw."""
    pages = defaultdict(list)
    for pair in pairs:
        pages[pair["page_id"]].append(pair)
    page_rows = [pages[page] for page in sorted(pages)]
    counts = [len(items) for items in page_rows]
    sums = [[sum(pair["candidate"][key] - pair["baseline"][key] for pair in items)
             for items in page_rows] for key in HUMAN_METRICS]
    samples, seed = 2000, 20261018
    bootstraps = [[] for _ in HUMAN_METRICS]
    generator = random.Random(seed)
    for _ in range(samples if pairs else 0):
        chosen = generator.choices(range(len(page_rows)), k=len(page_rows))
        denominator = sum(counts[index] for index in chosen)
        for values, boot in zip(sums, bootstraps):
            boot.append(sum(values[index] for index in chosen) / denominator)

    def percentile(values, probability):
        if not values:
            return None
        index = (len(values) - 1) * probability
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)

    metrics = {}
    for key, differences, boot in zip(HUMAN_METRICS, sums, bootstraps):
        boot.sort()
        metrics[key] = {
            "regions": len(pairs), "pages": len(page_rows), "paired_judgments": len(pairs),
            "baseline": fmean(pair["baseline"][key] for pair in pairs) if pairs else None,
            "candidate": fmean(pair["candidate"][key] for pair in pairs) if pairs else None,
            "delta": sum(differences) / len(pairs) if pairs else None,
            "ci95": [percentile(boot, .025), percentile(boot, .975)],
            "unit": "page; reviewers averaged within each region",
            "delta_direction": "candidate - baseline", "samples": samples, "seed": seed,
            "interval_status": "descriptive_small_sample",
        }
    return metrics


def _human_comparison(baseline, candidate, grouped, reviewer):
    left, right = grouped[baseline], grouped[candidate]
    coverage = dict.fromkeys(("paired", "baseline_only", "candidate_only", "neither_complete"), 0)
    pairs = []
    for block in sorted(left):
        a, b = _human_complete(left[block]), _human_complete(right[block])
        coverage["paired" if a and b else "baseline_only" if a else
                 "candidate_only" if b else "neither_complete"] += 1
        if a and b:
            pairs.append({"reviewer": reviewer, "block_id": block, "page_id": left[block]["page_id"],
                          "baseline": left[block], "candidate": right[block]})
    severe = dict.fromkeys(("both_severe", "resolved", "introduced", "neither_severe"), 0)
    tags = {tag: dict.fromkeys(("both_flagged", "baseline_only", "candidate_only"), 0)
            for tag in HUMAN_TAGS}
    for pair in pairs:
        a, b = pair["baseline"], pair["candidate"]
        severe["both_severe" if a["severe_error"] and b["severe_error"] else
               "resolved" if a["severe_error"] else "introduced" if b["severe_error"] else
               "neither_severe"] += 1
        for tag in set(a["tags"]) | set(b["tags"]):
            tags[tag]["both_flagged" if tag in a["tags"] and tag in b["tags"] else
                      "baseline_only" if tag in a["tags"] else "candidate_only"] += 1
    return {"baseline": baseline, "candidate": candidate, "expected_judgment_pairs": len(left),
            "coverage": coverage, "metrics": _human_metrics(pairs), "severe_transitions": severe,
            "tag_transitions": tags, "transitions_denominator": len(pairs), "paired_rows": pairs}


def rebuild_human_summary(summary=None, ratings=None, freeze=None, predictions=None):
    """Rebuild this archive's single-reviewer R1 statistics without changing files or inputs.

    The recovered shuffle matches all 183 originally stored item/system assignments and
    all 192 saved translations. The bootstrap reproduces all nine archived intervals.
    """
    summary = load("results/R1/analysis/summary.json") if summary is None else summary
    ratings = load("results/R1/source/ratings_1.json") if ratings is None else ratings
    freeze = load("freezes/openmantra_v1_test.json") if freeze is None else freeze
    predictions = rows("results/test24-systems/cases.csv") if predictions is None else predictions
    grouped = _human_rows(ratings, freeze, predictions)
    assignments = {row["item"]: system for system, blocks in grouped.items() for row in blocks.values()}
    for comparison in summary["comparisons"]:
        for pair in comparison["paired_rows"]:
            for side in ("baseline", "candidate"):
                if assignments.get(pair[side]["item"]) != comparison[side]:
                    raise ValueError(f"{pair[side]['item']}: saved pair has incorrect system assignment")
    result = deepcopy(summary)
    result["systems"] = {system: _human_system(list(grouped[system].values()))
                         for system in sorted(HUMAN_SYSTEMS)}
    completed = sum(system["completed"] for system in result["systems"].values())
    expected = len(ratings["rows"])
    attested = (ratings["reviewer_kind"] == "human" and ratings["review_status"] == "human_reviewed"
                and ratings["mode"]["kind"] == "human_review")
    result.update(expected_regions=len(freeze["human_review"]["block_ids"]), expected_candidates=expected,
                  completed_judgments=completed, human_attested_exports=int(attested),
                  status="complete" if completed == expected else "partial")
    result["reviewers"] = [{"reviewer": ratings["reviewer"], "expected": expected,
                            "completed": completed, "human_attested": attested,
                            "assistance": {"independent_human_review": attested,
                                           "reviewed_items": list(ratings["mode"]["reviewed_items"])}}]
    result["comparisons"] = [_human_comparison(left, right, grouped, ratings["reviewer"])
                             for left, right in HUMAN_COMPARISONS]
    return result


def _human_differences(saved, expected, path="summary"):
    if isinstance(expected, dict) and isinstance(saved, dict):
        for key in sorted(saved.keys() | expected.keys()):
            if key not in saved or key not in expected:
                yield f"{path}.{key}: missing or unexpected field"
            else:
                yield from _human_differences(saved[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list) and isinstance(saved, list):
        if len(saved) != len(expected):
            yield f"{path}: expected {len(expected)} entries, got {len(saved)}"
        else:
            for index, (actual, wanted) in enumerate(zip(saved, expected)):
                yield from _human_differences(actual, wanted, f"{path}[{index}]")
    elif isinstance(expected, float) and type(saved) in (int, float):
        if not abs(saved - expected) <= 1e-12:
            yield f"{path}: expected {expected!r}, got {saved!r}"
    elif type(saved) is not type(expected) or saved != expected:
        yield f"{path}: expected {expected!r}, got {saved!r}"


def check_human_summary(summary=None, ratings=None, freeze=None, predictions=None):
    summary = load("results/R1/analysis/summary.json") if summary is None else summary
    try:
        rebuilt = rebuild_human_summary(summary, ratings, freeze, predictions)
    except (KeyError, ValueError, TypeError) as error:
        return [f"human source validation: {error}"]
    return list(_human_differences(summary, rebuilt))


def build_claims():
    """Return (name, {language: (README text, text rebuilt from evidence)}, evidence file)."""
    claims = []

    def claim(name, zh, en, source):
        claims.append((name, {"zh": zh, "en": en}, source))

    tojime_file = "results/test24-systems/comparison.json"
    rasetugari_file = "results/riskflag/automatic/comparison.json"
    tojime, rasetugari = load(tojime_file), load(rasetugari_file)
    for name, data, source, readme in (
            ("tojime VNTL - Qwen chrF++", tojime, tojime_file, "2.497 [0.707, 4.503]"),
            ("rasetugari VNTL - Qwen chrF++", rasetugari, rasetugari_file, "5.768 [3.873, 7.555]")):
        item = contrast(data, "qwen-gold-c0", "vntl-gold-c0")
        text = (readme, f"{num(item['delta'], 3)} {interval(*item['ci95'], 3)}")
        claim(name, text, text, source)

    def failures(data, label):
        score = data["runs"][label]["score"]
        return f"{score['failed_blocks']}/{score['eligible_blocks']}"
    vntl = failures(tojime, "vntl-gold-c0"), failures(rasetugari, "vntl-gold-c0")
    qwen = failures(tojime, "qwen-gold-c0"), failures(rasetugari, "qwen-gold-c0")
    claim("translation failures",
          ("VNTL 为 0/224、3/358，Qwen 为 1/224、11/358", f"VNTL 为 {vntl[0]}、{vntl[1]}，Qwen 为 {qwen[0]}、{qwen[1]}"),
          ("VNTL 0/224 and 3/358; Qwen 1/224 and 11/358", f"VNTL {vntl[0]} and {vntl[1]}; Qwen {qwen[0]} and {qwen[1]}"),
          f"{tojime_file}, {rasetugari_file}")
    score = rasetugari["runs"]["vntl-gold-c0"]["score"]
    claim("rasetugari size",
          ("54 页、358 区", f"{score['pages']} 页、{score['eligible_blocks']} 区"),
          ("54 pages, 358 regions", f"{score['pages']} pages, {score['eligible_blocks']} regions"),
          rasetugari_file)
    score = tojime["runs"]["vntl-gold-c0"]["score"]
    claim("tojime size",
          ("前 24 页，共224 区", f"前 {score['pages']} 页，共{score['eligible_blocks']} 区"),
          ("first 24 pages of the test work tojime_no_siora, 224 regions in total",
           f"first {score['pages']} pages of the test work tojime_no_siora, {score['eligible_blocks']} regions in total"),
          tojime_file)

    others = [contrast(data, "vntl-gold-c0", candidate) for data in (tojime, rasetugari)
              for candidate in ("vntl-ocr-c0", "vntl-gold-c1200")]
    cross = all(map(crosses_zero, others))
    zh = "换成 OCR 输入、加上 1200 字符邻近上下文，两个区间都跨零"
    en = "with OCR input and with 1200 characters of nearby context added, both intervals cross zero"
    claim("OCR input and context intervals cross zero",
          (zh, zh if cross else "at least one interval excludes zero"),
          (en, en if cross else "at least one interval excludes zero"),
          f"{tojime_file}, {rasetugari_file}")

    human_file = "results/R1/analysis/summary.json"
    human = load(human_file)["comparisons"]

    def severe_difference(baseline, candidate):
        severe = next(c for c in human if c["baseline"] == baseline
                      and c["candidate"] == candidate)["metrics"]["severe_error"]
        low, high = (100 * value for value in severe["ci95"])
        return num(-100 * severe["delta"], 2), interval(low, high, 2)
    value, ci = severe_difference("qwen-gold-c0", "vntl-gold-c0")
    claim("tojime human severe-error difference",
          ("低 14.89 个百分点 [−28.26, −2.08]", f"低 {value} 个百分点 {ci}"),
          ("14.89 percentage points lower [−28.26, −2.08]", f"{value} percentage points lower {ci}"),
          human_file)
    value, ci = severe_difference("vntl-gold-c0", "vntl-gold-c1200")
    claim("tojime human severe-error difference with context",
          ("加上下文后严重错误率低 10.87 个百分点 [−19.57, −4.17]", f"加上下文后严重错误率低 {value} 个百分点 {ci}"),
          ("adding context lowered the severe-error rate by 10.87 percentage points [−19.57, −4.17]",
           f"adding context lowered the severe-error rate by {value} percentage points {ci}"),
          human_file)

    ocr_file = "results/test24-label/scores.json"
    ocr = load(ocr_file)
    for key, zh, en in (("cer_nfc", "10.24%（266/2,598 字）", "10.24% (266/2,598 characters)"),
                        ("cer_width_space", "1.89%（49/2,598 字）", "1.89% (49/2,598 characters)")):
        cer = ocr[key]
        percent, edits = f"{100 * cer['cer']:.2f}%", f"{cer['edit_errors']}/{cer['reference_characters']:,}"
        claim(f"OCR {key}", (zh, f"{percent}（{edits} 字）"), (en, f"{percent} ({edits} characters)"), ocr_file)
    claim("OCR format-only regions",
          ("224 个区域里，有 78 个只是格式不同",
           f"{ocr['blocks']} 个区域里，有 {ocr['format_only_regions']} 个只是格式不同"),
          ("of the 224 regions, 78 differ only in formatting",
           f"of the {ocr['blocks']} regions, {ocr['format_only_regions']} differ only in formatting"),
          ocr_file)
    item = contrast(tojime, "vntl-gold-c0", "vntl-ocr-c0")
    text = ("+0.353 [−0.472, 1.321]", f"{num(item['delta'], 3, sign=True)} {interval(*item['ci95'], 3)}")
    claim("OCR input chrF++ difference", text, text, tojime_file)

    ocr_case = next(r for r in rows("results/test24-label/cases.csv")
                    if r["block_id"] == "tojime_no_siora:p0001:t000")
    translated = next(r for r in rows("results/test24-systems/cases.csv")
                      if r["system"] == "vntl-ocr-c0" and r["block_id"] == "tojime_no_siora:p0001:t000")
    found = ("シオラ" in ocr_case["source_text"] and "シボラ" in ocr_case["ocr_text"]
             and "Shibo-ra" in translated["hypothesis"])
    zh = "把「シオラ」识别成「シボラ」，译文里的名字就从 `Shiora` 变成了 `Shibo-ra`"
    en = "recognized 「シオラ」 as 「シボラ」, and the name in the translation changed from `Shiora` to `Shibo-ra`"
    claim("OCR name error propagates", (zh, zh if found else "case text differs"),
          (en, en if found else "case text differs"),
          "results/test24-label/cases.csv, results/test24-systems/cases.csv")

    risk_file = "results/riskflag/v1/summary.json"
    risk = load(risk_file)
    signal = risk["analysis"]["systems"]["vntl-gold-c0"]["S1"]
    budget = next(top for top in signal["point"]["top"] if top["fraction"] == 0.3)
    claim("risk-flag budget",
          ("在 151 个区域上只检查最可疑的 30%（前 46 名）",
           f"在 {signal['point']['n']} 个区域上只检查最可疑的 30%（前 {budget['k']} 名）"),
          ("Across 151 regions, only the most suspicious 30% (the top 46) are checked",
           f"Across {signal['point']['n']} regions, only the most suspicious 30% (the top {budget['k']}) are checked"),
          risk_file)
    auroc = signal["point"]["auroc"]
    claim("risk-flag AUROC",
          ("AUROC 低于 0.5（0.342）", f"AUROC {'低于' if auroc < 0.5 else '不低于'} 0.5（{auroc:.3f}）"),
          ("AUROC below 0.5 (0.342)", f"AUROC {'below' if auroc < 0.5 else 'not below'} 0.5 ({auroc:.3f})"),
          risk_file)
    failed = risk["primary_decision"] == "failed"
    claim("risk-flag preregistered decision",
          ("**放弃该方向。**", "**放弃该方向。**" if failed else risk["primary_decision"]),
          ("**Abandon this direction.**", "**Abandon this direction.**" if failed else risk["primary_decision"]),
          risk_file)

    posthoc_file = "results/riskflag/provenance/posthoc.json"
    posthoc = load(posthoc_file)
    group, population = posthoc["groups"]["severe_reference_or_alignment"], posthoc["population"]
    claim("severe errors by type",
          ("18 个严重错误中，15 个", f"{population['severe']} 个严重错误中，{group['count']} 个"),
          ("of the 18 severe errors on the new work, 15",
           f"of the {population['severe']} severe errors on the new work, {group['count']}"),
          posthoc_file)
    expected = group["count"] * posthoc["top_budget"]["k"] / population["sampled_regions"]
    claim("reference errors ranked low",
          ("风险排名中位数是 151 名中的第 125 名",
           f"风险排名中位数是 {population['sampled_regions']} 名中的第 {group['risk_rank_median']} 名"),
          ("median risk rank of 125th out of 151",
           f"median risk rank of {ordinal(group['risk_rank_median'])} out of {population['sampled_regions']}"),
          posthoc_file)
    claim("reference errors caught vs random",
          ("只抓到其中 2 个，随机抽也能抽中约 4.6 个",
           f"只抓到其中 {group['top46_count']} 个，随机抽也能抽中约 {expected:.1f} 个"),
          ("caught only 2 of them, while random sampling would catch about 4.6",
           f"caught only {group['top46_count']} of them, while random sampling would catch about {expected:.1f}"),
          posthoc_file)

    api_file = "results/multimodal-api-qwen38/replacement-v1/analysis-v1/results.json"
    api = load(api_file)
    severe = api["comparisons"]["A_vs_VNTL"]["fields"]["severe_error"]
    reference = api["comparisons"]["A_vs_VNTL"]["fields"]["reference"]

    def rate(arm, field):
        values = api["arms"][arm]["fields"][field]
        return f"{values['true']}/{values['assessed']}", f"{100 * values['rate']:.2f}%"
    (a_count, a_rate), (v_count, v_rate) = rate("A", "severe_error"), rate("VNTL", "severe_error")
    claim("API severe errors",
          ("2/24（8.33%），同一批区域上，本地 VNTL 为 14/24（58.33%）",
           f"{a_count}（{a_rate}），同一批区域上，本地 VNTL 为 {v_count}（{v_rate}）"),
          ("2/24 (8.33%); on the same regions, local VNTL had 14/24 (58.33%)",
           f"{a_count} ({a_rate}); on the same regions, local VNTL had {v_count} ({v_rate})"),
          api_file)
    low, high = (100 * value for value in severe["ci95"]["delta"])
    value, ci = num(100 * severe["delta"], 2), interval(low, high, 2)
    claim("API severe-error difference",
          ("−50.00 个百分点 [−69.23, −30.43]", f"{value} 个百分点 {ci}"),
          ("−50.00 percentage points [−69.23, −30.43]", f"{value} percentage points {ci}"),
          api_file)
    right = f"{reference['right_true']}/{reference['total_regions']}"
    left = f"{reference['left_true']}/{reference['total_regions']}"
    claim("API reference errors",
          ("是 2/24 对 12/24，VNTL 的 12 个指代错误里有 11 个被 API 译对了",
           f"是 {right} 对 {left}，VNTL 的 {reference['left_true']} "
           f"个指代错误里有 {reference['repair_count']} 个被 API 译对了"),
          ("2/24 versus 12/24, and the API fixed 11 of VNTL's 12 (",
           f"{right} versus {left}, and the API fixed {reference['repair_count']} of VNTL's "
           f"{reference['left_true']} ("),
          api_file)

    fragment_file = "validation/fragment-bench.json"
    timings = sorted(load(fragment_file)["results"], key=lambda r: r["blocks"])
    text = ("| 0.119 ms | 1.117 ms | 15.188 ms |",
            "| " + " | ".join(f"{1000 * r['median_seconds']:.3f} ms" for r in timings) + " |")
    claim("continuation scan benchmark", text, text, fragment_file)
    boundary_file = "validation/boundary-bench.json"
    boundary = load(boundary_file)
    first, warm = (f"{1000 * boundary[key]:.1f} ms" for key in ("first_scan_seconds", "warm_median_seconds"))
    claim("word-boundary benchmark",
          ("首次 56.1 ms，预热后中位数 18.1 ms", f"首次 {first}，预热后中位数 {warm}"),
          ("the first run takes 56.1 ms, and the median after warm-up is 18.1 ms",
           f"the first run takes {first}, and the median after warm-up is {warm}"),
          boundary_file)
    grammar_file = "validation/vntl-grammar.json"
    cases = load(grammar_file)["cases"]
    matched = sum(case["expected"] == case["accepted"] for case in cases)
    every = matched == len(cases)
    claim("native grammar sampler",
          ("14 项接受／拒绝判定全部符合预期",
           f"{len(cases)} 项接受／拒绝判定" + ("全部符合预期" if every else f"{matched} 项符合预期")),
          ("All 14 accept/reject decisions matched expectations",
           f"All {len(cases)} accept/reject decisions matched expectations" if every
           else f"{matched} of {len(cases)} accept/reject decisions matched expectations"),
          grammar_file)
    followup_file = "validation/VNTL.md"
    followup = (EVIDENCE / followup_file).read_text(encoding="utf-8")
    for text in ("`Misaki Takanashi, where is that key?`", "`Misaki Takanashi, go to Silver Gate first.`"):
        pair = (text, text if text.strip("`") in followup else "not in report")
        claim("locked-name output", pair, pair, followup_file)
    return claims


def check(readme_path=README_CN, language="zh"):
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
    results = []
    for name, texts, source in build_claims():
        expected, rebuilt = texts[language]
        problems = []
        if rebuilt != expected:
            problems.append(f"evidence gives {rebuilt!r}")
        if readme is not None and expected not in readme:
            problems.append(f"text no longer in {readme_path.name}")
        results.append((name, expected, source, problems))
    results.append(("tojime human summary matches original ratings", "source-derived statistics",
                    "results/R1/source/ratings_1.json, results/R1/analysis/summary.json",
                    check_human_summary()))
    return results, readme is not None


def main():
    failed = 0
    for language, path in READMES.items():
        results, has_readme = check(path, language)
        for name, expected, source, problems in results:
            print(f"{'FAIL' if problems else 'PASS'}  {name}: {expected}  <- {source}")
            for problem in problems:
                print(f"      {problem}")
        missed = sum(bool(problems) for *_, problems in results)
        failed += missed
        scope = f"evidence and {path.name}" if has_readme else f"evidence only ({path.name} not found)"
        print(f"\n{len(results) - missed}/{len(results)} claims match ({scope}).\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
