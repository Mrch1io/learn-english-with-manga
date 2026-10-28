# Summary of Experimental Evidence

**Language / 语言**: English | [简体中文](SUMMARY.md)

This page summarizes the saved experimental results; for the data, see the [evidence index](README_EN.md).

The local four-condition comparison uses OpenMantra, with development and test sets split by work and configurations frozen before the official runs; reference translations never enter the prompt, and failed translations are scored as empty translations. All intervals below are 95% intervals from a page-paired bootstrap (2,000 iterations); the sampling and missing-data rules for each experiment are in the corresponding report.

## 1. Model choice: VNTL 8B (Q4_K_M) vs. Qwen2.5-3B, chrF++

| Work | Regions | VNTL | + OCR input | + 1200-character context | Qwen | Failures VNTL / Qwen |
|---|---:|---:|---:|---:|---:|---:|
| tojime (test work, first 24 pages) | 224 | 30.463 | 30.816 | 31.409 | 27.966 | 0 / 1 |
| rasetugari (new work, 54 pages) | 358 | 29.773 | 29.298 | 29.603 | 24.005 | 3 / 11 |

| Difference | tojime | rasetugari |
|---|---|---|
| VNTL − Qwen | **+2.497 [0.707, 4.503]** | **+5.768 [3.873, 7.555]** |
| OCR input − official Japanese | +0.353 [−0.472, 1.321] | −0.475 [−1.841, 0.477] |
| 1200-character context − no context | +0.946 [−0.642, 2.461] | −0.170 [−1.206, 0.811] |

tojime human review, severe-error rate difference (percentage points): VNTL − Qwen **−14.89 [−28.26, −2.08]** (47 pairs); context − no context −10.87 [−19.57, −4.17] (46 pairs); OCR − official Japanese −2.13 [−6.52, 0.00] (47 pairs). The two models differ in size, training, quantization and prompting, so the results only support a comparison of the current deployment configurations. The experiments use official text boxes; a small number of works and a single reviewer cannot represent overall or end-to-end quality.

Sources: [tojime](results/test24-systems/comparison.json) · [rasetugari](results/riskflag/automatic/comparison.json) · [human review](results/R1/analysis/summary.json)

## 2. OCR: Manga-OCR, tojime, 224 regions, 2,598 characters

Strict NFC CER is **10.24%** (266 character edits); after NFC, converting only full-width ASCII to half-width and removing Unicode whitespace, it is **1.89%** (49 character edits), and 78 regions differ only in formatting; all other punctuation and kana are kept. p0001:t000 recognized 「シオラ」 as 「シボラ」, and the translated name became `Shibo-ra`.

Sources: [scores.json](results/test24-label/scores.json) · [case-by-case comparison](results/test24-label/cases.csv)

## 3. Risk flagging (four-system disagreement, 30% review budget): failed, abandoned

Preregistered criteria: the lower bound of the interval for the recall gain over random > 0, and the lower bound of the AUROC interval > 0.5.

| | Seen pilot tojime | New work rasetugari |
|---|---|---|
| Labeled regions / severe errors | 46 / 9 | 151 / 18 |
| Severe errors caught / regions checked | 6 / 14 | 3 / 46 |
| Recall gain over random (percentage points) | +36.23 [+8.89, +68.09] | −13.80 [−30.41, +3.27] |
| AUROC | 0.754 [0.626, 0.882] | **0.342 [0.206, 0.496]** |

Post hoc analysis: 15 of the 18 severe errors involve character reference or cross-bubble correspondence; their median rank is 125th out of 151, only 2 of them are in the top 46, and the random expectation is about 4.6. This grouping was analyzed after seeing the results; it is an explanatory clue and cannot be treated as an independently confirmed error mechanism.

Sources: [protocol](results/riskflag/PROTOCOL_EN.md) · [report](results/riskflag/REPORT_EN.md) · [summary.json](results/riskflag/v1/summary.json) · [post hoc analysis](results/riskflag/POSTHOC_EN.md)

## 4. API comparison

The 24 seen regions were selected by their old labels as 12 severe errors and 12 normal controls (23 pages), so the proportions are not overall error rates. qwen3.8-max, text only (given the whole page of Japanese): **2/24** severe errors versus **14/24** for local VNTL, a difference of −50.00 [−69.23, −30.43] percentage points; character/action reference 2/24 versus 12/24, fixing 11 of VNTL's 12 and introducing 1 new one. The main comparison (with image − text only) is 3/24 versus 2/24, which did not meet the predefined go-ahead criteria. One sample was replaced after the provider blocked it, so the results are exploratory; differences in model and context limit what the API−VNTL comparison can be attributed to.

Sources: [report](results/multimodal-api-qwen38/replacement-v1/REPORT_EN.md) · [results.json](results/multimodal-api-qwen38/replacement-v1/analysis-v1/results.json)

## 5. Engineering validation

Model-free CPU microbenchmark: the linked-candidate scan takes 0.119 / 1.117 / 15.188 ms for 20 / 200 / 2,000 boxes (median of 5 runs after warm-up); 2,000 boxes plus the intra-word comma check take 56.1 ms on the first run and 18.1 ms after warm-up, using six groups of repeated word examples, so the caching benefit depends on how repetitive the input is. llama.cpp's native grammar sampler, loading only the vocabulary, matched expectations on all 14 decisions, which shows that the constraint implementation passes these cases but does not guarantee that translations are semantically correct.

Locked-name samples from separate real generation runs: `Misaki Takanashi, where is that key?` and `Misaki Takanashi, go to Silver Gate first.` The microbenchmarks are not full-page inference times, and a few generated samples cannot prove accuracy for arbitrary terms.

Sources: [scan](validation/fragment-bench.json) · [intra-word comma](validation/boundary-bench.json) · [grammar](validation/vntl-grammar.json) · [follow-up record](validation/VNTL_EN.md)
