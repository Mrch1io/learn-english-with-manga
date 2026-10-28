# Research Evidence Index

**Language / 语言**: English | [简体中文](README.md)

This directory holds the predictions, scores and statistical results from the translation experiments and engineering validation. Start with the [research summary](SUMMARY_EN.md); the design of the local four-condition experiment is in the [methods](PROTOCOL_EN.md). The full original images, run logs and historical source code are not all included.

| Finding | Evidence |
|---|---|
| tojime: VNTL−Qwen chrF++ +2.497 [0.707, 4.503]; OCR-input and context ablations | [statistics](results/test24-systems/comparison.json), [per-item predictions](results/test24-systems/cases.csv) |
| rasetugari: VNTL−Qwen +5.768 [3.873, 7.555]; failures 3/358 and 11/358 | [statistics](results/riskflag/automatic/comparison.json), [per-item predictions](results/riskflag/automatic/cases.csv) |
| tojime human review: severe-error rate difference −14.89 [−28.26, −2.08] percentage points | [analysis](results/R1/analysis/summary.json), [original ratings](results/R1/source/ratings_1.json) |
| OCR: strict CER 10.24%, 1.89% after normalization; 78 regions differ only in formatting | [metrics](results/test24-label/scores.json), [case-by-case comparison](results/test24-label/cases.csv) |
| Risk flagging: 151 regions, top 46 checked; AUROC 0.342, failed | [protocol](results/riskflag/PROTOCOL_EN.md), [supplementary analysis rules](results/riskflag/SENSITIVITY_EN.md), [report](results/riskflag/REPORT_EN.md), [statistics](results/riskflag/v1/summary.json) |
| Post hoc analysis: 15 reference/cross-bubble errors, median rank 125; only 2 caught in the top 46 | [analysis notes](results/riskflag/POSTHOC_EN.md), [computed results](results/riskflag/provenance/posthoc.json) |
| API exploration: text-only severe errors 2/24, VNTL 14/24; no additional improvement from adding images | [report](results/multimodal-api-qwen38/replacement-v1/REPORT_EN.md), [statistics](results/multimodal-api-qwen38/replacement-v1/analysis-v1/results.json) |
| Frozen configurations and samples | [local four conditions](freezes/openmantra_v1_test.json), [risk flagging](freezes/riskflag-confirm.json) |
| Candidate scan: 0.119/1.117/15.188 ms for 20/200/2,000 boxes | [scan microbenchmark](validation/fragment-bench.json), [runnable scan script](../tools/benchmark_fragments.py), [word-boundary microbenchmark](validation/boundary-bench.json) |
| Locked names: 14 native grammar decisions; real-generation follow-up | [grammar results](validation/vntl-grammar.json), [validation report](validation/VNTL_EN.md) |

For data attribution and licensing, see [LICENSE-OpenMantra_EN.md](LICENSE-OpenMantra_EN.md).
