# Supplementary Methods: Exploratory Sensitivity Analysis

**Language / 语言**: English | [简体中文](SENSITIVITY.md)

The original supplement was written after the first stage was complete and before any human ratings existed. It adds an empty-output sensitivity analysis and error-type counts, and **does not change the primary decision for the original S1 at the 30% budget**; the primary test, sampling and statistical rules are in the [methods summary](PROTOCOL_EN.md).

## Excluding empty outputs

If any of the four systems has an empty hypothesis, that sampled region is excluded, and the main system's S1 is then analyzed. An original prediction whose status is not `translated`, or whose `translated_text.strip()` is empty, is treated as an empty hypothesis. Exclusions are determined from the saved outputs; the rule does not change based on severity labels, risk level or translation quality, and each region is excluded only once.

Recall, precision, the difference from random and AUROC are recomputed for the 20%/30%/50% budgets, reporting the number of excluded regions, how many of them are severe errors or null, and the numbers of retained regions and valid labels. This analysis checks whether empty outputs affect the disagreement signal's performance and is an exploratory robustness analysis.

The primary and sensitivity analyses keep the same full set of 54 sampled pages, including pages left with no valid items after exclusion, and reuse the 2,000 iterations, seed=2026102003 and the same page-draw sequence. N, k and the ranking are recomputed after exclusion, but the text features are not. Severity true/false determines inclusion, and null stays missing; there is no filtering by fidelity, naturalness or `reviewer_confidence`.

Whatever the sensitivity results, the primary decision still uses the full sample: for S1 at the 30% budget, the lower bound of the interval for the recall gain over random > 0, and the lower bound of the AUROC interval > 0.5. The sensitivity analysis cannot replace or rescue a failed primary test.

## Error-type counts

Regions with `severe_error=true` whose `tags` contain `reference` or `alignment` are deduplicated, and the count of each type, their intersection and their union are reported. These counts describe the failure types and are not used as a new risk signal; the later group analysis of risk rankings is listed separately as a [post hoc analysis after the results were revealed](POSTHOC_EN.md).

## Saved data

- [Summary results](v1/summary.json): `sensitivity` contains the exclusions, retained counts, full metrics and intervals; `failure_pool` stores the error-type counts.
- [Original ratings](v1/ratings_original.json) and the [page-draw sequence](v1/bootstrap.json).
- [Risk scores](automatic/risk_scores.json) and [per-region translations](automatic/cases.csv).

The private mapping and research source code that the original analysis relied on are not included in the reduced package; this page keeps the methods and saved results, and provides no reproduction commands that depend on the missing files.
