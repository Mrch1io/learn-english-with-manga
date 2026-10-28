# Risk-Signal Confirmation Experiment: Methods Summary

**Language / 语言**: English | [简体中文](PROTOCOL.md)

This file summarizes the original preregistered rules; the machine-readable settings and sampled IDs are in the [freeze record](../../freezes/riskflag-confirm.json). The research question is: can disagreement among four translation systems find the main system's severe translation errors within a limited review budget?

## Systems, sample and signals

The main system is `vntl-gold-c0`; the other three conditions are `vntl-ocr-c0`, `vntl-gold-c1200` and `qwen-gold-c0`. Inputs and deployment settings for the four conditions are in the [shared experimental methods](../../PROTOCOL_EN.md). S1 measures disagreement as the average chrF++ difference between the main system's translation and the translations of the other three conditions. This round keeps the frozen S1 and the secondary signal S2, and does not change the signals, the risk direction or the budget based on human results; the [risk-score archive](automatic/risk_scores.json) records each region's scores and declares that the signal computation used no human labels or reference translations.

The confirmation sample is the 54 pages of rasetugari, a different work from the seen pilot tojime_no_siora. The freeze record keeps the dataset's original `train` label; "confirmation" here means testing on another work chosen by this project, and does not mean proving that the work was absent from upstream models' training data.

Sampling was completed before any model output: regions are sorted by `page_id` and `block_id`, with the only requirement that the Japanese and the English reference are non-empty; up to 3 regions are drawn uniformly without replacement from each page, seed=2026102001. Short pages are not topped up from other pages, and samples are not chosen by difficulty, model output, risk or automatic metrics. 151 regions were actually drawn; the four systems ran on all 54 pages, and unsampled regions covered by the risk archive do not enter the denominator of the human evaluation.

## Human review

The predefined presentation shows one main-system translation per item, shuffled with seed=2026102002 and given new anonymous IDs, hiding the system name, other candidates, the English reference, risk scores and automatic metrics. Severe errors are recorded as true/false/null; error types use labels such as `reference` and `alignment`. Fidelity, naturalness and self-reported confidence do not determine inclusion in the main analysis.

There is only one reviewer, with no second reviewer or repeated-rating agreement estimate. The [original ratings](v1/ratings_original.json) mark the reviewer as R1, `reviewer_kind=human`. This reduced package does not include the original blind-review pages or the private mapping, so these files alone cannot independently verify the full presentation and unblinding process.

## Primary endpoint and decision

The single primary test is fixed as **`vntl-gold-c0`, S1, 30% review budget**. Passing requires both:

- The lower bound of the 95% interval for the recall gain over random > 0.
- The lower bound of the 95% AUROC interval > 0.5.

If both hold, the result is `passed`; if not, it is `failed`; if a required interval is undefined, it is `failed_insufficient_evidence`. S2, the 20%/50% budgets, and misses and false alarms are secondary analyses and cannot replace the primary decision. The other three systems are not tested for risk against human labels. Even a pass would only support testing proofreading efficiency separately; it would not prove that users correct translations faster.

## Statistical rules

`severe_error=true` is positive, false is negative, and null is excluded and reported; unsampled or unrated regions are not treated as correct. N is the number of labeled regions and P the number of severe errors. Regions are sorted by descending risk, with ties broken by lexicographic `(page_id, block_id)` order; the budget is k=`ceil(N × fraction)`, with fraction 0.20/0.30/0.50. The reported values are the number checked k, the number caught TP, recall TP/P and precision TP/k.

AUROC is the pairwise probability that a positive has higher risk than a negative, with ties counted as 0.5. Random ordering uses the analytical expectation: TP=kP/N, recall=k/N, precision=P/N, AUROC=0.5. Recall is undefined when there are no positives; AUROC is undefined when either class is missing.

A page-paired bootstrap over the 54 originally sampled pages is run 2,000 times, seed=2026102003; each iteration draws the same number of pages with replacement and keeps all non-missing labels within each page. All signals, budgets and random controls share the page-draw sequence, and N, k and the ranking are recomputed each time. The 95% interval uses linearly interpolated 2.5/97.5 percentiles, and the numbers of valid and undefined iterations are reported. Page resampling does not include reviewer error and does not model cross-page scene correlation.

Exploratory exclusion rules added before rating are in the [supplementary methods](SENSITIVITY_EN.md), and the actual results are in the [report](REPORT_EN.md). This package keeps the scores, labels, summaries and page-draw sequence, but not the original research source code or the full input archive, so it cannot be claimed that the original experiment can be fully rerun from it.
