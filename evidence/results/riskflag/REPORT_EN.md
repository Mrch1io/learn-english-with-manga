# Risk-Flagging Confirmation Experiment: Results

**Language / 语言**: English | [简体中文](REPORT.md)

**The primary test failed (`failed`).** Among the 151 labeled regions of the new work rasetugari, checking the top 46 by S1 caught only 3 of the 18 severe errors; AUROC is 0.3419. The current evidence does not support using this disagreement signal for automatic risk flagging.

## Sample and methods

151 regions were presampled from 54 pages: 18 severe errors, 133 non-severe and 0 excluded as null; all 54 pages have labels. The main system is `vntl-gold-c0`, with a single reviewer; only these sampled regions take part in the human risk evaluation. The methods and primary thresholds are in the [summary of the preregistered rules](PROTOCOL_EN.md), and the ratings are in the [original file](v1/ratings_original.json).

In the tables below, proportions are percentages and differences are percentage points; AUROC is not multiplied by 100. The fixed budget is k=`ceil(N × fraction)`. The 95% intervals are percentile intervals from a page-paired bootstrap with 2,000 iterations, seed=2026102003; the saved primary and sensitivity analyses both have 2,000 valid and 0 undefined iterations. Full precision is in the [summary results](v1/summary.json), and the page-draw sequence is in [bootstrap.json](v1/bootstrap.json).

## Primary analysis and secondary signal

| Signal | Budget | Checked k | Caught TP | Recall % [95% CI] | Precision % [95% CI] | Recall gain over random, percentage points [95% CI] |
|---|---:|---:|---:|---|---|---|
| S1 | 20% | 31 | 2 | 11.1111 [0.0000, 30.0000] | 6.4516 [0.0000, 16.6667] | -9.4187 [-20.5128, 9.7451] |
| S1 | 30% | 46 | 3 | 16.6667 [0.0000, 33.3333] | 6.5217 [0.0000, 13.6364] | -13.7969 [-30.4069, 3.2680] |
| S1 | 50% | 76 | 6 | 33.3333 [11.1111, 54.5455] | 7.8947 [2.5000, 14.2857] | -16.9978 [-38.8974, 4.5455] |
| S2 | 20% | 31 | 2 | 11.1111 [0.0000, 30.7692] | 6.4516 [0.0000, 19.3548] | -9.4187 [-20.5128, 10.3778] |
| S2 | 30% | 46 | 4 | 22.2222 [0.0000, 44.4697] | 8.6957 [0.0000, 19.1489] | -8.2414 [-30.1282, 14.3907] |
| S2 | 50% | 76 | 9 | 50.0000 [29.4118, 71.4286] | 11.8421 [5.1948, 19.4805] | -0.3311 [-20.9197, 21.1141] |

| Signal | AUROC [95% CI] |
|---|---|
| S1 | 0.3419 [0.2063, 0.4958] |
| S2 | 0.4720 [0.3547, 0.6062] |
| Random ordering | 0.5000 |

The random control is an analytical expectation, not another sample. The 20%/30%/50% budgets check 31/46/76 regions, with expected catches of 3.6954/5.4834/9.0596 severe errors; the corresponding recall is 20.5298%/30.4636%/50.3311%, and the expected precision is 11.9205% in every case.

At the primary budget, S1 has 15 misses and 43 false alarms. The lower bound of the recall gain over random, −30.4069 percentage points, is not above 0, and the lower bound of AUROC, 0.2063, is not above 0.5, so both criteria failed. S2 and the other budgets are secondary analyses and cannot replace the primary decision; after seeing an AUROC below 0.5, reversing the ranking and claiming a successful confirmation is not allowed.

## Differences from the seen pilot

| Item | Seen pilot tojime_no_siora | New-work confirmation rasetugari |
|---|---:|---:|
| Sampled / labeled / severe errors | 48 / 46 / 9 | 151 / 151 / 18 |
| Pages | 24 | 54 |
| S1, 30% budget: checked / caught | 14 / 6 | 46 / 3 |
| Recall gain over random, percentage points [95% CI] | +36.2319 [8.8854, 68.0851] | −13.7969 [−30.4069, 3.2680] |
| S1 AUROC [95% CI] | 0.7538 [0.6264, 0.8824] | 0.3419 [0.2063, 0.4958] |

The pilot figures keep the original report's records to explain why a confirmation test on another work was run; this reduced package does not include the pilot's per-item risk records or page-draw sequence, so the pilot intervals cannot be independently recomputed from it. The pilot reached the exploratory threshold while the confirmation sample failed, and the pilot results cannot stand in for the confirmation conclusion.

## Exploratory sensitivity analysis

Following the [supplementary rules set before rating](SENSITIVITY_EN.md), the 9 sampled regions where any of the four systems produced an empty output were excluded, including 1 severe error and 0 nulls; N=142, P=17 and 125 negatives remain, and the original 54 pages and the same page-draw sequence are kept.

At the 30% budget, 43 regions are checked and 2 severe errors caught: recall 11.7647% [0.0000, 31.5789], precision 4.6512% [0.0000, 12.8261], recall gain over random −18.5170 percentage points [−30.5349, 1.2994]; AUROC is 0.3362 [0.1905, 0.4993]. The 20%/50% budgets catch 2/5 respectively, and all intervals are saved in the `sensitivity` field of the summary results.

After excluding empty outputs, the results still do not support the fixed S1 high-risk direction; this neither rescues the main hypothesis nor allows a causal mechanism to be inferred from the exclusion analysis.

## Error types and limitations

Of the 18 severe errors, 12 are `reference`, 4 are `alignment` and 1 is in both, for a union of 15. Their risk-ranking analysis is in the [exploratory post hoc analysis](POSTHOC_EN.md). Missed examples include `item100`: “しばらくは尻に敷かせてもらわんとな” was translated as “For a while, I’ll have to let you boss me around.”, which the reviewer marked as a severe reference error; for the false-alarm example `item032`, the reviewer marked the empty translation as an omission, but with severity false. The severe-error label and whether an output is empty are not the same concept.

The conclusions apply only to this work, these presampled regions, the fixed signal and the single reviewer's labels. At most 3 regions were drawn per page, which is not equal-probability sampling from all regions; page resampling does not include reviewer error or cross-page scene correlation. Only the main system went through this round of human risk testing, so the semantic correctness of the other three systems cannot be judged from it. Both the failure of the primary test and the limited post hoc explanation should be kept on record.
