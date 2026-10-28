# Exploratory Paired Experiment on Image Input for qwen3.8-max

**Language / 语言**: English | [简体中文](REPORT.md)

**No additional improvement from images was observed this time.** The text-only arm had 2/24 severe errors and the text-plus-image arm 3/24; the main comparison did not meet the predefined go-ahead criteria (`not_met`). The text-only arm's improvement over the old VNTL is a descriptive comparison of deployment setups and cannot be attributed to images.

## Research question and design

Research question: with the same API model and text context, does adding the whole-page image reduce severe translation errors?

- **A: text only**, given the target region's Japanese, the target ID and the whole-page Japanese context.
- **B: text + image**, the same text conditions as A, plus a whole-page image with the target region marked.
- **VNTL ctx0: historical baseline**, reusing the old translations of the same regions and rescoring them in this round; its model, prompt and context setup differ from A/B.

Both A and B use `qwen3.8-max`, `max_completion_tokens=256`, `enable_thinking=false`. English reference translations and human labels never enter the model input. The sample is 24 seen regions from 23 pages, selected by old labels as 12 severe errors and 12 normal controls. In this round's rescoring VNTL has 14 severe errors, but the original 12/12 grouping is unchanged. The sample is deliberately enriched for known errors, so the proportions below are not overall error rates for the works or the product.

## Sample replacement and exploratory scope

The first round stopped after 18 complete A/B pairs had been obtained, before any human review or effect analysis, because the image request for `tojime_no_siora:p0010:t010` was blocked by the provider. That region was one of the originally mandatory cases; its whole-page image, Japanese and target ID were then replaced together. **The original fixed sample list was not completed, and this report gives exploratory results after the sample was changed.**

The replacement rule was fixed before selection and does not use this round's model outputs or new ratings: same work and same old severe-error label as the original region, matching the original `reference/alignment` priority; neither the region nor its page is on the original list; the old VNTL translation, the Japanese and the English reference are all valid and non-empty. The original region's error-type `tags` are empty and its severe-error label is `true`; under the original rule its priority is recorded as false, which is not taken to mean that it has no reference or cross-bubble error. Eligible items were sorted by region ID and chosen with seed=2026102104; there was in fact only 1 eligible item, and the English reference was used only for the eligibility check.

The other 23 regions, the condition parameters and the A/B order are unchanged. The original 36 valid responses were reused directly, and 12 responses were added for the replacement region and the remaining unfinished regions, giving the current 48 valid API candidates. The originally blocked sample is not counted in the current 24 regions and is not recorded as a success; the replacement cannot remove the selection bias caused by the provider's filtering.

## Scoring and statistics

All 72 candidates across the three arms were rated by one reviewer. The four judgments (severe error, character/action reference, cross-bubble correspondence and action direction) have no missing values; there is no second reviewer or inter-rater agreement estimate.

The main comparison is fixed as B−A. A bootstrap paired by actual page ID is run 2,000 times (seed=2026102102), keeping the regions within each page and the pairing, with 95% intervals formed from linearly interpolated 2.5/97.5 percentiles. Differences are in percentage points; positive values mean the candidate arm has more errors. The statistical rules require valid responses and non-missing relevant labels; currently all candidates are valid, all 24 pairs are complete, and no region was excluded for failures or missing ratings. The intervals do not account for reviewer error and cannot remove the selection bias.

## Main results

| Arm | Severe errors | Severe-error rate | 95% interval | Character/action reference errors |
|---|---:|---:|---:|---:|
| A: text only | 2/24 | 8.33% | [0.00, 21.74]% | 2/24 |
| B: text + image | 3/24 | 12.50% | [0.00, 26.09]% | 3/24 |
| Old VNTL ctx0 | 14/24 | 58.33% | [40.00, 76.02]% | 12/24 |

**Main comparison B−A:** the severe-error rate difference is **+4.17 percentage points [0.00, 13.04]**, with 0 fixed, 1 regressed and a net reduction of −1; 2 severe errors are shared by both arms. Among the fixed normal controls, A has 0/12 and B 1/12 severe errors. The interval includes zero, so it cannot be claimed that images reliably increase severe errors.

The predefined go-ahead criteria require at least 20/24 complete pairs, all 12 normal controls complete, a net reduction of at least 3 severe errors for B, and no new severe errors in the normal controls. The completeness conditions are met but the last two are not, so the decision is `not_met`.

**A−VNTL descriptive comparison:** the severe-error rate difference is **−50.00 percentage points [−69.23, −30.43]**, with 12 fixed and 0 regressed. The character/action reference error rate difference is **−41.67 percentage points [−64.00, −17.39]**: of VNTL's 12 reference errors, 11 were fixed while 1 new one appeared, a net reduction of 10. The model, prompt and context all differ, so the improvement cannot be attributed to model size or to any single input change alone.

All three arms have 0 cross-bubble correspondence errors, so this round cannot evaluate repairs of that kind; zero events do not prove that the true risk is zero. All three arms have 1 action-direction error each, and A/B did not fix the same error; relative to VNTL, each has one fix and one regression, so equal totals do not mean the errors are identical.

## Cases

The table below shows one VNTL→A improvement, the only A→B severe regression, and both severe errors shared by A and B. The judgments come from this round's review; full translations and notes are kept in the per-item records.

| Region | Observation |
|---|---|
| `rasetugari:p0005:t003` | 「家族を悪鬼に喰われて以来」: VNTL translated it as “my family” and A changed it to “his family”, correcting the kinship reference in the narration. |
| `rasetugari:p0007:t001` | 「さま...」: A keeps “…sama…” while B translates it as “Lord…”; the reviewer judged that B misread the cross-bubble form of address calling out to an older brother. This is the only A→B severe regression, and it was also one of the original normal controls. |
| `rasetugari:p0026:t003` | 「幻術とはいえ妹の顔を...」: both A and B use “my sister's face”; the reviewer judged that this wrongly attributes the boy's sister to the current speaker. |
| `tojime_no_siora:p0023:t006` | 「つっつけようったって...」: neither A nor B correctly recovered the relationship between the pursuer and the pursued; this is a shared severe error. |

## Conclusions and data

The current sample does not support an additional benefit from adding images with the same API model. The text-only arm's improvement over the old VNTL is worth further validation, but the sample has already been seen, is enriched for errors, had one replacement and has only one reviewer, so the conclusions cannot be generalized to overall translation quality, nor do they show that images are ineffective for other tasks.

- [Analysis results and full intervals](analysis-v1/results.json)
- [Original ratings and source tags](analysis-v1/ratings_1_reviewed.json)
- [Per-item translations, labels and reviewer notes](analysis-v1/joined_rows.json)
