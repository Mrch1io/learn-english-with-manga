# Methods for the Local Manga Translation Comparison Experiment

**Language / 语言**: English | [简体中文](PROTOCOL.md)

This file outlines the four-condition `openmantra-v1` experiment. The configuration, test pages and human-review sample are in the [freeze record](freezes/openmantra_v1_test.json), and the results are in the [summary](SUMMARY_EN.md). The research goal is to separate, with the text boxes held fixed, the effects of OCR input, nearby context and the choice of locally deployed model on the translations.

## Samples and controls

The data comes from OpenMantra, with development and test sets split by work. The first N consecutive pages of designated works are used; this is not a random sample of manga in general. Test pages were chosen by metadata, and pages were not swapped based on model performance.

| Purpose | Work and pages | Regions |
|---|---|---:|
| Development and debugging | tencho_isoro, p0001–p0012; the first 3 pages were used for initial run checks | 96 |
| Official test | tojime_no_siora, p0001–p0024 | 224 |

Configurations and samples were frozen before the official test; the reference English never enters the model input. Changes made after seeing test results are listed separately as exploratory; they do not overwrite the original results and are not called independent confirmation.

| Condition ID | Model | Current-region Japanese | Nearby Japanese budget |
|---|---|---|---:|
| vntl-gold-c0 | VNTL Llama3 8B, Q4_K_M | Official Japanese | 0 |
| vntl-ocr-c0 | Same VNTL | Manga-OCR output from the same official text box | 0 |
| vntl-gold-c1200 | Same VNTL | Official Japanese | 1,200 characters |
| qwen-gold-c0 | Qwen2.5-3B-Instruct, no LoRA | Official Japanese | 0 |

Shared settings: region-by-region translation, at most 256 output tokens and at most one retry on failure; continuation merging, English history, and term extraction and injection are turned off, and an independent empty database is used. VNTL uses its native prompt format, temperature=0, repeat_penalty=1 and a 2,048-token context; Qwen uses its native chat template and greedy decoding, with a limit of 8,192 tokens. Parameter count, training, quantization and prompt format differ, so comparisons between models only represent these deployment configurations.

`gold` means the official Japanese text and text boxes; reading order uses annotation indices that were not manually verified (`annotation_index_unverified`). The OCR condition does not fill in recognition failures with the correct Japanese. All conditions share the text boxes and the ordering proxy, so they cannot be used to evaluate end-to-end quality after real detection, reading order, text removal and typesetting.

## Metrics and statistics

The two primary contrasts are `vntl-ocr-c0 − vntl-gold-c0` (effect of OCR input) and `vntl-gold-c1200 − vntl-gold-c0` (effect of context); `vntl-gold-c0 − qwen-gold-c0` is a secondary deployment reference.

The translation metric is corpus chrF++, computed with SacreBLEU 2.6.0 (char_order=6, word_order=2, beta=2). Scores are computed per work and then macro-averaged; each split here has only one work, so in practice it is that work's corpus score. Regions whose reference Japanese or English is empty in the original are not scored; failed translations are kept as empty translations and counted in the denominator. In the test, Qwen's output for `p0013:t003` contained untranslated Japanese both times and was scored as a failure.

The page-paired bootstrap within each work was run 2,000 times (seed=20261017); each iteration draws the same pages for both conditions and re-aggregates the corpus counts. Candidate−baseline differences are reported with 95% percentile intervals.

For OCR, both strict NFC CER and a CER computed after NFC with only full-width ASCII converted to half-width and Unicode whitespace removed are reported; other punctuation and kana are kept. The two measures reflect formatting differences, and the lower normalized CER is not treated as a model improvement. The differences from input substitution cannot be added up and interpreted as each stage's share of the errors.

## Human review and detection audit

The human review preselected 48 regions: with seed=20261017, at most two regions with non-empty Japanese and English were drawn from each test page; all four systems were rated on the same set of IDs, with the original image and the necessary Japanese context provided. Ratings cover fidelity 0–4, naturalness 1–3, severe error and error type; missing ratings are not fabricated, and each comparison reports its own number of valid pairs. The current archive has a single rater: 47 pairs each for VNTL−Qwen and OCR−official Japanese, and 46 pairs for context−no context; human-review intervals use a page-paired bootstrap with 2,000 iterations, seed=20261018. See the [original ratings](results/R1/source/ratings_1.json) and the [analysis](results/R1/analysis/summary.json).

Detection was also audited on the first 6 pages each of the development and test sets, using the geometric criteria of text coverage ≥90% and area ratio ≤4. Official text boxes are not the same as bubble boundaries, so bubble mAP is not reported; this audit does not replace a translation-quality evaluation of the full pipeline.

## Limits of the conclusions and evidence

chrF++ measures character and word overlap; on its own it cannot judge whether meaning, character relationships or names are correct. The 95% intervals are not corrected for multiple comparisons; consecutive pages and scene correlations are not modeled by page resampling, so the intervals may be too narrow. Single-rater scoring has no inter-rater agreement estimate. Results on a few works do not support significance or accuracy guarantees for manga in general.

This directory keeps the configuration and sampling records, predictions, scores and statistical results; the full original images, run logs and historical source-code snapshots are not all included, so re-checking the saved results is not the same as fully rerunning the original experiment.

- [Four-condition automatic metrics](results/test24-systems/comparison.json) and [per-item predictions](results/test24-systems/cases.csv)
- [OCR metrics](results/test24-label/scores.json) and [case-by-case comparison](results/test24-label/cases.csv)
