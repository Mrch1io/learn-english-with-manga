# Exploratory Analysis After the Risk Signal Failed

**Language / 语言**: English | [简体中文](POSTHOC.md)

## Data and methods

The subject is `vntl-gold-c0` on rasetugari: 151 presampled regions, all with severity labels, 18 severe and 133 non-severe. They are grouped by the existing ratings into severe with `reference` or `alignment` (overlap counted once), other severe, and non-severe; no labels were re-adjudicated.

S1 risk is sorted from high to low, with ties broken by lexicographic `(page_id, block_id)` order; rank 1 has the highest risk, out of 151 ranks. Using the top 46 from the 30% budget, the median S1, the median risk rank and the number in the top 46 are reported for each group.

Japanese length is the Python `len` of the raw `source_text`: the number of Unicode code points, including punctuation, whitespace and line breaks, with no normalization. The Spearman coefficient is computed from the average ranks of S1 and length, with tied values given the average rank; this is distinct from the unique ordinal ranks used for the fixed budget above.

## Results

| Group | Regions | Median S1 | Median risk rank | Number in top 46 |
|---|---:|---:|---:|---:|
| Severe with reference or alignment | 15 | 0.206494 | 125 | 2 |
| Other severe | 3 | 0.331955 | 70 | 1 |
| Non-severe | 133 | 0.328540 | 72 | 43 |

Only 2 of the 15 reference/cross-bubble severe errors were caught; the expectation for random sampling at the same budget is `15 × 46 / 151 = 4.57`. Between S1 and Japanese length, **Spearman ρ=−0.145817**. Full precision and definitions are in the [computation record](provenance/posthoc.json), and the main results are in the [confirmation report](REPORT_EN.md).

## Interpretation and limitations

Among these rated regions, reference/cross-bubble severe errors usually sit at low risk ranks, which shows that the fixed strategy of checking high-disagreement regions first missed this kind of error. One possible explanation is that the systems share errors, but only the main system received human risk evaluation in this round; low disagreement by itself cannot prove that the other systems made the same semantic errors, nor can it confirm the failure mechanism.

This analysis is a post hoc description of a single work with a single reviewer, and no new inferential tests were constructed for the group differences or the correlation coefficient. It cannot be generalized to other manga, causally attributed to text length, or used to claim a validated new signal by reversing the ranking. The results should serve as a source of hypotheses for later independent experiments.
