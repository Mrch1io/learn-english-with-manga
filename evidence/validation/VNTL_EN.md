# Validation of VNTL Cross-Bubble Linking and Locked Names

**Language / 语言**: English | [简体中文](VNTL.md)

This record tests the integrity of cross-bubble continuations, locked names and single-bubble retranslation. This directory keeps the native grammar check and two CPU microbenchmark JSON files; the GPU translation, image pipeline and local-editing results below come from earlier experiment summaries, and the full samples and per-run raw responses are not included in this directory, so these files alone cannot independently verify all of the results.

## Methods and conditions

The [continuation and splitting implementation](../../translation/vntl.py) groups recognized adjacent continuations into temporary translation units, for example 「〜ちゃ／〜ては」 followed by 「ダメ／いけません」, or 「〜ても」 followed by 「いい」. Finished sentences, failed OCR and fragments confirmed as different speakers stay separate. After the model generates the linked whole sentence, it is split into consecutive fragments only at English word boundaries, keeping the word order, punctuation, full translated names and original bubble IDs; a locked name must sit in a bubble that contains the corresponding Japanese or alias, and if that can't be satisfied, it fails explicitly.

Single-bubble retranslation uses the full linked Japanese, with the unselected bubble's old translation as a fixed splitting constraint. Apart from whitespace normalization, the new and old fragments must reconstruct the model's whole sentence, which prevents repeated or dropped words; incompatible results are not committed. Prohibition sentences get an additional conservative-wording check, but it cannot prove that an arbitrary translation is semantically correct. This behavior is covered by the [translation-unit tests](../../tests/test_translation_units.py).

Locked names are normalized by exact Japanese span only in the model input, and the saved OCR stays as it is; the GBNF grammar constrains the corresponding target IDs, the output is checked again afterwards, and the generated English paragraph is never replaced. Ambiguous mappings are rejected; each translation unit allows at most four distinct locked names, 256 characters in total. See the [locked-term scope tests](../../tests/test_vntl_locked_scope.py).

The real-model experiment fixed the OCR and ran each of six sample groups twice, reusing the same model pool; candidate term extraction was turned off, and historical answers and review criteria were not fed to the model. In addition, the full pipeline was run twice in a row on the same 1280×720 image, and the left and right single bubbles were each retranslated from independent copies. The experiment scripts and full samples are kept locally by the author and are not published with the source code.

Environment: Windows 11, RTX 4070 Laptop 8GB, 16GB RAM, Python 3.12.12 / torch 2.10.0+cu126; VNTL Q4_K_M, llama.cpp b11706, 2048-token context, temperature=0, repeat_penalty=1. The recorded GPU offload is 33/33 layers, with 4403.49 MiB of CUDA weights, 256 MiB of KV cache and another 281.81 MiB of CPU_Mapped memory.

## Translation and editing results

Each round of the six sample groups covers 22 bubbles in total: 21 valid translations and 1 empty text; repeated outputs were identical, and the glossary contents did not change. These are run-state and repeatability results, not accuracy.

| Japanese | VNTL before the fix | This round's output |
| --- | --- | --- |
| 海賊版は読んじゃ / ダメですっ | You shouldn’t read pirated copies. / No, you can’t! | You absolutely cannot read / pirated copies! |
| みーちゃん、その鍵はどこ？ | Mii-chan, where’s that key? | Misaki Takanashi, where is that key? |
| 美咲、先に白銀門へ行って。 | Misaki, go ahead to Silver Gate. | Misaki Takanashi, go to Silver Gate first. |
| この薬は飲んじゃ / いけないんだ。 | You can’t drink this medicine. / It’s not allowed. | You can’t drink / this medicine. |

`/` marks a bubble boundary. Manual review found that both prohibition sentences kept their prohibitive tone without repeating the whole sentence, and that the name and its alias used the locked full name. The second bubble may be an English fragment and needs to be read together with the linked whole sentence. In the medicine sentence, `drink` is more literal than `take`; the source doesn't say what form the medicine takes.

There is a clear regression: `健太はもう帰った？` went from `Kenta already went home?` to `Kenta, did he already leave?`, mixing direct address with the third person, and needs proofreading. The six sample groups do not prove correctness for other images, character identities or arbitrary sentences with omissions.

The left and right single-bubble retranslations, and the retranslations after the two OCR edits, all passed the full concatenation check, and the unselected bubbles stayed unchanged. In the first round of local editing, `read` was repeated or dropped, and the old script's `checks=True` failed to catch it; after adding a check that "the saved split sentences reconstruct the model's whole sentence," the test was rerun. This failure shows that checking local fields is not enough to prove the integrity of cross-bubble text.

Trials with only the grammar constraint, before input normalization was added, kept generating until the token limit; these are failures and cannot be counted as successful translations. In the final review of the same image, the text of both bubbles sat inside them and was not marked as overflow; the text-removal algorithm was not changed in this round, so the existing text-removal results cannot be attributed to the translation fix.

## Grammar check and CPU microbenchmarks

All 14 accept/reject decisions in the [native grammar check JSON](vntl-grammar.json) matched expectations, covering full translated names, abbreviations, target bubbles, word boundaries, apostrophes, containment and four locked terms. Only the vocabulary was loaded, taking 0.281 seconds; no inference weights were loaded. Samples that contain Japanese along with the correct translated name are also accepted by the grammar and must still be rejected by the production validator, so **passing these 14 checks does not guarantee semantic or linguistic correctness of the translations**.

| CPU measurement scope | Result | Data |
| --- | --- | --- |
| Adjacent-fragment selection; 20 / 200 / 2000 blocks, 5 repetitions each | Median 0.119 / 1.117 / 15.188 ms; source fields unchanged | [fragment selection](fragment-bench.json) |
| Grouping and comma normalization; 2000 blocks, six kinds of word examples repeated | First run 56.059 ms, median of 5 warm runs 18.119 ms; source fields unchanged | [word-boundary normalization](boundary-bench.json) |

Both measure only CPU rule processing and include no model inference, OCR or translation quality. The word-boundary benchmark uses fugashi 1.5.2 / unidic-lite 1.0.8, with 9992 cache hits and 10 misses; warm-run results on repeated word examples cannot represent large amounts of varied text. The adjacent-fragment scan can be re-measured with [tools/benchmark_fragments.py](../../tools/benchmark_fragments.py), which keeps the original samples, warm-up and five-run timing method; the historical script for the word-boundary benchmark is not published with the source code. The JSON files above keep the measurements from that time, and re-measuring does not overwrite them.

## GPU timings and limitations

| Item | Measured summary |
| --- | ---: |
| VNTL cold load: full-image job / fixed-OCR job | 8.242 / 14.687 s |
| Warm translation of this image with fixed OCR | 0.556 s |
| Full image, first round / second round | 28.897 / 1.689 s |
| Average per page over two rounds (including the cold page) | 15.293 s |
| Total batch time divided by two pages (including disk writes and other overhead) | 17.038 s |
| Single-bubble retranslation: first item including load / next item with a warm model | 5.797 / 0.786 s |
| English reflow and save only | 0.209 s |

Both rounds used the same image; detection, OCR and VNTL were each loaded only once, and solid-color text removal did not load LaMa. So this does not represent cross-work throughput, and no general speedup factor can be derived from the difference between cold and warm pages. The difference in cold-load times shows that system state has a noticeable effect on the results.

For the fixed-OCR job, the whole-card peak sampled at roughly one-second intervals was 5366 MiB; it includes other applications and may miss momentary peaks. The PyTorch peak of 576 MiB reported for the image excludes the external llama.cpp and cannot be taken as whole-card usage.

Early VRAM contention once caused a long stall in OCR on the second page. The newly added stage-headroom check can release an idle VNTL in the model pool, but VRAM was free in the final measurement, so that branch was not triggered; unit tests are not enough to prove that real low-VRAM stalls have been eliminated.
