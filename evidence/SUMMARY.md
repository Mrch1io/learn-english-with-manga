# 实验证据摘要

本页汇总已保存的实验结果，数据入口见[证据索引](README.md)。

本地四条件比较使用 OpenMantra，按作品划分开发集与测试集，正式运行前冻结配置；参考译文不进入提示词，翻译失败按空译文计分。以下区间均为按页配对 bootstrap（2,000 次）的 95% 区间；各实验的抽样与缺失规则见对应报告。

## 1. 模型选择：VNTL 8B（Q4_K_M）对 Qwen2.5-3B，chrF++

| 作品 | 区域 | VNTL | + OCR 输入 | + 上下文 1200 字 | Qwen | 失败 VNTL / Qwen |
|---|---:|---:|---:|---:|---:|---:|
| tojime（测试作品，前 24 页） | 224 | 30.463 | 30.816 | 31.409 | 27.966 | 0 / 1 |
| rasetugari（新作品，54 页） | 358 | 29.773 | 29.298 | 29.603 | 24.005 | 3 / 11 |

| 差值 | tojime | rasetugari |
|---|---|---|
| VNTL − Qwen | **+2.497 [0.707, 4.503]** | **+5.768 [3.873, 7.555]** |
| OCR 输入 − 官方日文 | +0.353 [−0.472, 1.321] | −0.475 [−1.841, 0.477] |
| 上下文 1200 字 − 无上下文 | +0.946 [−0.642, 2.461] | −0.170 [−1.206, 0.811] |

tojime 人评，严重错误率差（百分点）：VNTL − Qwen **−14.89 [−28.26, −2.08]**（47 对）；上下文 − 无上下文 −10.87 [−19.57, −4.17]（46 对）；OCR − 官方日文 −2.13 [−6.52, 0.00]（47 对）。两模型的规模、训练、量化和提示不同，结果仅支持当前部署组合的比较。实验使用官方文字框；少量作品与单人评审不能代表总体或端到端质量。

来源：[tojime](results/test24-systems/comparison.json) · [rasetugari](results/riskflag/automatic/comparison.json) · [人评](results/R1/analysis/summary.json)

## 2. OCR：Manga-OCR，tojime 224 个区域、2,598 字

严格 NFC CER **10.24%**（266 次字符编辑），在 NFC 后仅将全角 ASCII 转半角、去除 Unicode 空白后 **1.89%**（49 次字符编辑），78 个区域只差格式；其他标点与假名保留。p0001:t000「シオラ」被识别成「シボラ」，译名变成 `Shibo-ra`。

来源：[scores.json](results/test24-label/scores.json) · [逐条对照](results/test24-label/cases.csv)

## 3. 风险提醒（四系统分歧，30% 检查预算）：未通过，已放弃

事先登记的标准：召回率比随机高出部分的区间下限 > 0，且 AUROC 区间下限 > 0.5。

| | 已见试点 tojime | 新作品 rasetugari |
|---|---|---|
| 有标签区域 / 严重错误 | 46 / 9 | 151 / 18 |
| 抓到严重错误 / 检查数 | 6 / 14 | 3 / 46 |
| 召回率比随机高出（百分点） | +36.23 [+8.89, +68.09] | −13.80 [−30.41, +3.27] |
| AUROC | 0.754 [0.626, 0.882] | **0.342 [0.206, 0.496]** |

事后分析：18 个严重错误中有 15 个属于人物指代或跨气泡对应，排名中位数是 151 名中的第 125 名，前 46 名里只有 2 个，随机期望约 4.6 个。该分组在看到结果后分析，属于解释线索，不能视为独立确认的错误机制。

来源：[协议](results/riskflag/PROTOCOL.md) · [报告](results/riskflag/REPORT.md) · [summary.json](results/riskflag/v1/summary.json) · [事后分析](results/riskflag/POSTHOC.md)

## 4. API 对照

24 个已见区域按旧标签选出 12 个严重错误和 12 个正常对照（23 页），比例不是总体错误率。qwen3.8-max 纯文本（给整页日文）：严重错误 **2/24**，本地 VNTL **14/24**，差 −50.00 [−69.23, −30.43] 个百分点；人物／动作指代 2/24 对 12/24，VNTL 的 12 个修好 11 个、新增 1 个。主比较（加图片 − 纯文本）为 3/24 对 2/24，未达到预定推进条件。实验因服务商拦截替换过一个样本，属于探索性结果；模型与上下文差异限制了 API−VNTL 比较的归因。

来源：[报告](results/multimodal-api-qwen38/replacement-v1/REPORT.md) · [results.json](results/multimodal-api-qwen38/replacement-v1/analysis-v1/results.json)

## 5. 工程验证

无模型 CPU 微基准：关联候选扫描，20 / 200 / 2,000 个框为 0.119 / 1.117 / 15.188 ms（预热后 5 次取中位数）；2,000 个框加词内逗号判断首次 56.1 ms，预热后 18.1 ms，使用六组重复词例，缓存收益依赖输入重复性。仅加载词表的 llama.cpp 原生语法采样器 14 项判定全部符合预期，说明约束实现通过这些用例，不保证翻译语义正确。

另行真实生成的锁定译名样本为 `Misaki Takanashi, where is that key?`、`Misaki Takanashi, go to Silver Gate first.`。微基准不是整页推理耗时，少量生成样本不能证明任意术语的正确率。

来源：[扫描](validation/fragment-bench.json) · [词内逗号](validation/boundary-bench.json) · [语法](validation/vntl-grammar.json) · [跟进记录](validation/VNTL.md)

