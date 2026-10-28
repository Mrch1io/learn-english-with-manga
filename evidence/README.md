# 研究证据索引

本目录保存翻译实验与工程验证的预测、评分和统计结果。先读[研究摘要](SUMMARY.md)；本地四条件实验设计见[方法说明](PROTOCOL.md)。完整原始图片、运行日志和历史源码未全部随包提供。

| 研究结果 | 证据 |
|---|---|
| tojime：VNTL−Qwen chrF++ +2.497 [0.707, 4.503]；OCR 输入与上下文消融 | [统计结果](results/test24-systems/comparison.json)、[逐条预测](results/test24-systems/cases.csv) |
| rasetugari：VNTL−Qwen +5.768 [3.873, 7.555]；失败 3/358 和 11/358 | [统计结果](results/riskflag/automatic/comparison.json)、[逐条预测](results/riskflag/automatic/cases.csv) |
| tojime 人评：严重错误率差 −14.89 [−28.26, −2.08] 个百分点 | [分析](results/R1/analysis/summary.json)、[评分原件](results/R1/source/ratings_1.json) |
| OCR：严格 CER 10.24%、归一后 1.89%；78 个区域仅格式不同 | [指标](results/test24-label/scores.json)、[逐条对照](results/test24-label/cases.csv) |
| 风险提醒：151 区域、检查前 46 个；AUROC 0.342，未通过 | [协议](results/riskflag/PROTOCOL.md)、[分析补充规则](results/riskflag/SENSITIVITY.md)、[报告](results/riskflag/REPORT.md)、[统计结果](results/riskflag/v1/summary.json) |
| 事后分析：15 个指代／跨泡错误，排名中位数 125；前 46 个仅抓到 2 个 | [分析说明](results/riskflag/POSTHOC.md)、[计算结果](results/riskflag/provenance/posthoc.json) |
| API 探索：纯文本严重错误 2/24、VNTL 14/24；加图未见额外改善 | [报告](results/multimodal-api-qwen38/replacement-v1/REPORT.md)、[统计结果](results/multimodal-api-qwen38/replacement-v1/analysis-v1/results.json) |
| 冻结配置与抽样 | [本地四条件](freezes/openmantra_v1_test.json)、[风险提醒](freezes/riskflag-confirm.json) |
| 候选扫描：20／200／2,000 框为 0.119／1.117／15.188 ms | [扫描微基准](validation/fragment-bench.json)、[可运行扫描脚本](../tools/benchmark_fragments.py)、[词边界微基准](validation/boundary-bench.json) |
| 锁定译名：14 项原生语法判定；真实生成跟进 | [语法结果](validation/vntl-grammar.json)、[验证报告](validation/VNTL.md) |

数据署名与许可见 [LICENSE-OpenMantra.md](LICENSE-OpenMantra.md)。
