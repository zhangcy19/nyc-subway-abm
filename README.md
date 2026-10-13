# NYC Subway ABM

大作业：比较纽约地铁天气变化下的规则模型与 LLM 行为模型。

- data/：两组共享的 2023 年 7–10 月站点客流、票种、坐标和 JFK 天气；[数据说明与附件取舍](data/README.md)。
- rule_based_abm/：协作者的规则模型工作区。
- llm_based_abm/：四个独立版本；[版本入口](llm_based_abm/README.md)。

当前多月份、三次强降雨的 Lite 对比实验：[V4说明](llm_based_abm/V4/README.md)、[流程](llm_based_abm/V4/PROTOCOL.md)、[结果](llm_based_abm/V4/RESULTS.md)。包含直接预测和画像式个人决策，各做 zero-shot / few-shot。

历史固定策略实验：[V1](llm_based_abm/V1/README.md)。动态群体预测：[V2](llm_based_abm/V2/README.md)。早期个人决策实验：[V3](llm_based_abm/V3/README.md)。各版本分别保存真实 API 响应和结果，不混用指标。

LLM 组使用 llm 分支，阶段完成后直接合入 main 同步小组；密钥不上传。

V1–V3 的已发布实验仍固定使用 2023 年 8–9 月。共享数据扩展不代表这些实验已重新训练或评估。
