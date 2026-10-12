# NYC Subway ABM

大作业：比较极端天气下纽约地铁出行的Rule-based ABM与LLM-based ABM。

- `data/`：两组共享的真实站点客流、票种、坐标和JFK天气。
- `rule_based_abm/`：协作者的规则模型工作区。
- `llm_based_abm/`：已实现Mesa群体代理、Pro策略调用、zero/few-shot、5/10/20代理、重复实验和图表。

LLM实验入口：[运行说明](llm_based_abm/README.md)。第一轮：[流程](llm_based_abm/PROTOCOL.md)、[结果](llm_based_abm/RESULTS.md)。第二轮：[诊断](llm_based_abm/V2_DIAGNOSIS.md)、[动态站点模型流程](llm_based_abm/V2_PROTOCOL.md)、[结果](llm_based_abm/V2_RESULTS.md)。

LLM组在 `llm` 分支开发，阶段完成后直接合入 `main` 与小组同步；PR可按需要使用。LLM API密钥不上传。
