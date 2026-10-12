# NYC Subway ABM

大作业：比较纽约地铁天气变化下的规则模型与 LLM 行为模型。

- data/：两组共享的真实站点客流、票种、坐标和 JFK 天气。
- rule_based_abm/：协作者的规则模型工作区。
- llm_based_abm/：三个独立版本；[版本入口](llm_based_abm/README.md)。

当前画像式个人决策实验：[V3说明](llm_based_abm/V3/README.md)、[流程](llm_based_abm/V3/PROTOCOL.md)、[结果](llm_based_abm/V3/RESULTS.md)。

历史固定策略实验：[V1](llm_based_abm/V1/README.md)。动态群体预测：[V2](llm_based_abm/V2/README.md)。各版本分别保存真实 API 响应和结果，不混用指标。

LLM 组使用 llm 分支，阶段完成后直接合入 main 同步小组；密钥不上传。
