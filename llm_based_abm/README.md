# LLM-based NYC subway cohort ABM

用相同站点需求基准比较天气规则、LLM整体判断和LLM群体代理。数据为2023年8–9月，12站，最终检验为9月22–30日。完整设计见 [PROTOCOL.md](PROTOCOL.md)，文献依据见 [LITERATURE.md](LITERATURE.md)，数据口径见 [DATA_DESIGN.md](DATA_DESIGN.md)。

## 运行

使用dev环境，终端进入 `llm_based_abm` 文件夹后运行：

```bash
conda activate dev
python -m pip install -r requirements.txt
python prepare_experiment.py
python -m unittest test_model.py
python run_experiment.py --pilot
python run_experiment.py
python analyze.py --extended
python report_results.py
```

Pro API密钥放本文件夹 `key.txt`，仅保存纯密钥；该文件被git忽略。API基址沿用已有测试代码。运行器逐次记录真实API使用量，总上限200,000 token，并预留输出额度。重复请求按完整内容哈希复用实际响应；请求改变会重新调用，不保证API采样逐字复现。

已有完整策略时，可不联网、不消费token重放：

```bash
python run_experiment.py --offline
python analyze.py --extended
python report_results.py
```

`--pilot`只运行zero-shot、5-agent、一次重复，用于检查接口。正式主实验为zero/few ×5/10/20代理 ×3次独立策略重复，并比较两种基准、拟合规则和LLM direct。原始响应和最终策略都保留，便于追踪概率来源。正式结果覆盖试跑预测，API记账仍包含试跑费用。

## 主要文件

| 文件 | 用途 |
|---|---|
| `prepare_experiment.py` | 按时间划分，选站、构建需求基准与群体分配 |
| `model.py` | Mesa代理、概率验证、逐小时推迟需求状态 |
| `run_experiment.py` | 规则拟合、Pro策略请求、缓存、实验与敏感性 |
| `analyze.py` | 指标表与科研图 |
| `experiment_data/` | 小型试验输入和基准参数 |
| `results/predictions.csv.gz` | 留出期各模型、配置、重复的逐组小时预测 |
| `results/api_calls.jsonl` | 请求、原始响应及API token使用量，不含密钥 |
| `results/policies.json` | 通过验证后实际用于Mesa的概率策略 |
| `results/metrics.csv` | 总体与各票种组的误差 |
| `results/figures/` | 配置比较、暴雨过程、群体、空间与代理数图 |

详细结果见最终 `RESULTS.md`。统计中的seed是重复样本编号，不是可保证重现LLM采样的服务端种子。

## 与规则模型组对接

这里的rule_based是为本实验准备的内部规则对照，协作者的 `../rule_based_abm/` 由其继续开发。两组共享 `../data/` 源数据和相同时间划分，模型预测可按同一结果列对比：

`timestamp, station_complex_id, group, observed_ridership, predicted_ridership, baseline_demand, total_observed, model, shot, n_agents, seed`

四组为 `students, fair_fare, seniors_disability, general`。`total_observed`是独立站点总客流，同小时每组重复存储；总评分取其一次，不能将它按组累加。原始票种缺失不补成真实零观测，群体指标排除缺失；仅群体基准质量分配使用零贡献约定。天气等附加列可保留供分层分析。

## 解释范围

这是已知实测天气条件下的历史预测，不是提前天气预报或真实个体轨迹预测。agent是加权群体代表，其灵活性是假设；没有真实人格、OD、换站或地铁线路容量。行动概率无法用客流数据单独验证为真实取消/改方式比例。老人和残障在源标签合并，无法分别验证。

9月29日只有一个确认极端事件，12站也不代表全纽约；天气与客流关系不等于降雨因果效应。推迟返回率是情境参数，结果中另有敏感性分析。此代码与草稿用于小组实验工作，课程提交需由组员理解、核查并按课程要求声明GAI使用。
