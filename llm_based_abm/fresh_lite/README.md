# 独立新站点Lite实验

本目录是2026-10-18的新实验，不加载历史实验的回复、权重或预测。模型与时段配置保持不变，在四个新站点上重新执行训练、验证和测试。固定设定见[PROTOCOL.md](PROTOCOL.md)，完成结果见[RESULTS.md](RESULTS.md)。

统一入口是run.py；experiment.py和timing.py保留原方法函数作为模块，不直接运行其旧入口。

从本目录运行，Python 3.11，依赖使用上一级requirements.txt：

```bash
python -m unittest test_fresh.py
python reproduce_numeric.py
```

上面使用已公开的数值概率重新拟合规则/角色方法，并核对预测；直接LLM复用已保存的数值预测。无需密钥或原始日志，不消耗额度。这是数值复现，不独立核验原始API响应。`python run.py`才允许联网；密钥在上一级key.txt，不提交。所有新请求及失败重试合计硬限500万token，日志中的已有用量也计入，不能删日志重置预算。不要同时启动两个联网进程。

如需从共享原始数据重新聚合：先运行`python select_stations.py`、`python prepare.py`、`python run.py --preflight`。运行时会校验冻结输入，未经记录改变数据不能复用为原实验。

本轮七方法：星期均值、Ridge、规则ABM日汇总、规则ABM时段、Lite角色zero-shot时段、Lite角色few-shot时段、Lite直接few-shot时段。外层角色权重与星期系数全部由本轮数据重新拟合，未继承Pro小测试权重。每配置一次推断，站点共享画像回复；不宣称已检验重复随机性。

数据与输出都在res：

| 文件 | 内容 |
|---|---|
| selected_stations.csv / dates.csv | 固定站点与逐日划分 |
| panel.csv | 本轮站点日真实客流、票种、天气、划分 |
| predictions.csv | 方法×站点×测试日真实值和预测 |
| summary.csv | Normal、Extreme、去掉9/29后的平均WAPE |
| metrics.csv / event_metrics.csv | 整测试窗口 / 事件当天误差 |
| decisions.csv / populations.csv | 角色概率和理由 / 本轮拟合规模与星期系数 |
| validation.csv | 只按验证集选择的原参数候选 |
| runtime.csv / budget.json | 阶段耗时、新调用token和总成本 |
| api_calls.jsonl.gz（仅本地） | 原始请求、回复及调用元数据，不上传GitHub |
| protocol.json / audit.json | 冻结哈希与核验结果 |

Normal平均是暑假、学期任务等权；Extreme平均仅取事件当天，不混入周边普通日。9/29的86 St少一小时，所有方法该日统一不评分；没有根据测试分数换站。画像、出发时段和暴露函数是原假设；旧时间段曾被探索过，不是新的时间盲测。

本轮曾发生连接中断，网络恢复后按相同请求继续；所有失败记录和保守预算保留在日志中，有效回复不重抽。联网运行会从本轮已完成方法的检查点继续。该工程性修复记录在protocol.json中，未改模型或提示。runtime.csv只统计完成方法的阶段耗时，中断阶段的额外等待不在其中；API原始日志保留每次请求的耗时。

离线复现得到的是缓存回放耗时，不能当首次联网耗时。累计token以压缩API日志和budget.json为准。

最终核验修复了重试缓存优先级：每个逻辑请求统一使用最早有效回复，包括非首次attempt的有效回复。18条额外有效回复留在日志并计入费用，不参与分析；全部拟合已离线重做。离线重跑耗时写入replay_runtime.csv，原始联网阶段runtime.csv保留。

公开文件中的reason是占位说明，不是LLM原话。根据用户选择，原始提示/回复日志和服务调用元数据仅保留本地。若本地有原始日志，可用run.py --offline及report.py进行完整溯源核验；GitHub接收方请用reproduce_numeric.py。audit.json记录本地已完成的核验，并不表示公开仓库包含原始证据。
