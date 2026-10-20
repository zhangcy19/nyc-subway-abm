# Normal 校准人口、按事件留一测试

当前为本地实验，不提交或推送 GitHub。使用 dev Python、Doubao Lite，本轮硬上限 500 万 token。API 密钥沿用 `../key.txt`，原始调用账本仅在本机 `res/api_calls.jsonl`。

## 固定设计

- 站点：Parsons Blvd（256）、176 St（385）；统一使用原 JFK 小时天气，不使用空间插值天气。
- Normal：保留原暑期/非暑期训练、验证、测试日期。各方法在 Normal 校准一次代表人数和星期活动系数。
- Extreme：7/29、8/13、9/29、10/7四个事件，均含前后两天。留出事件的所有站点、全部五天客流均不可用于该折示例或拟合。允许其余事件在时间上晚于目标，这是事后事件泛化实验。
- Normal 全部日期与四个事件窗口互不重叠。Normal 的测试标签不用于任何事件训练或历史参照。
- Extreme 人口参数严格复用对应季节 Normal 参数，不做人口再拟合或根据事件结果调整。

## 方法

1. 星期均值：对应季节 Normal 训练均值。
2. Ridge：Normal 用原训练/验证选正则化；Extreme 使用固定的 Normal 正则化，在对应季节 Normal 训练集加其他三个事件完整窗口上拟合回归系数。它不含人口参数。
3. 规则 ABM（日汇总/时段）：Normal 拟合人口和响应系数；Extreme 固定人口，仅在其他三个事件中心日上选择响应系数，按三个事件的WAPE等权选优。参考事件也使用各自季节已经固定的Normal人口。
4. 角色 zero-shot：Normal 拟合人口；Extreme 使用相同画像、天气输入，不提供历史客流，人口固定。
5. 角色 few-shot：Normal 仍使用最多四条更早训练群体历史校准；Extreme 增加其他三个完整事件，人口固定。事件参考是群体客流相对于Normal训练同星期均值的指数；按站归一后平均，不是个人选择标签。普通参照为零时指数置空，不编造比例。
6. 直接 few-shot：Normal 最多八条历史；Extreme 给本站四条Normal训练参照及其余三个事件的五日真实客流，直接输出本站日客流。

角色概率在两站共享，人口参数按站分别拟合。人口份额约束、12个画像及Mesa汇总方式沿用原实现。所有LLM提供逐小时降雨，规则同时保留日汇总和时段两版。

固定人口与Normal/Extreme历史上下文变化之间可能不匹配；本实验忠实测试这种迁移，而不是在每折偷偷重新校准。另做普通天气探针诊断，不据此修改测试预测。观察到的事件客流变化不等于纯天气因果效应。

## 运行

在本文件所在目录：

```bash
conda activate dev
python design.py
python run.py --preflight
python run.py
```

新调用单独记账；同一实验完全相同请求复用。本轮不读取原实验回复或权重。`--offline`只允许读取本轮缓存。七种方法共42个任务/方法组合。主得分：Normal两个任务等权；Extreme四个中心日等权（先按任务汇总两站），事件前后日另报。展示得分为100×(1−WAPE)，不是分类正确率。

`res/panel.csv`记录日期和实际标签，`fold_reference_dates.csv`列出每折可见事件日期，`populations.csv`只应有两个Normal任务，`decisions.csv`保存概率和固定人口来源。原始API文本不公开。

主实验完成后，依次运行：

```bash
python probe_context.py
python analyze.py
python report.py
```

`probe_context.py`会在同一本轮预算内调用API做干燥日诊断；相同请求可从本轮缓存恢复。`analyze.py`与`report.py`均不调用模型。报告为 `RESULTS.md`，图表为 `res/performance.png`、`res/events.png`。核心源文件和输入哈希在 `res/protocol.json`；结果检查在 `res/result_audit.json`。
