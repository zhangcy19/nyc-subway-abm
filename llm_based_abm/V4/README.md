# V4：扩大数据后的两条 LLM 路线

同一组站点与日期上比较：规则 ABM、数值基线、Lite 直接预测、Lite 扮演乘客。两条 LLM 路线分别做 zero-shot / few-shot，并分开看干燥、普通雨天、强降雨。

先看 [RESULTS.md](RESULTS.md) 的误差和图；实验约定在 [PROTOCOL.md](PROTOCOL.md)，实际日期和站点在 `res/design.json`、`res/stations.csv`。

## 运行

从本目录运行，使用 `/opt/anaconda3/envs/dev/bin/python`。密钥仍只放 `../key.txt`，不上传。
下面命令按顺序执行，避免多个进程同时写同一缓存。已有缓存可以直接加 `--offline` 重放；`prepare.py` 生成不上传的 panel。

```bash
python prepare.py
python -m unittest test_experiment.py
python experiment.py pilot
python experiment.py validation
python experiment.py test
python experiment.py test --subset
python experiment.py test --subset --enriched
python experiment.py test --subset --replicate 1
python experiment.py test --subset --agents 16
python experiment.py test --subset --agents 24
python analyze.py
```

`--offline` 只重放已有真实缓存，缺少结果会停止，不调用 API。新运行会命中相同提示的缓存；改提示、记忆、画像或 replicate 会产生新调用并计入预算。

子集的四个配置（8/16/24 人，以及 8 人 + enriched）各运行 replicate 0/1/2。首轮发现采样波动后补做重复，主实验不变。用 `--replicate 1`、`--replicate 2` 指定重复，文件分别以 `_repeat`、`_repeat2` 结尾；这些重复是完整四日序列，后续个人记忆可能不同。

`prepare.py` 从共享数据提取八站完整时间网格，构造历史计划量和可选人口/网络背景；不改变共享 data 或 V1–V3。`model.py` 是 Mesa 代表人物和透明规则；`experiment.py` 调用 Lite 并汇总；`analyze.py` 免费评分及画图。
这里的训练只用于统计模型与历史示例，不对 Lite 权重做微调。

## 输出

- `*_predictions.csv.gz`：每方法、站点、小时的真值与预测，缺失真值仍为 NaN；test 是主结果，validation 是参数选择，probe 是额外对照。
- `*_decisions.csv.gz`：个人行动、理由、代表权重和请求哈希；直接预测只记录该站点的一段理由，不冒充个人行动。
- `representatives.csv`：主实验和人数对照的合成人物参数；职业、目的、预算与性格都是假设，组内权重不是全站的人群比例。
- `*_mass.csv`：计划量、当天地铁量、其他方式与跨午夜量的守恒。
- `api_calls.jsonl.gz`：真实输入、原文、usage、失败及修复记录；`api_usage.csv`、`budget.json` 汇总费用相关用量。
- `metrics.csv`、`weather_metrics.csv`、`event_metrics.csv` 等：总误差、分天气、分事件；`calibrated_metrics.csv` 仅使用验证期拟合的总量校正。
- `probe_metrics.csv`：人数、重复调用和新背景维度的同一子集对照；不能与不同日期的主结果直接比高低。
- `probe_repeat_summary.csv`：各配置三次完整序列的均值、标准差和范围；三次采样不足以作可靠的显著性判断。
- `probe_ensemble_metrics.csv`：同一配置三次小时预测等权平均后的误差，与“平均三次误差”不同；只作子集诊断。
- `rule_sensitivity.csv`、`rule_weather_sensitivity.csv`：事后免费诊断事先列出的规则参数档位，不用于重新选择主模型。
- `figures/`：整体误差、天气、三次事件、站点空间、群体选择及人数敏感性。

角色路线每站 8 人代表四个票种层，各两人。人物分别覆盖固定与灵活出行；老人和残障需求在合成人物中分开，但原始票种标签无法分开评分。一个人的六个决策是六种可能出发时段，不是六次真实出行。概率由多人的加权选择得到，小时量沿用历史出行分布，不让 LLM 猜全站人数。

例如，某时段历史计划中有 100 次学生出行，两个学生代表各占一半。一人选地铁、一人取消，则预测 50 次学生入站；推迟的人会把自己的份额移到后一小时。增加到 16/24 个代表会让选择份额更细，但不会自动获得更多真实的人群信息。

新 ACS/当前路网只进入单独的回顾性背景对照，避免时间泄漏。三次强降雨仅有三个事件样本；本实验能比较聚合预测与约束一致性，不能证明虚拟人物等同于真实乘客，也不是完整的拥挤、列车调度和人与人交互仿真。

WAPE 是“各站预测与实际的绝对差之和 ÷ 实际客流之和”，越低越好。日总指标先按站点、日期加总，小时指标直接算每小时差值；有缺失观测的日期只比较共同有观测的小时。

`res/manifest.json` 记录实验设计及实现修正。角色记忆中的雨天额外耗时只是模拟参数；取消或远程时为零。额外重复测试重新执行整个四日序列，首日输入相同，后续记忆可能不同，检验的是完整序列的随机波动。
