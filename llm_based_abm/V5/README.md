# V5：正常天气与极端天气下的逐日客流预测

本版比较六种方法：星期均值、Ridge、规则 ABM、LLM 直接 few-shot、LLM 角色无历史、LLM 角色有历史。四个代表站点，两个 Normal 窗口，一个 Extreme 事件任务。结果见 `RESULTS.md`，日期、阈值、参数与方法边界见 [PROTOCOL.md](PROTOCOL.md)。

## 运行

从本目录运行，Python 使用 `/opt/anaconda3/envs/dev/bin/python`。密钥在上一级 `key.txt`，不上传。

```bash
python prepare.py
python -m unittest test_experiment.py
python experiment.py primary
python experiment.py repeat --replicate 1
python experiment.py repeat --replicate 2
python experiment.py transfer
python sensitivity.py
python analyze.py
python animation_demo.py
python audit.py
```

主实验和补充实验均使用 Lite。每个请求记录输入、原始回复、usage、耗时及哈希；总保守额度硬限 400 万 token。相同请求读取缓存，`--offline` 强制不访问 API。一次只运行一个调用进程，避免两个进程各自计算预算。运行前检查已有缓存；改提示或 replicate 会产生新请求。重复实验只重新抽取测试决策，训练系数保持不变。

本次主实验使用默认 6 并发；补充实验使用 `V5_WORKERS=12 python experiment.py ...`，输入和预算规则不变。`sensitivity.py` 固定人物、星期、学期及历史示例，只替换雨量，是事后行为诊断，不用于调参或替换主结果。

只检查已有结果时运行 `python audit.py`，不会调用 API。重新执行整套实验会重新生成 runtime 表，缓存重放时间不能当成首次真实 API 运行时间；本次首次运行时间已写入 RESULTS。

## 代码

- `prepare.py`：日聚合、完整性和独立任务切分。
- `model.py`：人物、规则、非负权重拟合、Mesa 汇总。
- `api.py`：Lite 请求、输出校验、缓存、费用保护。
- `experiment.py`：提示构建、训练/验证/测试、重复和迁移实验。
- `analyze.py`：误差、群体选择、成本、图。
- `animation_demo.py`：生成少量网页对接样本。

## 结果文件

| 文件 | 一行代表什么 | 主要字段 |
|---|---|---|
| `res/dates.csv` | 一个任务的一天 | task, date, split, event, event_day, rain_mm, peak_mm |
| `res/panel.csv` | 任务 × 站点 × 天 | station_complex_id, total_ridership, 四票种层, observed_hours, 天气 |
| `res/predictions.csv` | 方法 × 站点 × 测试日 | task, method, replicate, actual, predicted |
| `res/metrics.csv` | 方法 × 任务 | WAPE_pct, MAE, RMSE, bias_pct, within_10pct, within_20pct |
| `res/extreme_metrics.csv` | 事件当天/对照日 × 方法 | condition 和同样的误差字段 |
| `res/populations.csv` | 方法 × 站点 × 人物 | effective_population；是拟合系数，不是真实人口 |
| `res/decisions.csv` | 方法 × 站点 × 测试日 × 人物 | p_subway / p_bus / p_taxi / p_walk / p_stay_home, reason |
| `res/group_actions.csv` | 群体 × 条件 × 动作 | probability；为有效规模加权的模型概率 |
| `res/api_cost.csv` | LLM 方法 | calls, tokens, charged_tokens, api_seconds_sum, mean_latency_s |
| `res/runtime.csv` | 主实验任务 × 方法 | seconds；包括该阶段训练、验证、推断及 API 等待 |

`*_policy.json` 保存每个人每天的真实策略回复，供离线重放与迁移复用；API ledger 保存原始证据，压缩后上传。`repeat*` 是固定拟合参数的重复推断；`transfer*` 是另一任务的训练数据、参数与示例直接迁移。`figures/` 为 PNG 图。CSV 使用 UTF-8，日期 `YYYY-MM-DD`，站点 ID 按字符串读取，雨量单位 mm，概率 0–1，客流单位 entries/day。

天气同城四站共用 JFK 观测。老人和残障群体在原始票种里无法拆开；人物设定区分需求，汇总评分仍使用 `seniors_disability`。角色有历史使用真实群体客流示例，未伪造个体行动历史。

## 给网页组的初步接口

这里只提供可行性样本，不是全站预测。`res/animation_demo/`：

| 文件 | 主键 | 含义 |
|---|---|---|
| nodes.csv | station_id | 全部当前站点复合体的经纬度，WGS84，经度在前、纬度在后 |
| edges.csv | from_station, to_station | 简化有向网络和静态旅行秒数 |
| agents.csv | id | 80 个合成展示人物，画像、出发站、抽样动作、代表权重 |
| trips.csv | agent_id | 乘地铁样本的合成目的站、出发/到达秒数和路径 |
| waypoints.csv | agent_id, sequence | 沿路径逐点位置与从当天零点起算的秒数 |

网页可按 `seconds` 在相邻 waypoint 之间插值移动图标。路线目前是站间示意折线，不是真实轨道形状；目的地和出发时间均为假设，当前网络不保证还原 2023 年状态。不要把动画标题写成“真实乘客轨迹”。人物理由是生成的简短解释，不是可验证的心理过程。

后续若正式扩展：先对全站输出日预测，再确定经过数据支持的 OD 分配方法、出发时段和静态/受扰网络；将入站规模拆为带权虚拟旅次，输出同样的表即可。真实 OD 不足时应明确做情景演示，不能靠这些入站量恢复真实目的地。
