# 纽约地铁逐日客流：规则 ABM 与 LLM 对比

新增：[降雨时段与Pro小测试](TIMING_RESULTS.md)，设定见[TIMING.md](TIMING.md)。运行 `python timing.py rules`、`python timing.py pro --offline`、`python timing_report.py` 可离线复现新增结果；去掉Pro命令的 `--offline` 才允许新调用。时段模块单独限额16万token，不与原Lite预算混用。

实验使用四个代表站点的真实进站量，比较六种方法。正常天气分暑假、学期；强降雨按四个事件滚动训练和测试。先看 [RESULTS.md](RESULTS.md)，具体日期与方法看 [PROTOCOL.md](PROTOCOL.md)。

## 交接：只复现最新时段对照

从仓库根目录开始，建议Python 3.11。新电脑可创建环境；已有dev环境则直接激活。

```bash
conda create -n subway-abm python=3.11 -y
conda activate subway-abm
python -m pip install -r llm_based_abm/requirements.txt
cd llm_based_abm/current
python reproduce.py
```

已有环境可用 `conda activate dev` 替代前两行。代码从current目录运行，使用相对路径。

`reproduce.py` 依次执行10项测试、重新拟合规则ABM的日汇总/时段两组、读取真实Pro缓存重算预测、核验并生成报告。缺缓存会报错，绝不会偷偷调用API。无需API密钥，无需下载额外数据，也无需跑prepare.py或旧主实验。仓库内的数据、已拟合权重和压缩API证据是本次对照的必要输入。

结果入口：

- `TIMING_RESULTS.md`：最新说明、两组对照、图和成本。
- `res/timing/pro_metrics.csv`：两个事件×三个Pro方法×两种天气输入的误差。
- `res/timing/rule_metrics.csv`：规则ABM六个任务的整窗口及事件日误差。
- `res/timing/pro_decisions.csv`：角色概率、代表权重和简短理由。
- `res/timing/audit.json`：原始回复、固定权重、相同样本等核验结果。

这里的“复现”是使用相同真实API回复重新计算，不是重新向Pro抽取一批回答。Pro外层沿用已有Lite校准权重；规则模型重新拟合。计时文件会更新，数值结果应一致，浮点末位允许微小差异。

## 如需联网调用

API密钥放在 `llm_based_abm/key.txt`，不要提交。`python timing.py pro` 仅在请求缓存缺失时联网，完整缓存存在时仍复用，不会自动生成第二次独立试验。新增调用写入 `res/timing/api_calls.jsonl`，与已提交的压缩日志共同计入160,000 token上限；已有98,196 token也计入。不要删除日志来重置预算。

要独立重复Pro实验，应另设replicate及预算记录，并重新安排验证，而不是直接把已有日志删掉。当前默认入口只交接可验证的已有结果。

## 代码与对照范围

最新新增代码为 `timing.py`、`timing_api.py`、`timing_report.py`、`reproduce.py`，固定设定见[TIMING.md](TIMING.md)。日汇总与加入时段的对比保留在同一模块中。

`model.py`、`experiment.py`等是当前模块使用的共同代码；主实验结果与旧API记录用于校准来源和基线核对，不是需要另跑的旧版本。当前仓库没有V1、V2等历史版本副本，历史实现通过Git提交查看。

## 文件入口

- `prepare.py`：日聚合、完整性、日期划分。
- `model.py`：人物、规则、训练规模、Mesa汇总。
- `experiment.py`：历史示例、训练/验证/测试、重复。
- `api.py`：提示、真实请求、缓存与费用上限。
- `analyze.py`：统计表和图；`audit.py`：离线回放检查。
- `animation_demo.py`：网页展示数据小样。
- `context_probe.py`：固定权重、强制暴雨历史的小型对照，见 [说明](CONTEXT_PROBE.md)。
- `res/`：数据、预测、真实API记录、图。

当前目录只保留最新版。此前完整实验在Git历史提交 `7f973e8018e0cec16dcebfbaf9f098f4fc280a07` 中；不需要复制回当前目录即可在GitHub浏览。`res/previous_reference.csv` 仅保留本次同日比较需要的旧预测，来源已记录在 protocol_manifest。

## 数据接口

CSV为UTF-8；日期 `YYYY-MM-DD`，站点ID按字符串读取；雨量mm，客流为每日进站次数，概率0—1。缺整小时的站点日不补零、不计分。

| 文件 | 一行代表 | 主要字段 |
|---|---|---|
| dates.csv | 任务×日期 | task, date, split, event_day, rain_mm, peak_mm, school_term |
| daily.csv | 站点×日期 | station_complex_id, date, total_ridership, students, fair_fare, seniors_disability, general, observed_hours |
| panel.csv | 任务×站点×日期 | 上述字段与train/validation/test归属 |
| predictions.csv | 方法×站点×测试日期 | task, method, date, station_complex_id, actual, predicted, replicate |
| metrics.csv | 方法×任务 | WAPE_pct, MAE, RMSE, bias_pct, within_20pct |
| event_summary.csv | 方法×事件 | WAPE_mean/std/min/max, bias_mean, runs；LLM为重复推断统计 |
| group_predictions.csv | 方法×票种×站点×日期 | actual, predicted；用于群体误差 |
| decisions.csv | 方法×角色×站点×日期 | base_population, weekday_multiplier, effective_population（前两者乘积）, p_subway/bus/taxi/walk/stay_home, reason |
| populations.csv | 方法×任务×站点×角色 | effective_population：基础规模；weekday_0…weekday_6：固定星期活跃度，均来自训练集 |
| capacity_diagnostic.csv | 方法×任务×票种 | unavoidable_underprediction_pct：规模上限导致的最低低估程度 |
| history_examples.csv | 角色few-shot的一条实际参考日 | task, target_date, group, example_date, entry_index, rain_mm |
| weather_range.csv | 一个事件 | 训练最大雨量、测试雨量、是否超范围 |
| previous_comparison.csv | 版本×同一事件×方法×重复 | 仅相同事件日与站点的旧/新对比 |
| api_cost.csv / budget.json | 方法 / 本轮总计 | 新调用、token、API延迟、预算 |
| runtime.csv | 任务×方法 | 本次离线重新拟合耗时，非全新调用成本 |

`*_policy.json` 保存主实验角色概率，支持数值回放；完整原始请求与回复见压缩API记录。`res/figures/` 为可直接用于汇报的PNG。群体是票种代理标签，不是完整人口分类；源数据合并老人和残障，不能由此拆出真实数量。角色few-shot接收群体历史指数，不是个人行动标签。

## 网页展示小样

`res/animation_demo/` 提供全网当前站点节点和简化边，以及四个起点的80个合成人物。人物动作来自本轮9/29角色zero-shot；目的地、出发时间和线路是演示设定，不能叫真实乘客轨迹。

| 文件 | 主键 | 用法 |
|---|---|---|
| nodes.csv | station_id | name, longitude, latitude，WGS84落图 |
| edges.csv | from_station, to_station | 简化有向边和seconds |
| agents.csv | id | 画像、动作、代表权重、reason |
| trips.csv | agent_id | 起终点、出发/到达秒数、站点路径 |
| waypoints.csv | agent_id, sequence | 沿路径位置、当天零点起seconds |

网页按相邻waypoint插值即可。当前路网不还原2023年临时关闭；无真实OD，因此动画用于展示模型机制。后续全站预测、OD与时段分配需独立验证。

## 周周期与权重检查

本次修复无新增 API 消耗，原始 LLM 概率不变。具体方法见 [REPAIR.md](REPAIR.md)。`repair_check.py` 生成修复前后误差、分星期偏差和角色份额表（res/repair_*）。同票种三个代表角色占比限制在10%—60%，属于技术约束，不是实测人口比例。
