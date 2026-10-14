# 纽约地铁逐日客流：规则 ABM 与 LLM 对比

实验使用四个代表站点的真实进站量，比较六种方法。正常天气分暑假、学期；强降雨按四个事件滚动训练和测试。先看 [RESULTS.md](RESULTS.md)，具体日期与方法看 [PROTOCOL.md](PROTOCOL.md)。

## 运行

使用 conda dev，从本目录运行：

```bash
python prepare.py
python -m unittest test_experiment.py
python experiment.py primary
python experiment.py repeat --replicate 1
python experiment.py repeat --replicate 2
python context_probe.py  # 可选：只替换10/7的一条历史示例
python analyze.py
python animation_demo.py
python audit.py
```

`prepare.py` 从共享 data 重建日数据与划分。`primary` 会调用 API，保存完整主实验；`repeat` 只重复四个事件当天的LLM推断，保持训练参数不变。已有完整请求读取缓存；加 `--offline` 禁止网络调用。只查看结果不必重跑实验；`analyze.py`、`audit.py` 均离线。不要为了检查结果重跑 primary，否则 runtime 将记录缓存重放耗时。

API密钥放上一级 `key.txt`，不上传。模型固定 Lite，新增预算硬限200万token（包括失败及重试），相同调用复用上次真实响应。`res/reused_api.jsonl.gz` 是本轮用到的旧调用证据，`res/api_calls.jsonl.gz` 是本轮新调用；代码会同时读取未压缩的新追加日志。一次只运行一个调用进程，避免跨进程重复消耗预算。并发默认6，本次设置 `ABM_WORKERS=10`。改变prompt、模型或replicate会产生新调用。

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
| decisions.csv | 方法×角色×站点×日期 | effective_population, p_subway/bus/taxi/walk/stay_home, reason |
| populations.csv | 方法×任务×站点×角色 | effective_population：训练出的有效规模，不是真实人数 |
| capacity_diagnostic.csv | 方法×任务×票种 | unavoidable_underprediction_pct：规模上限导致的最低低估程度 |
| history_examples.csv | 角色few-shot的一条实际参考日 | task, target_date, group, example_date, entry_index, rain_mm |
| weather_range.csv | 一个事件 | 训练最大雨量、测试雨量、是否超范围 |
| previous_comparison.csv | 版本×同一事件×方法×重复 | 仅相同事件日与站点的旧/新对比 |
| api_cost.csv / budget.json | 方法 / 本轮总计 | 新调用、token、API延迟、预算 |
| runtime.csv | 任务×方法 | 本轮实际阶段耗时，含缓存读取，非全新调用成本 |

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
