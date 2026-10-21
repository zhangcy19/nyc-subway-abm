# 网页可视化数据

基于Role LLM few-shot，共152站、12类角色，覆盖2023年7月29日、8月13日、9月29日、10月7日及各前后两天。

| 文件 | 一条记录 / 主要字段 |
|---|---|
| `events.json` | 事件：`task, event_date, dates` |
| `stations.csv` | 152个评估站：`station_complex_id, station_name, latitude, longitude` |
| `station_flows.csv` | 站点每天客流：`task, date, station_complex_id, actual, predicted` |
| `profiles.json` | 角色：`agent_id, group, routine, budget_usd, mobility_limited` 等 |
| `policies.json` | 角色每天决策：`policy_id, task, date, agent_id, probabilities, reason` |
| `station_decisions.csv` | 站点角色权重：`task, date, station_complex_id, agent_id, policy_id, effective_population` |
| `individuals.json` | 动画个体：`individual_id, task, date, origin, agent_id, policy_id, decision, destination, route_id, departure_hour, display_weight` |
| `routes.json` | 地铁示意路径：`route_id, stations, coordinates, elapsed_seconds, lines` |
| `network.geojson` / `network_stations.csv` | 线路连边 / 全网节点（含路径途经站） |
| `weather_hourly.csv` | 纽约当地小时天气：`timestamp, rain_mm, temp_c, wind_ms` 等 |

网页连接方式：按 `task + date` 筛选个体；`origin` 查站点，`agent_id` 查角色，`policy_id` 查概率和理由，`route_id` 查动画路径。坐标为 `[经度, 纬度]`，路径累计时间单位为秒；编号按字符串读取。

`probabilities` 包含 `subway / bus / taxi / walk / stay_home`，总和为1。`reason` 是模型原始简短理由，没有完整思维链；同类角色的概率与理由在各站共享。

个体行动、出发小时、目的地和路径是动画用的模拟分配，非真实乘客轨迹。仅地铁行动提供路径；其他行动的 `destination/route_id` 为 `null`，不出行的小时也为 `null`。线路为当前GTFS连边示意，非2023年真实轨道或无障碍路径。

`display_weight` 是拟合代表量，不是人数。客流图请用 `station_flows.csv`，不要累加抽样动画个体。模型客流等于各角色 `effective_population × probabilities.subway` 之和。

## 可能的网页动画

| 动画 | 展示效果 | 数据支持 |
|---|---|---|
| 地铁移动粒子 | 小圆点从起点站沿线路移动到目的地，不同颜色表示不同角色群体 | `individuals.json`、`routes.json` |
| 天气与出行时间轴 | 播放事件前后五天，降雨图随时间变化，地图同步显示当天个体和站点客流 | `events.json`、`weather_hourly.csv`、个体与客流数据 |
| 点击个体看决策 | 点击圆点显示角色属性、五种方式的概率和英文决策理由 | `profiles.json`、`policies.json` |
| 站点客流变化 | 站点圆圈随预测进站量变大变小；点击显示真实与预测客流折线 | `station_flows.csv`、`stations.csv` |
| 同一角色跨日对比 | 跟踪同一个模拟角色，看事件前、当天、之后的出行选择与理由 | `individuals.json` 中的稳定 `individual_id`、`policies.json` |
