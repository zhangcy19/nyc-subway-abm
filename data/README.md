# 共享数据

模型通过 `load_station_hourly()` 读取完整客流，站点和天气直接读取根目录的 CSV。范围是 **2023 年 7–10 月、纽约地铁 424 个 station complex**；一个 complex 可能包含多条线路或多个站台，不能把它当作单独的站台。

## 模型输入与替换对应

| 文件 | 每行是什么 | 主键 | 记录数 |
| --- | --- | --- | ---: |
| `station_hourly.csv.gz` | 原有 8–9 月站点小时客流、票种与天气 | `timestamp, station_complex_id` | 606,750 |
| `extensions/station_hourly_2023_07_*.csv.gz` | 新增 7 月，同样的字段 | 同上 | 309,176 |
| `extensions/station_hourly_2023_10_*.csv.gz` | 新增 10 月，同样的字段 | 同上 | 309,326 |
| `stations.csv` | 一个站点的名称与坐标 | `station_complex_id` | 424 |
| `weather_hourly.csv` | JFK 一小时天气 | `timestamp` | 2,952 |

完整客流共 1,225,252 行。附件 `model_input_2023_jul_oct/` 的整文件压缩后约 36 MB，整文件及单月上传均遇到接口服务端错误，因此保留已有 8–9 月文件，新增 7 月、10 月按七天分块的压缩 CSV，共 10 个文件，集中放在 `extensions/`。不再另存合并副本。`load_station_hourly()` 将这些数据读成一个相同格式的 DataFrame，模型组不需要自己拼接。

对应关系：原 `station_hourly.csv.gz` 的 8–9 月已有值保留；`stations.csv` 对应附件同名站点表；`weather_hourly.csv` 扩展为四个月，并保留原有 8–9 月记录的文本精度。新旧重叠的 606,750 个站点小时键、客流和票种数值相同；坐标及客流表内天气的少量小数表示差异小于 `1e-12`。新增客流 CSV 只做无损 gzip 压缩。

| 月份 | 站点小时记录 | 全网进站量 |
| --- | ---: | ---: |
| 2023-07 | 309,176 | 91,580,897 |
| 2023-08 | 308,659 | 94,666,920 |
| 2023-09 | 298,091 | 95,990,088 |
| 2023-10 | 309,326 | 105,225,533 |

已有客流记录全部匹配到了天气，站点小时键无重复、总客流无空值或负数。不过，424 站点 × 2,952 小时应有 1,251,648 行，实际少了 26,396 行（2.11%）。缺失的站点小时不能自动补成零。

## 读取和字段约定

从项目根目录运行：

```python
import pandas as pd

from data.load_data import load_station_hourly

panel = load_station_hourly()  # 全部四个月，22 列
# panel = load_station_hourly('2023-09-01', '2023-10-01')  # 左闭右开
stations = pd.read_csv('data/stations.csv', dtype={'station_complex_id': str})
weather = pd.read_csv('data/weather_hourly.csv', parse_dates=['timestamp'])
```

| 字段 | 格式与含义 |
| --- | --- |
| `timestamp` | `YYYY-MM-DD HH:MM:SS`，纽约当地时间，表示该小时；本窗口没有夏令时切换 |
| `station_complex_id` | 字符串；用于连接站点表，不能用站名连接 |
| `station_name, borough` | 名称与行政区；行政区为英文全称 |
| `latitude, longitude` | 十进制度坐标，用于落图 |
| `total_ridership` | 该站点该小时进站量；不是列车人数、出站量或独立乘客人数 |
| `metrocard_fair_fare` | Fair Fares 票种进站量，可作低收入优惠群体的代理标签 |
| `metrocard_students` | 学生票进站量，可作学生群体代理标签 |
| `metrocard_seniors_disability, omny_seniors_disability` | 老人及残障优惠票进站量；原始票种已合并，不能据此拆出两类真实人数 |
| 其余 `metrocard_* / omny_*` | 各票种进站量；完整列名保留原接口，未合并 |
| `rain_mm, temp_c, wind_ms` | 小时降水量（mm）、气温（°C）、风速（m/s） |
| `weather_available` | 布尔值；当前窗口已有记录均为 `True` |
| `weather_station` | 天气站代码，当前为 JFK；所有地铁站共享该天气站，并非站点局地降雨 |

空单元格读为 `NaN`，与零不同。票种列有大量空值，例如学生票有 730,257 个空值；不应无条件补零后当成真实人群人数，也不应强行要求可见票种之和等于总客流。天气已融合进客流表；独立天气表用于补齐时间网格、计算累计降雨及构建未来天气情景。

## 可以继续用的补充数据

`supplementary/` 是可选输入，当前 V1–V3 没有使用它们。保留原始字段，不把尚未验证的数据包装成模型特征。

| 文件 | 可用于什么 | 连接方式与限制 |
| --- | --- | --- |
| `station_complexes_current.csv` | 线路、GTFS stop 映射、当前无障碍属性 | `complex_id` 对 `station_complex_id`，均按字符串读取；445 行，覆盖主数据全部 424 站。`gtfs_stop_ids` 用分号分隔；`ada` 保留源编码，使用前查官方说明。是 2026-10-12 下载的当前状态，不能当作 2023 年无障碍状态 |
| `network_edges_current.csv` | 线路连接与计划运行时间 | 2,084 条有向边、29 个 route；键为 `from_stop_id, to_stop_id, route_id, direction_id`。`scheduled_travel_time_s` 为秒。GTFS 站台 ID 与客流 complex ID 不同，需处理方向后缀 N/S 并经上表映射；只有车内边，没有建立换乘边，`accessible_path` 未知 |
| `accessible_platform_availability_2023_jul_oct.csv` | 研究期行政区层面的无障碍背景 | 按月份与行政区连接，20 行；`availability` 是比例，分钟字段保留。不能说明某站某小时电梯故障；预测时不能提前使用尚未发布的当月统计 |
| `acs_2023_nyc_tract_age_B01001.csv` | 居民年龄构成 | 与另外两张 ACS 表按 `GEO_ID` 连接 |
| `acs_2023_nyc_tract_school_enrollment_B14001.csv` | 居民在校情况 | 同上 |
| `acs_2023_nyc_tract_disability_B18101.csv` | 居民残障情况 | 同上 |
| `service_2023_09_29_trips.csv.gz` | 暴雨当天列车服务记录的辅助分析 | 6,814 行，`trip_uid` 唯一；`route_id` 按字符串读取 |
| `service_2023_09_29_stop_times.csv.gz` | 上述班次的到站、离站记录 | 163,339 行，通过 `trip_uid` 多对一连接 trips；所有 trip 均能匹配。时间字段为 Unix 秒，先按 UTC 转时间，再转为 `America/New_York` |

三张 ACS 表各有 2,327 个纽约 census tract，以 `GEO_ID` 一对一连接；`*_E###` 是估计值，`*_M###` 是误差幅度，保留原字段。它们是 **2023 ACS 五年估计**，不是 2023 年单年样本。不同表的人口范围不同，使用变量前应查表定义；附件旧说明写的 S1401、S1810、DP05 不是这次实际下载的表。现在没有 tract 边界或站点服务区对应表，不能直接把居民比例当成地铁乘客比例，也不能用它们拆分优惠票中的老人和残障人数。

服务记录来自第三方整理的实时信息。stop_times 中有 19 条重复 trip-stop 键，已保留而未擅自聚合；记录延续到纽约时间 9 月 30 日 05:52:41，不能仅凭文件日期归到 9 月 29 日。其 stop ID 有 99.27% 能匹配附件当前 GTFS，余下需单独处理。未出现的班次不等于停运；只有事件日数据，还缺普通日对照。没有把这些记录变成站点关闭标签或直接作为预测日已知特征，以免误用或泄漏未来信息。

## 附件中未导入的内容

| 原附件 | 处理原因 |
| --- | --- |
| `model_input_2023_aug_sep/`、`model_input_2023_jul/`、`model_input_2023_oct/` | 已被完整的 `model_input_2023_jul_oct/` 覆盖，不再重复存储 |
| `data_handoff.zip` 及其中 raw/output | 老交接包，当前三件套已覆盖对应输入 |
| 全国 ACS `.dat` | 已有纽约提取结果，不上传全国大文件 |
| OD `.csv.partial` 与分页 CSV | 下载不完整且两套重复；当前模型没有使用 OD。未来若做出行链或目的地选择，可重新按所需月份完整下载，不能将当前片段当成完整分布 |
| GTFS ZIP、服务 `.tar.xz` | 已有提取结果，不再上传重复压缩包；未来要做线路几何或换乘，需要另取 GTFS stops/shapes 等字段 |
| 2024 出入口表 | 对出入口步行距离研究有用，但当前预测单位是 complex；已有站点坐标及 GTFS 对应，不纳入本轮输入 |
| `event_windows.csv`、`disruption_events_v1.csv` | 是候选实验日期及全网事件线索，不能证明具体站点的停运时间；后续情景设计仍可参考原附件 |
| 清单 XLSX、预览 PNG、inspect、audit JSON、状态与计划文档 | 是工作记录，不是模型输入；部分“待下载/无十月数据”的状态已过时，不复制到代码项目 |
| `agent_group_proxy_contract.csv` | 早期建议映射与实际 ACS 表不同，不作为自动生成画像的接口 |

这些内容保留在原 `5011` 文件夹，未删除。真正不需要进入模型项目的是重复存储、日志和过时清单；OD、出入口等资料属于暂未接入，不能一概视为没有研究价值。

## 来源和重新整理

- 主客流：[MTA Subway Hourly Ridership，wujg-7c2s](https://data.ny.gov/d/wujg-7c2s)。
- 天气：[Iowa Environmental Mesonet ASOS 下载](https://mesonet.agron.iastate.edu/request/download.phtml)，JFK。
- 当前站点及 GTFS 对应：[MTA Subway Stations and Complexes，5f5g-n3cz](https://data.ny.gov/d/5f5g-n3cz)。
- 当前线路：[MTA 静态 GTFS](https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip)，附件于 2026-10-12 下载。
- 月度无障碍统计：[MTA Accessible Platform Availability，thh2-syn7](https://data.ny.gov/d/thh2-syn7)。
- ACS：[Census 2023 五年详细表下载](https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/)，表 B01001、B14001、B18101。
- 服务辅助记录：[SubwayData.nyc 的 2023-09-29 归档](https://subwaydata.nyc/data/subwaydatanyc_2023-09-29_csv.tar.xz)。

源附件在项目旁边的 `5011/数据附件合集/` 时，从 `data/` 运行 `python import_handoff.py` 可重新整理同样的文件。脚本不调用 API，也不下载额外数据。

V1–V3 的准备脚本明确筛选 8 月 1 日至 9 月 30 日，以保持已发布实验的日期和选站方式；结果及 API 缓存没有重新运行。旧 manifest 中的代码哈希记录的是当时的实验版本，新增日期筛选后不代表当前准备脚本的哈希。新实验通过 `load_station_hourly()` 读取四个月主数据，再另行确定训练、验证、测试窗口。
