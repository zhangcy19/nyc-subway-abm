# 分气象站逐小时天气数据

本目录整理用户指定的10个气象站。时间范围已确认为 **2023-07-01 00:00:00—2023-10-31 23:00:00**，使用纽约当地时间 `America/New_York`。本时段全程为EDT（UTC−4），每站完整时间网格共 **2,952小时**。

目前已获取并导出6站；BRON、MANH、BKLN、STAT需向NYS Mesonet申请历史记录，尚未取得观测数据，未生成占位天气CSV。用户所写的EWER按纽瓦克机场 **EWR** 处理。

## 文件清单与覆盖率

“完整小时”表示降水、温度、风速三项都有值且通过本次明确的缺测/PNO检查，不表示经过完整气象学质量认证。缺失小时仍保留该行及该小时可用的其他变量。

| 代码 | 站点 | 文件名（待申请站仅为计划文件名） | 完整小时 / 2952 | 降水缺失小时 | 温度缺失小时 | 风速缺失小时 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| BRON | 布朗克斯 | weather_hourly_BRON.csv（待申请） | — | — | — | — |
| NYC | 中央公园 | weather_hourly_NYC.csv | 2804 | 17 | 2 | 134 |
| LGA | 拉瓜迪亚机场 | weather_hourly_LGA.csv | 2946 | 5 | 0 | 1 |
| MANH | 曼哈顿 | weather_hourly_MANH.csv（待申请） | — | — | — | — |
| JRB | 曼哈顿市中心直升机场 | weather_hourly_JRB.csv | 1471 | 1481 | 11 | 0 |
| BKLN | 布鲁克林 | weather_hourly_BKLN.csv（待申请） | — | — | — | — |
| JFK | 肯尼迪机场 | weather_hourly_JFK.csv | 2898 | 54 | 0 | 0 |
| TEB | 泰特伯勒机场 | weather_hourly_TEB.csv | 2897 | 47 | 29 | 37 |
| STAT | 斯塔滕岛 | weather_hourly_STAT.csv（待申请） | — | — | — | — |
| EWR | 纽瓦克自由国际机场 | weather_hourly_EWR.csv | 2741 | 209 | 0 | 2 |

## 与原文件一致的六列格式

```text
timestamp,rain_mm,temp_c,wind_ms,weather_available,weather_station
```

| 字段 | 含义 |
| --- | --- |
| timestamp | 纽约当地小时时间标签，`YYYY-MM-DD HH:00:00`，无时区后缀以兼容原文件 |
| rain_mm | IEM处理后的小时降水量，英寸×25.4转换为毫米，并按PNO质量标记剔除不可用值；保留原有字段名，数据实际为液态当量降水 |
| temp_c | 该钟表小时内最后一个非缺测常规METAR气温，摄氏度；不是时间加权小时均温 |
| wind_ms | 该钟表小时内最后一个非缺测常规METAR持续风速，m/s；不是阵风或时间加权小时均速 |
| weather_available | 三个天气变量都有有效数值时为`True`，否则`False` |
| weather_station | 当前气象站代码；不混入其他站观测 |

缺失值写为空字段，既不补0，也不做时间插值或跨站填充。对降水进行空间插值时应单独检查`rain_mm`是否缺失；不能仅因同小时风速缺失，就丢弃仍然可用的降水记录。

IEM小时降水来自常规报文的降水累计周期，周期起止不一定恰好是整点。当前保留提供方小时标签；该产品不能声称具有严格整点至整点的分钟级降雨时序。微量降水用提供方数值0.0001英寸（0.00254mm）编码，这是trace代理值，并非雨量计具备该测量精度。

## 已发现的质量问题

原始METAR中的独立标记`PNO`表示**降水量不可用**。部分IEM数值记录却为0，因此本次将包含PNO的小时降水设为缺测，而不按“无雨”使用。采用保守规则：同一小时任一常规报文出现PNO，即剔除该小时降水。详见[美国国家气象局METAR说明](https://www.weather.gov/media/zse/metar_decode_key.pdf)。

本批PNO小时数为：JRB 1480、JFK 49、TEB 11、EWR 203；NYC和LGA未见该标记。这些小时与原本缺失小时可能重叠，因此不能将计数简单相加。

**JRB不宜未经进一步核查就作为稳定的降水插值站。** 9月29日和10月7日全天的降水均因PNO不可用，8月13日也仅有2小时降水值；剩余时段的报告降水总量亦明显偏低。温度、风速可按各自缺测情况使用。已保留原始报文和逐小时处理记录，未用附近站的雨量冒充JRB观测。

## 来源与追溯

- 六个已下载站来自[Iowa Environmental Mesonet（IEM）ASOS/METAR档案](https://mesonet.agron.iastate.edu/request/download.phtml)。降水使用[processed hourly precipitation接口](https://mesonet.agron.iastate.edu/cgi-bin/request/hourlyprecip.py?help)，不直接累加同小时所有METAR的累计降水字段。温度和风速使用`report_type=3`的常规报文，避免暴雨期间特别报文较多带来的采样权重变化。
- 原始接口响应保存在`sources/`，原始文件未改写。`station_inventory.json`记录所有10站的经纬度、来源链接、下载状态、缺测统计以及原始与输出文件SHA-256。坐标为当前站点元数据，尚未核查2023年期间是否存在仪器迁移。
- `quality_control_events.json`记录PNO对应的站点、小时、原数值和剔除原因。
- `validation_report.json`记录CSV结构、2,952小时连续性、字段单位转换、原始观测逐值核对及三个研究事件日的可用小时数。部分缺测日仅报告“已观测部分之和”，不冒充完整日降水量。
- BRON、MANH、BKLN、STAT使用[NYS Mesonet申请入口](https://www.nysmesonet.org/weather/requestdata)。所需参数与用途已整理在`NYS_Mesonet_request.md`，目前未提交。

## 处理脚本

1. `scripts/prepare_weather.py`下载并缓存原始数据，执行单位转换、PNO处理，生成JSON中间表与清单。默认时段与本批数据相同。
2. `scripts/export_weather.mjs`使用Artifact Tool写入数据、校验往返值并导出CSV。运行目录需有指向bundled runtime的`node_modules`链接；内部预览放在运行目录的`previews/`。
3. `scripts/verify_weather.py`独立核对导出的六站CSV及原始记录，更新校验报告和文件哈希。本次校验通过。该校验脚本固定检查本批2023年7—10月的六站数据；改变研究时段或补入NYS站点时应相应扩展校验。

## 地铁站逐小时天气表（2026-10-18新增）

已根据`../stations.csv`中的424个地铁站复合体生成 **`subway_weather_hourly.csv`**，共 **424 × 2952 = 1,251,648行数据、11列**。每行代表一个地铁站在一个当地小时时间标签上的天气估计。记录按原地铁站文件的顺序排列，每站内部按时间升序排列。

```text
station_complex_id,station_name,borough,latitude,longitude,timestamp,rain_mm,temp_c,wind_ms,weather_available,weather_station
```

前5列原样继承地铁站元数据。地铁站编号作为标识符处理，不是连续的1—424序号。空间汇总使用上述6个已下载气象站，未将尚未获取的4个NYS站点当成有观测值的站点；以后补齐4站后，需要重算最近邻和全部权重。

### 固定最近3站与逐变量缺测处理

1. 采用经纬度Haversine球面距离，地球平均半径6371.0088 km。为每个地铁站在当前6站中选定距离最近的3个气象站；这3站固定，不随小时或变量变化。
2. 对每个小时的`rain_mm`、`temp_c`、`wind_ms`分别筛选这3站中的有效值，不用来源文件的整行`weather_available`标记来筛掉仍有效的单项变量。
3. 在可用子集中使用反距离平方权重：`w_i = (1/d_i²) / Σ(1/d_j²)`，输出`Σ(w_i × x_i)`。缺测后重新归一化；只剩1站时其权重为1。只要3站中还有一个有效值，就不从第4站补足到3站。
4. 若固定3站对当前变量全部缺测，才在其余气象站中选择同小时最近的可用站，直接使用该值，不做跨小时回填。
5. 若全部6站对当前变量都缺测，则该变量留空。输出行仍保留，`weather_available=False`。三项都得到值时为`True`。
6. 与地铁站重合的有效气象站距离为0时优先使用；若有多个重合站，则等权平均，避免除以0。本批没有此情形。

计算使用双精度，三个天气输出值保留最多6位小数；原始缺测不被替换为0。源文件中的PNO降水剔除已经生效。本批继续使用JRB中未被PNO剔除的降水值，因此仍需注意前述JRB可能低估降水的问题；输出不等同于经过全面气象质量认证的数据。

### `weather_station`来源列

该列记录三个变量**实际使用**的气象站，按距离从近到远排列。例如：

```text
rain_mm=IDW:LGA|NYC;temp_c=IDW:LGA|NYC|JRB;wind_ms=IDW:LGA|NYC|JRB
```

`IDW:`表示从固定最近3站的可用子集中加权；`NEAREST:`表示3站全部缺测后的最近其他站回退；`NONE`表示所有候选站都没有该小时的该变量。未使用的站不会列入来源列。

每站候选气象站距离、固定3站和三站均有值时的基础权重，保存于`subway_weather_weights.json`。对任一CSV行，将来源列列出的站点距离代入上述公式即可复算该行的实际权重；只有一个来源时权重为1。

### 本批数据结果与校验

| 变量 | 最近3站均有效 | 使用最近3站的部分有效值 | 回退到其他最近站 | 所有候选站均缺测 |
| --- | ---: | ---: | ---: | ---: |
| rain_mm | 782395 | 467557 | 0 | 1696 |
| temp_c | 1244548 | 7100 | 0 | 0 |
| wind_ms | 1198922 | 52726 | 0 | 0 |

三项天气均可用的行数为 **1,249,952**；另有 **1,696行降水缺失**，即以下4个时刻的全部424个地铁站：

- 2023-07-14 12:00:00
- 2023-07-19 04:00:00
- 2023-08-14 06:00:00
- 2023-08-14 09:00:00

这4个时刻当前6个气象站均没有可用降水值。它们保留温度、风速和地铁站信息，没有被删除。其他站回退逻辑本次未实际触发，已通过模拟缺测边界检查验证。

`scripts/aggregate_subway_weather.mjs`生成该表及距离、权重和来源清单；`scripts/verify_subway_weather.py`对全部1,251,648行用独立距离公式和源气象文件重新核算。检查包括11列格式、每站完整时间网格、站点元数据、唯一键、加权值、实际来源、缺测标记和数值范围；结果见`subway_weather_validation.json`。`subway_weather_metadata.json`记录参数、输入哈希与输出哈希。

CSV约200 MiB，行数超过单张Excel工作表的1,048,576行上限，建议使用Python/R或数据库读取全量数据；直接用Excel打开不能完整显示全表。

原有`../weather_hourly.csv`及既有实验输入保持原样。模型接入已完成：从`llm_based_abm/current`运行`python spatial_experiment.py prepare`，新输入和结果保存在独立的`res/station_weather/qwen_all_stations/`。日汇总和24小时雨量均按站点读取，角色决策也按站点生成；缺测不补零。完整流程见[模型README](../../llm_based_abm/current/README.md#站点空间天气实验2026-10-18)。
