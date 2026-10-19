# BRON / MANH / BKLN / STAT 历史数据申请草稿

状态：**未提交**。申请人姓名、邮箱、所属机构和数据政策确认尚缺。

官方表单：https://www.nysmesonet.org/weather/requestdata

## 表单选择

| 项目 | 内容 |
| --- | --- |
| Network | Standard Network |
| Stations | BRON — Bronx; MANH — Manhattan; BKLN — Brooklyn; STAT — Staten Island |
| Start | 2023-07-01 |
| End | 2023-10-31（含当天） |
| Time zone | America/New_York / Eastern Time，保留夏令时 |
| Time resolution | All 5-minute observations，用于在本地统一整理逐小时文件 |
| Units | Metric |
| Variables | Latitude/longitude/elevation; air temperature at 2 m; liquid-equivalent precipitation, incremental totals; wind speed from the best available sensor at 10 m |
| Output format | CSV |
| File organization | 每站单独文件（如可选） |
| Name / Email / Organization | 由申请人填写 |

申请5分钟原始记录是为保留质量标记、明确降水累积区间，并使温度和风速能按与现有文件相近的小时代表值规则处理；最终仍导出逐小时表格。若只能提供小时统计，需要先确认其时间标签和变量统计方式，不能直接混用累计降水与小时降水。

## Planned Use（可直接粘贴）

Academic research on weather-related changes in New York City subway ridership during July–October 2023. We will compare station-level rainfall exposure and examine demand responses using statistical models and rule-based / LLM-assisted behavioral models. Weather observations will be aligned to hourly local-time ridership data and, after quality checks, used for spatial interpolation to subway stations. The study needs observed precipitation, temperature and wind; no personal information about weather-station operators or individual transit passengers is requested.

## Comments（可直接粘贴）

Please provide Standard Network observations for BRON, MANH, BKLN and STAT covering 2023-07-01 00:00 through 2023-10-31 23:59 America/New_York (UTC equivalent: 2023-07-01 04:00 inclusive to 2023-11-01 04:00 exclusive).

We request 5-minute incremental precipitation totals (mm), air temperature at 2 m (degrees C), and sustained wind speed from the best available sensor at 10 m (m/s), with quality-control / missing-value flags and site coordinates. Please document whether timestamps label the start or end of an accumulation interval, the missing-value codes, and the aggregation used for temperature and wind. Please include the boundary record necessary to calculate the first and last complete local clock hours, if applicable.

If only hourly products are available, please specify whether precipitation is an hourly incremental total, rolling accumulation, or cumulative total, and specify the temperature and wind statistics. Please advise the applicable academic-use terms and any fee before fulfilling the request.

## 数据到手后的导入要求

1. 保留原始文件、变量说明、许可与质量标记。
2. 根据提供方定义处理时间标签；5分钟降水仅在完整覆盖相应小时且通过质量检查时求和，不能把滚动1小时量或当日累计量直接相加。
3. 温度和风速优先使用该小时最后一个通过质量检查的观测值，与本批IEM文件的代表值方法保持接近；如改用小时均值，应统一全体站点的统计口径。
4. 按 `README.md` 的六列格式导出 `weather_hourly_BRON.csv`、`weather_hourly_MANH.csv`、`weather_hourly_BKLN.csv`、`weather_hourly_STAT.csv`，保留缺测并更新清单和校验结果。
5. 申请页面要求接受数据政策；使用、共享及发布须依实际获批条款执行。当前未代表申请人接受条款、提交申请或承诺费用。
