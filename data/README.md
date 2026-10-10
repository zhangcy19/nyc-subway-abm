# 交接数据

范围：2023 年 8–9 月，纽约地铁。

- `stations.csv`：站点编号、名称、行政区与坐标。
- `station_hourly.csv.gz`：完整站点小时客流，已融合票种、站点和天气信息。因上传接口大小限制使用无损 gzip 压缩，解压后与原 CSV 完全一致。
- `weather_hourly.csv`：JFK 小时天气。

时间为纽约当地时间，站点编号按字符串读取，空值表示缺失。

```python
import pandas as pd
panel = pd.read_csv('data/station_hourly.csv.gz', dtype={'station_complex_id': str}, parse_dates=['timestamp'])
```

来源：[MTA 客流](https://data.ny.gov/d/wujg-7c2s) 与 [Iowa Environmental Mesonet 天气](https://mesonet.agron.iastate.edu/request/download.phtml)。
