# 分站点天气：两站完整重跑

在 `llm_based_abm/spatial_weather` 运行，Python 使用 `/opt/anaconda3/envs/dev/bin/python`。

## 数据与范围

- 原始天气：`../../data/weather/subway_weather_hourly.zip` 内同名 CSV；Git 源提交 `714cc5a`。
- 424 个地铁站、2023 年 7–10 月、每站 2952 小时；共 1,251,648 行。
- 天气来自 6 个气象站。每个地铁站采用最近 3 个气象站的距离平方倒数加权；这是空间估计，不是地铁站现场观测。
- 选择原实验中的 Parsons Blvd（256，Queens）和 176 St（385，Bronx）。选择依据是地理差异以及能够与原结果配对，不依据新模型得分。二者的最近三个气象站均不包含降雨质量存疑的 JRB。
- 所有站点在 7/14 12:00、7/19 04:00、8/14 06:00 和 09:00 缺降雨。涉及的日期整体排除，不填零；新旧评分使用相同测试日期。`res/excluded_weather_days.csv` 列明被排除记录。
- 独立重算了全表的降雨、温度、风速插值；误差不超过 CSV 六位小数的舍入范围。核查通过不代表气象站观测本身已经得到独立验证。

## 模型和对照

七种方法：星期均值、Ridge、规则 ABM（日汇总/加入时段）、角色 LLM（zero-shot/few-shot）、直接 LLM（few-shot）。复用 `../fresh_lite` 的模型实现、12 个画像、参数搜索范围、提示词以及 Lite 模型。规则的“加入时段”使用画像假设出发时间内的降雨暴露；两类 LLM 读取逐小时雨量。

每站单独提供当天及历史示例的本地天气，单独调用角色 LLM；不能把一站回复直接当作另一站的决策。角色 few-shot 继续使用原四站归一化票种客流历史，不是个人出行记录；配套天气使用当前目标站的本地天气。直接预测最多给 8 条本站历史，角色 few-shot 最多给 4 条更早的训练历史。

训练、验证、测试时间窗沿用旧实验，剔除缺天气日期。每站独立从训练集拟合、验证集选参数，测试数据不参与选择。此前原实验在四站验证集上共同选择超参数，因此新旧结果还存在验证范围差异；加上训练缺失日排除和 LLM 采样随机性，不能把全部分差解释成天气来源的因果效果。

保留原来的 Normal / Extreme 任务名称便于比较。这些名称沿用原先基于 JFK 的日期选择：事件日在某个站点可能仅有小雨，不能说每个 Extreme 样本都是当地暴雨。`res/event_weather_comparison.csv` 给出四个事件日的新旧雨量。

## 顺序

```bash
conda activate dev
cd llm_based_abm/spatial_weather
python check_weather.py
python prepare_spatial.py
python check_inputs.py
python run_spatial.py --preflight
python run_spatial.py
python report_spatial.py
```

天气已准备好时可直接从 `check_inputs.py` 开始。API 密钥沿用 `../key.txt`，不会输出到结果。新一轮单独记账，不读取以前实验的 API 回复；可中断后用同一命令续跑。本轮预算经授权放宽，硬上限为 750 万 token。失败但拿不到实际用量时按保守估计计费。运行时间包含调用等待和参数拟合。

## 输出

- `res/panel.csv`：两站每日真实客流、票种组客流、本地天气、task/split/event_day。
- `res/predictions.csv`：task、method、station_complex_id、date、actual、predicted；部分方法另有 request_hash。
- `res/decisions.csv`：画像的五类决策概率、校准人口与星期系数；只保存可公开数值。
- `res/populations.csv`、`res/validation.csv`：训练人口参数和验证选参记录。
- `res/runtime.csv`：各站、任务、方法的墙钟秒数与新增 token；本轮缓存命中不再次计费。
- `res/summary.csv`：Normal / Extreme 的新旧得分；`task_metrics.csv`、`station_task_metrics.csv` 可展开到任务和站点。
- `res/paired_predictions.csv`：相同站点、日期、方法的新旧预测。
- `res/accuracy_comparison.png`、`res/event_predictions.png`：比较图。

“准确率”按展示口径为 `100 × (1 − WAPE)`，不是分类正确率，不裁剪负数。Normal 先在每个正常任务内汇总两站，再对两个任务等权平均；Extreme 只评分四个事件当天，先汇总两站，再对四个事件等权平均。事件前后参考日的预测也保存在结果里。

原始 API 日志仅在本机 `res/api_calls.jsonl`，已加入 Git 忽略，不上传提示词和模型原文。GitHub 同步代码、说明和数值结果。

另外提供 `local_weather_subset_metrics.csv`：Normal 测试子集要求本站日雨量 <10 mm 且小时峰值 <10 mm；Extreme 子集要求日雨量 ≥25 mm 或小时峰值 ≥10 mm。这是固定阈值下的补充评分，未重新训练模型，也未重新调用 API。因日期减少，不能直接与主表比较来宣称提升。主表仍是原日期配对结果。

完整跑完后执行 `python validate_results.py` 检查全部 84 个组合、共同测试样本、概率、人口权重及 token 预算。`api_cost.csv` 根据完整调用账本统计，包含重试；其中请求秒数之和不是并发运行的墙钟时间。`runtime.csv` 记录各组合的完成段耗时和该段新增 token，中断前已经入账但尚未完成的段可能不在续跑耗时内；总 token 以 `result_audit.json` 为准。
