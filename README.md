# NYC Subway ABM

最新实验：[分站点天气两站重跑结果](llm_based_abm/spatial_weather/RESULTS.md)。使用空间插值天气，覆盖 Parsons Blvd 与 176 St、七种方法、Normal 与四个事件日；[运行说明](llm_based_abm/spatial_weather/README.md)。新版执行入口为 `llm_based_abm/spatial_weather/run_spatial.py`。

上一轮实验：[四个新站点的独立Lite重跑](llm_based_abm/fresh_lite/RESULTS.md)，2026-10-18。模型配置保持不变，重新生成训练、验证、测试的LLM回复，并重新拟合全部外层权重。没有加载此前实验的回复、权重或预测结果。

- [上一轮结果与图](llm_based_abm/fresh_lite/RESULTS.md)
- [运行与交接](llm_based_abm/fresh_lite/README.md)
- [固定样本与模型设定](llm_based_abm/fresh_lite/PROTOCOL.md)
- [共享数据与来源](data/README.md)

四站为176 St（4）、Fulton St（G）、86 St（1）、Parsons Blvd（F）。比较星期均值、Ridge、规则ABM日汇总/时段，以及Lite角色zero-shot、角色few-shot和直接few-shot；LLM统一使用最新时段输入。正常天气分暑假和学期，极端天气按四个事件分别训练/验证/测试。结果同时列出排除9/29后的对照。

上一轮执行入口为llm_based_abm/fresh_lite/run.py；离线复现只读本轮缓存。此前current目录是上一轮实验，未参与本轮数据加载，不要混合其成绩。历史实现和发布记录可通过Git提交查看。

data是小组共享数据，rule_based_abm是协作者工作区。阶段成果同步llm/main，无需PR。密钥不上传。
