# LLM 实验

最新的分站点天气实验入口是 [spatial_weather](spatial_weather/README.md)，结果见 [RESULTS](spatial_weather/RESULTS.md)。使用队友提供的空间插值天气，在 Parsons Blvd 和 176 St 重跑七种方法、两个 Normal 任务及四个 Extreme 事件。每站单独调用 Lite，重新拟合权重；原始 API 日志保留在本地。

此前统一 JFK 天气的四站实验在 [fresh_lite](fresh_lite/README.md)，作为新旧对照，也提供本轮复用的模型和 API 模块。运行新版仍需要保留该目录。

依赖使用 requirements.txt。API 密钥在本地 key.txt，不提交。旧 current 目录不是本轮运行依赖。
