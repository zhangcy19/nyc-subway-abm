# 纽约地铁 LLM 实验

| 版本 | LLM 做什么 | 入口 |
|---|---|---|
| V1 | 固定天气情境的群体行动概率 | [说明](V1/README.md)、[结果](V1/RESULTS.md) |
| V2 | 读取站点历史，修正群体需求和概率 | [说明](V2/README.md)、[结果](V2/RESULTS.md) |
| V3 | 扮演有画像和记忆的虚拟乘客，作个人选择 | [说明](V3/README.md)、[结果](V3/RESULTS.md) |
| V4 | 两条 Lite 路线 × zero/few-shot，8 站与多次降雨 | [说明](V4/README.md)、[结果](V4/RESULTS.md) |
| V5 | 逐日预测；Normal/Extreme 独立任务；直接 few-shot、角色无历史/有历史；不使用共同计划值 | [说明](V5/README.md)、[结果](V5/RESULTS.md) |

各版本从自己的文件夹运行，使用 dev 环境。共享数据在项目根目录 data/，依赖在本目录 requirements.txt。密钥只放本目录 key.txt，不上传；各版本通过 ../key.txt 读取。

V1/V2 保存代码、真实API响应、策略、压缩预测和图表。panel 可重建，不上传。不同站点、日期和任务的版本误差不能直接作为升级前后比较。代码在 llm 分支开发，阶段完成后直接合入 main。
