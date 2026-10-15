# NYC Subway ABM

纽约地铁逐日客流：比较规则ABM、LLM直接预测和LLM角色决策。

- [降雨时段与Pro小测试](llm_based_abm/current/TIMING_RESULTS.md)
- [主实验结果与图](llm_based_abm/current/RESULTS.md)
- [运行与输出接口](llm_based_abm/current/README.md)
- [实验设定与数据划分](llm_based_abm/current/PROTOCOL.md)
- [共享数据与来源](data/README.md)

`llm_based_abm/current/` 只保留最新版。当前实验使用2023年7—10月数据，正常天气区分暑假和学期，四个强降雨事件按时间滚动测试。四个代表站点、逐日预测、Lite模型；报告准确率、群体行为和实际API成本。

`data/` 是小组共享数据，`rule_based_abm/` 是协作者工作区。LLM组在llm分支开发，阶段完成后同步main，无需PR。密钥不上传。

历史代码与结果保留在Git提交中；此前V1—V5完整目录可在[7f973e8](https://github.com/zhangcy19/nyc-subway-abm/tree/7f973e8018e0cec16dcebfbaf9f098f4fc280a07/llm_based_abm)浏览。删除当前目录副本不删除Git历史。
