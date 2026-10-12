# 文献与实验设计依据

核查日期：2026-10-10。Zotero 的 `UPAD 7001`（集合 `Z32ZE5D7`）当前有 6 篇。它们主要讨论因果关系、证据和交通行为，不是 LLM-ABM 方法论文。因此下面分开说明：哪些是用户已有文献，哪些是补充的方法参考，以及本实验实际借用了什么。

## 用户已有的 6 篇

| 文献 | 本次读取范围 | 对本项目的实际作用 |
|---|---|---|
| Wu et al. (2020), *Air pollution and COVID-19 mortality in the United States: Strengths and limitations of an ecological regression analysis* ([DOI](https://doi.org/10.1126/sciadv.abd4049)) | Zotero 元数据与摘要 | 区域总量不能直接推出个体机制。本项目可以检验站点客流，但不能由此确认某个具体乘客是否取消了出行。 |
| Duranton & Turner (2011), *The Fundamental Law of Road Congestion: Evidence from US Cities* ([DOI](https://doi.org/10.1257/aer.101.6.2616)) | Zotero 元数据与摘要 | 交通需求会对供给变化作出反应。它是道路长期研究，不能直接提供纽约暴雨的客流参数。 |
| Little (1995), *Objectivity, Truth, and Method: A Philosopher’s Perspective on the Social Sciences* ([作者原文](https://www-personal.umd.umich.edu/~delittle/anthnew3.pdf)) | Zotero 元数据；作者公开原文关于实证检验的段落 | 对代理的行为解释也应提出可观察、可检验的预测。模型自己说“我担心暴雨”，不是验证通过。Zotero 日期为空，1995 来自原文标题页。 |
| Næss (2015), *Built Environment, Causality and Travel* ([DOI](https://doi.org/10.1080/01441647.2015.1017751)) | Zotero 元数据与摘要 | 写清“天气→出行决策→进站客流”的假设机制；相关性、预测效果和因果识别分别报告。 |
| Hoeken (2001), *Anecdotal, Statistical, and Causal Evidence: Their Perceived and Actual Persuasiveness* ([DOI](https://doi.org/10.1023/A:1012075630523)) | Zotero 摘要；大学公开论文首段及作者大学出版列表 | 区分有说服力的故事和实证结果。展示代理理由时同时给客流误差；本文研究论证说服力，不是 LLM 准确性的直接证据。 |
| Handy, Cao & Mokhtarian (2005), *Correlation or causality between the built environment and travel behavior? Evidence from Northern California* ([DOI](https://doi.org/10.1016/j.trd.2005.05.002)) | Zotero 元数据与摘要 | 偏好、自选择和环境可能共同影响出行。本项目不能把全部客流变化都归因于天气，也不能把设定的人物性格当作真实人口属性。 |

以上不是全面系统综述，也未完整精读全部 6 篇。表中“对项目的作用”是本项目的方法判断，不表示论文直接研究了地铁暴雨或证明 LLM 更好。

## 补充的直接方法参考

1. Park et al. (2023), *Generative Agents: Interactive Simulacra of Human Behavior* ([论文](https://arxiv.org/abs/2304.03442), [正式出版](https://doi.org/10.1145/3586183.3606763))。摘要与论文资料提出观察、记忆、计划和反思组成的代理架构，并展示 25 个代理的小镇行为。可借鉴稳定人物设定和情境输入；“行为看起来可信”不等于真实客流预测准确，不据此预设 LLM 胜过规则模型。
2. Liu, Yang & Yin (2025), *LLM-ABM for Transportation: Assessing the Potential of LLM Agents in System Analysis* ([原文](https://arxiv.org/html/2503.22718v1))。已读取框架、两组实验和局限段落。40 个代理完成通勤出发时间与路径选择，系统结果对照理论基准，并让交通状态反馈影响后续决策。对我们最直接的启发是同时检验行为和系统总量；论文没有完成纽约极端天气的实证验证。
3. Liu et al. (2025), *Aligning LLM agents with human learning and adjustment behavior: a dual agent approach* ([原文](https://arxiv.org/html/2511.00993v1))。已读取摘要、数据划分与实验段落。用真实日际路径选择实验校准人物设定，前 80 轮训练、后 80 轮评价。可借鉴只用过去数据校准和严格留出测试；本项目没有同等级个体轨迹，不能声称实现了真实个体行为对齐。

## MUDT5011 收藏中的补充材料

另以只读方式核查了 `MUDT5011`（`S98NDV5N`），当前有 10 条记录，其中包含直接相关的 LLM 文献与代码。上面 Liu et al. 的 transportation benchmark 已在该集合中；其他值得参考的是：

- Liu, Yang & Yin (2025), *Toward LLM-agent-based modeling of transportation systems: A conceptual framework* ([期刊原文](https://doi.org/10.1016/j.ait.2025.100001), [预印本](https://arxiv.org/abs/2412.06681))。核查摘要和关于行为对齐、混合框架的段落。支持把人物、感知、决策、行动和传统模拟器分开，但不支持仅凭角色提示就认定代理已与真实人群一致。
- Yifan Liu et al., *Human Mobility Modeling with Limited Information via Large Language Models*。Zotero 保存的是旧标题网页；[当前 arXiv v2](https://arxiv.org/abs/2409.17495v2) 标题为 *Human Mobility Modeling with Household Coordination Activities under Limited Information via Retrieval-Augmented LLMs*。本次核查摘要：利用公共统计和社会人口信息生成含家庭协调的活动链，以 NHTS 和 SCAG-ABM 数据验证。可用于后续扩展出行链，首轮站点需求实验尚未实现这些功能。
- Li, Yang & Zhou, *Exploring dissatisfaction in bus route reduction through LLM-calibrated agent-based modelling* ([DOI](https://doi.org/10.1016/j.tranpol.2026.104431))。本次只读取 Zotero 元数据及摘要：使用刷卡和线路/站点空间数据分析公交削减下不同票种群体的模拟不满。可参考群体分层与供给扰动，不能把“模拟不满”当成已观测心理状态。Zotero 日期为 2026-12-01，晚于本次核查日期；未据此推断正式出版状态。
- `qiliuchn/gatsim` 是集合中的软件记录，相关 [GATSim 论文](https://doi.org/10.1016/j.simpat.2025.103234) 是潜在框架参考。本次未审计其代码，不声称已复用该框架。

## 本实验建议采用的流程

- **任务边界**：在给定天气条件下预测站点小时客流。使用当天实际观测天气时，这是历史条件预测；要声称提前预报，需要换成预测时点可获得的天气预报和客流信息。
- **人口表示**：每个代理代表一类出行者及其份额，份额来自训练期票种客流。年龄、收入、性格、工作弹性等没有逐人观测，明确作为人物假设。学生票可单列；来源把老人和残障合并时，实证评价继续使用合并标签，二者分别设定只能做情景探索。
- **输出与聚合**：代理输出有限行为选项的概率，程序检查概率和并按份额聚合。正常客流基准提供规模，LLM 提供天气条件下的行为变化。换站需求应从原站扣除并加入目标站；取消和改用其他方式都减少地铁需求，但数据无法验证二者的具体比例。
- **对照组**：相同训练窗口、测试记录和正常客流基准下，至少比较季节/星期/小时基准、规则天气模型和 LLM 群体代理。若增加 LLM 直接总量预测，可判断优势是否真的来自代理结构。
- **zero-shot / few-shot**：zero-shot 没有带答案的情境示例，但可以共享训练期正常基准和人物份额；few-shot 只加入训练窗口内的少量真实情境及其客流变化。模拟的个人选择不能伪装成真实 few-shot 标签。
- **5 / 10 / 20 个代理**：总人口权重和正常需求保持不变，新增代理只细分人物类型。5→10→20 比较的是表示复杂度和敏感性，不是人口翻倍。可先固定人物组，再增加组内性格差异，保留相同基准份额。
- **时间隔离**：先确定训练、验证和测试窗口，再锁定提示词及规则参数。测试客流不能进入正常基准、few-shot 示例、缓存键所引用的示例库或人格校准。测试期间若没有设定在线学习，不把真实测试客流反馈给下一时刻。
- **历史事件记忆**：纽约 2023 年 9 月暴雨是公开事件，LLM 可能在预训练中见过。使用匿名站点编号、相对时间、星期/小时和数值天气，尽量不提供真实日期、站名或新闻描述；此举减轻风险，不能证明训练污染完全消失。需要保留该局限，最好追加近期事件复验。
- **评价与成本**：先记录站点/群体 MAE、WAPE、带符号偏差，分别报告干天、湿天和极端天，避免低客流小时的百分比误差失真。使用固定配置、少量重复，保存 API 真实响应、模型名、提示词版本、token 用量；LLM 没有执行成功的记录不能填成模拟结果。
- **反事实探索**：在相同正常需求下改变雨强、站点关闭或人物设定，观察决策和客流变化。若没有相应实测标签，图题和结论明确写“情景模拟”。解释文本用于了解模型如何反应，数值评价用于判断预测效果。

## 首轮实现的结论边界

已核查 `PROTOCOL.md`、`prepare_experiment.py`、`model.py`、`run_experiment.py` 的机制。正式运行期间不根据测试误差调整策略；下列区别应在解释结果时保留：

- 当前是 **LLM 生成策略、Mesa 执行持久群体状态** 的模型。API 先生成 8 个天气/时段情境下的概率，跨站点复用；每小时执行固定策略并维护推迟需求，不是逐小时重新请求 LLM，也没有学习闭环、线路拥堵或社会传播。
- 地图上的站点差异来自正常需求、票种组合及推迟状态。LLM 不接收站点空间特征，不能把空间误差图解读成模型学到了地理差异。
- 无雨策略被程序统一置为继续原计划，这有助于保持相同基准；因此干天合理响应是实验约束，不能作为 LLM 自发学到正常行为的证据。
- 直接预测允许湿天乘数超过 1；群体模型的新计划需求由继续/推迟/离开地铁分配，不从其他方式吸引新需求，只可能因前一小时推迟回流而使某小时客流增加。因此二者的输出可行范围不完全相同，结果比较同时包含结构差异。
- 代理数量变化同时改变人物灵活度的离散表示，并使用各配置独立的 API 策略样本。即使平均灵活度和人口份额固定，也不能把误差变化全归因于代理数量本身。
- 60% 的推迟回流率是假设；0%/100% 是敏感性情景。未回流部分及测试终点的待出行需求没有真实标签支撑，不代表观测到的取消率。
- 站点在训练期已经出现；留出的是未来时间窗口，并非从未见过的站点。可以说时间留出检验，不宜称未见站点泛化。

这些是实验设计依据；实际执行范围、配置和结果以实验说明及保存的运行结果为准。
