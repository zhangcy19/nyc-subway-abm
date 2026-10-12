# V3：让 LLM 扮演虚拟乘客

同一批合成乘客分别用规则或LLM作个人选择，再加权得到站点预测。人物有固定画像、当天计划和最近三次自己的模拟经历；每个HTTP请求只有一个人的信息。

设计见 [PROTOCOL.md](PROTOCOL.md)，文献见 [LITERATURE.md](LITERATURE.md)，结果见 [RESULTS.md](RESULTS.md)。

在本文件夹使用 dev：

```bash
python prepare.py
python -m unittest test_model.py
python run.py feedback --offline
python run.py test --offline
python run.py checks --offline
python run.py evaluate
python analyze.py
```

源数据从 ../../data/ 读取，来源与格式见 [数据说明](../../data/README.md)。密钥只放 ../key.txt、不上传。去掉 offline 为未缓存的个人决策调用API。python run.py pilot 单独检查接口。Pro/Lite 分开保护额度；在线阶段依次运行，不同时启动多个进程。

| 文件 | 用途 |
|---|---|
| prepare.py | 过去数据的计划需求、选站与合成人物 |
| model.py | Mesa乘客、个人感知、规则、约束、执行与记忆 |
| run.py | 单人请求、真实响应缓存、时间窗口与配对检查 |
| analyze.py | 行为汇总、客流误差、图表和报告 |
| res/personas_*.csv | 24/48/96分辨率的人物与权重 |
| res/decisions.csv.gz | 主测试每人每天行动、理由、权重、入站时刻 |
| res/predictions.csv.gz | 汇总站点小时预测与真实客流 |
| res/matched_checks.csv.gz | 配对无记忆、少样本和重复检查 |
| res/resolution_decisions.csv.gz | 固定一天的人数敏感性 |
| res/api_calls.jsonl.gz | 真实请求/响应/用量，支持离线重放 |
| res/reproducibility.json | 重建与缓存重放的逐值比较、测试和环境版本 |
| res/figures/ | 图表 |

panel.csv.gz 是可重建中间表，不上传。weight 代表计划进站机会，action 是单项选择，delay_min 是允许的出发延迟，reason 是个人简短说明。当天某票种计划量为零时，该组没有出行或 API 请求。actual_entry_timestamp 仅地铁有值；其他方式不计入本站。block 区分独立三日窗口，不推断跨空缺日期连续历史。

职业、性格、个人行程和经历都是合成假设。真实客流验证汇总预测，不验证画像/理由是否真实。模型只有原计划地铁乘客，尚不能生成其他方式转来的新增地铁需求。
