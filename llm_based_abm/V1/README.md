# V1：固定天气策略

12站，固定天气策略，比较规则、LLM zero/few-shot和代理数量。流程见 [PROTOCOL.md](PROTOCOL.md)，结果见 [RESULTS.md](RESULTS.md)。这版代理是需求群体代表，没有逐个乘客的自主决策。

在本文件夹使用 dev：

```bash
python prepare_experiment.py
python -m unittest test_model.py
python run_experiment.py --offline
python analyze.py --extended
python report_results.py
```

源数据从 ../../data/ 读取，密钥从 ../key.txt 读取。offline 使用 results/api_calls.jsonl 的真实响应，不调用API。experiment_data/panel.csv 由 prepare 重建。results 保存预测、行动、统计及图表。移除 offline 才会为未缓存请求消费额度。

票种是人群代理，老人/残障源标签合并；不能由概率推断真实取消或换方式比例。
