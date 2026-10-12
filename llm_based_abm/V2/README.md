# V2：动态群体客流修正

12个新站点，每天零点预测24小时。LLM读取历史、数值基线和天气，输出群体倍率与概率；不是独立乘客画像决策。

先看 [RESULTS.md](RESULTS.md)，再看 [DIAGNOSIS.md](DIAGNOSIS.md)、[PROTOCOL.md](PROTOCOL.md)、[LITERATURE.md](LITERATURE.md)。

在本文件夹使用 dev：

```bash
python v2.py prepare
python -m unittest test_v2.py
python v2.py test --offline
python v2_analyze.py
```

源数据为 ../../data/；排除V1站点读取 ../V1/experiment_data/stations.csv。密钥在 ../key.txt。results/api_calls.jsonl.gz 是真实响应缓存；predictions.csv.gz 是最终预测；panel 重建、不上传。

使用实测天气、旧数据回测，不是外部盲测。部分检索群体缺失口径和critic/统计校准容量差异已写入结果。路径迁移不改变提示或预测算法。
