# test_1 测试集

共 200 条，全部使用项目现有 `text` + `tasks` 标注格式，只用于评估，不参与训练。

- `single_intent.jsonl`：50 条，每题一个任务，包含两种业务意图。
- `same_intent_multi_parameter.jsonl`：50 条，每题两个任务，意图相同、地区/时间/蚊种等参数不同。
- `two_different_intents.jsonl`：50 条，每题一个数量查询和一个地区排名查询。
- `mixed_intents.jsonl`：50 条，每题 3 或 4 个任务，两种意图交错出现，覆盖重复意图和混合意图。

覆盖实体按当前 `schema.py` 支持范围：省、市、区、街道、时间、蚊种、性别、排名层级和排名数量。句子中的“每天/按天”作为统计粒度出现；当前 schema 没有 `time_unit`，因此不标为实体。
