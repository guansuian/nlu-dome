# 蚊虫监测意图识别

使用 `hfl/chinese-macbert-base` 微调多标签文本分类模型，完成数据读取、模型加载、训练、保存和本地预测。一句话可以识别一个或两个业务意图，两个均未命中时返回 `unknown`。

## 安装依赖

在项目目录运行，已有 `.venv` 时直接使用：

```bash
.venv/bin/python -m pip install -r requirements.txt
```

首次使用且没有虚拟环境时，先运行 `python3 -m venv .venv`。建议使用 Python 3.11。

## 准备数据

`data/train.csv` 使用 UTF-8 编码，首行为字段名，后续每行为一条样本：

```csv
text,label
今天监测了多少只蚊虫？,species_compose_report
最近7天哪个省份蚊虫最多？,region_rank_report
统计今天蚊虫总数，再列出数量最多的五个省份。,species_compose_report|region_rank_report
我的父亲是谁？,unknown
```

标签自动从 CSV 中生成，不必在代码里手动维护。文本含英文逗号、引号或换行时，应按 CSV 规则转义，例如使用 Python 的 `csv.writer` 写入。

当前 `train.csv` 有 252 条人工编写样本，分为四种标注模式：

- `species_compose_report`：72 条，蚊虫数量与蚊种构成查询，含“蚊子”等口语说法。
- `region_rank_report`：60 条，省、市、区、街道等行政区域的蚊虫数量排名。
- `species_compose_report|region_rank_report`：60 条，同时包含两种业务需求，以 `|` 分隔。
- `unknown`：60 条，无关聊天、标点、不完整输入，以及当前尚未支持的查询。

`data/test.csv` 另有 60 条未参与训练的样本：两个单意图各 10 条、双意图 20 条、`unknown` 20 条。训练集与测试集没有完全相同的问句。

训练时只为两个业务意图建立输出位置。例如标签顺序为 `[region_rank_report, species_compose_report]` 时，排名为 `[1, 0]`，数量为 `[0, 1]`，双意图为 `[1, 1]`，`unknown` 为 `[0, 0]`。`unknown` 不能和业务标签同时标注。使用 `multi_label_classification` 和浮点多标签向量训练，模型内部采用 `BCEWithLogitsLoss`。

设备、趋势、环境、生境等需求目前也标为 `unknown`，因为当前只支持两个业务意图；未来增加对应意图时，需要重新标注这部分样本。用户已反馈的“你”“？”“我的父亲是谁？”已纳入训练，测试使用其他问法，不能把这三个已见样本的正确预测当作独立测试效果。

## 训练并保存

```bash
.venv/bin/python train.py
```

首次训练需要联网下载模型，默认缓存到 `.cache/huggingface/`。本机训练默认使用 Apple GPU（`mps`），由 `config.py` 中的 `TRAIN_DEVICE` 控制；不可用时会明确报错，不会自动回退 CPU。预测仍自动选择可用设备。可用 `--device cpu` 显式使用 CPU，或用 `--device cuda` 指定 NVIDIA GPU。

本机安装依赖时遇到 Python CA 证书配置问题，可使用系统证书：

```bash
.venv/bin/python -m pip install --cert /etc/ssl/cert.pem -r requirements.txt
REQUESTS_CA_BUNDLE=/etc/ssl/cert.pem HF_HUB_DISABLE_XET=1 .venv/bin/python train.py
```

模型完整缓存后，可用 `HF_HUB_OFFLINE=1 .venv/bin/python train.py --device mps` 离线进行 GPU 训练。训练日志会打印 `设备：mps`，保存的模型配置也记录训练设备、样本数和轮数。

默认训练 3 轮，批次大小为 2。路径、模型名称和默认训练参数在 `config.py` 中，也可通过命令行调整，例如：

```bash
.venv/bin/python train.py --epochs 5 --batch-size 2 --device cpu
```

训练结果保存在 `models/intent_classifier/`，包含模型权重、分词器和带标签映射的 `config.json`。再次训练会更新该目录；需要保留不同版本时，用 `--output-dir` 指定其他目录。

模型同时保存多标签分类模式和预测阈值。旧的单标签模型需要重新训练，预测脚本会拒绝将其作为多标签模型使用。

## 重新加载并预测

单次预测：

```bash
.venv/bin/python predict.py --text "哪个城市的蚊虫最多？"
```

交互式预测：

```bash
.venv/bin/python predict.py
```

输入 `/exit` 退出。预测只读取本地训练结果，不下载模型。自定义训练输出目录时，预测需通过 `--model-dir` 指定同一目录。

结果包含 `text`、意图列表 `intents`、每个业务意图的分数 `scores` 和阈值 `threshold`。例如双意图问题可能输出：

```json
{
  "text": "统计今天蚊虫总数，再列出数量最多的五个省份。",
  "intents": ["region_rank_report", "species_compose_report"],
  "scores": {
    "region_rank_report": 0.95,
    "species_compose_report": 0.97
  },
  "threshold": 0.5
}
```

预测使用 sigmoid，两个分数独立计算，不要求相加为 1。达到阈值的业务意图都会选中；均未达到时，`intents` 为 `["unknown"]`。不单独输出 unknown 分数，避免将它误当成经过模型预测的第三个概率。

默认阈值 `0.5` 是起始设置，尚未用独立验证集校准；训练时可用 `--threshold` 保存其他阈值，预测时也可用 `--threshold` 临时覆盖。例如 `.venv/bin/python predict.py --threshold 0.6`。分数不代表经过校准的正确概率，未知样本也不能保证覆盖所有无关问题。

代码、数据或模型更新后，请退出原来的交互进程，再重新运行 `predict.py`。

## 检查代码

```bash
.venv/bin/python -m unittest discover -s tests -v
```

覆盖多标签解析、未知样本、标签冲突、同时选中多个意图、阈值和旧模型检查。实际识别效果还需要加载训练结果，用未参与训练的数据检查。

## 当前验证范围

当前识别意图集合，不负责给每个意图绑定各自的地区、时间，也不执行工具调用。训练样本和测试样本都是人工编写的，测试集不参与训练；初步测试结果不能代表真实用户问题上的准确率。应分别检查单意图、双意图和未知问题的效果，并补充真实问句与独立验证集。
