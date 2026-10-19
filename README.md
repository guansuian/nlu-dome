# 蚊虫监测 NLU：Ollama 本地模型

当前分支 `codex/local-model-nlu` 使用 Ollama 的 `qwen3:4b-q4_K_M`，一次请求完成意图识别、查询任务划分及实体提取。通过提示词指定业务规则，不训练或修改模型权重。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `config.py` | 本地服务地址、模型名称、超时和生成参数 |
| `prompts/nlu.txt` | 两种意图、任务划分、共享条件和实体标注规则 |
| `schema.py` | Pydantic 输出结构、实体位置和意图字段校验 |
| `nlu.py` | 核心函数 `parse(text)`，调用 Ollama 并校验结果 |
| `predict.py` | 交互输入或单次预测 |
| `evaluate.py` | 递归读取测试集、逐题对比并生成报告 |
| `tests/test_local_nlu.py` | 无需模型服务的代码测试 |

Python 端直接调用官方 Ollama 客户端，不启动 `ollama run` 子进程。项目文件名不要命名为 `ollama.py`，以免遮蔽官方包。

## 安装与启动

已有项目虚拟环境可直接使用。新环境推荐 Python 3.11：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

先打开 Ollama 应用，确认 `ollama list` 中存在 `qwen3:4b-q4_K_M`。应用已运行时，无需再启动一个 `ollama serve`。Ollama 管理 Qwen3 的模型文件，不需要放进项目的 `models/`。

在本机安装依赖时若遇到证书校验失败，可以使用 macOS 系统证书包：

```bash
.venv/bin/python -m pip install --cert /etc/ssl/cert.pem -r requirements.txt
```

默认配置：本地地址 `http://localhost:11434`、`think=False`、`stream=False`、`temperature=0`、随机种子 42、上下文 8192 tokens、最多生成 4096 tokens、请求超时 180 秒。低温度和固定种子不代表不同机器或服务版本下的结果必然一致。

本机接口使用直连，不读取系统代理设置；本机实测启用环境代理会返回 HTTP 502。输入最多 500 个字符，超长输入明确报错，不静默截断。修改配置后重新运行脚本。

## 交互预测

在 PyCharm 中直接运行 `predict.py` 的 `main()` 即可，无需设置参数。也可以执行：

```bash
/Applications/pythoncunfangchu/nlu-demo/.venv/bin/python /Applications/pythoncunfangchu/nlu-demo/predict.py
```

连续输入问题，空行跳过，输入 `/exit` 退出。每个问题独立解析，不保存多轮上下文。首次请求可能需要加载模型。单次输出无效后仍可继续输入，Ctrl+C 或输入结束也可以退出。

单次预测：

```bash
.venv/bin/python predict.py --text '厦门市昨天监测到多少只白纹伊蚊？'
```

业务代码只需要调用：

```python
from nlu import parse

result = parse('今天青岛市有多少蚊虫？昨天郑州市有多少蚊虫？')
```

返回格式保持与原标注一致：

```json
{
  "text": "厦门市昨天监测到多少只白纹伊蚊？",
  "tasks": [
    {
      "intent": "species_compose_report",
      "entities": [
        {"type": "city", "text": "厦门市", "start": 0, "end": 3},
        {"type": "time", "text": "昨天", "start": 3, "end": 5},
        {"type": "species", "text": "白纹伊蚊", "start": 11, "end": 15}
      ]
    }
  ]
}
```

无关问题返回 `tasks: []`。连接失败、超时或模型不存在抛出 `RuntimeError`；输入、JSON 结构或实体位置无效抛出 `ValueError`，不会把失败伪装成无关问题。实体提取成功也不意味着业务参数已齐全；本项目暂不执行后台查询、日期标准化、必填项追问或接口参数转换。

## 输出检查

模型通过 JSON Schema 返回任务及实体，结构定义使用 Pydantic。支持 `species_compose_report`、`region_rank_report` 两种意图；同意图的多个任务分别保留，没有旧模型固定 4 个任务位置的限制，实际仍受上下文和生成长度限制。

实体沿用 `province`、`city`、`district`、`street`、`time`、`species`、`sex`、`rank_level`、`top_n`。`rank_level` 和 `top_n` 只能属于排名任务。字段遵循现有 NLU 标注，并非直接对应后台 API 参数。

字符位置从 0 开始，包含起点、不包含终点；输入的空格和换行保留。模型负责识别实体原文和归属：

- 实体原文只出现一次时，代码用原文精确匹配计算位置，避免让小模型数错字符。
- 实体原文出现多次时，保留模型给出的位置，并检查对应片段；位置错误明确报错，不默认选择第一次出现。
- 实体原文不存在、位置越界或同一任务重复列出同一实体时，明确报错。
- 不补造地名，不纠正模型选择的实体种类或任务归属。即使片段匹配，语义仍可能识别错误，需要测试集检查。

原始模型输出与定位后的业务输出可能不同，评估统计的是完整 `parse()` 流程的结果。低参数量模型仍可能分错任务、漏实体或选择错误的同名位置，结构化输出不能保证语义正确。

## 批量评估

直接运行 `evaluate.py`，默认递归读取 `data/test/` 的全部 142 条数据，逐题打印进度，保存新的带时间戳 JSON 报告到 `reports/`。不会读取训练集进行训练，也不会把全量标注发给模型；每次请求只发送业务规则、输出结构和当前问题。

```bash
.venv/bin/python evaluate.py
```

先检查前 5 条流程（按文件路径及文件内顺序选取，不代表整体效果）：

```bash
.venv/bin/python evaluate.py --limit 5
```

单独评估 v10 新增的 64 条测试题：

```bash
.venv/bin/python evaluate.py --data-dir data/test/multitask_context
```

测试数据同时支持标准 JSONL 和项目现有的展开 JSON 对象格式。加载时先检查结构、实体位置及重复问题。报告记录数据、提示词和输出结构的指纹，以及模型名称、调用参数、每题答案、预测和错误。

评估的三个指标为：任务数量正确、任务数量及意图列表正确、完整任务正确。完整正确要求每项意图、实体类型、字符位置及任务归属全部正确；忽略任务顺序但保留同意图任务数量，与 v9/v10 完整匹配口径一致。

调用或校验失败计为错误，不能算未知问题拒识成功。格式/位置错误继续下一题；服务故障则停止，保存已完成部分并标记 `completed=false`。Ctrl+C 也保存已完成部分。原报告不覆盖；报告中的 `limited_run` 表示是否只选取了部分题目。

已完成的 7 道人工流程测试结果见 `reports/ollama_qwen3_implementation_smoke.json`：3 条完整正确，2 条存在实体类型或任务分组错误，2 条因实体位置无效被校验拦截。该报告用于实现检查，并非对原有 142 条测试集的正式评估，也不能作为与 v9/v10 的准确率对比。

## 代码测试

无需启动 Ollama，不向业务数据添加样本：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

检查共享实体、重复名称的字符位置、意图字段限制、输出截断、无效响应、交互错误处理及评估失败计数。

## 保留资源

`models/` 中的所有旧模型、`data/` 全部训练/验证/测试/待整理数据、历史 `reports/`、`.cache/` 和原虚拟环境均保留。模型、缓存和虚拟环境不纳入 Git。

数据目录内的旧说明和历史报告保留原样，其中的训练命令仅作为历史记录。旧联合训练方案代码保存在 `codex/joint-nlu-shared-encoder` 分支，提交 `dfaeff8`。

官方参考：[Python 客户端](https://github.com/ollama/ollama-python)、[结构化输出](https://docs.ollama.com/capabilities/structured-outputs)、[思考模式参数](https://docs.ollama.com/capabilities/thinking)。
