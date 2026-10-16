"""读取 CSV，微调中文多标签意图分类模型，并保存到本地。"""

import argparse
import csv
from pathlib import Path

import config  # 必须先设置缓存位置，再导入 transformers。
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


def load_samples(path):
    samples = []
    seen = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or set(reader.fieldnames) != {"text", "label"}:
            raise ValueError("CSV 表头必须为 text,label。")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV 第 {reader.line_num} 行的列数不正确。")
            text, label = row["text"].strip(), row["label"].strip()
            if not text or not label:
                raise ValueError(f"CSV 第 {reader.line_num} 行的 text 和 label 不能为空。")
            parts = [part.strip() for part in label.split("|")]
            if any(not part for part in parts) or len(parts) != len(set(parts)):
                raise ValueError(f"CSV 第 {reader.line_num} 行有空标签或重复标签。")
            if "unknown" in parts and len(parts) != 1:
                raise ValueError("unknown 不能与业务意图同时标注。")
            # unknown 表示所有业务意图均不存在，不作为独立输出类别。
            labels = () if parts == ["unknown"] else tuple(sorted(parts))
            if text in seen and seen[text] != labels:
                raise ValueError(f"同一句话对应了不同标签：{text}")
            seen[text] = labels
            samples.append((text, labels))
    labels = sorted({label for _, names in samples for label in names})
    if len(labels) < 2:
        raise ValueError("训练数据至少需要两个不同的业务意图标签（不含 unknown）。")
    return samples, {label: index for index, label in enumerate(labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.TRAIN_FILE)
    parser.add_argument("--model-name", default=config.MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=config.MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--threshold", type=float, default=config.PREDICTION_THRESHOLD)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=config.TRAIN_DEVICE)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_length) <= 0 or args.learning_rate <= 0:
        parser.error("轮数、批次大小、最大长度和学习率必须大于 0。")
    if not 0 < args.threshold < 1:
        parser.error("意图阈值必须大于 0 且小于 1。")

    samples, label2id = load_samples(args.data)
    # 设置随机种子
    set_seed(config.SEED)
    # 确定训练设备
    device = config.resolve_device(args.device)
    print(f"训练样本：{len(samples)}；标签：{label2id}；设备：{device}", flush=True)
    print(f"加载预训练模型：{args.model_name}（首次运行需要下载）", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label={index: label for label, index in label2id.items()},
        problem_type="multi_label_classification",
    ).to(device)
    if args.max_length > model.config.max_position_embeddings:
        parser.error(f"最大长度不能超过模型上限 {model.config.max_position_embeddings}。")

    def collate(batch):
        texts, labels = zip(*batch)
        inputs = tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        )
        # 每个业务意图一个位置；unknown 为全 0，同时命中两个意图则为全 1。
        targets = torch.zeros((len(batch), len(label2id)), dtype=torch.float32)
        for row, names in enumerate(labels):
            for name in names:
                targets[row, label2id[name]] = 1.0
        inputs["labels"] = targets
        return inputs

    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失不是有限值，已停止训练。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            print(f"轮次 {epoch + 1}/{args.epochs}，批次 {step}/{len(loader)}，loss={loss.item():.4f}", flush=True)
        print(f"本轮平均 loss={total_loss / len(samples):.4f}", flush=True)

    # 标签映射和训练时的截断长度随模型一起保存，预测时自动读取。
    model.config.intent_max_length = args.max_length
    model.config.intent_threshold = args.threshold
    model.config.intent_training_device = str(device)
    model.config.intent_training_samples = len(samples)
    model.config.intent_training_epochs = args.epochs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu").save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    # 新模型不能沿用上一次训练的测试结果。
    (args.output_dir / "test_results.json").unlink(missing_ok=True)
    print(f"模型、分词器和标签映射已保存：{args.output_dir.resolve()}")
    print("当前仅验证训练流程；识别效果需要使用独立测试集评估。")


if __name__ == "__main__":
    main()
