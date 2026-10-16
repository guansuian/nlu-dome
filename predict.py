"""从本地加载多标签模型，预测一个或多个业务意图，无匹配时返回 unknown。"""

import argparse
import json
from pathlib import Path

import config
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict(text, model, tokenizer, device, threshold=None):
    text = text.strip()
    if not text:
        raise ValueError("问题不能为空。")
    if model.config.problem_type != "multi_label_classification":
        raise ValueError("当前模型不是多标签模型，请先重新运行 train.py。")
    if threshold is None:
        threshold = model.config.intent_threshold
    if not 0 < threshold < 1:
        raise ValueError("意图阈值必须大于 0 且小于 1。")
    inputs = tokenizer(
        text, truncation=True,
        max_length=model.config.intent_max_length, return_tensors="pt",
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(**inputs).logits)[0]
    scores = probabilities.cpu().tolist()
    intents = [model.config.id2label[i] for i, score in enumerate(scores) if score >= threshold]
    return {
        "text": text,
        "intents": intents or ["unknown"],
        "scores": {model.config.id2label[i]: round(score, 6) for i, score in enumerate(scores)},
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="指定问题；省略时进入交互模式")
    parser.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    parser.add_argument("--threshold", type=float, help="覆盖模型保存的阈值，必须在 0 与 1 之间")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if not (args.model_dir / "config.json").is_file():
        parser.error("未找到训练后的模型，请先运行 train.py。")
    if args.text is not None and not args.text.strip():
        parser.error("问题不能为空。")
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("意图阈值必须大于 0 且小于 1。")
    device = config.resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True,
    ).to(device)
    model.eval()
    if model.config.problem_type != "multi_label_classification":
        parser.error("当前模型不是多标签模型，请先重新运行 train.py。")

    if args.text is not None:
        print(json.dumps(predict(args.text, model, tokenizer, device, args.threshold), ensure_ascii=False, indent=2))
        return
    print(f"模型已加载（{device}）。输入问题开始预测，输入 /exit 退出。")
    while True:
        try:
            text = input("问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text == "/exit":
            break
        if text:
            print(json.dumps(predict(text, model, tokenizer, device, args.threshold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
