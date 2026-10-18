"""训练、验证并保存联合 NLU 模型。"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import config
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from dataset import build_schema, collate_batch, encode_samples, load_samples
from model import JointNLU, build_intent_weights, compute_loss, load_model, save_model
from predict import predict


def task_key(task):
    return (task['intent'], tuple(sorted((e['type'], e['start'], e['end']) for e in task['entities'])))


def evaluate(model, tokenizer, samples, device):
    # 独立预测，不向模型提供正确意图；完整任务匹配包含实体归属
    exact = 0
    intent_exact = 0
    for sample in samples:
        predicted = predict(sample['text'], model, tokenizer, device)['tasks']
        exact += sorted(map(task_key, predicted)) == sorted(map(task_key, sample['tasks']))
        intent_exact += sorted(t['intent'] for t in predicted) == sorted(t['intent'] for t in sample['tasks'])
    return {'task_exact_match': exact / len(samples), 'intent_exact_match': intent_exact / len(samples)}


def read_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, default=config.TRAIN_DIR)
    parser.add_argument('--validation-dir', type=Path, default=config.VALIDATION_DIR)
    parser.add_argument('--output-dir', type=Path, default=config.OUTPUT_DIR)
    parser.add_argument('--model', default=config.MODEL_NAME)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'auto'], default='mps')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--head-learning-rate', type=float, default=1e-3,
                        help='新建预测层的学习率，与预训练 MacBERT 分开设置')
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--max-tasks', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke-test', action='store_true', help='只训练一个批次并验证保存加载，不评估识别效果')
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.max_length, args.max_tasks) < 1
            or not 0 < args.learning_rate < 1 or not 0 < args.head_learning_rate < 1):
        parser.error('轮数、批次大小、长度和任务数必须为正数，学习率必须在 0 与 1 之间。')
    if args.smoke_test and args.output_dir == config.OUTPUT_DIR:
        args.output_dir = config.ROOT / 'models/joint_nlu_smoke'
    return args


def main():
    # 1. 读取配置并固定随机种子
    args = read_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('输出目录非空，请使用新的 --output-dir，避免覆盖已有模型。')

    # 2. 读取全部训练标注，验证集暂未准备好时允许先试训
    samples = load_samples(args.train_dir)
    validation = [] if args.smoke_test else load_samples(args.validation_dir, allow_empty=True)
    if {s['text'] for s in samples} & {s['text'] for s in validation}:
        raise ValueError('训练集与验证集存在重复问题。')
    schema = build_schema(samples, args.max_tasks, args.max_length)
    device = config.resolve_device(args.device)
    print(f'设备：{device}；训练样本：{len(samples)}；验证样本：{len(validation)}', flush=True)
    if not validation and not args.smoke_test:
        print('未提供验证集：将完整训练所有数据并保存最后一轮，不报告验证准确率。', flush=True)

    # 3. 加载分词器，检查标签和实体位置是否能用于训练
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    encoded = encode_samples(samples, tokenizer, schema)
    if validation:
        encode_samples(validation, tokenizer, schema)
    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    intent_weights = build_intent_weights(encoded, len(schema['intents'])).to(device)
    weight_config = dict(zip(schema['intents'], intent_weights.cpu().tolist()))
    print(f'意图训练权重：{weight_config}', flush=True)

    # 4. 加载预训练骨干和新建的预测层
    model = JointNLU(AutoModel.from_pretrained(args.model), schema).to(device)
    # 预训练骨干小步调整，新建的随机预测层需要更快学习
    head_parameters = [parameter for name, parameter in model.named_parameters()
                       if not name.startswith('encoder.')]
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.learning_rate},
        {'params': head_parameters, 'lr': args.head_learning_rate},
    ])
    print(f'学习率：MacBERT={args.learning_rate}，新预测层={args.head_learning_rate}', flush=True)
    fingerprint = hashlib.sha256(json.dumps(samples, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update({'training_data_sha256': fingerprint, 'training_samples': len(samples),
                     'device': str(device), 'intent_loss_weights': weight_config})
    best_score = -1

    # 5. 每轮处理全部批次；有验证集选最优模型，没有则保存最后一轮
    epochs = 1 if args.smoke_test else args.epochs
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            batch = {'inputs': {k: v.to(device) for k, v in batch['inputs'].items()},
                     'intents': batch['intents'].to(device), 'entities': batch['entities'].to(device)}
            optimizer.zero_grad()
            loss = compute_loss(model(batch['inputs'], batch['intents']), batch, intent_weights)
            if not torch.isfinite(loss):
                raise ValueError('训练损失出现非有限值，停止保存。')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if args.smoke_test:
                break
        metrics = evaluate(model, tokenizer, validation, device) if validation else {}
        print(f'第 {epoch + 1}/{epochs} 轮：完成 {len(losses)} 批，loss={sum(losses) / len(losses):.4f}', flush=True)
        if validation:
            print(f'验证结果：{metrics}', flush=True)
        should_save = metrics['task_exact_match'] > best_score if validation else epoch == epochs - 1
        if should_save:
            best_score = metrics.get('task_exact_match', best_score)
            save_model(model, tokenizer, args.output_dir, {
                **metadata, 'epoch': epoch + 1, 'metrics': metrics,
                'batches_per_epoch': len(losses), 'validation_samples': len(validation),
            })

    # 6. 从磁盘重新加载，确认保存的模型可以预测
    saved = load_model(args.output_dir, device)
    saved_tokenizer = AutoTokenizer.from_pretrained(args.output_dir, local_files_only=True, use_fast=True)
    if not args.smoke_test:
        training_metrics = evaluate(saved, saved_tokenizer, samples, device)
        print(f'训练集拟合检查（不代表新问题准确率）：{training_metrics}', flush=True)
    result = predict(samples[0]['text'], saved, saved_tokenizer, device)
    print(f'模型已保存并重新加载：{args.output_dir}', flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.smoke_test:
        print('流程检查通过；以上预测不代表模型识别效果。')


if __name__ == '__main__':
    main()
