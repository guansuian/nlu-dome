"""加载本地模型，输出任务和实体原文。"""
import argparse
import json

import config
import torch
from transformers import AutoTokenizer
from model import decode_tasks, load_model


def predict(text, model, tokenizer, device):
    # 1. 检查输入并保留实体所需的字符位置
    if not isinstance(text, str):
        raise ValueError('问题必须是文本。')
    if not text.strip():
        raise ValueError('问题不能为空。')
    try:
        text.encode('utf-8')
    except UnicodeEncodeError as error:
        raise ValueError('输入中包含无效编码字符，请清空当前输入并重新输入问题。') from error
    tokens = tokenizer(text, return_offsets_mapping=True, return_tensors='pt', truncation=False)
    offsets = tokens.pop('offset_mapping')[0].tolist()
    if len(offsets) > model.schema['max_length']:
        raise ValueError('问题超过模型支持的长度，请拆分查询。')
    inputs = {key: value.to(device) for key, value in tokens.items()}
    # 2. 一次编码完成任务意图和实体预测
    model.eval()
    with torch.inference_mode():
        intents, entities = model(inputs)
    tasks = decode_tasks(text, offsets, intents[0], entities[0], model.schema)
    result = {'text': text, 'tasks': tasks}
    if not tasks:
        result['message'] = '请说明你想查询什么蚊虫监测信息。'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default=str(config.OUTPUT_DIR))
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--text', help='指定时预测一次；不指定则进入交互输入')
    args = parser.parse_args()
    # 1. 启动时加载一次模型，后续问题复用模型和分词器
    device = config.resolve_device(args.device)
    model = load_model(args.model_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    if args.text is not None:
        print(json.dumps(predict(args.text, model, tokenizer, device), ensure_ascii=False, indent=2))
        return

    # 2. 连续读取问题，输入 /exit 或按 Ctrl+C 退出
    print('模型已加载。请输入问题，输入 /exit 退出。')
    while True:
        try:
            text = input('问题：')
        except (EOFError, KeyboardInterrupt):
            print('\n已退出。')
            break
        if text.strip().lower() == '/exit':
            print('已退出。')
            break
        if not text.strip():
            continue

        # 3. 输出当前问题的结果，输入不合法时提示并继续
        try:
            result = predict(text, model, tokenizer, device)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except ValueError as error:
            print(f'输入错误：{error}')


if __name__ == '__main__':
    main()
