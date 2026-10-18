"""读取标注，并将字符位置转换为模型使用的 token 标签。"""
import json
from pathlib import Path

import torch


def load_samples(directory, allow_empty=False):
    # 1. 读取目录及其子目录，支持标准 JSONL 和当前使用的换行展开格式
    samples = []
    decoder = json.JSONDecoder()
    for path in sorted(Path(directory).rglob('*.jsonl')):
        content = path.read_text(encoding='utf-8-sig')
        position = 0
        while position < len(content):
            if content[position].isspace():
                position += 1
                continue
            try:
                sample, position = decoder.raw_decode(content, position)
                validate_sample(sample)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f'{path}: {error}') from error
            samples.append(sample)
    if not samples and not allow_empty:
        raise ValueError(f'{directory} 没有已标注的数据。')
    # 2. 重复问题会影响采样权重，要求先整理数据
    texts = [sample['text'] for sample in samples]
    if len(set(texts)) != len(texts):
        raise ValueError('存在重复问题，请先合并或修正标注。')
    return samples


def validate_sample(sample):
    if not isinstance(sample, dict) or set(sample) != {'text', 'tasks'}:
        raise ValueError('每条样本必须包含且只包含 text、tasks。')
    if not isinstance(sample['text'], str) or not sample['text'].strip():
        raise ValueError('text 必须是非空字符串。')
    if not isinstance(sample['tasks'], list):
        raise ValueError('tasks 必须是列表；未知问题使用空列表。')
    for task in sample['tasks']:
        if not isinstance(task, dict) or set(task) != {'intent', 'entities'}:
            raise ValueError('每个任务必须包含 intent、entities。')
        if not isinstance(task['intent'], str) or not task['intent'].strip() or task['intent'] == '__none__':
            raise ValueError('intent 必须是业务标签，不能使用保留标签 __none__。')
        if not isinstance(task['entities'], list):
            raise ValueError('entities 必须是列表。')
        occupied = set()
        for entity in task['entities']:
            if not isinstance(entity, dict) or set(entity) != {'type', 'text', 'start', 'end'}:
                raise ValueError('实体必须包含 type、text、start、end。')
            start, end = entity['start'], entity['end']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(sample['text']):
                raise ValueError('实体位置无效。')
            if sample['text'][start:end] != entity['text']:
                raise ValueError('实体原文与位置不匹配。')
            if not isinstance(entity['type'], str) or not entity['type'].strip():
                raise ValueError('实体类型不能为空。')
            positions = set(range(start, end))
            if occupied & positions:
                raise ValueError('同一任务内暂不支持重叠实体。')
            occupied |= positions


def build_schema(samples, max_tasks, max_length):
    intents = sorted({task['intent'] for s in samples for task in s['tasks']})
    types = sorted({e['type'] for s in samples for t in s['tasks'] for e in t['entities']})
    if not intents or not types:
        raise ValueError('训练集需要包含业务意图和实体标注。')
    return {'intents': ['__none__'] + intents,
            'tags': ['O'] + [f'{prefix}-{kind}' for kind in types for prefix in ('B', 'I')],
            'max_tasks': max_tasks, 'max_length': max_length}


def encode_samples(samples, tokenizer, schema):
    # 1. 保留字符偏移；禁止静默截断，否则实体标注可能丢失
    encoded = []
    for sample in samples:
        if len(sample['tasks']) > schema['max_tasks']:
            raise ValueError('任务数超过 max_tasks，请增大配置。')
        tokens = tokenizer(sample['text'], return_offsets_mapping=True, truncation=False)
        offsets = tokens.pop('offset_mapping')
        if len(offsets) > schema['max_length']:
            raise ValueError(f'问题超过 max_length：{sample["text"]}')
        length = len(offsets)
        padding = schema['max_length'] - length
        inputs = {key: torch.tensor(value + [tokenizer.pad_token_id if key == 'input_ids' else 0] * padding)
                  for key, value in tokens.items()}
        intents = torch.zeros(schema['max_tasks'], dtype=torch.long)
        labels = torch.full((schema['max_tasks'], schema['max_length']), -100, dtype=torch.long)
        # 2. 每个任务占一个位置，实体标签只属于对应任务
        for slot, task in enumerate(sample['tasks']):
            intents[slot] = schema['intents'].index(task['intent'])
            for i, (start, end) in enumerate(offsets):
                if end > start:
                    labels[slot, i] = 0
            for entity in task['entities']:
                indices = [i for i, (a, b) in enumerate(offsets) if b > a and a < entity['end'] and b > entity['start']]
                if not indices or offsets[indices[0]][0] != entity['start'] or offsets[indices[-1]][1] != entity['end']:
                    raise ValueError(f'实体边界不能与 token 对齐：{entity}')
                for n, i in enumerate(indices):
                    tag = ('B-' if n == 0 else 'I-') + entity['type']
                    labels[slot, i] = schema['tags'].index(tag)
        encoded.append({'inputs': inputs, 'intents': intents, 'entities': labels})
    return encoded


def collate_batch(rows):
    return {'inputs': {key: torch.stack([r['inputs'][key] for r in rows]) for key in rows[0]['inputs']},
            'intents': torch.stack([r['intents'] for r in rows]),
            'entities': torch.stack([r['entities'] for r in rows])}
