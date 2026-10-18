"""一次编码，按查询任务预测意图和实体。"""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class JointNLU(nn.Module):
    def __init__(self, encoder, schema):
        super().__init__()
        self.encoder = encoder
        self.schema = schema
        hidden = encoder.config.hidden_size
        self.slots = nn.Embedding(schema['max_tasks'], hidden)
        self.intent_head = nn.Linear(hidden, len(schema['intents']))
        self.intent_embedding = nn.Embedding(len(schema['intents']), hidden)
        self.entity_head = nn.Linear(hidden, len(schema['tags']))

    def forward(self, inputs, intent_targets=None):
        # 1. MacBERT 只运行一次，后面的两项预测复用这份特征
        features = self.encoder(**inputs).last_hidden_state
        task_features = torch.tanh(features[:, :1, :] + self.slots.weight.unsqueeze(0))
        intent_logits = self.intent_head(task_features)
        # 2. 训练使用标注意图，预测使用模型自己识别出的意图
        intent_ids = intent_logits.argmax(-1) if intent_targets is None else intent_targets
        conditions = task_features + self.intent_embedding(intent_ids)
        entity_features = torch.tanh(features[:, None, :, :] + conditions[:, :, None, :])
        return intent_logits, self.entity_head(entity_features)


def build_intent_weights(samples, num_intents):
    # 1. 只统计训练集中的真实任务和空位置，不使用验证集
    labels = torch.cat([sample['intents'] for sample in samples])
    real_count = int((labels != 0).sum())
    empty_count = int((labels == 0).sum())
    # 2. 空位置更多时降低其权重，数据比例变化后自动重新计算
    weights = torch.ones(num_intents)
    if real_count and empty_count:
        weights[0] = min(1.0, real_count / empty_count)
    return weights


def compute_loss(outputs, batch, intent_weights=None):
    intent_logits, entity_logits = outputs
    # 使用逐位置损失再取平均，使全为空任务的批次也遵守相同权重
    loss = nn.functional.cross_entropy(
        intent_logits.flatten(0, 1), batch['intents'].flatten(),
        weight=intent_weights, reduction='none',
    ).mean()
    # 未知问题没有实体监督，避免全是忽略标签时产生 NaN
    if (batch['entities'] != -100).any():
        loss = loss + nn.functional.cross_entropy(entity_logits.reshape(-1, entity_logits.shape[-1]),
                                                  batch['entities'].flatten(), ignore_index=-100)
    return loss


def decode_tasks(text, offsets, intent_logits, entity_logits, schema):
    tasks = []
    for slot, intent_id in enumerate(intent_logits.argmax(-1).tolist()):
        if intent_id == 0:
            continue
        entities = []
        active = None
        for (start, end), tag_id in zip(offsets, entity_logits[slot].argmax(-1).tolist()):
            if start == end:
                continue
            tag = schema['tags'][tag_id]
            if tag == 'O':
                active = None
                continue
            prefix, kind = tag.split('-', 1)
            if prefix == 'I' and active is not None and active['type'] == kind and active['end'] == start:
                active['end'] = end
                active['text'] = text[active['start']:end]
            else:
                active = {'type': kind, 'text': text[start:end], 'start': start, 'end': end}
                entities.append(active)
        tasks.append({'intent': schema['intents'][intent_id], 'entities': entities})
    return tasks


def save_model(model, tokenizer, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, directory / 'joint_model.pt')
    (directory / 'nlu_config.json').write_text(json.dumps({'schema': model.schema, 'training': metadata}, ensure_ascii=False, indent=2), encoding='utf-8')


def load_model(directory, device):
    directory = Path(directory)
    settings = json.loads((directory / 'nlu_config.json').read_text(encoding='utf-8'))
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(directory, local_files_only=True))
    model = JointNLU(encoder, settings['schema'])
    model.load_state_dict(torch.load(directory / 'joint_model.pt', map_location='cpu', weights_only=True))
    return model.to(device).eval()
