"""定义返回格式，检查实体是否确实来自原文。"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    type: Literal['province', 'city', 'district', 'street', 'time', 'species', 'sex',
                  'rank_level', 'top_n']
    text: str = Field(min_length=1)
    start: int = Field(ge=0, description='原文中的字符起点，从 0 开始')
    end: int = Field(gt=0, description='原文中的字符终点，不包含该位置')


class Task(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    intent: Literal['species_compose_report', 'region_rank_report']
    entities: list[Entity]


class Prediction(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    tasks: list[Task]


def validate_entities(prediction: Prediction, text: str) -> None:
    # 1. 每个任务单独检查，共享实体允许出现在不同任务中
    for number, task in enumerate(prediction.tasks, 1):
        seen = set()
        for entity in task.entities:
            # 2. 位置必须与原文完全一致，不自动替换成第一次出现的位置
            if not (0 <= entity.start < entity.end <= len(text)) or text[entity.start:entity.end] != entity.text:
                raise ValueError(f'任务 {number} 的实体位置与原文不一致：{entity.text!r}'
                                 f' [{entity.start}:{entity.end}]。')
            if task.intent == 'species_compose_report' and entity.type in {'rank_level', 'top_n'}:
                raise ValueError(f'任务 {number}：数量查询不能携带 {entity.type}。')
            key = (entity.type, entity.start, entity.end)
            if key in seen:
                raise ValueError(f'任务 {number} 重复返回了同一实体：{entity.text!r}。')
            seen.add(key)
