"""唯一的业务入口：parse(text) 返回问题及各查询任务。"""
import json

import httpx
from ollama import Client, ResponseError
from pydantic import ValidationError

import config
from schema import Prediction, validate_entities

# 本机服务直接连接，避免系统代理把 localhost 请求转发出去
client = Client(host=config.OLLAMA_HOST, timeout=config.REQUEST_TIMEOUT, trust_env=False)


def parse(text: str) -> dict:
    # 1. 检查问题，保留原文空格，避免改变实体的字符位置
    if not text.strip():
        raise ValueError('问题不能为空。')
    if len(text) > config.MAX_INPUT_LENGTH:
        raise ValueError(f'问题不能超过 {config.MAX_INPUT_LENGTH} 个字符，请缩短后重试。')
    try:
        text.encode('utf-8')
    except UnicodeEncodeError as error:
        raise ValueError('输入包含无效编码，请重新输入。') from error

    # 2. 将业务规则和输出结构一并交给模型，每个问题独立处理
    output_schema = Prediction.model_json_schema()
    instructions = config.PROMPT_FILE.read_text(encoding='utf-8')
    instructions += '\n输出必须符合以下 JSON Schema：\n' + json.dumps(output_schema, ensure_ascii=False)
    try:
        response = client.chat(
            model=config.MODEL_NAME,
            messages=[{'role': 'system', 'content': instructions},
                      {'role': 'user', 'content': text}],
            format=output_schema, think=config.THINK, stream=False,
            options=config.MODEL_OPTIONS,
        )
    except ResponseError as error:
        raise RuntimeError(f'Ollama 返回错误（{error.status_code}）：{error.error}') from error
    except (ConnectionError, httpx.HTTPError) as error:
        raise RuntimeError('无法连接 Ollama 或请求超时，请检查服务是否启动及 config.py 中的配置。') from error

    # 3. 校验返回结构；不完整或无效结果不能冒充未知意图
    if response.done_reason == 'length':
        raise ValueError('模型输出达到长度上限，结果不完整，请缩短问题后重试。')
    try:
        prediction = Prediction.model_validate_json(response.message.content or '')
    except ValidationError as error:
        raise ValueError(f'模型输出不符合约定的 JSON 结构：{error.errors()[0]["msg"]}') from error
    # 4. 唯一出现的实体由代码定位；多次出现时保留模型位置并严格校验
    for task in prediction.tasks:
        for entity in task.entities:
            start = text.find(entity.text)
            if start >= 0 and start == text.rfind(entity.text):
                entity.start = start
                entity.end = start + len(entity.text)
    validate_entities(prediction, text)
    return {'text': text, **prediction.model_dump()}
