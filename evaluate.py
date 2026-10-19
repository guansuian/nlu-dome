"""批量评估本地模型，完整匹配标准与旧版 NLU 一致。"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import config
from nlu import parse
from schema import Prediction, validate_entities


def load_cases(directory: Path) -> list:
    # 1. 同时兼容一行一条 JSONL 和项目现有的多行 JSON 对象
    cases = []
    for path in sorted(directory.rglob('*.jsonl')):
        content = path.read_text(encoding='utf-8-sig')
        offset = 0
        decoder = json.JSONDecoder()
        while offset < len(content):
            if content[offset].isspace():
                offset += 1
                continue
            try:
                sample, offset = decoder.raw_decode(content, offset)
                if not isinstance(sample, dict) or set(sample) != {'text', 'tasks'}:
                    raise ValueError('每条样本必须包含且仅包含 text 和 tasks。')
                if not isinstance(sample['text'], str) or not sample['text'].strip():
                    raise ValueError('text 必须是非空字符串。')
                result = Prediction.model_validate({'tasks': sample['tasks']})
                validate_entities(result, sample['text'])
            except ValueError as error:
                raise ValueError(f'{path}：{error}') from error
            cases.append((str(path.relative_to(directory)), sample))
    if not cases:
        raise ValueError(f'目录中没有测试数据：{directory}')
    if len({sample['text'] for _, sample in cases}) != len(cases):
        raise ValueError('测试集中存在重复问题，请先检查数据。')
    return cases


def task_key(task: dict) -> tuple:
    return task['intent'], tuple(sorted((e['type'], e['start'], e['end']) for e in task['entities']))


def compare(expected: list, predicted: list | None) -> dict:
    # 2. 忽略任务顺序，但保留同意图任务的数量；调用失败不能算拒识成功
    return {
        'count_correct': predicted is not None and len(expected) == len(predicted),
        'intent_correct': predicted is not None and sorted(t['intent'] for t in expected) == sorted(t['intent'] for t in predicted),
        'complete_correct': predicted is not None and sorted(map(task_key, expected)) == sorted(map(task_key, predicted)),
    }


def summarize(rows: list) -> dict:
    total = len(rows)
    return {'total': total, 'errors': sum(r['error'] is not None for r in rows),
            **{key: {'correct': sum(r[key] for r in rows),
                     'rate': sum(r[key] for r in rows) / total if total else None}
               for key in ['count_correct', 'intent_correct', 'complete_correct']}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=config.TEST_DIR)
    parser.add_argument('--limit', type=int, help='只检查前 N 条，用于流程检查；默认检查全部数据')
    parser.add_argument('--output', type=Path, help='报告路径；默认生成带时间的文件，不覆盖历史报告')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit 必须大于 0。')
    output = args.output or config.REPORT_DIR / f'ollama_evaluation_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    if output.exists():
        parser.error(f'报告已存在，不覆盖：{output}')
    try:
        cases = load_cases(args.data_dir)
    except ValueError as error:
        parser.error(str(error))
    available = len(cases)
    cases = cases[:args.limit]

    # 3. 逐题预测；无效输出记为失败，服务故障或用户中断时保存已完成部分
    rows = []
    interrupted = False
    start = time.monotonic()
    try:
        for index, (category, sample) in enumerate(cases, 1):
            predicted = None
            error_message = None
            service_failed = False
            try:
                predicted = parse(sample['text'])['tasks']
            except ValueError as error:
                error_message = str(error)
            except RuntimeError as error:
                error_message = str(error)
                service_failed = True
            row = {'category': category, 'text': sample['text'], 'expected': sample['tasks'],
                   'predicted': predicted, 'error': error_message, **compare(sample['tasks'], predicted)}
            rows.append(row)
            print(f'{index}/{len(cases)}：' + ('完整正确' if row['complete_correct'] else error_message or '结果与标注不一致'), flush=True)
            if service_failed:
                interrupted = True
                break
    except KeyboardInterrupt:
        interrupted = True

    # 4. 分文件统计，记录配置、提示词和数据指纹，便于后续复核
    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'model': config.MODEL_NAME, 'host': config.OLLAMA_HOST, 'think': config.THINK,
        'options': config.MODEL_OPTIONS, 'request_timeout': config.REQUEST_TIMEOUT,
        'prompt_sha256': hashlib.sha256(config.PROMPT_FILE.read_bytes()).hexdigest(),
        'schema_sha256': hashlib.sha256(json.dumps(Prediction.model_json_schema(), sort_keys=True).encode()).hexdigest(),
        'cases_sha256': hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        'data_dir': str(args.data_dir.resolve()), 'available': available, 'requested': len(cases),
        'completed': not interrupted, 'limited_run': len(cases) < available,
        'seconds': round(time.monotonic() - start, 2), 'overall': summarize(rows),
        'by_file': {name: summarize([r for r in rows if r['category'] == name])
                    for name in sorted({r['category'] for r in rows})},
        'details': rows,
        'definition': '任务数量、意图、实体类型、位置及归属全部正确；忽略任务顺序，保留重复任务。调用或校验失败计为错误。',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps(report['overall'], ensure_ascii=False, indent=2))
    print(f'报告已保存：{output}')
    return 1 if interrupted or report['overall']['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
