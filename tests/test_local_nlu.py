"""检查输出校验和评估口径；测试不需要启动 Ollama。"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluate
import nlu
import predict
from schema import Prediction, validate_entities


class LocalNLUTests(unittest.TestCase):
    def test_entity_offsets_and_shared_conditions(self):
        text = '今天甲市和甲市'
        time = {'type': 'time', 'text': '今天', 'start': 0, 'end': 2}
        city = {'type': 'city', 'text': '甲市', 'start': 2, 'end': 4}
        tasks = [{'intent': 'species_compose_report', 'entities': [time, city]},
                 {'intent': 'species_compose_report', 'entities': [time, {**city, 'start': 5, 'end': 7}]}]
        result = Prediction.model_validate({'tasks': tasks})
        validate_entities(result, text)
        result.tasks[1].entities[1].start = 4
        with self.assertRaisesRegex(ValueError, '位置与原文不一致'):
            validate_entities(result, text)

    def test_intent_fields_and_duplicate_entities(self):
        entity = {'type': 'top_n', 'text': '5', 'start': 0, 'end': 1}
        result = Prediction.model_validate({'tasks': [
            {'intent': 'species_compose_report', 'entities': [entity]}]})
        with self.assertRaisesRegex(ValueError, '数量查询不能携带'):
            validate_entities(result, '5')
        result.tasks[0].intent = 'region_rank_report'
        result.tasks[0].entities.append(result.tasks[0].entities[0])
        with self.assertRaisesRegex(ValueError, '重复返回'):
            validate_entities(result, '5')

    def test_parse_request_and_invalid_response(self):
        response = SimpleNamespace(done_reason='stop', message=SimpleNamespace(content='{"tasks": []}'))
        with patch.object(nlu.client, 'chat', return_value=response) as chat:
            self.assertEqual(nlu.parse(' 你好 '), {'text': ' 你好 ', 'tasks': []})
            request = chat.call_args.kwargs
            self.assertFalse(request['think'])
            self.assertFalse(request['stream'])
            self.assertEqual(request['format'], Prediction.model_json_schema())
            self.assertEqual(request['messages'][-1]['content'], ' 你好 ')
            response.done_reason = 'length'
            with self.assertRaisesRegex(ValueError, '长度上限'):
                nlu.parse('你好')
            response.done_reason = 'stop'
            response.message.content = '{"tasks": "不是列表"}'
            with self.assertRaisesRegex(ValueError, 'JSON 结构'):
                nlu.parse('你好')
        with patch.object(nlu.client, 'chat') as chat:
            with self.assertRaisesRegex(ValueError, '不能为空'):
                nlu.parse(' ')
            chat.assert_not_called()

    def test_comparison_preserves_task_count_and_errors(self):
        task = {'intent': 'species_compose_report', 'entities': []}
        self.assertFalse(evaluate.compare([task, task], [task])['complete_correct'])
        self.assertFalse(evaluate.compare([], None)['complete_correct'])
        self.assertTrue(evaluate.compare([], [])['complete_correct'])

    def test_unique_entity_alignment_without_guessing_repeated_names(self):
        task = {'intent': 'species_compose_report', 'entities': [
            {'type': 'city', 'text': '甲市', 'start': 0, 'end': 2}]}
        response = SimpleNamespace(done_reason='stop', message=SimpleNamespace(
            content=json.dumps({'tasks': [task]}, ensure_ascii=False)))
        with patch.object(nlu.client, 'chat', return_value=response):
            result = nlu.parse('今天甲市有多少蚊虫')
            self.assertEqual(result['tasks'][0]['entities'][0]['start'], 2)
            with self.assertRaisesRegex(ValueError, '位置与原文不一致'):
                nlu.parse('今天甲市和昨天甲市有多少蚊虫')

    def test_multiline_data_and_service_failure_report(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'data'
            (directory / 'nested').mkdir(parents=True)
            samples = [{'text': '你好', 'tasks': []}, {'text': '天气如何', 'tasks': []}]
            (directory / 'nested/cases.jsonl').write_text(
                '\n\n'.join(json.dumps(s, ensure_ascii=False, indent=2) for s in samples), encoding='utf-8')
            self.assertEqual(len(evaluate.load_cases(directory)), 2)
            output = Path(temp) / 'report.json'
            with patch('sys.argv', ['evaluate.py', '--data-dir', str(directory), '--output', str(output)]), \
                    patch('evaluate.parse', side_effect=RuntimeError('服务不可用')), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.main(), 1)
            report = json.loads(output.read_text())
            self.assertFalse(report['completed'])
            self.assertEqual(report['overall']['errors'], 1)
            self.assertEqual(report['overall']['complete_correct']['correct'], 0)

    def test_interaction_continues_after_invalid_output(self):
        output = io.StringIO()
        with patch('sys.argv', ['predict.py']), \
                patch('builtins.input', side_effect=['问题', '/exit']), \
                patch('predict.parse', side_effect=ValueError('实体位置错误')), \
                contextlib.redirect_stdout(output):
            self.assertEqual(predict.main(), 0)
        self.assertIn('实体位置错误', output.getvalue())
        self.assertIn('已退出', output.getvalue())


if __name__ == '__main__':
    unittest.main()
