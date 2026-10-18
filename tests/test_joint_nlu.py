"""用微型模型检查结构，不向业务训练集添加任何样本。"""
import tempfile
import json
import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import config
import predict as prediction
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast
from dataset import build_schema, encode_samples, load_samples, validate_sample
from model import JointNLU, build_intent_weights, compute_loss, decode_tasks, load_model, save_model


class JointTests(unittest.TestCase):
    def setUp(self):
        self.samples = load_samples(config.TRAIN_DIR)
        self.schema = build_schema(self.samples, 4, 128)

    def test_data_and_invalid_offset(self):
        self.assertTrue(self.samples)
        bad = {'text': '深圳', 'tasks': [{'intent': 'count', 'entities': [
            {'type': 'city', 'text': '深圳', 'start': 0, 'end': 1}]}]}
        with self.assertRaises(ValueError):
            validate_sample(bad)

    def test_invalid_terminal_encoding_keeps_interaction_alive(self):
        tokenizer = Mock()
        with self.assertRaisesRegex(ValueError, '无效编码'):
            prediction.predict('广州\udc80今天', None, tokenizer, 'cpu')
        tokenizer.assert_not_called()
        output = io.StringIO()
        with patch('sys.argv', ['predict.py']), \
                patch('predict.config.resolve_device', return_value='cpu'), \
                patch('predict.load_model'), \
                patch('predict.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                patch('builtins.input', side_effect=['广州\udc80今天', '/exit']), \
                contextlib.redirect_stdout(output):
            prediction.main()
        self.assertIn('无效编码', output.getvalue())
        self.assertIn('已退出', output.getvalue())
        tokenizer.assert_not_called()

    def test_empty_task_weight_and_gradients(self):
        # 1 个真实任务、3 个空位置时，两组损失的总影响应当平衡
        labels = torch.tensor([[1, 0, 0, 0]])
        weights = build_intent_weights([{'intents': labels[0]}], 2)
        torch.testing.assert_close(weights, torch.tensor([1 / 3, 1.0]))
        logits = torch.zeros(1, 4, 2, requires_grad=True)
        entities = torch.zeros(1, 4, 1, 1)
        batch = {'intents': labels, 'entities': torch.full((1, 4, 1), -100)}
        loss = compute_loss((logits, entities), batch, weights)
        loss.backward()
        torch.testing.assert_close(logits.grad[0, 0, 1].abs(), logits.grad[0, 1:, 1].sum().abs())

        # 数据比例改变后重新计算，全为空或没有空位置时不产生零权重
        mixed = build_intent_weights([{'intents': torch.tensor([1, 1, 0, 0])}], 2)
        torch.testing.assert_close(mixed, torch.ones(2))
        for targets in [torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)]:
            edge_weights = build_intent_weights([{'intents': targets}], 2)
            torch.testing.assert_close(edge_weights, torch.ones(2))

        # 全为空的批次仍应降低损失，避免批内权重归一化抵消调整
        batch['intents'] = torch.zeros_like(labels)
        weighted = compute_loss((logits, entities), batch, weights)
        unweighted = compute_loss((logits, entities), batch)
        torch.testing.assert_close(weighted, unweighted / 3)
        self.assertTrue(torch.isfinite(weighted))

    def test_alignment_and_save_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            vocab = Path(temp) / 'vocab.txt'
            # 词表每行一个字符，空白由分词器处理，不能写成词表条目
            chars = sorted({char for s in self.samples for char in s['text'] if not char.isspace()})
            vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + chars))
            tokenizer = BertTokenizerFast(vocab_file=str(vocab))
            rows = encode_samples(self.samples, tokenizer, self.schema)
            encoder = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
                                           num_hidden_layers=1, num_attention_heads=2, intermediate_size=32))
            model = JointNLU(encoder, self.schema).eval()
            inputs = {k: v.unsqueeze(0) for k, v in rows[0]['inputs'].items()}
            calls = []
            hook = encoder.register_forward_hook(lambda *args: calls.append(1))
            expected = model(inputs)
            self.assertEqual(len(calls), 1)
            hook.remove()
            unknown = {'intents': torch.zeros(1, 4, dtype=torch.long),
                       'entities': torch.full((1, 4, self.schema['max_length']), -100, dtype=torch.long)}
            loss = compute_loss(expected, unknown)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            output = Path(temp) / 'saved'
            save_model(model, tokenizer, output, {})
            restored = load_model(output, torch.device('cpu'))
            actual = restored(inputs)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)

    def test_training_with_independent_validation(self):
        # 使用临时微型模型验证完整训练流程，不改动业务数据或原模型
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / 'base'
            base.mkdir()
            (base / 'vocab.txt').write_text('\n'.join(
                ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '甲', '乙', '丙', '丁', '戊', '己']))
            tokenizer = BertTokenizerFast(vocab_file=str(base / 'vocab.txt'))
            tokenizer.save_pretrained(base)
            BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
                                num_hidden_layers=1, num_attention_heads=2,
                                intermediate_size=32)).save_pretrained(base)
            def sample(text, intent):
                tasks = [] if intent is None else [{'intent': intent, 'entities': [
                    {'type': 'item', 'text': text, 'start': 0, 'end': len(text)}]}]
                return {'text': text, 'tasks': tasks}
            groups = {'train': [sample('甲', 'a'), sample('乙', 'b'), sample('丙', None)],
                      'validation': [sample('丁', 'a'), sample('戊', 'b'), sample('己', None)]}
            for name, records in groups.items():
                directory = root / name
                directory.mkdir()
                (directory / 'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            result = subprocess.run([
                sys.executable, str(config.ROOT / 'train.py'), '--model', str(base),
                '--train-dir', str(root / 'train'), '--validation-dir', str(root / 'validation'),
                '--output-dir', str(root / 'output'), '--device', 'cpu', '--epochs', '1',
                '--max-length', '16', '--max-tasks', '2'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metadata = json.loads((root / 'output/nlu_config.json').read_text())
            self.assertIn('task_exact_match', metadata['training']['metrics'])
            self.assertFalse(metadata['training']['smoke_test'])
            self.assertAlmostEqual(metadata['training']['intent_loss_weights']['__none__'], 0.5)

            # 单意图、无验证集时，每轮也必须处理全部数据并保存最后一轮
            (root / 'train/samples.jsonl').write_text('\n'.join(
                json.dumps(sample(text, 'a')) for text in ['甲', '乙', '丙']))
            result = subprocess.run([
                sys.executable, str(config.ROOT / 'train.py'), '--model', str(base),
                '--train-dir', str(root / 'train'), '--validation-dir', str(root / 'missing'),
                '--output-dir', str(root / 'single_output'), '--device', 'cpu', '--epochs', '2',
                '--batch-size', '2', '--max-length', '16', '--max-tasks', '2'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metadata = json.loads((root / 'single_output/nlu_config.json').read_text())
            self.assertEqual(metadata['training']['epoch'], 2)
            self.assertEqual(metadata['training']['training_samples'], 3)
            self.assertEqual(metadata['training']['batches_per_epoch'], 2)
            self.assertEqual(metadata['training']['validation_samples'], 0)
            self.assertEqual(metadata['training']['metrics'], {})

    def test_same_intent_separate_entities(self):
        schema = {'intents': ['__none__', 'count'], 'tags': ['O', 'B-city', 'I-city']}
        intents = torch.tensor([[0., 9.], [0., 9.]])
        entities = torch.zeros(2, 4, 3)
        entities[0, :, 0] = 1
        entities[1, :, 0] = 1
        entities[0, 0, 1] = 9
        entities[0, 1, 2] = 9
        entities[1, 2, 1] = 9
        entities[1, 3, 2] = 9
        tasks = decode_tasks('深圳广州', [(0, 1), (1, 2), (2, 3), (3, 4)], intents, entities, schema)
        self.assertEqual([t['entities'][0]['text'] for t in tasks], ['深圳', '广州'])
        self.assertEqual(len(decode_tasks('深圳广州', [(0, 1)] * 4, intents.flip(-1), entities, schema)), 0)


if __name__ == '__main__':
    unittest.main()
