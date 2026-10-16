"""多标签数据读取和预测的回归检查，无需下载模型或使用 GPU。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from predict import predict
from train import load_samples


class MultiLabelDataTests(unittest.TestCase):
    def load_csv(self, content):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            path.write_text(content, encoding="utf-8")
            return load_samples(path)

    def test_unknown_and_multiple_labels(self):
        samples, mapping = self.load_csv(
            "text,label\n数量,a\n排名,b\n两项,b | a\n闲聊,unknown\n"
        )
        self.assertEqual(mapping, {"a": 0, "b": 1})
        self.assertEqual(samples[2], ("两项", ("a", "b")))
        self.assertEqual(samples[3], ("闲聊", ()))

    def test_label_order_does_not_create_a_conflict(self):
        samples, _ = self.load_csv("text,label\n两项,a|b\n两项,b|a\n")
        self.assertEqual(samples[0], samples[1])

    def test_invalid_labels_are_rejected(self):
        for labels in ["unknown|a", "a||b", "a|a", "|a"]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                self.load_csv(f"text,label\n问题,{labels}\n其他,b\n")

    def test_conflicting_annotations_are_rejected(self):
        with self.assertRaises(ValueError):
            self.load_csv("text,label\n相同问题,a\n相同问题,a|b\n")


class MultiLabelPredictionTests(unittest.TestCase):
    def classify(self, logits, threshold=None, problem_type="multi_label_classification"):
        settings = SimpleNamespace(
            problem_type=problem_type, intent_threshold=0.5,
            intent_max_length=128, id2label={0: "region_rank_report", 1: "species_compose_report"},
        )
        class FakeModel:
            config = settings

            def __call__(self, **inputs):
                return SimpleNamespace(logits=torch.tensor([logits], dtype=torch.float32))

        def tokenizer(text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        return predict("测试问题", FakeModel(), tokenizer, torch.device("cpu"), threshold)

    def test_both_intents_can_be_selected(self):
        result = self.classify([2, 2])
        self.assertEqual(set(result["intents"]), {"region_rank_report", "species_compose_report"})
        self.assertGreater(sum(result["scores"].values()), 1)
        self.assertNotIn("unknown", result["scores"])

    def test_no_business_intent_returns_unknown(self):
        self.assertEqual(self.classify([-2, -2])["intents"], ["unknown"])

    def test_single_intent_is_preserved(self):
        self.assertEqual(self.classify([-2, 2])["intents"], ["species_compose_report"])

    def test_threshold_can_be_overridden(self):
        self.assertEqual(self.classify([2, 2], threshold=0.9)["intents"], ["unknown"])

    def test_invalid_threshold_is_rejected(self):
        for threshold in [0, 1, -1, float("nan")]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.classify([2, 2], threshold=threshold)

    def test_old_single_label_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "重新运行 train.py"):
            self.classify([2, 2], problem_type="single_label_classification")


if __name__ == "__main__":
    unittest.main()
