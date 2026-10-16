"""训练和预测共用的配置。所有默认路径均相对于项目目录。"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "hfl/chinese-macbert-base"
TRAIN_FILE = PROJECT_DIR / "data" / "train.csv"
MODEL_DIR = PROJECT_DIR / "models" / "intent_classifier"
MAX_LENGTH = 128
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42
TRAIN_DEVICE = "mps"  # 本机 Apple GPU；不可用时明确报错，避免悄悄改用 CPU。
PREDICTION_THRESHOLD = 0.5  # 每个业务意图独立判断，未达到阈值则不选中。

# 将下载缓存放在项目中；仍允许通过环境变量指定其他位置。
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".cache" / "huggingface"))
# 关闭 Transformers 对远端模型的自动格式转换；本项目自行保存 safetensors。
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")


def resolve_device(name="auto"):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境无法使用 CUDA，请选择 cpu 或 auto。")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("当前环境无法使用 MPS，请选择 cpu 或 auto。")
    return torch.device(name)
