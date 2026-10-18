"""训练和预测共用的默认配置。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('DISABLE_SAFETENSORS_CONVERSION', 'true')
MODEL_NAME = 'hfl/chinese-macbert-base'
TRAIN_DIR = ROOT / 'data/train'
VALIDATION_DIR = ROOT / 'data/validation'
OUTPUT_DIR = ROOT / 'models/joint_nlu'


def resolve_device(name):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('当前环境无法使用 Apple GPU（MPS）。')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('当前环境无法使用 CUDA。')
    return torch.device(name)
