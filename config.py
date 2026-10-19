"""本地模型调用的配置，路径不依赖终端的当前目录。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_NAME = 'qwen3:4b-q4_K_M'
OLLAMA_HOST = 'http://localhost:11434'
REQUEST_TIMEOUT = 180.0
MAX_INPUT_LENGTH = 500
THINK = False
MODEL_OPTIONS = {'temperature': 0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 4096}
PROMPT_FILE = ROOT / 'prompts/nlu.txt'
TEST_DIR = ROOT / 'data/test'
REPORT_DIR = ROOT / 'reports'
