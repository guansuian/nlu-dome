"""直接运行进入交互模式，也可以用 --text 预测一个问题。"""
import argparse
import json
import sys

import config
from nlu import parse


def main() -> int:
    # 1. 命令行指定问题时，预测一次后退出
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', help='需要识别的问题；不填写则进入交互模式')
    args = parser.parse_args()
    if args.text is not None:
        try:
            print(json.dumps(parse(args.text), ensure_ascii=False, indent=2))
            return 0
        except (ValueError, RuntimeError) as error:
            print(f'错误：{error}', file=sys.stderr)
            return 1

    # 2. 交互输入，每次独立解析；空行跳过，单次失败后仍可继续
    print(f'使用本地模型：{config.MODEL_NAME}。输入 /exit 退出；首次调用可能需要加载模型。')
    try:
        while True:
            text = input('问题：')
            if text.strip() == '/exit':
                break
            if not text.strip():
                continue
            try:
                print(json.dumps(parse(text), ensure_ascii=False, indent=2))
            except (ValueError, RuntimeError) as error:
                print(f'错误：{error}')
    except (EOFError, KeyboardInterrupt):
        pass
    print('\n已退出。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
