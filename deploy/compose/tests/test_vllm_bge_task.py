#!/usr/bin/env python3
'''Regression test: vllm-bge must be launched with --task embedding.'''
import sys
from pathlib import Path

def main() -> int:
    compose = Path(__file__).resolve().parents[1] / 'compose.yaml'
    lines = compose.read_text().splitlines()
    for i, line in enumerate(lines):
        if line == '  vllm-bge:':
            break
    else:
        print('FAIL: vllm-bge service not found')
        return 1

    command_lines = []
    in_command = False
    for line in lines[i:]:
        if line.startswith('    command: |'):
            in_command = True
            continue
        if in_command:
            if not line.startswith('      '):
                break
            command_lines.append(line.strip())
    command = ' '.join(command_lines)
    if '--task embedding' in command or '--task embed' in command:
        print('PASS')
        return 0
    print('FAIL: vllm-bge command missing --task embedding')
    return 1

if __name__ == '__main__':
    sys.exit(main())
