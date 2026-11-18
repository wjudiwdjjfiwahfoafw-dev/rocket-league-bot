#!/usr/bin/env python3
"""Headless render regression: capture real RocketSimVis UDP frames, never train.
Requires a built GigaLearnBot; no GUI, W&B, restored campaign or external services.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def run_case(modes, expected):
    checkpoint = ROOT / 'training-backup/best'
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in checkpoint.rglob('*') if p.is_file()}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver, tempfile.TemporaryFile(mode='w+t') as log:
        receiver.bind(('127.0.0.1', 9273))
        receiver.settimeout(0.3)
        cmd = [str(ROOT / 'build/GigaLearnBot'), '--render', '--checkpoints', str(checkpoint),
               '--device', 'cpu', '--threads', '1', '--modes', modes, '--games', '99',
               '--mode-balance', 'players', '--no-touch-timeout', '0.15', '--no-metrics',
               '--train-vs-old', '1', '--seed', '123']
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, GGL_WANDB='0'))
        seen, frames = set(), 0
        first_frame = None
        deadline = time.monotonic() + 15
        try:
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise AssertionError(f'Render exited unexpectedly: {child.returncode}')
                try:
                    payload, _ = receiver.recvfrom(65535)
                except socket.timeout:
                    continue
                state = json.loads(payload)
                size = len(state['cars']) // 2
                assert len(state['cars']) == size * 2 and size in expected, state
                assert sum(c['team_num'] == 0 for c in state['cars']) == size
                assert sum(c['team_num'] == 1 for c in state['cars']) == size
                seen.add(size)
                frames += 1
                if first_frame is None:
                    first_frame = time.monotonic()
                if seen == expected and time.monotonic() - first_frame >= 3:
                    break
            assert seen == expected and frames >= 20, (modes, seen, frames)
        finally:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            log.seek(0)
            output = log.read()
            assert 'one arena, random mode each episode, deterministic argmax' in output, output[-3000:]
            assert 'Saving to folder' not in output and 'PPO Learn Time' not in output
        print(f'{modes}: {frames} real UDP frames, observed modes {sorted(seen)}, no training/saving')
    for name, digest in before.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name


def main():
    run_case('1v1,2v2,3v3', {1, 2, 3})
    run_case('1v1,3v3', {1, 3})
    run_case('2v2', {2})


if __name__ == '__main__':
    main()
