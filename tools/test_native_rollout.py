#!/usr/bin/env python3
"""Short end-to-end fixed-rollout regression using the real bundled model and a saved opponent.
Run from any directory after building. Only trains in a temporary copy.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from train_ranked import REPO, newest


def main():
    os.chdir(REPO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bot', default='build/GigaLearnBot')
    p.add_argument('--base', type=Path, default=None,
                   help='Full numbered checkpoint directory (default: newest in training-backup/best)')
    p.add_argument('--pool', type=Path, default=Path('training-backup/pool'),
                   help='Policy versions pool directory')
    args = p.parse_args()
    base = args.base if args.base is not None else newest(REPO / 'training-backup/best')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ckpts = root / 'ckpt'
        shutil.copytree(base, ckpts / base.name)
        shutil.copytree(args.pool, ckpts / 'policy_versions')
        start = int(base.name)
        # In a vs-old iteration only half the cars belong to the learner. Counting both teams
        # in the stop callback used to stop before this absolute target was actually reached.
        cmd = [args.bot, '--checkpoints', str(ckpts), '--model', 's', '--modes', '1v1,2v2,3v3',
               '--device', 'cpu', '--threads', '1', '--games', '11', '--mode-balance', 'players',
               '--fixed-rollout', '--lean-metrics', '--train-vs-old', '1', '--ts-per-itr', '360',
               '--batch-size', '360', '--minibatch-size', '180', '--epochs', '1', '--new-run',
               '--ts-per-save', '1b', '--seed', '9031', '--max-timesteps', str(start + 721)]
        env = dict(os.environ, GGL_WANDB='0', GGL_METRICS_EVERY='1', GGL_METRICS_DIR=str(root / 'metrics'),
                   OMP_WAIT_POLICY='passive')
        run = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env, timeout=60)
        if run.returncode:
            raise RuntimeError(run.stdout[-1000:] + run.stderr[-1000:])
        rows = [json.loads(line) for f in (root / 'metrics').glob('*.jsonl') for line in f.read_text().splitlines()]
        assert [int(r['Total Timesteps']) - start for r in rows] == [360, 720, 1080], rows
        assert all(r['Experience Timesteps'] == r['Collected Timesteps'] == 360 for r in rows)
        assert '6 x 1v1 (12 players), 3 x 2v2 (12 players), 2 x 3v3 (12 players)' in run.stdout
        assert int(newest(ckpts).name) == start + 1080
        print('PASS: equal player counts; three on-policy batches vs a saved opponent; exact learner-step stop and resumable checkpoint.')


if __name__ == '__main__':
    main()
