#!/usr/bin/env python3
"""Verify the canonical paused snapshot, restore a working copy, and explicitly resume.
Use --restore-only to prepare without starting training. Never trains in training-backup/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'training-backup'
WORK = ROOT / 'results/ranking_3043214022/campaign'


def verify(snapshot):
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot.resolve()):
            raise ValueError(f'Unsafe manifest path: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Backup hash mismatch: {name}')
    return manifest


def restore(snapshot=SNAPSHOT, work=WORK):
    manifest = verify(snapshot)
    if work.exists():
        if not (work / 'state.json').exists():
            raise ValueError('Existing work directory is not a campaign; refusing to overwrite')
        print('Existing working campaign retained; resume it rather than overwrite newer progress.')
    else:
        work.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot, work, ignore=shutil.ignore_patterns('manifest.json'))
        print(f'Restored checkpoint {manifest["checkpoint"]} into {work}')
    return json.loads((work / 'state.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-only', action='store_true')
    parser.add_argument('--without-monitor', action='store_true')
    args = parser.parse_args()
    saved = json.loads((SNAPSHOT / 'state.json').read_text())
    work = ROOT / saved.get('working_directory', str(WORK.relative_to(ROOT)))
    if not work.resolve().is_relative_to((ROOT / 'results').resolve()):
        raise ValueError('Saved working campaign must be under results/, never the canonical backup')
    state = restore(work=work)
    if args.restore_only:
        return
    config = state['config']
    if state['status'] == 'complete':
        print('Campaign complete; create a new experiment to train further.')
        return
    # Exact options of the paused campaign. The driver validates config and pool digest.
    cmd = [sys.executable, str(ROOT / 'tools/train_ranked.py'), '--base', config['base'],
           '--out', str(work.relative_to(ROOT)), '--minutes', str(sum(config['minutes'])),
           '--pilot-minutes', str(config['minutes'][0]),
           '--block-minutes', str(max(config['minutes'][1:] or config['minutes'])),
           '--rounds', str(config['rounds']), '--seed', str(config['seed'])]
    cmd += ['--audit-seed', str(config['audit_seed'])]
    if state.get('target_ranking'):
        cmd += ['--target-ranking', str(state['target_ranking'])]
    if config.get('reference_pool'):
        cmd += ['--reference-pool', config['reference_pool']]
    train_args = config['train_args']
    for flag in ('--threads', '--minibatch-size', '--lr', '--entropy-scale',
                 '--random-state-prob', '--train-vs-old', '--rewards'):
        if flag not in train_args:
            continue
        value = train_args[train_args.index(flag) + 1]
        if flag == '--lr' and value == '1e-4':
            continue  # preserve the exact default config, not float-formatted 0.0001
        cmd += ['--minibatch' if flag == '--minibatch-size' else flag, value]
    if not args.without_monitor:
        cmd += ['--monitor', '--github-backup']
    print('Resuming only because this script was explicitly invoked.', flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
