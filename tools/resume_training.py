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
WORK = ROOT / 'results/weights1700/campaign'


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
    state = json.loads((work / 'state.json').read_text())
    runtime = state.get('runtime_overrides') or {}
    if runtime.get('evolving_opponents'):
        # Compact GitHub snapshots keep just the latest model and its ten opponents. Rebuild
        # local-only evaluation/telemetry directories without duplicating model snapshots.
        versions = work / 'learner/policy_versions'
        if not versions.is_dir():
            raise ValueError('Backup is missing its evolving old-policy pool')
        pool = work / 'pool'
        if not pool.exists():
            shutil.copytree(versions, pool)
        digest = hashlib.sha256()
        for path in sorted(pool.glob('*/*')):
            if path.is_file():
                digest.update(str(path.relative_to(pool)).encode() + b'\0' + path.read_bytes())
        state['pool_sha256'] = digest.hexdigest()
        for folder in ('eval', 'metrics', 'best'):
            (work / folder).mkdir(parents=True, exist_ok=True)
        (work / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    return state


def build_resume_command(state, work, without_services=False, backup_interval=1800,
                         reward_metrics=False, player_metrics=False, throughput_profile=None,
                         continuous=False, evolving_opponents=False, opponent_pool_size=None,
                         promotion_margin=None):
    config = state['config']
    cmd = [sys.executable, str(ROOT / 'tools/train_ranked.py'), '--base', config['base'],
           '--out', str(work.relative_to(ROOT)), '--minutes', str(sum(config['minutes'])),
           '--pilot-minutes', str(config['minutes'][0]),
           '--block-minutes', str(max(config['minutes'][1:] or config['minutes'])),
           '--rounds', str(config['rounds']), '--seed', str(config['seed']),
           '--audit-seed', str(config['audit_seed'])]
    if state.get('target_ranking'):
        cmd += ['--target-ranking', str(state['target_ranking'])]
    if config.get('reference_pool'):
        cmd += ['--reference-pool', config['reference_pool']]
    if config.get('extra_versions'):
        cmd += ['--extra-versions', *config['extra_versions']]
    train_args = config['train_args']
    defaults = {'--lr': '1e-4', '--batch-size': '12096', '--skill-minutes': '15', '--ts-per-save': '5m'}
    for flag in ('--threads', '--minibatch-size', '--lr', '--critic-lr', '--epochs', '--batch-size',
                 '--entropy-scale', '--norm-advantages', '--target-kl', '--gamma', '--gae-lambda',
                 '--return-stat-max-count', '--random-state-prob', '--tactical-state-prob',
                 '--train-vs-old', '--ts-per-version', '--max-versions', '--skill-minutes',
                 '--ts-per-save', '--rewards'):
        if flag not in train_args:
            continue
        value = train_args[train_args.index(flag) + 1]
        if defaults.get(flag) == value:
            continue  # preserve exact default config entries
        cmd += ['--minibatch' if flag == '--minibatch-size' else flag, value]
    for flag in ('--mask-entropy', '--fuzzed-kickoffs', '--pfsp'):
        if flag in train_args:
            cmd.append(flag)
    if reward_metrics or (state.get('runtime_overrides') or {}).get('reward_metrics'):
        cmd.append('--reward-metrics')
    if player_metrics or (state.get('runtime_overrides') or {}).get('player_metrics'):
        cmd.append('--player-metrics')
    profile = throughput_profile or (state.get('runtime_overrides') or {}).get('throughput_profile')
    if profile:
        cmd += ['--throughput-profile', profile]
    if continuous or (state.get('runtime_overrides') or {}).get('continuous'):
        cmd.append('--continuous')
    runtime = state.get('runtime_overrides') or {}
    if evolving_opponents or runtime.get('evolving_opponents'):
        cmd.append('--evolving-opponents')
        cmd += ['--opponent-pool-size', str(opponent_pool_size if opponent_pool_size is not None
                                            else runtime.get('opponent_pool_size', 10))]
        cmd += ['--promotion-margin', str(promotion_margin if promotion_margin is not None
                                           else runtime.get('promotion_margin', 50.0))]
    if not without_services:
        cmd += ['--github-backup', '--backup-interval', str(backup_interval), '--wandb-bridge']
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-only', action='store_true')
    parser.add_argument('--without-services', action='store_true',
                        help='Run trainer only, without automatic GitHub backup or W&B bridge')
    parser.add_argument('--reward-metrics', action='store_true',
                        help='Enable and persist sampled reward-component telemetry on the resumed campaign')
    parser.add_argument('--player-metrics', action='store_true',
                        help='Enable and persist sampled per-player telemetry on the resumed campaign')
    parser.add_argument('--throughput-profile', choices=('cpu-20k',),
                        help='Use the benchmarked two-vCPU profile targeting about 20k overall steps/second')
    parser.add_argument('--continuous', action='store_true',
                        help='Continue with repeated training/evaluation blocks indefinitely, until interrupted')
    parser.add_argument('--evolving-opponents', action='store_true',
                        help='Enable the rolling historical opponent pool gated by Elo')
    parser.add_argument('--opponent-pool-size', type=int,
                        help='Number of old policies in the evolving pool (default: 10)')
    parser.add_argument('--promotion-margin', type=float,
                        help='Required Elo lead over the pool strongest policy (default: 50)')
    parser.add_argument('--backup-interval', type=int, default=1800,
                        help='Minimum seconds between GitHub auto-backups (default: 1800 = 30 minutes)')
    args = parser.parse_args()
    saved = json.loads((SNAPSHOT / 'state.json').read_text())
    work = ROOT / saved.get('working_directory', str(WORK.relative_to(ROOT)))
    if not work.resolve().is_relative_to((ROOT / 'results').resolve()):
        raise ValueError('Saved working campaign must be under results/, never the canonical backup')
    state = restore(work=work)
    if args.restore_only:
        return
    if state['status'] == 'complete':
        print('Campaign complete; create a new experiment to train further.')
        return
    cmd = build_resume_command(state, work, without_services=args.without_services,
                               backup_interval=args.backup_interval, reward_metrics=args.reward_metrics,
                               player_metrics=args.player_metrics, throughput_profile=args.throughput_profile,
                               continuous=args.continuous, evolving_opponents=args.evolving_opponents,
                               opponent_pool_size=args.opponent_pool_size,
                               promotion_margin=args.promotion_margin)
    print('Resuming only because this script was explicitly invoked.', flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
