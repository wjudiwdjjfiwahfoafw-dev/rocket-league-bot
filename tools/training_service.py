#!/usr/bin/env python3
"""Headless GitHub checkpoint backup worker for ranked training campaigns."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
FILES = {'CRITIC.lt', 'CRITIC_OPTIM.lt', 'POLICY.lt', 'POLICY_OPTIM.lt',
         'SHARED_HEAD.lt', 'SHARED_HEAD_OPTIM.lt', 'SHARED_HEAD_SLOTS.lt',
         'INFO.json', 'RUNNING_STATS.json'}
DEFAULT_BACKUP_INTERVAL = 1800  # 30 minutes between GitHub auto-backups
BACKUP_RETRY_SECONDS = 60       # Retry cadence only when a due backup is incomplete or failed


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def signature(path):
    return [(str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(path.rglob('*')) if p.is_file()]


def assess_goal(state, target=1700.0):
    best = state.get('best') or {}
    screen = (best.get('screen') or {}).get('Ranking')
    confirm = (best.get('confirm') or {}).get('Ranking')
    audit = ((state.get('audit') or {}).get(str(best.get('steps'))) or {}).get('Ranking')
    selection_met = all(v is not None and v >= target for v in (screen, confirm))
    return {'target': target, 'screen': screen, 'confirmation': confirm, 'audit': audit,
            'selection_met': selection_met,
            'verified': selection_met and audit is not None and audit >= target,
            'screen_gap': max(0.0, target - screen) if screen is not None else None}


def read_manifest_created(destination):
    marker = destination / 'manifest.json'
    if not marker.exists():
        return None
    try:
        created = json.loads(marker.read_text()).get('created')
        return float(created) if created is not None else None
    except (OSError, ValueError, TypeError):
        return None


def last_backup_time(destination, status):
    if status.get('push_pending'):
        return None
    times = [t for t in (status.get('last_success'), read_manifest_created(destination)) if t is not None]
    return max(times) if times else None


def seconds_until_backup_due(destination, status, min_interval, now=None):
    if min_interval <= 0 or status.get('push_pending'):
        return 0.0
    last_ts = last_backup_time(destination, status)
    if last_ts is None:
        return 0.0
    cur = time.time() if now is None else float(now)
    return max(0.0, (last_ts + min_interval) - cur)


def verify_frozen_versions(versions, state):
    """Refuse to publish a snapshot in which a saved old policy was rewritten after creation."""
    expected = {int(step): files for step, files in (state.get('frozen_versions') or {}).items()}
    if not expected:
        return
    for version in versions:
        step = int(version.name)
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(version.iterdir()) if p.is_file()}
        if step in expected and files != expected[step]:
            raise RuntimeError(f'Policy version {step} changed after creation; refusing backup')


def snapshot_layout(campaign, state, latest):
    """Return only policy-model snapshot sources for the saved campaign format."""
    policy_pool = campaign / 'learner/policy_versions'
    evolving = bool((state.get('runtime_overrides') or {}).get('evolving_opponents'))
    if evolving:
        versions = sorted((p for p in policy_pool.iterdir() if p.is_dir() and p.name.isdecimal()),
                          key=lambda p: int(p.name))
        expected = int(state['runtime_overrides'].get('opponent_pool_size', 10))
        # The pool of an evolving campaign starts with the policies history offers and grows
        # through promotions until it reaches its target size, so a smaller pool is expected
        # while it is still filling up. Only a pool that outgrew its target is a fault.
        if len(versions) > expected:
            raise RuntimeError(f'Evolving policy pool has {len(versions)} entries; expected at most {expected}')
        if not versions:
            raise RuntimeError('Evolving policy pool is empty; nothing to back up')
        verify_frozen_versions(versions, state)
        return [latest, policy_pool], ['learner/' + latest.name, 'learner/policy_versions'], True
    return ([latest, policy_pool, campaign / 'pool', campaign / 'best' / str(state['best']['steps'])],
            ['learner/' + latest.name, 'learner/policy_versions', 'pool',
             'best/' + str(state['best']['steps'])], False)


def backup(campaign, destination, branch, status, min_interval=0, now=None):
    cur_now = time.time() if now is None else float(now)
    status['interval_seconds'] = min_interval
    wait_s = seconds_until_backup_due(destination, status, min_interval, now=cur_now)
    if wait_s > 0:
        status['next_due'] = cur_now + wait_s
        return False
    if git('branch', '--show-current') != branch:
        raise RuntimeError('Branch changed; refusing backup')
    # Retry a previously unpushed local commit before inspecting new checkpoints.
    local_head = git('rev-parse', 'HEAD')
    remote_line = git('ls-remote', 'origin', 'refs/heads/' + branch).split()
    if status.get('push_pending') or not remote_line or remote_line[0] != local_head:
        try:
            git('push', 'origin', branch)
        except subprocess.CalledProcessError:
            git('pull', '--rebase', '--autostash', 'origin', branch)
            git('push', 'origin', branch)
        status['push_pending'] = False
    candidates = sorted((p for p in (campaign / 'learner').iterdir() if p.name.isdecimal()), key=lambda p: int(p.name))
    latest = candidates[-1]
    state = json.loads((campaign / 'state.json').read_text())
    if state.get('best') is None:
        raise RuntimeError('Waiting for baseline evaluation; prior remote snapshot remains protected')
    sources, targets, evolving = snapshot_layout(campaign, state, latest)
    before = [signature(p) for p in sources]
    if any(not sig for sig in before) or not FILES.issubset({p.name for p in latest.iterdir()}):
        raise RuntimeError('Checkpoint incomplete; waiting for next poll')
    if any(time.time_ns() - mtime < 10_000_000_000 for sig in before for _, _, mtime in sig):
        raise RuntimeError('Checkpoint files still settling; waiting for next poll')
    identity = hashlib.sha256(json.dumps({'files': before, 'history': state['history'],
        'best': state['best'], 'audit': state.get('audit'), 'status': state['status']}, sort_keys=True).encode()).hexdigest()
    marker = destination / 'manifest.json'
    if marker.exists() and json.loads(marker.read_text()).get('identity') == identity:
        done = time.time() if now is None else cur_now
        status.update(last_success=done, next_due=(done + min_interval) if min_interval > 0 else None,
                      checkpoint=int(latest.name), commit=git('rev-parse', 'HEAD'), push_pending=False, error=None)
        return True
    (ROOT / 'build').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / 'build', prefix='backup-') as tmp:
        staging = Path(tmp) / 'snapshot'
        staging.mkdir()
        for source, target in zip(sources, targets):
            shutil.copytree(source, staging / target)
        if before != [signature(p) for p in sources]:
            raise RuntimeError('Checkpoint changed during copy; retrying next poll')
        # Restore training from the selected checkpoint boundary, not a newer in-flight heartbeat.
        steps = int(latest.name)
        history = [e for e in state['history'] if e['steps'] <= steps]
        if state.get('pending') and state['pending']['start_steps'] <= steps:
            if state['status'] != 'paused':
                state['pending']['seconds'] = 0  # crash snapshot: don't claim unbacked trainer time
                state['pending']['trained'] = False
        state['history'] = history
        state['status'] = 'paused' if state['status'] != 'complete' else 'complete'
        (staging / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        if not evolving:
            for folder in ('metrics', 'eval'):
                shutil.copytree(campaign / folder, staging / folder)
        hashes = {str(p.relative_to(staging)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(staging.rglob('*')) if p.is_file()}
        created_ts = time.time() if now is None else cur_now
        (staging / 'manifest.json').write_text(json.dumps({'checkpoint': steps, 'identity': identity,
            'created': created_ts, 'sha256': hashes, 'source_commit': git('rev-parse', 'HEAD')}, indent=2) + '\n')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(staging, destination)
    relative = str(destination.relative_to(ROOT))
    git('add', '--', relative)
    if git('diff', '--cached', '--name-only', '--', relative):
        git('commit', '--only', '-m', f'Back up ranked checkpoint {latest.name}', '--', relative)
        status['push_pending'] = True
    try:
        git('push', 'origin', branch)
    except subprocess.CalledProcessError:
        git('pull', '--rebase', '--autostash', 'origin', branch)
        git('push', 'origin', branch)
    commit = git('rev-parse', 'HEAD')
    # Verify the remote points at the pushed commit, not merely that a local commit exists.
    if git('ls-remote', 'origin', 'refs/heads/' + branch).split()[0] != commit:
        raise RuntimeError('Remote commit verification failed')
    done = time.time() if now is None else cur_now
    status.update(last_success=done, next_due=(done + min_interval) if min_interval > 0 else None,
                  checkpoint=steps, commit=commit, push_pending=False, error=None)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--interval', type=int, default=DEFAULT_BACKUP_INTERVAL,
                        help='Minimum seconds between automatic GitHub backups (default: 1800 = 30 minutes)')
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    interval = max(60, int(args.interval))
    os.chdir(ROOT)
    (ROOT / 'build').mkdir(exist_ok=True)
    lock = (ROOT / 'build/training-service.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    branch = git('branch', '--show-current')
    dest = ROOT / 'training-backup'
    prior_ts = read_manifest_created(dest)
    status = {
        'enabled': True,
        'interval_seconds': interval,
        'last_success': prior_ts,
        'next_due': (prior_ts + interval) if prior_ts is not None else None,
        'push_pending': False,
        'error': None,
    }
    print(f'GitHub checkpoint backup worker active (interval={interval}s, branch={branch})', flush=True)
    while True:
        wait_s = seconds_until_backup_due(dest, status, interval)
        if wait_s > 0:
            status['next_due'] = time.time() + wait_s
            time.sleep(wait_s)
            continue
        try:
            ran = backup(args.campaign, dest, branch, status, min_interval=interval)
            if ran:
                print('Backup verified:', json.dumps(status), flush=True)
            time.sleep(max(1.0, seconds_until_backup_due(dest, status, interval)))
        except Exception as exc:
            status['error'] = str(exc)
            print('Backup pending/failed:', str(exc), flush=True)
            time.sleep(min(BACKUP_RETRY_SECONDS, interval))


if __name__ == '__main__':
    main()
