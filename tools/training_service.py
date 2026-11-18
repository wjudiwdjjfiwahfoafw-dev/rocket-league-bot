#!/usr/bin/env python3
"""Sandbox metrics viewer and GitHub checkpoint backup worker. No external assets."""
import argparse
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
FILES = {'CRITIC.lt', 'CRITIC_OPTIM.lt', 'POLICY.lt', 'POLICY_OPTIM.lt',
         'SHARED_HEAD.lt', 'SHARED_HEAD_OPTIM.lt', 'SHARED_HEAD_SLOTS.lt',
         'INFO.json', 'RUNNING_STATS.json'}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def signature(path):
    return [(str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(path.rglob('*')) if p.is_file()]


def assess_goal(state, target=1600.0):
    best = state.get('best') or {}
    screen = (best.get('screen') or {}).get('Ranking')
    confirm = (best.get('confirm') or {}).get('Ranking')
    audit = ((state.get('audit') or {}).get(str(best.get('steps'))) or {}).get('Ranking')
    selection_met = all(v is not None and v >= target for v in (screen, confirm))
    return {'target': target, 'screen': screen, 'confirmation': confirm, 'audit': audit,
            'selection_met': selection_met,
            'verified': selection_met and audit is not None and audit >= target,
            'screen_gap': max(0.0, target - screen) if screen is not None else None}


def snapshot(campaign):
    state = json.loads((campaign / 'state.json').read_text())
    rows = {}
    for path in sorted((campaign / 'metrics').glob('*.jsonl')):
        for line in path.read_text().splitlines(keepends=True):
            if line.endswith('\n') and line.strip():
                row = json.loads(line)
                rows[int(row['Total Timesteps'])] = row
    return {'state': state, 'goal': assess_goal(state), 'rows': [rows[k] for k in sorted(rows)]}


def backup(campaign, destination, branch, status):
    if git('branch', '--show-current') != branch:
        raise RuntimeError('Branch changed; refusing backup')
    # Retry a previous failed push even when there is no new checkpoint.
    git('push', 'origin', branch)
    candidates = sorted((p for p in (campaign / 'learner').iterdir() if p.name.isdecimal()), key=lambda p: int(p.name))
    latest = candidates[-1]
    state = json.loads((campaign / 'state.json').read_text())
    sources = [latest, campaign / 'learner/policy_versions', campaign / 'pool',
               campaign / 'best' / str(state['best']['steps'])]
    before = [signature(p) for p in sources]
    if any(not sig for sig in before) or not FILES.issubset({p.name for p in latest.iterdir()}):
        raise RuntimeError('Checkpoint incomplete; waiting for next poll')
    if any(time.time_ns() - mtime < 10_000_000_000 for sig in before for _, _, mtime in sig):
        raise RuntimeError('Checkpoint files still settling; waiting for next poll')
    identity = hashlib.sha256(json.dumps({'files': before, 'history': state['history'],
        'best': state['best'], 'audit': state.get('audit'), 'status': state['status']}, sort_keys=True).encode()).hexdigest()
    marker = destination / 'manifest.json'
    if marker.exists() and json.loads(marker.read_text())['identity'] == identity:
        status.update(last_success=time.time(), checkpoint=int(latest.name), commit=git('rev-parse', 'HEAD'), error=None)
        return
    with tempfile.TemporaryDirectory(dir=ROOT / 'build', prefix='backup-') as tmp:
        staging = Path(tmp) / 'snapshot'
        staging.mkdir()
        for source, target in zip(sources, ['learner/' + latest.name, 'learner/policy_versions', 'pool', 'best/' + str(state['best']['steps'])]):
            shutil.copytree(source, staging / target)
        if before != [signature(p) for p in sources]:
            raise RuntimeError('Checkpoint changed during copy; retrying next poll')
        # Restore training from the selected checkpoint boundary, not a newer in-flight heartbeat.
        steps = int(latest.name)
        history = [e for e in state['history'] if e['steps'] <= steps]
        if state.get('pending') and state['pending']['start_steps'] <= steps:
            state['pending']['seconds'] = 0  # conservative: never claim unbacked training time
            state['pending']['trained'] = False
        state['history'] = history
        state['status'] = 'paused' if state['status'] != 'complete' else 'complete'
        (staging / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        for folder in ('metrics', 'eval'):
            shutil.copytree(campaign / folder, staging / folder)
        hashes = {str(p.relative_to(staging)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in staging.rglob('*') if p.is_file()}
        (staging / 'manifest.json').write_text(json.dumps({'checkpoint': steps, 'identity': identity,
            'created': time.time(), 'sha256': hashes, 'source_commit': git('rev-parse', 'HEAD')}, indent=2) + '\n')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(staging, destination)
    relative = str(destination.relative_to(ROOT))
    git('add', '--', relative)
    if git('diff', '--cached', '--name-only', '--', relative):
        git('commit', '--only', '-m', f'Back up ranked checkpoint {latest.name}', '--', relative)
    git('push', 'origin', branch)
    commit = git('rev-parse', 'HEAD')
    # Verify the remote points at the pushed commit, not merely that a local commit exists.
    if git('ls-remote', 'origin', 'refs/heads/' + branch).split()[0] != commit:
        raise RuntimeError('Remote commit verification failed')
    status.update(last_success=time.time(), checkpoint=steps, commit=commit, error=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--backup', action='store_true', help='Commit checkpoint snapshots and push current branch')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    os.chdir(ROOT)
    (ROOT / 'build').mkdir(exist_ok=True)
    lock = (ROOT / 'build/training-service.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    branch = git('branch', '--show-current')
    status = {'enabled': args.backup, 'last_success': None, 'error': None}
    def worker():
        while True:
            try:
                backup(args.campaign, ROOT / 'training-backup', branch, status)
                print('Backup verified:', json.dumps(status), flush=True)
            except Exception as exc:
                status['error'] = str(exc)
                print('Backup pending/failed:', str(exc), flush=True)
            time.sleep(max(10, args.interval))
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                data = (ROOT / 'tools/training_view.html').read_bytes()
                content_type = 'text/html; charset=utf-8'
            elif self.path == '/status.json':
                data = json.dumps({**snapshot(args.campaign), 'backup': dict(status)}, allow_nan=False).encode()
                content_type = 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    if args.backup:
        threading.Thread(target=worker, daemon=True).start()
    print(f'Training metrics and backup status on port {args.port}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
