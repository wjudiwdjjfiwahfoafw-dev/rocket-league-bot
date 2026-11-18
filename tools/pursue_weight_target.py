#!/usr/bin/env python3
"""Run the predeclared, bounded weight-only 1700 experiment sequence.
No calibration, temperature override or deterministic evaluation is invoked.
"""
import fcntl
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from resume_training import ROOT, WORK, restore
from training_service import backup, git

child = None
stopping = False


def stop(*_):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.terminate()  # driver requests native checkpoint before exiting


def run(command):
    global child
    if stopping:
        raise InterruptedError
    child = subprocess.Popen(command, cwd=ROOT)
    result = child.wait()
    if stopping:
        raise InterruptedError
    if result:
        raise RuntimeError(f'Experiment command exited {result}')


def achieved(state):
    best = state['best']
    audit = state.get('audit', {}).get(str(best['steps']), {})
    return all(v >= 1700 for v in (best['screen']['Ranking'], best['confirm']['Ranking'], audit.get('Ranking', 0)))


def main():
    (ROOT / 'build').mkdir(exist_ok=True)
    with (ROOT / 'build/weight-target.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        restore()
        driver = [sys.executable, 'tools/train_ranked.py']
        run(driver + ['--base', 'checkpoints/3043214022', '--out', str(WORK.relative_to(ROOT)),
                      '--minutes', '180', '--rounds', '40', '--target-ranking', '1700',
                      '--monitor', '--github-backup'])
        backup(WORK, ROOT / 'training-backup', git('branch', '--show-current'), {})
        state = json.loads((WORK / 'state.json').read_text())
        if state['status'] != 'complete':
            print('Original campaign paused; not starting another trainer.', flush=True)
            return
        if achieved(state):
            print('1700 WEIGHT-ONLY TARGET VERIFIED', flush=True)
            return
        steps = state['best']['steps']
        source = ROOT / 'checkpoints' / str(steps)
        if not source.exists():
            shutil.copytree(WORK / 'best' / str(steps), source)
            branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
            relative = str(source.relative_to(ROOT))
            subprocess.run(['git', 'add', '--', relative], cwd=ROOT, check=True)
            subprocess.run(['git', 'commit', '--only', '-m', f'Preserve native PPO source {steps} for weight-only curriculum', '--', relative], cwd=ROOT, check=True)
            subprocess.run(['git', 'push', 'origin', branch], cwd=ROOT, check=True)
        rewards = ('VelocityBallToGoal=8,Goal=600,VelocityPlayerToBall=0,NearestVelocityToBall=0,'
                   'TeamSpacing=1,TouchAccel=5,BehindBall=0,FaceBall=0,StrongTouch=25,'
                   'PickupBoost=2,SaveBoost=0.2,Bump=10,Demo=20')
        out = 'results/weights1700/campaign'
        run(driver + ['--base', str(source.relative_to(ROOT)), '--reference-pool', 'training-backup/pool',
                      '--out', out, '--minutes', '120', '--pilot-minutes', '12', '--block-minutes', '18',
                      '--rounds', '40', '--audit-seed', '9001', '--target-ranking', '1700',
                      '--lr', '0.00008', '--random-state-prob', '0.25', '--train-vs-old', '0.35',
                      '--rewards', rewards, '--monitor', '--github-backup'])
        backup(ROOT / out, ROOT / 'training-backup', git('branch', '--show-current'), {})
        final = json.loads((ROOT / out / 'state.json').read_text())
        print('1700 WEIGHT-ONLY TARGET VERIFIED' if achieved(final) else
              'Bounded experiment finished without a verified 1700 result; inspect state.json.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except InterruptedError:
        print('Weight-only experiments paused safely.', flush=True)
