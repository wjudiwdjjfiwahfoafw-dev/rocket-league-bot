#!/usr/bin/env python3
"""Run the predeclared, bounded weight-only 1700 experiment sequence.
No calibration, temperature override or deterministic evaluation is invoked.
"""
import argparse
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
    audit = (state.get('audit') or {}).get(str(best['steps']), {})
    return all(v >= 1700 for v in (best['screen']['Ranking'], best['confirm']['Ranking'], audit.get('Ranking', 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curriculum-only', action='store_true', help='Skip the original continuation and branch a fresh recipe from its selected native best')
    args = parser.parse_args()
    (ROOT / 'build').mkdir(exist_ok=True)
    with (ROOT / 'build/weight-target.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        restore()
        driver = [sys.executable, 'tools/train_ranked.py']
        if not args.curriculum_only:
            run(driver + ['--base', 'checkpoints/3043214022', '--out', str(WORK.relative_to(ROOT)),
                          '--minutes', '180', '--rounds', '40', '--target-ranking', '1700',
                          '--monitor', '--github-backup', '--backup-interval', '1800'])
        backup(WORK, ROOT / 'training-backup', git('branch', '--show-current'), {})
        state = json.loads((WORK / 'state.json').read_text())
        if state['status'] != 'complete' and not args.curriculum_only:
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
        rewards = (
            'Air=0,FaceBall=0,VelocityPlayerToBall=0,NearestVelocityToBall=0,TeamVelocityToBall=0,'
            'StrongTouch=0,TouchAccel=0,VelocityBallToGoal=0,BehindBall=0,TeamSpacing=0,PickupBoost=0,'
            'Goal=600,GoalSpeed=150,ScorerAssist=100,Save=65,Shot=40,DirectionalTouch=35,'
            'ObjectiveBallVel=7.0,KickoffSpeed=4.0,TeamPressure=1.5,DefensiveCover=0.65,DoubleCommit=1.8,'
            'PickupBoostZS=8.0,SaveBoost=0.04,Demo=30,Bump=4'
        )
        out = 'results/weights1700-v2/campaign'
        run(driver + ['--base', str(source.relative_to(ROOT)), '--reference-pool', 'training-backup/pool',
                      '--out', out, '--minutes', '180', '--pilot-minutes', '12', '--block-minutes', '20',
                      '--rounds', '40', '--audit-seed', '9007', '--target-ranking', '1700',
                      '--epochs', '2', '--lr', '0.00011', '--critic-lr', '0.0002',
                      '--entropy-scale', '0.0025', '--mask-entropy',
                      '--norm-advantages', 'mode', '--target-kl', '0.02',
                      '--return-stat-max-count', '200000',
                      '--random-state-prob', '0', '--tactical-state-prob', '0.22', '--fuzzed-kickoffs',
                      '--train-vs-old', '0.30', '--pfsp', '--ts-per-version', '12m', '--max-versions', '16',
                      '--ts-per-save', '10m', '--rewards', rewards,
                      '--monitor', '--github-backup', '--backup-interval', '1800'])
        backup(ROOT / out, ROOT / 'training-backup', git('branch', '--show-current'), {})
        final = json.loads((ROOT / out / 'state.json').read_text())
        print('1700 WEIGHT-ONLY TARGET VERIFIED' if achieved(final) else
              'Bounded experiment finished without a verified 1700 result; inspect state.json.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except InterruptedError:
        print('Weight-only experiments paused safely.', flush=True)
