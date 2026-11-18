#!/usr/bin/env python3
"""Evaluate copied, calibrated policies against the unchanged campaign frozen pool.
Never changes source weights, opponents, fixtures, sampling code or rating rules.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import train_ranked as ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.65, 0.4, 1.3])
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, ranked.stop)
    signal.signal(signal.SIGINT, ranked.stop)
    pool = args.campaign / 'pool'
    digest = ranked.pool_digest(pool)
    env = dict(os.environ, OMP_WAIT_POLICY='passive', GGL_WANDB='0')
    results = []
    for temperature in args.temperatures:
        trial = args.out / ('temperature-' + str(temperature))
        path = trial / 'weights' / args.base.name
        trial.mkdir(parents=True, exist_ok=True)
        if not (trial / 'pool').exists():
            (trial / 'pool').symlink_to(pool.resolve(), target_is_directory=True)
        (trial / 'eval').mkdir(exist_ok=True)
        logdir = Path('build/calibration-logs') / trial.name
        logdir.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            shutil.copytree(args.base, path)
            subprocess.run(['build/calibrate_policy', str(args.base / 'POLICY.lt'),
                            str(path / 'POLICY.lt'), str(temperature)], check=True)
            provenance = {'source': str(args.base), 'temperature': temperature,
                          'source_policy_sha256': hashlib.sha256((args.base / 'POLICY.lt').read_bytes()).hexdigest(),
                          'policy_sha256': hashlib.sha256((path / 'POLICY.lt').read_bytes()).hexdigest(),
                          'optimizer_note': 'Original optimizer archives retained; transformed weights have not been trained.'}
            (trial / 'calibration.json').write_text(json.dumps(provenance, indent=2) + '\n')
        scoring = argparse.Namespace(out=trial, bot=Path('build/GigaLearnBot'), threads=2)
        screen = ranked.score(scoring, path, 1009, 40, digest, env, logdir)
        result = {'temperature': temperature, 'screen': screen}
        # Confirm candidates beating the selected original baseline; audit only potential target models.
        if screen['Ranking'] > 1543.1810031467014:
            result['confirm'] = ranked.score(scoring, path, 2003, 80, digest, env, logdir)
        if screen['Ranking'] >= 1600 and result.get('confirm', {}).get('Ranking', 0) >= 1600:
            result['audit'] = ranked.score(scoring, path, 4001, 80, digest, env, logdir)
        results.append(result)
        (args.out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print('CALIBRATION RESULT:', json.dumps(result), flush=True)
        if result.get('audit', {}).get('Ranking', 0) >= 1600:
            print('TARGET VERIFIED', str(path), flush=True)
            break


if __name__ == '__main__':
    main()
