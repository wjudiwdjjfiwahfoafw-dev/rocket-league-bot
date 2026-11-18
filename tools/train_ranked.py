#!/usr/bin/env python3
"""Run a bounded CPU training campaign. Rerun the same command to resume. Linux, stdlib only
unless --wandb-offline is requested. See RANKED_TRAINING.md for the selection protocol.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
MODES = ('1v1', '2v2', '3v3')
# Predeclared selection and confirmation seeds; the audit seed is not used for selection.
SCREEN, CONFIRM, AUDIT = 1009, 2003, 4001
MODELS = ('POLICY', 'CRITIC', 'SHARED_HEAD')
child = None
stopping = False


def stop(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)  # native learner finishes the iteration and saves


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def checkpoint(path):
    stats = read(path / 'RUNNING_STATS.json')
    info = read(path / 'INFO.json')
    if stats['total_timesteps'] != int(path.name) or info['model'] != 's' or stats['mirror_x'] != -1:
        raise ValueError(f'Not a matching full S checkpoint: {path}')
    for name in (*MODELS, *(k + '_OPTIM' for k in MODELS), 'SHARED_HEAD_SLOTS'):
        if not (path / (name + '.lt')).stat().st_size:
            raise ValueError(f'Empty {name} in {path}')
    return path


def newest(root):
    return checkpoint(max((p for p in root.iterdir() if p.is_dir() and p.name.isdecimal()), key=lambda p: int(p.name)))


def pool_digest(root):
    h = hashlib.sha256()
    for p in sorted(root.glob('*/*')):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + b'\0' + p.read_bytes())
    return h.hexdigest()


def run(cmd, env, log, heartbeat=None):
    global child
    if stopping:
        raise InterruptedError
    with log.open('w') as f:
        child = subprocess.Popen(list(map(str, cmd)), stdin=subprocess.DEVNULL, stdout=f,
                                 stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if heartbeat:
                        heartbeat()
            if stopping:
                raise InterruptedError
            if child.returncode:
                raise RuntimeError(f'Trainer exited {child.returncode}; see {log}\n{log.read_text()[-1500:]}')
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
            child = None


def policy_digest(path):
    """Fingerprint actual inference weights, not their timestep directory name."""
    digest = hashlib.sha256()
    for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt'):
        digest.update(name.encode() + b'\0' + (path / name).read_bytes())
    return digest.hexdigest()


def validate_execution(result):
    settings = result.get('settings', {})
    if settings.get('deterministic') is not False or settings.get('policy_temperature') != 1.0:
        raise ValueError('Ranked training requires sampled actions at temperature 1; no evaluation calibration')


def score(args, path, seed, rounds, pool_hash, env, logdir):
    """Cache only a complete eval of this immutable step, pool and protocol."""
    out = args.out / 'eval' / f'{path.name}-{seed}-{rounds}.json'
    weights_sha = policy_digest(path)
    cache_valid = out.exists() and read(out).get('policy_sha256') == weights_sha
    if not cache_valid:
        # --checkpoints takes a *parent*. Give it exactly one checkpoint, even to rate an old best.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / path.name).symlink_to(path.resolve(), target_is_directory=True)
            run([args.bot, '--device', 'cpu', '--model', 's', '--modes', ','.join(MODES), '--games', '3',
                 '--checkpoints', tmp, '--rate-anchored', '--rate-versions', args.out / 'pool',
                 '--rate-rounds', rounds, '--skill-arenas', '36', '--seed', seed, '--threads', args.threads,
                 '--rate-out', tmp / 'score.json'], env, logdir / 'eval.log')
            result = read(tmp / 'score.json')
            result.update(checkpoints=str(path), pool_sha256=pool_hash, policy_sha256=weights_sha)
    else:
        result = read(out)
    validate_execution(result)
    if policy_digest(path) != weights_sha:
        raise ValueError('Policy weights changed during evaluation')
    if (not result['anchored'] or result['seed'] != seed or result['rounds'] != rounds or
            result['pool_sha256'] != pool_hash or result['checkpoint_timesteps'] != int(path.name)):
        raise ValueError(f'Incomplete or mismatched rating: {out}')
    if pool_digest(args.out / 'pool') != pool_hash:
        raise ValueError('Frozen pool changed during campaign')
    ratings = checked_ranking(result, args.out / 'pool')
    if not cache_valid:
        write(out, result)
    print(f'{path.name} seed {seed}: {ratings}', flush=True)
    return ratings


def checked_ranking(result, pool):
    # Fail closed: a partial match, drifting opponent or 2-mode score cannot promote a bot.
    refs = {int(p.parent.name): read(p)['skill_ratings'] for p in pool.glob('*/STATS.json')}
    got = {o['timesteps']: o for o in result['opponents']}
    if not refs or got.keys() != refs.keys():
        raise ValueError('The anchored pool did not play every opponent')
    for step, ref in refs.items():
        if got[step]['rounds'] * len(refs) != result['rounds'] or any(
                not math.isclose(got[step]['ratings'][m], ref[m], abs_tol=1e-4) for m in MODES):
            raise ValueError('Unbalanced fixtures or opponent rating drift')
    ratings = {m: result['ratings'][m]['average'] for m in MODES}
    ratings['Ranking'] = sum(ratings.values()) / len(MODES)
    for m in MODES:
        if sum(result['mode_goals'].get(m, {}).values()) <= 0:
            raise ValueError(f'No goals in {m}; cannot select a checkpoint')
    if not math.isclose(ratings['Ranking'], result['ratings']['Ranking']['average'], abs_tol=1e-5):
        raise ValueError('Ranking is not the three-mode mean')
    return ratings


def minutes_plan(total, pilot, block):
    minutes = [min(total, pilot)]
    while sum(minutes) < total:
        minutes.append(min(total - sum(minutes), block))
    return minutes


def promote(screen, confirm, best):
    if best is None:
        return True
    a = screen['Ranking'] - best['screen']['Ranking']
    b = confirm['Ranking'] - best['confirm']['Ranking']
    return a > 0 and b > 0 and (a + b) / 2 >= 8


def campaign(args):
    out, learner = args.out, args.out / 'learner'
    base = checkpoint(args.base)
    if not args.bot.is_file():
        raise ValueError(f'Build {args.bot} before starting the campaign')
    plan = minutes_plan(args.minutes, args.pilot_minutes, args.block_minutes)
    train_args = ['--threads', str(args.threads), '--games', '88', '--mode-balance', 'players',
                  '--ts-per-itr', '48384', '--batch-size', '12096',
                  '--fixed-rollout', '--lean-metrics', '--minibatch-size', str(args.minibatch),
                  '--lr', '1e-4', '--skill-minutes', '15', '--ts-per-save', '5m', '--keep-checkpoints', '2']
    # Explicit recipe overrides make follow-up trials reproducible without editing presets.
    overrides = [('lr', '--lr'), ('entropy_scale', '--entropy-scale'),
                 ('random_state_prob', '--random-state-prob'), ('train_vs_old', '--train-vs-old')]
    for attribute, flag in overrides:
        value = getattr(args, attribute, None)
        if value is not None:
            if flag in train_args:
                train_args[train_args.index(flag) + 1] = str(value)
            else:
                train_args.extend([flag, str(value)])
    if getattr(args, 'rewards', None):
        train_args.extend(['--rewards', args.rewards])
    config = dict(base=str(base), preset_sha256=hashlib.sha256((REPO / 'train_1v1_2v2_3v3.sh').read_bytes()).hexdigest(),
                  parent_steps=int(base.name), seed=args.seed, screen_seed=SCREEN,
                  confirm_seed=CONFIRM, audit_seed=getattr(args, 'audit_seed', AUDIT), rounds=args.rounds, train_args=train_args,
                  minutes=plan, modes=list(MODES), protocol='anchored-v1')
    if getattr(args, 'reference_pool', None) is not None:
        config['reference_pool'] = str(args.reference_pool)
    statefile = out / 'state.json'
    if not statefile.exists():
        if any(p.name != '.lock' for p in out.iterdir()):
            raise ValueError(f'Refusing to overwrite a non-empty campaign directory: {out}')
        src_pool = getattr(args, 'reference_pool', None) or (base.parent / 'policy_versions')
        count = len([p for p in src_pool.iterdir() if p.name.isdecimal() and p.is_dir()])
        if not count or args.rounds % (2 * count):
            raise ValueError('--rounds must be a multiple of twice the size of the (nonempty) pool')
        shutil.copytree(src_pool, out / 'pool')
        shutil.copytree(src_pool, learner / 'policy_versions')
        shutil.copytree(base, learner / base.name)
        (out / 'eval').mkdir()
        (out / 'best').mkdir()
        state = dict(config=config, pool_sha256=pool_digest(out / 'pool'), history=[], best=None,
                     pending=None, status='initialized')
        write(statefile, state)
    state = read(statefile)
    if config != state['config'] or pool_digest(out / 'pool') != state['pool_sha256']:
        raise ValueError('Campaign settings or frozen pool changed. Use the original settings or a fresh --out.')
    if getattr(args, 'target_ranking', None) is not None:
        state['target_ranking'] = args.target_ranking
        write(statefile, state)
    if state['status'] == 'complete':
        print('Campaign already complete:', statefile)
        return
    logdir = REPO / 'build' / ('rank-logs-' + hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:10])
    logdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GGL_WANDB='0', GGL_METRICS_EVERY='20', GGL_METRICS_DIR=str(out / 'metrics'),
               OMP_WAIT_POLICY='passive', BUILD_DIR=str(args.bot.parent))
    wb = None
    if args.wandb_offline:
        import ranked_wandb
        wb = ranked_wandb.start(out, config, f'ranking-{base.name}')
        state['wandb_dir'] = str(Path(wb.dir).parent.relative_to(REPO)) if Path(wb.dir).is_relative_to(REPO) else str(Path(wb.dir).parent)
    service = None
    if getattr(args, 'monitor', False) or getattr(args, 'github_backup', False):
        service_cmd = [sys.executable, str(REPO / 'tools/training_service.py'), str(out),
                       '--port', str(args.monitor_port)]
        if args.github_backup:
            service_cmd.append('--backup')
        service = subprocess.Popen(service_cmd)
    streamer = None
    if getattr(args, 'wandb_online', False):
        streamer = subprocess.Popen([sys.executable, str(REPO / 'tools/ranked_wandb.py'),
                                     str(out), '--out', str(out / 'online-stream'), '--online'])
    evaluate = lambda p, seed, n: score(args, p, seed, n, state['pool_sha256'], env, logdir)
    try:
        # Stage 0 is the parent, not a trained candidate. Later stages always continue the latest
        # weights, irrespective of short-term rating noise. Only the independent best copy is selected.
        for i, minutes in enumerate([0, *plan]):
            if i < len(state['history']):
                continue
            if state['pending'] is None:
                state['pending'] = dict(stage=i, start_steps=int(newest(learner).name), seconds=0, trained=False)
            pending = state['pending']
            if minutes and not pending['trained']:
                state['status'] = 'training'
                write(statefile, state)
                start, previous = time.monotonic(), pending['seconds']
                def heartbeat():
                    pending['seconds'] = previous + time.monotonic() - start
                    write(statefile, state)
                cmd = ['bash', 'train_1v1_2v2_3v3.sh', '--checkpoints', learner, *train_args,
                       '--seed', args.seed + i, '--run-name', 'three-mode-ranking',
                       '--max-minutes', max(0.02, minutes - previous / 60)]
                if int(newest(learner).name) == int(base.name):
                    cmd.append('--new-run')
                print(f'Stage {i}/{len(plan)}: training from {newest(learner).name} ({minutes} min budget)', flush=True)
                try:
                    run(cmd, env, logdir / f'train-{i}.log', heartbeat)
                finally:
                    heartbeat()
                pending['trained'] = True
            state['status'] = 'evaluating'
            write(statefile, state)
            ckpt = newest(learner)
            steps = int(ckpt.name)
            if minutes and steps <= pending['start_steps']:
                raise ValueError('Training did not advance')
            screen = evaluate(ckpt, SCREEN, args.rounds)
            best = state['best']
            confirm = None
            if best is None or screen['Ranking'] > best['screen']['Ranking']:
                confirm = evaluate(ckpt, CONFIRM, 2 * args.rounds)
            won = confirm is not None and promote(screen, confirm, best)
            if won:
                dest = out / 'best' / ckpt.name
                if not dest.exists():
                    tmp = dest.with_suffix('.copying')
                    shutil.copytree(ckpt, tmp, dirs_exist_ok=True)
                    tmp.rename(dest)
                state['best'] = dict(steps=steps, screen=screen, confirm=confirm)
            seconds = pending['seconds']
            entry = dict(stage=i, start_steps=pending['start_steps'], steps=steps, train_seconds=seconds,
                         steps_per_second=(steps - pending['start_steps']) / seconds if seconds else None,
                         screen=screen, confirm=confirm, promoted=won)
            state['history'].append(entry)
            state['pending'] = None
            write(statefile, state)
            for p in (out / 'best').iterdir():
                if p.name.isdecimal() and int(p.name) != state['best']['steps']:
                    shutil.rmtree(p)
            if wb is not None:
                ranked_wandb.log_stage(wb, out, entry, state['best']['steps'])
            print(f'Stage {i} complete. Best: {state["best"]["steps"]}; latest: {steps}', flush=True)
        # An untouched seed measures the selected checkpoint against the original baseline on the same scale.
        state['status'] = 'auditing'
        write(statefile, state)
        state['audit'] = {p.name: evaluate(p, config['audit_seed'], 2 * args.rounds) for p in
                          (base, out / 'best' / str(state['best']['steps']), newest(learner))}
        if wb is not None:
            ranked_wandb.log_audit(wb, state)
        state['status'] = 'complete'
    except InterruptedError:
        state['status'] = 'paused'
        print('Paused; rerun the same command to resume.', flush=True)
    except BaseException:
        state['status'] = 'failed'
        raise
    finally:
        write(statefile, state)
        if service is not None:
            service.terminate()
            service.wait(timeout=30)
        if streamer is not None:
            try:
                streamer.wait(timeout=15)
            except subprocess.TimeoutExpired:
                streamer.terminate()
                streamer.wait(timeout=15)
            if streamer.returncode:
                print('W&B online streamer exited unsuccessfully; JSON metrics remain saved.', flush=True)
        if wb is not None:
            wb.finish(exit_code=int(state['status'] == 'failed'))


def main():
    os.chdir(REPO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Full numbered S checkpoint with a sibling policy_versions pool')
    parser.add_argument('--out', type=Path, required=True, help='Fresh campaign directory or a matching campaign to resume')
    parser.add_argument('--bot', type=Path, default=Path('build/GigaLearnBot'))
    for name, default in [('minutes', 180), ('pilot-minutes', 6), ('block-minutes', 30), ('rounds', 40),
                          ('minibatch', 1008), ('threads', 2), ('seed', 20261118)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--wandb-offline', action='store_true', help='Write a syncable W&B run in --out; no credential or network needed')
    parser.add_argument('--wandb-online', action='store_true', help='Tail metrics to W&B online in a separate sandbox process; requires W&B credentials/network')
    parser.add_argument('--monitor', action='store_true', help='Serve all recorded metrics in a browser')
    parser.add_argument('--monitor-port', type=int, default=8080)
    parser.add_argument('--github-backup', action='store_true', help='Automatically commit resumable snapshots and push the current branch (also starts monitor)')
    parser.add_argument('--reference-pool', type=Path, help='Explicit immutable opponent pool for a fresh experiment')
    parser.add_argument('--audit-seed', type=int, default=AUDIT, help='Predeclared held-out audit seed')
    parser.add_argument('--target-ranking', type=float, help='Report a weight-only temperature-1 Ranking target; does not change immutable training settings')
    parser.add_argument('--lr', type=float, help='Explicit learning-rate override for a fresh experiment')
    parser.add_argument('--entropy-scale', type=float)
    parser.add_argument('--random-state-prob', type=float)
    parser.add_argument('--train-vs-old', type=float)
    parser.add_argument('--rewards', help='Native comma-separated reward weights for a fresh experiment')
    args = parser.parse_args()
    if args.target_ranking is not None and not (0 < args.target_ranking < 10000):
        parser.error('--target-ranking must be finite and between 0 and 10000')
    if args.lr is not None and not (0 < args.lr <= 0.01):
        parser.error('--lr must be in (0, 0.01]')
    if args.entropy_scale is not None and not (0 <= args.entropy_scale <= 1):
        parser.error('--entropy-scale must be in [0, 1]')
    for value in (args.random_state_prob, args.train_vs_old):
        if value is not None and not (0 <= value <= 1):
            parser.error('Start/opponent probabilities must be in [0, 1]')
    if args.bot.name != 'GigaLearnBot' or min(args.minutes, args.pilot_minutes, args.block_minutes, args.rounds, args.threads, args.minibatch) <= 0:
        parser.error('Budgets, rounds, threads and minibatch must be positive; --bot must name GigaLearnBot')
    if args.out.resolve().is_relative_to(args.base.parent.resolve()):
        parser.error('The campaign directory must be outside the source checkpoint tree')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, 'A campaign is already running here.\n')
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        campaign(args)


if __name__ == '__main__':
    main()
