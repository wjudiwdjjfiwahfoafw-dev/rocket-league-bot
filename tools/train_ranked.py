#!/usr/bin/env python3
"""Run a staged CPU training campaign, optionally repeating blocks until stopped. Rerun to resume.
Linux, stdlib only unless --wandb-offline is requested. See RANKED_TRAINING.md for the protocol.
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


def policy_versions(root):
    return sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdecimal()),
                  key=lambda p: int(p.name))


def pool_size(pool):
    """Number of policies in the opponent pool; the native rating tournament is sized by it."""
    return len(policy_versions(pool))


def rounds_for_pool(base_rounds, count):
    """Round count the native anchored tournament accepts against a pool of `count` policies.

    The native skill tracker requires the rating rounds to be a multiple of twice the number of
    loaded policy versions, so every opponent is played equally often on both team colors. The
    opponent pool of an evolving campaign grows while it runs, so the round count is derived from
    the pool size at evaluation time instead of being fixed for the whole campaign. The derived
    count is never below the requested budget, and it is a pure function of the budget and the
    pool size, so a resumed campaign rates exactly the same rounds as the run before it.
    """
    if count < 1:
        raise ValueError('The opponent pool must contain at least one policy')
    if base_rounds < 1:
        raise ValueError('The evaluation round budget must be positive')
    step = 2 * count
    return -(-base_rounds // step) * step


def file_hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.iterdir()) if p.is_file()}


def frozen_versions(root):
    """Content manifest of every saved policy version. A version may never change after creation."""
    return {int(v.name): file_hashes(v) for v in policy_versions(root)}


def verify_frozen_versions(root, expected):
    """Fail if any still-present old version differs from its creation-time manifest."""
    current = frozen_versions(root)
    expected = {int(steps): files for steps, files in expected.items()}
    changed = [steps for steps, files in expected.items()
               if steps in current and current[steps] != files]
    if changed:
        raise RuntimeError(f'Saved policy version(s) {changed} changed after creation '
                           '(no Elo or metadata updates are allowed)')
    return current


def sync_policy_versions(pool, destination):
    """Copy the pool into the trainer only when content differs; never touch identical files."""
    if destination.is_dir() and frozen_versions(destination) == frozen_versions(pool):
        return False
    replace_policy_pool(pool, destination)
    return True


def diverse_versions(source, count):
    """Select evenly spaced checkpoints so the seed pool spans the full historical range."""
    versions = policy_versions(source)
    if len(versions) < count:
        raise ValueError(f'Need {count} historical policies, found {len(versions)} in {source}')
    if count == 1:
        return [versions[-1]]
    indices = [round(i * (len(versions) - 1) / (count - 1)) for i in range(count)]
    return [versions[i] for i in indices]


def copy_policy_version(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt', 'STATS.json'):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    if not all((destination / name).is_file() for name in
               ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt', 'STATS.json')):
        if (source / 'RUNNING_STATS.json').is_file() and not (destination / 'STATS.json').exists():
            stats = read(source / 'RUNNING_STATS.json')
            write(destination / 'STATS.json', {'skill_ratings': stats.get('skill_ratings', {}),
                                               'mirror_x': stats.get('mirror_x', -1)})
        if not all((destination / name).is_file() for name in
                   ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt', 'STATS.json')):
            shutil.rmtree(destination, ignore_errors=True)
            raise ValueError(f'Incomplete policy version: {source}')


def seed_policy_pool(source, destination, count):
    destination.mkdir(parents=True, exist_ok=False)
    try:
        # A campaign can ask for a bigger pool than history offers. Seed with everything that exists (evenly
        # spaced, so it still spans the historical range) and let promotions grow the pool to `count`; nothing
        # is evicted until the pool actually exceeds `count`. This is what makes a 10 -> 15 pool change possible
        # without inventing historical policies that were never saved.
        available = len(policy_versions(source))
        for version in diverse_versions(source, min(count, available)):
            copy_policy_version(version, destination / version.name)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def replace_policy_pool(source, destination):
    """Atomically replace a native policy_versions folder from the managed opponent pool."""
    tmp = destination.with_name(destination.name + '.updating')
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(source, tmp)
    old = destination.with_name(destination.name + '.previous')
    shutil.rmtree(old, ignore_errors=True)
    if destination.exists():
        destination.rename(old)
    tmp.rename(destination)
    shutil.rmtree(old, ignore_errors=True)


def add_policy_to_pool(checkpoint_path, pool, count):
    """Add a qualified current policy, then evict the oldest timestamp FIFO once the pool is full."""
    target = pool / checkpoint_path.name
    if target.exists():
        return False
    copy_policy_version(checkpoint_path, target)
    versions = policy_versions(pool)
    # FIFO eviction only once the pool is over its target: a pool that is still filling up keeps every
    # member, and the promotion gate keeps comparing against whichever member is currently strongest.
    while len(versions) > count:
        shutil.rmtree(versions.pop(0))
    if len(versions) > count:
        raise ValueError(f'Evolving opponent pool cannot exceed {count} policies')
    return True


def strongest_pool_ranking(pool):
    ratings = []
    for version in policy_versions(pool):
        stats = read(version / 'STATS.json').get('skill_ratings', {})
        if not all(mode in stats for mode in MODES):
            raise ValueError(f'Missing three-mode Elo in {version / "STATS.json"}')
        ratings.append((sum(float(stats[m]) for m in MODES) / len(MODES), int(version.name)))
    if not ratings:
        raise ValueError('Opponent policy pool is empty')
    return max(ratings)


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
    if settings.get('fuzzed_kickoffs') is not True:
        # Without a randomized kickoff every round of a tournament would replay the same start state.
        raise ValueError('Rating requires fuzzed kickoffs; a round must not repeat its start state')


def checkpoint_elo(path):
    """The current policy's own Elo. A tournament continues from it; it is never reset to a pool value."""
    ratings = (read(path / 'RUNNING_STATS.json').get('skill_ratings') or {})
    return {mode: float(ratings[mode]) for mode in MODES if mode in ratings}


def write_back_elo(path, ratings):
    """A tournament's result belongs to the current policy, so its own metadata carries the new Elo."""
    stats = read(path / 'RUNNING_STATS.json')
    stats['skill_ratings'] = {mode: float(ratings[mode]) for mode in MODES}
    write(path / 'RUNNING_STATS.json', stats)


def score(args, path, seed, rounds, pool_hash, env, logdir, write_back=False):
    """Cache only a complete eval of this immutable step, pool and protocol."""
    out = args.out / 'eval' / f'{path.name}-{seed}-{rounds}.json'
    weights_sha = policy_digest(path)
    start = checkpoint_elo(path)
    cached = read(out) if out.exists() else None
    if cached is not None:
        try:
            validate_execution(cached)
        except ValueError:
            # The rating protocol changed since this result was cached: rate the checkpoint again.
            cached = None
    # A cached result stays valid either before or after its Elo was written back into the policy.
    cache_valid = cached is not None and all(cached.get(key) == value for key, value in (
        ('policy_sha256', weights_sha), ('pool_sha256', pool_hash),
        ('seed', seed), ('rounds', rounds), ('checkpoint_timesteps', int(path.name)))) and (
        cached.get('start_elo') == start or cached.get('end_elo') == start)
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
            # The tournament must have started from the Elo this policy already had, not from a shared anchor.
            for mode, value in result['ratings'].items():
                if mode in start and 'start' in value and not math.isclose(value['start'], start[mode], abs_tol=1e-3):
                    raise ValueError(f'{path.name} started {mode} at {value["start"]}, not at its own Elo {start[mode]}')
    else:
        result = cached
    validate_execution(result)
    if policy_digest(path) != weights_sha:
        raise ValueError('Policy weights changed during evaluation')
    if (not result['anchored'] or result['seed'] != seed or result['rounds'] != rounds or
            result['pool_sha256'] != pool_hash or result['checkpoint_timesteps'] != int(path.name)):
        raise ValueError(f'Incomplete or mismatched rating: {out}')
    if pool_digest(args.out / 'pool') != pool_hash:
        raise ValueError('Frozen pool changed during campaign')
    ratings = checked_ranking(result, args.out / 'pool')
    if write_back:
        # Only the current policy moves: its tournament result is stored in the policy itself, and it keeps
        #accumulating from there. Opponents are never written to.
        write_back_elo(path, ratings)
    if not cache_valid:
        # A cached checkpoint result is a record, not a rating journal: keep the current Elo only.
        result['ratings'] = ratings
        result['start_elo'] = start
        result['end_elo'] = {mode: ratings[mode] for mode in MODES}
        write(out, result)
    print(f'{path.name} seed {seed}: {ratings}', flush=True)
    return ratings


def endpoint_ratings(result):
    """Keep only the current (endpoint) Elo per mode; rating series and averages are discarded."""
    ratings = {}
    for mode in (*MODES, 'Ranking'):
        value = result['ratings'][mode]
        ratings[mode] = float(value['end'] if isinstance(value, dict) else value)
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
    # Report the exact endpoint Elo produced by the final rated round, never a smoothed
    # average across the rating session. The per-mode endpoint values are preserved verbatim.
    ratings = endpoint_ratings(result)
    if not math.isclose(ratings['Ranking'], sum(ratings[m] for m in MODES) / len(MODES), abs_tol=1e-5):
        raise ValueError('Ranking endpoint is not the three-mode endpoint mean')
    for m in MODES:
        if sum(result['mode_goals'].get(m, {}).values()) <= 0:
            raise ValueError(f'No goals in {m}; cannot select a checkpoint')
    return ratings


def minutes_plan(total, pilot, block):
    minutes = [min(total, pilot)]
    while sum(minutes) < total:
        minutes.append(min(total - sum(minutes), block))
    return minutes


def budget_for_stage(plan, stage, continuous=False, block=15):
    """Return a cycle budget; continuous runs repeat one fixed block after the parent rating."""
    if continuous:
        return 0 if stage == 0 else block
    budgets = [0, *plan]  # stage 0 rates the parent; training stages follow it
    return budgets[stage] if stage < len(budgets) else None


def promote(screen, confirm, best):
    if best is None:
        return True
    a = screen['Ranking'] - best['screen']['Ranking']
    b = confirm['Ranking'] - best['confirm']['Ranking']
    return a > 0 and b > 0 and (a + b) / 2 >= 8


def beats_strongest(screen, confirm, pool, margin):
    strongest, version = strongest_pool_ranking(pool)
    return (screen['Ranking'] >= strongest + margin and
            confirm['Ranking'] >= strongest + margin), strongest, version


def campaign(args):
    out, learner = args.out, args.out / 'learner'
    statefile = out / 'state.json'
    base = checkpoint(args.base) if not statefile.exists() else Path(args.base)
    if not base.name.isdecimal():
        raise ValueError(f'Not a numbered checkpoint path: {base}')
    if not args.bot.is_file():
        raise ValueError(f'Build {args.bot} before starting the campaign')
    plan = minutes_plan(args.minutes, args.pilot_minutes, args.block_minutes)
    train_args = ['--threads', str(args.threads), '--games', '88', '--mode-balance', 'players',
                  '--ts-per-itr', '48384', '--batch-size', '12096',
                  '--fixed-rollout', '--lean-metrics', '--minibatch-size', str(args.minibatch),
                  '--lr', '1e-4', '--skill-minutes', '15', '--ts-per-save', '5m', '--keep-checkpoints', '2']
    # Explicit recipe overrides make follow-up trials reproducible without editing presets.
    overrides = [
        ('lr', '--lr'), ('critic_lr', '--critic-lr'),
        ('epochs', '--epochs'), ('batch_size', '--batch-size'),
        ('entropy_scale', '--entropy-scale'), ('norm_advantages', '--norm-advantages'),
        ('target_kl', '--target-kl'), ('gamma', '--gamma'), ('gae_lambda', '--gae-lambda'),
        ('return_stat_max_count', '--return-stat-max-count'),
        ('random_state_prob', '--random-state-prob'),
        ('tactical_state_prob', '--tactical-state-prob'),
        ('train_vs_old', '--train-vs-old'),
        ('ts_per_version', '--ts-per-version'), ('max_versions', '--max-versions'),
        ('skill_minutes', '--skill-minutes'), ('ts_per_save', '--ts-per-save'),
    ]
    for attribute, flag in overrides:
        value = getattr(args, attribute, None)
        if value is not None:
            if flag in train_args:
                train_args[train_args.index(flag) + 1] = str(value)
            else:
                train_args.extend([flag, str(value)])
    bool_overrides = [
        ('mask_entropy', '--mask-entropy'),
        ('fuzzed_kickoffs', '--fuzzed-kickoffs'),
        ('pfsp', '--pfsp'),
    ]
    for attribute, flag in bool_overrides:
        if getattr(args, attribute, False) and flag not in train_args:
            train_args.append(flag)
    if '--fuzzed-kickoffs' not in train_args:
        # Rollouts against a near-deterministic policy need varying start states just like the rating rounds,
        #so a campaign never trains on the plain centre-spot kickoff.
        train_args.append('--fuzzed-kickoffs')
    if getattr(args, 'rewards', None):
        train_args.extend(['--rewards', args.rewards])
    config = dict(base=str(base), preset_sha256=hashlib.sha256((REPO / 'train_1v1_2v2_3v3.sh').read_bytes()).hexdigest(),
                  parent_steps=int(base.name), seed=args.seed, screen_seed=SCREEN,
                  confirm_seed=CONFIRM, audit_seed=getattr(args, 'audit_seed', AUDIT), rounds=args.rounds, train_args=train_args,
                  minutes=plan, modes=list(MODES), protocol='anchored-v1')
    if getattr(args, 'reference_pool', None) is not None:
        config['reference_pool'] = str(args.reference_pool)
    extra_versions = [str(p) for p in (getattr(args, 'extra_versions', None) or [])]
    if extra_versions:
        config['extra_versions'] = extra_versions
    if not statefile.exists():
        if any(p.name != '.lock' for p in out.iterdir()):
            raise ValueError(f'Refusing to overwrite a non-empty campaign directory: {out}')
        src_pool = getattr(args, 'reference_pool', None) or (base.parent / 'policy_versions')
        if getattr(args, 'evolving_opponents', False):
            if getattr(args, 'extra_versions', None):
                raise ValueError('--extra-versions cannot be combined with --evolving-opponents')
            pool_count = int(getattr(args, 'opponent_pool_size', 10))
            if pool_count < 1:
                raise ValueError('--opponent-pool-size must be at least 1')
            # The pool starts with the policies history offers and grows through promotions, so the
            # rating rounds are derived from the pool size at evaluation time, not fixed here.
            seed_policy_pool(src_pool, out / 'pool', pool_count)
            shutil.copytree(out / 'pool', learner / 'policy_versions')
        else:
            count = len(policy_versions(src_pool))
            if not count:
                raise ValueError('The reference opponent pool is empty')
            shutil.copytree(src_pool, out / 'pool')
            shutil.copytree(src_pool, learner / 'policy_versions')
        # Optional stronger native sparring partners for learner/policy_versions ONLY; out/pool stays immutable.
        for extra in (getattr(args, 'extra_versions', None) or []):
            extra = Path(extra)
            if not extra.name.isdecimal() or not (extra / 'POLICY.lt').is_file():
                raise ValueError(f'Invalid --extra-versions entry: {extra}')
            if int(extra.name) > int(base.name):
                raise ValueError(f'--extra-versions entry {extra.name} is newer than --base {base.name}')
            vdest = learner / 'policy_versions' / extra.name
            if not vdest.exists():
                vdest.mkdir(parents=True)
                for fname in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt', 'STATS.json'):
                    if (extra / fname).is_file():
                        shutil.copy2(extra / fname, vdest / fname)
                if not (vdest / 'STATS.json').is_file() and (extra / 'RUNNING_STATS.json').is_file():
                    rstats = read(extra / 'RUNNING_STATS.json')
                    write(vdest / 'STATS.json', {
                        'skill_ratings': rstats.get('skill_ratings', {}),
                        'mirror_x': rstats.get('mirror_x', -1),
                    })
        shutil.copytree(base, learner / base.name)
        (out / 'eval').mkdir()
        (out / 'best').mkdir()
        state = dict(config=config, pool_sha256=pool_digest(out / 'pool'), history=[], best=None,
                     pending=None, status='initialized')
        if getattr(args, 'evolving_opponents', False):
            state['runtime_overrides'] = {
                'evolving_opponents': True,
                'opponent_pool_size': int(getattr(args, 'opponent_pool_size', 10)),
                'promotion_margin': float(getattr(args, 'promotion_margin', 50.0)),
            }
        write(statefile, state)
    state = read(statefile)
    if config != state['config'] or pool_digest(out / 'pool') != state['pool_sha256']:
        raise ValueError('Campaign settings or frozen pool changed. Use the original settings or a fresh --out.')
    # Old policies are immutable records: pin every version's files once and re-verify them at
    # each cycle boundary, so no later save can re-rate or re-annotate an existing version.
    state.setdefault('frozen_versions', frozen_versions(out / 'pool'))
    verify_frozen_versions(out / 'pool', state['frozen_versions'])
    write(statefile, state)
    existing_runtime = state.setdefault('runtime_overrides', {})
    want_evolving = bool(getattr(args, 'evolving_opponents', False) or existing_runtime.get('evolving_opponents'))
    if want_evolving:
        pool_count = int(getattr(args, 'opponent_pool_size', None) or existing_runtime.get('opponent_pool_size', 10))
        margin = float(getattr(args, 'promotion_margin', None) if getattr(args, 'promotion_margin', None) is not None
                       else existing_runtime.get('promotion_margin', 50.0))
        if pool_count < 1:
            raise ValueError('--opponent-pool-size must be at least 1')
        if not existing_runtime.get('evolving_opponents'):
            # Migrate only at an explicit resume boundary. Keep the old frozen eval pool for provenance,
            # but seed both training and ratings from evenly spaced historical policy versions.
            old_hash = state['pool_sha256']
            legacy = out / f'pool-legacy-{old_hash[:10]}'
            if not legacy.exists():
                shutil.copytree(out / 'pool', legacy)
            staged = out / 'pool-evolving.new'
            shutil.rmtree(staged, ignore_errors=True)
            seed_policy_pool(learner / 'policy_versions', staged, pool_count)
            shutil.rmtree(out / 'pool')
            staged.rename(out / 'pool')
            replace_policy_pool(out / 'pool', learner / 'policy_versions')
            existing_runtime.update(evolving_opponents=True, opponent_pool_size=pool_count,
                                    promotion_margin=margin, pool_generation=0,
                                    pool_steps=[int(p.name) for p in policy_versions(out / 'pool')])
            state['pool_sha256'] = pool_digest(out / 'pool')
            state['status'] = 'paused'
            write(statefile, state)
        elif (int(existing_runtime.get('opponent_pool_size', pool_count)) != pool_count or
              not math.isclose(float(existing_runtime.get('promotion_margin', margin)), margin)):
            raise ValueError('Evolving opponent-pool size and promotion margin are immutable after migration')
    # Runtime instrumentation/performance options may be enabled on an existing campaign without
    # changing its frozen reward recipe or evaluation protocol. Persist them across clean resumes.
    runtime_changed = False
    if getattr(args, 'reward_metrics', False):
        state.setdefault('runtime_overrides', {})['reward_metrics'] = True
        runtime_changed = True
    if getattr(args, 'player_metrics', False):
        state.setdefault('runtime_overrides', {})['player_metrics'] = True
        runtime_changed = True
    if getattr(args, 'throughput_profile', None):
        state.setdefault('runtime_overrides', {})['throughput_profile'] = args.throughput_profile
        runtime_changed = True
    if getattr(args, 'continuous', False):
        state.setdefault('runtime_overrides', {})['continuous'] = True
        runtime_changed = True
    if runtime_changed:
        write(statefile, state)
    runtime_overrides = state.get('runtime_overrides') or {}
    runtime_train_args = []
    if runtime_overrides.get('reward_metrics'):
        runtime_train_args.append('--reward-metrics')
    if runtime_overrides.get('player_metrics'):
        runtime_train_args.append('--player-metrics')
    if runtime_overrides.get('fuzzed_kickoffs') is not True:
        # Record the start-state randomization this campaign trains and rates with, so it cannot drift.
        runtime_overrides['fuzzed_kickoffs'] = True
        state['runtime_overrides'] = runtime_overrides
        write(statefile, state)
    if runtime_overrides.get('throughput_profile') == 'cpu-20k':
        # Benchmarked on the two-vCPU CPU runner: this is the fastest stable all-mode profile,
        # with best clean measurements near 19.5k overall SPS. Shared-host variance still applies.
        runtime_train_args.extend(['--games', '64', '--epochs', '1', '--collection-infer-threads', '1'])
    if out.resolve().is_relative_to(REPO.resolve()):
        state['working_directory'] = str(out.resolve().relative_to(REPO.resolve()))
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
    if getattr(args, 'github_backup', False):
        service = subprocess.Popen([sys.executable, str(REPO / 'tools/training_service.py'), str(out),
                                    '--interval', str(getattr(args, 'backup_interval', 1800))])
    bridge_proc = None
    if getattr(args, 'wandb_bridge', False):
        bridge_proc = subprocess.Popen([sys.executable, str(REPO / 'tools/wandb_live_bridge.py'),
                                        'publish', str(out), '--interval', '20'])
    streamer = None
    if getattr(args, 'wandb_online', False):
        streamer = subprocess.Popen([sys.executable, str(REPO / 'tools/ranked_wandb.py'),
                                     str(out), '--out', str(out / 'online-stream'), '--online'])
    def evaluate(p, seed, multiplier=1, write_back=False):
        """Rate a policy against the pool as it is right now.

        The native tournament needs a multiple of twice the pool size, and the pool of an
        evolving campaign changes as qualified policies are promoted into it. The screen, the
        confirmation, the incumbent rerating and the audit all read the current pool, so they all
        use the round count that pool size allows. The confirmation keeps its twofold budget:
        it runs exactly twice the rounds of the screen it confirms.
        """
        rounds = rounds_for_pool(args.rounds, pool_size(out / 'pool')) * multiplier
        return score(args, p, seed, rounds, state['pool_sha256'], env, logdir, write_back=write_back)
    def rerate_incumbent():
        """Re-rate the current best against the pool as it is now, if the pool moved on.

        A pool update changes the Elo scale: the incumbent has to meet the new opponent set to
        be comparable with the candidate that is rated next. This runs before the first
        comparison of a campaign and again after every pool change, so a promotion is never
        decided by two Elo values that were measured against different opponents.
        """
        if not runtime_overrides.get('evolving_opponents', False):
            return
        if state.get('best') and state['best'].get('opponent_pool_sha256') == state['pool_sha256']:
            return
        state['status'] = 'evaluating'
        write(statefile, state)
        incumbent_steps = (state.get('best') or {}).get('steps')
        incumbent = out / 'best' / str(incumbent_steps) if incumbent_steps is not None else None
        if incumbent is None or not incumbent.is_dir():
            incumbent = newest(learner)
            incumbent_steps = int(incumbent.name)
        incumbent_screen = evaluate(incumbent, SCREEN)
        incumbent_confirm = evaluate(incumbent, CONFIRM, 2)
        state['best'] = dict(steps=int(incumbent_steps), screen=incumbent_screen,
                             confirm=incumbent_confirm,
                             opponent_pool_sha256=state['pool_sha256'])
        incumbent_copy = out / 'best' / str(incumbent_steps)
        if not incumbent_copy.exists():
            shutil.copytree(incumbent, incumbent_copy)
        write(statefile, state)
    try:
        rerate_incumbent()
        # Stage 0 is the parent, not a trained candidate. Later stages always continue the latest
        # weights, irrespective of short-term rating noise. Only the independent best copy is selected.
        stage = len(state['history'])
        while True:
            minutes = budget_for_stage(
                plan, stage, continuous=runtime_overrides.get('continuous', False),
                block=getattr(args, 'block_minutes', 15))
            if minutes is None:
                break
            i = stage
            stage_wall_start = time.monotonic()
            if state['pending'] is None:
                state['pending'] = dict(stage=i, start_steps=int(newest(learner).name), seconds=0, wall_seconds=0, trained=False)
            pending = state['pending']
            prior_wall = float(pending.get('wall_seconds', pending.get('seconds', 0)) or 0.0)
            if minutes and not pending['trained']:
                if runtime_overrides.get('evolving_opponents', False):
                    sync_policy_versions(out / 'pool', learner / 'policy_versions')
                state['status'] = 'training'
                write(statefile, state)
                start, previous = time.monotonic(), pending['seconds']
                def heartbeat():
                    now = time.monotonic()
                    pending['seconds'] = previous + now - start
                    pending['wall_seconds'] = prior_wall + now - stage_wall_start
                    write(statefile, state)
                cmd = ['bash', 'train_1v1_2v2_3v3.sh', '--checkpoints', learner, *train_args,
                       *runtime_train_args, '--seed', args.seed + i, '--run-name', 'three-mode-ranking',
                       '--max-minutes', max(0.02, minutes - previous / 60)]
                if runtime_overrides.get('evolving_opponents', False):
                    # Do not let the native saver append unqualified policies: only the outer
                    # evaluation gate may change this ten-policy training/rating pool.
                    cmd.extend(['--ts-per-version', '9000000000000000',
                                '--max-versions', str(runtime_overrides['opponent_pool_size'])])
                if int(newest(learner).name) == int(base.name):
                    cmd.append('--new-run')
                if runtime_overrides.get('continuous', False):
                    stage_label = f'Cycle {i} (continuous)'
                else:
                    stage_label = f'Stage {i}/{len(plan)}'
                print(f'{stage_label}: training from {newest(learner).name} ({minutes} min budget)', flush=True)
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
            versions_verified = False
            if runtime_overrides.get('evolving_opponents', False):
                # Prove the trainer neither re-rated nor re-annotated any old policy version.
                verify_frozen_versions(out / 'pool', state['frozen_versions'])
                trainer_versions = verify_frozen_versions(learner / 'policy_versions',
                                                          state['frozen_versions'])
                if set(trainer_versions) != {int(s) for s in state['frozen_versions']}:
                    raise RuntimeError('Trainer policy versions do not match the frozen pool')
                versions_verified = True
                print(f'Frozen policy versions verified unchanged ({len(trainer_versions)})', flush=True)
            # The pool may have grown since the last stage. Bring the incumbent onto the current
            # Elo scale before the candidate is compared with it.
            rerate_incumbent()
            pool_steps_before = [int(p.name) for p in policy_versions(out / 'pool')]
            pool_hash_before = state['pool_sha256']
            # The rating rounds follow the pool that is rated right now; record them with the stage
            # so a later reader can tell which tournament produced which Elo.
            eval_rounds = rounds_for_pool(args.rounds, pool_size(out / 'pool'))
            screen = evaluate(ckpt, SCREEN, write_back=True)
            best = state['best']
            confirm = None
            strongest = strongest_version = None
            pool_screen_candidate = False
            if runtime_overrides.get('evolving_opponents', False):
                strongest, strongest_version = strongest_pool_ranking(out / 'pool')
                pool_screen_candidate = screen['Ranking'] >= strongest + float(runtime_overrides['promotion_margin'])
            if (best is None or screen['Ranking'] > best['screen']['Ranking'] or
                    pool_screen_candidate):
                confirm = evaluate(ckpt, CONFIRM, 2, write_back=True)
            won = confirm is not None and promote(screen, confirm, best)
            pool_promoted = False
            if runtime_overrides.get('evolving_opponents', False) and confirm is not None:
                pool_promoted, strongest, strongest_version = beats_strongest(
                    screen, confirm, out / 'pool', float(runtime_overrides['promotion_margin']))
            if won:
                dest = out / 'best' / ckpt.name
                if not dest.exists():
                    tmp = dest.with_suffix('.copying')
                    shutil.copytree(ckpt, tmp, dirs_exist_ok=True)
                    tmp.rename(dest)
                state['best'] = dict(steps=steps, screen=screen, confirm=confirm,
                                     opponent_pool_sha256=pool_hash_before)
            if pool_promoted:
                pool_promoted = add_policy_to_pool(ckpt, out / 'pool', runtime_overrides['opponent_pool_size'])
                if pool_promoted:
                    replace_policy_pool(out / 'pool', learner / 'policy_versions')
                    runtime_overrides['pool_generation'] = int(runtime_overrides.get('pool_generation', 0)) + 1
                    runtime_overrides['pool_steps'] = [int(p.name) for p in policy_versions(out / 'pool')]
                    state['pool_sha256'] = pool_digest(out / 'pool')
                    state['frozen_versions'] = frozen_versions(out / 'pool')
            seconds = pending['seconds']
            wall_seconds = prior_wall + (time.monotonic() - stage_wall_start)
            delta_steps = steps - pending['start_steps']
            # Include 100% of stage wall-clock time (subprocess startup, rollout, PPO, saves,
            # screen/confirmation evaluations, and checkpoint promotion copy).
            entry = dict(stage=i, start_steps=pending['start_steps'], steps=steps,
                         train_seconds=seconds, wall_seconds=wall_seconds,
                         steps_per_second=(delta_steps / wall_seconds) if (minutes and wall_seconds > 0) else None,
                         eval_rounds=eval_rounds, eval_rounds_confirm=2 * eval_rounds,
                         screen=screen, confirm=confirm, promoted=won,
                         opponent_pool_steps=pool_steps_before, opponent_pool_sha256=pool_hash_before,
                         opponent_pool_generation=int(runtime_overrides.get('pool_generation', 0)) -
                         (1 if pool_promoted else 0),
                         strongest_opponent_ranking=strongest,
                         strongest_opponent_steps=strongest_version,
                         opponent_pool_promoted=pool_promoted,
                         opponent_pool_margin=float(runtime_overrides['promotion_margin'])
                         if runtime_overrides.get('evolving_opponents', False) else None,
                         old_versions_verified=versions_verified)
            state['history'].append(entry)
            state['pending'] = None
            write(statefile, state)
            for p in (out / 'best').iterdir():
                if p.name.isdecimal() and int(p.name) != state['best']['steps']:
                    shutil.rmtree(p)
            if wb is not None:
                ranked_wandb.log_stage(wb, out, entry, state['best']['steps'])
            label = (f'Cycle {i} complete; starting next cycle' if runtime_overrides.get('continuous', False)
                     else f'Stage {i} complete')
            print(f'{label}. Best: {state["best"]["steps"]}; latest: {steps}', flush=True)
            if (not runtime_overrides.get('continuous', False) and
                    getattr(args, 'stop_on_target', False) and
                    getattr(args, 'target_ranking', None) is not None and i > 0):
                margin = float(getattr(args, 'target_margin', 0.0) or 0.0)
                goal = args.target_ranking + margin
                cur_best = state['best']
                if (cur_best['screen']['Ranking'] >= goal and
                        (cur_best.get('confirm') or {}).get('Ranking', 0.0) >= goal):
                    print(f'Target threshold {goal:.2f} reached on screen & confirm at stage {i}; proceeding to held-out audit.', flush=True)
                    break
            stage += 1
        # An untouched seed measures the selected checkpoint against the original baseline on the same scale.
        state['status'] = 'auditing'
        write(statefile, state)
        audit_targets = []
        if base.is_dir() and (base / 'POLICY.lt').is_file():
            audit_targets.append(base)
        for cand in (out / 'best' / str(state['best']['steps']), newest(learner)):
            if all(p.name != cand.name for p in audit_targets):
                audit_targets.append(cand)
        state['audit'] = {p.name: evaluate(p, config['audit_seed'], 2) for p in audit_targets}
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
        if bridge_proc is not None:
            try:
                bridge_proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                bridge_proc.terminate()
                bridge_proc.wait(timeout=15)
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
    for name, default in [('minutes', 180), ('pilot-minutes', 6), ('block-minutes', 30),
                          ('minibatch', 1008), ('threads', 2), ('seed', 20261118)]:
        parser.add_argument('--' + name, type=int, default=default)
    # The anchored tournament needs a multiple of twice the opponent count, so --rounds is a budget:
    # every evaluation derives the smallest valid count at least as large as it.
    parser.add_argument('--rounds', type=int, default=40,
                        help='Rating round budget per screen; the actual count follows the opponent '
                             'pool size (a multiple of twice the pool size, never below the budget)')
    parser.add_argument('--wandb-offline', action='store_true', help='Write a syncable W&B run in --out; no credential or network needed')
    parser.add_argument('--wandb-online', action='store_true', help='Tail metrics to W&B online in a separate sandbox process; requires W&B credentials/network')
    parser.add_argument('--wandb-bridge', action='store_true', help='Stream live metrics to W&B via the GitHub Actions cloud relay even when direct W&B egress is blocked')
    parser.add_argument('--github-backup', action='store_true', help='Automatically commit resumable snapshots and push the current branch every --backup-interval seconds')
    parser.add_argument('--backup-interval', type=int, default=1800, help='Minimum seconds between GitHub auto-backups (default: 1800 = 30 minutes)')
    parser.add_argument('--reference-pool', type=Path, help='Explicit immutable opponent pool for a fresh experiment')
    parser.add_argument('--extra-versions', type=Path, nargs='*', help='Optional native checkpoints copied into learner/policy_versions only (frozen eval pool untouched)')
    parser.add_argument('--audit-seed', type=int, default=AUDIT, help='Predeclared held-out audit seed')
    parser.add_argument('--target-ranking', type=float, help='Report a weight-only temperature-1 Ranking target; does not change immutable training settings')
    parser.add_argument('--stop-on-target', action='store_true', help='Proceed straight to held-out audit once best clears target_ranking + target_margin on screen and confirm')
    parser.add_argument('--target-margin', type=float, default=0.0, help='Optional Elo safety margin above --target-ranking before early audit')
    parser.add_argument('--lr', type=float, help='Explicit learning-rate override for a fresh experiment')
    parser.add_argument('--critic-lr', type=float, help='Separate critic learning rate override')
    parser.add_argument('--epochs', type=int, help='PPO epochs per iteration')
    parser.add_argument('--batch-size', type=int, help='PPO batch size override')
    parser.add_argument('--entropy-scale', type=float)
    parser.add_argument('--mask-entropy', action='store_true', help='Normalize entropy over valid actions')
    parser.add_argument('--norm-advantages', choices=('off', 'all', 'mode'), help='Normalize PPO policy advantages')
    parser.add_argument('--target-kl', type=float, help='KL early-stop target for multi-epoch PPO')
    parser.add_argument('--gamma', type=float, help='Discount factor override')
    parser.add_argument('--gae-lambda', type=float, help='GAE lambda override')
    parser.add_argument('--return-stat-max-count', type=int, help='Effective sample cap on return standardizer')
    parser.add_argument('--random-state-prob', type=float)
    parser.add_argument('--tactical-state-prob', type=float)
    parser.add_argument('--fuzzed-kickoffs', action='store_true')
    parser.add_argument('--train-vs-old', type=float)
    parser.add_argument('--pfsp', action='store_true', help='Prioritized Fictitious Self-Play opponent weighting')
    parser.add_argument('--ts-per-version', help='Timesteps between saved policy versions (e.g. 12m)')
    parser.add_argument('--max-versions', type=int, help='Maximum saved policy versions in learner pool')
    parser.add_argument('--skill-minutes', type=float, help='Minutes between internal skill tracker matches')
    parser.add_argument('--ts-per-save', help='Timesteps between local checkpoint saves (e.g. 10m)')
    parser.add_argument('--rewards', help='Native comma-separated reward weights for a fresh experiment')
    parser.add_argument('--reward-metrics', action='store_true',
                        help='Enable sampled reward-component telemetry for this campaign and persist the choice across resumes')
    parser.add_argument('--player-metrics', action='store_true',
                        help='Enable sampled per-player telemetry for this campaign and persist the choice across resumes')
    parser.add_argument('--throughput-profile', choices=('cpu-20k',),
                        help='Use the benchmarked two-vCPU profile targeting about 20k overall steps/second')
    parser.add_argument('--continuous', action='store_true',
                        help='Repeat evaluation-backed training blocks indefinitely, until interrupted')
    parser.add_argument('--evolving-opponents', action='store_true',
                        help='Use an evenly seeded pool of old policies; add a candidate only when it clears the strongest by the promotion margin')
    parser.add_argument('--opponent-pool-size', type=int, default=10,
                        help='Number of saved historical policies in the evolving train/eval pool (default: 10)')
    parser.add_argument('--promotion-margin', type=float, default=50.0,
                        help='Raw endpoint Elo margin above the strongest pool member required for promotion (default: 50)')
    args = parser.parse_args()
    if args.backup_interval < 60:
        parser.error('--backup-interval must be at least 60 seconds')
    if args.target_ranking is not None and not (0 < args.target_ranking < 10000):
        parser.error('--target-ranking must be finite and between 0 and 10000')
    if args.opponent_pool_size < 1:
        parser.error('--opponent-pool-size must be positive')
    if not math.isfinite(args.promotion_margin) or args.promotion_margin < 0:
        parser.error('--promotion-margin must be finite and nonnegative')
    if args.lr is not None and not (0 < args.lr <= 0.01):
        parser.error('--lr must be in (0, 0.01]')
    if args.critic_lr is not None and not (0 < args.critic_lr <= 0.01):
        parser.error('--critic-lr must be in (0, 0.01]')
    if args.epochs is not None and args.epochs <= 0:
        parser.error('--epochs must be positive')
    if args.entropy_scale is not None and not (0 <= args.entropy_scale <= 1):
        parser.error('--entropy-scale must be in [0, 1]')
    for value in (args.random_state_prob, args.tactical_state_prob, args.train_vs_old):
        if value is not None and not (0 <= value <= 1):
            parser.error('Start/opponent probabilities must be in [0, 1]')
    if (args.random_state_prob or 0.0) + (args.tactical_state_prob or 0.0) > 1.0:
        parser.error('--random-state-prob + --tactical-state-prob must be <= 1')
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
