#!/usr/bin/env python3
"""Write a closed, syncable W&B run from a ranked campaign's completed stages.

No trainer, simulator, W&B account or network access is required. Needs the wandb SDK.
Useful after a hard stop or to take a snapshot without interrupting training. Sync either
this snapshot or the full offline run, not both, unless you want two runs in W&B.
"""
import argparse
import json
import time
from pathlib import Path


def start(directory, config, name, online=False, run_id=None):
    import wandb  # optional dependency, never needed for JSON-only training or viewing
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(mode='online' if online else 'offline', id=run_id,
                     resume='allow' if online and run_id else None, project='rocket-league-ggl', group='three-mode-ranking',
                     name=name, dir=str(directory.resolve()), config=config,
                     settings=wandb.Settings(console='off', disable_code=True, disable_git=True,
                                             x_disable_stats=True, x_disable_meta=True, init_timeout=30))
    run.define_metric('Total Timesteps')
    run.define_metric('*', step_metric='Total Timesteps')
    return run


def log_stage(run, campaign, entry, best_steps):
    lo, hi = entry['start_steps'], entry['steps']
    training_rows = []
    for path in (campaign / 'metrics').glob('*.jsonl'):
        with path.open() as f:
            for line in f:
                # A concurrently-running stage may have its last row open for writing.
                if not line.endswith('\n') or not line.strip():
                    continue
                row = json.loads(line)
                if lo < row['Total Timesteps'] <= hi:
                    training_rows.append(row)
    for row in sorted(training_rows, key=lambda r: r['Total Timesteps']):
        run.log({('Live/' + k if k.startswith('Rating/') else k): v
                 for k, v in row.items() if k != '_time' and v is not None})
    record = {f'Rating/{k}': v for k, v in entry['screen'].items()}
    record.update({f'Confirmation/{k}': v for k, v in (entry['confirm'] or {}).items()})
    record.update({'Total Timesteps': hi, 'Campaign/Best Steps': best_steps, 'Campaign/Stage': entry['stage']})
    if entry.get('steps_per_second') is not None:
        record['Campaign/Steps Per Second'] = entry['steps_per_second']
    run.log(record)


def log_audit(run, state):
    audit = state.get('audit') or {}
    for name, step in [('Parent', state['config']['parent_steps']), ('Best', state['best']['steps']),
                       ('Latest', state['history'][-1]['steps'])]:
        for metric, value in (audit.get(str(step)) or {}).items():
            run.summary[f'Audit/{name} {metric}'] = value



def stream(campaign, directory, interval=10):
    """Tail durable JSON in-process; never acquire the trainer lock or alter its state."""
    state = json.loads((campaign / 'state.json').read_text())
    import hashlib
    run_id = 'live-' + hashlib.sha256(str(campaign.resolve()).encode()).hexdigest()[:16]
    run = start(directory, state['config'], 'three-mode-ranking-live', online=True, run_id=run_id)
    print(f'Online W&B run: {run.url}', flush=True)
    last = int(run.summary.get('Stream/Last Timesteps', -1))
    stages = set(run.summary.get('Stream/Stages', []))
    try:
        while True:
            state = json.loads((campaign / 'state.json').read_text())
            events = {}
            for path in sorted((campaign / 'metrics').glob('*.jsonl')):
                for line in path.read_text().splitlines(keepends=True):
                    if not line.endswith('\n') or not line.strip():
                        continue
                    row = json.loads(line)
                    step = int(row['Total Timesteps'])
                    if step > last:
                        events.setdefault(step, {}).update({
                            ('Live/' + k if k.startswith('Rating/') else k): v
                            for k, v in row.items() if k != '_time' and v is not None})
            for entry in state['history']:
                if entry['stage'] in stages:
                    continue
                record = events.setdefault(entry['steps'], {'Total Timesteps': entry['steps']})
                record.update({f'Rating/{k}': v for k, v in entry['screen'].items()})
                record.update({f'Confirmation/{k}': v for k, v in (entry.get('confirm') or {}).items()})
                record['Campaign/Stage'] = entry['stage']
                stages.add(entry['stage'])
            for step in sorted(events):
                run.log(events[step])
                last = max(last, step)
            run.summary['Stream/Last Timesteps'] = last
            run.summary['Stream/Stages'] = sorted(stages)
            run.summary['Campaign/Status'] = state['status']
            if state.get('best'):
                run.summary['Campaign/Best Steps'] = state['best']['steps']
            if state.get('audit'):
                log_audit(run, state)
            if state['status'] in ('complete', 'paused', 'failed'):
                break
            time.sleep(interval)
    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path, help='Directory containing state.json and metrics/')
    parser.add_argument('--out', type=Path, required=True, help='Destination for the separate offline run')
    parser.add_argument('--online', action='store_true', help='Stream the running campaign directly to W&B instead of offline export')
    parser.add_argument('--interval', type=float, default=10)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    if args.online:
        stream(args.campaign, args.out, args.interval)
        return
    state = json.loads((args.campaign / 'state.json').read_text())  # writer renames atomically
    if not state['history']:
        parser.error('Wait for at least one completed evaluation')
    last = state['history'][-1]['steps']
    run = start(args.out, state['config'], f'ranking-{state["config"]["parent_steps"]}-through-{last}')
    failed = True
    try:
        for entry in state['history']:
            if entry['promoted']:
                best_steps = entry['steps']
            log_stage(run, args.campaign, entry, best_steps)
        if 'audit' in state:
            log_audit(run, state)
        failed = False
    finally:
        run.finish(exit_code=int(failed))
    print(f'Finished offline W&B run: {Path(run.dir).parent}')


if __name__ == '__main__':
    main()
