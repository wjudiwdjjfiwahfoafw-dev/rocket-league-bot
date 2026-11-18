#!/usr/bin/env python3
"""Live Weights & Biases telemetry bridge for sandboxed Rocket League bot training.

Works even when direct outbound traffic from the training sandbox to api.wandb.ai is blocked:

1. Sandbox publisher mode (`publish`):
   Tails a running campaign's durable `state.json` and `metrics/*.jsonl` (read-only, never
   acquiring the trainer lock) and publishes compact gzip+base64 metric snapshots to a single
   dedicated GitHub Issue via `gh api` (`api.github.com`), plus the active live-preview
   `/status.json` URL when available. Uses near-zero CPU and creates zero Git commits.

2. Cloud / external relay mode (`relay`):
   Runs anywhere with outbound access to `api.wandb.ai` (e.g. the included GitHub Actions
   workflow `.github/workflows/wandb-live.yml`, or any external machine). Polls the live
   preview `/status.json` endpoint first (with automatic fallback to the GitHub Issue live
   payload and `training-backup/` snapshot) every few seconds and streams every training row,
   frozen-pool Screen/Confirmation Elo rating, and held-out Audit summary idempotently into a
   single resumable online W&B run (`step_metric='Total Timesteps'`).
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_LABEL_TITLE = '[telemetry] Live W&B metrics feed'
PAYLOAD_RE = re.compile(r'<!-- WANDB_SNAPSHOT_GZ_B64_START -->\s*([A-Za-z0-9+/=\s]+?)\s*<!-- WANDB_SNAPSHOT_GZ_B64_END -->')
META_RE = re.compile(r'<!-- WANDB_RELAY_META_START -->\s*(\{.*?\})\s*<!-- WANDB_RELAY_META_END -->', re.DOTALL)
# GitHub issue/comment body limit is 65,536 chars; keep comfortable headroom for header + metadata.
MAX_BODY_B64 = 52_000


def campaign_run_id(config, prefix='rlbot'):
    """Deterministic W&B run ID for a campaign so restarts/backfills resume one run cleanly."""
    ident = {
        'base': config.get('base'),
        'parent_steps': config.get('parent_steps'),
        'seed': config.get('seed'),
        'audit_seed': config.get('audit_seed'),
        'protocol': config.get('protocol'),
    }
    digest = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
    return f'{prefix}-{config.get("parent_steps", "run")}-{digest}'


def read_campaign_snapshot(campaign: Path, status_url: str | None = None) -> dict:
    """Read atomic state.json + completed JSONL metric lines without locking the trainer."""
    state = json.loads((campaign / 'state.json').read_text())
    rows_by_step = {}
    metrics_dir = campaign / 'metrics'
    if metrics_dir.is_dir():
        for path in sorted(metrics_dir.glob('*.jsonl')):
            for line in path.read_text(errors='replace').splitlines(keepends=True):
                if not line.endswith('\n') or not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'Total Timesteps' in row:
                    rows_by_step[int(row['Total Timesteps'])] = row
    # Also fold detailed anchored eval stats (goal ratios, kickoff first-touch %, end Elo) per stage.
    eval_extras = {}
    eval_dir = campaign / 'eval'
    if eval_dir.is_dir():
        for path in sorted(eval_dir.glob('*.json')):
            try:
                ev = json.loads(path.read_text())
                key = f"{ev['checkpoint_timesteps']}-{ev['seed']}-{ev['rounds']}"
                kf = float(ev.get('kickoff_first_touches_for', 0.0))
                ka = float(ev.get('kickoff_first_touches_against', 0.0))
                mg = ev.get('mode_goals') or {}
                mode_goal_pct = {}
                for m in ('1v1', '2v2', '3v3'):
                    gf = float((mg.get(m) or {}).get('for', 0.0))
                    ga = float((mg.get(m) or {}).get('against', 0.0))
                    if gf + ga > 0:
                        mode_goal_pct[m] = round(100.0 * gf / (gf + ga), 3)
                eval_extras[key] = {
                    'kickoff_pct': round(100.0 * kf / (kf + ka), 3) if (kf + ka) > 0 else None,
                    'goals_for': ev.get('goals_for'),
                    'goals_against': ev.get('goals_against'),
                    'mode_goal_pct': mode_goal_pct,
                }
            except Exception:
                continue
    payload = {
        'updated': time.time(),
        'run_id': campaign_run_id(state.get('config') or {}),
        'state': state,
        'rows': [rows_by_step[k] for k in sorted(rows_by_step)],
        'eval_extras': eval_extras,
    }
    if status_url:
        payload['status_url'] = status_url
    return payload


def encode_snapshot_body(snapshot: dict, max_b64: int = MAX_BODY_B64, min_step: int = -1) -> tuple[str, int]:
    """Pack snapshot into a GitHub Issue body with readable summary + gzip/base64 JSON payload.
    If full history exceeds GitHub's 65 KB limit, retains newest rows while always keeping state/eval.
    Returns (markdown_body, rows_included).
    """
    all_rows = [r for r in snapshot.get('rows', []) if int(r['Total Timesteps']) > min_step]
    rows = list(all_rows)
    while True:
        candidate = dict(snapshot, rows=rows, total_row_count=len(snapshot.get('rows', [])))
        raw = json.dumps(candidate, separators=(',', ':'), allow_nan=False).encode()
        b64 = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode('ascii')
        if len(b64) <= max_b64 or len(rows) <= 1:
            break
        # Drop oldest quarter of un-trimmed rows to stay comfortably under 65 KB.
        rows = rows[max(1, len(rows) // 4):]

    state = snapshot.get('state') or {}
    best = state.get('best') or {}
    best_screen = (best.get('screen') or {}).get('Ranking')
    best_confirm = (best.get('confirm') or {}).get('Ranking')
    latest_step = int(rows[-1]['Total Timesteps']) if rows else (best.get('steps') or 0)
    target = state.get('target_ranking', 1700.0)
    wrapped = '\n'.join(b64[i:i + 120] for i in range(0, len(b64), 120))
    header = (
        f"### Live W&B Telemetry Feed (`{snapshot.get('run_id', 'unknown')}`)\n"
        f"- **Status**: `{state.get('status', 'unknown')}` | **Target**: `{target}` Elo\n"
        f"- **Latest step**: `{latest_step:,}` | **Stages completed**: `{max(0, len(state.get('history') or []) - 1)}`\n"
        f"- **Selected best**: `{best.get('steps')}` (Screen `{best_screen}`, Confirm `{best_confirm}`)\n"
        f"- **Rows buffered**: `{len(rows)}` / `{len(snapshot.get('rows', []))}` | "
        f"**Updated**: `{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(snapshot.get('updated', time.time())))}`\n\n"
        f"<details><summary>Compressed telemetry frame ({len(b64)} bytes)</summary>\n\n"
        f"<!-- WANDB_SNAPSHOT_GZ_B64_START -->\n{wrapped}\n<!-- WANDB_SNAPSHOT_GZ_B64_END -->\n"
        f"</details>\n"
    )
    return header, len(rows)


def decode_snapshot_body(body: str) -> dict | None:
    """Extract and decompress a telemetry snapshot from a GitHub Issue or comment body."""
    if not body:
        return None
    match = PAYLOAD_RE.search(body)
    if not match:
        return None
    b64 = re.sub(r'\s+', '', match.group(1))
    raw = gzip.decompress(base64.b64decode(b64))
    return json.loads(raw.decode('utf-8'))


def encode_relay_meta(meta: dict) -> str:
    clean = json.dumps(meta, sort_keys=True)
    lines = ['### Live W&B Relay Status']
    if meta.get('run_url'):
        lines.append(f"- **Live W&B Run**: {meta['run_url']}")
    lines.append(f"- **Run ID**: `{meta.get('run_id', '')}` | **Auth**: `{meta.get('auth_mode', '')}`")
    lines.append(f"- **Last synced step**: `{meta.get('last_step', -1):,}` | **Stages synced**: `{meta.get('stages', [])}`")
    lines.append(f"- **Source**: `{meta.get('source', '')}` | **Updated**: `{meta.get('updated_iso', '')}`")
    lines.append(f"\n<!-- WANDB_RELAY_META_START -->\n{clean}\n<!-- WANDB_RELAY_META_END -->\n")
    return '\n'.join(lines)


def decode_relay_meta(body: str) -> dict | None:
    if not body:
        return None
    match = META_RE.search(body)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def build_wandb_events(snapshot: dict, last_step: int, logged_stages: set[int]) -> tuple[list[tuple[int, dict]], set[int], dict]:
    """Build strictly ordered, idempotent `(step, metrics)` pairs and run summary updates."""
    state = snapshot.get('state') or {}
    config = state.get('config') or {}
    rounds = int(config.get('rounds', 40))
    screen_seed = int(config.get('screen_seed', 1009))
    confirm_seed = int(config.get('confirm_seed', 2003))
    target = float(state.get('target_ranking') or 1700.0)
    extras = snapshot.get('eval_extras') or {}

    events: dict[int, dict] = {}
    for row in snapshot.get('rows') or []:
        if 'Total Timesteps' not in row:
            continue
        step = int(row['Total Timesteps'])
        if step <= last_step:
            continue
        rec = events.setdefault(step, {'Total Timesteps': step})
        for k, v in row.items():
            if k == '_time' or v is None or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out_key = ('Live/' + k) if k.startswith('Rating/') else k
                rec[out_key] = v

    new_stages: set[int] = set()
    best_screen_so_far = None
    best_confirm_so_far = None
    for entry in state.get('history') or []:
        stage = int(entry['stage'])
        step = int(entry['steps'])
        if entry.get('promoted'):
            best_screen_so_far = entry['screen']['Ranking']
            if entry.get('confirm'):
                best_confirm_so_far = entry['confirm']['Ranking']
        if stage in logged_stages:
            continue
        rec = events.setdefault(step, {'Total Timesteps': step})
        for m, v in (entry.get('screen') or {}).items():
            rec[f'Rating/{m}'] = v
        for m, v in (entry.get('confirm') or {}).items():
            rec[f'Confirmation/{m}'] = v
        rec['Campaign/Stage'] = stage
        rec['Campaign/Promoted'] = 1 if entry.get('promoted') else 0
        rec['Campaign/Target Ranking'] = target
        if best_screen_so_far is not None:
            rec['Campaign/Best Screen Ranking'] = best_screen_so_far
            rec['Campaign/Screen Gap To Target'] = max(0.0, target - best_screen_so_far)
        if best_confirm_so_far is not None:
            rec['Campaign/Best Confirm Ranking'] = best_confirm_so_far
            rec['Campaign/Confirm Gap To Target'] = max(0.0, target - best_confirm_so_far)
        if entry.get('steps_per_second') is not None:
            rec['Campaign/Steps Per Second'] = entry['steps_per_second']

        s_extra = extras.get(f'{step}-{screen_seed}-{rounds}') or {}
        c_extra = extras.get(f'{step}-{confirm_seed}-{2 * rounds}') or {}
        if s_extra.get('kickoff_pct') is not None:
            rec['Eval/Screen Kickoff First Touch %'] = s_extra['kickoff_pct']
        if c_extra.get('kickoff_pct') is not None:
            rec['Eval/Confirm Kickoff First Touch %'] = c_extra['kickoff_pct']
        for m, pct in (c_extra.get('mode_goal_pct') or s_extra.get('mode_goal_pct') or {}).items():
            rec[f'Eval/Goal Share {m} %'] = pct
        new_stages.add(stage)

    summary = {
        'Campaign/Status': state.get('status', 'unknown'),
        'Campaign/Target Ranking': target,
        'Campaign/Stages Completed': max(0, len(state.get('history') or []) - 1),
    }
    best = state.get('best') or {}
    if best:
        summary['Campaign/Best Steps'] = best.get('steps')
        for m, v in (best.get('screen') or {}).items():
            summary[f'Best/Screen {m}'] = v
        for m, v in (best.get('confirm') or {}).items():
            summary[f'Best/Confirm {m}'] = v
    audit = state.get('audit') or {}
    if audit and state.get('history'):
        mapping = [
            ('Parent', config.get('parent_steps')),
            ('Best', best.get('steps')),
            ('Latest', state['history'][-1].get('steps')),
        ]
        for label, astep in mapping:
            scores = audit.get(str(astep)) or {}
            for m, v in scores.items():
                summary[f'Audit/{label} {m}'] = v

    ordered = [(s, events[s]) for s in sorted(events)]
    return ordered, new_stages, summary


def default_status_url(port: int = 8080) -> str | None:
    sandbox_id = os.environ.get('E2B_SANDBOX_ID', '').strip()
    if sandbox_id:
        return f'https://{port}-{sandbox_id}.e2b.app/status.json'
    return None


def gh_api_json(method: str, endpoint: str, payload: dict | None = None, repo: str | None = None):
    if endpoint.startswith('/'):
        url = endpoint
    else:
        if not repo:
            raise ValueError('repo is required for relative GitHub API endpoints')
        url = f'/repos/{repo}/{endpoint}'
    cmd = ['gh', 'api', '--method', method, url]
    stdin_data = None
    if payload is not None:
        cmd.extend(['--input', '-'])
        stdin_data = json.dumps(payload)
    out = subprocess.check_output(cmd, cwd=ROOT, input=stdin_data, text=True, stderr=subprocess.STDOUT)
    return json.loads(out) if out.strip() else {}


def detect_repo() -> str:
    env_repo = os.environ.get('GITHUB_REPOSITORY', '').strip()
    if env_repo:
        return env_repo
    url = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=ROOT, text=True).strip()
    match = re.search(r'github\.com[:/]([^/]+/[^/.]+?)(?:\.git)?$', url)
    if not match:
        raise RuntimeError(f'Cannot parse GitHub owner/repo from remote URL: {url}')
    return match.group(1)


def ensure_telemetry_issue(repo: str, branch: str, issue_number: int | None = None) -> int:
    if issue_number:
        return int(issue_number)
    issues = gh_api_json('GET', f'issues?state=open&per_page=30', repo=repo)
    tag = f'{TELEMETRY_LABEL_TITLE} ({branch})'
    for item in issues:
        if 'pull_request' not in item and item.get('title') == tag:
            return int(item['number'])
    created = gh_api_json('POST', 'issues', {
        'title': tag,
        'body': f'Initializing live W&B telemetry channel for branch `{branch}`...\n',
    }, repo=repo)
    return int(created['number'])


def find_relay_meta_on_issue(repo: str, issue_number: int) -> tuple[int | None, dict | None]:
    comments = gh_api_json('GET', f'issues/{issue_number}/comments?per_page=50', repo=repo)
    for c in reversed(comments):
        meta = decode_relay_meta(c.get('body') or '')
        if meta is not None:
            return int(c['id']), meta
    return None, None


def fetch_url_json(url: str, timeout: float = 10.0, token: str | None = None) -> dict:
    headers = {'Accept': 'application/json', 'User-Agent': 'rocket-league-bot-wandb-bridge/1.0'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def run_publisher(args):
    """Sandbox daemon: publish live metrics snapshots via GitHub API and write local relay status."""
    repo = args.repo or detect_repo()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    status_url = args.status_url if args.status_url is not None else default_status_url(args.port)
    issue_number = ensure_telemetry_issue(repo, branch, args.issue)
    status_file = args.status_file or (args.campaign / 'wandb_bridge.json')

    print(f'Publishing live telemetry for {args.campaign} -> {repo}#{issue_number} (interval={args.interval}s)', flush=True)
    last_digest = None
    while True:
        state_status = 'running'
        try:
            snap = read_campaign_snapshot(args.campaign, status_url=status_url)
            state_status = (snap.get('state') or {}).get('status', 'running')
            _, relay_meta = find_relay_meta_on_issue(repo, issue_number)
            # Only trim below steps already confirmed synced by the cloud relay when needed.
            body, included = encode_snapshot_body(snap)
            digest = hashlib.sha256(json.dumps({
                'rows_len': len(snap['rows']),
                'last_step': snap['rows'][-1]['Total Timesteps'] if snap['rows'] else -1,
                'history_len': len((snap['state'].get('history') or [])),
                'status': state_status,
                'audit': bool(snap['state'].get('audit')),
                'status_url': status_url,
            }, sort_keys=True).encode()).hexdigest()
            if digest != last_digest:
                gh_api_json('PATCH', f'issues/{issue_number}', {'body': body}, repo=repo)
                last_digest = digest
                print(
                    f'Published telemetry #{issue_number}: status={state_status} rows={included}/{len(snap["rows"])} '
                    f'stages={max(0, len(snap["state"].get("history") or []) - 1)}'
                    + (f' wandb={relay_meta.get("run_url")}' if relay_meta and relay_meta.get('run_url') else ''),
                    flush=True,
                )
            bridge_state = {
                'enabled': True,
                'repo': repo,
                'issue_number': issue_number,
                'issue_url': f'https://github.com/{repo}/issues/{issue_number}',
                'run_id': snap['run_id'],
                'status_url': status_url,
                'last_publish': time.time(),
                'rows_published': included,
                'rows_total': len(snap['rows']),
                'relay': relay_meta,
                'error': None,
            }
            tmp = status_file.with_suffix('.tmp')
            tmp.write_text(json.dumps(bridge_state, indent=2) + '\n')
            tmp.replace(status_file)
            if args.once or state_status in ('complete', 'paused', 'failed'):
                if not args.once:
                    # Stay briefly to capture the final relay confirmation URL.
                    time.sleep(min(20.0, args.interval))
                    _, relay_meta = find_relay_meta_on_issue(repo, issue_number)
                    bridge_state['relay'] = relay_meta
                    status_file.write_text(json.dumps(bridge_state, indent=2) + '\n')
                break
        except Exception as exc:
            print(f'Telemetry publish warning: {exc}', flush=True)
            if args.once:
                raise
        time.sleep(args.interval)


def load_best_remote_snapshot(args, repo: str | None, gh_token: str | None) -> tuple[dict | None, str]:
    """Fetch freshest telemetry snapshot: 1) direct HTTPS status_url, 2) GitHub Issue, 3) local fallback."""
    candidates: list[tuple[int, int, str, dict]] = []
    discovered_url = args.status_url

    # Read GitHub Issue payload (also discovers live preview URL automatically when rotated).
    if repo and args.issue:
        try:
            issue = fetch_url_json(f'https://api.github.com/repos/{repo}/issues/{args.issue}', token=gh_token)
            isnap = decode_snapshot_body(issue.get('body') or '')
            if isnap:
                if not discovered_url and isnap.get('status_url'):
                    discovered_url = isnap['status_url']
                max_step = max((int(r['Total Timesteps']) for r in isnap.get('rows') or []), default=-1)
                hist_len = len(((isnap.get('state') or {}).get('history')) or [])
                candidates.append((hist_len, max_step, f'github-issue#{args.issue}', isnap))
        except Exception as exc:
            print(f'Issue poll note: {exc}', flush=True)

    # Try direct live-preview /status.json for complete un-trimmed real-time rows.
    if discovered_url:
        try:
            raw = fetch_url_json(discovered_url, timeout=8.0)
            if 'state' in raw and 'rows' in raw:
                raw.setdefault('run_id', campaign_run_id((raw['state'].get('config')) or {}))
                # Merge eval_extras from issue snapshot if present.
                if candidates and 'eval_extras' in candidates[0][3]:
                    raw.setdefault('eval_extras', candidates[0][3]['eval_extras'])
                max_step = max((int(r['Total Timesteps']) for r in raw.get('rows') or []), default=-1)
                hist_len = len((raw['state'].get('history')) or [])
                candidates.append((hist_len + 1, max_step, 'live-preview-http', raw))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            print(f'Preview poll note ({discovered_url}): {exc}', flush=True)

    # Optional local fallback (e.g. checked-out training-backup/ inside GitHub Actions).
    if args.fallback_campaign and (args.fallback_campaign / 'state.json').is_file():
        try:
            lsnap = read_campaign_snapshot(args.fallback_campaign)
            max_step = max((int(r['Total Timesteps']) for r in lsnap.get('rows') or []), default=-1)
            hist_len = len(((lsnap.get('state') or {}).get('history')) or [])
            candidates.append((hist_len, max_step, f'local:{args.fallback_campaign.name}', lsnap))
        except Exception as exc:
            print(f'Fallback campaign note: {exc}', flush=True)

    if not candidates:
        return None, 'none'

    # Merge rows from all available sources sharing the same run_id so trimmed frames never miss early steps!
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, best_source, best_snap = candidates[0]
    target_run_id = best_snap.get('run_id')
    merged_rows: dict[int, dict] = {}
    merged_extras: dict = {}
    for _, _, _, snap in reversed(candidates):
        if snap.get('run_id') != target_run_id:
            continue
        merged_extras.update(snap.get('eval_extras') or {})
        for row in snap.get('rows') or []:
            if 'Total Timesteps' in row:
                merged_rows[int(row['Total Timesteps'])] = row
    best_snap = dict(best_snap, rows=[merged_rows[k] for k in sorted(merged_rows)], eval_extras=merged_extras)
    return best_snap, best_source


def post_relay_meta_comment(repo: str, issue_number: int, comment_id: int | None, meta: dict) -> int | None:
    body = encode_relay_meta(meta)
    try:
        if comment_id is None:
            comment_id, _ = find_relay_meta_on_issue(repo, issue_number)
        if comment_id is not None:
            gh_api_json('PATCH', f'issues/comments/{comment_id}', {'body': body}, repo=repo)
            return comment_id
        created = gh_api_json('POST', f'issues/{issue_number}/comments', {'body': body}, repo=repo)
        return int(created['id'])
    except Exception as exc:
        print(f'Relay metadata comment note: {exc}', flush=True)
        return comment_id


def run_relay(args):
    """Cloud / external daemon: pull live telemetry and stream directly to online W&B."""
    import wandb

    repo = args.repo or (detect_repo() if args.issue else None)
    gh_token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    deadline = time.monotonic() + args.max_minutes * 60.0

    # Wait for first snapshot to obtain campaign config & deterministic run_id.
    snap, source = None, 'none'
    while time.monotonic() < deadline:
        snap, source = load_best_remote_snapshot(args, repo, gh_token)
        if snap is not None:
            break
        if args.once:
            raise RuntimeError('No telemetry snapshot available to relay')
        print('Waiting for first telemetry snapshot...', flush=True)
        time.sleep(args.interval)
    if snap is None:
        raise RuntimeError('Timed out waiting for telemetry snapshot')

    has_api_key = bool(os.environ.get('WANDB_API_KEY', '').strip())
    if not has_api_key and not args.allow_anonymous:
        raise RuntimeError('WANDB_API_KEY is not set (add it as a GitHub Actions secret or pass --allow-anonymous)')

    auth_mode = 'api_key' if has_api_key else 'anonymous'
    state = snap.get('state') or {}
    config = state.get('config') or {}
    base_run_id = args.run_id or snap.get('run_id') or campaign_run_id(config)
    # Keep anonymous and authenticated run IDs distinct so switching to WANDB_API_KEY backfills cleanly.
    run_id = base_run_id if has_api_key else f'{base_run_id}-anon'
    run_name = args.run_name or f'weights1700-{config.get("parent_steps", "live")}'
    project = args.project or os.environ.get('WANDB_PROJECT') or 'rocket-league-ggl'
    entity = (args.entity or os.environ.get('WANDB_ENTITY') or '').strip() or None
    group = args.group or os.environ.get('WANDB_RUN_GROUP') or 'weights-1700'

    args.work_dir.mkdir(parents=True, exist_ok=True)
    init_kwargs = dict(
        mode='online',
        id=run_id,
        resume='allow',
        project=project,
        group=group,
        name=run_name,
        dir=str(args.work_dir.resolve()),
        config=config,
        settings=wandb.Settings(
            console='off',
            disable_code=True,
            disable_git=True,
            x_disable_stats=True,
            x_disable_meta=True,
            init_timeout=60,
        ),
    )
    if has_api_key and entity:
        init_kwargs['entity'] = entity
    if not has_api_key:
        init_kwargs['anonymous'] = 'must'

    run = wandb.init(**init_kwargs)
    run.define_metric('Total Timesteps')
    run.define_metric('*', step_metric='Total Timesteps')
    run_url = str(run.url or '')
    print(f'WANDB_LIVE_URL={run_url} (run_id={run_id}, auth={auth_mode}, project={project})', flush=True)

    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_path:
        with open(summary_path, 'a') as sf:
            sf.write(f"## Live Weights & Biases Stream\n- **Run URL**: {run_url}\n- **Run ID**: `{run_id}`\n- **Auth**: `{auth_mode}`\n")

    last_step = int(run.summary.get('Stream/Last Timesteps', -1))
    logged_stages = {int(s) for s in (run.summary.get('Stream/Stages') or [])}
    comment_id = None
    last_comment_push = 0.0
    idle_since = None

    try:
        while True:
            if snap is not None:
                cur_run_id = args.run_id or snap.get('run_id') or base_run_id
                if cur_run_id != base_run_id:
                    print(f'New campaign detected ({cur_run_id} != {base_run_id}); closing current run cleanly.', flush=True)
                    break
                ordered, new_stages, summary = build_wandb_events(snap, last_step, logged_stages)
                for step, record in ordered:
                    run.log(record)
                    last_step = max(last_step, step)
                logged_stages |= new_stages
                if ordered or new_stages:
                    idle_since = None
                    print(
                        f'Logged {len(ordered)} step(s), +{len(new_stages)} stage(s) via {source} -> '
                        f'last_step={last_step:,} status={summary.get("Campaign/Status")} url={run_url}',
                        flush=True,
                    )
                elif idle_since is None:
                    idle_since = time.monotonic()

                run.summary['Stream/Last Timesteps'] = last_step
                run.summary['Stream/Stages'] = sorted(logged_stages)
                run.summary['Stream/Source'] = source
                for k, v in summary.items():
                    if v is not None:
                        run.summary[k] = v

                now_m = time.monotonic()
                if repo and args.issue and (ordered or new_stages or comment_id is None or now_m - last_comment_push >= 120.0):
                    meta = {
                        'run_url': run_url,
                        'run_id': run_id,
                        'project': project,
                        'auth_mode': auth_mode,
                        'last_step': last_step,
                        'stages': sorted(logged_stages),
                        'source': source,
                        'campaign_status': summary.get('Campaign/Status'),
                        'updated_iso': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                    }
                    comment_id = post_relay_meta_comment(repo, args.issue, comment_id, meta)
                    last_comment_push = now_m

                cstatus = summary.get('Campaign/Status')
                if args.once or cstatus in ('complete', 'failed'):
                    print(f'Campaign reached terminal state ({cstatus}); finishing W&B stream.', flush=True)
                    break
                if cstatus == 'paused' and idle_since and (time.monotonic() - idle_since) >= 180.0:
                    print('Campaign paused and all metrics flushed; finishing W&B stream.', flush=True)
                    break
                if idle_since and (time.monotonic() - idle_since) >= args.idle_timeout_minutes * 60.0:
                    print(f'No new metrics for {args.idle_timeout_minutes}m; exiting cleanly.', flush=True)
                    break

            if time.monotonic() >= deadline:
                print('Reached relay max_minutes budget; run remains resumable by ID.', flush=True)
                break
            time.sleep(args.interval)
            snap, source = load_best_remote_snapshot(args, repo, gh_token)
    finally:
        run.finish()
        print(f'Closed/flushed W&B stream cleanly: {run_url}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)

    pub = sub.add_parser('publish', help='Sandbox side: publish live campaign telemetry to a GitHub Issue')
    pub.add_argument('campaign', type=Path)
    pub.add_argument('--repo', help='GitHub owner/repo (default: inferred from git origin)')
    pub.add_argument('--issue', type=int, help='Existing telemetry issue number (default: find or create)')
    pub.add_argument('--status-url', help='Live preview /status.json URL (default: inferred from E2B_SANDBOX_ID)')
    pub.add_argument('--port', type=int, default=8080)
    pub.add_argument('--interval', type=float, default=20.0)
    pub.add_argument('--status-file', type=Path, help='Local JSON file recording live W&B bridge status/URL')
    pub.add_argument('--once', action='store_true')

    rel = sub.add_parser('relay', help='Cloud/external side: stream live telemetry into online W&B')
    rel.add_argument('--status-url', help='Direct HTTPS URL of training_service.py /status.json')
    rel.add_argument('--repo', help='GitHub owner/repo for Issue fallback & status writeback')
    rel.add_argument('--issue', type=int, help='GitHub telemetry issue number')
    rel.add_argument('--fallback-campaign', type=Path, help='Optional local snapshot folder (e.g. training-backup)')
    rel.add_argument('--project', default='')
    rel.add_argument('--entity', default='')
    rel.add_argument('--group', default='weights-1700')
    rel.add_argument('--run-id', default='')
    rel.add_argument('--run-name', default='')
    rel.add_argument('--interval', type=float, default=15.0)
    rel.add_argument('--max-minutes', type=float, default=330.0)
    rel.add_argument('--idle-timeout-minutes', type=float, default=45.0)
    rel.add_argument('--work-dir', type=Path, default=Path('/tmp/wandb-live-relay'))
    rel.add_argument('--allow-anonymous', action='store_true',
                     help='Stream to a claimable anonymous online W&B run when WANDB_API_KEY is unset')
    rel.add_argument('--once', action='store_true')

    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    if args.mode == 'publish':
        run_publisher(args)
    else:
        run_relay(args)


if __name__ == '__main__':
    main()
