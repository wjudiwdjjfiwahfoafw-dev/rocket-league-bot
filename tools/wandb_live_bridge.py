#!/usr/bin/env python3
"""Live Weights & Biases telemetry bridge for sandboxed Rocket League bot training.

Works even when direct outbound traffic from the training sandbox to api.wandb.ai is blocked:

1. Sandbox publisher mode (`publish`):
   Tails a running campaign's durable `state.json` and `metrics/*.jsonl` (read-only, never
   acquiring the trainer lock) and publishes compact gzip+base64 metric snapshots to a single
   dedicated GitHub Issue via `gh api` (`api.github.com`). Uses near-zero CPU, runs no local
   web server, and creates zero Git commits.

2. Cloud / external relay mode (`relay`):
   Runs anywhere with outbound access to `api.wandb.ai` (e.g. `.github/workflows/wandb-live.yml`
   or any external machine). Polls the GitHub Issue live telemetry feed (with `training-backup/`
   fallback) and streams every training row, frozen-pool Screen/Confirmation Elo rating, and
   held-out Audit summary idempotently into a single resumable online W&B run
   (`step_metric='Total Timesteps'`).
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
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY_FILE = Path('/dev/shm/.wandb_bridge_key')
TELEMETRY_LABEL_TITLE = '[telemetry] Live W&B metrics feed'
PAYLOAD_RE = re.compile(r'<!-- WANDB_SNAPSHOT_GZ_B64_START -->\s*([A-Za-z0-9+/=\s]+?)\s*<!-- WANDB_SNAPSHOT_GZ_B64_END -->')
META_RE = re.compile(r'<!-- WANDB_RELAY_META_START -->\s*(\{.*?\})\s*<!-- WANDB_RELAY_META_END -->', re.DOTALL)
KEYREQ_RE = re.compile(r'<!-- WANDB_KEYREQ_START -->\s*(\{.*?\})\s*<!-- WANDB_KEYREQ_END -->', re.DOTALL)
KEYSEAL_RE = re.compile(r'<!-- WANDB_KEYSEAL_START -->\s*(\{.*?\})\s*<!-- WANDB_KEYSEAL_END -->', re.DOTALL)
# GitHub issue/comment body limit is 65,536 chars; keep comfortable headroom for header + metadata.
MAX_BODY_B64 = 52_000


def _shm_dir() -> Path:
    shm = Path('/dev/shm')
    return shm if shm.is_dir() else Path(tempfile.gettempdir())


def generate_ephemeral_rsa_keypair() -> tuple[bytes, str, str]:
    """Generate a one-time 3072-bit RSA keypair in RAM. Returns (priv_pem_bytes, pub_pem_str, key_id)."""
    with tempfile.TemporaryDirectory(dir=_shm_dir(), prefix='wbk-') as td:
        os.chmod(td, 0o700)
        priv_path = Path(td) / 'priv.pem'
        pub_path = Path(td) / 'pub.pem'
        subprocess.run(
            ['openssl', 'genpkey', '-algorithm', 'RSA', '-pkeyopt', 'rsa_keygen_bits:3072', '-out', str(priv_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.chmod(priv_path, 0o600)
        subprocess.run(
            ['openssl', 'pkey', '-in', str(priv_path), '-pubout', '-out', str(pub_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        priv_bytes = priv_path.read_bytes()
        pub_pem = pub_path.read_text('ascii')
        priv_path.write_bytes(b'\x00' * len(priv_bytes))
    key_id = hashlib.sha256(pub_pem.encode('ascii')).hexdigest()[:16]
    return priv_bytes, pub_pem, key_id


def rsa_oaep_seal(pub_pem: str, secret: str) -> str:
    """Encrypt `secret` under a 3072-bit RSA public key using RSA-OAEP-SHA256; return base64 ciphertext."""
    with tempfile.TemporaryDirectory(dir=_shm_dir(), prefix='wbs-') as td:
        os.chmod(td, 0o700)
        pub_path = Path(td) / 'pub.pem'
        pub_path.write_text(pub_pem, encoding='ascii')
        # Verify public key strength (>= 3072-bit RSA) before encrypting.
        info = subprocess.check_output(
            ['openssl', 'pkey', '-pubin', '-in', str(pub_path), '-text', '-noout'],
            text=True, stderr=subprocess.DEVNULL,
        )
        if 'Public-Key: (3072 bit)' not in info and 'Public-Key: (4096 bit)' not in info:
            raise ValueError('Rejected public key with insufficient bit length')
        ct = subprocess.check_output(
            ['openssl', 'pkeyutl', '-encrypt', '-pubin', '-inkey', str(pub_path),
             '-pkeyopt', 'rsa_padding_mode:oaep', '-pkeyopt', 'rsa_oaep_md:sha256',
             '-pkeyopt', 'rsa_mgf1_md:sha256'],
            input=secret.encode('utf-8'), stderr=subprocess.DEVNULL,
        )
    return base64.b64encode(ct).decode('ascii')


def rsa_oaep_open(priv_pem: bytes, sealed_b64: str) -> str:
    """Decrypt a base64 RSA-OAEP-SHA256 ciphertext using the in-RAM ephemeral private key."""
    ct = base64.b64decode(sealed_b64)
    with tempfile.TemporaryDirectory(dir=_shm_dir(), prefix='wbo-') as td:
        os.chmod(td, 0o700)
        priv_path = Path(td) / 'priv.pem'
        priv_path.write_bytes(priv_pem)
        os.chmod(priv_path, 0o600)
        try:
            pt = subprocess.check_output(
                ['openssl', 'pkeyutl', '-decrypt', '-inkey', str(priv_path),
                 '-pkeyopt', 'rsa_padding_mode:oaep', '-pkeyopt', 'rsa_oaep_md:sha256',
                 '-pkeyopt', 'rsa_mgf1_md:sha256'],
                input=ct, stderr=subprocess.DEVNULL,
            )
        finally:
            priv_path.write_bytes(b'\x00' * len(priv_pem))
    return pt.decode('utf-8').strip()


PERSISTENT_KEY_FILE = Path(__file__).resolve().parents[1] / '.secrets' / 'wandb_api_key'


def _load_local_wandb_key(key_file: Path) -> str:
    """WANDB_API_KEY first, then --key-file, then the gitignored workspace copy.

    /dev/shm is wiped when the sandbox restarts, so a key that lives only there has to be pasted again for
    every session. .secrets/wandb_api_key lives with the workspace (and is gitignored, so the 30-minute
    backup worker can never commit it). The repository secret WANDB_API_KEY is still the best home: the
    relay then authenticates itself and needs no sandbox key at all.
    """
    env_val = os.environ.get('WANDB_API_KEY', '').strip()
    if env_val:
        return env_val
    for candidate in (key_file, PERSISTENT_KEY_FILE):
        try:
            if candidate.is_file():
                key = candidate.read_text(encoding='utf-8').strip()
                if key:
                    return key
        except OSError:
            pass
    return ''


def _verify_active_workflow_run(repo: str, run_id: str, branch: str) -> bool:
    if not run_id or not str(run_id).isdecimal():
        return False
    try:
        info = gh_api_json('GET', f'actions/runs/{run_id}', repo=repo)
    except Exception:
        return False
    return (
        info.get('status') == 'in_progress'
        and info.get('head_branch') == branch
        and str(info.get('path', '')).endswith('/wandb-live.yml')
    )


def answer_pending_key_requests(repo: str, branch: str, issue_number: int, comments: list[dict],
                                secret: str, answered: set[str]) -> bool:
    """Seal `secret` only for a verified in-progress github-actions[bot] workflow run of this branch."""
    if not secret:
        return False
    already_sealed = {}
    active_reqs = set()
    for c in comments:
        body_c = c.get('body') or ''
        m = KEYSEAL_RE.search(body_c)
        if m:
            try:
                already_sealed[json.loads(m.group(1)).get('key_id', '')] = int(c['id'])
            except ValueError:
                pass
        mr = KEYREQ_RE.search(body_c)
        if mr:
            try:
                active_reqs.add(json.loads(mr.group(1)).get('key_id', ''))
            except ValueError:
                pass
    # Delete any already-consumed sealed replies whose request comment was deleted by the runner.
    for kid, cid in list(already_sealed.items()):
        if kid and kid not in active_reqs:
            try:
                gh_api_json('DELETE', f'issues/comments/{cid}', repo=repo)
            except Exception:
                pass
    did_seal = False
    for c in comments:
        user = c.get('user') or {}
        if user.get('login') != 'github-actions[bot]' or user.get('type') != 'Bot':
            continue
        m = KEYREQ_RE.search(c.get('body') or '')
        if not m:
            continue
        try:
            req = json.loads(m.group(1))
        except ValueError:
            continue
        key_id = str(req.get('key_id', ''))
        pub_pem = str(req.get('pub_pem', ''))
        run_id = str(req.get('run_id', ''))
        if not key_id or key_id in answered or key_id in already_sealed:
            continue
        if hashlib.sha256(pub_pem.encode('ascii')).hexdigest()[:16] != key_id:
            continue
        if not _verify_active_workflow_run(repo, run_id, branch):
            continue
        sealed_b64 = rsa_oaep_seal(pub_pem, secret)
        payload = json.dumps({'key_id': key_id, 'run_id': run_id, 'sealed_b64': sealed_b64})
        body = f"Ephemeral RSA-OAEP handshake reply for `{key_id}`.\n<!-- WANDB_KEYSEAL_START -->\n{payload}\n<!-- WANDB_KEYSEAL_END -->\n"
        gh_api_json('POST', f'issues/{issue_number}/comments', {'body': body}, repo=repo)
        answered.add(key_id)
        did_seal = True
        print(f'Completed ephemeral RSA-3072-OAEP handshake for verified Actions run {run_id} (key_id={key_id})', flush=True)
    return did_seal


def request_sealed_wandb_key(repo: str, issue_number: int, timeout_s: float = 150.0) -> str | None:
    """From inside a GitHub Actions runner, acquire WANDB_API_KEY via one-time RSA-3072-OAEP exchange."""
    run_id = os.environ.get('GITHUB_RUN_ID', '').strip()
    if not run_id:
        return None
    priv_pem, pub_pem, key_id = generate_ephemeral_rsa_keypair()
    req_comment_id = None
    seal_comment_id = None
    try:
        req_json = json.dumps({'key_id': key_id, 'run_id': run_id, 'pub_pem': pub_pem})
        body = f"Ephemeral RSA-3072-OAEP handshake request (`{key_id}`, run `{run_id}`).\n<!-- WANDB_KEYREQ_START -->\n{req_json}\n<!-- WANDB_KEYREQ_END -->\n"
        created = gh_api_json('POST', f'issues/{issue_number}/comments', {'body': body}, repo=repo)
        req_comment_id = int(created['id'])
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(4.0)
            comments = gh_api_json('GET', f'issues/{issue_number}/comments?per_page=50', repo=repo)
            for c in comments:
                m = KEYSEAL_RE.search(c.get('body') or '')
                if not m:
                    continue
                try:
                    data = json.loads(m.group(1))
                except ValueError:
                    continue
                if data.get('key_id') == key_id and data.get('sealed_b64'):
                    seal_comment_id = int(c['id'])
                    key = rsa_oaep_open(priv_pem, data['sealed_b64'])
                    if os.environ.get('GITHUB_ACTIONS') == 'true' and key:
                        print(f'::add-mask::{key}', flush=True)
                    return key
        return None
    finally:
        priv_pem = b''
        for cid in (req_comment_id, seal_comment_id):
            if cid is not None:
                try:
                    gh_api_json('DELETE', f'issues/comments/{cid}', repo=repo)
                except Exception:
                    pass


def campaign_run_id(config, prefix='rlbot-v4'):
    """Deterministic clean-schema W&B run ID; v4 excludes transport and Live-prefixed series."""
    ident = {
        'base': config.get('base'),
        'parent_steps': config.get('parent_steps'),
        'seed': config.get('seed'),
        'audit_seed': config.get('audit_seed'),
        'protocol': config.get('protocol'),
    }
    digest = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
    return f'{prefix}-{config.get("parent_steps", "run")}-{digest}'


def snapshot_digest(snapshot: dict, state_status: str) -> str:
    """Fingerprint of everything the published Issue body carries.

    The anchored evaluation files are part of that body, so a change in them has to count as a
    change: without them an evaluation that only refreshes an already known stage would leave the
    published snapshot untouched and its numbers stale.
    """
    rows = snapshot.get('rows') or []
    state = snapshot.get('state') or {}
    return hashlib.sha256(json.dumps({
        'rows_len': len(rows),
        'last_step': rows[-1]['Total Timesteps'] if rows else -1,
        'history_len': len(state.get('history') or []),
        'status': state_status,
        'audit': bool(state.get('audit')),
        'run_id': snapshot['run_id'],
        'eval_extras': hashlib.sha256(
            json.dumps(snapshot.get('eval_extras') or {}, sort_keys=True).encode()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()


def read_campaign_snapshot(campaign: Path) -> dict:
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
    return {
        'updated': time.time(),
        'run_id': campaign_run_id(state.get('config') or {}),
        'state': state,
        'rows': [rows_by_step[k] for k in sorted(rows_by_step)],
        'eval_extras': eval_extras,
    }


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
    if meta.get('phase'):
        lines.append(f"- **Relay phase**: `{meta['phase']}`")
    if meta.get('error'):
        lines.append(f"- **Relay note**: `{meta['error']}`")
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

    stage_boundaries = sorted(int(e['steps']) for e in (state.get('history') or []) if int(e.get('stage', 0)) > 0)
    events: dict[int, dict] = {}
    prev_row = None
    for row in snapshot.get('rows') or []:
        if 'Total Timesteps' not in row:
            continue
        step = int(row['Total Timesteps'])
        if step <= last_step:
            prev_row = row
            continue
        rec = events.setdefault(step, {'Total Timesteps': step})
        for k, v in row.items():
            if k == '_time' or v is None or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out_key = ('Elo/' + k.removeprefix('Rating/')) if k.startswith('Rating/') else k
                rec[out_key] = v
        # Across a stage evaluation/restart boundary, fold fixed-pool eval, promotion copy,
        # and trainer relaunch wall time into Overall Steps/Second so zero runtime is excluded.
        if prev_row is not None and row.get('_time') and prev_row.get('_time'):
            prev_step = int(prev_row['Total Timesteps'])
            dt = float(row['_time']) - float(prev_row['_time'])
            crossed_eval = any(prev_step <= b < step for b in stage_boundaries)
            if crossed_eval and step > prev_step and 0.0 < dt <= 1800.0:
                rec['Overall Steps/Second'] = (step - prev_step) / dt
        prev_row = row

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
        if entry.get('opponent_pool_steps') is not None:
            rec['Campaign/Opponent Pool Size'] = len(entry['opponent_pool_steps'])
            rec['Campaign/Opponent Pool Generation'] = entry.get('opponent_pool_generation', 0)
            rec['Campaign/Opponent Pool Promoted'] = int(bool(entry.get('opponent_pool_promoted')))
            if entry.get('strongest_opponent_ranking') is not None:
                rec['Campaign/Strongest Opponent Elo'] = entry['strongest_opponent_ranking']
                rec['Campaign/Pool Promotion Margin'] = entry.get('opponent_pool_margin')
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
    issue_number = ensure_telemetry_issue(repo, branch, args.issue)
    status_file = args.status_file or (args.campaign / 'wandb_bridge.json')

    print(f'Publishing live telemetry for {args.campaign} -> {repo}#{issue_number} (interval={args.interval}s)', flush=True)
    last_digest = None
    answered_keys: set[str] = set()
    while True:
        state_status = 'running'
        relay_ready = False
        try:
            snap = read_campaign_snapshot(args.campaign)
            state_status = (snap.get('state') or {}).get('status', 'running')
            comments = gh_api_json('GET', f'issues/{issue_number}/comments?per_page=50', repo=repo)
            secret = _load_local_wandb_key(args.key_file)
            if secret:
                answer_pending_key_requests(repo, branch, issue_number, comments, secret, answered_keys)
                # Clean up any leftover consumed keyseal comments older than our latest handshake.
            relay_meta = None
            for c in reversed(comments):
                relay_meta = decode_relay_meta(c.get('body') or '')
                if relay_meta is not None:
                    break
            relay_ready = bool(relay_meta and relay_meta.get('run_url') and relay_meta.get('run_id') == snap['run_id'])
            body, included = encode_snapshot_body(snap)
            digest = snapshot_digest(snap, state_status)
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
                'last_publish': time.time(),
                'rows_published': included,
                'rows_total': len(snap['rows']),
                'relay': relay_meta,
                'error': None,
            }
            tmp = status_file.with_suffix('.tmp')
            tmp.write_text(json.dumps(bridge_state, indent=2) + '\n')
            tmp.replace(status_file)
            terminal = state_status in ('complete', 'failed') or (state_status == 'paused' and not args.keep_alive)
            if args.once or terminal:
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
        # Poll faster (5s) while waiting for initial relay handshake, then relax to --interval.
        time.sleep(args.interval if relay_ready else min(5.0, args.interval))


def load_best_remote_snapshot(args, repo: str | None, gh_token: str | None) -> tuple[dict | None, str]:
    """Fetch freshest telemetry snapshot from the live GitHub Issue feed (with local snapshot fallback)."""
    candidates: list[tuple[int, int, int, str, dict]] = []

    if repo and args.issue:
        try:
            issue = fetch_url_json(f'https://api.github.com/repos/{repo}/issues/{args.issue}', token=gh_token)
            isnap = decode_snapshot_body(issue.get('body') or '')
            if isnap:
                max_step = max((int(r['Total Timesteps']) for r in isnap.get('rows') or []), default=-1)
                hist_len = len(((isnap.get('state') or {}).get('history')) or [])
                candidates.append((2, hist_len, max_step, f'github-issue#{args.issue}', isnap))
        except Exception as exc:
            print(f'Issue poll note: {exc}', flush=True)

    # Optional local fallback (e.g. checked-out training-backup/ inside GitHub Actions).
    if args.fallback_campaign and (args.fallback_campaign / 'state.json').is_file():
        try:
            lsnap = read_campaign_snapshot(args.fallback_campaign)
            max_step = max((int(r['Total Timesteps']) for r in lsnap.get('rows') or []), default=-1)
            hist_len = len(((lsnap.get('state') or {}).get('history')) or [])
            candidates.append((1, hist_len, max_step, f'local:{args.fallback_campaign.name}', lsnap))
        except Exception as exc:
            print(f'Fallback campaign note: {exc}', flush=True)

    if not candidates:
        return None, 'none'

    # Prefer live GitHub Issue over checked-out backup, and merge all rows with same run_id.
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _, _, _, best_source, best_snap = candidates[0]
    target_run_id = best_snap.get('run_id')
    merged_rows: dict[int, dict] = {}
    merged_extras: dict = {}
    for _, _, _, _, snap in reversed(candidates):
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


def load_relay_cursor(summary, relay_meta=None, expected_run_id=None):
    """Load the relay cursor from GitHub metadata and remove legacy Stream summary fields.

    The issue comment is the durable cursor for cloud relays; W&B summary fields are user-facing
    metrics and should not be polluted with transport bookkeeping.
    """
    legacy_step = int(summary.get('Stream/Last Timesteps', -1) or -1)
    legacy_stages = {int(s) for s in (summary.get('Stream/Stages') or [])}
    for key in ('Stream/Last Timesteps', 'Stream/Stages', 'Stream/Source'):
        try:
            del summary[key]
        except (KeyError, TypeError, AttributeError):
            pass

    if relay_meta and (expected_run_id is None or relay_meta.get('run_id') == expected_run_id):
        last_step = max(legacy_step, int(relay_meta.get('last_step', -1) or -1))
        stages = legacy_stages | {int(s) for s in (relay_meta.get('stages') or [])}
        return last_step, stages
    return legacy_step, legacy_stages


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
    auth_mode = 'repo_secret' if has_api_key else 'none'
    if not has_api_key and repo and args.issue:
        print('WANDB_API_KEY secret unset; requesting one-time RSA-3072-OAEP sealed key from sandbox...', flush=True)
        sealed_key = request_sealed_wandb_key(repo, args.issue, timeout_s=150.0)
        if sealed_key:
            os.environ['WANDB_API_KEY'] = sealed_key
            has_api_key = True
            auth_mode = 'ephemeral_rsa_oaep'
    if not has_api_key and not args.allow_anonymous:
        raise RuntimeError('WANDB_API_KEY is not set (add it as a GitHub Actions secret or keep sandbox publisher active)')
    if not has_api_key:
        auth_mode = 'anonymous'

    state = snap.get('state') or {}
    config = state.get('config') or {}
    base_run_id = args.run_id or snap.get('run_id') or campaign_run_id(config)
    run_id = base_run_id if has_api_key else f'{base_run_id}-anon'
    run_name = args.run_name or f'weights1700-{config.get("parent_steps", "live")}'
    project = args.project or os.environ.get('WANDB_PROJECT') or 'rocket-league-ggl'
    entity = (args.entity or os.environ.get('WANDB_ENTITY') or '').strip() or None
    group = args.group or os.environ.get('WANDB_RUN_GROUP') or 'weights-1700'

    comment_id = None
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

    try:
        if has_api_key:
            wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        # Scrub any potential credential fragment before reporting status.
        msg = type(exc).__name__ + ': ' + str(exc)[:200]
        cur_key = os.environ.get('WANDB_API_KEY', '')
        if cur_key:
            msg = msg.replace(cur_key, '[REDACTED]')
        if repo and args.issue:
            post_relay_meta_comment(repo, args.issue, None, {
                'run_url': '',
                'run_id': run_id,
                'project': project,
                'auth_mode': auth_mode,
                'last_step': -1,
                'stages': [],
                'source': source,
                'error': msg,
                'updated_iso': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            })
        raise
    run.define_metric('Total Timesteps')
    run.define_metric('*', step_metric='Total Timesteps')
    run_url = str(run.url or '')
    print(f'WANDB_LIVE_URL={run_url} (run_id={run_id}, auth={auth_mode}, project={project})', flush=True)

    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_path:
        with open(summary_path, 'a') as sf:
            sf.write(f"## Live Weights & Biases Stream\n- **Run URL**: {run_url}\n- **Run ID**: `{run_id}`\n- **Auth**: `{auth_mode}`\n")

    # Relay position lives in the GitHub issue metadata, not W&B's user-facing metrics.
    _, relay_meta = find_relay_meta_on_issue(repo, args.issue) if repo and args.issue else (None, None)
    last_step, logged_stages = load_relay_cursor(run.summary, relay_meta, expected_run_id=run_id)
    last_comment_push = 0.0
    if repo and args.issue:
        comment_id = post_relay_meta_comment(repo, args.issue, None, {
            'run_url': run_url, 'run_id': run_id, 'project': project, 'auth_mode': auth_mode,
            'last_step': last_step, 'stages': sorted(logged_stages), 'source': source,
            'phase': 'initialized',
            'updated_iso': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        })
        last_comment_push = time.monotonic()
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
                        'phase': 'streaming',
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
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)[:240]
        secret = os.environ.get('WANDB_API_KEY', '')
        if secret:
            error = error.replace(secret, '[REDACTED]')
        if repo and args.issue:
            post_relay_meta_comment(repo, args.issue, comment_id, {
                'run_url': run_url, 'run_id': run_id, 'project': project,
                'auth_mode': auth_mode, 'last_step': last_step,
                'stages': sorted(logged_stages), 'source': source,
                'campaign_status': (snap or {}).get('state', {}).get('status'),
                'phase': 'error', 'error': error,
                'updated_iso': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            })
        raise
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
    pub.add_argument('--interval', type=float, default=20.0)
    pub.add_argument('--status-file', type=Path, help='Local JSON file recording live W&B bridge status/URL')
    pub.add_argument('--key-file', type=Path, default=DEFAULT_KEY_FILE,
                     help='RAM-only path holding optional WANDB_API_KEY for ephemeral RSA-OAEP relay handshake')
    pub.add_argument('--keep-alive', action='store_true',
                     help='Keep the publisher available for relay handshakes while a campaign is paused')
    pub.add_argument('--once', action='store_true')

    rel = sub.add_parser('relay', help='Cloud/external side: stream live telemetry into online W&B')
    rel.add_argument('--repo', help='GitHub owner/repo for live Issue telemetry & status writeback')
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
