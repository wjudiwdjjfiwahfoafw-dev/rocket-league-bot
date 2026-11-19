#!/usr/bin/env python3
"""Report whether every saved old policy version is still byte-identical to its creation record.

Usage: python3 tools/verify_frozen_versions.py results/weights1700-v3/campaign
Exit code 1 means an old version was re-rated, re-annotated or otherwise rewritten.
"""
import datetime
import json
from pathlib import Path
import sys

from train_ranked import (file_hashes, frozen_versions, policy_versions, verify_frozen_versions)


def report(campaign):
    campaign = Path(campaign)
    state = json.loads((campaign / 'state.json').read_text())
    expected = state.get('frozen_versions') or {}
    rows, ok = [], True
    if not expected:
        print('No frozen-version manifest in state.json (legacy campaign); hashes below are informational.')
    pool = campaign / 'pool'
    trainer = campaign / 'learner/policy_versions'
    trainer_files = frozen_versions(trainer) if trainer.is_dir() else {}
    for version in policy_versions(pool):
        steps = int(version.name)
        files = file_hashes(version)
        recorded = expected.get(str(steps))
        stable = recorded is None or recorded == files
        copy_ok = trainer_files.get(steps) == files
        rows.append((steps, len(files), stable, copy_ok,
                     max(p.stat().st_mtime for p in version.iterdir() if p.is_file())))
        ok &= stable and copy_ok
    try:
        verify_frozen_versions(pool, expected)
        verify_frozen_versions(trainer, expected)
    except RuntimeError as error:
        ok = False
        print(f'FAIL: {error}')
    print(f'{"version":>13} {"files":>5} {"unchanged":>9} {"trainer copy":>12} {"newest file mtime":>20}')
    for steps, count, stable, copy_ok, mtime in rows:
        print(f'{steps:>13,} {count:>5} {"yes" if stable else "NO":>9} '
              f'{"same" if copy_ok else "DIFF":>12} '
              f'{datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"):>20}')
    print(f'\n{len(rows)} saved policy versions; '
          f'{"all constant since creation" if ok else "MISMATCH DETECTED"}')
    return ok


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip())
    sys.exit(0 if report(sys.argv[1]) else 1)
