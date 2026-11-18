# Training status — paused

Training was paused gracefully at **3155997126**. The final full checkpoint and exact paused stage state are committed in [`training-backup/`](training-backup/), pushed to GitHub and SHA256-verified. Stage 5 has **1366.76 seconds** recorded; it is not complete or evaluated. No trainer, viewer or backup worker remains running.

## Selected PPO model

Checkpoint **3137998278** passed both fixed-pool tests and was promoted:

| Test | 1v1 | 2v2 | 3v3 | Combined Ranking |
|---|---:|---:|---:|---:|
| Screen | 1472.77 | 1652.35 | 1585.03 | **1570.05** |
| Confirmation | 1477.46 | 1656.52 | 1594.76 | **1576.24** |

The campaign parent's matching rankings were 1543.18 / 1555.39. The mean matched improvement is **+23.86 Elo**. There is no completed PPO held-out audit yet. Latest checkpoint does not mean strongest checkpoint.

## Verified 1600+ policy

The separately calibrated original policy passed **1608.18 screen / 1662.50 confirmation / 1659.84 held-out audit** against the unchanged frozen pool. This is an action-distribution calibration gain, not a claim that PPO training alone reached 1600. The policy, hashes and complete evaluations remain in the current tree. See [TARGET_1600.md](TARGET_1600.md).

## Continue later

```bash
python3 tools/resume_training.py
```

Build first if needed. This explicit command validates the canonical snapshot, restores a separate working campaign if absent, and resumes with browser monitoring and GitHub backups. It is **not** being run as part of cleanup. See [RANKED_TRAINING.md](RANKED_TRAINING.md).

Obsolete reports, two-mode/1v1 experiment scripts, old checkpoint `3034023852`, temporary experiments and duplicate live campaign data were removed. Historical evidence remains recoverable from Git history; no history was rewritten. Current source, tests, build scripts, original calibration source, campaign base, selected/latest PPO states and the verified calibrated policy are preserved.
