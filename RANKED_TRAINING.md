# Three-mode training and recovery

## Paused campaign

Training stopped gracefully at **3155997126**, partway through stage 5. The checkpoint includes policy, critic, shared head, slot encoder, Adam archives and running statistics. The authoritative copy is [`training-backup/`](training-backup/); `state.json` preserves the stage's elapsed budget. This snapshot is committed and pushed, not merely present on the sandbox disk.

```bash
./setup_linux.sh
python3 tools/resume_training.py
```

On restricted Linux x86-64 / CPython 3.11 machines, `bash tools/setup_pypi_cpu.sh` is the no-root build fallback (GCC/make required, ~880 MiB wheel download cached outside the repository). It was tested with Python 3.11.2. Build dependencies and compilation intermediates are ignored by Git.

`resume_training.py` verifies `manifest.json`, restores the snapshot into `results/ranking_3043214022/campaign/` only if no working campaign exists, and invokes the driver. Existing working progress is never overwritten. The snapshot itself is never used as a writable training directory. To restore without starting any process:

```bash
python3 tools/resume_training.py --restore-only
```

After restoring, the exact original campaign invocation also works:

```bash
python3 tools/train_ranked.py --base checkpoints/3043214022 \
  --out results/ranking_3043214022/campaign --minutes 180 --rounds 40 \
  --monitor --github-backup
```

Omit the monitoring/backup flags if intentionally running without them. The driver checks immutable settings and the frozen-pool hash, skips finished stages and uses the pending stage's remaining time. A lock prevents two drivers sharing a working campaign. SIGTERM/Ctrl+C once requests a native save and pause. Do not use SIGKILL if you can avoid it.

No training, viewer or background worker is left running after the cleanup pause.

## Selection protocol

**Rating/Ranking = (1v1 Elo + 2v2 Elo + 3v3 Elo) / 3.** These are goal-based simulator ratings against a specific pool, not human competitive ranks.

- Five frozen opponent versions, immutable original per-mode ratings; pool SHA256 `aa048b453da50d8c7c96913692d3e4d70bd141b5247cbd6519fe11589539d81c`.
- Each candidate starts at the reference pool's mean per mode. Only candidate Elo changes, K=5 per goal.
- Native sampled actions, seeded per-arena kickoff streams, scheduled opponents on both colors, 36 arenas (12 per mode). Scores average the final three quarters of rounds.
- Screen: seed **1009**, **40 rounds**. Potential improvements get confirmation: seed **2003**, **80 rounds**.
- Promote only if both matching-seed rankings improve and mean gain is at least **8 Elo**.
- End-of-campaign audit: seed **4001**, **80 rounds** for parent, best and latest. Audit does not select the PPO best.
- The 1600 target requires screen, confirmation **and** held-out audit >=1600. The separately calibrated policy passed; see [TARGET_1600.md](TARGET_1600.md).

The selected PPO best and latest weights differ. Continue training from `learner/`, not `best/`. The planned trainer budget is `[6,30,30,30,30,30,24]` minutes; evaluation adds wall time. A stronger latest timestep is never assumed without evaluation.

## Snapshot contents

| Path inside `training-backup/` | Purpose |
|---|---|
| `learner/3155997126/` | Latest full, resumable checkpoint |
| `learner/policy_versions/` | Mutable training opponent versions and their rolling ratings |
| `best/3137998278/` | Selected full PPO model, preserved independently |
| `pool/` | Immutable evaluation opponents and ratings |
| `state.json` | Config, history, selected model, paused stage and elapsed time |
| `metrics/*.jsonl` | Numeric metrics every 20 iterations and on live rating sessions |
| `eval/*.json` | Completed fixed-pool measurements |
| `manifest.json` | SHA256 of every snapshot file |

The base `checkpoints/3043214022/` must remain available because it is part of the campaign configuration and final audit. `checkpoints/3035220582/` preserves the source of the verified calibrated policy. Do not edit either source in place. The standalone preset writes to its `--checkpoints` directory; use a copy for experiments.

## Monitoring and off-machine backups

`--monitor --github-backup` starts a sandbox-local service with the trainer. Open the platform's **Live training metrics** preview, not a saved sandbox URL. The viewer refreshes every 10 seconds and plots every numeric field actually recorded in the JSONL, plus fixed-pool scores and separate calibration results. It cannot reconstruct diagnostics that were not recorded. It is not the W&B website.

The worker checks every minute for complete, settled checkpoint files. New snapshots are committed under `training-backup/` and pushed **only to the current branch**. It verifies the remote branch hash before reporting success. A failed push is retried. Latest, selected best, pools, state, evaluations and metrics are copied; credentials and W&B diagnostics are not.

The repository is public, so snapshots and metrics are public. An abrupt shutdown can lose work since the last successful remote backup. Git history retains older snapshots; this is suitable for these small CPU models, not unlimited large checkpoints. No force push/history rewrite is performed. A crash snapshot resets pending elapsed time conservatively; a graceful pause preserves the saved boundary's elapsed time. The viewer stops when the sandbox/service stops; pushed checkpoints survive.

To inspect paused scores without starting a process:

```bash
python3 tools/show_rating.py training-backup
```

## Training recipe and follow-up experiments

The current engine uses equal-player mode balance: 88 arenas split 48/24/16, each mode supplying 96 player decisions per step. Fixed rollouts bootstrap unfinished trajectories at each PPO update without resetting games or leaking trajectories across policy switches. GAE uses the matching bootstrap in reverse traversal order.

PPO receives 48,384 learner steps per iteration, four batches of 12,096, minibatches of 1,008, one epoch, LR `1e-4`, entropy `0.009`, half random starts and 15% saved opponents. Joint actor/critic inference shares head features; lean metrics reduce CPU overhead. The short benchmark measured ~18% more learner rows/s than the previous whole setup, **not** a guarantee of better Elo.

New experiments can explicitly override `--lr`, `--entropy-scale`, `--random-state-prob`, `--train-vs-old` and `--rewards`. Use a new `--out`: overrides are recorded in immutable configuration and cannot change a resumed run silently.

For policy calibration, build `cmake --build build --target CalibratePolicy -j 1`. The utility scales a **copied** policy's output weight and bias. It does not change fixture rules or opponent ratings. `tools/test_calibration_candidates.py` preserves provenance and separate score caches; it stops after a qualifying held-out audit. The saved target policy is already calibrated: no additional runtime temperature setting is needed. Its optimizer archives were copied, not adapted to transformed weights, so fine-tuning it requires a fresh experiment and reevaluation.

## Optional W&B

W&B is not needed to build, train, resume, monitor or back up. `--wandb-offline` requires the W&B SDK in the driver's Python and records a local offline segment; it is not live website reporting. `--wandb-online` tails metrics in a separate sandbox process with a stable resumable run ID. SDK credentials and permitted W&B outbound traffic are required. Both were unavailable during this session, and no online upload was verified. No external-PC relay remains.

## Regression tests

```bash
python3 -m unittest discover -s tools -p 'test_*.py'
python3 tools/test_native_rollout.py  # short real-model test, after building
```

Native `GigaLearnTests` and `GigaLearnMultiModeTests` cover GAE math, observations and joint inference. The Python tests exercise selection, budget/recovery, streaming, target verification and snapshot restoration.
