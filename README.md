# Rocket League PPO training

CPU training for one small (`s`) policy across **1v1, 2v2 and 3v3**, using the bundled GigaLearnCPP / RLGymCPP / RocketSim stack. Upstream source and license notices remain in the dependency directories. The original upstream README and historical experiments remain available in Git history.

## Current state: paused

- **Latest resumable checkpoint:** `3155997126`, saved during stage 5.
- **Selected PPO model:** `3137998278`, screen / confirmation Ranking **1570.05 / 1576.24**.
- **Calibrated sampled-action benchmark (not a learned-model improvement):** original at temperature 0.4, **1608.18 screen / 1662.50 confirmation / 1659.84 held-out audit**. See [model and evidence](TARGET_1600.md). Calibration is distinct from PPO training progress.
- **Canonical paused campaign:** [`training-backup/`](training-backup/), including optimizer state, opponent pools, evaluations, metrics and SHA256 manifest. This snapshot is committed and pushed to GitHub.

## Build and explicitly resume

```bash
./setup_linux.sh
# Restricted Linux CPU alternative: bash tools/setup_pypi_cpu.sh
python3 tools/resume_training.py
```

The resume command validates the backup hashes, restores a separate writable campaign, and starts the trainer, browser metrics viewer and GitHub backup worker. **Nothing starts automatically on checkout.** To restore without training:

```bash
python3 tools/resume_training.py --restore-only
```

See [training and recovery guide](RANKED_TRAINING.md) for requirements, direct commands, evaluation rules and limitations.

## Kept intentionally

- `src/`, `GigaLearnCPP/`, `RLBotCPP/`: trainer and its build dependencies; do not remove vendored dependencies needed by CMake.
- `checkpoints/3043214022/`: original base required by the paused campaign configuration.
- `checkpoints/3035220582/`: unmodified source of the calibrated policy, for reproducibility.
- `checkpoints/policy_versions/`: standalone preset's mutable opponent pool; distinct from the immutable evaluation pool.
- `training-backup/`: latest recoverable PPO state and selected best, not a scratch directory.
- `results/target1600-calibration/temperature-0.4/`: verified policy and all three evaluations.
- `tools/`: current training, evaluation, monitoring, recovery, setup and regression tests.

Older checkpoints, superseded 1v1/two-mode scripts, reports, trial outputs and benchmark builds were removed from the current tree. They remain in Git history; no history was rewritten.

## Deterministic viewing

```bash
build/GigaLearnBot --render --checkpoints training-backup/best --device cpu --modes 1v1,2v2,3v3
```

`--render` uses exactly one arena even if `--games` is supplied. At each goal/no-touch episode reset it independently samples a mode uniformly from `--modes` (repeats are possible). Actions are masked argmax, never sampled. RocketSimVis receives UDP frames; viewing does not train or save.

**Training remains stochastic at temperature 1.** Resume uses the uncalibrated PPO weights in `training-backup/learner/`, not the temperature-scaled artifact. Calibrated policies may be used as separately identified opponents; their sampled-action Elo is not a new learned-model rating. Positive temperature scaling does not change deterministic argmax actions.
