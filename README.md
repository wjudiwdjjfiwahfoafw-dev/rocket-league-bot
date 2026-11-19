# Rocket League Multi-Mode PPO Bot

Native C++ PPO training and RLBot integration for a single shared policy (`s`, left-mirrored `MultiModeObs`, 90-action `DefaultAction`) across **1v1, 2v2, and 3v3**, built on the bundled GigaLearnCPP / RLGymCPP / RocketSim stack.

## Current status — paused

Training is cleanly paused with no running background processes.

- **Canonical snapshot:** [`training-backup/`](training-backup/) (`status: paused`, SHA-256 verified via `training-backup/manifest.json`).
- **Latest resumable learner checkpoint:** `training-backup/learner/3250007238/` (full policy, critic, shared head, slot encoder, Adam optimizer states, and running stats; mode-aware v3 zero-sum reward recipe).
- **Selected best checkpoint in snapshot:** `training-backup/best/3244539846/` (native sampled actions at temperature `1.0` — Screen `1594.89`, Confirmation `1608.13` against the immutable 5-bot reference pool `aa048b45...`).
- **Target:** `1700.0` combined Elo (`(1v1 + 2v2 + 3v3) / 3`) on fixed-pool screen, confirmation, and held-out audit.
- **Metrics & backups:** Live metrics stream directly to Weights & Biases (`tools/wandb_live_bridge.py` + `.github/workflows/wandb-live.yml`; no local web dashboard). Automatic GitHub backups run headlessly every 30 minutes (`1800s`) via `tools/training_service.py`. `Overall Steps/Second` is measured over contiguous wall-clock windows so zero training, saving, evaluation, or reporting overhead is excluded.

## Build and resume

```bash
./setup_linux.sh
# Restricted Linux x86-64 / Python 3.11 CPU fallback:
# bash tools/setup_pypi_cpu.sh

# Inspect paused ratings without starting training:
python3 tools/show_rating.py training-backup

# Restore working copy only (no training started):
python3 tools/resume_training.py --restore-only

# Explicitly resume paused training (with 30m GitHub backup + W&B bridge):
python3 tools/resume_training.py
```

See [RANKED_TRAINING.md](RANKED_TRAINING.md) for the evaluation protocol, CLI options, W&B streaming, and recovery details.

## Repository layout

- `src/` — Bot entrypoint (`ExampleMain.cpp`), mode-aware zero-sum rewards, tactical state setter, and RLBot client.
- `GigaLearnCPP/`, `RLBotCPP/`, `rlbot/` — Bundled PPO/RLGymCPP/RocketSim framework and RLBot runner.
- `training-backup/` — Canonical paused campaign (`learner/`, `best/`, immutable `pool/`, `eval/`, `metrics/`, `state.json`, `manifest.json`).
- `tools/` — Campaign driver (`train_ranked.py`), recovery (`resume_training.py`), headless 30-minute GitHub backup worker (`training_service.py`), W&B streaming (`wandb_live_bridge.py`, `ranked_wandb.py`), rating viewer (`show_rating.py`), throughput profiler (`benchmark.py`), CPU wheel bootstrap (`setup_pypi_cpu.sh`), and regression tests (`test_*.py`).
- `.github/workflows/wandb-live.yml` — Cloud relay workflow for streaming live metrics to W&B from network-restricted environments.

## Deterministic viewing

```bash
build/GigaLearnBot --render --checkpoints training-backup/best --device cpu --modes 1v1,2v2,3v3
```

Streams RocketSimVis UDP frames in a single arena using masked argmax actions without training or saving.
