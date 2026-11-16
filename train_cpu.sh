#!/usr/bin/env bash
# CPU training preset (the settings used for the run documented in TRAINING_REPORT.md).
#
# Usage:   ./train_cpu.sh [extra GigaLearnBot options]
# Example: ./train_cpu.sh --max-minutes 30        # train for 30 minutes, then save and stop
#          ./train_cpu.sh --max-timesteps 50m     # train until 50 million total timesteps
#
# Running it again continues from the newest checkpoint in ./checkpoints.
# Stop it at any time with Ctrl+C (or `kill <pid>`): it finishes the current iteration, saves, and exits.
#
# Why these settings (the program defaults are the GPU values of the original example):
#   --games 64            enough games to keep both CPU threads busy, with little memory
#   --ts-per-itr 24k      smaller iterations than the example's 50k ...
#   --batch-size 6k       ... split into ~4 batches, so every iteration makes several policy updates
#                         (measured peak memory: 0.38 GB with these settings, 0.74 GB with the example's)
#   --epochs 2            the framework's default
#   --lr 2e-4             slightly above the example's 1.5e-4
#   --random-state-prob   half of the episodes start from a random position instead of a kickoff, to give the agent
#                         (NOTE: until the 1v1 work of ELO_REPORT.md, the ball of these episodes hung in the air out of
#                         reach, which wasted most of them. It now starts with up to 2000 uu/s, see --random-ball-speed.
#                         The run of TRAINING_REPORT.md was made with the hanging ball.)
#                         more situations close to the ball early on (not compared against kickoffs only)
#
# Not set here, so the example's value of 0.035 applies: --entropy-scale. In a short follow-up test,
# continuing with "--entropy-scale 0.01" learned faster (see "Follow-up test" in TRAINING_REPORT.md).
#
# Speed: about 21,000 steps per second on a CPU with 2 threads (see SPEED_REPORT.md, measure with tools/benchmark.py).
# "--epochs 1" gives about 28,000, with half as many policy updates per step.
#
# BUILD_DIR selects the build folder if you did not use the default ./build.
set -euo pipefail
cd "$(dirname "$0")"

BOT="${BUILD_DIR:-build}/GigaLearnBot"
if [ ! -x "$BOT" ]; then
	echo "$BOT does not exist yet. Run ./setup_linux.sh first (it downloads the dependencies and builds)." >&2
	exit 1
fi

# Don't let idle libtorch worker threads spin, they would steal CPU time from the game simulation threads
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-passive}"

# Where the metrics receiver writes its .jsonl files (set GGL_WANDB=1 to also log to Weights & Biases)
export GGL_METRICS_DIR="${GGL_METRICS_DIR:-metrics}"
# Every 10th iteration is written there (about 0.5 MB per hour). GGL_METRICS_EVERY=1 writes every iteration
export GGL_METRICS_EVERY="${GGL_METRICS_EVERY:-10}"

exec "$BOT" \
	--device cpu \
	--games 64 \
	--ts-per-itr 24k \
	--batch-size 6k \
	--epochs 2 \
	--lr 2e-4 \
	--random-state-prob 0.5 \
	--ts-per-save 1m \
	--keep-checkpoints 2 \
	"$@"
