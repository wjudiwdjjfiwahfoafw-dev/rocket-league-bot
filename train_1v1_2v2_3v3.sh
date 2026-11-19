#!/usr/bin/env bash
# Continues the bundled model in 1v1, 2v2 and 3v3 on a CPU, and rates it in all three modes.
#
# Usage:   ./train_1v1_2v2_3v3.sh [extra GigaLearnBot options]
# Example: ./train_1v1_2v2_3v3.sh --max-minutes 120                     # train for 2 hours, then save and stop
#          ./train_1v1_2v2_3v3.sh --wandb --wandb-project my-project    # also log to Weights & Biases
#                                                                   # (needs WANDB_API_KEY or `wandb login`)
#
# Loads the newest numbered checkpoint and policy_versions pool from --checkpoints.
# Saves periodically, after rating sessions, and on exit; only the newest two checkpoints are kept by default.
# Ctrl+C / SIGTERM finishes the current iteration, saves, and exits.
#
# Each mode gets 15 arenas. The small slot-encoder model and its left-mirrored view match the shipped checkpoint.
# 15% of iterations train against the saved policy pool rather than pure self-play.
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

# Where the metrics receiver writes its .jsonl files
export GGL_METRICS_DIR="${GGL_METRICS_DIR:-metrics_1v1_2v2}"
# ... and how often: every 20th iteration, and every one with a rating (about 0.3 MB per hour instead of 5).
# wandb gets every iteration
export GGL_METRICS_EVERY="${GGL_METRICS_EVERY:-20}"
# W&B keeps its local run cache in /tmp (not in the workspace, which has a 128 MB limit)
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb_local}"
mkdir -p "$WANDB_DIR"
# W&B gets every 10th iteration, plus every rating (wandb overhead costs throughput on 2 cores)
export GGL_WANDB_EVERY="${GGL_WANDB_EVERY:-10}"

exec "$BOT" \
	--model s \
	--modes 1v1,2v2,3v3 \
	--mirror left \
	--checkpoints checkpoints \
	--device cpu \
	--games 45 \
	--ts-per-itr 48k \
	--batch-size 12000 \
	--epochs 1 \
	--lr 1.2e-4 \
	--entropy-scale 0.009 \
	--random-state-prob 0.5 \
	--random-ball-speed 2000 \
	--rewards VelocityBallToGoal=8,Goal=300,VelocityPlayerToBall=0,NearestVelocityToBall=0,TeamSpacing=1,TouchAccel=15,BehindBall=0,FaceBall=0 \
	--train-vs-old 0.15 \
	--ts-per-save 2m \
	--keep-checkpoints 2 \
	--skill-tracker \
	--skill-minutes 10 \
	--skill-rounds 10 \
	--skill-arenas 48 \
	--ts-per-version 150m \
	--max-versions 8 \
	--wandb-group training-1v1-2v2 \
	--run-name s-1v1-2v2 \
	"$@"
