#!/usr/bin/env bash
# Trains the bot for 1v1, 2v2 and 3v3 (model "l", see ARCHITECTURE.md) on a CPU.
#
# Usage:   ./train_multi.sh [extra GigaLearnBot options]
# Example: ./train_multi.sh --max-minutes 60        # train for 60 minutes, then save and stop
#          ./train_multi.sh --model m               # a new run with the medium model (in an empty checkpoint folder)
#
# Running it again continues from the newest checkpoint in ./checkpoints_multi.
#
# NOTE on disk space: a checkpoint of the model "l" is 57 MB, and two exist while one is being saved.
# In a workspace with a size limit (128 MB where this was made), give --checkpoints a folder outside of it.
# Stop it at any time with Ctrl+C (or `kill <pid>`): it finishes the current iteration, saves, and exits.
#
# The settings:
#   --model l             the network, about 5 million parameters. The model can't be changed for an existing checkpoint
#   --games 44            24 x 1v1, 12 x 2v2 and 8 x 3v3 at once: 48 players, and so the same number of timesteps, in each mode
#   --ts-per-itr 24k      a learning phase after every 24,000 timesteps ...
#   --batch-size 1500     ... with one policy update per 1,500 timesteps, each timestep used once (--epochs 1).
#                         In short test runs, more and smaller updates learned faster for the same computing time
#                         than fewer and bigger ones, or than using every timestep twice
#   --entropy-scale 0.018 the framework's default. The example's 0.035 kept the policy close to random for longer in two tests
#   --lr 2e-4, --random-state-prob 0.5    as in train_cpu.sh
#                         (NOTE: the run of ARCHITECTURE.md was made while the ball of the random start states still
#                         hung in the air. It now starts moving, see --random-ball-speed and ELO_REPORT.md.)
#
# Speed on a CPU with 2 threads: about 3,000 timesteps per second. On a GPU, start without "--device cpu" and with
# more games and bigger batches (e.g. --games 264 --ts-per-itr 100k --batch-size 25k).
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
export GGL_METRICS_DIR="${GGL_METRICS_DIR:-metrics_multi}"
# Every 10th iteration is written there (about 0.5 MB per hour). GGL_METRICS_EVERY=1 writes every iteration
export GGL_METRICS_EVERY="${GGL_METRICS_EVERY:-10}"

exec "$BOT" \
	--model l \
	--checkpoints checkpoints_multi \
	--device cpu \
	--games 44 \
	--ts-per-itr 24k \
	--batch-size 1500 \
	--epochs 1 \
	--lr 2e-4 \
	--entropy-scale 0.018 \
	--random-state-prob 0.5 \
	--ts-per-save 2m \
	--keep-checkpoints 2 \
	"$@"
