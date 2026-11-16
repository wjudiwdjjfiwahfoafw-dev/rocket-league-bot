#!/usr/bin/env bash
# Trains one variant of the training setup from a common starting checkpoint, with GigaLearn's skill tracker on.
# A variant is train_1v1.sh plus the options you add, e.g. another batch size or other reward weights.
#
# Usage:   BASE_CKPT=<checkpoint folder> tools/run_setup_experiment.sh NAME [extra GigaLearnBot options]
# Example: BASE_CKPT=checkpoints_1v1/123456789 tools/run_setup_experiment.sh control
#          BASE_CKPT=checkpoints_1v1/123456789 tools/run_setup_experiment.sh lr-1e-4 --lr 1e-4
#
# With --wandb (and --wandb-project) each variant becomes a run of its own in the group "experiments".
#
# All variants start from the same policy. They play no rating games while training. Compare them afterwards: rate
# their final checkpoints against the same saved policy versions (see "Comparing training setups" in ELO_REPORT.md),
# e.g. the ones of the run they start from:
#   build/GigaLearnBot --modes 1v1 --device cpu --checkpoints experiments/NAME/ckpt --rate-rounds 90 \
#       --skill-arenas 32 --rate-versions checkpoints_1v1/policy_versions --rate-out ratings/NAME.json
#   tools/show_rating.py ratings/*.json
#
# Environment:
#   BASE_CKPT  checkpoint to start from (of the model of train_1v1.sh, or add --model)
#   EXP_DIR    folder for the runs (default: experiments)
#   STEPS      timesteps to train (default: 8000000)
#   BUILD_DIR  build folder (default: build)
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?usage: BASE_CKPT=<checkpoint> $0 NAME [extra options]}"
shift
BASE_CKPT="${BASE_CKPT:?set BASE_CKPT to the checkpoint folder to start from}"
EXP_DIR="${EXP_DIR:-experiments}"
STEPS="${STEPS:-8000000}"

RUN_DIR="$EXP_DIR/$NAME"
mkdir -p "$RUN_DIR/ckpt"
if [ -z "$(ls -A "$RUN_DIR/ckpt")" ]; then
	cp -r "$BASE_CKPT" "$RUN_DIR/ckpt/"
fi
BASE_STEPS="$(basename "$BASE_CKPT")"

export GGL_METRICS_DIR="$RUN_DIR/metrics"

# The settings of train_1v1.sh, without rating games during the run (--skill-minutes with a time that is never
# reached): they would only cost time, the variants are rated afterwards.
# --new-run: a metrics run of its own, instead of continuing the one of the starting checkpoint
exec ./train_1v1.sh \
	--checkpoints "$RUN_DIR/ckpt" \
	--skill-minutes 100000 \
	--new-run \
	--wandb-group experiments \
	--run-name "$NAME" \
	--max-timesteps "$((BASE_STEPS + STEPS))" \
	"$@"
