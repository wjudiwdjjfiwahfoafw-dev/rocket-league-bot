#!/usr/bin/env bash
# Trains one bot for 1v1, 2v2 and 3v3 on a CPU, and rates it in both modes with GigaLearn's skill tracker
# (the setup of the last part of ELO_REPORT.md).
#
# Usage:   ./train_1v1_2v2_3v3.sh [extra GigaLearnBot options]
# Example: ./train_1v1_2v2_3v3.sh --max-minutes 120                     # train for 2 hours, then save and stop
#          ./train_1v1_2v2_3v3.sh --wandb --wandb-project my-project    # also log to Weights & Biases
#                                                                   # (needs WANDB_API_KEY or `wandb login`)
#
# Running it again continues from the newest checkpoint in ./checkpoints, with the same ratings and wandb run.
# Stop it at any time with Ctrl+C (or `kill <pid>`): it finishes the current iteration, saves, and exits.
#
# The model and the games:
#   --model s             the small multi-mode network (537k parameters), the same one for both modes
#   --modes 1v1,2v2       two 1v1 games for every 2v2 game, so both modes get the same number of timesteps
#   --mirror left         a car on the right half of the field sees the game mirrored (and its action is mirrored
#                         back), so the network only ever sees its own car on the left half and learns every
#                         situation once instead of twice. A checkpoint has to be continued with the view it has
#
# Reward phase-out (stage 2): KickoffProximity, BehindBall and FaceBall are off (0); the speed-towards-ball shaping is halved
# (VelocityPlayerToBall 1->0.5, NearestVelocityToBall 3->1.5). Goal, VelocityBallToGoal, TouchAccel, TeamSpacing stay.
# The learning settings are those of train_1v1.sh (each one the result of a test judged by the 1v1 rating).
# Two rewards are different, because with a teammate the 1v1 versions pay both players for the same thing:
#   KickoffProximity=3    for the whole team, by its nearest player: one player going for the kickoff is enough
#   VelocityPlayerToBall=1,NearestVelocityToBall=3
#                         speed towards the ball: 4 for the team's nearest player as before, 1 for its teammate,
#                         which leaves the teammate free to take a position instead of chasing the ball too.
#                         In 1v1 both are exactly the rewards of train_1v1.sh
#
# The rating (see "How the rating works" in ELO_REPORT.md):
#   --skill-minutes 10, --skill-rounds 10, --skill-arenas 48
#                         one session every 10 minutes: 10 rounds of 32 1v1 games and 16 2v2 games at once against
#                         a saved older version. It gives the metrics "Rating/1v1" and "Rating/2v2"
#   --ts-per-version 150m, --max-versions 8   versions 150M steps apart, so the opponents differ (25M apart they were the same bot)
#   --train-vs-old 0.15                        15% of the iterations train against an old version
#
# Checkpoints:
#   --ts-per-save 5m, --keep-checkpoints 2
#                         a checkpoint every 10 million timesteps and after every rating session, the newest 2 are
#                         kept. With the 10 policy versions that is about 30 MB in ./checkpoints
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
	--checkpoints checkpoints_1v1_2v2 \
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
