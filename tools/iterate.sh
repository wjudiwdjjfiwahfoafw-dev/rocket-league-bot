#!/usr/bin/env bash
# One short experiment: continue from a checkpoint for a fixed WALLCLOCK budget, then rate the result
# against the frozen opponent pool (eval/opponents). Appends one row to results/iterations.tsv.
#
# Usage:   tools/iterate.sh <name> <minutes> [extra GigaLearnBot options]
# Example: tools/iterate.sh control 12
#          tools/iterate.sh batch12k 12 --batch-size 12000
# Env:     BASE (default checkpoints) = checkpoint folder to start from (copied, never modified)
#          RATE_ROUNDS (default 60) = rating rounds for the final score
#
# The score that is optimized: Rating/Ranking = mean of the 1v1 and 2v2 ratings (see ELO_REPORT / PolicyVersionManager).
# A 12-minute run moves the rating by a few points, and one rating eval with 60 rounds is about +-12,
# so compare rows only with the control run and treat small gaps as noise.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?name}"; MIN="${2:?minutes}"; shift 2
BASE="${BASE:-checkpoints_1v1_2v2}"
RATE_ROUNDS="${RATE_ROUNDS:-60}"
EXP="experiments/$NAME"
BOT="build/GigaLearnBot"

rm -rf "$EXP"; mkdir -p "$EXP"
cp -r "$BASE" "$EXP/ckpt"
START_TS=$(ls "$EXP/ckpt" | grep -E '^[0-9]+$' | sort -n | tail -1)

export OMP_WAIT_POLICY=passive
export GGL_METRICS_DIR="$EXP/metrics" GGL_METRICS_EVERY=20
T0=$(date +%s)
"$BOT" --model s --modes 1v1,2v2 --mirror left --checkpoints "$EXP/ckpt" --device cpu --threads 2 \
	--games 44 --ts-per-itr 24k --batch-size 6000 --epochs 1 --lr 1e-4 --entropy-scale 0.009 \
	--random-state-prob 0.5 --random-ball-speed 2000 \
	--rewards VelocityBallToGoal=8,Goal=300,KickoffProximity=3,VelocityPlayerToBall=1,NearestVelocityToBall=3,BehindBall=0.5,TeamSpacing=1,TouchAccel=15 \
	--train-vs-old 0.15 --ts-per-save 10m --keep-checkpoints 2 \
	--skill-tracker --skill-minutes 10 --skill-rounds 10 --skill-arenas 48 \
	--ts-per-version 25m --max-versions 10 \
	--max-minutes "$MIN" "$@" > "$EXP/train.log" 2>&1
T1=$(date +%s)

END_TS=$(ls "$EXP/ckpt" | grep -E '^[0-9]+$' | sort -n | tail -1)
tools/compare_checkpoints.sh "$EXP/ckpt" "$RATE_ROUNDS" "$EXP/rating.json" > "$EXP/rate.log" 2>&1 || true

python3 - "$EXP/rating.json" "$NAME" "$MIN" "$START_TS" "$END_TS" "$((T1-T0))" "$*" <<'PY'
import json, sys, os
path, name, minutes, s, e, secs, args = sys.argv[1:]
r = json.load(open(path))["ratings"]
v1, v2 = r["1v1"]["average"], r["2v2"]["average"]
rank = (v1 + v2) / 2
row = [name, minutes, s, e, f"{(int(e)-int(s))/max(int(secs),1):.0f}", f"{rank:.1f}", f"{v1:.1f}", f"{v2:.1f}", args or "-"]
hdr = not os.path.exists("results/iterations.tsv")
with open("results/iterations.tsv", "a") as f:
    if hdr: f.write("name\tminutes\tstart_ts\tend_ts\tsteps_per_s\tranking\t1v1\t2v2\targs\n")
    f.write("\t".join(row) + "\n")
print("\t".join(row))
PY
