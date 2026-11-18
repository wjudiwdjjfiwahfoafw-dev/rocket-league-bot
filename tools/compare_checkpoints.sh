#!/usr/bin/env bash
# Rates a checkpoint against a FROZEN pool of opponents, so that ratings from different
# runs and different days are on the same scale and can be compared.
#
# Usage:   tools/compare_checkpoints.sh <checkpoint folder> [rounds] [out.json]
# Example: tools/compare_checkpoints.sh checkpoints 60 results/eval/compare_latest.json
#
# The opponents are eval/opponents/policy_versions (the policy versions of the 1v1+2v2 run at
# 1,292.9M timesteps). Rating games are sampled actions, 45 s each, 16 at once: 40 rounds take about 35 s.
# Note: each goal moves the rating by at most 5 points, so a difference below ~20 points (1v1)
# is within noise. Use more rounds (e.g. 150) before trusting a small difference.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:?checkpoint folder}"
ROUNDS="${2:-60}"
OUT="${3:-}"
OPP="eval/opponents/policy_versions"
BOT="${BUILD_DIR:-build}/GigaLearnBot"

ARGS=(--model s --modes 1v1,2v2 --mirror left --device cpu --threads 2
      --checkpoints "$CKPT" --rate-rounds "$ROUNDS" --rate-versions "$OPP")
if [ -n "$OUT" ]; then ARGS+=(--rate-out "$OUT"); fi

"$BOT" "${ARGS[@]}" 2>&1 | grep -E "Rating/|Goals:|Rating of the checkpoint|FATAL|ERROR" || true
