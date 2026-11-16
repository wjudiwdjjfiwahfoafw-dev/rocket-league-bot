#!/usr/bin/env python3
"""Prints a compact progress table from a metrics .jsonl file written by python_scripts/metric_receiver.py.

Usage: python3 tools/show_progress.py [metrics folder or .jsonl file] [number of rows]
"""
import glob
import json
import os
import sys

STEPS_PER_SECOND = 15  # 120 ticks per second / tick skip of 8


def load_rows(path):
	if os.path.isdir(path):
		files = sorted(glob.glob(os.path.join(path, "*.jsonl")), key = os.path.getmtime)
		if not files:
			sys.exit(f"No .jsonl files in {path}")
		path = files[-1]
	with open(path) as f:
		return [json.loads(line) for line in f if line.strip()], path


def main():
	path = sys.argv[1] if len(sys.argv) > 1 else "metrics"
	num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 12
	rows, path = load_rows(path)
	if not rows:
		sys.exit("No metrics yet")

	# Runs with several game modes also report the ball touches of each mode
	modes = [mode for mode in ("1v1", "2v2", "3v3") if any(f"Mode {mode}/Ball Touch Ratio" in row for row in rows)]

	print(f"{path}: {len(rows)} iterations")
	print(
		f"{'steps':>8} {'reward':>7} {'touch%':>7} {'->ball':>7} {'dist':>6} {'air%':>5} {'goals/h':>8} {'entropy':>8} {'KL':>8} {'clip%':>6} {'steps/s':>8}"
		+ "".join(f" {'touch% ' + mode:>10}" for mode in modes)
	)

	# Evenly spaced rows, always including the last one
	count = min(num_rows, len(rows))
	indices = sorted(set(round(i * (len(rows) - 1) / max(count - 1, 1)) for i in range(count)))
	for i in indices:
		r = rows[i]
		g = lambda key, scale = 1.0: (r.get(key) or 0) * scale
		print(
			f"{g('Total Timesteps') / 1e6:7.2f}M {g('Average Step Reward'):7.3f} {g('Player/Ball Touch Ratio', 100):7.3f} "
			f"{g('Player/Speed Towards Ball'):7.0f} {g('Player/Ball Distance'):6.0f} {g('Player/In Air Ratio', 100):5.1f} "
			f"{g('Game/Goal Ratio', STEPS_PER_SECOND * 3600):8.2f} {g('Policy Entropy'):8.4f} {g('Mean KL Divergence'):8.5f} "
			f"{g('SB3 Clip Fraction', 100):6.2f} {g('Overall Steps/Second'):8.0f}"
			+ "".join(f" {g(f'Mode {mode}/Ball Touch Ratio', 100):10.3f}" for mode in modes)
		)


if __name__ == "__main__":
	main()
