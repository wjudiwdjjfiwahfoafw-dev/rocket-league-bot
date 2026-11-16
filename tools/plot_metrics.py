#!/usr/bin/env python3
"""Plots training curves from a metrics .jsonl file written by python_scripts/metric_receiver.py.

Usage: python3 tools/plot_metrics.py [metrics folder or .jsonl file] [output.png]

Needs matplotlib (pip install matplotlib).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS_PER_SECOND = 15  # 120 ticks per second / tick skip of 8

LINE_COLOR = "#2563eb"
RAW_COLOR = "#93c5fd"
SECOND_COLOR = "#d97706"
SECOND_RAW_COLOR = "#fcd34d"

# One color per game mode (from the Okabe-Ito palette, which stays distinguishable with color vision deficiencies)
MODES = ["1v1", "2v2", "3v3"]
MODE_COLORS = {"1v1": "#0072B2", "2v2": "#D55E00", "3v3": "#009E73"}


def load_rows(path):
	if os.path.isdir(path):
		files = sorted(glob.glob(os.path.join(path, "*.jsonl")), key = os.path.getmtime)
		if not files:
			sys.exit(f"No .jsonl files in {path}")
		path = files[-1]
	with open(path) as f:
		rows = [json.loads(line) for line in f if line.strip()]
	if not rows:
		sys.exit(f"{path} has no metrics yet")
	return rows, path


def series(rows, key, scale = 1.0):
	"""Returns (timesteps in millions, values) of a metric, skipping iterations where it is missing."""
	xs, ys = [], []
	for row in rows:
		val = row.get(key)
		if val is None:
			continue
		xs.append(row["Total Timesteps"] / 1e6)
		ys.append(val * scale)
	return xs, ys


def smooth(ys, window):
	"""Trailing moving average."""
	result, total = [], 0.0
	for i, y in enumerate(ys):
		total += y
		if i >= window:
			total -= ys[i - window]
		result.append(total / min(i + 1, window))
	return result


def plot_metric(ax, rows, key, title, ylabel, scale = 1.0, window = 15, color = LINE_COLOR, raw_color = RAW_COLOR, label = None):
	xs, ys = series(rows, key, scale)
	if not xs:
		ax.set_title(f"{title} (no data)")
		return
	ax.plot(xs, ys, color = raw_color, linewidth = 0.8, alpha = 0.7)
	ax.plot(xs, smooth(ys, window), color = color, linewidth = 2, label = label)
	ax.set_title(title, fontsize = 11, loc = "left")
	ax.set_ylabel(ylabel, fontsize = 9)


def plot_modes(ax, rows, name, title, ylabel, scale = 1.0, window = 15):
	"""One smoothed line per game mode, for the metrics that training reports as "Mode 1v1/<name>" etc."""
	for mode in MODES:
		xs, ys = series(rows, f"Mode {mode}/{name}", scale)
		if xs:
			ax.plot(xs, smooth(ys, window), color = MODE_COLORS[mode], linewidth = 2, label = mode)
	ax.set_title(title, fontsize = 11, loc = "left")
	ax.set_ylabel(ylabel, fontsize = 9)
	ax.legend(fontsize = 9, frameon = False)


def main():
	path = sys.argv[1] if len(sys.argv) > 1 else "metrics"
	out_path = sys.argv[2] if len(sys.argv) > 2 else "training_curves.png"
	rows, path = load_rows(path)

	total_steps = rows[-1]["Total Timesteps"]
	window = max(3, len(rows) // 40)

	# Runs with several game modes report some metrics per mode as well
	modes = [mode for mode in MODES if any(f"Mode {mode}/Ball Touch Ratio" in row for row in rows)]

	fig, axes = plt.subplots(3, 3, figsize = (16, 11))
	if modes:
		fig.suptitle(
			f"GigaLearnCPP PPO training, {', '.join(modes)}: {total_steps / 1e6:.1f}M timesteps, {len(rows)} iterations\n"
			f"(lines: average of the last {window} iterations. Light line: every iteration)",
			fontsize = 13
		)

		plot_modes(axes[0][0], rows, "Step Reward", "Average reward per step, by game mode", "reward", window = window)
		plot_modes(axes[0][1], rows, "Ball Touch Ratio", "Ball touches per player, by game mode", "touches per minute of game time", scale = STEPS_PER_SECOND * 60, window = window)
		plot_modes(axes[0][2], rows, "Goal Ratio", "Goals, by game mode", "goals per hour of game time", scale = STEPS_PER_SECOND * 3600, window = window)
		plot_modes(axes[1][0], rows, "Ball Distance", "Average distance to the ball, by game mode", "unreal units (field length = 10240)", window = window)
	else:
		fig.suptitle(
			f"GigaLearnCPP PPO training, 1v1: {total_steps / 1e6:.1f}M timesteps, {len(rows)} iterations\n"
			f"(light line: every iteration, dark line: average of the last {window} iterations)",
			fontsize = 13
		)

		ax = axes[0][0]
		plot_metric(ax, rows, "Average Step Reward", "Average reward per step", "reward", window = window)

		ax = axes[0][1]
		plot_metric(ax, rows, "Player/Ball Touch Ratio", "Ball touches per player", "touches per minute of game time", scale = STEPS_PER_SECOND * 60, window = window)

		ax = axes[0][2]
		plot_metric(ax, rows, "Game/Goal Ratio", "Goals", "goals per hour of game time", scale = STEPS_PER_SECOND * 3600, window = window)

		ax = axes[1][0]
		plot_metric(ax, rows, "Player/Ball Distance", "Average distance to the ball", "unreal units (field length = 10240)", window = window)

	ax = axes[1][1]
	plot_metric(ax, rows, "Player/Speed", "Car speed", "unreal units per second", window = window, label = "speed")
	plot_metric(ax, rows, "Player/Speed Towards Ball", "Car speed", "unreal units per second", window = window, color = SECOND_COLOR, raw_color = SECOND_RAW_COLOR, label = "speed towards the ball")
	ax.legend(fontsize = 9, frameon = False)

	ax = axes[1][2]
	plot_metric(ax, rows, "Player/In Air Ratio", "Time in the air", "% of steps", scale = 100, window = window)

	ax = axes[2][0]
	plot_metric(ax, rows, "Policy Entropy", "Policy entropy (normalized)", "1 = uniformly random actions", window = window)

	ax = axes[2][1]
	plot_metric(ax, rows, "Mean KL Divergence", "Policy change per iteration", "mean KL divergence", window = window)

	ax = axes[2][2]
	plot_metric(ax, rows, "Overall Steps/Second", "Training speed", "timesteps per second", window = window)

	for row_axes in axes:
		for ax in row_axes:
			ax.grid(True, color = "#e5e7eb", linewidth = 0.8)
			ax.set_axisbelow(True)
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
			ax.tick_params(labelsize = 9)
			ax.set_xlim(left = 0)
	for ax in axes[2]:
		ax.set_xlabel("total timesteps (millions)", fontsize = 9)

	fig.tight_layout(rect = (0, 0, 1, 0.94))
	fig.savefig(out_path, dpi = 110)
	print(f"Wrote {out_path} ({len(rows)} iterations from {path})")


if __name__ == "__main__":
	main()
