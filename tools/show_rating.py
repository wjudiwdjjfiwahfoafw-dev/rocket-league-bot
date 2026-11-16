#!/usr/bin/env python3
"""Shows the ratings of GigaLearn's skill tracker.

For training runs: reads the metrics .jsonl files that the metrics receiver writes (GGL_METRICS_DIR) and prints, per run,
the "Rating/<mode>" at regular timestep intervals and after every rating session (--skill-minutes), together with the
run time and the goals of the rating games.

    tools/show_rating.py RUN_METRICS_DIR_OR_FILE [...] [--mode 1v1] [--every 1000000] [--last 2000000] [--plot rating.png]

For rated checkpoints: reads the .json files of "GigaLearnBot --rate-rounds N --rate-out FILE" and prints them as one table.
Checkpoints that were rated against the same policy versions (--rate-versions) are on the same scale.

    tools/show_rating.py RATING.json [...] [--mode 1v1]
"""
import argparse
import glob
import json
import math
import os


def load_rows(path):
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "*.jsonl")))
    rows = []
    for file in files:
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # A line that was cut off when the run was stopped
    rows.sort(key=lambda r: r.get("Total Timesteps", 0))
    return rows


def add_run_hours(rows):
    """Adds "_hours" to every row: the run time up to it.
    Gaps of more than 2 minutes between two iterations (the run was stopped) don't count, unless a rating session filled them"""
    total = 0.0
    prev = None
    for r in rows:
        t = r.get("_time")
        if t is not None and prev is not None and t > prev:
            gap = t - prev
            if gap < 120 or (r.get("Skill Matches/Rounds", 0) > 1 and gap < 1800):
                total += gap
        if t is not None:
            prev = t
        r["_hours"] = total / 3600


def is_session(row):
    """A rating session (--skill-minutes): many rounds of rating games in a row, its rating is already an average"""
    return row.get("Skill Matches/Rounds", 1) > 1


def run_name(path):
    path = os.path.normpath(path)
    name = os.path.basename(path)
    if name == "metrics" or name.endswith(".jsonl"):
        name = os.path.basename(os.path.dirname(path))
    return name


def show_rated(paths, mode):
    """Table of the results of --rate-rounds"""
    results = []
    for path in paths:
        with open(path) as f:
            j = json.load(f)
        if mode not in j.get("ratings", {}):
            print(f"{path}: no rating for {mode}")
            continue
        name = os.path.basename(path)[:-len(".json")]
        results.append((name, j, j["ratings"][mode]))

    print(f"{'checkpoint':<28} {'timesteps':>11} {'rating':>7} {'+-':>4} {'goals':>11} {'share':>6} {'rounds':>6} {'start':>7} {'end':>7}")
    for name, j, r in sorted(results, key=lambda x: -x[2]["average"]):
        goals_for, goals_against = j["goals_for"], j["goals_against"]
        goals = max(goals_for + goals_against, 1)
        share = goals_for / goals

        # Rule of thumb for how far a repeat of the session lands from this one (1 standard deviation).
        # Measured with repeated sessions of about 1,350 goals: about 15 points. It shrinks with the square root of the goals.
        error = 550 / math.sqrt(goals)

        print(f"{name:<28} {j['checkpoint_timesteps']:>11,} {r['average']:>7.1f} {error:>4.0f} "
              f"{goals_for:>5.0f}:{goals_against:<5.0f} {100 * share:>5.1f}% {j['rounds']:>6} {r['start']:>7.1f} {r['end']:>7.1f}")
    print("\nrating: average of \"Rating/" + mode + "\" over the last three quarters of the rounds.\n"
          "+-: how far a repeat of the session typically lands from it (about 550 / sqrt(goals) points).\n"
          "share: goals of the checkpoint out of all goals against the versions it played.")


def plot_runs(runs, mode, out_path, marks=(), hours=False):
    """Rating over the timesteps (or the run time) of each run (needs matplotlib).
    Single rounds of rating games are drawn as small dots with a running average as the line, rating sessions as bigger dots
    (the biggest for sessions of 30 rounds or more)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Okabe-Ito palette, which stays distinguishable with color vision deficiencies
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=110)
    first_line = []
    for i, (name, rated) in enumerate(runs):
        color = colors[i % len(colors)]
        x_of = (lambda r: r["_hours"]) if hours else (lambda r: r["Total Timesteps"] / 1e6)
        rounds = [r for r in rated if not is_session(r)]
        sessions = [r for r in rated if is_session(r)]
        line_x, line_y = [], []
        if rounds:
            steps = [x_of(r) for r in rounds]
            ratings = [r["Rating/" + mode] for r in rounds]
            window = max(1, min(15, len(ratings) // 8))
            smooth = [sum(ratings[max(0, j - window + 1):j + 1]) / len(ratings[max(0, j - window + 1):j + 1]) for j in range(len(ratings))]
            ax.plot(steps, ratings, ".", color=color, markersize=3, alpha=0.35)
            line_x, line_y = steps, smooth
        if sessions:
            # The line goes on through the sessions. Long sessions (30 rounds or more) are exact enough to stand for
            # themselves. Short ones scatter by about 17 points each, so the line takes the average of each one and
            # its two neighbours.
            values = [r["Rating/" + mode] for r in sessions]
            is_long = [r.get("Skill Matches/Rounds", 0) >= 30 for r in sessions]
            for j, r in enumerate(sessions):
                window = [values[j]]
                if not is_long[j]:
                    window += [values[k] for k in (j - 1, j + 1) if 0 <= k < len(sessions) and not is_long[k]]
                line_x.append(x_of(r))
                line_y.append(sum(window) / len(window))
            for long_ones, size in ((True, 6), (False, 3.5)):
                points = [(x_of(r), v) for r, v, l in zip(sessions, values, is_long) if l == long_ones]
                if points:
                    ax.plot([x for x, _ in points], [y for _, y in points], "o", color=color, markersize=size, alpha=1.0 if long_ones else 0.6)
        ax.plot(line_x, line_y, "-", color=color, linewidth=2, label=name)
        if i == 0:
            first_line = list(zip(line_x, line_y))
    y_low, y_high = ax.get_ylim()
    for mark in marks:
        # "42.9=text": a vertical line at 42.9 million timesteps with a label
        position, _, text = mark.partition("=")
        ax.axvline(float(position), color="#555555", linestyle="--", linewidth=1)
        # The label runs along the left side of its line, so that labels of nearby lines don't overlap.
        # It starts at the top of the plot, or at the bottom where the curve is in the upper half
        curve_y = min(first_line, key=lambda point: abs(point[0] - float(position)))[1] if first_line else y_low
        at_top = curve_y < (y_low + y_high) / 2
        ax.annotate(text, xy=(float(position), 0.98 if at_top else 0.02), xycoords=("data", "axes fraction"), xytext=(-4, 0),
                    textcoords="offset points", fontsize=9, color="#333333", rotation=90, ha="right", va="top" if at_top else "bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.set_xlabel("Run time (hours)" if hours else "Timesteps (millions)")
    ax.set_ylabel(f"Rating/{mode} (GigaLearn skill tracker)")
    ax.set_title(f"Rating/{mode} during training (dots: single rounds or rating sessions, line: their running average)")
    ax.grid(True, alpha=0.3)
    if len(runs) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Plot written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--mode", default="1v1")
    parser.add_argument("--every", type=float, default=1e6, help="timesteps between the printed readings")
    parser.add_argument("--last", type=float, default=2e6, help="timesteps at the end of the run that the average covers")
    parser.add_argument("--plot", default="", help="also draw the rating curves of the runs into this image file")
    parser.add_argument("--mark", action="append", default=[], metavar="MILLIONS=TEXT",
                        help="plot: a labelled vertical line at this many million timesteps (can be given several times)")
    parser.add_argument("--hours", action="store_true", help="plot: run time in hours as the x-axis instead of timesteps (marks are then in hours too)")
    args = parser.parse_args()

    rated = [p for p in args.paths if p.endswith(".json")]
    if rated:
        show_rated(rated, args.mode)
        args.paths = [p for p in args.paths if not p.endswith(".json")]
        if not args.paths:
            return

    key = "Rating/" + args.mode
    summary = []
    plot_data = []
    for path in args.paths:
        rows = load_rows(path)
        add_run_hours(rows)
        rated = [r for r in rows if key in r]
        if not rated:
            print(f"{run_name(path)}: no \"{key}\" in the metrics (was --skill-tracker on?)")
            continue

        plot_data.append((run_name(path), rated))
        first_ts, last_ts = rows[0]["Total Timesteps"], rows[-1]["Total Timesteps"]
        num_sessions = sum(1 for r in rated if is_session(r))
        print(f"\n{run_name(path)}: {first_ts:,.0f} -> {last_ts:,.0f} timesteps in {rows[-1]['_hours']:.2f} hours of run time, "
              f"{len(rated) - num_sessions} single rounds of rating games and {num_sessions} rating sessions")
        print(f"  {'timesteps':>12} {'hours':>6} {'rating':>8} {'goals for:against':>18} {'versions':>9}")
        next_ts = first_ts
        goals_for = goals_against = 0
        for i, r in enumerate(rated):
            goals_for += r.get("Skill Matches/Goals For", 0)
            goals_against += r.get("Skill Matches/Goals Against", 0)
            # Single rounds every --every timesteps, and every rating session
            if r["Total Timesteps"] >= next_ts or i == len(rated) - 1 or is_session(r):
                print(f"  {r['Total Timesteps']:>12,.0f} {r['_hours']:>6.2f} {r[key]:>8.1f} {goals_for:>9.0f}:{goals_against:<8.0f} "
                      f"{r.get('Skill Matches/Versions', float('nan')):>9.0f}{'  session' if is_session(r) else ''}")
                goals_for = goals_against = 0
                next_ts = r["Total Timesteps"] + args.every

        # (With rating sessions there may be no reading in the last timesteps of the run: then the last one counts)
        tail = [r for r in rated if r["Total Timesteps"] >= last_ts - args.last] or rated[-1:]
        total_for = sum(r.get("Skill Matches/Goals For", 0) for r in rated)
        total_against = sum(r.get("Skill Matches/Goals Against", 0) for r in rated)
        summary.append((run_name(path), last_ts - first_ts, rated[0][key], rated[-1][key],
                        sum(r[key] for r in tail) / len(tail), max(r[key] for r in rated), total_for, total_against))

    if summary:
        print(f"\n{'run':<28} {'timesteps':>11} {'first':>7} {'final':>7} {'avg of end':>10} {'best':>7} {'goals':>11}")
        for name, steps, first, final, avg, best, gf, ga in sorted(summary, key=lambda s: -s[4]):
            print(f"{name:<28} {steps:>11,.0f} {first:>7.1f} {final:>7.1f} {avg:>10.1f} {best:>7.1f} {gf:>5.0f}:{ga:<5.0f}")

    if args.plot and plot_data:
        plot_runs(plot_data, args.mode, args.plot, args.mark, args.hours)


if __name__ == "__main__":
    main()
