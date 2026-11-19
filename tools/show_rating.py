#!/usr/bin/env python3
"""Shows the ratings of GigaLearn's skill tracker.

For training runs: reads the metrics .jsonl files that the metrics receiver writes (GGL_METRICS_DIR) and prints, per run,
the "Rating/<mode>" at regular timestep intervals and after every rating session (--skill-minutes), together with the
run time and the goals of the rating games.

    tools/show_rating.py RUN_METRICS_DIR_OR_FILE [...] [--mode 1v1] [--every 1000000] [--last 2000000] [--plot rating.png]

For rated checkpoints: reads the .json files of "GigaLearnBot --rate-rounds N --rate-out FILE" and prints them as one table.
Compare checkpoints evaluated with the same frozen pool, modes, fixtures and round count.
Use --mode Ranking for the equal-weight mean over the evaluated modes.
Pass a train_ranked.py campaign directory for its stage history and the selected checkpoint.

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

    def endpoint(rating):
        """Saved results keep only the current endpoint Elo, never the rating series or averages."""
        return float(rating["end"] if isinstance(rating, dict) else rating)

    print(f"{'checkpoint':<28} {'timesteps':>11} {'elo':>7} {'+-':>4} {'goals':>11} {'share':>6} {'rounds':>6}")
    for name, j, r in sorted(results, key=lambda x: -endpoint(x[2])):
        per_mode = j.get("mode_goals", {}).get(mode)
        goals_for, goals_against = (per_mode["for"], per_mode["against"]) if per_mode else (j["goals_for"], j["goals_against"])
        goals = max(goals_for + goals_against, 1)
        share = goals_for / goals

        # Rule of thumb for how far a repeat of the session lands from this one (1 standard deviation).
        # Measured with repeated sessions of about 1,350 goals: about 15 points. It shrinks with the square root of the goals.
        # This heuristic has not been calibrated for fixed opponent ratings.
        error = "-" if j.get("anchored") else f"{550 / math.sqrt(goals):.0f}"

        print(f"{name:<28} {j['checkpoint_timesteps']:>11,} {endpoint(r):>7.1f} {error:>4} "
              f"{goals_for:>5.0f}:{goals_against:<5.0f} {100 * share:>5.1f}% {j['rounds']:>6}")
    print("\nelo: current (endpoint) \"Rating/" + mode + "\" after the final rated round.\n"
          "+-: legacy tracker heuristic (not calibrated for anchored matches).\n"
          "share: fraction of goals scored by the checkpoint, per mode if available, otherwise across modes.")


def show_campaign(path):
    with open(os.path.join(path, "state.json")) as f:
        state = json.load(f)
    config = state["config"]
    print(f"{path}: {state['status']} | {sum(config['minutes'])} min scheduled")
    print(f"{'stage':>5} {'checkpoint':>13} {'train min':>9} {'steps/s':>8} | {'1v1':>7} {'2v2':>7} {'3v3':>7} | {'screen':>7} {'confirm':>7}  selection")
    for r in state["history"]:
        screen = r["screen"]
        confirm = f"{r['confirm']['Ranking']:.1f}" if r['confirm'] else "-"
        speed = f"{r['steps_per_second']:,.0f}" if r['steps_per_second'] else "-"
        decision = 'parent' if r['stage'] == 0 else ('promoted' if r['promoted'] else 'kept incumbent')
        print(f"{r['stage']:>5} {r['steps']:>13,} {r['train_seconds']/60:>9.1f} {speed:>8} | "
              f"{screen['1v1']:>7.1f} {screen['2v2']:>7.1f} {screen['3v3']:>7.1f} | "
              f"{screen['Ranking']:>7.1f} {confirm:>7}  {decision}")
    print(f"Screen seed: {config['screen_seed']}; confirmation: {config['confirm_seed']}. "
          "Each Ranking is the mean of all three mode Elos.")
    if state.get('best'):
        print(f"Selected: {os.path.join(path, 'best', str(state['best']['steps']))}")
    if state.get('pending'):
        print(f"Pending stage {state['pending']['stage']}: {state['pending']['seconds']/60:.1f} training minutes recorded.")
    if state.get('audit'):
        audit = state['audit']
        baseline = audit[str(config['parent_steps'])]['Ranking']
        roles = {'parent': config['parent_steps'], 'selected': state['best']['steps'], 'latest': state['history'][-1]['steps']}
        print(f"\nHeld-out seed {config['audit_seed']} (not used for selection)")
        print(f"{'checkpoint':>13} {'1v1':>7} {'2v2':>7} {'3v3':>7} {'Ranking':>8} {'vs parent':>10}  role")
        for step, r in audit.items():
            role = ', '.join(name for name, n in roles.items() if n == int(step))
            print(f"{int(step):>13,} {r['1v1']:>7.1f} {r['2v2']:>7.1f} {r['3v3']:>7.1f} "
                  f"{r['Ranking']:>8.1f} {r['Ranking']-baseline:>+10.1f}  {role}")


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

    campaigns = [p for p in args.paths if os.path.isfile(os.path.join(p, "state.json"))]
    for p in campaigns:
        show_campaign(p)
    args.paths = [p for p in args.paths if p not in campaigns]
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
