#!/usr/bin/env python3
"""
Measures the training speed (steps per second) of GigaLearnBot and shows where the time goes.

It copies a checkpoint to a temporary folder, trains from it for a number of iterations and reads the
timing metrics of every iteration. The first iteration is left out (warm-up). Nothing in ./checkpoints is changed.

Usage:
    python3 tools/benchmark.py [options] [-- extra GigaLearnBot options]

Examples:
    python3 tools/benchmark.py                                   # the legacy CPU benchmark preset (not the ranked training recipe)
    python3 tools/benchmark.py -- --epochs 1                     # the same with one PPO epoch
    python3 tools/benchmark.py --no-preset -- --device cuda      # the program defaults (the original example) on a GPU
    python3 tools/benchmark.py --fresh --no-preset -- --model l --device cpu --games 44 --ts-per-itr 24k --batch-size 1500
                                                                 # a fresh large three-mode model
    python3 tools/benchmark.py --env OMP_NUM_THREADS=1 --repeat 3

Options:
    --bot PATH          the executable                                        (default: build/GigaLearnBot)
    --checkpoints DIR   start from the newest checkpoint in this folder       (default: checkpoints)
    --fresh             start from an untrained network instead
    --iterations N      measured iterations, after one warm-up iteration      (default: 10)
    --repeat N          run the whole measurement N times                     (default: 1)
    --no-preset         don't add the legacy CPU benchmark preset (not the ranked training recipe)
    --env NAME=VALUE    extra environment variable for the run (can be repeated)
    --label TEXT        name shown in the output
    --json FILE         append the result as one JSON line to FILE
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

# Legacy CPU benchmark settings; use explicit --no-preset arguments for the ranked recipe
CPU_PRESET = ["--device", "cpu", "--games", "64", "--ts-per-itr", "24k", "--batch-size", "6k", "--epochs", "2",
              "--lr", "2e-4", "--random-state-prob", "0.5"]
CPU_PRESET_ENV = {"OMP_WAIT_POLICY": "passive"}

PHASES = [  # (name shown, how to get the seconds from one metrics row)
    ("env step", lambda r: r["Env Step Time"]),
    ("inference", lambda r: r["Inference Time"]),
    ("other collection", lambda r: r["Collection Time"] - r["Env Step Time"] - r["Inference Time"]),
    ("value prediction + GAE", lambda r: r["Consumption Time"] - r["PPO Learn Time"]),
    ("PPO learn", lambda r: r["PPO Learn Time"]),
]


def parse_num(text):
    text = text.lower()
    scale = {"k": 1e3, "m": 1e6, "b": 1e9}.get(text[-1])
    return int(float(text[:-1]) * scale) if scale else int(float(text))


def get_option(args, name, default):
    """Value of the last occurrence of `name` in a GigaLearnBot argument list"""
    value = default
    for i, arg in enumerate(args[:-1]):
        if arg == name:
            value = args[i + 1]
    return value


def newest_checkpoint(folder):
    names = [n for n in os.listdir(folder) if n.isdigit() and os.path.isdir(os.path.join(folder, n))]
    if not names:
        raise SystemExit(f"No checkpoint found in {folder} (use --fresh to start from an untrained network)")
    return max(names, key=int)


def run_once(opt, bot_args, env):
    work_dir = tempfile.mkdtemp(prefix="ggl_bench_")
    try:
        ckpt_dir = os.path.join(work_dir, "checkpoints")
        os.makedirs(ckpt_dir)
        start_steps = 0
        if not opt["fresh"]:
            name = newest_checkpoint(opt["checkpoints"])
            shutil.copytree(os.path.join(opt["checkpoints"], name), os.path.join(ckpt_dir, name))
            start_steps = int(name)

        metrics_dir = os.path.join(work_dir, "metrics")
        run_env = dict(os.environ)
        run_env.update(env)
        run_env["GGL_METRICS_DIR"] = metrics_dir
        run_env.pop("GGL_WANDB", None)

        ts_per_itr = parse_num(get_option(bot_args, "--ts-per-itr", "50000"))
        # One warm-up iteration, the measured iterations, and a little reserve (iterations are a bit larger than ts_per_itr)
        max_steps = start_steps + int(ts_per_itr * (opt["iterations"] + 1) * 1.3) + 1
        cmd = [opt["bot"]] + bot_args + ["--checkpoints", ckpt_dir, "--max-timesteps", str(max_steps),
                                         "--ts-per-save", "1b", "--keep-checkpoints", "1"]
        start = time.time()
        proc = subprocess.run(cmd, env=run_env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        wall = time.time() - start
        if proc.returncode != 0:
            print(proc.stdout[-3000:])
            raise SystemExit(f"GigaLearnBot failed with exit code {proc.returncode}")

        rows = []
        for name in sorted(os.listdir(metrics_dir)):
            if name.endswith(".jsonl"):
                with open(os.path.join(metrics_dir, name)) as f:
                    rows += [json.loads(line) for line in f if line.strip()]
        if len(rows) < 3:
            print(proc.stdout[-3000:])
            raise SystemExit("Too few iterations were completed")

        measured = rows[1:1 + opt["iterations"]]
        steps = sum(r["Collected Timesteps"] for r in measured)
        # Timesteps that were given to the learner (those of finished episodes). Older builds don't report this.
        # With many games this is lower than the collected timesteps for the first dozens of iterations of a run,
        # because the steps of unfinished episodes pile up until their number levels off.
        learned = sum(r.get("Experience Timesteps", r["Collected Timesteps"]) for r in measured)
        collection = sum(r["Collection Time"] for r in measured)
        consumption = sum(r["Consumption Time"] for r in measured)
        # Time between the reports of two iterations: also contains the metrics/report overhead between iterations
        wall_between = rows[len(measured)]["_time"] - rows[0]["_time"]
        result = {
            "iterations": len(measured),
            "steps_per_iteration": steps / len(measured),
            "overall_sps": steps / (collection + consumption),
            "learned_sps": learned / (collection + consumption),
            "wall_sps": steps / wall_between,
            "collection_sps": steps / collection,
            "consumption_sps": steps / consumption,
            "overall_sps_stdev": statistics.pstdev([r["Overall Steps/Second"] for r in measured]),
            "phases_ms_per_1k_steps": {name: 1e6 * sum(fn(r) for r in measured) / steps for name, fn in PHASES},
            "process_seconds": wall,
        }
        return result
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    argv = sys.argv[1:]
    bot_args = []
    if "--" in argv:
        split = argv.index("--")
        argv, bot_args = argv[:split], argv[split + 1:]

    opt = {"bot": "build/GigaLearnBot", "checkpoints": "checkpoints", "fresh": False, "iterations": 10, "repeat": 1,
           "preset": True, "label": "", "json": ""}
    env = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            print(__doc__)
            return
        elif arg == "--fresh":
            opt["fresh"] = True
        elif arg == "--no-preset":
            opt["preset"] = False
        elif arg == "--env":
            name, _, value = argv[i + 1].partition("=")
            env[name] = value
            i += 1
        elif arg in ("--bot", "--checkpoints", "--label", "--json"):
            opt[arg[2:]] = argv[i + 1]
            i += 1
        elif arg in ("--iterations", "--repeat"):
            opt[arg[2:]] = int(argv[i + 1])
            i += 1
        else:
            raise SystemExit(f"Unknown option {arg} (options for GigaLearnBot go after \"--\")")
        i += 1

    if not os.path.exists(opt["bot"]):
        raise SystemExit(f"{opt['bot']} does not exist. Run setup_linux.sh first.")

    if opt["preset"]:
        bot_args = CPU_PRESET + bot_args
        env = {**CPU_PRESET_ENV, **env}

    label = opt["label"] or (" ".join(bot_args[len(CPU_PRESET):] if opt["preset"] else bot_args) or "legacy CPU benchmark settings")
    results = []
    for run in range(opt["repeat"]):
        r = run_once(opt, bot_args, env)
        results.append(r)
        phases = "  ".join(f"{name} {ms:.1f}" for name, ms in r["phases_ms_per_1k_steps"].items())
        print(f"[{label}] run {run + 1}: {r['overall_sps']:8.0f} steps/s overall "
              f"(collection {r['collection_sps']:.0f}, consumption {r['consumption_sps']:.0f}, "
              f"with reporting {r['wall_sps']:.0f}), {r['steps_per_iteration']:.0f} steps/iteration, "
              f"given to the learner {r['learned_sps']:.0f} steps/s")
        print(f"    ms per 1000 steps: {phases}")
        sys.stdout.flush()

    if len(results) > 1:
        values = [r["overall_sps"] for r in results]
        print(f"[{label}] {len(values)} runs: mean {statistics.mean(values):.0f}, min {min(values):.0f}, "
              f"max {max(values):.0f} steps/s overall")

    if opt["json"]:
        with open(opt["json"], "a") as f:
            for r in results:
                f.write(json.dumps({"label": label, "args": bot_args, "env": env, **r}) + "\n")


if __name__ == "__main__":
    main()
