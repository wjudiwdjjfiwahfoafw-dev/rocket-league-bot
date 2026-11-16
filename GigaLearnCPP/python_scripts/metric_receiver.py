"""
Metrics receiver for GigaLearnCPP.

The C++ learner imports this module, calls init() once at startup, and then calls
add_metrics() once per iteration with a dict of {metric name: value}.

What this version does:
  * ALWAYS appends the reports to a local JSON-lines file:  <GGL_METRICS_DIR>/<run id>.jsonl
	(one JSON object per iteration, plus a wall-clock "_time" field).
	This needs no account, no network and no extra Python packages.
	With GGL_METRICS_EVERY=N only every Nth iteration is written, and every iteration that has a rating:
	a small model writes about 5 MB per hour otherwise.
  * ADDITIONALLY logs to Weights & Biases when the environment variable GGL_WANDB=1 is set
	(needs `pip install wandb`, and the API key in WANDB_API_KEY or a previous `wandb login`).

Environment variables (the bot executable sets some of them from its command-line options):
  GGL_METRICS_DIR     Folder for the .jsonl files (default: "metrics", relative to the working directory)
  GGL_METRICS_EVERY   Write only every Nth iteration to the .jsonl file (default: 1, every iteration).
					  Iterations with a rating ("Rating/...") are always written. wandb gets every iteration
  GGL_WANDB           Set to 1 to also log to wandb
  WANDB_ENTITY        wandb user or team (default: the default entity of the API key)
  GGL_RUN_CONFIG      JSON object with the settings of the run, shown as the run config in wandb
  GGL_NEW_RUN         Set to 1 to start a new run even if the checkpoint belongs to an earlier run
					  (for a run that branches off another run's checkpoint)
"""

import json
import math
import os
import site
import sys
import time

TIMESTEPS_KEY = "Total Timesteps"

wandb_run = None
jsonl_file = None
jsonl_path = None
jsonl_every = 1 # GGL_METRICS_EVERY
jsonl_count = 0


def _env_flag(name):
	return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _init_wandb(py_exec_path, project, group, name, id):
	global wandb_run

	# Fix the path of our interpreter so wandb doesn't run the bot executable instead of Python
	# Very strange fix for a very strange problem
	sys.executable = py_exec_path

	try:
		# Windows-style site-packages next to the interpreter (harmless if it does not exist)
		site_packages_dir = os.path.join(os.path.join(os.path.dirname(py_exec_path), "Lib"), "site-packages")
		if os.path.isdir(site_packages_dir):
			sys.path.append(site_packages_dir)
			site.addsitedir(site_packages_dir)
		import wandb
	except Exception as e:
		raise Exception(f"""
			FAILED to import wandb! Make sure GigaLearnCPP isn't using the wrong Python installation.
			This installation's site packages: {site.getsitepackages()}
			Exception: {repr(e)}"""
		)

	print("Calling wandb.init()...")
	kwargs = {
		"project": project,
		"group": group,
		"name": name,
	}
	if os.environ.get("WANDB_ENTITY"):
		kwargs["entity"] = os.environ["WANDB_ENTITY"]
	if not (id is None) and len(id) > 0:
		kwargs["id"] = id
		kwargs["resume"] = "allow"
		# A run that is continued keeps the name it has
		del kwargs["name"]

	wandb_run = wandb.init(**kwargs)

	# The settings of the run. A run that is continued with other settings (e.g. another batch size) shows the latest ones
	try:
		wandb_run.config.update(_get_run_config(), allow_val_change = True)
	except Exception as e:
		print(f"Metrics receiver: can't set the run config ({repr(e)})")

	# Plot everything over the timesteps trained, so that runs with different iteration sizes can be compared
	try:
		wandb_run.define_metric(TIMESTEPS_KEY)
		wandb_run.define_metric("*", step_metric = TIMESTEPS_KEY)
	except Exception as e:
		print(f"Metrics receiver: define_metric failed ({repr(e)}), wandb will plot over iterations")

	print(f"Metrics receiver: wandb run {wandb_run.url}", flush = True)
	return wandb_run.id


def _get_run_config():
	try:
		config = json.loads(os.environ.get("GGL_RUN_CONFIG", "") or "{}")
		return config if isinstance(config, dict) else {}
	except Exception as e:
		print(f"Metrics receiver: can't read GGL_RUN_CONFIG ({repr(e)})")
		return {}


# Takes in the python executable path, the three wandb init strings, and optionally the current run ID
# Returns the ID of the run (either newly created or resumed)
def init(py_exec_path, project, group, name, id = None):
	global jsonl_file, jsonl_path, jsonl_every

	run_id = id if (id is not None and len(id) > 0) else None

	if _env_flag("GGL_NEW_RUN"):
		run_id = None # This run branches off the checkpoint of another run, so it doesn't continue that run's metrics

	if _env_flag("GGL_WANDB"):
		run_id = _init_wandb(py_exec_path, project, group, name, run_id)

	if run_id is None:
		# Not resuming and no wandb ID, so make our own
		run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid())

	metrics_dir = os.environ.get("GGL_METRICS_DIR", "metrics")
	try:
		jsonl_every = max(1, int(os.environ.get("GGL_METRICS_EVERY", "1")))
	except ValueError:
		jsonl_every = 1
	os.makedirs(metrics_dir, exist_ok = True)
	jsonl_path = os.path.join(metrics_dir, run_id + ".jsonl")

	# Append mode, so that a resumed run (same ID) continues in the same file
	jsonl_file = open(jsonl_path, "a", buffering = 1)
	print(f"Metrics receiver: logging to \"{jsonl_path}\"" + (" and wandb" if wandb_run is not None else ""), flush = True)

	return run_id


def _to_json_value(v):
	try:
		f = float(v)
	except Exception:
		return str(v)

	if math.isnan(f) or math.isinf(f):
		return None # Not valid in JSON
	return f


def add_metrics(metrics):
	global wandb_run, jsonl_file

	if jsonl_file is not None:
		global jsonl_count
		jsonl_count += 1
		if (jsonl_count - 1) % jsonl_every == 0 or any(str(k).startswith("Rating/") for k in metrics):
			row = {str(k): _to_json_value(v) for k, v in metrics.items()}
			row["_time"] = time.time()
			jsonl_file.write(json.dumps(row, sort_keys = True) + "\n")
			jsonl_file.flush()

	if wandb_run is not None:
		wandb_run.log(metrics)


# Called by the learner when the run is over
def finish():
	global wandb_run, jsonl_file

	if wandb_run is not None:
		try:
			wandb_run.finish()
		except Exception as e:
			print(f"Metrics receiver: wandb finish failed ({repr(e)})", flush = True)
		wandb_run = None

	if jsonl_file is not None:
		jsonl_file.close()
		jsonl_file = None
