"""
Metrics receiver for GigaLearnCPP.

The C++ learner imports this module, calls init() once at startup, and then calls
add_metrics() once per iteration with a dict of {metric name: value}.

What this version does:
  * ALWAYS appends the reports to a local JSON-lines file:  <GGL_METRICS_DIR>/<run id>.jsonl
	(one JSON object per reporting window, plus a wall-clock "_time" field).
	This needs no account, no network and no extra Python packages.
	With GGL_METRICS_EVERY=N, numeric diagnostics are averaged over each N-iteration window;
	timestep/iteration counters retain their latest cumulative values, and rating events flush immediately.
  * ADDITIONALLY logs to Weights & Biases when the environment variable GGL_WANDB=1 is set
	(needs `pip install wandb`, and the API key in WANDB_API_KEY or a previous `wandb login`).

Environment variables (the bot executable sets some of them from its command-line options):
  GGL_METRICS_DIR     Folder for the .jsonl files (default: "metrics", relative to the working directory)
  GGL_METRICS_EVERY   Average N iterations per JSONL row (default: 1); rating events flush immediately.
  GGL_WANDB_EVERY     Average N iterations per W&B log record (default: 10); rating events flush immediately
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
COLLECTED_KEY = "Collected Timesteps"
OVERALL_SPS_KEY = "Overall Steps/Second"
ITERATION_TIME_KEY = "Iteration Time"
OVERHEAD_TIME_KEY = "Overhead Time"

# Process-start anchor so the very first emitted metric window also includes
# executable/Python/W&B startup, checkpoint loading, and arena initialization.
_PROCESS_START_MONO = time.monotonic()

wandb_run = None
jsonl_file = None
jsonl_path = None
jsonl_every = 1 # GGL_METRICS_EVERY
jsonl_count = 0
_jsonl_aggregates = {}
_wandb_aggregates = {}
_jsonl_window = {"last_mono": _PROCESS_START_MONO, "steps": 0.0, "active": 0.0}
_wandb_window = {"last_mono": _PROCESS_START_MONO, "steps": 0.0, "active": 0.0}


def _add_aggregate(aggregate, metrics):
	for key, value in metrics.items():
		key = str(key)
		try:
			number = float(value)
		except (TypeError, ValueError, OverflowError):
			aggregate[key] = {"last": value}
			continue
		if not math.isfinite(number):
			aggregate[key] = {"last": None}
			continue
		# A metric that was undefined or non-finite in an earlier report only carries "last", so a later
		# numeric report has to upgrade that placeholder: setdefault would keep the "sum"-less dict and
		# raise KeyError ('Episode Length' is null until the first episode ends; rating keys appear late).
		entry = aggregate.get(key)
		if entry is None or "sum" not in entry or "count" not in entry:
			entry = {"sum": 0.0, "count": 0, "last": value}
			aggregate[key] = entry
		entry["sum"] += number
		entry["count"] += 1
		entry["last"] = value


def _take_aggregate(aggregate):
	"""Average sampled scalar metrics; retain cumulative counters and step axis as last values."""
	latest_keys = {TIMESTEPS_KEY, "Total Iterations"}
	payload = {}
	for key, entry in aggregate.items():
		if key in latest_keys or key.startswith("Rating/") or not entry.get("count"):
			payload[key] = entry.get("last")
		elif key == COLLECTED_KEY:
			payload[key] = entry["sum"]
		else:
			payload[key] = entry["sum"] / entry["count"]
	aggregate.clear()
	return payload


def _flush_jsonl_aggregate():
	if jsonl_file is None or not _jsonl_aggregates:
		return
	row = _take_aggregate(_jsonl_aggregates)
	_flush_window_sps(_jsonl_window, row)
	row["_time"] = time.time()
	jsonl_file.write(json.dumps(row, sort_keys=True) + "\n")
	jsonl_file.flush()


def _flush_wandb_aggregate():
	if wandb_run is None or not _wandb_aggregates:
		return
	payload = _take_aggregate(_wandb_aggregates)
	_flush_window_sps(_wandb_window, payload)
	wandb_run.log(payload)


def _accumulate_window(window, metrics):
	try:
		window["steps"] += float(metrics.get(COLLECTED_KEY, 0.0) or 0.0)
		window["active"] += float(metrics.get("Collection Time", 0.0) or 0.0) + float(metrics.get("Consumption Time", 0.0) or 0.0)
	except Exception:
		pass


def _flush_window_sps(window, payload):
	"""Compute all-inclusive Overall Steps/Second over the contiguous wall-clock
	window since the previous emitted report (zero CPU/wall time excluded)."""
	now_mono = time.monotonic()
	elapsed = max(now_mono - window["last_mono"], 1e-9)
	steps = window["steps"]
	active = window["active"]
	window["last_mono"] = now_mono
	window["steps"] = 0.0
	window["active"] = 0.0
	if steps > 0.0:
		payload[OVERALL_SPS_KEY] = steps / elapsed
		payload[ITERATION_TIME_KEY] = elapsed
		payload[OVERHEAD_TIME_KEY] = max(0.0, elapsed - active)


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
	global jsonl_file, jsonl_path, jsonl_every, jsonl_count, _jsonl_aggregates, _wandb_aggregates

	if jsonl_count > 0:
		now_mono = time.monotonic()
		_jsonl_window.update(last_mono = now_mono, steps = 0.0, active = 0.0)
		_wandb_window.update(last_mono = now_mono, steps = 0.0, active = 0.0)
	jsonl_count = 0
	_jsonl_aggregates = {}
	_wandb_aggregates = {}
	add_metrics._wandb_count = 0

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
		_accumulate_window(_jsonl_window, metrics)
		_add_aggregate(_jsonl_aggregates, metrics)
		jsonl_count += 1
		if (jsonl_count - 1) % jsonl_every == 0 or any(str(k).startswith("Rating/") for k in metrics):
			_flush_jsonl_aggregate()

	if wandb_run is not None:
		# Aggregate skipped iterations rather than logging a single noisy iteration sample.
		# GGL_WANDB_EVERY=N controls the averaging window; ratings flush the window immediately.
		_accumulate_window(_wandb_window, metrics)
		_add_aggregate(_wandb_aggregates, metrics)
		add_metrics._wandb_count = getattr(add_metrics, "_wandb_count", 0) + 1
		try:
			wandb_every = max(1, int(os.environ.get("GGL_WANDB_EVERY", "10")))
		except ValueError:
			wandb_every = 10
		if (add_metrics._wandb_count - 1) % wandb_every == 0 or any(str(k).startswith("Rating/") for k in metrics):
			_flush_wandb_aggregate()


# Called by the learner when the run is over
def finish():
	global wandb_run, jsonl_file

	_flush_wandb_aggregate()
	_flush_jsonl_aggregate()

	if wandb_run is not None:
		try:
			wandb_run.finish()
		except Exception as e:
			print(f"Metrics receiver: wandb finish failed ({repr(e)})", flush = True)
		wandb_run = None

	if jsonl_file is not None:
		jsonl_file.close()
		jsonl_file = None
