import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'GigaLearnCPP/python_scripts'))
import metric_receiver  # noqa: E402


class MetricReceiverThroughputTests(unittest.TestCase):
    def test_overall_sps_includes_all_inter_iteration_overhead(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = metric_receiver.os.environ.get('GGL_METRICS_DIR')
            old_every = metric_receiver.os.environ.get('GGL_METRICS_EVERY')
            old_wandb = metric_receiver.os.environ.get('GGL_WANDB')
            try:
                metric_receiver.os.environ['GGL_METRICS_DIR'] = tmp
                metric_receiver.os.environ['GGL_METRICS_EVERY'] = '2'
                metric_receiver.os.environ['GGL_WANDB'] = '0'
                metric_receiver.jsonl_count = 0
                now0 = time.monotonic()
                metric_receiver._jsonl_window.update(last_mono=now0, steps=0.0, active=0.0)
                run_id = metric_receiver.init(sys.executable, 'proj', 'grp', 'run', None)
                try:
                    # Iteration 1 emitted immediately (index 0 % 2 == 0).
                    time.sleep(0.02)
                    metric_receiver.add_metrics({
                        'Total Timesteps': 1000,
                        'Collected Timesteps': 1000,
                        'Collection Time': 0.005,
                        'Consumption Time': 0.005,
                        'Overall Steps/Second': 100_000.0,
                    })
                    t_after_first = time.monotonic()
                    # Iteration 2 skipped in JSONL, carrying unmeasured overhead (e.g. checkpoint save).
                    time.sleep(0.03)
                    metric_receiver.add_metrics({
                        'Total Timesteps': 2000,
                        'Collected Timesteps': 1000,
                        'Collection Time': 0.005,
                        'Consumption Time': 0.005,
                        'Overall Steps/Second': 100_000.0,
                    })
                    # Iteration 3 emitted (index 2 % 2 == 0): must divide ALL 2000 steps by FULL wall time.
                    time.sleep(0.03)
                    metric_receiver.add_metrics({
                        'Total Timesteps': 3000,
                        'Collected Timesteps': 1000,
                        'Collection Time': 0.005,
                        'Consumption Time': 0.005,
                        'Overall Steps/Second': 100_000.0,
                    })
                    t_after_third = time.monotonic()
                finally:
                    metric_receiver.finish()

                rows = [json.loads(line) for line in (Path(tmp) / f'{run_id}.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                full_wall = t_after_third - t_after_first
                # Reported Overall Steps/Second must reflect full ~60ms wall time (~33k sps), NOT compute-only 100k sps.
                self.assertLess(rows[1]['Overall Steps/Second'], 40_000.0)
                self.assertAlmostEqual(rows[1]['Iteration Time'], full_wall, delta=0.01)
                self.assertGreater(rows[1]['Overhead Time'], 0.035)
            finally:
                for k, v in [('GGL_METRICS_DIR', old_dir), ('GGL_METRICS_EVERY', old_every), ('GGL_WANDB', old_wandb)]:
                    if v is None:
                        metric_receiver.os.environ.pop(k, None)
                    else:
                        metric_receiver.os.environ[k] = v
