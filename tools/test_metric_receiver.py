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
    def test_sparse_rows_average_skipped_iterations_and_keep_cumulative_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = metric_receiver.os.environ.get('GGL_METRICS_DIR')
            old_every = metric_receiver.os.environ.get('GGL_METRICS_EVERY')
            old_wandb = metric_receiver.os.environ.get('GGL_WANDB')
            try:
                metric_receiver.os.environ['GGL_METRICS_DIR'] = tmp
                metric_receiver.os.environ['GGL_METRICS_EVERY'] = '3'
                metric_receiver.os.environ['GGL_WANDB'] = '0'
                run_id = metric_receiver.init(sys.executable, 'proj', 'grp', 'run', None)
                try:
                    for i in range(1, 5):
                        metric_receiver.add_metrics({
                            'Total Timesteps': i * 1000,
                            'Total Iterations': i,
                            'Collected Timesteps': 1000,
                            'Player/Speed': float(i),
                            'Average Step Reward': float(i * 2),
                        })
                finally:
                    metric_receiver.finish()

                rows = [json.loads(line) for line in (Path(tmp) / f'{run_id}.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[1]['Total Timesteps'], 4000)
                self.assertEqual(rows[1]['Total Iterations'], 4)
                self.assertEqual(rows[1]['Collected Timesteps'], 3000)
                self.assertEqual(rows[1]['Player/Speed'], 3.0)
                self.assertEqual(rows[1]['Average Step Reward'], 6.0)
            finally:
                for k, v in [('GGL_METRICS_DIR', old_dir), ('GGL_METRICS_EVERY', old_every), ('GGL_WANDB', old_wandb)]:
                    if v is None:
                        metric_receiver.os.environ.pop(k, None)
                    else:
                        metric_receiver.os.environ[k] = v

    def test_wandb_sparse_rows_average_skipped_iterations(self):
        class FakeRun:
            def __init__(self):
                self.rows = []
                self.finished = False

            def log(self, payload):
                self.rows.append(dict(payload))

            def finish(self):
                self.finished = True

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = metric_receiver.os.environ.get('GGL_METRICS_DIR')
            old_every = metric_receiver.os.environ.get('GGL_METRICS_EVERY')
            old_wandb = metric_receiver.os.environ.get('GGL_WANDB')
            old_wandb_every = metric_receiver.os.environ.get('GGL_WANDB_EVERY')
            try:
                metric_receiver.os.environ['GGL_METRICS_DIR'] = tmp
                metric_receiver.os.environ['GGL_METRICS_EVERY'] = '1'
                metric_receiver.os.environ['GGL_WANDB'] = '0'
                metric_receiver.os.environ['GGL_WANDB_EVERY'] = '3'
                metric_receiver.init(sys.executable, 'proj', 'grp', 'run', None)
                fake = FakeRun()
                metric_receiver.wandb_run = fake
                for i in range(1, 5):
                    metric_receiver.add_metrics({
                        'Total Timesteps': i * 1000,
                        'Collected Timesteps': 1000,
                        'Player/Speed': float(i),
                    })
                metric_receiver.finish()
                self.assertEqual(len(fake.rows), 2)
                self.assertEqual(fake.rows[1]['Total Timesteps'], 4000)
                self.assertEqual(fake.rows[1]['Collected Timesteps'], 3000)
                self.assertEqual(fake.rows[1]['Player/Speed'], 3.0)
                self.assertTrue(fake.finished)
            finally:
                metric_receiver.finish()
                for k, v in [('GGL_METRICS_DIR', old_dir), ('GGL_METRICS_EVERY', old_every),
                             ('GGL_WANDB', old_wandb), ('GGL_WANDB_EVERY', old_wandb_every)]:
                    if v is None:
                        metric_receiver.os.environ.pop(k, None)
                    else:
                        metric_receiver.os.environ[k] = v

    def test_rating_events_are_logged_exactly_and_never_averaged(self):
        class FakeRun:
            def __init__(self):
                self.rows = []
                self.finished = False

            def log(self, payload):
                self.rows.append(dict(payload))

            def finish(self):
                self.finished = True

        with tempfile.TemporaryDirectory() as tmp:
            old_values = {k: metric_receiver.os.environ.get(k) for k in
                          ('GGL_METRICS_DIR', 'GGL_METRICS_EVERY', 'GGL_WANDB', 'GGL_WANDB_EVERY')}
            try:
                metric_receiver.os.environ.update(GGL_METRICS_DIR=tmp, GGL_METRICS_EVERY='50',
                                                  GGL_WANDB='0', GGL_WANDB_EVERY='50')
                metric_receiver.init(sys.executable, 'proj', 'grp', 'run', None)
                fake = FakeRun()
                metric_receiver.wandb_run = fake
                for i, elo in enumerate((1512.25, 1498.75, 1530.5), 1):
                    metric_receiver.add_metrics({'Total Timesteps': i * 1000,
                                                 'Collected Timesteps': 1000,
                                                 'Player/Speed': float(i),
                                                 'Rating/1v1': elo})
                self.assertEqual([row['Rating/1v1'] for row in fake.rows],
                                 [1512.25, 1498.75, 1530.5])
                self.assertEqual([row['Total Timesteps'] for row in fake.rows], [1000, 2000, 3000])
            finally:
                metric_receiver.wandb_run = None
                metric_receiver.finish()
                for key, value in old_values.items():
                    if value is None:
                        metric_receiver.os.environ.pop(key, None)
                    else:
                        metric_receiver.os.environ[key] = value

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

    def test_undefined_then_numeric_metric_keeps_aggregating(self):
        # "Episode Length" is null until the first episode ends, so its aggregate starts as {"last": None}
        aggregate = {}
        metric_receiver._add_aggregate(aggregate, {'Episode Length': None, 'Total Iterations': 1})
        metric_receiver._add_aggregate(aggregate, {'Episode Length': 4.0, 'Total Iterations': 2})
        metric_receiver._add_aggregate(aggregate, {'Episode Length': 6.0, 'Total Iterations': 3})
        payload = metric_receiver._take_aggregate(aggregate)
        self.assertEqual(payload['Episode Length'], 5.0)
        self.assertEqual(payload['Total Iterations'], 3)
