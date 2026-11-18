import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wandb_live_bridge as bridge


class WandbLiveBridgeTest(unittest.TestCase):
    def _make_campaign(self, root: Path, num_rows: int = 5):
        (root / 'metrics').mkdir(parents=True)
        (root / 'eval').mkdir(parents=True)
        state = {
            'config': {
                'base': 'checkpoints/3244539846',
                'parent_steps': 3244539846,
                'seed': 20261120,
                'screen_seed': 1009,
                'confirm_seed': 2003,
                'audit_seed': 9007,
                'rounds': 40,
                'protocol': 'anchored-v1',
            },
            'target_ranking': 1700.0,
            'status': 'training',
            'history': [
                {
                    'stage': 0,
                    'start_steps': 3244539846,
                    'steps': 3244539846,
                    'steps_per_second': None,
                    'screen': {'1v1': 1524.4, '2v2': 1658.8, '3v3': 1592.7, 'Ranking': 1592.0},
                    'confirm': {'1v1': 1551.0, '2v2': 1647.7, '3v3': 1608.6, 'Ranking': 1602.4},
                    'promoted': True,
                },
                {
                    'stage': 1,
                    'start_steps': 3244539846,
                    'steps': 3244539846 + num_rows * 48384,
                    'steps_per_second': 16200.0,
                    'screen': {'1v1': 1540.0, '2v2': 1710.0, '3v3': 1690.0, 'Ranking': 1646.7},
                    'confirm': {'1v1': 1560.0, '2v2': 1725.0, '3v3': 1705.0, 'Ranking': 1663.3},
                    'promoted': True,
                },
            ],
            'best': {
                'steps': 3244539846 + num_rows * 48384,
                'screen': {'1v1': 1540.0, '2v2': 1710.0, '3v3': 1690.0, 'Ranking': 1646.7},
                'confirm': {'1v1': 1560.0, '2v2': 1725.0, '3v3': 1705.0, 'Ranking': 1663.3},
            },
        }
        (root / 'state.json').write_text(json.dumps(state) + '\n')
        lines = []
        for i in range(1, num_rows + 1):
            lines.append(json.dumps({
                'Total Timesteps': 3244539846 + i * 48384,
                'Policy Entropy': 0.35 - 0.001 * i,
                'Mean KL Divergence': 0.0031,
                'Rating/1v1': 1500.0 + i,
                '_time': 1700000000 + i,
            }) + '\n')
        # Add an unterminated partial row to ensure reader ignores half-written lines.
        lines.append('{"Total Timesteps": 9999999999, "Policy Entropy": 0.1')
        (root / 'metrics/run.jsonl').write_text(''.join(lines))

        ev_step = 3244539846 + num_rows * 48384
        (root / f'eval/{ev_step}-2003-80.json').write_text(json.dumps({
            'checkpoint_timesteps': ev_step,
            'seed': 2003,
            'rounds': 80,
            'kickoff_first_touches_for': 420,
            'kickoff_first_touches_against': 380,
            'goals_for': 1800,
            'goals_against': 700,
            'mode_goals': {
                '1v1': {'for': 700, 'against': 300},
                '2v2': {'for': 600, 'against': 220},
                '3v3': {'for': 500, 'against': 180},
            },
        }) + '\n')

    def test_snapshot_round_trip_and_idempotent_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_campaign(root, num_rows=6)
            snap = bridge.read_campaign_snapshot(root, status_url='https://8080-demo.e2b.app/status.json')
            self.assertEqual(len(snap['rows']), 6)
            body, included = bridge.encode_snapshot_body(snap)
            self.assertEqual(included, 6)
            self.assertLess(len(body), 60_000)

            decoded = bridge.decode_snapshot_body(body)
            self.assertEqual(decoded['run_id'], snap['run_id'])
            self.assertEqual(decoded['status_url'], 'https://8080-demo.e2b.app/status.json')
            self.assertEqual(len(decoded['rows']), 6)

            events, new_stages, summary = bridge.build_wandb_events(decoded, last_step=-1, logged_stages=set())
            self.assertEqual(new_stages, {0, 1})
            # Stage 0 at parent_steps + 6 training steps (stage 1 merges into step 6).
            self.assertEqual(len(events), 7)
            self.assertEqual([s for s, _ in events], sorted(s for s, _ in events))
            # Native live tracker rating must be prefixed with Live/ while frozen eval uses Rating/
            self.assertIn('Live/Rating/1v1', events[-1][1])
            self.assertAlmostEqual(events[-1][1]['Rating/Ranking'], 1646.7)
            self.assertAlmostEqual(events[-1][1]['Confirmation/Ranking'], 1663.3)
            self.assertAlmostEqual(events[-1][1]['Eval/Goal Share 1v1 %'], 70.0)
            self.assertEqual(summary['Campaign/Status'], 'training')

            # Re-running with updated cursor produces zero duplicate events!
            events2, new_stages2, _ = bridge.build_wandb_events(
                decoded, last_step=events[-1][0], logged_stages=new_stages
            )
            self.assertEqual(events2, [])
            self.assertEqual(new_stages2, set())

    def test_large_payload_trims_gracefully_under_github_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_campaign(root, num_rows=80)
            snap = bridge.read_campaign_snapshot(root)
            body, included = bridge.encode_snapshot_body(snap, max_b64=2200)
            self.assertLess(included, 80)
            self.assertGreater(included, 0)
            decoded = bridge.decode_snapshot_body(body)
            # Always keeps the newest step and complete campaign stage history.
            self.assertEqual(decoded['rows'][-1]['Total Timesteps'], snap['rows'][-1]['Total Timesteps'])
            self.assertEqual(len(decoded['state']['history']), 2)

    def test_relay_meta_round_trip(self):
        meta = {
            'run_url': 'https://wandb.ai/demo/rocket-league-ggl/runs/abc123',
            'run_id': 'abc123',
            'auth_mode': 'api_key',
            'last_step': 3265973958,
            'stages': [0, 1, 2],
            'source': 'live-preview-http',
            'updated_iso': '2026-11-18 21:15:00 UTC',
        }
        parsed = bridge.decode_relay_meta(bridge.encode_relay_meta(meta))
        self.assertEqual(parsed, meta)


if __name__ == '__main__':
    unittest.main()
