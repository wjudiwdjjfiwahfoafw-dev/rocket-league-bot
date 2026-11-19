import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import ranked_wandb


class StreamTests(unittest.TestCase):
    def test_backfill_partial_line_and_metric_names(self):
        class Run:
            url = 'mock'
            def __init__(self):
                self.summary = {}
                self.rows = []
                self.closed = False
            def log(self, row):
                self.rows.append(row)
            def finish(self):
                self.closed = True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'metrics').mkdir()
            (root / 'metrics/a.jsonl').write_text(
                json.dumps({'Total Timesteps': 2, 'Rating/Ranking': 1500}) + '\n' + '{"partial":')
            (root / 'state.json').write_text(json.dumps({
                'config': {}, 'status': 'complete', 'best': {'steps': 1},
                'history': [{'stage': 0, 'steps': 1, 'screen': {'Ranking': 1400}, 'confirm': None}]}))
            run = Run()
            with patch.object(ranked_wandb, 'start', return_value=run) as start:
                ranked_wandb.stream(root, root / 'out')
            self.assertTrue(start.call_args.kwargs['online'])
            self.assertEqual([r['Total Timesteps'] for r in run.rows], [1, 2])
            self.assertEqual(run.rows[0]['Rating/Ranking'], 1400)
            self.assertEqual(run.rows[1]['Elo/Ranking'], 1500)
            self.assertFalse(any(key.startswith('Live/') for row in run.rows for key in row))
            self.assertFalse(any(key.startswith('Stream/') for key in run.summary))
            self.assertTrue((root / 'out/.relay_cursor.json').is_file())
            self.assertTrue(run.closed)
            # Resuming the stream doesn't replay rows or completed evaluations.
            run.rows.clear()
            with patch.object(ranked_wandb, 'start', return_value=run):
                ranked_wandb.stream(root, root / 'out')
            self.assertEqual(run.rows, [])
