import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from resume_training import restore, verify


class RestoreTests(unittest.TestCase):
    def make_snapshot(self, path):
        path.mkdir()
        state = b'{"status":"paused","pending":{"seconds":1366.75}}'
        (path / 'state.json').write_bytes(state)
        (path / 'manifest.json').write_text(json.dumps({'checkpoint': 123,
            'sha256': {'state.json': hashlib.sha256(state).hexdigest()}}))

    def test_restore_verifies_and_never_overwrites_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, work = root / 'snapshot', root / 'work'
            self.make_snapshot(snapshot)
            result = restore(snapshot, work)
            self.assertEqual(result['pending']['seconds'], 1366.75)
            self.assertFalse((work / 'manifest.json').exists())
            (work / 'state.json').write_text('{"status":"newer"}')
            self.assertEqual(restore(snapshot, work)['status'], 'newer')
            self.assertEqual(json.loads((snapshot / 'state.json').read_text())['status'], 'paused')

    def test_tampered_backup_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, work = root / 'snapshot', root / 'work'
            self.make_snapshot(snapshot)
            (snapshot / 'state.json').write_text('corrupt')
            with self.assertRaises(ValueError):
                restore(snapshot, work)
            self.assertFalse(work.exists())

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_snapshot(root / 'snapshot')
            (root / 'snapshot/manifest.json').write_text(json.dumps({'sha256': {'../outside': 'bad'}}))
            with self.assertRaises(ValueError):
                verify(root / 'snapshot')
