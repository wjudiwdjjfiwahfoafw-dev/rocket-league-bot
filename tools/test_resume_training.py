import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import resume_training
import train_ranked
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

    def test_compact_evolving_backup_rebuilds_local_only_campaign_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, work = root / 'snapshot', root / 'work'
            (snapshot / 'learner/9000').mkdir(parents=True)
            versions = snapshot / 'learner/policy_versions'
            versions.mkdir()
            for step in range(10):
                version = versions / str(step)
                version.mkdir()
                for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt'):
                    (version / name).write_bytes(f'{step}:{name}'.encode())
                (version / 'STATS.json').write_text(json.dumps({'skill_ratings': {
                    '1v1': 1500 + step, '2v2': 1500 + step, '3v3': 1500 + step}}))
            state = {'status': 'paused', 'runtime_overrides': {
                'evolving_opponents': True, 'opponent_pool_size': 10}}
            (snapshot / 'state.json').write_text(json.dumps(state))
            hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in snapshot.rglob('*') if p.is_file()}
            (snapshot / 'manifest.json').write_text(json.dumps({'checkpoint': 9000, 'sha256': hashes}))
            restored = restore(snapshot, work)
            self.assertEqual(len(list((work / 'pool').iterdir())), 10)
            self.assertTrue(all((work / folder).is_dir() for folder in ('eval', 'metrics', 'best')))
            self.assertEqual(restored['pool_sha256'], train_ranked.pool_digest(work / 'pool'))

    def test_tampered_backup_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, work = root / 'snapshot', root / 'work'
            self.make_snapshot(snapshot)
            (snapshot / 'state.json').write_text('corrupt')
            with self.assertRaises(ValueError):
                restore(snapshot, work)
            self.assertFalse(work.exists())

    def test_restore_only_uses_saved_campaign_path_without_starting_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / 'snapshot'
            self.make_snapshot(snapshot)
            state = json.loads((snapshot / 'state.json').read_text())
            state['working_directory'] = 'results/weights1700/campaign'
            payload = json.dumps(state).encode()
            (snapshot / 'state.json').write_bytes(payload)
            (snapshot / 'manifest.json').write_text(json.dumps({'checkpoint': 123,
                'sha256': {'state.json': hashlib.sha256(payload).hexdigest()}}))
            with patch.object(resume_training, 'ROOT', root), patch.object(resume_training, 'SNAPSHOT', snapshot), \
                 patch.object(resume_training, 'WORK', root / 'results/old'), \
                 patch.object(resume_training, 'restore', wraps=lambda **kw: restore(snapshot=snapshot, **kw)), \
                 patch.object(resume_training.subprocess, 'run') as launch, \
                 patch('sys.argv', ['resume_training.py', '--restore-only']):
                resume_training.main()
            self.assertTrue((root / 'results/weights1700/campaign/state.json').exists())
            self.assertFalse((root / 'results/old').exists())
            launch.assert_not_called()

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_snapshot(root / 'snapshot')
            (root / 'snapshot/manifest.json').write_text(json.dumps({'sha256': {'../outside': 'bad'}}))
            with self.assertRaises(ValueError):
                verify(root / 'snapshot')

    def test_build_resume_command_preserves_new_recipe_and_30m_backup(self):
        state = {
            'target_ranking': 1700.0,
            'runtime_overrides': {'reward_metrics': True, 'player_metrics': True, 'throughput_profile': 'cpu-20k', 'continuous': True,
                                'evolving_opponents': True, 'opponent_pool_size': 10,
                                'promotion_margin': 50.0},
            'config': {
                'base': 'checkpoints/3244539846',
                'minutes': [12, 20, 20],
                'rounds': 40,
                'seed': 20261118,
                'audit_seed': 9007,
                'reference_pool': 'training-backup/pool',
                'extra_versions': ['checkpoints/3137998278'],
                'train_args': [
                    '--threads', '2', '--games', '88', '--minibatch-size', '1008',
                    '--lr', '0.00011', '--critic-lr', '0.0002', '--epochs', '2',
                    '--entropy-scale', '0.0025', '--norm-advantages', 'mode',
                    '--target-kl', '0.02', '--tactical-state-prob', '0.22',
                    '--train-vs-old', '0.3', '--ts-per-save', '10m',
                    '--mask-entropy', '--fuzzed-kickoffs', '--pfsp',
                    '--rewards', 'Goal=600,Save=65',
                ],
            },
        }
        cmd = resume_training.build_resume_command(
            state, resume_training.ROOT / 'results/weights1700-v3/campaign'
        )
        self.assertNotIn('--monitor', cmd)
        for token in ('--github-backup', '--backup-interval', '1800', '--wandb-bridge',
                      '--mask-entropy', '--fuzzed-kickoffs',
                      '--pfsp', '--norm-advantages', 'mode', '--epochs', '2',
                      '--critic-lr', '0.0002', '--tactical-state-prob', '0.22',
                      '--extra-versions', 'checkpoints/3137998278', '--reward-metrics', '--player-metrics',
                      '--throughput-profile', 'cpu-20k', '--continuous', '--evolving-opponents',
                      '--opponent-pool-size', '10', '--promotion-margin', '50.0'):
            self.assertIn(token, cmd)
