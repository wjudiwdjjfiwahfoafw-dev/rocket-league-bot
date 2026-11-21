import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training_service import (
    BACKUP_RETRY_SECONDS,
    DEFAULT_BACKUP_INTERVAL,
    assess_goal,
    backup,
    seconds_until_backup_due,
    snapshot_layout,
    verify_frozen_versions,
)


class GoalTests(unittest.TestCase):
    def test_requires_all_three_fixed_measurements(self):
        state = {'best': {'steps': 1, 'screen': {'Ranking': 1701},
                          'confirm': {'Ranking': 1699}}}
        self.assertFalse(assess_goal(state)['selection_met'])
        state['best']['confirm']['Ranking'] = 1700
        self.assertTrue(assess_goal(state)['selection_met'])
        self.assertFalse(assess_goal(state)['verified'])
        state['audit'] = {'1': {'Ranking': 1699}}
        self.assertFalse(assess_goal(state)['verified'])
        state['audit']['1']['Ranking'] = 1700
        self.assertTrue(assess_goal(state)['verified'])

    def test_missing_measurements(self):
        result = assess_goal({})
        self.assertFalse(result['verified'])
        self.assertIsNone(result['screen_gap'])


class BackupIntervalTests(unittest.TestCase):
    def test_evolving_backup_contains_only_latest_model_and_exact_old_policy_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / 'campaign'
            latest = campaign / 'learner/9000'
            latest.mkdir(parents=True)
            versions = campaign / 'learner/policy_versions'
            versions.mkdir()
            for step in range(10):
                (versions / str(step)).mkdir()
            sources, targets, evolving = snapshot_layout(
                campaign, {'runtime_overrides': {'evolving_opponents': True, 'opponent_pool_size': 10}}, latest)
            self.assertTrue(evolving)
            self.assertEqual(sources, [latest, versions])
            self.assertEqual(targets, ['learner/9000', 'learner/policy_versions'])
            self.assertEqual(len(list(versions.iterdir())), 10)

    def test_evolving_backup_accepts_a_pool_that_is_still_growing(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / 'campaign'
            latest = campaign / 'learner/9000'
            latest.mkdir(parents=True)
            versions = campaign / 'learner/policy_versions'
            versions.mkdir()
            # A campaign that asks for fifteen policies but only has ten so far is still filling up.
            for step in range(10):
                (versions / str(step)).mkdir()
            sources, targets, evolving = snapshot_layout(
                campaign, {'runtime_overrides': {'evolving_opponents': True, 'opponent_pool_size': 15}}, latest)
            self.assertTrue(evolving)
            self.assertEqual(sources, [latest, versions])
            self.assertEqual(targets, ['learner/9000', 'learner/policy_versions'])
            # A pool that outgrew its target is a fault and must stop the backup.
            for step in range(10, 16):
                (versions / str(step)).mkdir()
            with self.assertRaises(RuntimeError):
                snapshot_layout(
                    campaign, {'runtime_overrides': {'evolving_opponents': True, 'opponent_pool_size': 15}}, latest)
            # An empty pool has nothing to restore.
            for version in versions.iterdir():
                version.rmdir()
            with self.assertRaises(RuntimeError):
                snapshot_layout(
                    campaign, {'runtime_overrides': {'evolving_opponents': True, 'opponent_pool_size': 15}}, latest)

    def test_backup_rejects_a_rewritten_old_policy_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            versions = Path(tmp) / 'policy_versions'
            versions.mkdir()
            for step in (100, 200):
                (versions / str(step)).mkdir()
                (versions / str(step) / 'STATS.json').write_text('{"skill_ratings": {}}')
            state = {'frozen_versions': {
                str(step): {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (versions / str(step)).iterdir()}
                for step in (100, 200)}}
            verify_frozen_versions(sorted(versions.iterdir()), state)
            # A re-rated or re-annotated old version must stop the backup, not ship silently.
            (versions / '200' / 'STATS.json').write_text('{"skill_ratings": {"1v1": 1600}}')
            with self.assertRaises(RuntimeError):
                verify_frozen_versions(sorted(versions.iterdir()), state)
            # Legacy snapshots without a manifest stay supported.
            verify_frozen_versions(sorted(versions.iterdir()), {})

    def test_default_backup_interval_is_thirty_minutes(self):
        self.assertEqual(DEFAULT_BACKUP_INTERVAL, 1800)
        self.assertEqual(BACKUP_RETRY_SECONDS, 60)

    def test_throttles_backups_within_thirty_minute_window_across_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'training-backup'
            dest.mkdir()
            (dest / 'manifest.json').write_text(json.dumps({'created': 10_000.0}))
            status = {'last_success': None, 'push_pending': False}
            # 3 minutes later (180s): must NOT trigger git/backup; 1620s remaining.
            self.assertAlmostEqual(
                seconds_until_backup_due(dest, status, DEFAULT_BACKUP_INTERVAL, now=10_180.0),
                1620.0,
            )
            with patch('training_service.git') as mock_git:
                ran = backup(Path(tmp) / 'campaign', dest, 'main', status,
                             min_interval=DEFAULT_BACKUP_INTERVAL, now=10_180.0)
            self.assertFalse(ran)
            mock_git.assert_not_called()
            self.assertAlmostEqual(status['next_due'], 11_800.0)

            # Exactly 30 minutes later (1800s): due immediately.
            self.assertEqual(
                seconds_until_backup_due(dest, status, DEFAULT_BACKUP_INTERVAL, now=11_800.0),
                0.0,
            )

            # Pending failed push bypasses wait so the existing commit is retried cleanly.
            status['push_pending'] = True
            self.assertEqual(
                seconds_until_backup_due(dest, status, DEFAULT_BACKUP_INTERVAL, now=10_240.0),
                0.0,
            )
