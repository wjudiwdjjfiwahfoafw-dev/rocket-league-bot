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
