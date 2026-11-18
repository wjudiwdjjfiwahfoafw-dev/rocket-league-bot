import unittest
from training_service import assess_goal


class GoalTests(unittest.TestCase):
    def test_requires_all_three_fixed_measurements(self):
        state = {'best': {'steps': 1, 'screen': {'Ranking': 1601},
                          'confirm': {'Ranking': 1599}}}
        self.assertFalse(assess_goal(state)['selection_met'])
        state['best']['confirm']['Ranking'] = 1600
        self.assertTrue(assess_goal(state)['selection_met'])
        self.assertFalse(assess_goal(state)['verified'])
        state['audit'] = {'1': {'Ranking': 1599}}
        self.assertFalse(assess_goal(state)['verified'])
        state['audit']['1']['Ranking'] = 1600
        self.assertTrue(assess_goal(state)['verified'])

    def test_missing_measurements(self):
        result = assess_goal({})
        self.assertFalse(result['verified'])
        self.assertIsNone(result['screen_gap'])
