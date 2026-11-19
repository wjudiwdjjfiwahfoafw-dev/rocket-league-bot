"""Campaign budget, score validation, and promotion tests (no Torch, W&B or simulator needed)."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import train_ranked as ranked
from train_ranked import MODES, checked_ranking, minutes_plan, pool_digest, promote


class RankingTests(unittest.TestCase):
    def test_weight_only_execution_guard(self):
        ranked.validate_execution({'settings': {'deterministic': False, 'policy_temperature': 1.0}})
        for settings in ({}, {'deterministic': True, 'policy_temperature': 1.0},
                         {'deterministic': False, 'policy_temperature': 0.4},
                         {'deterministic': False, 'policy_temperature': 2.0}):
            with self.assertRaises(ValueError):
                ranked.validate_execution({'settings': settings})

    def test_policy_digest_tracks_actual_inference_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt'):
                (root / name).write_bytes(b'original')
            before = ranked.policy_digest(root)
            (root / 'POLICY.lt').write_bytes(b'changed without changing timestep')
            self.assertNotEqual(before, ranked.policy_digest(root))

    def test_budget(self):
        self.assertEqual(minutes_plan(180, 6, 30), [6, 30, 30, 30, 30, 30, 24])
        self.assertEqual(minutes_plan(3, 6, 30), [3])

    def test_continuous_campaign_repeats_regular_blocks_after_finite_plan(self):
        plan = [10, 15, 5]
        self.assertEqual(ranked.budget_for_stage(plan, 0), 0)
        self.assertEqual(ranked.budget_for_stage(plan, 1), 10)
        self.assertEqual(ranked.budget_for_stage(plan, 3), 5)
        self.assertIsNone(ranked.budget_for_stage(plan, 4))
        # Continuous mode discards the pilot/final short-block schedule and repeats one regular block.
        self.assertEqual(ranked.budget_for_stage(plan, 0, continuous=True, block=15), 0)
        self.assertEqual(ranked.budget_for_stage(plan, 1, continuous=True, block=15), 15)
        self.assertEqual(ranked.budget_for_stage(plan, 3, continuous=True, block=15), 15)
        self.assertEqual(ranked.budget_for_stage(plan, 10_000, continuous=True, block=15), 15)

    def test_both_seeds_and_margin_required(self):
        def rating(x):
            return {'Ranking': x}
        best = {'screen': rating(1500), 'confirm': rating(1490)}
        self.assertFalse(promote(rating(1540), rating(1489), best))
        self.assertFalse(promote(rating(1503), rating(1493), best))
        self.assertTrue(promote(rating(1511), rating(1495), best))

    def test_equal_weight_score_and_invalid_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '100').mkdir()
            ref = dict(zip(MODES, [1200, 1300, 1400]))
            stats = root / '100' / 'STATS.json'
            stats.write_text(json.dumps({'skill_ratings': ref}))
            result = dict(opponents=[dict(timesteps=100, rounds=10, ratings=ref.copy())], rounds=10,
                          ratings={m: {'average': v} for m, v in ref.items()},
                          mode_goals={m: {'for': n, 'against': 1} for m, n in zip(MODES, [50, 20, 5])})
            result['ratings']['Ranking'] = {'average': 1300}
            before = pool_digest(root)
            self.assertEqual(checked_ranking(result, root)['Ranking'], 1300)
            result['ratings']['Ranking']['average'] = 1250  # average over the first two modes only
            with self.assertRaises(ValueError):
                checked_ranking(result, root)
            result['ratings']['Ranking']['average'] = 1300
            result['opponents'][0]['ratings']['2v2'] += 10
            with self.assertRaises(ValueError):
                checked_ranking(result, root)
            self.assertEqual(before, pool_digest(root))

    def test_resume_evaluation_without_retraining_or_promoting_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, out = root / 'checkpoints/100', root / 'run'
            base.mkdir(parents=True)
            out.mkdir()
            bot = root / 'GigaLearnBot'
            bot.touch()
            (root / 'train_1v1_2v2_3v3.sh').touch()
            ranked.write(base / 'INFO.json', {'model': 's'})
            ranked.write(base / 'RUNNING_STATS.json', dict(total_timesteps=100, mirror_x=-1))
            for name in (*ranked.MODELS, *(k + '_OPTIM' for k in ranked.MODELS), 'SHARED_HEAD_SLOTS'):
                (base / (name + '.lt')).write_bytes(b'weights')
            opponent = base.parent / 'policy_versions/1'
            opponent.mkdir(parents=True)
            ref = {m: 1500 for m in MODES}
            ranked.write(opponent / 'STATS.json', dict(skill_ratings=ref))
            args = argparse.Namespace(base=base, out=out, bot=bot, minutes=2, pilot_minutes=1,
                                      block_minutes=1, rounds=2, threads=1, minibatch=1008,
                                      seed=1, wandb_offline=False, reward_metrics=True, player_metrics=True,
                                      throughput_profile='cpu-20k')
            trainings, training_commands, matches = [], [], []

            def native(cmd, env, log, heartbeat=None):
                checkpoints = Path(cmd[cmd.index('--checkpoints') + 1])
                ckpt = ranked.newest(checkpoints)
                n = int(ckpt.name)
                if cmd[0] == 'bash':
                    trainings.append(n)
                    training_commands.append(cmd)
                    dest = checkpoints / str(n + 20)
                    shutil.copytree(ckpt, dest)
                    ranked.write(dest / 'RUNNING_STATS.json', dict(total_timesteps=n + 20, mirror_x=-1))
                    return
                seed = cmd[cmd.index('--seed') + 1]
                if (n, seed) == (120, ranked.CONFIRM) and (n, seed) not in matches:
                    matches.append((n, seed))
                    raise InterruptedError
                matches.append((n, seed))
                rounds = cmd[cmd.index('--rate-rounds') + 1]
                elo = {100: 1500, 120: 1530, 140: 1520}[n]
                ranked.write(cmd[cmd.index('--rate-out') + 1], dict(
                    anchored=True, seed=seed, rounds=rounds, checkpoint_timesteps=n,
                    settings=dict(deterministic=False, policy_temperature=1.0),
                    ratings={m: {'average': elo} for m in (*MODES, 'Ranking')},
                    opponents=[dict(timesteps=1, rounds=rounds, ratings=ref)],
                    mode_goals={m: {'for': 3, 'against': 1} for m in MODES}))

            with patch.object(ranked, 'run', native), patch.object(ranked, 'REPO', root), contextlib.redirect_stdout(io.StringIO()):
                ranked.campaign(args)
                self.assertEqual(ranked.read(out / 'state.json')['status'], 'paused')
                ranked.campaign(args)
                state = ranked.read(out / 'state.json')
                self.assertEqual(state['status'], 'complete')
                self.assertEqual(trainings, [100, 120])
                self.assertEqual(matches.count((120, ranked.SCREEN)), 1)
                self.assertNotIn((140, ranked.CONFIRM), matches)
                self.assertEqual(state['best']['steps'], 120)
                self.assertTrue(state['runtime_overrides']['reward_metrics'])
                self.assertTrue(state['runtime_overrides']['player_metrics'])
                self.assertEqual(len(training_commands), 2)
                self.assertTrue(all('--reward-metrics' in cmd for cmd in training_commands))
                self.assertTrue(all('--player-metrics' in cmd for cmd in training_commands))
                for cmd in training_commands:
                    for flag, expected in (('--games', '64'), ('--epochs', '1'),
                                           ('--collection-infer-threads', '1')):
                        values = [cmd[i + 1] for i, token in enumerate(cmd[:-1]) if token == flag]
                        self.assertEqual(values[-1], expected)
                self.assertEqual(state['runtime_overrides']['throughput_profile'], 'cpu-20k')
                self.assertTrue((out / 'best/120/POLICY_OPTIM.lt').is_file())
                self.assertEqual(state['history'][-1]['steps'], 140)
                self.assertEqual(set(state['audit']), {'100', '120', '140'})
                self.assertEqual(ranked.pool_digest(opponent.parent), state['pool_sha256'])
                self.assertEqual(ranked.read(base / 'RUNNING_STATS.json')['total_timesteps'], 100)


if __name__ == '__main__':
    unittest.main()
