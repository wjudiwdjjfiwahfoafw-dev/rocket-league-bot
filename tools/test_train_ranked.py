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

    def test_diverse_pool_spans_historical_versions_and_fifo_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, pool, candidate = root / 'source', root / 'pool', root / '1200'
            source.mkdir()
            for step in range(100, 1200, 100):
                version = source / str(step)
                version.mkdir()
                for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt'):
                    (version / name).write_bytes(f'{step}-{name}'.encode())
                ranked.write(version / 'STATS.json', {'skill_ratings': {m: 1500 + step / 100 for m in MODES}})
            ranked.seed_policy_pool(source, pool, 10)
            chosen = [int(p.name) for p in ranked.policy_versions(pool)]
            self.assertEqual(len(chosen), 10)
            self.assertEqual(chosen[0], 100)
            self.assertEqual(chosen[-1], 1100)
            self.assertGreaterEqual(min(b - a for a, b in zip(chosen, chosen[1:])), 100)

            candidate.mkdir()
            for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt'):
                (candidate / name).write_bytes(b'new')
            ranked.write(candidate / 'RUNNING_STATS.json', {'skill_ratings': {m: 1700 for m in MODES}})
            self.assertTrue(ranked.add_policy_to_pool(candidate, pool, 10))
            promoted = [int(p.name) for p in ranked.policy_versions(pool)]
            self.assertEqual(len(promoted), 10)
            self.assertNotIn(100, promoted)
            self.assertIn(1200, promoted)

    def test_promotion_requires_margin_over_strongest_policy_in_both_evals(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp)
            for step, elo in ((100, 1500), (200, 1600)):
                version = pool / str(step)
                version.mkdir()
                ranked.write(version / 'STATS.json', {'skill_ratings': {m: elo for m in MODES}})
            self.assertEqual(ranked.strongest_pool_ranking(pool), (1600.0, 200))
            ok, strongest, version = ranked.beats_strongest(
                {'Ranking': 1650}, {'Ranking': 1651}, pool, 50)
            self.assertTrue(ok)
            self.assertEqual((strongest, version), (1600.0, 200))
            self.assertFalse(ranked.beats_strongest(
                {'Ranking': 1649}, {'Ranking': 1700}, pool, 50)[0])

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
                          ratings={m: {'average': v + 40, 'end': v} for m, v in ref.items()},
                          mode_goals={m: {'for': n, 'against': 1} for m, n in zip(MODES, [50, 20, 5])})
            result['ratings']['Ranking'] = {'average': 1340, 'end': 1300}
            before = pool_digest(root)
            self.assertEqual(checked_ranking(result, root)['Ranking'], 1300)
            result['ratings']['Ranking']['end'] = 1250  # raw endpoint must match the mode endpoints
            with self.assertRaises(ValueError):
                checked_ranking(result, root)
            result['ratings']['Ranking']['end'] = 1300
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
                    ratings={m: {'average': elo + 10, 'end': elo} for m in (*MODES, 'Ranking')},
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

    def test_evolving_campaign_trains_and_rates_against_same_ten_then_fifo_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, out = root / 'checkpoints/100', root / 'campaign'
            base.mkdir(parents=True)
            out.mkdir()
            bot = root / 'GigaLearnBot'
            bot.touch()
            (root / 'train_1v1_2v2_3v3.sh').touch()
            ranked.write(base / 'INFO.json', {'model': 's'})
            ranked.write(base / 'RUNNING_STATS.json', dict(total_timesteps=100, mirror_x=-1,
                                                            skill_ratings={m: 1500 for m in MODES}))
            for name in (*ranked.MODELS, *(k + '_OPTIM' for k in ranked.MODELS), 'SHARED_HEAD_SLOTS'):
                (base / (name + '.lt')).write_bytes(b'weights')
            versions = base.parent / 'policy_versions'
            for step in range(1, 11):
                version = versions / str(step)
                version.mkdir(parents=True)
                for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt'):
                    (version / name).write_bytes(f'{step}-{name}'.encode())
                ranked.write(version / 'STATS.json', {'skill_ratings': {m: 1500 + step for m in MODES}})
            args = argparse.Namespace(
                base=base, out=out, bot=bot, minutes=1, pilot_minutes=1, block_minutes=1,
                rounds=20, threads=1, minibatch=1008, seed=1, audit_seed=9007,
                wandb_offline=False, wandb_online=False, wandb_bridge=False, github_backup=False,
                reward_metrics=False, player_metrics=False, throughput_profile=None, continuous=False,
                evolving_opponents=True, opponent_pool_size=10, promotion_margin=50.0,
                reference_pool=None, extra_versions=None, target_ranking=None, stop_on_target=False,
                rewards=None,
            )
            train_commands = []
            def native(cmd, env, log, heartbeat=None):
                checkpoints = Path(cmd[cmd.index('--checkpoints') + 1])
                if cmd[0] == 'bash':
                    train_commands.append(cmd)
                    parent = ranked.newest(checkpoints)
                    candidate = checkpoints / '120'
                    shutil.copytree(parent, candidate)
                    ranked.write(candidate / 'RUNNING_STATS.json', dict(
                        total_timesteps=120, mirror_x=-1,
                        skill_ratings={m: 1700 for m in MODES}))
                    return
                candidate_path = ranked.newest(Path(cmd[cmd.index('--checkpoints') + 1]))
                step = int(candidate_path.name)
                seed = int(cmd[cmd.index('--seed') + 1])
                rounds = int(cmd[cmd.index('--rate-rounds') + 1])
                pool = Path(cmd[cmd.index('--rate-versions') + 1])
                refs = {int(p.name): ranked.read(p / 'STATS.json')['skill_ratings']
                        for p in ranked.policy_versions(pool)}
                elo = 1700.0 if step == 120 else 1500.0
                ratings = {m: {'average': elo - 25, 'end': elo} for m in MODES}
                ratings['Ranking'] = {'average': elo - 25, 'end': elo}
                result = dict(
                    anchored=True, seed=seed, rounds=rounds, checkpoint_timesteps=step,
                    settings={'deterministic': False, 'policy_temperature': 1.0}, ratings=ratings,
                    opponents=[dict(timesteps=t, rounds=rounds // len(refs), ratings=r)
                               for t, r in refs.items()],
                    mode_goals={m: {'for': 3, 'against': 1} for m in MODES})
                ranked.write(cmd[cmd.index('--rate-out') + 1], result)

            with patch.object(ranked, 'run', native), patch.object(ranked, 'REPO', root), \
                    contextlib.redirect_stdout(io.StringIO()):
                ranked.campaign(args)
            state = ranked.read(out / 'state.json')
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['history'][-1]['steps'], 120)
            self.assertTrue(state['history'][-1]['opponent_pool_promoted'])
            self.assertEqual(state['history'][-1]['strongest_opponent_ranking'], 1510)
            self.assertEqual(len(ranked.policy_versions(out / 'pool')), 10)
            self.assertNotIn(1, [int(p.name) for p in ranked.policy_versions(out / 'pool')])
            self.assertIn(120, [int(p.name) for p in ranked.policy_versions(out / 'pool')])
            self.assertEqual({p.name for p in ranked.policy_versions(out / 'pool')},
                             {p.name for p in ranked.policy_versions(out / 'learner/policy_versions')})
            train_cmd = train_commands[0]
            self.assertEqual(train_cmd[train_cmd.index('--max-versions') + 1], '10')
            self.assertEqual(train_cmd[train_cmd.index('--ts-per-version') + 1], '9223372036854775807')


if __name__ == '__main__':
    unittest.main()
