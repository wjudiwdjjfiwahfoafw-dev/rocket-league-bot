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
from train_ranked import (MODES, checked_ranking, endpoint_ratings, frozen_versions, minutes_plan,
                             pool_digest, promote, sync_policy_versions, verify_frozen_versions)
import verify_frozen_versions as frozen_report


class RankingTests(unittest.TestCase):
    def test_weight_only_execution_guard(self):
        ranked.validate_execution({'settings': {'deterministic': False, 'policy_temperature': 1.0,
                                                'fuzzed_kickoffs': True}})
        for settings in ({}, {'deterministic': True, 'policy_temperature': 1.0, 'fuzzed_kickoffs': True},
                         {'deterministic': False, 'policy_temperature': 0.4, 'fuzzed_kickoffs': True},
                         {'deterministic': False, 'policy_temperature': 2.0, 'fuzzed_kickoffs': True},
                         # A rating without randomized kickoffs would replay its start states every round.
                         {'deterministic': False, 'policy_temperature': 1.0},
                         {'deterministic': False, 'policy_temperature': 1.0, 'fuzzed_kickoffs': False}):
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

    def test_pool_grows_to_the_requested_size_before_it_evicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, pool = root / 'source', root / 'pool'
            source.mkdir()
            for step in (100, 200, 300, 400):
                version = source / str(step)
                version.mkdir()
                for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt'):
                    (version / name).write_bytes(f'{step}-{name}'.encode())
                ranked.write(version / 'STATS.json', {'skill_ratings': {m: 1500.0 for m in MODES}})

            # Asking for more policies than history holds must seed what exists instead of raising, so a
            # 10 -> 15 pool increase works on a campaign that only ever saved 10 versions.
            ranked.seed_policy_pool(source, pool, 6)
            self.assertEqual(len(ranked.policy_versions(pool)), 4)

            # Promotions fill the pool without evicting anything...
            for step in (500, 600, 700):
                candidate = root / str(step)
                candidate.mkdir()
                for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt'):
                    (candidate / name).write_bytes(f'{step}-{name}'.encode())
                ranked.write(candidate / 'RUNNING_STATS.json', {'skill_ratings': {m: 1700.0 for m in MODES}})
                self.assertTrue(ranked.add_policy_to_pool(candidate, pool, 6))
                self.assertEqual(len(ranked.policy_versions(pool)), min(4 + (step - 400) // 100, 6))
            # The pool is full at six, so the seventh member already evicted the oldest one.
            self.assertEqual([int(p.name) for p in ranked.policy_versions(pool)], [200, 300, 400, 500, 600, 700])

            # ...and from then on FIFO eviction keeps it at the requested size.
            final = root / '800'
            final.mkdir()
            for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt'):
                (final / name).write_bytes(b'800')
            ranked.write(final / 'RUNNING_STATS.json', {'skill_ratings': {m: 1700.0 for m in MODES}})
            self.assertTrue(ranked.add_policy_to_pool(final, pool, 6))
            self.assertEqual([int(p.name) for p in ranked.policy_versions(pool)], [300, 400, 500, 600, 700, 800])

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
                    settings=dict(deterministic=False, policy_temperature=1.0, fuzzed_kickoffs=True),
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
                # Start states must vary even though this campaign never asked for it explicitly.
                self.assertTrue(all('--fuzzed-kickoffs' in cmd for cmd in training_commands))
                self.assertTrue(state['runtime_overrides']['fuzzed_kickoffs'])
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
                start = ranked.checkpoint_elo(candidate_path)
                if not start:
                    start = {m: 1500.0 for m in MODES}
                elo = {m: value + 10.0 for m, value in start.items()}
                ratings = {m: {'start': start[m], 'average': elo[m] - 25, 'end': elo[m]} for m in MODES}
                ratings['Ranking'] = {'start': sum(start.values()) / 3,
                                      'average': sum(elo.values()) / 3 - 25,
                                      'end': sum(elo.values()) / 3}
                result = dict(
                    anchored=True, seed=seed, rounds=rounds, checkpoint_timesteps=step,
                    settings={'deterministic': False, 'policy_temperature': 1.0, 'fuzzed_kickoffs': True},
                    ratings=ratings,
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
            # The accepted policy carries the Elo its own tournaments earned (screen 1700 then confirm 1710,
            #each +10), and its checkpoint keeps it for the next tournament.
            self.assertEqual(ranked.read(out / 'pool/120/STATS.json')['skill_ratings'],
                             {m: 1720.0 for m in MODES})
            self.assertEqual(ranked.checkpoint_elo(out / 'learner/120'), {m: 1720.0 for m in MODES})
            self.assertEqual({p.name for p in ranked.policy_versions(out / 'pool')},
                             {p.name for p in ranked.policy_versions(out / 'learner/policy_versions')})
            train_cmd = train_commands[0]
            self.assertIn('--fuzzed-kickoffs', train_cmd)
            self.assertTrue(state['runtime_overrides']['fuzzed_kickoffs'])
            self.assertEqual(train_cmd[train_cmd.index('--max-versions') + 1], '10')
            self.assertEqual(train_cmd[train_cmd.index('--ts-per-version') + 1], '9000000000000000')

    def test_rounds_for_pool_derives_a_count_every_pool_size_accepts(self):
        # The native anchored tournament only accepts a multiple of twice the opponent count, so
        # every pool size an evolving campaign can reach needs its own valid round count.
        for count in range(1, 16):
            rounds = ranked.rounds_for_pool(40, count)
            self.assertEqual(rounds % (2 * count), 0)
            self.assertGreaterEqual(rounds, 40)
            self.assertLess(rounds, 40 + 2 * count)
        # The reachable sizes of a 10 -> 15 pool, with the counts a 40-round budget derives.
        self.assertEqual([ranked.rounds_for_pool(40, n) for n in range(10, 16)],
                         [40, 44, 48, 52, 56, 60])
        # A budget below one full cycle still yields a complete, balanced set of fixtures.
        self.assertEqual(ranked.rounds_for_pool(1, 10), 20)
        self.assertEqual(ranked.rounds_for_pool(3, 1), 4)
        # The derivation is a pure function of budget and pool size, so a resumed campaign rates
        # exactly the same rounds as the run before it.
        self.assertEqual(ranked.rounds_for_pool(40, 11), ranked.rounds_for_pool(40, 11))
        with self.assertRaises(ValueError):
            ranked.rounds_for_pool(40, 0)
        with self.assertRaises(ValueError):
            ranked.rounds_for_pool(0, 10)

    def test_growing_pool_changes_the_rating_rounds_and_rerates_the_incumbent(self):
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
                base=base, out=out, bot=bot, minutes=2, pilot_minutes=1, block_minutes=1,
                rounds=40, threads=1, minibatch=1008, seed=1, audit_seed=9007,
                wandb_offline=False, wandb_online=False, wandb_bridge=False, github_backup=False,
                reward_metrics=False, player_metrics=False, throughput_profile=None, continuous=False,
                evolving_opponents=True, opponent_pool_size=15, promotion_margin=50.0,
                reference_pool=None, extra_versions=None, target_ranking=None, stop_on_target=False,
                rewards=None,
            )
            trainings, ratings = [], []

            def native(cmd, env, log, heartbeat=None):
                checkpoints = Path(cmd[cmd.index('--checkpoints') + 1])
                if cmd[0] == 'bash':
                    trainings.append(len(trainings))
                    parent = ranked.newest(checkpoints)
                    candidate = checkpoints / str(100 + 20 * len(trainings))
                    shutil.copytree(parent, candidate)
                    ranked.write(candidate / 'RUNNING_STATS.json', dict(
                        total_timesteps=int(candidate.name), mirror_x=-1,
                        skill_ratings={m: 1700 for m in MODES}))
                    return
                candidate_path = ranked.newest(Path(cmd[cmd.index('--checkpoints') + 1]))
                step = int(candidate_path.name)
                seed = int(cmd[cmd.index('--seed') + 1])
                rounds = int(cmd[cmd.index('--rate-rounds') + 1])
                pool = Path(cmd[cmd.index('--rate-versions') + 1])
                refs = {int(p.name): ranked.read(p / 'STATS.json')['skill_ratings']
                        for p in ranked.policy_versions(pool)}
                ratings.append((step, seed, rounds, len(refs)))
                start = ranked.checkpoint_elo(candidate_path) or {m: 1500.0 for m in MODES}
                elo = {m: value + 10.0 for m, value in start.items()}
                ratings_json = {m: {'start': start[m], 'average': elo[m] - 25, 'end': elo[m]} for m in MODES}
                ratings_json['Ranking'] = {'start': sum(start.values()) / 3,
                                           'average': sum(elo.values()) / 3 - 25,
                                           'end': sum(elo.values()) / 3}
                ranked.write(cmd[cmd.index('--rate-out') + 1], dict(
                    anchored=True, seed=seed, rounds=rounds, checkpoint_timesteps=step,
                    settings={'deterministic': False, 'policy_temperature': 1.0, 'fuzzed_kickoffs': True},
                    ratings=ratings_json,
                    opponents=[dict(timesteps=t, rounds=rounds // len(refs), ratings=r)
                               for t, r in refs.items()],
                    mode_goals={m: {'for': 3, 'against': 1} for m in MODES}))

            with patch.object(ranked, 'run', native), patch.object(ranked, 'REPO', root), \
                    contextlib.redirect_stdout(io.StringIO()):
                ranked.campaign(args)
            state = ranked.read(out / 'state.json')
            self.assertEqual(state['status'], 'complete')
            # The pool started with the ten historical policies and grew through the promotion.
            self.assertEqual(len(ranked.policy_versions(out / 'pool')), 11)
            # Every rating ran a round count the native tournament accepts for its pool: a
            # multiple of twice the opponent count, and never below the 40-round budget.
            for step, seed, rounds, pool_count in ratings:
                self.assertEqual(rounds % (2 * pool_count), 0)
                self.assertGreaterEqual(rounds, 40)
            # The ten-policy pool rates with 40 rounds; the eleven-policy pool with 44.
            self.assertEqual({rounds for _, _, rounds, count in ratings if count == 10}, {40, 80})
            self.assertEqual({rounds for _, _, rounds, count in ratings if count == 11}, {44, 88})
            # The incumbent was re-rated onto the new pool before the next candidate was rated.
            incumbent_reratings = [r for r in ratings if r[0] == 120]
            self.assertTrue(incumbent_reratings)
            self.assertEqual({rounds for _, _, rounds, _ in incumbent_reratings}, {40, 80, 44, 88})
            self.assertEqual(state['history'][-1]['eval_rounds'], 44)
            self.assertEqual(state['history'][-1]['eval_rounds_confirm'], 88)

    def test_tournament_elo_is_continuous_and_written_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'campaign'
            (out / 'eval').mkdir(parents=True)
            pool = out / 'pool'
            (pool / '100').mkdir(parents=True)
            ref = {m: 1500.0 for m in MODES}
            ranked.write(pool / '100/STATS.json', {'skill_ratings': ref})
            bot = root / 'GigaLearnBot'
            bot.touch()
            args = argparse.Namespace(bot=bot, out=out, threads=1, rounds=2)

            def make(path, elo):
                path.mkdir()
                for name in (*ranked.MODELS, *(k + '_OPTIM' for k in ranked.MODELS), 'SHARED_HEAD_SLOTS'):
                    (path / (name + '.lt')).write_bytes(b'weights')
                ranked.write(path / 'INFO.json', {'model': 's'})
                ranked.write(path / 'RUNNING_STATS.json', dict(total_timesteps=int(path.name), mirror_x=-1,
                                                               skill_ratings=elo))
            current, frozen = root / '1200', root / '1100'
            make(current, {m: 1500.0 + 10 * i for i, m in enumerate(MODES)})
            make(frozen, {m: 1400.0 for m in MODES})

            starts = []
            def native(cmd, env, log, heartbeat=None):
                ckpt = ranked.newest(Path(cmd[cmd.index('--checkpoints') + 1]))
                start = ranked.checkpoint_elo(ckpt)
                starts.append((int(ckpt.name), start))
                delta = 5.0
                ends = {m: start[m] + delta for m in MODES}
                ratings = {m: {'start': start[m], 'average': start[m] + delta / 2,
                               'average_rounds': 1, 'series': [start[m], ends[m]], 'end': ends[m]}
                           for m in MODES}
                ratings['Ranking'] = {'start': sum(start.values()) / 3, 'average': sum(ends.values()) / 3,
                                      'end': sum(ends.values()) / 3}
                rounds = int(cmd[cmd.index('--rate-rounds') + 1])
                ranked.write(cmd[cmd.index('--rate-out') + 1], dict(
                    anchored=True, seed=int(cmd[cmd.index('--seed') + 1]),
                    rounds=rounds, checkpoint_timesteps=int(ckpt.name),
                    settings=dict(deterministic=False, policy_temperature=1.0, fuzzed_kickoffs=True),
                    ratings=ratings,
                    opponents=[dict(timesteps=100, rounds=rounds, ratings=ref)],
                    mode_goals={m: {'for': 3, 'against': 1} for m in MODES}))

            digest, first = pool_digest(pool), {m: 1500.0 + 10 * i for i, m in enumerate(MODES)}
            ranking = lambda values: dict(values, Ranking=sum(values.values()) / len(MODES))
            with patch.object(ranked, 'run', native):
                screen = ranked.score(args, current, 1009, 2, digest, {}, root, write_back=True)
                self.assertEqual(screen, ranking({m: first[m] + 5 for m in MODES}))
                self.assertEqual(ranked.checkpoint_elo(current), {m: screen[m] for m in MODES})
                frozen_before = ranked.read(frozen / 'RUNNING_STATS.json')
                # The next tournament continues where this one ended, from the policy itself.
                confirm = ranked.score(args, current, 2003, 4, digest, {}, root, write_back=True)
                self.assertEqual(confirm, ranking({m: first[m] + 10 for m in MODES}))
                self.assertEqual(starts[-1], (1200, {m: screen[m] for m in MODES}))
                self.assertEqual(ranked.checkpoint_elo(current), {m: confirm[m] for m in MODES})
                # A cached tournament is not replayed, and does not move the Elo a second time.
                self.assertEqual(ranked.score(args, current, 2003, 4, digest, {}, root, write_back=True), confirm)
                self.assertEqual(ranked.checkpoint_elo(current), {m: confirm[m] for m in MODES})
                self.assertEqual(len(starts), 2)
                # Opponents are only ever read: evaluating one must not create or change its Elo.
                self.assertEqual(ranked.score(args, frozen, 1009, 2, digest, {}, root),
                                 ranking({m: 1405.0 for m in MODES}))
                self.assertEqual(ranked.read(frozen / 'RUNNING_STATS.json'), frozen_before)

                # A tournament that did not start from the policy's own Elo is a failure, not a rating.
                def detached(cmd, env, log, heartbeat=None):
                    native(cmd, env, log)
                    out_file = cmd[cmd.index('--rate-out') + 1]
                    result = ranked.read(Path(out_file))
                    for value in result['ratings'].values():
                        value['start'] = 1500.0
                    ranked.write(Path(out_file), result)
                with patch.object(ranked, 'run', detached):
                    with self.assertRaises(ValueError):
                        ranked.score(args, frozen, 4001, 2, digest, {}, root)

    def test_saved_evaluations_keep_current_elo_without_history(self):
        result = dict(ratings={m: {'start': 10.0, 'series': [1.0, 2.0], 'average': 1.5,
                                   'average_rounds': 2, 'end': value}
                               for m, value in zip((*MODES, 'Ranking'), (1500.0, 1500.0, 1500.0, 1500.0))})
        self.assertEqual(endpoint_ratings(result), {m: 1500.0 for m in (*MODES, 'Ranking')})
        # Already-compacted results are accepted unchanged, so cached evaluations stay reusable.
        compact = dict(result, ratings={m: 1500.0 for m in (*MODES, 'Ranking')})
        self.assertEqual(endpoint_ratings(compact), {m: 1500.0 for m in (*MODES, 'Ranking')})

    def test_eval_cache_stores_endpoint_elo_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, ckpt = root / 'campaign', root / '1200'
            (out / 'eval').mkdir(parents=True)
            ckpt.mkdir()
            for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt'):
                (ckpt / name).write_bytes(b'weights')
            ranked.write(ckpt / 'RUNNING_STATS.json',
                         dict(total_timesteps=1200, mirror_x=-1, skill_ratings={m: 1500.0 for m in MODES}))
            pool = out / 'pool'
            (pool / '100').mkdir(parents=True)
            ref = {m: 1500.0 for m in MODES}
            ranked.write(pool / '100/STATS.json', {'skill_ratings': ref})
            bot = root / 'GigaLearnBot'
            bot.touch()
            args = argparse.Namespace(bot=bot, out=out, threads=1, rounds=2)
            ends = {m: value for m, value in zip(MODES, (1600.0, 1610.0, 1620.0))}
            ends['Ranking'] = 1610.0

            def native(cmd, env, log, heartbeat=None):
                ranked.write(cmd[cmd.index('--rate-out') + 1], dict(
                    anchored=True, seed=1009, rounds=2, checkpoint_timesteps=1200,
                    settings=dict(deterministic=False, policy_temperature=1.0, fuzzed_kickoffs=True),
                    ratings={m: {'start': 1500.0 if m in MODES else 1500.0, 'series': [value - 10.0, value],
                                 'average': value - 5.0, 'average_rounds': 2, 'end': value}
                             for m, value in ends.items()},
                    opponents=[dict(timesteps=100, rounds=2, ratings=ref)],
                    mode_goals={m: {'for': 3, 'against': 1} for m in MODES}))

            with patch.object(ranked, 'run', native):
                ratings = ranked.score(args, ckpt, 1009, 2, pool_digest(pool), {}, root)
            self.assertEqual(ratings, ends)
            saved = ranked.read(out / 'eval/1200-1009-2.json')
            self.assertEqual(saved['ratings'], ends)
            self.assertTrue(all(isinstance(v, float) for v in saved['ratings'].values()))
            # Reusing the cache must not require the discarded rating series.
            with patch.object(ranked, 'run', side_effect=AssertionError('cache was not reused')):
                self.assertEqual(ranked.score(args, ckpt, 1009, 2, pool_digest(pool), {}, root), ends)
            # A result cached under an older protocol is rated again instead of failing validation.
            stale = ranked.read(out / 'eval/1200-1009-2.json')
            stale['settings'] = {'deterministic': False, 'policy_temperature': 1.0}
            ranked.write(out / 'eval/1200-1009-2.json', stale)
            calls = []
            def again(cmd, env, log, heartbeat=None):
                calls.append(cmd)
                native(cmd, env, log)
            with patch.object(ranked, 'run', again):
                self.assertEqual(ranked.score(args, ckpt, 1009, 2, pool_digest(pool), {}, root), ends)
            self.assertEqual(len(calls), 1)
            self.assertTrue(ranked.read(out / 'eval/1200-1009-2.json')['settings']['fuzzed_kickoffs'])

    def test_old_policy_versions_are_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, trainer = root / 'pool', root / 'learner/policy_versions'
            for step in (100, 200):
                version = pool / str(step)
                version.mkdir(parents=True)
                for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt', 'STATS.json'):
                    (version / name).write_bytes(f'{step}-{name}'.encode())
                ranked.write(version / 'STATS.json', {'skill_ratings': {m: 1500.0 for m in MODES}})
            manifest = frozen_versions(pool)
            shutil.copytree(pool, trainer)
            untouched = {p: (p.stat().st_ino, p.stat().st_mtime_ns)
                         for p in trainer.rglob('*') if p.is_file()}
            # Identical content: the trainer copy is left byte-for-byte and file-for-file alone.
            self.assertFalse(sync_policy_versions(pool, trainer))
            self.assertEqual(untouched, {p: (p.stat().st_ino, p.stat().st_mtime_ns)
                                         for p in trainer.rglob('*') if p.is_file()})
            self.assertEqual(verify_frozen_versions(pool, manifest), manifest)
            # FIFO eviction removes whole versions; surviving versions must still match.
            shutil.rmtree(pool / '100')
            self.assertEqual(set(verify_frozen_versions(pool, manifest)), {200})
            # A re-rated old STATS.json is exactly the mutation the campaign must reject.
            ranked.write(pool / '200/STATS.json', {'skill_ratings': {m: 1600.0 for m in MODES}})
            with self.assertRaises(RuntimeError):
                verify_frozen_versions(pool, manifest)

    def test_frozen_version_report_flags_a_mutated_old_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)
            pool = campaign / 'pool'
            trainer = campaign / 'learner/policy_versions'
            for root in (pool, trainer):
                for step in (100, 200):
                    version = root / str(step)
                    version.mkdir(parents=True)
                    for name in ('POLICY.lt', 'SHARED_HEAD.lt', 'SHARED_HEAD_SLOTS.lt', 'STATS.json'):
                        ranked.write(version / name, {'skill_ratings': {m: 1500.0 for m in MODES}}) \
                            if name.endswith('.json') else (version / name).write_bytes(f'{step}'.encode())
            manifest = frozen_versions(pool)
            ranked.write(campaign / 'state.json', {'frozen_versions': manifest})
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(frozen_report.report(campaign))
                # The trainer copy drifting from the pool is enough to fail the report.
                ranked.write(trainer / '200/STATS.json', {'skill_ratings': {m: 1600.0 for m in MODES}})
                self.assertFalse(frozen_report.report(campaign))
            # A whole-version eviction from both copies is not a mutation.
            ranked.write(trainer / '200/STATS.json', {'skill_ratings': {m: 1500.0 for m in MODES}})
            for root in (pool, trainer):
                shutil.rmtree(root / '100')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(frozen_report.report(campaign))


if __name__ == '__main__':
    unittest.main()
