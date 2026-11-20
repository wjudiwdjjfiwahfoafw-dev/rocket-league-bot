# Ranked Multi-Mode Training, W&B Telemetry, and Recovery

## Resuming and pausing

`training-backup/` holds the single canonical paused snapshot (verified by `training-backup/manifest.json`) and is never trained in directly.

```bash
# Restore into results/ without launching training:
python3 tools/resume_training.py --restore-only

# Verify and resume the finite training schedule (starts 30m GitHub auto-backup + live W&B publisher):
python3 tools/resume_training.py

# Resume one fixed 15-minute train/evaluate cycle and repeat it indefinitely with a rolling pool of 10 old policies:
python3 tools/resume_training.py --continuous --evolving-opponents

# Resume trainer only (no backup worker or W&B bridge):
python3 tools/resume_training.py --without-services
```

Send `SIGINT` (`Ctrl+C`) or `SIGTERM` once to pause cleanly: the native learner completes the current iteration, writes a full checkpoint, updates `state.json` to `paused`, and exits. In `--continuous` mode the original pilot/final-short-block schedule is ignored: one regular 15-minute training block is followed by evaluation, and the next identical cycle begins from the latest checkpoint. This repeats without a stage-count/time limit; the held-out audit is only part of a finite schedule.

To start a fresh campaign (for example, a 12-hour run = `720` minutes) from the paused checkpoint against the same frozen reference pool:

```bash
python3 tools/train_ranked.py \
  --base training-backup/learner/3250007238 \
  --reference-pool training-backup/pool \
  --out results/weights1700/campaign \
  --minutes 720 --pilot-minutes 15 --block-minutes 30 --rounds 40 \
  --target-ranking 1700 \
  --github-backup --backup-interval 1800 --wandb-bridge
```

## Fixed-pool evaluation protocol

`Rating/Ranking = (1v1 Elo + 2v2 Elo + 3v3 Elo) / 3` using native sampled actions at softmax temperature `1.0` (`deterministic=False`, `policy_temperature=1.0`).

- **Frozen opponent pool:** 5 immutable left-mirrored policies in `training-backup/pool/` (SHA-256 `aa048b453da50d8c7c96913692d3e4d70bd141b5247cbd6519fe11589539d81c`).
- **Fixtures:** Balanced across all opponents and both team colors, 36 arenas (12 per mode). The reported rating is the exact endpoint of the final rated round.
- **Tournament Elo:** One Elo update per round per mode, from that round's share of its goals:
  `delta = (C / rounds) * (goals_for / (goals_for + goals_against) - expected)`, `expected = 1 / (1 + 10^((opponent - current) / 400))`, `C = 100` (`--skill-tournament-elo-budget`). The sum over the rounds is at most `C` however many goals, arenas or players a tournament has, so there is no cap on top of the rating — only a fixed budget per tournament.
  Because the update is `C * 0.00144 * (true - current)`, `C` sets how fast the rating follows real improvement: `100` closes about 14 % of the gap per tournament (1/e in ~7 tournaments, 90 % of a 50-point gap in ~16) while a 40-round screen still estimates the change to only about +-2 points.
- **Promotion gate:** Screen on seed `1009` (`40` rounds); if above current best screen, confirm on seed `2003` (`80` rounds). Promotes only when both matching-seed scores improve and their mean gain is at least `+8.0` Elo.
- **Held-out audit:** Evaluated once at campaign completion (`80` rounds on the predeclared `--audit-seed`, never used for selection).
- **Cache integrity:** Cached `eval/*.json` entries fingerprint `POLICY.lt`, `SHARED_HEAD.lt`, and `SHARED_HEAD_SLOTS.lt` by SHA-256 and validate pool digest and temperature-1 sampled execution.
- **Current Elo only:** A saved result stores the endpoint Elo of each mode (`1v1`, `2v2`, `3v3`, `Ranking`), plus the Elo the tournament started from and the one written back into the policy so a cached result stays verifiable. The per-round rating series and averaged Elo are discarded, so checkpoint JSON stays a compact record of the current rating instead of a rating history.
- **Continuous rating:** The evaluated policy keeps the Elo it already carries — it never starts from the pool's mean or any other shared anchor. The tournament's endpoint is written back into that policy's own `RUNNING_STATS.json` (`skill_ratings`), so the next tournament starts where this one ended, and an accepted pool copy carries exactly the Elo it had when it was accepted. Opponents are only ever read.

### Start-state randomness

Sampled actions at temperature `1.0` are not the only source of variation: every round of a rating tournament also starts from its own randomized kickoff. Skill matches always install `RLGC::FuzzedKickoffState`, seeded per arena from the evaluation seed (`PrepareAnchoredPool`), and each arena re-draws from its own deterministic RNG stream on every reset — so rounds never replay the same start state, and the same seed still reproduces a rating exactly. Training does the same with `--fuzzed-kickoffs`, which `train_ranked.py` always appends and records as `runtime_overrides.fuzzed_kickoffs`. A rating whose `settings.fuzzed_kickoffs` is not `true` is rejected, so a recipe or protocol change cannot silently bring back repeated rounds.

## Reward presets

`--rewards` accepts either explicit `Name=weight` pairs or a named preset; every preset lists *all* weights, so a
preset never inherits a changed default. The resolved weights are logged (`Rewards (preset v4): ...`) and recorded in
the run config as `rewards`, `reward_weights`, and `reward_preset`.

- `v3` — the recipe the 66-cycle campaign was trained with: `Goal=750` plus mode-aware event rewards, every dense
  RLGym-style primitive off.
- `v4` — resumes from a `v3` checkpoint with the dense shaping that RLGym ships by default switched back on, which is
  the intended fix for the ~1795 Elo plateau: the policy was only paid for the outcome of a touch, never for the
  approach. `Air=0.05` and `FaceBall=0.1` pay for aerial play, `TouchAccel=10` for flicks and power shots,
  `TeamSpacing=0.75` for rotation, and `NearestVelocityToBall=0.5` for the nearest player of each team driving at the
  ball. Each is deliberately below the weight the `RewardDefs` table suggests (0.1 / 0.25 / 15 / 1 / 0), because those
  suggestions assume an episode with no event rewards, whereas here every term is layered on top of `Goal=750`.
  `NearestVelocityToBall` stays at half of RLGym's `1.0` since `DirectionalTouch=16` and `ObjectiveBallVel=10` already
  pay for ball advancement. `Goal` remains the anchor and the event rewards move only slightly (`GoalSpeed 120 -> 110`,
  `Save 28 -> 30`, `Shot 18 -> 20`, `ObjectiveBallVel 6.5 -> 10`, `TeamPressure 1.6 -> 2`, `SoloDiscipline 0.55 -> 0.7`,
  `DefensiveCover 0.45 -> 0.6`, `DoubleCommit 0.9 -> 1`, `PickupBoostZS 3.5 -> 4`, `DemoZS 25 -> 15`, `Bump 4 -> 3`).

- `v5` — `v4` without `FaceBall`. `FaceBallReward` is `rotMat.forward . normalize(ball - player)`, paid every step
  no matter what the car is doing, so it is a reward for *looking* at the ball: at weight 0.1 it paid about
  0.032 per step, i.e. ~19-22 per 600-step episode, more than a whole `Shot=20` event, and its cheapest optimum is
  hovering near the ball with the nose pointed at it. Measured on the 4.2B-step policy the raw term averaged 0.32
(on a fresh policy, 0.15), so the per-step figure is the weight times that fraction of steps.

Keep `Goal` first when adding a term: a dense reward that out-earns a goal over an episode makes hovering and ball
chasing optimal, which is exactly the plateau this preset set out to break.

## Evolving old-policy pool

`--evolving-opponents` migrates/resumes a campaign with a single shared set of 10 historical policies for both `--train-vs-old`/skill matches and anchored evaluation. The initial set is sampled evenly across all available prior policy-version timestamps so it spans the historical range. Native automatic policy saving is effectively disabled; only the campaign's promotion gate changes the set.

After each training block, the candidate is rated against all ten opponents on the screen and confirmation seeds. It is added only if its raw endpoint `Ranking` is at least 50 Elo above the strongest opponent on both evaluations. On promotion it enters the set and the oldest timestamp is evicted. The incumbent is re-rated whenever the pool changes so comparisons stay on the same pool-specific scale. Pool membership, digest, generation, and strongest-opponent margin are recorded in campaign history. The threshold and pool size are configurable with `--promotion-margin` and `--opponent-pool-size`.

### Immutable old policies

Every policy version accepted into the pool is an immutable snapshot: its weights and a small `STATS.json` holding only the Elo it had at acceptance (`skill_ratings`) plus the mirror flag are written exactly once, when the copy is created. This does not apply to the active learner — its weights keep training, and its own Elo is continuous: native sessions move it, and every anchored evaluation tournament writes its endpoint back into the learner's `RUNNING_STATS.json`. Only the accepted copy is frozen. Nothing rewrites an accepted copy afterwards — no Elo updates from later skill matches, no metadata refreshes — so old policy folders stay byte-identical and backup commits stop churning.

`train_ranked.py` pins a SHA-256 manifest of every version (`frozen_versions` in `state.json`), re-checks it at each cycle boundary, and compares the trainer's `learner/policy_versions/` copy against the pool after every training block. Any re-rated or re-annotated old version aborts the cycle instead of being silently accepted, and the GitHub backup worker refuses to publish a snapshot that no longer matches the pinned manifest. Identical copies are never recopied, and FIFO eviction only removes whole versions.

In this mode, GitHub `training-backup/` stores only the latest full model checkpoint, the exact old-policy set under `learner/policy_versions/`, and the small state/manifest metadata needed to verify and resume; it does not duplicate the pool, best model, evaluation cache, or training metric logs. Local campaign history/metrics remain in `results/...`.

The reported Elo values use the exact final rating endpoint from a match/session, not a cross-round smoothed or interpolated series. W&B logs each rating event as a discrete sample (`Elo/...` for the native tracker and `Rating/...` for the anchored candidate evaluation); other sparse scalar diagnostics may still be window-averaged.

Inspect saved ratings anytime with:

```bash
python3 tools/show_rating.py training-backup
```

Verify that the ten old policies are still byte-identical to their creation record (exit code `1` on any rewrite):

```bash
python3 tools/verify_frozen_versions.py results/weights1700-v3/campaign
```

## Throughput accounting (`Overall Steps/Second`)

`Overall Steps/Second` uses contiguous monotonic wall-clock intervals with zero excluded CPU or wall time:

- Inside `GigaLearnCPP` (`Learner.cpp` and `python_scripts/metric_receiver.py`), each emitted window spans from process start (first report) or the exact end of the previous emitted report through opponent switching, rollout collection, inference, trajectory packing, GAE, per-mode advantage normalization, PPO updates, skill-tracker rating matches, policy-version writes, checkpoint saves, and metric serialization/flushing (`Iteration Time` and `Overhead Time` are logged alongside `Collection Time` and `Consumption Time`).
- Across ranked campaign stages (`train_ranked.py` and `wandb_live_bridge.py`), stage `steps_per_second` and stage-boundary `Overall Steps/Second` also fold in subprocess startup/teardown, fixed-pool screen/confirmation evaluations, and checkpoint promotion copies.

## Automatic 30-minute GitHub backups

`--github-backup` runs `tools/training_service.py` as a headless background worker (no HTTP server or web UI). Every `--backup-interval` seconds (default `1800` = 30 minutes, respected across restarts via `manifest.json`), it stages the latest checkpoint and minimal campaign state. In evolving-pool mode it includes exactly the current `learner/policy_versions/` set (10 by default), excludes duplicate pool/best/eval/metrics copies, hashes the snapshot, and pushes a verified commit to the current branch. The legacy fixed-pool backup format is retained for campaigns that have not enabled evolving opponents.

## Weights & Biases reporting

- **Sandboxed / restricted network (`--wandb-bridge`):** `tools/wandb_live_bridge.py publish` writes compact compressed metric frames to a branch telemetry issue; `.github/workflows/wandb-live.yml` streams them live to online W&B (`step_metric='Total Timesteps'`), authenticating via repository secret `WANDB_API_KEY` or an ephemeral RSA-3072-OAEP-SHA256 handshake against `/dev/shm/.wandb_bridge_key` (never stored in Git or logs).
- **Direct network (`--wandb-online` / `--wandb-offline`):** `tools/ranked_wandb.py` streams directly to online W&B or exports an offline run when direct outbound W&B access is available.
- Reward-component diagnostics are sampled during training and appear under `Rewards/…`; use `python3 tools/resume_training.py --reward-metrics` to enable them on a resumed campaign without changing its frozen PPO/evaluation recipe. Lean-metrics campaigns can independently retain sampled per-player telemetry under `Player/…` with `python3 tools/resume_training.py --player-metrics`. Sparse JSONL/W&B output windows average ordinary numeric diagnostics across skipped iterations (while keeping cumulative counters and the timestep axis at their latest values); rating events are immediate and are never window-averaged. On this two-vCPU runner, `python3 tools/resume_training.py --throughput-profile cpu-20k` selects the fastest measured all-mode profile (64 games, one PPO epoch, one collection-inference thread); clean benchmarks peak near 19.5k overall SPS, though shared-host variability can lower that. The live run stores its delivery cursor in GitHub issue metadata, uses `Elo/…` for raw internal skill-tracker endpoints, and publishes neither `Live/…` nor `Stream/…` transport fields.

### Where the W&B API key lives

Three homes, best first:

1. **Repository secret `WANDB_API_KEY`** — Settings -> Secrets and variables -> Actions -> New repository secret.
   The relay then authenticates itself: `has_api_key` skips the sealed-key handshake, so no copy has to exist in the
   sandbox and nothing can be lost when the sandbox restarts. Setting it needs admin rights on the repository; a
   GitHub App integration token gets `403 Resource not accessible by integration` even though it can push.
2. **`.secrets/wandb_api_key`** in the workspace, mode `600` — this is where the publisher looks when `/dev/shm` is
   empty. `/.secrets/` is gitignored, so the automatic 30-minute backup worker (`git add -A`) can never commit it,
   and it survives with the workspace across sandbox restarts.
3. **`/dev/shm/.wandb_bridge_key`** — the default `--key-file`. It is tmpfs: wiped on restart, which is why the feed
   went dark after the last cleanup. Good for one session, not for storage.

Precedence is `WANDB_API_KEY` env, then `--key-file`, then `.secrets/wandb_api_key`. Never put a key in tracked files,
and never in a path the backup worker stages.

The repository secret is the configured path here: the relay reports `auth_mode=repo_secret` and the handshake never
runs, so a sandbox copy should stay deleted. Delete a key the moment it is rotated out — a revoked key left in
`/dev/shm` or `.secrets/` only makes the publisher answer a handshake with a credential that no longer works.

## Regression tests

```bash
python3 -m unittest discover -s tools -p 'test_*.py' -v

# After building build/GigaLearnBot:
build/GigaLearnTests
build/GigaLearnMultiModeTests
python3 tools/test_native_rollout.py
python3 tools/test_native_render.py
```
