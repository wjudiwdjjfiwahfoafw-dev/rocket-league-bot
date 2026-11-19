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
- **Fixtures:** Balanced across all opponents and both team colors, 36 arenas (12 per mode), `K=5` per goal (only candidate Elo updates); reported rating averages the final 75% of rounds.
- **Promotion gate:** Screen on seed `1009` (`40` rounds); if above current best screen, confirm on seed `2003` (`80` rounds). Promotes only when both matching-seed scores improve and their mean gain is at least `+8.0` Elo.
- **Held-out audit:** Evaluated once at campaign completion (`80` rounds on the predeclared `--audit-seed`, never used for selection).
- **Cache integrity:** Cached `eval/*.json` entries fingerprint `POLICY.lt`, `SHARED_HEAD.lt`, and `SHARED_HEAD_SLOTS.lt` by SHA-256 and validate pool digest and temperature-1 sampled execution.

## Evolving old-policy pool

`--evolving-opponents` migrates/resumes a campaign with a single shared set of 10 historical policies for both `--train-vs-old`/skill matches and anchored evaluation. The initial set is sampled evenly across all available prior policy-version timestamps so it spans the historical range. Native automatic policy saving is effectively disabled; only the campaign's promotion gate changes the set.

After each training block, the candidate is rated against all ten opponents on the screen and confirmation seeds. It is added only if its raw endpoint `Ranking` is at least 50 Elo above the strongest opponent on both evaluations. On promotion it enters the set and the oldest timestamp is evicted. The incumbent is re-rated whenever the pool changes so comparisons stay on the same pool-specific scale. Pool membership, digest, generation, and strongest-opponent margin are recorded in campaign history. The threshold and pool size are configurable with `--promotion-margin` and `--opponent-pool-size`.

In this mode, GitHub `training-backup/` stores only the latest full model checkpoint, the exact old-policy set under `learner/policy_versions/`, and the small state/manifest metadata needed to verify and resume; it does not duplicate the pool, best model, evaluation cache, or training metric logs. Local campaign history/metrics remain in `results/...`.

The reported Elo values use the exact final rating endpoint from a match/session, not a cross-round smoothed or interpolated series. W&B logs each rating event as a discrete sample (`Elo/...` for the native tracker and `Rating/...` for the anchored candidate evaluation); other sparse scalar diagnostics may still be window-averaged.

Inspect saved ratings anytime with:

```bash
python3 tools/show_rating.py training-backup
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

## Regression tests

```bash
python3 -m unittest discover -s tools -p 'test_*.py' -v

# After building build/GigaLearnBot:
build/GigaLearnTests
build/GigaLearnMultiModeTests
python3 tools/test_native_rollout.py
python3 tools/test_native_render.py
```
