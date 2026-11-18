# Train and select on three-mode Elo

The [campaign state](results/ranking_3043214022/campaign/state.json) is the source of truth for **finished** stages, time spent training, scores and best checkpoint. `config.minutes` is a budget, not elapsed time. Standalone evaluation takes additional time. No human Rocket League ranks are inferred from these numbers.

## Run or resume

Build first with `./setup_linux.sh`. On a restricted Linux x86-64 / CPython 3.11 host without system development packages or the libtorch download, `bash tools/setup_pypi_cpu.sh` provides the PyPI-wheel CPU fallback used here: no sudo, ~880 MiB download cached outside the checkout, GCC and make required. It reuses installed Python development files or can build matching ones for Debian's Python 3.11.2. Then run:

```bash
python3 tools/train_ranked.py --base checkpoints/3043214022 \
  --out results/ranking_3043214022/campaign --minutes 180 --rounds 40 --wandb-offline
```

`--wandb-offline` requires the W&B Python package (tested with `wandb==0.19.11`) in the **driver's** Python, not the learner's embedded Python. Omit that flag if you only want JSON. The driver never requires a key. To upload a finished offline run, run `wandb login` on a machine with W&B access, then `wandb sync <offline-run-directory>` using `wandb_dir` in `state.json`. Keep credentials out of the repository. A resumed offline job opens a new W&B segment; the cumulative results are in the campaign directory.

To sync the pilot now rather than waiting for the running SDK to flush, a **closed and verified pilot snapshot** is in [`pilot-export/wandb/`](results/ranking_3043214022/pilot-export/wandb/). To make one later from the durable JSON, run `python3 tools/ranked_wandb.py results/ranking_3043214022/campaign --out /tmp/ranking-export`. Sync the resulting `offline-run-*` directory from your authenticated machine. A snapshot and the full campaign are separate W&B runs; you don't need to upload both.

The launch above uses a **6-minute pilot**, five 30-minute stages and one 24-minute stage. It copies the base checkpoint, its optimizer state and opponent pool; it never trains in `checkpoints/`. A file lock prevents two drivers from training in the same directory. Ctrl+C / SIGTERM the driver once to request a checkpoint and pause. After a stop, rerun the same command: it validates settings and the pool digest, skips finished stages, resumes the latest weights and uses the remaining budget. The job needs a running workspace or host; if suspended, rebuild if necessary and resume. Use a **new** `--out` for a different experiment.

To see which checkpoint is actually winning (including an in-progress run):

```bash
python3 tools/show_rating.py results/ranking_3043214022/campaign
```


Within the output directory:

| Path | What it means |
|---|---|
| `learner/` | Latest resumable run: two rolling full checkpoints plus live policy versions. Advances even if a noisy score drops. |
| `best/<steps>/` | Full checkpoint selected for play, separate from the latest training weights. The parent stays here until beaten. |
| `pool/` | Frozen reference versions **and their original per-mode Elos**. Never written by training. |
| `eval/<steps>-<seed>-<rounds>.json` | Standalone measurements: per-mode Elo, round series, goals and opponent counts. |
| `metrics/*.jsonl` | Every 20th training iteration and every live rating session. |
| `state.json` | Selection history; `best.steps` and `history[-1].steps` may differ. `pending.seconds` is active training time in the unfinished stage. |

## What is being optimized?

**Rating/Ranking = (1v1 Elo + 2v2 Elo + 3v3 Elo) / 3.** A 3v3 car, goal, or arena does not get extra weight in this score. Two-mode averages from earlier experiments are not comparable.

For selection, `--rate-anchored` runs the native skill matches against the same five policies for every checkpoint. Each mode starts at the mean of its reference opponents' ratings, not the candidate's live Elo. Only the candidate's Elo moves (K=5 per goal). The fixtures cycle through every opponent on both colors. There are 36 arenas, 12 per mode; each has its own seeded kickoff stream. Actions are sampled. The score averages the last three quarters of the rounds for each mode, then averages the three modes. The driver checks opponent coverage, opponent ratings, goals in every mode and the resulting mean before accepting an evaluation.

- **Screen** every stage on seed 1009 (40 rounds). A potential record gets a **confirmation** on seed 2003 (80 rounds). Promote only if both rankings beat the incumbent on the matching seed and the mean gain is at least 8 Elo. This is a noise guard, not a statistical guarantee.
- **Audit** the parent, best and latest on a third seed (4001, 80 rounds) at campaign end. This seed does not select the best.
- W&B `Rating/*` is the **anchored screen**, `Confirmation/*` the second measurement. `Live/Rating/*` is the native training tracker against the evolving training pool, not a promotion criterion. In the native JSONL, `Rating/*` retains its original live meaning.

## Training changes

- **Equal player experience:** `--mode-balance players --games 88` allocates 48 / 24 / 16 arenas to 1v1 / 2v2 / 3v3. Each supplies 96 of 288 player decisions per step (48 per mode when playing an older opponent). The old equal-arena preset supplied decisions in a 1:2:3 ratio.
- **On-policy rollout boundaries:** `--fixed-rollout` bootstraps unfinished trajectories with their own next-state value at each PPO update. It does not reset those games or turn truncations into goals. No unfinished trajectory straddles policy updates or a switch to an old opponent. GAE now consumes bootstrap values in the same reverse order in which it visits truncations; the previous forward indexing could use a different player's value.
- **Aligned batches:** 48,384 learner steps = 168 full self-play decisions (336 vs old) = four PPO batches of 12,096. Each batch splits evenly into twelve minibatches of 1,008. No backlog of incomplete episodes and no dropped batch remainder at these settings. LR `1e-4`, one epoch, entropy scale `0.009`; same model `s`, left mirror, rewards, starts and 15% old-opponent chance as the previous preset.
- **Cheaper CPU updates:** in fixed rollouts, actor and critic reuse the same shared features during collection; only bootstrap observations need a separate critic pass before GAE. Slot observations are written into one buffer without allocating a temporary vector per other car. Sampling computes log probabilities only for selected actions when needed. `--lean-metrics` keeps PPO, reward, timing and skill reports but skips per-player/reward-component diagnostics.

### Measured throughput on this sandbox

Same 2 CPU threads, PyTorch 2.5.1, parent weights, one epoch and all three modes. The [raw benchmarks](results/ranking_3043214022/benchmarks.jsonl) include the arguments and timing breakdown. Each measurement excludes one warm-up iteration, measures the next 12 and uses pure self-play (no rating sessions or historical opponents). The comparison is for the **whole setup**, not a claim about one code change.

| Setup | Experience rows delivered to PPO/s, two runs | Mean |
|---|---:|---:|
| Previous engine + 45 equal arenas | 16,682; 17,727 | 17,204 |
| Ranked preset + joint critic pass | 20,356; 20,148 | 20,252 |

About **18% more experience rows/s** in this short benchmark. The native live run rate also includes older opponents, ratings and saves. Neither speed nor higher reward proves higher Elo; use the anchored results above. The learner-math and multi-mode test binaries cover the GAE truncation order, joint vs separate critic predictions, observations, gradients and checkpoint compatibility.

## Sandbox-only W&B reporting: blocked

Training and offline W&B recording run entirely in this sandbox. The user requires
no external uploader or local-PC commands. The proposed relay was stopped and
removed. Live website reporting is not working: this environment allows outbound
access only to GitHub and Python/npm package hosts, not W&B. The direct
`https://api.wandb.ai/graphql` probe failed during TLS connection. W&B credentials
are also absent from the current environment. Platform/network configuration must
permit W&B API/upload traffic and securely provision W&B authentication before
online reporting can work here. Offline recording is not a live website upload.

### Implemented online streamer

`tools/train_ranked.py --wandb-online` now starts a sandbox-local W&B uploader
alongside the native learner. It tails complete JSONL records every 10 seconds,
backfills existing metrics, uses a stable resumable W&B run ID, and reports screen,
confirmation and held-out results as they become available. It does not require
restarting training to attach it to an already-running campaign:

```bash
python3 tools/ranked_wandb.py results/ranking_3043214022/campaign \
  --out results/ranking_3043214022/campaign/online-stream --online
```

This is a real online SDK path, not an offline export. A live attempt in this
sandbox failed with `api_key not configured (no-tty)`; there is no verified online
run URL. Network access to W&B is still required after authentication is configured.
An uploader failure does not stop training or delete its durable JSON metrics.
The existing offline campaign need not be interrupted or duplicated to use this.

## Browser metrics and automatic GitHub backups (sandbox only)

For a new or resumed campaign, add `--monitor --github-backup` to the training
command. The driver starts the service automatically; no PC-side process or
external CDN is required. Open the **Training metrics and GitHub backups** preview.
Every numeric field actually recorded in the trainer's JSONL is selectable and
plotted against timesteps. Anchored screen/confirmation scores are separate from
live moving-pool ratings. This is not the W&B website and does not reproduce
unrecorded diagnostics or all W&B product features.

For the already-running campaign, attach without restarting the learner:

```bash
python3 tools/training_service.py results/ranking_3043214022/campaign --backup
```

Every minute the worker checks for a complete, settled checkpoint. When changed,
it copies the latest full checkpoint (including optimizer and running statistics),
selected best, mutable opponent pool, frozen evaluation pool, campaign state,
evaluations and metrics into `training-backup/`, commits **only that directory**,
and pushes **the current branch**. The UI reports success only after GitHub's
remote branch hash is checked. Failed pushes are retried; no local commit alone
is called an off-machine backup. SHA256 hashes are in `manifest.json`.

The repository is public: these snapshots and metrics will be public too. Git
history retains old versions; this is appropriate for this small CPU experiment,
not unlimited multi-GB checkpoints. No credentials or W&B diagnostics are copied.
An abrupt termination loses unsaved steps and anything since the most recent
successful remote snapshot. The viewer is live only while the sandbox runs;
pushed snapshots remain on GitHub afterward.

### Recovery after sandbox loss

Clone this same branch and build the trainer. Verify the backup manifest hashes.
Copy `training-backup/` to a new campaign directory outside `checkpoints/`, remove
its `manifest.json`, then resume with the original configuration and the copied
campaign as `--out`. The base `checkpoints/3043214022` is included on this branch.
The backup's pending stage is reset conservatively to zero elapsed minutes, so a
resumed stage may exceed the original time budget but does not skip unbacked
training. The metrics/evaluations are historical records; resumable weights are
exactly the manifest checkpoint, not the newest heartbeat timestep. Do not resume
from `best/` if you intend to continue the latest training trajectory.

## Current strength target: 1600 combined Elo

Target the unweighted three-mode Ranking against the unchanged frozen pool.
The viewer distinguishes a selected model clearing 1600 on both screen and
confirmation from a **verified** target, which additionally requires its held-out
seed-4001 audit to reach 1600. No reward, throughput or moving-pool live-rating
threshold substitutes for this test. This is an experimental target, not a promise
of improvement or a human Rocket League rank. The existing promotion margin stays
unchanged. Stage 3 recovered to 1542.59 screen Ranking but did not beat the parent;
stage 4 continues the current recipe while its effects are measured.

### Reproducible follow-up recipes and policy calibration

Fresh experiments can pass `--lr`, `--entropy-scale`, `--random-state-prob`,
`--train-vs-old` and `--rewards` to `tools/train_ranked.py`. Overrides become part
of the immutable campaign configuration; they cannot silently change a resumed
experiment. Unspecified options preserve the original campaign configuration.

A separate target-1600 experiment tests action-distribution calibration from the
original `3035220582` checkpoint. `tools/calibrate_policy.cpp` scales the copied
policy's final linear weight and bias by `1 / temperature`; evaluation still uses
the identical native masks, sampled actions, seeds, goal Elo and frozen pool.
The unmodified original is preserved. This is policy post-processing, **not** a
claim that PPO training improved. Candidate directories include source/output
SHA256 provenance, full copied model state and measurements. Optimizer archives
are copied from the source and have not been adapted to the transformed weights.

`tools/test_calibration_candidates.py` screens each temperature on seed 1009/40,
confirms screen improvements on 2003/80, and audits candidates reaching 1600 on
both against seed 4001/80. Candidate results do not overwrite the main PPO
campaign's selected checkpoint or masquerade as its training history. Changing
temperatures repeatedly based on scores is model selection, so the held-out test
must remain unused until a candidate passes the declared screen/confirmation.

Build the calibration utility with the same configured libtorch/ABI as the trainer:
`cmake --build build --target CalibratePolicy -j 1`. Then run, for example:

```bash
python3 tools/test_calibration_candidates.py --base checkpoints/3035220582 \
  --campaign results/ranking_3043214022/campaign \
  --out results/target1600-calibration --temperatures 0.65 0.4 1.3
```

The script stops when the target is verified. It reuses only the measurements
within each temperature's own directory, not a different policy's cached scores.
