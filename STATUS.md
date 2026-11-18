# Training status

## 2026-11-18: three-hour ranked campaign

The campaign starts from `3043214022`. Its objective is **Rating/Ranking**, the unweighted mean of the 1v1, 2v2 and 3v3 Elos against a frozen five-opponent pool. [Live campaign state](results/ranking_3043214022/campaign/state.json) · [Protocol, throughput benchmarks and resume command](RANKED_TRAINING.md).

**First completed training stage:** `3049891014` (+6,676,992 timesteps). Screen Ranking rose **1543.2 → 1549.8** and confirmation **1555.4 → 1560.0**. Both seeds favored the pilot in 1v1 and 3v3 but not in 2v2. The mean gain (+5.6 Elo) was below the predeclared 8-point promotion margin, so the parent remained selected after the pilot. Later stages train from the latest weights regardless; read `state.json` for current best and final audit, not the latest timestep alone.

**Second completed training stage:** `3079308486` (+29,417,472 timesteps in 30.43 native trainer minutes). Its screen Ranking was **1511.43**, down **31.76** from the selected parent on the same seed/pool: 1v1 **1426.39**, 2v2 **1612.60**, 3v3 **1495.29**. It failed the screen, so no confirmation was run and the parent remains selected. Stage 3 is running from the latest checkpoint; there is still no demonstrated strength improvement or completed held-out audit. The independent clean CPU build was stopped while compiling to prioritize campaign CPU; only its configure/embedded-Python checks are verified so far.

## 2026-11-18: continuation of the 1v1 / 2v2 / 3v3 bot

Resumed [`3035220582`](checkpoints/3035220582/) from `5a6b850` and saved [`3043214022`](checkpoints/3043214022/): **7,993,440 additional timesteps**, from iteration 112,600 to 112,757. The learner reported 7 min 53 s (16,915 steps/s) on two CPU threads. This is a resumable checkpoint, **not a demonstrated strength upgrade**.

### Head-to-head against the starting checkpoint

RocketSim kickoff games, sampled actions on both sides, 32 arenas with the new bot blue in half and orange in half. Each seed got 180 simulated minutes per mode. A goal or 10 seconds without a touch resets an arena. These are **goals for : against**, not match wins or competitive ranks.

| Mode | [Seed 2718](results/continuation_3035220582/head_to_head_2718.json) | [Seed 31415](results/continuation_3035220582/head_to_head_31415.json) | Combined | New bot's share |
|---|---:|---:|---:|---:|
| 1v1 | 142 : 165 | 128 : 156 | 270 : 321 | 45.7% |
| 2v2 | 107 : 114 | 115 : 99 | 222 : 213 | 51.0% |
| 3v3 | 58 : 69 | 65 : 66 | 123 : 135 | 47.7% |

The parent outscored the continuation in 1v1 on both seeds. The small 2v2 lead changed direction between seeds; in 3v3 the new bot finished slightly behind. Keep the parent as the 1v1 baseline. Neither checkpoint was tested against humans.

### What ran

- `train_1v1_2v2_3v3.sh`: model `s`, 294 inputs, left-mirrored view, unchanged rewards, 15 arenas of each size. PPO: 48k steps per iteration, 12k batch, one epoch, LR `1.2e-4`, entropy scale `0.009`, seed `20261118`. Self-play with a 15% chance per iteration of playing a saved policy.
- Training used a **copy** of the parent, including its Adam optimizers, running statistics and five-version opponent pool. Both original numbered checkpoints remain unchanged. The saved continuation includes the critic, policy, shared head, slot encoder and all three optimizer archives; only opponent **ratings**, not opponent weights, changed in `policy_versions/`.
- CPU backend of PyTorch 2.5.1 (from the Linux PyPI wheel in this sandbox), RocketSim 2.1.1, GCC 12.2, Python 3.11.2. No W&B: [local metrics](results/continuation_3035220582/training.jsonl) contain every tenth iteration and the final rating session. All 537,179 model parameters were finite, and weights changed in the policy, critic, shared layers and slot encoder. A fresh learner loaded the new checkpoint and all five versions without resetting a model or optimizer.

To continue from the **new** checkpoint after building, run `./train_1v1_2v2_3v3.sh --max-minutes 10`. The preset keeps the newest two checkpoints; use `--checkpoints <copy-of-checkpoints>` to preserve the originals.

For reference, this was the bounded training invocation, starting with the parent checkpoint and pool from `5a6b850`:

```bash
mkdir -p build/continuation/checkpoints
cp -a checkpoints/{3035220582,policy_versions} build/continuation/checkpoints/
GGL_WANDB=0 GGL_METRICS_DIR=build/continuation/metrics GGL_METRICS_EVERY=10 \
  bash train_1v1_2v2_3v3.sh --checkpoints build/continuation/checkpoints \
    --threads 2 --seed 20261118 --new-run --run-name s-continuation \
    --max-timesteps 3043220582 --max-minutes 8
```

The stop target is an absolute timestep count. The saved count was 6,560 short of the target: the stop callback counts both teams, while the learner credits only current-policy player steps in a vs-old iteration.

To repeat the evaluation of the saved checkpoint:

```bash
for seed in 2718 31415; do
  OMP_WAIT_POLICY=passive OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build/GigaLearnBot \
    --eval checkpoints/3043214022 --vs checkpoints/3035220582 \
    --eval-modes 1v1,2v2,3v3 --eval-arenas 32 --eval-minutes 180 --seed "$seed"
done
```

## Historical 1v1 + 2v2 run (different opponent pool)

Run: W&B `2bzqz9po` (project `rocket-league-ggl`). Settings: lr 1e-4, batch 6000 (set for policy update magnitude ~0.03), ts-per-itr 24k, rating sessions every 10 min, 2 checkpoints kept.

Frozen-pool evaluation (150 rounds, `tools/compare_checkpoints.sh`):

| checkpoint (steps) | batch | 1v1 | 2v2 | ranking |
|---|---|---|---|---|
| 1,375M | 24k | 1281 | 1348 | 1325 |
| 1,387M | 24k | 1267 | 1338 | 1303 (60 rounds) |
| 1,540M | 6k  | 1241 | 1336 | 1289 |

Observation: ranking has not improved since the batch change to 6k (policy update magnitude target). Trade-off to decide: restore batch 24k and raise lr to reach a similar update magnitude (untested), or keep 6k.
