# 1600 combined Elo: verified

On 2026-11-18, the calibrated original policy passed all three predeclared fixed-pool checks:

| Test | Seed / rounds | 1v1 | 2v2 | 3v3 | Combined Ranking |
|---|---|---:|---:|---:|---:|
| Screen | 1009 / 40 | 1464.26 | 1694.75 | 1665.54 | **1608.18** |
| Confirmation | 2003 / 80 | 1512.98 | 1751.54 | 1722.98 | **1662.50** |
| Held-out audit | 4001 / 80 | 1529.31 | 1757.19 | 1693.02 | **1659.84** |

Combined Ranking is the unweighted average of all three mode Elos. This does not
mean each mode is above 1600, nor does it establish a human competitive rank.
The source original scored 1556.84 screen and 1558.90 confirmation; matched gains
are **+51.35** and **+103.60**. The source was not independently audited on 4001,
so no matched held-out gain is claimed. Seed-to-seed variability is substantial.

## What improved

**Policy calibration**, not a claim that PPO alone reached 1600: the original
`3035220582` policy's last linear layer weight and bias were multiplied by 2.5
(equivalent to temperature 0.4), reducing action randomness. Critic, shared head,
slot encoder and other policy layers are unchanged. The native evaluator still
uses its original action masks, sampled actions, fixture schedule and Elo rules.
Two calibration temperatures were screened. Temperature 0.65 failed the 1600
screen; 0.4 passed screen and confirmation and was the first candidate audited.
No candidate was selected based on the held-out audit and no later temperature
was evaluated after success.

The same frozen five-policy pool was used throughout, with SHA256
`aa048b453da50d8c7c96913692d3e4d70bd141b5247cbd6519fe11589539d81c`.
Scores are specific to this pool and protocol, not universal Elo ratings.

PPO training separately promoted checkpoint `3137998278` with **1570.05 screen /
1576.24 confirmation**, versus the campaign parent's 1543.18 / 1555.39. Its mean
matched gain was +23.86. The ongoing PPO campaign and its selected model remain
separate from the calibrated target-achieving policy.

## Saved model and evidence

- **Target-achieving policy directory:**
  [`results/target1600-calibration/temperature-0.4/weights/3035220582/`](results/target1600-calibration/temperature-0.4/weights/3035220582/).
- [Screen, confirmation and held-out audit JSON](results/target1600-calibration/temperature-0.4/eval/).
- [Manifest with per-file SHA256 and verified scores](results/target1600-calibration/temperature-0.4/manifest.json).
- [Source/output policy hashes and transformation](results/target1600-calibration/temperature-0.4/calibration.json).

All these artifacts are committed and pushed on branch
`arena/21280682-rocket-league-bot`. The directory retains the original timestep
number because calibration does not add training steps. Model files can be used
with the existing small (`s`), left-mirrored MultiModeObs configuration and all
three modes; no runtime temperature change is needed because calibration is baked
into the weights. The original optimizer archives are retained for completeness,
but **are not adapted to the transformed weights**; any future PPO fine-tuning
must be treated as a new experiment and reevaluated, not assumed to preserve strength.

The live metrics viewer displays the calibration results separately from PPO
training results. [Training protocol and automatic recovery](RANKED_TRAINING.md).
