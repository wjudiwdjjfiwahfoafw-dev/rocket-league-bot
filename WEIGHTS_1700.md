# Target: 1700 combined Elo from learned weights alone

This attempt uses the uncalibrated PPO trajectory and **sampled actions at
softmax temperature 1** in training and ranked evaluation. No output rescaling,
calibration utility, deterministic scoring, or calibrated learner warm-start is
used. The fixed five-opponent evaluation pool and screen/confirmation fixtures
remain unchanged. 1700 is a goal, not an achieved result.

## Baseline

- Latest saved native PPO weights: `3155997126` (partway through stage 5).
- Selected best native PPO weights: `3137998278`.
- Selected matching-seed screen/confirmation: **1570.05 / 1576.24**.
- The earlier temperature-0.4 sampled-action results are **excluded** from this attempt.

## Evaluation safeguards added

Each new native rating JSON explicitly records policy temperature and action
selection. The ranked driver rejects results unless actions were sampled and
temperature was exactly 1. Cache identity fingerprints the policy, shared head
and slot encoder, not just a timestep folder. Changed weights invalidate cached
scores; weights are checked again after evaluation. These checks prevent a
transformed policy's cached score from being reused for a different model.

## Predeclared experiment sequence

1. Resume and finish the original bounded PPO campaign, retaining its original
   settings and remaining stage budgets. This tests whether the existing upward
   trend continues. Screen/confirmation remain 1009/40 and 2003/80.
2. If the native best does not reach the target, start a **new** campaign from its
   learned weights, retaining optimizers. Keep the same evaluation pool but use
   a fresh held-out audit seed **9001**.
3. Follow-up hypothesis: the learner spends too much experience on random starts
   and dense auxiliary rewards relative to scoring. Test a kickoff-heavier,
   goal-focused curriculum for 120 trainer minutes: random starts 0.25, old-policy
   opponents 0.35, LR 8e-5, entropy coefficient unchanged at 0.009. Reward overrides:
   VelocityBallToGoal=8, Goal=600, VelocityPlayerToBall=0,
   NearestVelocityToBall=0, TeamSpacing=1, TouchAccel=5, BehindBall=0,
   FaceBall=0, StrongTouch=25, PickupBoost=2, SaveBoost=0.2, Bump=10, Demo=20.
   Changes are experimental, not established improvements. No reduction of
   temperature or entropy coefficient is used to inflate performance.
4. Promote only on positive gains on both matching seeds and >=8 mean Elo.
   The 1700 target requires selected native weights >=1700 on both screen and
   confirmation plus a held-out audit. Fixed-pool Elo does not imply a human rank.

Browser metrics and GitHub checkpoint backups run with training. Only the current
branch is pushed. `training-backup/` is the newest recoverable native campaign;
source weights for a fresh campaign are separately preserved under `checkpoints/`.

## Adaptive decision after the resumed stage-5 screen

The remaining 7.56 native trainer minutes of stage 5 produced `3165722310`:
1v1 1459.52, 2v2 1619.35, 3v3 1561.07, mean **1546.65**, below the selected
1570.05. No confirmation or promotion occurred. The original continuation was
paused early in stage 6 rather than spending another hour on that recipe.
The next experiment now starts immediately from selected native PPO `3137998278`
using `tools/pursue_weight_target.py --curriculum-only`; the reward/curriculum
settings above were specified before seeing this screen. This change in scheduling
is adaptive, not a predeclared stop rule. No held-out audit informed it.
