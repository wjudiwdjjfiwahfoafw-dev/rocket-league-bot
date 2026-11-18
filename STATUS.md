# Training status — active native-weight 1700 attempt

On the new working branch, training was resumed without calibration. The remainder of stage 5 produced checkpoint `3165722310` with **1546.65 screen Ranking** (1459.52 / 1619.35 / 1561.07), below the selected PPO checkpoint. No confirmation or promotion occurred. The old continuation was paused early in stage 6 and backed up before starting the new recipe.

A fresh goal-focused trial is running from native PPO `3137998278`, using sampled actions at **temperature 1**, unchanged entropy coefficient, the same frozen pool, and predeclared held-out seed **9001**. The first fresh-build baseline screen is **1575.92**; compare candidates against this run's matching-seed baseline, not the earlier build's scores. No candidate from the new recipe has completed evaluation yet, and **1700 is not achieved**.

See [WEIGHTS_1700.md](WEIGHTS_1700.md) for the full recipe, reasoning and execution safeguards. The browser viewer and current-branch GitHub backups run with the driver. `training-backup/state.json` is the authoritative persisted campaign state; `python3 tools/resume_training.py` resumes its recorded working directory only when training is stopped.

## Historical results (not a current 1700 claim)

Native PPO `3137998278` previously measured 1570.05 screen / 1576.24 confirmation on the older build. The temperature-0.4 artifact measured 1608.18 / 1662.50 / 1659.84 with less-random sampled execution, **not learned-model improvement**; it is excluded from this attempt. Full evidence is retained in [TARGET_1600.md](TARGET_1600.md).
