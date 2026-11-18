# Training status (1v1 + 2v2, one continuously trained model)

Run: W&B `2bzqz9po` (project `rocket-league-ggl`). Settings: lr 1e-4, batch 6000 (set for policy update magnitude ~0.03), ts-per-itr 24k, rating sessions every 10 min, 2 checkpoints kept.

Frozen-pool evaluation (150 rounds, `tools/compare_checkpoints.sh`):

| checkpoint (steps) | batch | 1v1 | 2v2 | ranking |
|---|---|---|---|---|
| 1,375M | 24k | 1281 | 1348 | 1325 |
| 1,387M | 24k | 1267 | 1338 | 1303 (60 rounds) |
| 1,540M | 6k  | 1241 | 1336 | 1289 |

Observation: ranking has not improved since the batch change to 6k (policy update magnitude target). Trade-off to decide: restore batch 24k and raise lr to reach a similar update magnitude (untested), or keep 6k.
