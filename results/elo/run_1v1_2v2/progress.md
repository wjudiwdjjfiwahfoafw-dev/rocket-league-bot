# The 1v1 + 2v2 run (`s-1v1-2v2`, wandb run `2bzqz9po`): running

Written 13:35 (sandbox clock). 1,165,559,784 timesteps -> 1,323,114,387 timesteps in 2.35 hours of run time, 13 rating sessions.

Speed over the last 400 iterations: 17704 timesteps per second by the clock.

| Run time | Timesteps | Rating/1v1 | Rating/2v2 | Goals 1v1 | Goals 2v2 | Kickoff first touches |
|---|---|---|---|---|---|---|
| first session (0.17 h) | 1177.1M | 1229 | 1264 | 166:162 | 93:60 | 57% |
| 1 h | 1234.6M | 1229 | 1246 | 441:443 | 208:201 | 54% |
| 2 h | 1304.0M | 1211 | 1271 | 400:446 | 232:170 | 58% |
| latest session (2.23 h) | 1315.5M | 1200 | 1276 | 138:152 | 84:58 | 61% |

The hourly rows are averages over three rating sessions (the one nearest to the hour and its two neighbours), and their goals are the sums of those three. A single session scatters by about 20 points in 1v1 and 30 in 2v2.
Goals are the bot's : the saved older version's. `Rating/2v2` started at the 1v1 rating, see ELO_REPORT.md.

