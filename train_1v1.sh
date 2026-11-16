#!/usr/bin/env bash
# Trains the bot for 1v1 on a CPU and rates it with GigaLearn's skill tracker (the 12-hour run of ELO_REPORT.md).
# For 1v1 and 2v2 with one bot, see train_1v1_2v2.sh.
#
# Usage:   ./train_1v1.sh [extra GigaLearnBot options]
# Example: ./train_1v1.sh --max-minutes 120                     # train for 2 hours, then save and stop
#          ./train_1v1.sh --wandb --wandb-project my-project    # also log to Weights & Biases, as the run "s-1v1"
#                                                               # (needs WANDB_API_KEY or `wandb login`)
#
# Running it again continues from the newest checkpoint in ./checkpoints_1v1, with the same rating and wandb run.
# Stop it at any time with Ctrl+C (or `kill <pid>`): it finishes the current iteration, saves, and exits.
#
# The model:
#   --model s-1v1         the small network "s" in its 1v1-only form (439k parameters): without the parts that only
#                         games with teammates or several opponents use, which makes it about 10% faster.
#                         It loads the checkpoints of "s" and plays 1v1 the same. The way back is
#                         GigaLearnBot --expand-1v1 (see "Adding 2v2" in ELO_REPORT.md)
#   --modes 1v1           only 1v1 games
#   --mirror left         a car on the right half of the field sees the game mirrored (and its action is mirrored
#                         back), so the network only ever sees its own car on the left half and learns every
#                         situation once instead of twice. A checkpoint has to be continued with the view it has
#
# The learning settings (each is the result of a test that was judged by the rating, see ELO_REPORT.md):
#   --games 44, --ts-per-itr 24k, --epochs 1    as in train_multi.sh
#   --batch-size 6000     one policy update per 6,000 timesteps, 4 per iteration
#   --lr 1e-4             learning rate. Twice as much (2e-4) was worse, four times as much clearly worse
#   --entropy-scale 0.009 how much the bot is paid for acting randomly. Twice as much (0.018) was worse
#   --rewards VelocityBallToGoal=8,Goal=300,KickoffProximity=3
#                         4 times the example's reward for moving the ball towards the opponent's goal, twice its
#                         reward for a goal, and a reward for being closer to the ball than the opponent at a kickoff.
#                         The other rewards are the example's
#   --train-vs-old 0.15   in 15% of the iterations the bot trains against a saved older version instead of itself
#
# The last two keep the kickoff alive. Without them the bot stopped going for the ball at kickoffs after 130 million
# timesteps: its copy on the other side did the same, so nothing in training punished it, but every older version
# then won the kickoffs.
#
# These are settings for a bot that already plays. The run of ELO_REPORT.md started with smaller updates, a higher
# learning rate, more entropy and the example's rewards
# (--batch-size 1500 --lr 2e-4 --entropy-scale 0.018 --rewards VelocityBallToGoal=2,Goal=150 --train-vs-old 0),
# and changed one setting after the other when the rating stopped rising. I have not trained a bot from scratch
# with the settings here.
#
# The start of the episodes:
#   --random-state-prob 0.5   half of the episodes start with a kickoff, half with ball and cars at random positions
#   --random-ball-speed 2000  ... with the ball moving at up to 2000 uu/s. (A ball without speed hangs in the air.)
#
# The rating (GigaLearn's policy versions and skill tracker, see "How the rating works" in ELO_REPORT.md):
#   --skill-tracker       the bot plays rating games against saved older versions of itself.
#                         Every goal moves its Elo rating, the metric "Rating/1v1"
#   --skill-minutes 10    the rating games are played as one short session every 10 minutes, and training runs
#                         undisturbed in between. (Without this option: one round every --skill-interval iterations)
#   --skill-rounds 10, --skill-arenas 32
#                         a session is 10 rounds of 32 games at once. That takes 11 to 15 seconds (2% of the run
#                         time) and rates the bot to about +-17 points. Longer sessions are only slowly more exact
#                         (+-13 with 60 rounds), so short ones buy the most looks at the bot for the time spent
#   --ts-per-version 25m, --max-versions 10
#                         after a session, a copy of the policy is saved as a new version with the rating it got,
#                         once 25 million timesteps have passed since the last one (about every third session on
#                         a CPU with 2 threads). The 10 newest are kept in checkpoints_1v1/policy_versions as opponents
#
# Checkpoints:
#   --ts-per-save 10m, --keep-checkpoints 2
#                         a checkpoint every 10 million timesteps and after every rating session. The newest 2 are
#                         kept, with the 10 policy versions about 25 MB. (More checkpoints give more to go back to
#                         if the rating slides, at 5 MB each. Mind the size limit of your workspace.)
#
# Weights & Biases (only used with --wandb):
#   --wandb-group training-1v1, --run-name s-1v1    the group and name of a new run.
#                         (A run that is continued from a checkpoint keeps its wandb run and its name)
#
# Speed on a CPU with 2 threads: 18,200 timesteps per second over the 12-hour run, including the rating sessions.
# Check the line "CPU threads: ... for learning" at the start of the log: the number is measured at the start, and
# comes out too low if another program is busy at that moment. --threads 2 (or your number of CPU cores) sets it.
#
# BUILD_DIR selects the build folder if you did not use the default ./build.
set -euo pipefail
cd "$(dirname "$0")"

BOT="${BUILD_DIR:-build}/GigaLearnBot"
if [ ! -x "$BOT" ]; then
	echo "$BOT does not exist yet. Run ./setup_linux.sh first (it downloads the dependencies and builds)." >&2
	exit 1
fi

# Don't let idle libtorch worker threads spin, they would steal CPU time from the game simulation threads
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-passive}"

# Where the metrics receiver writes its .jsonl files
export GGL_METRICS_DIR="${GGL_METRICS_DIR:-metrics_1v1}"
# ... and how often: every 20th iteration, and every one with a rating (about 0.3 MB per hour instead of 5).
# wandb gets every iteration
export GGL_METRICS_EVERY="${GGL_METRICS_EVERY:-20}"

exec "$BOT" \
	--model s-1v1 \
	--modes 1v1 \
	--mirror left \
	--checkpoints checkpoints_1v1 \
	--device cpu \
	--games 44 \
	--ts-per-itr 24k \
	--batch-size 6000 \
	--epochs 1 \
	--lr 1e-4 \
	--entropy-scale 0.009 \
	--random-state-prob 0.5 \
	--random-ball-speed 2000 \
	--rewards VelocityBallToGoal=8,Goal=300,KickoffProximity=3 \
	--train-vs-old 0.15 \
	--ts-per-save 10m \
	--keep-checkpoints 2 \
	--skill-tracker \
	--skill-minutes 10 \
	--skill-rounds 10 \
	--skill-arenas 32 \
	--ts-per-version 25m \
	--max-versions 10 \
	--wandb-group training-1v1 \
	--run-name s-1v1 \
	"$@"
