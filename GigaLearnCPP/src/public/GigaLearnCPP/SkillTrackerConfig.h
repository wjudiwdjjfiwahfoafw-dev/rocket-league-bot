#pragma once

#include "Framework.h"

namespace GGL {
	struct SkillTrackerConfig {
		bool enabled = false;
		// Standalone evaluation only: common starting ratings, immutable opponent Elos, balanced fixtures.
		bool anchored = false;

		// Number of arenas for evaluation
		// Don't put this much higher than your CPU thread count
		int numArenas = 16;

		// Time (in seconds) to simulate each skill rating run, per arena
		float simTime = 45;

		// Maximum time (in seconds) to simulate games for (games are reset after the maximum time is exceeded)
		// If this max time is too low, the rating quality may be poor because the bots wont have enough time to score goals
		float maxSimTime = 240; 

		// Number of iterations between running skill rating games
		int updateInterval = 16;

		// Rating sessions, for runs where the rating games should cost as little run time as possible (0 = off):
		//	instead of one round of games every updateInterval iterations, sessionRounds rounds are played in a row
		//	every updateIntervalMinutes minutes of run time. The reported rating is the endpoint after the last round:
		//	the policy's own rating, which keeps accumulating from tournament to tournament instead of being reset.
		// New policy versions are then saved right after a session (the first session after tsPerVersion timesteps
		//	since the newest version), so that every version starts with an up-to-date rating.
		// How many rounds: the session's fixed budget is spread over its rounds, so a longer session moves the
		//	rating in smaller steps that follow a change of skill more smoothly. Short sessions are cheap and a
		//	long one reads the same budget more finely, so short sessions more often tell more for the same cost.
		float updateIntervalMinutes = 0;
		int sessionRounds = 10;

		// Maximum net rating change of the current policy, per mode, in one tournament: a rating session,
		//a --rate-rounds run, or one round of the older update-interval mode. The per-round increment is
		//budget / rounds, so the maximum does not depend on how many goals, games, arenas or players the
		//tournament has. Only the current policy's rating moves; saved versions never change.
		// The tournament estimate itself is low-noise (about +-2 points at this budget), so this is the rate at
		//which the rating follows real improvement: it closes ~14% of the gap to the true rating per tournament.
		float tournamentEloBudget = 100;

		// Initial rating of the first version
		float initialRating = 0; 

		// Policies will be inferred deterministically
		// Off by default since the learning algorithm is trying to optimize the stochastic policy
		bool deterministic = false;
	};
}