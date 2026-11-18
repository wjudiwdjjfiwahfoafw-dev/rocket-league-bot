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
		//	every updateIntervalMinutes minutes of run time. The reported rating is then the average after each of the
		//	last three quarters of the rounds (the first quarter lets the rating catch up with what was learned since
		//	the last session), and the current rating is set to this average.
		// New policy versions are then saved right after a session (the first session after tsPerVersion timesteps
		//	since the newest version), so that every version starts with an up-to-date rating.
		// How many rounds: the rating follows a change of skill over about 7 rounds of 32 games (at 5 points per goal),
		//	so a session should not be much shorter. More rounds make the reported rating only slowly more exact
		//	(measured: +-17 points with 10 rounds, +-16 with 20, +-12 with 60), so short sessions more often
		//	tell more for the same cost than long ones rarely.
		float updateIntervalMinutes = 0;
		int sessionRounds = 10;

		// Rating increment scale per-goal
		float ratingInc = 5; 

		// Initial rating of the first version
		float initialRating = 0; 

		// Policies will be inferred deterministically
		// Off by default since the learning algorithm is trying to optimize the stochastic policy
		bool deterministic = false;
	};
}