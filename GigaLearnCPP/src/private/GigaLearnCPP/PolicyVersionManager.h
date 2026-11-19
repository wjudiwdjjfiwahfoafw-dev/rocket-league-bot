#pragma once

#include "Util/Models.h"
#include <GigaLearnCPP/SkillTrackerConfig.h>
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/RenderSender.h>
#include <GigaLearnCPP/Util/Timer.h>

#include <nlohmann/json.hpp>

namespace GGL {

	struct SkillRating {
		std::map<std::string, float> data;

		static std::string GetModeName(const RLGC::GameState& state) {
			int playersOnTeams[2] = { 0, 0 };
			for (auto& player : state.players)
				playersOnTeams[(int)player.team]++;

			int minPlayersOnTeam = RS_MIN(playersOnTeams[0], playersOnTeams[1]);
			int maxPlayersOnTeam = RS_MAX(playersOnTeams[0], playersOnTeams[1]);

			std::string name = RS_STR(minPlayersOnTeam << "v" << maxPlayersOnTeam);
			return name;
		}

		float& GetRating(std::string name, float defaultRating) {
			if (data.contains(name)) {
				return data[name];
			} else {
				// A game mode that is rated for the first time starts at the rating of the modes that are rated already
				// (their average), not at the default. When a mode is added to the training of an existing bot, the current
				// policy and its saved versions then start the new mode as far apart as they are in the old ones, which is
				// the best guess there is. With all of them at the default, the first sessions would only sort that out.
				float startRating = defaultRating;
				if (!data.empty()) {
					startRating = 0;
					for (auto& pair : data)
						startRating += pair.second / data.size();
				}
				data[name] = startRating;
				return data[name];
			}
		}

		float& GetRating(const RLGC::GameState& state, float defaultRating) {
			return GetRating(GetModeName(state), defaultRating);
		}

		nlohmann::json ToJSON() {
			nlohmann::json j = {};
			for (auto pair : data)
				j[pair.first] = pair.second;
			return j;
		}

		void ReadFromJSON(const nlohmann::json& j) {
			data = {};
			for (auto& pair : j.items())
				data[pair.key()] = pair.value();
		}
	};

	struct PolicyVersion {
		uint64_t timesteps;
		ModelSet models;
		SkillRating ratings;

		// The mirrored view that this version was trained with, and has to play with: 0 = off, 1 = right, -1 = left
		// (see RLGC::ObsBuilder::mirrorXTeam)
		int mirrorX = 0;
	};

	struct PolicyVersionManager {
		std::vector<PolicyVersion> versions;
		std::filesystem::path saveFolder;

		// The mirrored view of the current policy (0 = off, 1 = right, -1 = left). New versions are copies of it and inherit this
		int curMirrorX = 0;
		int maxVersions;
		uint64_t tsPerVersion;

		//////////////////

		struct {
			SkillTrackerConfig config;

			RLGC::EnvSet* envSet;
			int curGoals = 0;

			bool doContinuation = false;
			int prevOldVersionIndex;
			Team prevNewTeam;
			float prevSimTime;

			int iterationsSinceRan = 0;

			// Per game: a kickoff is waiting for its first ball touch (for the kickoff metrics)
			std::vector<uint8_t> kickoffPending = {};

			// Rating sessions (SkillTrackerConfig::updateIntervalMinutes)
			Timer sessionTimer = {}; // Run time since the last session ended
			bool sessionJustRan = false; // Set by OnIteration(): a session ran in this iteration
			std::function<bool()> sessionAbortCheck = nullptr; // Returns true to stop a running session early

			SkillRating curRatings = {};

			// Per-tournament Elo update: the current policy may move by at most config.tournamentEloBudget per
			//mode, whatever the number of goals, so each round of the tournament applies budget / rounds.
			float curTournamentK = 0;
			std::map<std::string, std::pair<int, int>> roundGoals = {}; // This round: current policy / opponent goals, per mode
		} skill;

		RenderSender* renderSender;

		PolicyVersionManager(
			std::filesystem::path saveFolder, int maxVersions, uint64_t tsPerVersion,
			const SkillTrackerConfig& skillTrackerConfig, const RLGC::EnvSetConfig& envSetConfig,
			RenderSender* renderSender = NULL);

		// NOTE: Passed models should not be already cloned
		PolicyVersion& AddVersion(ModelSet modelsToClone, uint64_t timesteps);

		void SaveVersions();
		// allowNewer: Accept versions with more timesteps than the current model (for versions that come from another run)
		void LoadVersions(ModelSet modelsTemplate, uint64_t curTimesteps, bool allowNewer = false);

		void SortVersions();

		void PrepareAnchoredPool(uint32_t seed);

		// Starts a tournament (a rating session or a --rate-rounds run) of the given number of rounds:
		//the current policy keeps its Elo, and each round may move it by budget / rounds.
		void BeginTournament(int rounds);

		// Applies the round's aggregated result as one Elo update for the current policy. Saved versions never move.
		void ApplyRoundElo(const SkillRating& opponent);

		void RunSkillMatches(struct PPOLearner* ppo, Report& report, int fixture = -1);

		// Several rounds of skill matches in a row, see SkillTrackerConfig::updateIntervalMinutes
		void RunSkillSession(struct PPOLearner* ppo, Report& report);

		// lastIteration: The run ends after this iteration. With rating sessions, a final session is played then
		void OnIteration(struct PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps, bool lastIteration = false);

		void AddRunningStatsToJSON(nlohmann::json& json);
		void LoadRunningStatsFromJSON(const nlohmann::json& json);

		// TODO: Add deconstructor
	};
}