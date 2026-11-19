#include "PolicyVersionManager.h"
#include <nlohmann/json.hpp>

#include <GigaLearnCPP/Util/Utils.h>
#include <GigaLearnCPP/Util/Timer.h>

#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>

#include <private/GigaLearnCPP/PPO/PPOLearner.h>

using namespace nlohmann;

GGL::PolicyVersionManager::PolicyVersionManager(
	std::filesystem::path saveFolder, int maxVersions, uint64_t tsPerVersion, 
	const SkillTrackerConfig& skillTrackerConfig, const RLGC::EnvSetConfig& envSetConfig, RenderSender* renderSender) : 
	saveFolder(saveFolder), maxVersions(maxVersions), tsPerVersion(tsPerVersion), 
	renderSender(renderSender) {

	skill.config = skillTrackerConfig;

	if (!std::filesystem::exists(saveFolder))
		std::filesystem::create_directories(saveFolder);

	if (skill.config.enabled) {
		RLGC::EnvSetConfig skillEnvSetConfig = envSetConfig;
		skillEnvSetConfig.numArenas = skill.config.numArenas;
		skill.envSet = new RLGC::EnvSet(skillEnvSetConfig);
		for (int i = 0; i < skill.envSet->arenas.size(); i++) {
			skill.envSet->rewards[i].clear();
			skill.envSet->stateSetters[i] = { new RLGC::FuzzedKickoffState() };
			skill.envSet->terminalConditions[i] = { new RLGC::GoalScoreCondition() };
		}
	} else {
		skill.envSet = NULL;
	}
}

GGL::PolicyVersion& GGL::PolicyVersionManager::AddVersion(ModelSet modelsToClone, uint64_t timesteps) {
	RG_NO_GRAD;

	auto models = modelsToClone.CloneAll();

	auto newVersion = PolicyVersion{
		timesteps,
		models
	};

	newVersion.ratings = skill.curRatings;
	newVersion.mirrorX = curMirrorX;

	versions.push_back(newVersion);

	SortVersions();

	// Remove old versions
	while (versions.size() > maxVersions) {
		auto& toRemove = versions[0];
		toRemove.models.Free();
		versions.erase(versions.begin());
	}

	return versions.back();
}

void GGL::PolicyVersionManager::SaveVersions() {
	RG_NO_GRAD;

	// Remove old saved versions
	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(saveFolder);

	for (int64_t savedTimesteps : allSavedTimesteps) {
		bool matchesVersion = false;
		for (auto& version : versions)
			matchesVersion |= (savedTimesteps == version.timesteps);

		if (matchesVersion) {
			// We want to keep this
			allSavedTimesteps.insert(savedTimesteps);
		} else {
			// Get rid of it
			std::filesystem::remove_all(saveFolder / std::to_string(savedTimesteps));
		}
	}

	for (auto& version : versions) {
		auto versionSaveFolder = saveFolder / std::to_string(version.timesteps);

		// A saved version is an immutable record of a trained policy: its weights and its STATS.json
		// (the current Elo at creation, with no rating history) are written exactly once, right here.
		// Later saves must never rewrite them, so the folder stays byte-identical for training,
		// evaluation, backups and external inspection. Ratings that move during skill matches stay
		// in memory for the running process and are intentionally not written back to old versions.
		if (!allSavedTimesteps.contains(version.timesteps)) {
			std::filesystem::create_directories(versionSaveFolder);
			version.models.Save(versionSaveFolder, false);

			auto jsonPath = versionSaveFolder / "STATS.json";
			std::ofstream fOut(jsonPath);
			RG_ASSERT(fOut.good());

			json j = {};
			j["skill_ratings"] = version.ratings.ToJSON();
			j["mirror_x"] = version.mirrorX;
			fOut << j.dump(4);

			allSavedTimesteps.insert(version.timesteps);
		}
	}
}

void GGL::PolicyVersionManager::LoadVersions(ModelSet modelsTemplate, uint64_t curTimesteps, bool allowNewer) {

	RG_NO_GRAD;

	RG_LOG("PolicyVersionManager::LoadVersions():");

	for (auto& version : versions)
		version.models.Free();
	versions.clear();

	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(saveFolder);

	for (int64_t savedTimesteps : allSavedTimesteps) {

		if (savedTimesteps > curTimesteps && !allowNewer) {
			RG_ERR_CLOSE(
				"Tried to load saved policy version that is newer than our current model (" << savedTimesteps << " > " << curTimesteps << ")!\n" <<
				"If you deleted some checkpoints, make sure to delete that far back in the saved policy versions as well");
		}
		auto path = saveFolder / std::to_string(savedTimesteps);
		PolicyVersion& version = AddVersion(modelsTemplate, savedTimesteps);
		version.models.Load(path, false, false);

		{ // Load JSON
			// TODO: Repetitive
			auto jsonPath = path / "STATS.json";
			std::ifstream fIn(jsonPath);
			RG_ASSERT(fIn.good());

			json j = json::parse(fIn);
			if (j.contains("skill_ratings"))
				version.ratings.ReadFromJSON(j["skill_ratings"]);

			// Versions from before the mirrored view existed don't have the entry, and were trained without it
			version.mirrorX = j.value("mirror_x", 0);
		}
	}

	SortVersions();

	RG_LOG(" > Loaded " << versions.size() << " versions(s)");
}

void GGL::PolicyVersionManager::SortVersions() {
	auto fnCompareVersions = [](const PolicyVersion& a, const PolicyVersion& b) {
		return a.timesteps < b.timesteps;
	};

	std::sort(versions.begin(), versions.end(), fnCompareVersions);
}

/////////////////////////////////////////////////////////////////////

void GGL::PolicyVersionManager::PrepareAnchoredPool(uint32_t seed) {
	if (versions.empty()) RG_ERR_CLOSE("Cannot rate against an empty anchored pool");

	// The current policy continues with the Elo it has already earned. A tournament never resets the rating
	//to a common starting point: the opponents are anchored, the current policy is not.
	for (size_t i = 0; i < skill.envSet->arenas.size(); ++i) {
		auto setter = static_cast<RLGC::FuzzedKickoffState*>(skill.envSet->stateSetters[i]);
		setter->SetSeed(seed + (uint32_t)i + 1);
		auto name = SkillRating::GetModeName(skill.envSet->state.gameStates[i]);
		for (const auto& v : versions) {
			if (!v.ratings.data.contains(name))
				RG_ERR_CLOSE("Anchored opponent " << v.timesteps << " has no " << name << " rating");
		}
		// Materialize a mode the checkpoint has not been rated in yet (same rule as every other first rating).
		skill.curRatings.GetRating(name, skill.config.initialRating);
	}

	{
		std::stringstream line;
		bool first = true;
		for (auto& pair : skill.curRatings.data) {
			line << (first ? "" : ", ") << pair.first << " " << pair.second;
			first = false;
		}
		RG_LOG("Anchored pool: the current policy continues at " << line.str());
	}
}

void GGL::PolicyVersionManager::BeginTournament(int rounds) {
	// Spread one constant Elo budget over the tournament's rounds, so its maximum effect does not depend on
	//how many goals or games those rounds produce.
	int n = RS_MAX(rounds, 1);
	skill.curTournamentK = skill.config.tournamentEloBudget / (float)n;
	skill.roundGoals.clear();
}

void GGL::PolicyVersionManager::ApplyRoundElo(const SkillRating& opponent) {
	for (auto& pair : skill.roundGoals) {
		auto& mode = pair.first;
		int goalsFor = pair.second.first, goalsAgainst = pair.second.second;
		int total = goalsFor + goalsAgainst;
		if (total <= 0)
			continue; // No goals in this mode this round: nothing was learned, so nothing moves.

		auto opponentRating = opponent.data.find(mode);
		if (opponentRating == opponent.data.end())
			continue;
		float& rating = skill.curRatings.GetRating(mode, skill.config.initialRating);

		// One Elo update from the round's aggregated result: the share of goals won against the expectation.
		float expected = 1 / (powf(10, (opponentRating->second - rating) / 400) + 1);
		float observed = (float)goalsFor / (float)total;
		rating += skill.curTournamentK * (observed - expected);
	}
	skill.roundGoals.clear();
}

void GGL::PolicyVersionManager::RunSkillMatches(PPOLearner* ppo, Report& report, int fixture) {
	RG_NO_GRAD;
	

	
	Team newTeam;
	int oldVersionIndex;
	float totalSimTime;
	if (fixture >= 0) {
		skill.doContinuation = false;
		skill.curGoals = 0;
	}
	if (skill.doContinuation) {
		RG_ASSERT(skill.prevOldVersionIndex < versions.size());
		oldVersionIndex = skill.prevOldVersionIndex;
		newTeam = skill.prevNewTeam;
		totalSimTime = skill.prevSimTime;
	} else {
		if (fixture >= 0) {
			oldVersionIndex = fixture % versions.size();
			newTeam = (Team)((oldVersionIndex + fixture / versions.size()) % 2);
		} else {
			oldVersionIndex = Math::RandInt(0, versions.size());
			newTeam = (Team)Math::RandInt(0, 2);
		}
		totalSimTime = 0;

		// Every set of matches starts from kickoffs.
		// NOTE: EnvSet::Reset() only resets the arenas that are flagged as terminal, so they all get flagged here.
		//	Without this, the games of the previous set just went on with the new opponent (and, in the very first set,
		//	from whatever the training state setter had set up). Games in which both bots were stuck, which never end without
		//	a time limit, then stayed in the set of games for the rest of the run and no longer produced any goals.
		//	(SkillTrackerConfig::maxSimTime: "games are reset after the maximum time is exceeded")
		std::fill(skill.envSet->state.terminals.begin(), skill.envSet->state.terminals.end(), RLGC::TerminalType::NORMAL);
		skill.envSet->Reset();
	}
	skill.doContinuation = false;

	auto& oldVersion = versions[oldVersionIndex];

	{ // Each side sees the game the way its policy was trained
		int teamMirrorX[2];
		teamMirrorX[(int)newTeam] = curMirrorX;
		teamMirrorX[1 - (int)newTeam] = oldVersion.mirrorX;
		skill.envSet->SetMirrorX(teamMirrorX[0], teamMirrorX[1]);
	}

	// Find which players are on which teams
	std::vector<int>
		newPlayers = {}, 
		oldPlayers = {};
	for (int i = 0; i < skill.envSet->arenas.size(); i++) {
		auto& state = skill.envSet->state.gameStates[i];
		for (int j = 0; j < state.players.size(); j++) {
			int playerIdx = skill.envSet->state.arenaPlayerStartIdx[i] + j;
			bool isNew = (state.players[j].team == newTeam);
			(isNew ? newPlayers : oldPlayers).push_back(playerIdx);
		}
	}

	torch::Tensor
		tNewPlayers = torch::tensor(newPlayers),
		tOldPlayers = torch::tensor(oldPlayers);

	int newGoals = 0, oldGoals = 0;
	std::map<std::string, std::pair<int, int>> modeGoals = {}; // Per game mode: goals of the new policy, goals of the old version
	int newKickoffTouches = 0, oldKickoffTouches = 0;
	if (skill.kickoffPending.size() != skill.envSet->arenas.size())
		skill.kickoffPending.assign(skill.envSet->arenas.size(), 0);
	Timer matchTimer = {};
	uint64_t oldVersionTimesteps = oldVersion.timesteps;

	RG_LOG("Running skill matches (simTime=" << skill.config.simTime << ")...");

	SkillRating prevCurRatings = skill.curRatings;

	float stepTime = skill.envSet->config.tickSkip * RLGC::CommonValues::TICK_TIME;
	int stepsPlayed = 0;
	for (float t = 0; 
		t < skill.config.simTime && totalSimTime < skill.config.maxSimTime && skill.curGoals < skill.envSet->arenas.size();
		t += stepTime, totalSimTime += stepTime) {

		skill.envSet->Reset();

		torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(skill.envSet->state.obs);
		torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(skill.envSet->state.actionMasks);

		torch::Tensor tNewStates = tStates.index_select(0, tNewPlayers);
		torch::Tensor tOldStates = tStates.index_select(0, tOldPlayers);

		torch::Tensor tNewActionMasks = tActionMasks.index_select(0, tNewPlayers);
		torch::Tensor tOldActionMasks = tActionMasks.index_select(0, tOldPlayers);

		skill.envSet->StepFirstHalf(true);

		torch::Tensor tNewActions, tOldActions;

		PPOLearner::InferActionsFromModels(
			ppo->models, tNewStates.to(ppo->device, true), tNewActionMasks.to(ppo->device, true), 
			skill.config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision, 
			&tNewActions, nullptr);
		PPOLearner::InferActionsFromModels(
			oldVersion.models, tOldStates.to(ppo->device, true), tOldActionMasks.to(ppo->device, true), 
			skill.config.deterministic, ppo->config.policyTemperature, ppo->config.useHalfPrecision,
			&tOldActions, nullptr);

		auto newActions = TENSOR_TO_VEC<int>(tNewActions);
		auto oldActions = TENSOR_TO_VEC<int>(tOldActions);

		auto combinedActions = std::vector<int>(skill.envSet->state.numPlayers, -1);
		for (int i = 0; i < newActions.size(); i++)
			combinedActions[newPlayers[i]] = newActions[i];
		for (int i = 0; i < oldActions.size(); i++)
			combinedActions[oldPlayers[i]] = oldActions[i];

		skill.envSet->Sync();
		skill.envSet->StepSecondHalf(combinedActions, false);

		for (int i = 0; i < skill.envSet->arenas.size(); i++) {
			auto& gs = skill.envSet->state.gameStates[i];
			if (gs.goalScored) {
				std::string modeName = SkillRating::GetModeName(gs);

				if (RS_TEAM_FROM_Y(gs.ball.pos.y) != newTeam) {
					newGoals++;
					modeGoals[modeName].first++;
					skill.roundGoals[modeName].first++;
				} else {
					oldGoals++;
					modeGoals[modeName].second++;
					skill.roundGoals[modeName].second++;
				}

				skill.curGoals++;
			}

			// Kickoffs: which side touches the ball first.
			// A check on the kickoff play, which training against itself alone can lose without any other metric showing it
			if (gs.ball.pos.x == 0 && gs.ball.pos.y == 0 && gs.ball.vel.LengthSq() == 0) {
				skill.kickoffPending[i] = 1; // The ball lies at rest on the centre spot
			} else if (skill.kickoffPending[i]) {
				for (auto& player : gs.players)
					if (player.ballTouchedStep)
						((player.team == newTeam) ? newKickoffTouches : oldKickoffTouches)++;
				skill.kickoffPending[i] = 0;
			}
		}

		if (renderSender)
			renderSender->Send(skill.envSet->state.gameStates[0]);

		stepsPlayed++;
	}

	// A rated round (--rate-rounds) is complete after its sim time even when some arenas did not score;
	//during training a round without enough goals continues from the same position, so its goals are kept.
	bool roundFinished = !(skill.curGoals < skill.envSet->arenas.size() && totalSimTime < skill.config.maxSimTime);
	if (fixture >= 0)
		roundFinished = true;
	if (roundFinished)
		ApplyRoundElo(oldVersion.ratings);

	for (auto& pair : skill.curRatings.data) {
		float prevRating = prevCurRatings.GetRating(pair.first, skill.config.initialRating);
		float delta = pair.second - prevRating;

		std::stringstream ratingLine;
		ratingLine << " > " << pair.first << " = " << prevRating;
		if (delta != 0)
			ratingLine << " (" << (delta >= 0 ? '+' : '-') << abs(delta) << ")";

		RG_LOG(" > " << ratingLine.str());

		report["Rating/" + pair.first] = pair.second;
		report["Skill Matches/Opponent Rating " + pair.first] = oldVersion.ratings.GetRating(pair.first, skill.config.initialRating);
	}
	if (!skill.curRatings.data.empty()) {
		double avgSum = 0;
		for (auto& pair : skill.curRatings.data)
			avgSum += pair.second;
		// The one number to optimize: the mean over all trained modes (1v1, 2v2, and 3v3 once it is trained)
		report["Rating/Ranking"] = (float)(avgSum / skill.curRatings.data.size());
	}

	// What the rating of this iteration is based on
	report["Skill Matches/Goals For"] = newGoals;
	report["Skill Matches/Goals Against"] = oldGoals;
	for (auto& pair : modeGoals) {
		report["Skill Matches/Goals For " + pair.first] = pair.second.first;
		report["Skill Matches/Goals Against " + pair.first] = pair.second.second;
	}
	report["Skill Matches/Kickoff First Touches For"] = newKickoffTouches;
	report["Skill Matches/Kickoff First Touches Against"] = oldKickoffTouches;
	report["Skill Matches/Opponent Timesteps"] = (double)oldVersionTimesteps;
	report["Skill Matches/Versions"] = (double)versions.size();
	report["Skill Matches/Game Minutes"] = stepsPlayed * stepTime * skill.envSet->arenas.size() / 60;
	report["Skill Matches/Time"] = matchTimer.Elapsed();

	if (skill.curGoals < skill.envSet->arenas.size() && totalSimTime < skill.config.maxSimTime) {
		// Not enough goals were scored, we will force a continuation where the same models keep playing from the end position
		// This COULD switch versions, but it wouldn't be a big deal as it would just be the immediate next version
		RG_LOG(" > Forcing continuation (" << skill.curGoals <<  "/" << skill.envSet->arenas.size() << ")");
		skill.doContinuation = true;
		skill.prevOldVersionIndex = oldVersionIndex;
		skill.prevNewTeam = newTeam;
		skill.prevSimTime = totalSimTime;
	} else {
		skill.curGoals = 0;
	}
}

void GGL::PolicyVersionManager::RunSkillSession(PPOLearner* ppo, Report& report) {
	int numRounds = RS_MAX(skill.config.sessionRounds, 1);
	BeginTournament(numRounds);
	RG_LOG("Rating session: " << numRounds << " rounds of skill matches against " << versions.size() << " policy versions...");

	Timer timer = {};
	std::map<std::string, std::vector<float>> ratingSeries = {};
	double goalsFor = 0, goalsAgainst = 0, gameMinutes = 0;
	std::map<std::string, std::pair<double, double>> modeGoals = {};
	double kickoffsFor = 0, kickoffsAgainst = 0;
	int roundsPlayed = 0;
	for (; roundsPlayed < numRounds; roundsPlayed++) {
		if (skill.sessionAbortCheck && skill.sessionAbortCheck()) {
			RG_LOG(" > Rating session stopped early after " << roundsPlayed << " rounds");
			break;
		}

		Report roundReport = {};
		RunSkillMatches(ppo, roundReport);
		goalsFor += roundReport["Skill Matches/Goals For"];
		goalsAgainst += roundReport["Skill Matches/Goals Against"];
		gameMinutes += roundReport["Skill Matches/Game Minutes"];
		kickoffsFor += roundReport["Skill Matches/Kickoff First Touches For"];
		kickoffsAgainst += roundReport["Skill Matches/Kickoff First Touches Against"];
		for (auto& pair : skill.curRatings.data) {
			ratingSeries[pair.first].push_back(pair.second);
			modeGoals[pair.first].first += roundReport.Has("Skill Matches/Goals For " + pair.first) ? roundReport["Skill Matches/Goals For " + pair.first] : 0;
			modeGoals[pair.first].second += roundReport.Has("Skill Matches/Goals Against " + pair.first) ? roundReport["Skill Matches/Goals Against " + pair.first] : 0;
		}
	}

	// The next session starts with a newly picked opponent, from kickoffs
	skill.doContinuation = false;
	skill.curGoals = 0;

	if (roundsPlayed == numRounds) {
		double modeEndSum = 0;
		for (auto& pair : ratingSeries) {
			auto& series = pair.second;
			const float endpoint = series.back();
			modeEndSum += endpoint;

			RG_LOG(" > Rating session endpoint: " << pair.first << " = " << endpoint);
			// Publish the last Elo state as-is. Do not smooth/interpolate across rounds.
			report["Rating/" + pair.first] = endpoint;
			report["Skill Matches/Rating After Last Round " + pair.first] = endpoint;
			skill.curRatings.data[pair.first] = endpoint;
		}
		if (!ratingSeries.empty()) {
			float ranking = (float)(modeEndSum / ratingSeries.size());
			RG_LOG(" > Rating session endpoint Ranking (mean of mode endpoints) = " << ranking);
			// Ranking is the cross-mode composite; each mode's reported Elo above remains unmodified.
			report["Rating/Ranking"] = ranking;
		}

		report["Skill Matches/Rounds"] = roundsPlayed;
		report["Skill Matches/Goals For"] = goalsFor;
		report["Skill Matches/Goals Against"] = goalsAgainst;
		if (modeGoals.size() > 1) {
			// Several game modes: the goals that each mode's rating is based on
			for (auto& pair : modeGoals) {
				report["Skill Matches/Goals For " + pair.first] = pair.second.first;
				report["Skill Matches/Goals Against " + pair.first] = pair.second.second;
				RG_LOG(" > Goals in " << pair.first << ": " << pair.second.first << ":" << pair.second.second);
			}
		}
		report["Skill Matches/Kickoff First Touches For"] = kickoffsFor;
		report["Skill Matches/Kickoff First Touches Against"] = kickoffsAgainst;
		if (kickoffsFor + kickoffsAgainst > 0)
			report["Skill Matches/Kickoff First Touch Share"] = kickoffsFor / (kickoffsFor + kickoffsAgainst);
		RG_LOG(" > Goals " << goalsFor << ":" << goalsAgainst << ", first touch after the kickoff " << kickoffsFor << ":" << kickoffsAgainst);
		report["Skill Matches/Versions"] = (double)versions.size();
		report["Skill Matches/Game Minutes"] = gameMinutes;
		report["Skill Matches/Time"] = timer.Elapsed();
	}
}

void GGL::PolicyVersionManager::OnIteration(struct PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps, bool lastIteration) {
	skill.sessionJustRan = false;
	if (skill.config.enabled && skill.config.updateIntervalMinutes > 0) {
		// Rating sessions by run time.
		// Versions are saved right after a session, with the rating it has found: at the first session after
		//	tsPerVersion timesteps since the newest version
		if (versions.empty())
			AddVersion(ppo->GetPolicyModels(), totalTimesteps);

		double intervalSeconds = skill.config.updateIntervalMinutes * 60.0;
		double elapsed = skill.sessionTimer.Elapsed();
		// At the end of the run, the final state gets rated too (unless a session has just been played)
		if (elapsed >= intervalSeconds || (lastIteration && elapsed >= intervalSeconds * 0.2)) {
			RunSkillSession(ppo, report);
			uint64_t newestVersionTimesteps = 0;
			for (auto& version : versions)
				newestVersionTimesteps = RS_MAX(newestVersionTimesteps, version.timesteps);
			if (report.Has("Skill Matches/Rounds") && (uint64_t)totalTimesteps >= newestVersionTimesteps + tsPerVersion)
				AddVersion(ppo->GetPolicyModels(), totalTimesteps);
			skill.sessionTimer.Reset();
			skill.sessionJustRan = true;
		}
		return;
	}

	// NOTE: Also when there is no version yet, which is the case when versions are turned on for an existing checkpoint.
	//	The policy that the run starts with then becomes the first version, instead of the one at the next multiple of tsPerVersion.
	if ((totalTimesteps / tsPerVersion > prevTotalTimesteps / tsPerVersion) || (prevTotalTimesteps == 0) || versions.empty()) {
		// Save version
		AddVersion(ppo->GetPolicyModels(), totalTimesteps);
	}

	if (skill.config.enabled) {
		skill.iterationsSinceRan++;
		if (skill.iterationsSinceRan >= skill.config.updateInterval && !versions.empty()) {
			skill.iterationsSinceRan = 0;
			BeginTournament(1);
			RunSkillMatches(ppo, report);
		}
	}
}

void GGL::PolicyVersionManager::AddRunningStatsToJSON(nlohmann::json& json) {
	if (skill.config.enabled)
		json["skill_ratings"] = skill.curRatings.ToJSON();
}

void GGL::PolicyVersionManager::LoadRunningStatsFromJSON(const nlohmann::json& json) {
	if (skill.config.enabled)
		if (json.contains("skill_ratings"))
			skill.curRatings.ReadFromJSON(json["skill_ratings"]);
}