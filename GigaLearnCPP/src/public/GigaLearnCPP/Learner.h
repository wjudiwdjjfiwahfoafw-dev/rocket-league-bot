#pragma once

#include <RLGymCPP/EnvSet/EnvSet.h>
#include "Util/MetricSender.h"
#include "Util/RenderSender.h"
#include "Util/Timer.h"
#include "LearnerConfig.h"
#include "PPO/TransferLearnConfig.h"

namespace GGL {

	typedef std::function<void(class Learner*, const std::vector<RLGC::GameState>& states, Report& report)> StepCallbackFn;

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	class RG_IMEXPORT Learner {
	public:
		// Continuous wall-clock anchor initialized on Learner construction so
		// "Overall Steps/Second" includes 100% of setup, rollout, learning,
		// skill-tracker rating, checkpoint saving, and reporting overhead.
		std::chrono::steady_clock::time_point lastIterationWallEnd = std::chrono::steady_clock::now();

		LearnerConfig config;

		RLGC::EnvSet* envSet;

		class PPOLearner* ppo;
		class PolicyVersionManager* versionMgr;

		RLGC::EnvCreateFn envCreateFn;
		MetricSender* metricSender;
		RenderSender* renderSender;

		int obsSize;
		int numActions;

		struct WelfordStat* returnStat;
		struct BatchedWelfordStat* obsStat;

		std::string runID = {};

		// Mirrored view (see LearnerConfig::mirrorX): what is used, and what the loaded checkpoint was trained with
		int mirrorX = 0;
		int loadedMirrorX = 0;
		bool hasLoadedMirrorX = false;

		uint64_t
			totalTimesteps = 0,
			totalIterations = 0;

		StepCallbackFn stepCallback = NULL;

		// CPU threads used for learning, and by torch while inferring actions during collection (see LearnerConfig).
		// 0 if the thread counts are left alone (when learning on a GPU).
		int learnThreads = 0, collectionInferThreads = 0;

		// Set this to true (e.g. from the step callback or a signal handler) to make the learner
		//	save a checkpoint and exit at the end of the current iteration.
		// This is the programmatic equivalent of pressing 'Q', for headless runs.
		volatile bool quitRequested = false;

		// With rating sessions (SkillTrackerConfig::updateIntervalMinutes):
		// Set this together with quitRequested to have a final rating session played before the last save
		//	(for a planned end of the run, not for an interrupt, which should save and exit quickly).
		volatile bool rateBeforeQuit = false;
		// Returns true to stop a running rating session early (e.g. when an interrupt signal has arrived)
		std::function<bool()> shouldAbortRating = nullptr;

		Learner(RLGC::EnvCreateFn envCreateFunc, LearnerConfig config, StepCallbackFn stepCallback = NULL);
		void Start();

		void StartTransferLearn(const TransferLearnConfig& transferLearnConfig);

		// Rates the loaded checkpoint with the skill tracker, without training:
		//	plays numRounds rounds of its matches against the saved policy versions (see config.policyVersionsFolder).
		// The callback gets the report of every round ("Rating/<mode>", "Skill Matches/...").
		// Nothing is saved, neither the checkpoint nor the versions change on disk.
		void StartSkillMatches(int numRounds, std::function<void(int round, const Report& report)> roundCallback = NULL);

		// Current ratings of the skill tracker, per game mode ("1v1", ...). Empty without skill tracker or before the first match
		std::map<std::string, float> GetSkillRatings();

		void StartQuitKeyThread(bool& quitPressed, std::thread& outThread);

		void Save();
		void Load();
		void SaveStats(std::filesystem::path path);
		void LoadStats(std::filesystem::path path);

		RG_NO_COPY(Learner);

		~Learner();
	};
}