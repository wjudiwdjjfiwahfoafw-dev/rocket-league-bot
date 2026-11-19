#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include "PPO/PPOLearnerConfig.h"

#include <functional>
#include <map>
#include <string>
#include "SkillTrackerConfig.h"

namespace GGL {
	enum class LearnerDeviceType {
		AUTO,
		CPU,
		GPU_CUDA
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	struct LearnerConfig {
		int numGames = 300;

		int tickSkip = 8;
		int actionDelay = 7;

		bool renderMode = false;
		// Optional episode-boundary selector for the ONE render arena; never used in training.
		std::function<int()> renderChooseTeamSize = nullptr;
		// If renderMode, this is the scaling of time for the game
		// 1.0 = Run the game at real time
		// 2.0 = Run the game twice as fast as real time
		float renderTimeScale = 1.0f; 

		PPOLearnerConfig ppo = {};

		// Checkpoints are saved here as timestep-numbered subfolders
		//	e.g. a checkpoint at 20,000 steps will save to a subfolder called "20000"
		// Set empty to disable saving
		std::filesystem::path checkpointFolder = "checkpoints"; 

		// Save every timestep
		// Set to zero to just use timestepsPerIteration
		int64_t tsPerSave = 1'000'000;
		uint64_t maxTimesteps = 0; // PPO: stop at an iteration boundary, counting learner steps only

		// Optional notes that are saved into every checkpoint as "INFO.json" (e.g. which model config and obs builder it needs).
		// The learner itself doesn't read them.
		std::map<std::string, std::string> checkpointInfo = {};

		int64_t randomSeed = -1; // Set to -1 to use the current time
		int checkpointsToKeep = 8; // Checkpoint storage limit before old checkpoints are deleted, set to -1 to disable
		LearnerDeviceType deviceType = LearnerDeviceType::AUTO; // Auto will use your CUDA GPU if available

		// Number of CPU threads used for learning (only matters when learning on the CPU).
		// Every batch is split into minibatches, each thread computes the gradients of its share of them, and the gradients are added up.
		// 0 = automatic: torch's default thread count (one thread per physical core that it detects), 
		//	or one thread per logical CPU if a short measurement at startup shows that this is clearly faster.
		//	Virtual machines often present their CPUs as hyperthreads of a single core, and torch then uses only one of them.
		int learnThreads = 0;

		// Number of CPU threads that torch uses to infer the actions while collecting (only matters when inferring on the CPU).
		// 0 = automatic: One thread if there are fewer than 1024 players, otherwise torch's default.
		//	(Matrix products that small gain little from more threads, and the game simulation is using all CPU threads at that moment)
		int collectionInferThreads = 0;

		// Standardize the obs values (doesn't seem to help much from my testing)
		bool standardizeObs = false;
		float minObsSTD = 1 / 10.f;
		float maxObsMeanRange = 3;
		int maxObsSamples = 100;

		// Standardize the returns to help the critic (don't disable this unless you know what you're doing)
		bool standardizeReturns = true;
		int maxReturnSamples = 150;
		// Optional effective sample cap on returnStat (0 = unlimited) so returnStd can adapt smoothly after reward changes.
		int64_t maxReturnStatCount = 0;

		// Will automatically add the rewards to metrics
		bool addRewardsToMetrics = true;
		int maxRewardSamples = 50; // Maximum reward samples per step for reward metrics
		int rewardSampleRandInterval = 8; // Randomized interval range between sampling rewards (per step)

		// Send metrics to the python metrics receiver
		// The receiver can then log them to wandb or whatever
		bool sendMetrics = true;
		std::string metricsProjectName = "gigalearncpp"; // Project name for the python metrics receiver
		std::string metricsGroupName = "unnamed-runs"; // Group name for the python metrics receiver
		std::string metricsRunName = "gigalearncpp-run"; // Run name for the python metrics receiver

		bool savePolicyVersions = false;
		int64_t tsPerVersion = 25'000'000;
		int maxOldVersions = 32;

		// Folder of the saved policy versions, empty for "policy_versions" inside the checkpoint folder.
		// Setting it is meant for Learner::StartSkillMatches(), to rate a checkpoint against the versions of another run.
		// Those versions are only read then, and training (Learner::Start()) refuses to run with it.
		std::filesystem::path policyVersionsFolder = {};

		bool trainAgainstOldVersions = false;
		float trainAgainstOldChance = 0.15f; // Chance (from 0 - 1) that an iteration will train against an old version
		// Prioritized Fictitious Self-Play: favor stronger saved opponents while retaining coverage of older styles.
		bool pfspOpponents = false;

		// Mirrored view: 0 = off, 1 = right, -1 = left.
		// With "right", a car on the left half of the field sees the whole game mirrored, and its action is mirrored back.
		// The network then only ever sees its own car on the right half, and has to learn every situation once instead of twice.
		// "left" is the same with the halves swapped (see RLGC::ObsBuilder::mirrorXTeam). Needs an obs builder that supports it.
		// Saved policy versions always play with the view they were trained with.
		int mirrorX = 0;

		// true: a checkpoint that is loaded decides instead (it was saved with the view it was trained with)
		bool mirrorXFromCheckpoint = true;

		SkillTrackerConfig skillTracker = {};
	};
}