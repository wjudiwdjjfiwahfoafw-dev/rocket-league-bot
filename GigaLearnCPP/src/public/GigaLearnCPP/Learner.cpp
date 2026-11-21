#include "Learner.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>

#include <torch/cuda.h>
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/PolicyVersionManager.h>

#include "Util/KeyPressDetector.h"
#include <private/GigaLearnCPP/Util/WelfordStat.h>
#include "Util/AvgTracker.h"

#ifdef __GLIBC__
#include <malloc.h>
#endif

using namespace RLGC;

// The learner allocates and frees tensors of several megabytes all the time.
// By default, glibc returns that memory to the operating system when it is freed, 
//	so every new tensor has to be mapped and zeroed by the kernel again, page by page 
//	(measured with the example bot on a CPU: 130,000 page faults per iteration, 8% of the learning time).
// These settings make glibc keep the memory instead.
static void TuneMemoryAllocator() {
#ifdef __GLIBC__
	if (getenv("GGL_NO_MALLOC_TUNING"))
		return;

	mallopt(M_MMAP_THRESHOLD, 32 * 1024 * 1024); // Largest value glibc accepts. Bigger blocks are still returned to the OS
	mallopt(M_TRIM_THRESHOLD, 1024 * 1024 * 1024);
	mallopt(M_TOP_PAD, 64 * 1024 * 1024);
#endif
}

// Torch (and the MKL library it uses for matrix products) defaults to one thread per physical CPU core.
// That is right for real hyperthreads, which share the units that do the math.
// But virtual machines often present all of their CPUs as hyperthreads of one core even if they are separate cores,
//	and then half (or more) of the machine goes unused while learning.
// So if there are more logical CPUs than torch's default, we just measure which is faster.
static int FindBestTorchThreadCount() {
	int defaultThreads = torch::get_num_threads();
	int logicalCPUs = (int)std::thread::hardware_concurrency();
	if (logicalCPUs <= defaultThreads)
		return defaultThreads;

	RG_NO_GRAD;
	torch::Tensor a = torch::randn({ 4096, 256 }), b = torch::randn({ 256, 256 });

	auto fnMeasure = [&](int numThreads) -> double {
		torch::set_num_threads(numThreads);
		for (int i = 0; i < 4; i++) // Warm-up, starts the threads
			at::mm(a, b);

		double bestTime = FLT_MAX;
		for (int i = 0; i < 12; i++) {
			GGL::Timer timer = {};
			at::mm(a, b);
			bestTime = RS_MIN(bestTime, timer.Elapsed());
		}
		return bestTime;
	};

	// Several rounds each, spread over about a second, to be less affected by other things running on the machine.
	// (With two rounds right after each other, a few seconds of another program at the start of a run were enough
	//	to get the lower thread count, and with it half the learning speed, for the whole run.)
	const int NUM_ROUNDS = 6;
	double timeDefault = FLT_MAX, timeAll = FLT_MAX;
	for (int round = 0; round < NUM_ROUNDS; round++) {
		timeDefault = RS_MIN(timeDefault, fnMeasure(defaultThreads));
		timeAll = RS_MIN(timeAll, fnMeasure(logicalCPUs));
		if (round + 1 < NUM_ROUNDS)
			std::this_thread::sleep_for(std::chrono::milliseconds(150));
	}
	torch::set_num_threads(defaultThreads);

	// Only use the logical CPU count if it is clearly faster
	bool useAll = timeAll * 1.25 < timeDefault;
	RG_LOG(
		"\tTorch CPU threads: " << defaultThreads << " by default, the machine has " << logicalCPUs << " logical CPUs. " <<
		"Matrix products are " << std::fixed << std::setprecision(2) << (timeDefault / timeAll) << "x as fast with " << logicalCPUs << " threads" <<
		" -> using " << (useAll ? logicalCPUs : defaultThreads) << " threads for learning" <<
		(useAll ? "" : " (if the machine was busy during this test, set the number with LearnerConfig::learnThreads)")
	);
	return useAll ? logicalCPUs : defaultThreads;
}

GGL::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig config, StepCallbackFn stepCallback) :
	envCreateFn(envCreateFn), config(config), stepCallback(stepCallback)
{
	// Rendering must never sample actions, even for library callers bypassing the CLI.
	if (config.renderMode) {
		config.ppo.deterministic = true;
		this->config.ppo.deterministic = true;
	}
	TuneMemoryAllocator();

	// The embedded interpreter must use the standard library of the same Python installation this
	// build links against (its headers and libpython). CPython normally derives that from the
	// interpreter's own location, but it only does so when it knows the interpreter's path.
	// Point it at the interpreter this build was configured with, so the two can never drift apart
	// (a mismatched standard library loads modules that were not built against this libpython).
#if PY_VERSION_HEX >= 0x03080000
	PyConfig pyConfig;
	PyConfig_InitPythonConfig(&pyConfig);
	pyConfig.parse_argv = 0;
	pyConfig.install_signal_handlers = 1;
	if (PyStatus_Exception(PyConfig_SetBytesString(&pyConfig, &pyConfig.program_name, PY_EXEC_PATH)) != 0)
		throw std::runtime_error("Failed to point the embedded Python interpreter at " + std::string(PY_EXEC_PATH));
	pybind11::initialize_interpreter(&pyConfig);
#else
	pybind11::initialize_interpreter();
#endif

#ifndef NDEBUG
	RG_LOG("===========================");
	RG_LOG("WARNING: GigaLearn runs extremely slowly in debug, and there are often bizzare issues with debug-mode torch.");
	RG_LOG("It is recommended that you compile in release mode without optimization for debugging.");
	RG_SLEEP(1000);
#endif

	if (config.tsPerSave == 0)
		config.tsPerSave = config.ppo.tsPerItr;

	RG_LOG("Learner::Learner():");

	if (config.randomSeed == -1)
		config.randomSeed = RS_CUR_MS();

	RG_LOG("\tCheckpoint Save/Load Dir: " << config.checkpointFolder);

	torch::manual_seed(config.randomSeed);

	at::Device device = at::Device(at::kCPU);
	if (
		config.deviceType == LearnerDeviceType::GPU_CUDA || 
		(config.deviceType == LearnerDeviceType::AUTO && torch::cuda::is_available())
		) {
		RG_LOG("\tUsing CUDA GPU device...");

		// Test out moving a tensor to GPU and back to make sure the device is working
		torch::Tensor t;
		bool deviceTestFailed = false;
		try {
			t = torch::tensor(0);
			t = t.to(at::Device(at::kCUDA));
			t = t.cpu();
		} catch (...) {
			deviceTestFailed = true;
		}

		if (!torch::cuda::is_available() || deviceTestFailed)
			RG_ERR_CLOSE(
				"Learner::Learner(): Can't use CUDA GPU because " <<
				(torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
			)
		device = at::Device(at::kCUDA);
	} else {
		RG_LOG("\tUsing CPU device...");
		device = at::Device(at::kCPU);
	}

	if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
		RG_LOG("\tInitializing RocketSim...");
		RocketSim::Init("collision_meshes", true);
	}

	{
		RG_LOG("\tCreating envs...");
		EnvSetConfig envSetConfig = {};
		envSetConfig.envCreateFn = envCreateFn;
		envSetConfig.numArenas = config.renderMode ? 1 : config.numGames;
		envSetConfig.tickSkip = config.tickSkip;
		envSetConfig.actionDelay = config.actionDelay;
		envSetConfig.saveRewards = config.addRewardsToMetrics;
		envSet = new RLGC::EnvSet(envSetConfig);
		obsSize = envSet->state.obs.size[1];
		numActions = envSet->actionParsers[0]->GetActionAmount();
	}

	{
		if (config.standardizeReturns) {
			this->returnStat = new WelfordStat();
		} else {
			this->returnStat = NULL;
		}

		if (config.standardizeObs) {
			for (auto modelConfig : { config.ppo.sharedHead, config.ppo.policy, config.ppo.critic })
				if (modelConfig.slots.IsValid())
					RG_ERR_CLOSE("standardizeObs can't be combined with a slot encoder (it needs the \"slot is used\" inputs to stay 0 or 1)");

			this->obsStat = new BatchedWelfordStat(obsSize);
		} else {
			this->obsStat = NULL;
		}
	}

	try {
		RG_LOG("\tMaking PPO learner...");
		ppo = new PPOLearner(obsSize, numActions, config.ppo, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Failed to create PPO learner: " << e.what());
	}

	if (device.is_cpu() && !config.renderMode) {
		// Decide how many CPU threads are used in each phase (see LearnerConfig)
		int defaultThreads = torch::get_num_threads();

		learnThreads = (config.learnThreads > 0) ? config.learnThreads : FindBestTorchThreadCount();

		if (config.collectionInferThreads > 0) {
			collectionInferThreads = config.collectionInferThreads;
		} else {
			collectionInferThreads = (envSet->state.numPlayers < 1024) ? 1 : defaultThreads;
		}

		ppo->SetNumLearnThreads(learnThreads);

		RG_LOG("\tCPU threads: " << learnThreads << " for learning, " << RS_MAX(collectionInferThreads, ppo->numInferThreads) << " for inferring actions during collection");
		RG_LOG("\tPPO minibatch size: " << ppo->miniBatchSize);
	}

	if (config.renderMode) {
		renderSender = new RenderSender(config.renderTimeScale);
	} else {
		renderSender = NULL;
	}

	if (config.skillTracker.enabled || config.trainAgainstOldVersions)
		config.savePolicyVersions = true;

	if (config.savePolicyVersions && !config.renderMode) {
		if (config.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint save folder");
		versionMgr = new PolicyVersionManager(
			config.policyVersionsFolder.empty() ? (config.checkpointFolder / "policy_versions") : config.policyVersionsFolder,
			config.maxOldVersions, config.tsPerVersion,
			config.skillTracker, envSet->config
		);
	} else {
		versionMgr = NULL;
	}

	if (!config.checkpointFolder.empty())
		Load();

	{ // Mirrored view
		mirrorX = (config.mirrorXFromCheckpoint && hasLoadedMirrorX) ? loadedMirrorX : config.mirrorX;
		mirrorX = (mirrorX > 0) - (mirrorX < 0);

		if (mirrorX != 0 || loadedMirrorX != 0) {
			RG_LOG("\tMirrored view: " << (mirrorX ? RS_STR("on, the network sees its car on the " << (mirrorX > 0 ? "right" : "left") << " half") : std::string("off")) <<
				((hasLoadedMirrorX && loadedMirrorX != mirrorX) ? " (NOTE: the checkpoint was trained with another setting)" : ""));
		}

		envSet->SetMirrorX(mirrorX, mirrorX);
		if (versionMgr)
			versionMgr->curMirrorX = mirrorX;
	}

	if (config.savePolicyVersions && !config.renderMode) {
		if (config.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint save folder");
		auto models = ppo->GetPolicyModels();
		// NOTE: Versions from another run's folder may well have more timesteps than this checkpoint
		versionMgr->LoadVersions(models, totalTimesteps, !config.policyVersionsFolder.empty());
		if (config.skillTracker.anchored)
			versionMgr->PrepareAnchoredPool((uint32_t)config.randomSeed);
	}

	if (config.sendMetrics && !config.renderMode) {
		if (!runID.empty())
			RG_LOG("\tRun ID: " << runID);
		metricSender = new MetricSender(config.metricsProjectName, config.metricsGroupName, config.metricsRunName, runID);
	} else {
		metricSender = NULL;
	}

	RG_LOG(RG_DIVIDER);
}

void GGL::Learner::SaveStats(std::filesystem::path path) {
	using namespace nlohmann;

	constexpr const char* ERROR_PREFIX = "Learner::SaveStats(): ";

	std::ofstream fOut(path);
	if (!fOut.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = {};
	j["total_timesteps"] = totalTimesteps;
	j["total_iterations"] = totalIterations;

	if (config.sendMetrics)
		j["run_id"] = metricSender->curRunID;

	j["mirror_x"] = mirrorX;

	if (returnStat)
		j["return_stat"] = returnStat->ToJSON();
	if (obsStat)
		j["obs_stat"] = obsStat->ToJSON();

	if (versionMgr)
		versionMgr->AddRunningStatsToJSON(j);

	std::string jStr = j.dump(4);
	fOut << jStr;
}

void GGL::Learner::LoadStats(std::filesystem::path path) {
	// TODO: Repetitive code, merge repeated code into one function called from both SaveStats() and LoadStats()

	using namespace nlohmann;
	constexpr const char* ERROR_PREFIX = "Learner::LoadStats(): ";

	std::ifstream fIn(path);
	if (!fIn.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = json::parse(fIn);
	totalTimesteps = j["total_timesteps"];
	totalIterations = j["total_iterations"];

	if (j.contains("run_id"))
		runID = j["run_id"];

	if (j.contains("mirror_x")) {
		loadedMirrorX = j["mirror_x"].get<int>();
		hasLoadedMirrorX = true;
	}

	if (returnStat)
		returnStat->ReadFromJSON(j["return_stat"]);
	if (obsStat)
		obsStat->ReadFromJSON(j["obs_stat"]);

	if (versionMgr)
		versionMgr->LoadRunningStatsFromJSON(j);
}

// Different than RLGym-PPO to show that they are not compatible
constexpr const char* STATS_FILE_NAME = "RUNNING_STATS.json";

void GGL::Learner::Save() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Save(): Cannot save because config.checkpointSaveFolder is not set");

	std::filesystem::path saveFolder = config.checkpointFolder / std::to_string(totalTimesteps);
	std::filesystem::create_directories(saveFolder);

	RG_LOG("Saving to folder " << saveFolder << "...");
	SaveStats(saveFolder / STATS_FILE_NAME);
	ppo->SaveTo(saveFolder);

	if (!config.checkpointInfo.empty()) {
		std::ofstream infoOut(saveFolder / "INFO.json");
		infoOut << nlohmann::json(config.checkpointInfo).dump(4) << std::endl;
	}

	// Remove old checkpoints
	if (config.checkpointsToKeep != -1) {
		std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);
		while (allSavedTimesteps.size() > config.checkpointsToKeep) {
			int64_t lowestCheckpointTS = INT64_MAX;
			for (int64_t savedTimesteps : allSavedTimesteps)
				lowestCheckpointTS = RS_MIN(lowestCheckpointTS, savedTimesteps);

			std::filesystem::path removePath = config.checkpointFolder / std::to_string(lowestCheckpointTS);
			try {
				std::filesystem::remove_all(removePath);
			} catch (std::exception& e) {
				RG_ERR_CLOSE("Failed to remove old checkpoint from " << removePath << ", exception: " << e.what());
			}
			allSavedTimesteps.erase(lowestCheckpointTS);
		}
	}

	if (versionMgr)
		versionMgr->SaveVersions();

	RG_LOG(" > Done.");
}

void GGL::Learner::Load() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Load(): Cannot load because config.checkpointLoadFolder is not set");

	RG_LOG("Loading most recent checkpoint in " << config.checkpointFolder << "...");

	int64_t highest = -1;
	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);
	for (int64_t timesteps : allSavedTimesteps)
		highest = RS_MAX(timesteps, highest);

	if (highest != -1) {
		std::filesystem::path loadFolder = config.checkpointFolder / std::to_string(highest);
		RG_LOG(" > Loading checkpoint " << loadFolder << "...");
		LoadStats(loadFolder / STATS_FILE_NAME);
		ppo->LoadFrom(loadFolder);
		RG_LOG(" > Done.");
	} else {
		RG_LOG(" > No checkpoints found, starting new model.")
	}
}

void GGL::Learner::StartQuitKeyThread(bool& quitPressed, std::thread& outThread) {
	quitPressed = false;

	if (!KeyPressDetector::IsInteractive()) {
		// Without this check, the key thread spins at 100% CPU on a closed/redirected stdin (and spams errors)
		RG_LOG("No interactive terminal detected, 'Q' to quit is disabled (set Learner::quitRequested instead).");
		return;
	}

	RG_LOG("Press 'Q' to save and quit!");
	outThread = std::thread(
		[this] {
			while (true) {
				char c = toupper(KeyPressDetector::GetPressedChar());
				if (c == 'Q') {
					RG_LOG("Save queued, will save and exit next iteration.");
					this->quitRequested = true;
					break; // Stop listening, which also gives the terminal back in its normal mode
				}
			}
		}
	);

	outThread.detach();
}
void GGL::Learner::StartTransferLearn(const TransferLearnConfig& tlConfig) {

	RG_LOG("Starting transfer learning...");

	// TODO: Lots of manual obs builder stuff going on which is quite volatile
	//	Although I can't really think another way to do this

	std::vector<ObsBuilder*> oldObsBuilders = {};
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders.push_back(tlConfig.makeOldObsFn());

	// Reset all obs builders initially
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders[i]->Reset(envSet->state.gameStates[0]);

	std::vector<ActionParser*> oldActionParsers = {};
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldActionParsers.push_back(tlConfig.makeOldActFn());

	int oldNumActions = oldActionParsers[0]->GetActionAmount();

	if (oldNumActions != numActions) {
		if (!tlConfig.mapActsFn) {
			RG_ERR_CLOSE(
				"StartTransferLearn: Old and new action parsers have a different number of actions, but tlConfig.mapActsFn is NULL.\n" <<
				"You must implement this function to translate the action indices."
			);
		};
	}

	// Determine old obs size
	int oldObsSize;
	{
		GameState testState = envSet->state.gameStates[0];
		oldObsSize = oldObsBuilders[0]->BuildObs(testState.players[0], testState).size();
	}

	ModelSet oldModels = {};
	{
		RG_NO_GRAD;
		PPOLearner::MakeModels(false, oldObsSize, oldNumActions, tlConfig.oldSharedHeadConfig, tlConfig.oldPolicyConfig, {}, ppo->device, oldModels);

		oldModels.Load(tlConfig.oldModelsPath, false, false);
	}

	try {
		bool saveQueued;
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);

		while (true) {
			Report report = {};

			// Collect obs
			std::vector<float> allNewObs = {};
			std::vector<float> allOldObs = {};
			std::vector<uint8_t> allNewActionMasks = {};
			std::vector<uint8_t> allOldActionMasks = {};
			std::vector<int> allActionMaps = {};
			int stepsCollected;
			{
				RG_NO_GRAD;
				for (stepsCollected = 0; stepsCollected < tlConfig.batchSize; stepsCollected += envSet->state.numPlayers) {
					
					auto terminals = envSet->state.terminals; // Backup
					envSet->Reset();
					for (int i = 0; i < envSet->arenas.size(); i++) // Manually reset old obs builders
						if (terminals[i])
							oldObsBuilders[i]->Reset(envSet->state.gameStates[i]);

					torch::Tensor tActions, tLogProbs;
					torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
					torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

					envSet->StepFirstHalf(true);

					allNewObs += envSet->state.obs.data;
					allNewActionMasks += envSet->state.actionMasks.data;

					// Run all old obs and old action parser on each player
					// TODO: Could be multithreaded
					for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
						auto& gs = envSet->state.gameStates[arenaIdx];
						for (auto& player : gs.players) {
							allOldObs += oldObsBuilders[arenaIdx]->BuildObs(player, gs);
							allOldActionMasks += oldActionParsers[arenaIdx]->GetActionMask(player, gs);

							if (tlConfig.mapActsFn) {
								auto curMap = tlConfig.mapActsFn(player, gs);
								if (curMap.size() != numActions)
									RG_ERR_CLOSE("StartTransferLearn: Your action map must have the same size as the new action parser's actions");
								allActionMaps += curMap;
							}
						}
					}

					ppo->InferActions(
						tStates.to(ppo->device, true), tActionMasks.to(ppo->device, true), 
						&tActions, &tLogProbs
					);

					auto curActions = TENSOR_TO_VEC<int>(tActions);

					envSet->Sync();
					envSet->StepSecondHalf(curActions, false);

					if (stepCallback)
						stepCallback(this, envSet->state.gameStates, report);
				}
			}

			uint64_t prevTimesteps = totalTimesteps;
			totalTimesteps += stepsCollected;
			report["Total Timesteps"] = totalTimesteps;
			report["Collected Timesteps"] = stepsCollected;
			totalIterations++;
			report["Total Iterations"] = totalIterations;

			// Make tensors
			torch::Tensor tNewObs = torch::tensor(allNewObs).reshape({ -1, obsSize }).to(ppo->device);
			torch::Tensor tOldObs = torch::tensor(allOldObs).reshape({ -1, oldObsSize }).to(ppo->device);
			torch::Tensor tNewActionMasks = torch::tensor(allNewActionMasks).reshape({ -1, numActions }).to(ppo->device);
			torch::Tensor tOldActionMasks = torch::tensor(allOldActionMasks).reshape({ -1, oldNumActions }).to(ppo->device);

			torch::Tensor tActionMaps = {};
			if (!allActionMaps.empty())
				tActionMaps = torch::tensor(allActionMaps).reshape({ -1, numActions }).to(ppo->device);

			// Transfer learn
			ppo->TransferLearn(oldModels, tNewObs, tOldObs, tNewActionMasks, tOldActionMasks, tActionMaps, report, tlConfig);

			if (versionMgr)
				versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

			if (saveQueued) {
				if (!config.checkpointFolder.empty())
					Save();
				exit(0);
			}

			if (!config.checkpointFolder.empty()) {
				if (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave) {
					// Auto-save
					Save();
				}
			}

			report.Finish();

			if (metricSender)
				metricSender->Send(report);

			report.Display(
				{
					"Transfer Learn Accuracy",
					"Transfer Learn Loss",
					"",
					"Policy Entropy",
					"Old Policy Entropy",
					"Policy Update Magnitude",
					"",
					"Collected Timesteps",
					"Total Timesteps",
					"Total Iterations"
				}
			);
		}

	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during transfer learn loop: " << e.what());
	}
}

std::map<std::string, float> GGL::Learner::GetSkillRatings() {
	if (!versionMgr)
		return {};
	return versionMgr->skill.curRatings.data;
}

void GGL::Learner::StartSkillMatches(int numRounds, std::function<void(int round, const Report& report)> roundCallback) {
	if (!versionMgr || !config.skillTracker.enabled)
		RG_ERR_CLOSE("Learner::StartSkillMatches(): The skill tracker is not enabled (config.skillTracker.enabled)");
	if (versionMgr->versions.empty())
		RG_ERR_CLOSE("Learner::StartSkillMatches(): There are no saved policy versions to play against in " << versionMgr->saveFolder);

	if (config.skillTracker.anchored && numRounds % (2 * versionMgr->versions.size()) != 0)
		RG_ERR_CLOSE("Anchored rounds must be a multiple of twice the opponent count");
	RG_LOG("Learner::StartSkillMatches(): " << numRounds << " rounds against " << versionMgr->versions.size() << " policy versions");
	// The rounds are one tournament: the current policy keeps the Elo it arrived with, and each round
	//may move it by a constant fraction of the tournament budget.
	versionMgr->BeginTournament(numRounds);

	try {
		for (int i = 0; i < numRounds && !quitRequested; i++) {
			Report report = {};
			versionMgr->RunSkillMatches(ppo, report, config.skillTracker.anchored ? i : -1);
			report["Total Timesteps"] = totalTimesteps;
			report.Finish();
			if (roundCallback)
				roundCallback(i, report);
		}
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during the skill matches: " << e.what());
	}
	quitRequested = false;
}

void GGL::Learner::Start() {

	if (!config.policyVersionsFolder.empty()) {
		// Training adds and removes versions, which would change the versions of the other run
		RG_ERR_CLOSE(
			"Learner::Start(): config.policyVersionsFolder is only meant for Learner::StartSkillMatches(). " <<
			"To train with the policy versions of another run, copy them into \"policy_versions\" inside the checkpoint folder.");
	}

	bool render = config.renderMode;

	RG_LOG("Learner::Start():");
	RG_LOG("\tObs size: " << obsSize);
	RG_LOG("\tAction amount: " << numActions);

	if (render)
		RG_LOG("\t(Render mode enabled)");

	try {
		bool saveQueued;
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);

		ExperienceBuffer experience = ExperienceBuffer(config.randomSeed, torch::kCPU);

		int numPlayers = envSet->state.numPlayers;

		struct Trajectory {
			FList states, nextStates, rewards, logProbs, values;
			std::vector<uint8_t> actionMasks;
			std::vector<int8_t> terminals;
			std::vector<int32_t> actions;
			std::vector<uint8_t> modes;

			void Clear() {
				*this = Trajectory();
			}

			void Append(const Trajectory& other) {
				states += other.states;
				nextStates += other.nextStates;
				rewards += other.rewards;
				logProbs += other.logProbs;
				values += other.values;
				actionMasks += other.actionMasks;
				terminals += other.terminals;
				actions += other.actions;
				modes += other.modes;
			}

			size_t Length() const {
				return actions.size();
			}
		};

		// Mode index (0=1v1, 1=2v2, 2=3v3) of each player slot in envSet, used for per-mode advantage normalization.
		std::vector<uint8_t> playerModes(numPlayers, 0);
		if (!render) {
			int pIdx = 0;
			for (auto& state : envSet->state.gameStates) {
				uint8_t m = (uint8_t)RS_CLAMP((int)state.players.size() / 2 - 1, 0, 2);
				for (size_t k = 0; k < state.players.size(); k++)
					playerModes[pIdx++] = m;
			}
		}

		// On fixed rollouts all samples are gathered before the next update. Reuse the actor's shared
		// features for the critic here, instead of encoding every car a second time before GAE.
		bool collectValues = config.ppo.fixedRollout && !render && ppo->device.is_cpu();
		auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
		int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));

		while (true) {
			Report report = {};

			bool isFirstIteration = (totalTimesteps == 0);

			// TODO: Old version switching messes up the gameplay potentially
			GGL::PolicyVersion* oldVersion = NULL;
			Team oldVersionTeam = Team::BLUE;
			std::vector<bool> oldVersionPlayerMask;
			std::vector<int> newPlayerIndices = {}, oldPlayerIndices = {};
			torch::Tensor tNewPlayerIndices, tOldPlayerIndices;

			for (int i = 0; i < numPlayers; i++)
				newPlayerIndices.push_back(i);

			if (!render && config.trainAgainstOldVersions) {
				RG_ASSERT(config.trainAgainstOldChance >= 0 && config.trainAgainstOldChance <= 1);
				bool shouldTrainAgainstOld =
					(RocketSim::Math::RandFloat() < config.trainAgainstOldChance)
					&& !versionMgr->versions.empty()
					&& !render;

				if (shouldTrainAgainstOld) {
					// Set up training against old versions

					int oldVersionIdx = 0;
					int nVers = (int)versionMgr->versions.size();
					if (config.pfspOpponents && nVers > 1) {
						// Prioritized Fictitious Self-Play: favor higher-rated saved versions while keeping a
						// healthy floor on all older versions so the policy never drops match-ups against earlier styles.
						std::vector<double> meanElos(nVers, 0.0);
						double bestElo = -1e9;
						bool anyRated = false;
						for (int v = 0; v < nVers; v++) {
							auto& rmap = versionMgr->versions[v].ratings.data;
							if (!rmap.empty()) {
								double s = 0.0;
								for (auto& kv : rmap)
									s += kv.second;
								meanElos[v] = s / rmap.size();
								bestElo = std::max(bestElo, meanElos[v]);
								anyRated = true;
							}
						}
						if (anyRated) {
							std::vector<double> weights(nVers, 1.0);
							double totalW = 0.0;
							for (int v = 0; v < nVers; v++) {
								double recencyBonus = 0.25 * (double)v / (double)(nVers - 1);
								double eloW = versionMgr->versions[v].ratings.data.empty()
									? 1.0
									: std::exp(std::clamp((meanElos[v] - bestElo) / 120.0, -4.0, 0.0));
								weights[v] = std::max(0.30, eloW + recencyBonus);
								totalW += weights[v];
							}
							double pick = RocketSim::Math::RandFloat() * totalW;
							double acc = 0.0;
							oldVersionIdx = nVers - 1;
							for (int v = 0; v < nVers; v++) {
								acc += weights[v];
								if (pick <= acc) {
									oldVersionIdx = v;
									break;
								}
							}
						} else {
							oldVersionIdx = RocketSim::Math::RandInt(0, nVers);
						}
					} else {
						oldVersionIdx = RocketSim::Math::RandInt(0, nVers);
					}
					oldVersion = &versionMgr->versions[oldVersionIdx];

					oldVersionTeam = Team(RocketSim::Math::RandInt(0, 2)); 
					
					newPlayerIndices.clear();
					oldVersionPlayerMask.resize(numPlayers);
					int i = 0;
					for (auto& state : envSet->state.gameStates) {
						for (auto& player : state.players) {
							if (player.team == oldVersionTeam) {
								oldVersionPlayerMask[i] = true;
								oldPlayerIndices.push_back(i);
							} else {
								oldVersionPlayerMask[i] = false;
								newPlayerIndices.push_back(i);
							}
							i++;
						}
					}

					tNewPlayerIndices = torch::tensor(newPlayerIndices);
					tOldPlayerIndices = torch::tensor(oldPlayerIndices);
				}
			}

			{ // Each side sees the game the way its policy was trained (see LearnerConfig::mirrorX)
				int teamMirrorX[2] = { mirrorX, mirrorX };
				if (oldVersion)
					teamMirrorX[(int)oldVersionTeam] = oldVersion->mirrorX;
				envSet->SetMirrorX(teamMirrorX[0], teamMirrorX[1]);
			}

			int numRealPlayers = oldVersion ? newPlayerIndices.size() : envSet->state.numPlayers;

			int stepsCollected = 0;
			{ // Generate experience

				// Only contains complete episodes
				auto combinedTraj = Trajectory();
				if (config.ppo.fixedRollout && !render) {
					size_t rows = config.ppo.tsPerItr + numRealPlayers;
					combinedTraj.states.reserve(rows * obsSize);
					combinedTraj.actionMasks.reserve(rows * numActions);
				}

				if (collectionInferThreads > 0)
					torch::set_num_threads(collectionInferThreads);

				Timer collectionTimer = {};
				{ // Collect timesteps
					RG_NO_GRAD;

					float inferTime = 0;
					float envStepTime = 0;

					for (int step = 0; combinedTraj.Length() < config.ppo.tsPerItr || render; step++, stepsCollected += numRealPlayers) {
						Timer stepTimer = {};
						if (render && envSet->state.terminals[0] && config.renderChooseTeamSize) {
							envSet->ResetSingleArenaWithTeamSize(config.renderChooseTeamSize());
							numPlayers = numRealPlayers = envSet->state.numPlayers;
						} else {
							envSet->Reset();
						}
						envStepTime += stepTimer.Elapsed();

						// Start simulating the action delay right away (it only depends on the previous actions).
						// It runs on the thread pool while this thread checks the obs, stores them and infers the next actions.
						// (This does not touch the obs or the action masks, those are set at the end of the step and by Reset())
						envSet->StepFirstHalf(true);

						{ // Make sure the obs are all finite
							bool allFinite = true;
							for (float f : envSet->state.obs.data)
								allFinite &= std::isfinite(f);
							if (!allFinite)
								RG_ERR_CLOSE("Obs builder produced a NaN/inf value");
						}

						if (!render && obsStat) {
							// TODO: This samples from old versions too
							int numSamples = RS_MAX(envSet->state.numPlayers, config.maxObsSamples);
							for (int i = 0; i < numSamples; i++) {
								int idx = Math::RandInt(0, envSet->state.numPlayers);
								obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
							}

							std::vector<double> mean = obsStat->GetMean();
							std::vector<double> std = obsStat->GetSTD();
							for (double& f : mean)
								f = RS_CLAMP(f, -config.maxObsMeanRange, config.maxObsMeanRange);
							for (double& f : std)
								f = RS_MAX(f, config.minObsSTD);
							for (int i = 0; i < envSet->state.numPlayers; i++) {
								for (int j = 0; j < obsSize; j++) {
									float& obsVal = envSet->state.obs.At(i, j);
									obsVal = (obsVal - mean[j]) / std[j];
								}
							}
						}

						torch::Tensor tActions, tLogProbs, tValues;
						// Zero-copy views of envSet buffers (read-only during inference; copied into trajectories below).
						torch::Tensor tStates = torch::from_blob(
							envSet->state.obs.data.data(),
							{ (int64_t)envSet->state.obs.size[0], (int64_t)envSet->state.obs.size[1] },
							torch::kFloat32
						);
						torch::Tensor tActionMasks = torch::from_blob(
							envSet->state.actionMasks.data.data(),
							{ (int64_t)envSet->state.actionMasks.size[0], (int64_t)envSet->state.actionMasks.size[1] },
							torch::kUInt8
						);

						if (!render) {
							const float* obsData = envSet->state.obs.data.data();
							const uint8_t* maskData = envSet->state.actionMasks.data.data();
							for (int newPlayerIdx : newPlayerIndices) {
								auto& traj = trajectories[newPlayerIdx];
								traj.states.insert(traj.states.end(), obsData + (size_t)newPlayerIdx * obsSize, obsData + (size_t)(newPlayerIdx + 1) * obsSize);
								traj.actionMasks.insert(traj.actionMasks.end(), maskData + (size_t)newPlayerIdx * numActions, maskData + (size_t)(newPlayerIdx + 1) * numActions);
							}
						}

						Timer inferTimer = {};

						if (oldVersion) {
							torch::Tensor tdNewStates = tStates.index_select(0, tNewPlayerIndices).to(ppo->device, true);
							torch::Tensor tdOldStates = tStates.index_select(0, tOldPlayerIndices).to(ppo->device, true);
							torch::Tensor tdNewActionMasks = tActionMasks.index_select(0, tNewPlayerIndices).to(ppo->device, true);
							torch::Tensor tdOldActionMasks = tActionMasks.index_select(0, tOldPlayerIndices).to(ppo->device, true);

							torch::Tensor tNewActions;
							torch::Tensor tOldActions;

							ppo->InferActions(tdNewStates, tdNewActionMasks, &tNewActions, &tLogProbs, nullptr, collectValues ? &tValues : nullptr);
							ppo->InferActions(tdOldStates, tdOldActionMasks, &tOldActions, NULL, &oldVersion->models);

							tActions = torch::zeros(numPlayers, tNewActions.dtype());
							tActions.index_copy_(0, tNewPlayerIndices, tNewActions.cpu());
							tActions.index_copy_(0, tOldPlayerIndices, tOldActions.cpu());
						} else {
							torch::Tensor tdStates = tStates.to(ppo->device, true);
							torch::Tensor tdActionMasks = tActionMasks.to(ppo->device, true);
							ppo->InferActions(tdStates, tdActionMasks, &tActions, &tLogProbs, nullptr, collectValues ? &tValues : nullptr);
							tActions = tActions.cpu();
						}
						inferTime += inferTimer.Elapsed();

						auto curActions = TENSOR_TO_VEC<int>(tActions);
						FList values, newLogProbs;
						if (collectValues) values = TENSOR_TO_VEC<float>(tValues);
						if (tLogProbs.defined() && !render)
							newLogProbs = TENSOR_TO_VEC<float>(tLogProbs);	

						stepTimer.Reset();
						envSet->Sync(); // Make sure the first half is done
						envSet->StepSecondHalf(curActions, false);
						envStepTime += stepTimer.Elapsed();

						if (stepCallback)
							stepCallback(this, envSet->state.gameStates, report);

						if (render) {
							renderSender->Send(envSet->state.gameStates[0]);
							continue;
						}

						// Calc average rewards
						if (config.addRewardsToMetrics && (Math::RandInt(0, config.rewardSampleRandInterval) == 0)) {
							int numSamples = RS_MIN(envSet->arenas.size(), config.maxRewardSamples);
							std::unordered_map<std::string, AvgTracker> avgRewards = {};
							for (int i = 0; i < numSamples; i++) {
								int arenaIdx = Math::RandInt(0, envSet->arenas.size());
								auto& prevRewards = envSet->state.lastRewards[arenaIdx];

								for (int j = 0; j < envSet->rewards[arenaIdx].size(); j++) {
									std::string rewardName = envSet->rewards[arenaIdx][j].reward->GetName();
									avgRewards[rewardName] += prevRewards[j];
								}
							}

							for (auto& pair : avgRewards)
								report.AddAvg("Rewards/" + pair.first, pair.second.Get());
						}

						// Now that we've inferred and stepped the env, we can add that stuff to the trajectories
						bool trackModes = (config.ppo.normalizeAdvantages == AdvantageNormMode::PER_MODE);
						int i = 0;
						for (int newPlayerIdx : newPlayerIndices) {
							trajectories[newPlayerIdx].actions.push_back(curActions[newPlayerIdx]);
							trajectories[newPlayerIdx].rewards += envSet->state.rewards[newPlayerIdx];
							trajectories[newPlayerIdx].logProbs += newLogProbs[i];
							if (collectValues) trajectories[newPlayerIdx].values.push_back(values[i]);
							if (trackModes) trajectories[newPlayerIdx].modes.push_back(playerModes[newPlayerIdx]);
							i++;
						}

						auto curTerminals = std::vector<uint8_t>(numPlayers, 0);
						for (int idx = 0; idx < envSet->arenas.size(); idx++) {
							uint8_t terminalType = envSet->state.terminals[idx];
							if (!terminalType)
								continue;

							auto playerStartIdx = envSet->state.arenaPlayerStartIdx[idx];
							int playersInArena = envSet->state.gameStates[idx].players.size();
							for (int i = 0; i < playersInArena; i++)
								curTerminals[playerStartIdx + i] = terminalType;
						}

						bool endOfRollout = config.ppo.fixedRollout && stepsCollected + numRealPlayers >= config.ppo.tsPerItr;
						for (int newPlayerIdx : newPlayerIndices) {
							int8_t terminalType = curTerminals[newPlayerIdx];
							auto& traj = trajectories[newPlayerIdx];

							if (!terminalType && (traj.Length() >= maxEpisodeLength || endOfRollout)) {
								// Bootstrap at a duration limit or the on-policy rollout boundary
								// This won't actually reset the env, but rather will just add it to experience buffer as truncated
								terminalType = RLGC::TerminalType::TRUNCATED;
							}

							traj.terminals.push_back(terminalType);
							if (terminalType) {

								if (terminalType == RLGC::TerminalType::TRUNCATED) {
									// Truncation requires an additional next state for the critic
									traj.nextStates += envSet->state.obs.GetRow(newPlayerIdx);
								}

								combinedTraj.Append(traj);
								traj.Clear();
							}
						}
					}

					report["Inference Time"] = inferTime;
					report["Env Step Time"] = envStepTime;
				}
				float collectionTime = collectionTimer.Elapsed();

				// While learning, the PPO learner runs its own threads (one minibatch each), torch itself should use one thread in each of them
				if (learnThreads > 0)
					torch::set_num_threads(1);

				Timer consumptionTimer = {};
				{ // Process timesteps
					RG_NO_GRAD;

					// Make and transpose tensors
					torch::Tensor tStates = torch::tensor(combinedTraj.states).reshape({ -1, obsSize });
					torch::Tensor tActionMasks = torch::tensor(combinedTraj.actionMasks).reshape({ -1, numActions });
					torch::Tensor tActions = torch::tensor(combinedTraj.actions);
					torch::Tensor tLogProbs = torch::tensor(combinedTraj.logProbs);
					torch::Tensor tRewards = torch::tensor(combinedTraj.rewards);
					torch::Tensor tTerminals = torch::tensor(combinedTraj.terminals);

					// States we truncated at (there could be none)
					torch::Tensor tNextTruncStates;
					if (!combinedTraj.nextStates.empty())
						tNextTruncStates = torch::tensor(combinedTraj.nextStates).reshape({ -1, obsSize });

					report["Average Step Reward"] = tRewards.mean().item<float>();
					report["Collected Timesteps"] = stepsCollected;

					// The timesteps of the episodes that ended in this iteration, which are the ones given to the learner now.
					// (Steps of unfinished episodes are given to the learner in the iteration their episode ends in,
					//	so over many iterations this is the same amount as the collected timesteps)
					report["Experience Timesteps"] = (double)combinedTraj.Length();
					
					torch::Tensor tValPreds;
					torch::Tensor tTruncValPreds;

					// Predict values, in parts of the PPO learner's minibatch size (limits memory use, and is faster on a CPU than all at once)
					tValPreds = collectValues ? torch::tensor(combinedTraj.values) : ppo->InferCriticInParts(tStates);
					if (tNextTruncStates.defined())
						tTruncValPreds = ppo->InferCriticInParts(tNextTruncStates);

					report["Episode Length"] = 1.f / (tTerminals == 1).to(torch::kFloat32).mean().item<float>();

					Timer gaeTimer = {};
					// Run GAE
					torch::Tensor tAdvantages, tTargetVals, tReturns;
					float rewClipPortion;
					GAE::Compute(
						tRewards, tTerminals, tValPreds, tTruncValPreds,
						tAdvantages, tTargetVals, tReturns, rewClipPortion,
						config.ppo.gaeGamma, config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange
					);
					report["GAE Time"] = gaeTimer.Elapsed();
					report["Clipped Reward Portion"] = rewClipPortion;

					if (returnStat) {
						report["GAE/Returns STD"] = returnStat->GetSTD();

						int numToIncrement = RS_MIN(config.maxReturnSamples, tReturns.size(0));
						if (numToIncrement > 0) {
							if (config.maxReturnStatCount > 0)
								returnStat->CapCount(config.maxReturnStatCount);
							auto selectedReturns = tReturns.index_select(0, torch::randint(tReturns.size(0), { (int64_t)numToIncrement }));
							returnStat->Increment(TENSOR_TO_VEC<float>(selectedReturns));
						}
					}
					report["GAE/Avg Return"] = tReturns.abs().mean().item<float>();
					report["GAE/Avg Advantage"] = tAdvantages.abs().mean().item<float>();
					report["GAE/Avg Val Target"] = tTargetVals.abs().mean().item<float>();
					{
						float targetVar = tTargetVals.var(false).item<float>();
						if (targetVar > 1e-8f) {
							float errVar = (tTargetVals - tValPreds.slice(0, 0, tTargetVals.size(0))).var(false).item<float>();
							report["Critic Explained Variance"] = 1.f - errVar / targetVar;
						}
					}

					// Normalize policy advantages AFTER tTargetVals has been computed, so critic targets remain in
					// exact return-standardized units while policy gradients are zero-mean and balanced across modes.
					if (config.ppo.normalizeAdvantages == AdvantageNormMode::ALL && tAdvantages.size(0) >= 8) {
						float mean = tAdvantages.mean().item<float>();
						float std = std::max(tAdvantages.std(false).item<float>(), 1e-5f);
						report["GAE/Raw Adv STD"] = std;
						tAdvantages = (tAdvantages - mean) * (1.f / std);
					} else if (config.ppo.normalizeAdvantages == AdvantageNormMode::PER_MODE &&
					           (int64_t)combinedTraj.modes.size() == tAdvantages.size(0)) {
						tAdvantages = tAdvantages.contiguous();
						float* advPtr = tAdvantages.data_ptr<float>();
						const uint8_t* modePtr = combinedTraj.modes.data();
						int64_t n = tAdvantages.size(0);

						double sum[3] = {}, sumSq[3] = {};
						int64_t cnt[3] = {};
						for (int64_t idx = 0; idx < n; idx++) {
							uint8_t m = modePtr[idx] < 3 ? modePtr[idx] : 0;
							double v = advPtr[idx];
							sum[m] += v;
							sumSq[m] += v * v;
							cnt[m]++;
						}
						static const char* MODE_STD_NAMES[3] = {
							"GAE/Raw Adv STD 1v1", "GAE/Raw Adv STD 2v2", "GAE/Raw Adv STD 3v3"
						};
						float mean[3] = {}, invStd[3] = { 1.f, 1.f, 1.f };
						for (int m = 0; m < 3; m++) {
							if (cnt[m] >= 8) {
								double mu = sum[m] / cnt[m];
								double var = std::max(0.0, sumSq[m] / cnt[m] - mu * mu);
								double sd = std::max(std::sqrt(var), 1e-5);
								mean[m] = (float)mu;
								invStd[m] = (float)(1.0 / sd);
								report[MODE_STD_NAMES[m]] = (float)sd;
							}
						}
						for (int64_t idx = 0; idx < n; idx++) {
							uint8_t m = modePtr[idx] < 3 ? modePtr[idx] : 0;
							if (cnt[m] >= 8)
								advPtr[idx] = (advPtr[idx] - mean[m]) * invStd[m];
						}
					}

					// Set experience buffer
					experience.data.actions = tActions;
					experience.data.logProbs = tLogProbs;
					experience.data.actionMasks = tActionMasks;
					experience.data.states = tStates;
					experience.data.advantages = tAdvantages;
					experience.data.targetValues = tTargetVals;
				}

				// Free CUDA cache
#ifdef RG_CUDA_SUPPORT
				if (ppo->device.is_cuda())
					c10::cuda::CUDACachingAllocator::emptyCache();
#endif

				// Learn
				Timer learnTimer = {};
				ppo->Learn(experience, report, isFirstIteration);
				report["PPO Learn Time"] = learnTimer.Elapsed();

				// Set collection/consumption metrics
				float consumptionTime = consumptionTimer.Elapsed();
				report["Collection Time"] = collectionTime;
				report["Consumption Time"] = consumptionTime;
				report["Collection Steps/Second"] = stepsCollected / collectionTime;
				report["Consumption Steps/Second"] = stepsCollected / consumptionTime;

				uint64_t prevTimesteps = totalTimesteps;
				totalTimesteps += stepsCollected;
				if (!quitRequested && config.maxTimesteps && totalTimesteps >= config.maxTimesteps) {
					quitRequested = true;
					rateBeforeQuit = true;
					RG_LOG("Stopping after this iteration (timestep limit reached).");
				}
				report["Total Timesteps"] = totalTimesteps;
				totalIterations++;
				report["Total Iterations"] = totalIterations;

				bool ratingSessionRan = false;
				if (versionMgr) {
					versionMgr->skill.sessionAbortCheck = shouldAbortRating;
					versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps, quitRequested && rateBeforeQuit);
					ratingSessionRan = versionMgr->skill.sessionJustRan;
				}

				bool shouldQuit = saveQueued || quitRequested;

				if (!config.checkpointFolder.empty()) {
					// NOTE: Also after a rating session, to have its rating and the new policy version on disk
					if (shouldQuit || ratingSessionRan || (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave)) {
						// Auto-save (or final save before quitting)
						Save();
					}
				}

				// Contiguous wall-clock delta since the previous report boundary (or Learner construction on
				// the first iteration): includes previous metric/console output, opponent switching, rollout,
				// GAE, PPO updates, skill-tracker rating rounds, policy-version writes, and checkpoint saves,
				// with zero unmeasured time between iterations.
				auto wallNow = std::chrono::steady_clock::now();
				double iterationWallTime = std::chrono::duration<double>(wallNow - lastIterationWallEnd).count();
				lastIterationWallEnd = wallNow;
				if (iterationWallTime < 1e-9)
					iterationWallTime = 1e-9;
				report["Iteration Time"] = iterationWallTime;
				report["Overhead Time"] = std::max(0.0, iterationWallTime - ((double)collectionTime + (double)consumptionTime));
				report["Overall Steps/Second"] = (double)stepsCollected / iterationWallTime;

				report.Finish();

				if (metricSender)
					metricSender->Send(report);

				report.Display(
					{
						"Average Step Reward",
						"Policy Entropy",
						"Mean KL Divergence",
						"First Accuracy",
						"",
						"Policy Update Magnitude",
						"Critic Update Magnitude",
						"Shared Head Update Magnitude",
						"",
						"Collection Steps/Second",
						"Consumption Steps/Second",
						"Overall Steps/Second",
						"",
						"Collection Time",
						"-Inference Time",
						"-Env Step Time",
						"Consumption Time",
						"-GAE Time",
						"-PPO Learn Time",
						"",
						"Collected Timesteps",
						"Total Timesteps",
						"Total Iterations"
					}
				);

				if (shouldQuit) {
					// NOTE: The final checkpoint is saved and the final report has been sent/displayed at this point.
					// We return to the caller (instead of exit(0)) so it can decide what to do next.
					RG_LOG("Quit requested, stopping after " << totalTimesteps << " total timesteps.");
					std::cout << std::flush;
					quitRequested = false;
					return;
				}
			}
		}
		
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during main learner loop: " << e.what());
	}
}

GGL::Learner::~Learner() {
	delete ppo;
	delete versionMgr;
	delete metricSender;
	delete renderSender;
	pybind11::finalize_interpreter();
}