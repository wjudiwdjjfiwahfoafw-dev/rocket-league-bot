#pragma once
#include "ExperienceBuffer.h";
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <GigaLearnCPP/Util/AvgTracker.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <GigaLearnCPP/PPO/TransferLearnConfig.h>

#include "../Util/Models.h"

#include <torch/optim/adam.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/modules/container/sequential.h>

#include "ExperienceBuffer.h"

namespace GGL {

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	class PPOLearner {
	public:
		ModelSet models = {};
		ModelSet guidingPolicyModels = {};

		PPOLearnerConfig config;
		torch::Device device;

		// How many samples are sent through the models at once in Learn() and InferCriticInParts().
		// It is config.miniBatchSize if that was set. Otherwise it is the whole batch on a GPU, 
		//	and a piece that fits into the CPU cache on a CPU (which is faster than the whole batch at once, and needs far less memory).
		// The gradients of all minibatches of a batch are added up before the optimizers step, 
		//	so this has no effect on the result.
		int64_t miniBatchSize;
		constexpr static int64_t CPU_DEFAULT_MINIBATCH_SIZE = 1024, CPU_MIN_MINIBATCH_SIZE = 256;

		// Sets numLearnThreads, and (if config.miniBatchSize was not set) a minibatch size that gives every thread the same number of minibatches
		void SetNumLearnThreads(int numThreads);
		bool _autoMiniBatchSize = false;

		// CPU only: The number of threads that work on a batch in Learn() and InferCriticInParts().
		// Every thread takes its own minibatches, and the results are combined.
		int numLearnThreads = 1;

		// CPU only: The number of threads that InferActions() splits its observations between.
		// SetNumLearnThreads() sets it to numLearnThreads if the policy is big enough for that to pay off, otherwise to 1:
		//	While the actions are inferred, the thread pool is also simulating the games. For a small policy that takes longer
		//	than the inference, so one thread is enough for it. A big policy would leave all other threads waiting for that one.
		int numInferThreads = 1;
		constexpr static uint64_t INFER_THREADS_MIN_PARAMS = 1'000'000; // Parameters of the shared head and the policy together
		constexpr static int64_t INFER_THREADS_MIN_ROWS = 16; // Observations per thread

		// Averages of the values that Learn() reports
		struct LearnStats {
			AvgTracker entropy, divergence, policyLoss, relEntropyLoss, criticLoss, guidingLoss, ratio, clip;
			float lastBatchKL = 0;
		};

		// The same values for a single minibatch
		struct MiniBatchStats {
			int64_t count = 0;
			float entropy = 0, divergence = 0, policyLoss = 0, criticLoss = 0, guidingLoss = 0, ratio = 0, clipFraction = 0;
		};

		PPOLearner(
			int obsSize, int numActions,
			PPOLearnerConfig config, torch::Device device
		);

		static void MakeModels(
			bool makeCritic, 
			int obsSize, int numActions, 
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
			torch::Device device,
			ModelSet& outModels
		);
		
		// If models is null, this->models will be used
		void InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models = NULL, torch::Tensor* outValues = nullptr);
		torch::Tensor InferCritic(torch::Tensor obs);

		// Same as InferCritic(), but the observations (on any device) are moved to our device and evaluated in parts of miniBatchSize.
		// Returns a CPU tensor.
		torch::Tensor InferCriticInParts(torch::Tensor obs);

		// Action probabilities from the output of the policy model
		static torch::Tensor PolicyProbsFromLogits(torch::Tensor logits, torch::Tensor actionMasks, float temperature);

		// Perhaps they should be somewhere else? Should probably make an inference interface...
		static torch::Tensor InferPolicyProbsFromModels(
			ModelSet& models, 
			torch::Tensor obs, torch::Tensor actionMasks, 
			float temperature,
			bool halfPrec, torch::Tensor* outValues = nullptr
		);
		static void InferActionsFromModels(
			ModelSet& models, 
			torch::Tensor obs, torch::Tensor actionMasks, 
			bool deterministic, float temperature, bool halfPrec,
			torch::Tensor* outActions, torch::Tensor* outLogProbs
		);

		// Picks an action from the probabilities of each row: the most likely one if deterministic, otherwise a random one
		static void ChooseActions(torch::Tensor probs, bool deterministic, torch::Tensor* outActions, torch::Tensor* outLogProbs);

		void Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration);

		// Adds the gradients of the PPO loss of one batch to the models. Used by Learn(), which then steps the optimizers.
		void AccumulateBatchGrads(const ExperienceTensors& batch, bool trainPolicy, bool trainCritic, LearnStats& stats);

		// The PPO loss of the samples [start, stop) of a batch, weighted by their share of the batch
		torch::Tensor ComputeMiniBatchLoss(
			const ExperienceTensors& batch, int64_t start, int64_t stop, int64_t batchSize,
			bool trainPolicy, bool trainCritic, MiniBatchStats& outStats
		);

		void TransferLearn(
			ModelSet& oldModels, 
			torch::Tensor newObs, torch::Tensor oldObs, 
			torch::Tensor newActionMasks, torch::Tensor oldActionMasks, 
			torch::Tensor actionMaps,
			Report& report, 
			const TransferLearnConfig& transferLearnConfig
		);

		void SaveTo(std::filesystem::path folderPath);
		void LoadFrom(std::filesystem::path folderPath);
		void SetLearningRates(float policyLR, float criticLR);

		ModelSet GetPolicyModels();
	};
}