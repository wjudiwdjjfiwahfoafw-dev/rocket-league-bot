#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <public/GigaLearnCPP/Util/AvgTracker.h>
#include <torch/autograd.h>
#include <RLGymCPP/ThreadPool.h>

using namespace torch;

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device) : config(_config), device(_device) {

	if (config.miniBatchSize > 0) {
		miniBatchSize = config.miniBatchSize;
	} else {
		_autoMiniBatchSize = true;
		miniBatchSize = device.is_cpu() ? CPU_DEFAULT_MINIBATCH_SIZE : config.batchSize;
		config.miniBatchSize = config.batchSize;
	}
	miniBatchSize = RS_CLAMP(miniBatchSize, (int64_t)1, config.batchSize);

	MakeModels(true, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, models);

	SetLearningRates(config.policyLR, config.criticLR);

	// Print param counts
	RG_LOG("Model parameter counts:");
	uint64_t total = 0;
	for (auto model : this->models) {
		uint64_t count = model->GetParamCount();
		RG_LOG("\t\"" << model->modelName << "\": " << Utils::NumToStr(count));
		total += count;
	}
	RG_LOG("\t[Total]: " << Utils::NumToStr(total));

	if (config.useGuidingPolicy) {
		RG_LOG("Guiding policy enabled, loading from " << config.guidingPolicyPath << "...");
		MakeModels(false, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, guidingPolicyModels);
		guidingPolicyModels.Load(config.guidingPolicyPath, false, false);
	}
}

void GGL::PPOLearner::SetNumLearnThreads(int numThreads) {
	numLearnThreads = RS_MAX(numThreads, 1);

	if (device.is_cpu() && _autoMiniBatchSize) {
		// Minibatches of about the default size, and the same number of them for every thread.
		// (With 3 minibatches on 2 threads, one thread would wait while the other does its second minibatch)
		int64_t numMiniBatches = (config.batchSize + CPU_DEFAULT_MINIBATCH_SIZE - 1) / CPU_DEFAULT_MINIBATCH_SIZE;
		numMiniBatches = ((numMiniBatches + numLearnThreads - 1) / numLearnThreads) * numLearnThreads;
		miniBatchSize = (config.batchSize + numMiniBatches - 1) / numMiniBatches;
		miniBatchSize = RS_MAX(miniBatchSize, CPU_MIN_MINIBATCH_SIZE);
		miniBatchSize = RS_CLAMP(miniBatchSize, (int64_t)1, config.batchSize);
	}

	uint64_t inferParams = models["policy"]->GetParamCount();
	if (models["shared_head"])
		inferParams += models["shared_head"]->GetParamCount();
	numInferThreads = (device.is_cpu() && inferParams >= INFER_THREADS_MIN_PARAMS) ? numLearnThreads : 1;
}

void GGL::PPOLearner::MakeModels(
	bool makeCritic,
	int obsSize, int numActions, 
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
	torch::Device device, 
	ModelSet& outModels) {

	ModelConfig fullPolicyConfig = policyConfig;
	fullPolicyConfig.numInputs = obsSize;
	fullPolicyConfig.numOutputs = numActions;

	ModelConfig fullCriticConfig = criticConfig;
	fullCriticConfig.numInputs = obsSize;
	fullCriticConfig.numOutputs = 1;

	if (sharedHeadConfig.IsValid()) {

		ModelConfig fullSharedHeadConfig = sharedHeadConfig;
		fullSharedHeadConfig.numInputs = obsSize;
		fullSharedHeadConfig.numOutputs = 0;

		RG_ASSERT(!sharedHeadConfig.addOutputLayer);

		fullPolicyConfig.numInputs = fullSharedHeadConfig.layerSizes.back();
		fullCriticConfig.numInputs = fullSharedHeadConfig.layerSizes.back();

		outModels.Add(new Model("shared_head", fullSharedHeadConfig, device));
	}

	outModels.Add(new Model("policy", fullPolicyConfig, device));

	if (makeCritic)
		outModels.Add(new Model("critic", fullCriticConfig, device));
}

torch::Tensor GGL::PPOLearner::PolicyProbsFromLogits(torch::Tensor logits, torch::Tensor actionMasks, float temperature) {

	actionMasks = actionMasks.to(torch::kBool);

	constexpr float ACTION_MIN_PROB = 1e-11f;
	constexpr float ACTION_DISABLED_LOGIT = -1e10f;

	if (temperature != 1)
		logits = logits / temperature;

	auto result = torch::softmax(logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(), -1);
	return result.view({ -1, logits.size(-1) }).clamp(ACTION_MIN_PROB, 1);
}

torch::Tensor GGL::PPOLearner::InferPolicyProbsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	float temperature, bool halfPrec, torch::Tensor* outValues) {

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec);

	if (outValues)
		*outValues = models["critic"]->Forward(obs, halfPrec).flatten();
	auto logits = models["policy"]->Forward(obs, halfPrec);
	return PolicyProbsFromLogits(logits, actionMasks, temperature);
}

void GGL::PPOLearner::InferActionsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks, 
	bool deterministic, float temperature, bool halfPrec,
	torch::Tensor* outActions, torch::Tensor* outLogProbs) {

	auto probs = InferPolicyProbsFromModels(models, obs, actionMasks, temperature, halfPrec);
	ChooseActions(probs, deterministic, outActions, outLogProbs);
}

void GGL::PPOLearner::ChooseActions(torch::Tensor probs, bool deterministic, torch::Tensor* outActions, torch::Tensor* outLogProbs) {
	if (deterministic) {
		auto action = probs.argmax(1);
		if (outActions)
			*outActions = action.flatten();
	} else {
		auto action = torch::multinomial(probs, 1, true);
		if (outActions)
			*outActions = action.flatten();

		if (outLogProbs)
			*outLogProbs = probs.gather(-1, action).log().flatten();
	}
}

void GGL::PPOLearner::InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models, torch::Tensor* outValues) {
	ModelSet& inferModels = models ? *models : this->models;

	int64_t amount = obs.size(0);
	int numThreads = (int)RS_MIN((int64_t)numInferThreads, amount / INFER_THREADS_MIN_ROWS);
	if (!device.is_cpu() || numThreads <= 1 || config.useHalfPrecision || torch::GradMode::is_enabled()) {
		auto probs = InferPolicyProbsFromModels(inferModels, obs, actionMasks, config.policyTemperature, config.useHalfPrecision, outValues);
		ChooseActions(probs, config.deterministic, outActions, outLogProbs);
		return;
	}

	// Several threads, each computes the action probabilities of its share of the observations (same scheme as in AccumulateBatchGrads()).
	// The actions are then chosen on this thread, so they don't depend on thread timing.
	std::vector<torch::Tensor> partProbs(numThreads), partValues(outValues ? numThreads : 0);
	std::vector<std::exception_ptr> threadErrors(numThreads);
	auto fnThread = [&](int threadIdx) {
		try {
			RG_NO_GRAD;
			torch::set_num_threads(1);
			int64_t start = amount * threadIdx / numThreads;
			int64_t stop = amount * (threadIdx + 1) / numThreads;
			partProbs[threadIdx] = InferPolicyProbsFromModels(
				inferModels, obs.slice(0, start, stop), actionMasks.slice(0, start, stop), config.policyTemperature, false,
				outValues ? &partValues[threadIdx] : nullptr
			);
		} catch (...) {
			threadErrors[threadIdx] = std::current_exception();
		}
	};

	for (int i = 1; i < numThreads; i++)
		RLGC::g_ThreadPool.StartJobAsync(fnThread, i);
	fnThread(0);
	RLGC::g_ThreadPool.WaitUntilDone();

	for (auto& error : threadErrors)
		if (error)
			std::rethrow_exception(error);

	if (outValues)
		*outValues = torch::cat(partValues, 0);
	ChooseActions(torch::cat(partProbs, 0), config.deterministic, outActions, outLogProbs);
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs) {

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, config.useHalfPrecision);

	return models["critic"]->Forward(obs, config.useHalfPrecision).flatten();
}

torch::Tensor GGL::PPOLearner::InferCriticInParts(torch::Tensor obs) {
	int64_t amount = obs.size(0);
	if (amount <= miniBatchSize)
		return InferCritic(obs.to(device, true)).cpu();

	int64_t numParts = (amount + miniBatchSize - 1) / miniBatchSize;
	torch::Tensor result = torch::empty({ amount });

	auto fnRunPart = [&](int64_t partIdx) {
		RG_NO_GRAD;
		int64_t start = partIdx * miniBatchSize;
		int64_t stop = RS_MIN(start + miniBatchSize, amount);
		auto part = InferCritic(obs.slice(0, start, stop).to(device, true)).cpu();
		RG_ASSERT(part.size(0) == (stop - start));
		result.slice(0, start, stop).copy_(part);
	};

	int numThreads = (int)RS_MIN((int64_t)numLearnThreads, numParts);
	if (!device.is_cpu() || numThreads <= 1) {
		for (int64_t i = 0; i < numParts; i++)
			fnRunPart(i);
	} else {
		// Same scheme as in AccumulateBatchGrads()
		std::vector<std::exception_ptr> threadErrors(numThreads);
		auto fnThread = [&](int threadIdx) {
			try {
				torch::set_num_threads(1);
				for (int64_t i = threadIdx; i < numParts; i += numThreads)
					fnRunPart(i);
			} catch (...) {
				threadErrors[threadIdx] = std::current_exception();
			}
		};

		for (int i = 1; i < numThreads; i++)
			RLGC::g_ThreadPool.StartJobAsync(fnThread, i);
		fnThread(0);
		RLGC::g_ThreadPool.WaitUntilDone();

		for (auto& error : threadErrors)
			if (error)
				std::rethrow_exception(error);
	}

	return result;
}

torch::Tensor ComputeEntropy(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy) {
	// Compute log probs and entropy
	auto entropy = -(probs.log() * probs).sum(-1);

	if (maskEntropy) {
		// Account for action masking in entropy
		// We will effectively narrow the entropy to the scope of the valid actions
		// This way states with more masked actions don't just have inherently lower entropy
		entropy /= actionMasks.to(torch::kFloat32).sum(-1).log();
	} else {
		entropy /= logf(actionMasks.size(-1));
	}

	return entropy.mean();
}

torch::Tensor GGL::PPOLearner::ComputeMiniBatchLoss(
	const ExperienceTensors& batch, int64_t start, int64_t stop, int64_t batchSize,
	bool trainPolicy, bool trainCritic, MiniBatchStats& outStats) {

	int64_t count = stop - start;
	outStats = {};
	outStats.count = count;

	// The losses are averages over the minibatch, so they are weighted by its share of the batch
	float batchSizeRatio = count / (float)batchSize;

	// Send everything to the device and enforce correct shapes
	auto acts = batch.actions.slice(0, start, stop).to(device, true);
	auto obs = batch.states.slice(0, start, stop).to(device, true);
	auto actionMasks = batch.actionMasks.slice(0, start, stop).to(device, true);

	auto advantages = batch.advantages.slice(0, start, stop).to(device, true);
	auto oldProbs = batch.logProbs.slice(0, start, stop).to(device, true);
	auto targetValues = batch.targetValues.slice(0, start, stop).to(device, true);

	// The shared head is run once and its output goes to both the policy and the critic.
	// (It used to be run separately for each, which gives the same gradients for more work)
	torch::Tensor headOutput = obs;
	if (Model* sharedHead = models["shared_head"])
		headOutput = sharedHead->Forward(obs, false);

	torch::Tensor totalLoss;

	torch::Tensor probs, logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
	if (trainPolicy) {

		// Get policy log probs and entropy
		{
			auto logits = models["policy"]->Forward(headOutput, false);
			probs = PolicyProbsFromLogits(logits, actionMasks, config.policyTemperature);
			logProbs = probs.log().gather(-1, acts.unsqueeze(-1));
			entropy = ComputeEntropy(probs, actionMasks, config.maskEntropy);
			outStats.entropy = entropy.detach().cpu().item<float>();
		}

		logProbs = logProbs.view_as(oldProbs);

		// Compute PPO loss
		ratio = exp(logProbs - oldProbs);
		outStats.ratio = ratio.mean().detach().cpu().item<float>();
		clipped = clamp(
			ratio, 1 - config.clipRange, 1 + config.clipRange
		);

		// Compute policy loss
		policyLoss = -min(
			ratio * advantages, clipped * advantages
		).mean();
		outStats.policyLoss = policyLoss.detach().cpu().item<float>();

		ppoLoss = (policyLoss - entropy * config.entropyScale) * batchSizeRatio;

		if (config.useGuidingPolicy) {
			torch::Tensor guidingProbs;
			{
				RG_NO_GRAD;
				guidingProbs = InferPolicyProbsFromModels(guidingPolicyModels, obs, actionMasks, config.policyTemperature, config.useHalfPrecision);
			}

			auto guidingLoss = (guidingProbs - probs).abs().mean();
			outStats.guidingLoss = guidingLoss.detach().cpu().item<float>();
			guidingLoss = guidingLoss * (config.guidingStrength * batchSizeRatio);
			ppoLoss = ppoLoss + guidingLoss;
		}

		totalLoss = ppoLoss;
	}

	if (trainCritic) {
		auto vals = models["critic"]->Forward(headOutput, false).flatten();

		// Compute value loss
		vals = vals.view_as(targetValues);
		auto valueLoss = torch::mse_loss(vals, targetValues);
		outStats.criticLoss = valueLoss.detach().cpu().item<float>();

		auto criticLoss = valueLoss * batchSizeRatio;
		totalLoss = totalLoss.defined() ? (totalLoss + criticLoss) : criticLoss;
	}

	if (trainPolicy) {
		// Compute KL divergence & clip fraction using SB3 method for reporting
		RG_NO_GRAD;

		auto logRatio = logProbs - oldProbs;
		auto klTensor = (exp(logRatio) - 1) - logRatio;
		outStats.divergence = klTensor.mean().detach().cpu().item<float>();

		auto clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat));
		outStats.clipFraction = clipFraction.cpu().item<float>();
	}

	return totalLoss;
}

void GGL::PPOLearner::AccumulateBatchGrads(const ExperienceTensors& batch, bool trainPolicy, bool trainCritic, LearnStats& stats) {

	if (!trainPolicy && !trainCritic)
		return;

	// NOTE: With overbatching, the last batch of an epoch has more than config.batchSize samples (all that are left over).
	//	Like the original implementation, this only learns from the first config.batchSize samples of a batch,
	//	so the left-over samples of an epoch are not used in that epoch (every epoch shuffles, so they differ each time).
	//	Using them would be a change to the training itself (more samples per update and more learning time), not a speed-up.
	int64_t batchSize = RS_MIN((int64_t)batch.states.size(0), (int64_t)config.batchSize);

	int64_t numMiniBatches = (batchSize + miniBatchSize - 1) / miniBatchSize;
	std::vector<MiniBatchStats> miniBatchStats(numMiniBatches);

	auto fnGetLoss = [&](int64_t miniBatchIdx) -> torch::Tensor {
		int64_t start = miniBatchIdx * miniBatchSize;
		int64_t stop = RS_MIN(start + miniBatchSize, batchSize);
		return ComputeMiniBatchLoss(batch, start, stop, batchSize, trainPolicy, trainCritic, miniBatchStats[miniBatchIdx]);
	};

	int numThreads = (int)RS_MIN((int64_t)numLearnThreads, numMiniBatches);
	if (!device.is_cpu() || numThreads <= 1) {
		// One minibatch after the other, the gradients add up in the parameters
		for (int64_t i = 0; i < numMiniBatches; i++)
			fnGetLoss(i).backward();

	} else {
		// Several threads, each computes the gradients of its own minibatches.
		// Unlike backward(), autograd::grad() doesn't write to the parameters, so the threads don't get in each other's way.
		// The gradients of all threads are added up at the end.
		std::vector<torch::Tensor> params;
		for (Model* model : models)
			for (auto& param : model->parameters())
				if (param.requires_grad())
					params.push_back(param);

		std::vector<std::vector<torch::Tensor>> threadGrads(numThreads);
		std::vector<std::exception_ptr> threadErrors(numThreads);

		auto fnThread = [&](int threadIdx) {
			try {
				// The threads are the parallelism, torch itself should not start more threads inside of them
				torch::set_num_threads(1);

				auto& gradSums = threadGrads[threadIdx];

				// Minibatches are assigned in a fixed order, so the result doesn't depend on thread timing
				for (int64_t i = threadIdx; i < numMiniBatches; i += numThreads) {
					auto grads = torch::autograd::grad({ fnGetLoss(i) }, params, {}, false, false, true);

					if (gradSums.empty()) {
						gradSums = grads;
					} else {
						RG_NO_GRAD;
						for (size_t j = 0; j < grads.size(); j++) {
							if (!grads[j].defined())
								continue;

							if (gradSums[j].defined()) {
								gradSums[j] += grads[j];
							} else {
								gradSums[j] = grads[j];
							}
						}
					}
				}
			} catch (...) {
				threadErrors[threadIdx] = std::current_exception();
			}
		};

		for (int i = 1; i < numThreads; i++)
			RLGC::g_ThreadPool.StartJobAsync(fnThread, i);
		fnThread(0);
		RLGC::g_ThreadPool.WaitUntilDone();

		for (auto& error : threadErrors)
			if (error)
				std::rethrow_exception(error);

		// Add everything to the gradients of the parameters, like backward() would have
		RG_NO_GRAD;
		for (size_t j = 0; j < params.size(); j++) {
			torch::Tensor total;
			for (auto& gradSums : threadGrads) {
				if (gradSums.empty() || !gradSums[j].defined())
					continue;

				if (total.defined()) {
					total += gradSums[j];
				} else {
					// A tensor of its own (what autograd returns can be a view of another tensor, which can't be used as a parameter's gradient)
					total = gradSums[j].clone(at::MemoryFormat::Contiguous);
				}
			}

			if (!total.defined())
				continue;

			if (params[j].grad().defined()) {
				params[j].mutable_grad() += total;
			} else {
				params[j].mutable_grad() = total;
			}
		}
	}

	// Add up the reported values. The minibatches only exist to save time and memory, so everything is an average over the batch.
	double batchEntropySum = 0, batchPolicyLossSum = 0;
	for (auto& mb : miniBatchStats) {
		if (trainPolicy) {
			stats.entropy.Add(mb.entropy * mb.count, mb.count);
			stats.ratio.Add(mb.ratio * mb.count, mb.count);
			stats.policyLoss.Add(mb.policyLoss * mb.count, mb.count);
			stats.divergence.Add(mb.divergence * mb.count, mb.count);
			stats.clip.Add(mb.clipFraction * mb.count, mb.count);
			if (config.useGuidingPolicy)
				stats.guidingLoss.Add(mb.guidingLoss * mb.count, mb.count);

			batchEntropySum += (double)mb.entropy * mb.count;
			batchPolicyLossSum += (double)mb.policyLoss * mb.count;
		}

		if (trainCritic)
			stats.criticLoss.Add(mb.criticLoss * mb.count, mb.count);
	}

	if (trainPolicy) {
		// Size of the entropy term relative to the policy loss, for the whole batch
		float batchEntropy = (float)(batchEntropySum / batchSize);
		float batchPolicyLoss = (float)(batchPolicyLossSum / batchSize);
		stats.relEntropyLoss += (batchEntropy * config.entropyScale) / batchPolicyLoss;
	}
}

void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {

	LearnStats stats = {};

	// Save parameters first
	auto policyBefore = models["policy"]->CopyParams();
	auto criticBefore = models["critic"]->CopyParams();

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;
	bool trainSharedHead = models["shared_head"] && (trainPolicy || trainCritic);

	for (int epoch = 0; epoch < config.epochs; epoch++) {

		// Get randomly-ordered timesteps for PPO
		auto batches = experience.GetAllBatchesShuffled(config.batchSize, config.overbatching);

		for (auto& batch : batches) {
			AccumulateBatchGrads(batch, trainPolicy, trainCritic, stats);

			if (trainPolicy)
				nn::utils::clip_grad_norm_(models["policy"]->parameters(), 0.5f);
			if (trainCritic)
				nn::utils::clip_grad_norm_(models["critic"]->parameters(), 0.5f);

			if (trainSharedHead)
				nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), 0.5f);

			models.StepOptims();
		}
	}

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = models["policy"]->CopyParams();
	auto criticAfter = models["critic"]->CopyParams();

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

	// Assemble and return report
	report["Policy Entropy"] = stats.entropy.Get();
	report["Mean KL Divergence"] = stats.divergence.Get();
	if (!isFirstIteration) {
		// These metrics give bad data on the first iteration, which will mess up graph scaling
		// So we'll just skip them for the first iteration
		report["Policy Loss"] = stats.policyLoss.Get();
		report["Policy Relative Entropy Loss"] = stats.relEntropyLoss.Get();
		report["Critic Loss"] = stats.criticLoss.Get();

		if (config.useGuidingPolicy)
			report["Guiding Loss"] = stats.guidingLoss.Get();

		report["SB3 Clip Fraction"] = stats.clip.Get();
		report["Policy Update Magnitude"] = policyUpdateMagnitude;
		report["Critic Update Magnitude"] = criticUpdateMagnitude;
	}
}

void GGL::PPOLearner::TransferLearn(
	ModelSet& oldModels,
	torch::Tensor newObs, torch::Tensor oldObs,
	torch::Tensor newActionMasks, torch::Tensor oldActionMasks,
	torch::Tensor actionMaps,
	Report& report,
	const TransferLearnConfig& tlConfig
) {

	torch::Tensor oldProbs;
	{ // No grad for old model inference
		RG_NO_GRAD;
		oldProbs = InferPolicyProbsFromModels(oldModels, oldObs, oldActionMasks, config.policyTemperature, config.useHalfPrecision);
		report["Old Policy Entropy"] = ComputeEntropy(oldProbs, oldActionMasks, config.maskEntropy).detach().cpu().item<float>();

		if (actionMaps.defined())
			oldProbs = oldProbs.gather(1, actionMaps);
	}

	for (auto& model : GetPolicyModels())
		model->SetOptimLR(tlConfig.lr);

	auto policyBefore = models["policy"]->CopyParams();
	
	for (int i = 0; i < tlConfig.epochs; i++) {
		torch::Tensor newProbs = InferPolicyProbsFromModels(models, newObs, newActionMasks, config.policyTemperature, false);

		// Non-summative KL div	loss
		torch::Tensor transferLearnLoss;
		if (tlConfig.useKLDiv) {
			transferLearnLoss = (oldProbs * torch::log(oldProbs / newProbs)).abs();
		} else {
			transferLearnLoss = (oldProbs - newProbs).abs();
		}
		transferLearnLoss = transferLearnLoss.pow(tlConfig.lossExponent);
		transferLearnLoss = transferLearnLoss.mean();
		transferLearnLoss *= tlConfig.lossScale;

		if (i == 0) {
			RG_NO_GRAD;
			torch::Tensor matchingActionsMask = (newProbs.detach().argmax(-1) == oldProbs.detach().argmax(-1));
			report["Transfer Learn Accuracy"] = matchingActionsMask.to(torch::kFloat).mean().cpu().item<float>();
			report["Transfer Learn Loss"] = transferLearnLoss.detach().cpu().item<float>();

			report["Policy Entropy"] = ComputeEntropy(newProbs, newActionMasks, config.maskEntropy).detach().cpu().item<float>();
		}

		transferLearnLoss.backward();

		models.StepOptims();
	}

	auto policyAfter = models["policy"]->CopyParams();
	report["Policy Update Magnitude"] = (policyBefore - policyAfter).norm().item<float>();
}

void GGL::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	models.Save(folderPath);
}

void GGL::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	models.Load(folderPath, true, true);

	SetLearningRates(config.policyLR, config.criticLR);
}

void GGL::PPOLearner::SetLearningRates(float policyLR, float criticLR) {
	config.policyLR = policyLR;
	config.criticLR = criticLR;

	models["policy"]->SetOptimLR(policyLR);
	models["critic"]->SetOptimLR(criticLR);

	if (models["shared_head"])
		models["shared_head"]->SetOptimLR(RS_MIN(policyLR, criticLR));

	RG_LOG("PPOLearner: " << RS_STR(std::scientific << "Set learning rate to [" << policyLR << ", " << criticLR << "]"));
}

GGL::ModelSet GGL::PPOLearner::GetPolicyModels() {
	ModelSet result = {};
	for (Model* model : models) {
		if (model->modelName == "critic")
			continue;
		
		result.Add(model);
	}
	return result;
}