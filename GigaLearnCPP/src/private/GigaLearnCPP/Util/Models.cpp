#include "Models.h"
#include "FusedLayer.h"
#include "SlotPool.h"

#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/csrc/api/include/torch/nn/utils/convert_parameters.h>
#include <torch/nn/modules/normalization.h>
#include <torch/optim/adam.h>

#include <functional>

// Adds hidden layers (linear -> [layer norm] -> activation) to a sequence, returns the size of the last one
static int AddHiddenLayers(torch::nn::Sequential& seq, int numInputs, const std::vector<int>& layerSizes, bool addLayerNorm, GGL::ModelActivationType activationType) {
	int lastSize = numInputs;
	for (int i = 0; i < layerSizes.size(); i++) {
		seq->push_back(torch::nn::Linear(lastSize, layerSizes[i]));
		if (addLayerNorm)
			seq->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ (int64_t)layerSizes[i] })));
		lastSize = layerSizes[i];
		GGL::AddActivationFunc(seq, activationType);
	}
	return lastSize;
}

// Lists the layers of a sequence in the form that FusedLayer needs, or returns nothing if the sequence can't be run that way.
// The sequence must be [linear, layer norm, ReLU] for every hidden layer, optionally followed by a linear output layer.
static std::vector<GGL::Model::FusedStep> MakeFusedSteps(torch::nn::Sequential& seq) {
	std::vector<GGL::Model::FusedStep> steps = {};
	for (size_t i = 0; i < seq->size();) {
		GGL::Model::FusedStep step = {};
		step.linear = seq[i]->as<torch::nn::Linear>();
		if (!step.linear)
			return {};

		if (i + 1 == seq->size()) {
			// Output layer
			i += 1;
		} else {
			step.layerNorm = seq[i + 1]->as<torch::nn::LayerNorm>();
			bool hasRelu = (i + 2 < seq->size()) && seq[i + 2]->as<torch::nn::ReLU>();
			if (!step.layerNorm || !hasRelu || !step.layerNorm->options.elementwise_affine() || !step.linear->options.bias())
				return {};
			i += 3;
		}
		steps.push_back(step);
	}
	return steps;
}

GGL::Model::Model(
	const char* modelName,
	ModelConfig config,
	torch::Device device) : 
	modelName(modelName), device(device), seq({}), seqHalf({}), slotSeq({}), slotSeqHalf({}), config(config) {

	if (!config.IsValid())
		RG_ERR_CLOSE("Failed to create model \"" << modelName << "\" with invalid config");

	// Number of inputs of the first layer of seq
	int numLayerInputs = config.numInputs;

	auto& slots = this->config.slots;
	if (slots.IsValid()) {
		numGlobalInputs = config.numInputs - slots.NumSlotInputs();
		if (numGlobalInputs < 0 || slots.contextSize < 0 || slots.contextSize > numGlobalInputs) {
			RG_ERR_CLOSE(
				"Failed to create model \"" << modelName << "\": the slot encoder config (" << slots.NumSlots() << " slots of " << slots.slotSize <<
				" inputs, context size " << slots.contextSize << ") doesn't fit the " << config.numInputs << " model inputs"
			);
		}

		// The encodings must not be negative: empty slots count as zeros, which then never win the maximum of a group
		AddHiddenLayers(slotSeq, slots.slotSize + slots.contextSize, slots.layerSizes, config.addLayerNorm, ModelActivationType::RELU);
		numLayerInputs = numGlobalInputs + slots.NumOutputs();
	} else {
		slots = {};
	}

	int lastSize = AddHiddenLayers(seq, numLayerInputs, config.layerSizes, config.addLayerNorm, config.activationType);

	if (slots.IsValid()) {
		// The values that come out of the slot encoder can far outnumber the global inputs (e.g. 1024 to 84).
		// With the default initialization they would make up nearly all of what the first layer computes at the start of training,
		//	and drown out the global inputs. So their weights start smaller: both groups of inputs then have about the same total weight.
		float scale = sqrtf(numGlobalInputs / (float)slots.NumOutputs());
		if (scale < 1) {
			RG_NO_GRAD;
			auto firstLayer = seq[0]->as<torch::nn::Linear>();
			firstLayer->weight.narrow(1, numGlobalInputs, slots.NumOutputs()).mul_(scale);
		}
	}
	
	if (config.addOutputLayer) {
		seq->push_back(torch::nn::Linear(lastSize, config.numOutputs));
	} else {
		this->config.numOutputs = config.layerSizes.back();
	}

	register_module("seq", seq);
	seq->to(device);
	if (slots.IsValid()) {
		register_module("slots", slotSeq);
		slotSeq->to(device);
	}
	optim = MakeOptimizer(config.optimType, this->parameters(), 0);

	// On the CPU, hidden layers of the form (linear -> layer norm -> ReLU) are run as one fused operation, which is much faster
	if (device.is_cpu() && FusedLayer::IsSupported() && config.addLayerNorm) {
		if (config.activationType == ModelActivationType::RELU)
			_fusedSteps = MakeFusedSteps(seq);
		if (slots.IsValid())
			_slotFusedSteps = MakeFusedSteps(slotSeq);
	}
}

// Runs the layers of a sequence, as fused operations if possible (same result as seq->forward() up to float rounding)
static torch::Tensor RunLayers(torch::nn::Sequential& seq, const std::vector<GGL::Model::FusedStep>& fusedSteps, torch::Tensor input) {
	using namespace GGL;

	if (!fusedSteps.empty() && FusedLayer::IsEnabled() && input.dim() == 2 && input.scalar_type() == torch::kFloat32 && input.device().is_cpu()) {
		for (auto& step : fusedSteps) {
			if (step.layerNorm) {
				input = FusedLayer::Forward(
					input,
					step.linear->weight, step.linear->bias,
					step.layerNorm->weight, step.layerNorm->bias, step.layerNorm->options.eps()
				);
			} else {
				input = step.linear->forward(input);
			}
		}
		return input;
	}

	return seq->forward(input);
}

torch::Tensor GGL::Model::EncodeSlots(torch::Tensor input, torch::nn::Sequential& encoderSeq, const std::vector<FusedStep>& fusedSteps) {
	auto& slots = config.slots;
	int64_t
		batchSize = input.size(0),
		numSlots = slots.NumSlots(),
		encodingSize = slots.layerSizes.back();

	RG_ASSERT(input.dim() == 2 && input.size(1) == config.numInputs);

	if (FusedLayer::IsEnabled() && input.device().is_cpu() && input.scalar_type() == torch::kFloat32 && !input.requires_grad()) {
		// On the CPU, picking the used slots and pooling their encodings are done as single passes over the data (see SlotPool.h).
		// Same result as the torch operations below.
		torch::Tensor contiguousInput = input.contiguous();
		std::vector<int64_t> usedSlots = {};
		torch::Tensor encoderInput = SlotPool::GatherUsedSlots(contiguousInput, numGlobalInputs, numSlots, slots.slotSize, slots.contextSize, usedSlots);

		torch::Tensor usedEncodings;
		if (usedSlots.empty()) {
			usedEncodings = torch::zeros({ (int64_t)0, encodingSize });
		} else {
			usedEncodings = RunLayers(encoderSeq, fusedSteps, encoderInput);
		}

		if (slots.singleEncoding) {
			torch::Tensor globalPart = contiguousInput.narrow(1, 0, numGlobalInputs);

			// The usual case (1v1): every sample has exactly one used slot, so the encodings already are one row per sample
			bool oneEach = (int64_t)usedSlots.size() == batchSize;
			for (int64_t i = 0; oneEach && i < batchSize; i++)
				oneEach = (usedSlots[i] / numSlots) == i;
			if (oneEach)
				return torch::cat({ globalPart, usedEncodings }, 1);

			// Otherwise: the sum over the used slots of each sample
			std::vector<int64_t> sampleIndices(usedSlots.size());
			for (size_t i = 0; i < usedSlots.size(); i++)
				sampleIndices[i] = usedSlots[i] / numSlots;
			torch::Tensor summed = torch::zeros({ batchSize, encodingSize }).index_add(0, torch::tensor(sampleIndices), usedEncodings);
			return torch::cat({ globalPart, summed }, 1);
		}

		return SlotPool::Pool(contiguousInput, numGlobalInputs, usedEncodings, usedSlots, slots.groupSizes);
	}

	torch::Tensor globalInputs = input.narrow(1, 0, numGlobalInputs);
	torch::Tensor slotInputs = input.narrow(1, numGlobalInputs, numSlots * slots.slotSize).reshape({ batchSize * numSlots, slots.slotSize });

	// Which slots are used, and their row indices in slotInputs
	torch::Tensor used, usedIndices;
	{
		RG_NO_GRAD;
		used = slotInputs.select(1, 0) > 0.5;
		usedIndices = used.nonzero().flatten();
	}

	// Encode the used slots only. The encodings of empty slots are zero.
	torch::Tensor encodings; // [batchSize * numSlots, encodingSize]
	if (usedIndices.numel() > 0) {
		torch::Tensor encoderInput = slotInputs.index_select(0, usedIndices);
		if (slots.contextSize > 0) {
			torch::Tensor sampleIndices = usedIndices.div(numSlots, "floor"); // Which sample of the batch each used slot belongs to
			torch::Tensor context = globalInputs.narrow(1, numGlobalInputs - slots.contextSize, slots.contextSize).index_select(0, sampleIndices);
			encoderInput = torch::cat({ encoderInput, context }, 1);
		}

		torch::Tensor usedEncodings = RunLayers(encoderSeq, fusedSteps, encoderInput);
		encodings = torch::zeros({ batchSize * numSlots, encodingSize }, usedEncodings.options()).index_copy(0, usedIndices, usedEncodings);
	} else {
		encodings = torch::zeros({ batchSize * numSlots, encodingSize }, input.options());
	}
	encodings = encodings.view({ batchSize, numSlots, encodingSize });

	if (slots.singleEncoding)
		return torch::cat({ globalInputs, encodings.sum(1) }, 1);

	torch::Tensor usedCounts = used.view({ batchSize, numSlots }).to(encodings.scalar_type());

	// Combine the encodings within every group: maximum and mean over its used slots (both zero if there are none).
	// As the encodings are never negative (they come out of a ReLU), the zeros of empty slots don't change the maximum.
	std::vector<torch::Tensor> parts = { globalInputs };
	int64_t groupStart = 0;
	for (int groupSize : slots.groupSizes) {
		torch::Tensor groupEncodings = encodings.narrow(1, groupStart, groupSize);
		torch::Tensor groupUsedCount = usedCounts.narrow(1, groupStart, groupSize).sum(1, true).clamp_min(1);

		parts.push_back(groupEncodings.amax(1));
		parts.push_back(groupEncodings.sum(1) / groupUsedCount);
		groupStart += groupSize;
	}

	return torch::cat(parts, 1);
}

// Makes "half" a half-precision copy of "full", or updates its parameters if it already is one
static void UpdateHalfSeq(torch::nn::Sequential& full, torch::nn::Sequential& half) {
	if (half->size() == 0) {
		for (auto& mod : *full)
			half->push_back(mod.clone());
		half->to(RG_HALFPERC_TYPE, true);
	} else {
		auto fromParams = full->parameters();
		auto toParams = half->parameters();
		for (int i = 0; i < fromParams.size(); i++) {
			auto scaledParams = fromParams[i].to(RG_HALFPERC_TYPE, true);
			toParams[i].copy_(scaledParams, true);
		}
	}
}

torch::Tensor GGL::Model::Forward(torch::Tensor input, bool halfPrec) {

	if (torch::GradMode::is_enabled())
		halfPrec = false;

	bool hasSlots = config.slots.IsValid();

	if (halfPrec) {

		if (_seqHalfOutdated) {
			_seqHalfOutdated = false;

			UpdateHalfSeq(seq, seqHalf);
			if (hasSlots)
				UpdateHalfSeq(slotSeq, slotSeqHalf);
		}
		
		auto halfInput = input.to(RG_HALFPERC_TYPE);
		if (hasSlots)
			halfInput = EncodeSlots(halfInput, slotSeqHalf, {});
		auto halfOutput = seqHalf->forward(halfInput);
		return halfOutput.to(torch::kFloat);
	} else {
		if (hasSlots)
			input = EncodeSlots(input, slotSeq, _slotFusedSteps);

		return RunLayers(seq, _fusedSteps, input);
	}
}

// Get sizes of all parameters in a sequence
std::vector<uint64_t> GetSeqSizes(torch::nn::Sequential& seq) {
	std::vector<uint64_t> result = {};

	for (int i = 0; i < seq->size(); i++)
		for (auto param : seq[i]->parameters())
			result.push_back(param.numel());

	return result;
}

void GGL::Model::SetOptimLR(float newLR) {
	SetOptimizerLR(optim, config.optimType, newLR);
}

void GGL::Model::StepOptim() {
	optim->step();
	optim->zero_grad();
	_seqHalfOutdated = true;
}

void GGL::Model::Save(std::filesystem::path folder, bool saveOptim) {
	std::filesystem::path path = GetSavePath(folder);
	auto streamOut = std::ofstream(path, std::ios::binary);
	torch::save(seq, streamOut);

	if (config.slots.IsValid()) {
		auto slotStreamOut = std::ofstream(GetSlotEncoderSavePath(folder), std::ios::binary);
		torch::save(slotSeq, slotStreamOut);
	}

	if (saveOptim) {
		torch::serialize::OutputArchive optimArchive;
		optim->save(optimArchive);
		optimArchive.save_to(GetOptimSavePath(folder).string());
	}
}

// Loads a sequence from a file, and makes sure that what was loaded has the sizes that the sequence had before.
// afterLoad (optional) is called with the sequence as it came out of the file, before the sizes are checked.
static void LoadSeq(
	torch::nn::Sequential& seq, std::filesystem::path path, torch::Device device, const char* modelName,
	std::function<void(torch::nn::Sequential&)> afterLoad = nullptr) {
	auto streamIn = std::ifstream(path, std::ios::binary);
	streamIn >> std::noskipws;

	if (!streamIn.good())
		RG_ERR_CLOSE("Failed to load from " << path << ", file does not exist or can't be accessed");

	auto sizesBefore = GetSeqSizes(seq);

	try {
		torch::load(seq, streamIn, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE(
			"Failed to load model \"" << modelName << ", checkpoint may be corrupt or of different model arch.\n" <<
			"Exception: " << e.what()
		);
	}

	if (afterLoad)
		afterLoad(seq);

	// Torch will happily load in a model of a totally different size, then we will crash when we try to use it
	// So we need to manually check if it is the same size
	auto sizesAfter = GetSeqSizes(seq);
	if (!std::equal(sizesBefore.begin(), sizesBefore.end(), sizesAfter.begin(), sizesAfter.end())) {
		std::stringstream stream;
		stream << "Saved model has different size than current model, cannot load model from " << path << ":\n";

		for (int i = 0; i < 2; i++) {
			stream << " > " << (i ? "Saved model:   [ " : "Current model: [ ");
			for (uint64_t size : (i ? sizesAfter : sizesBefore))
				stream << size << ' ';

			stream << " ]";
			if (i == 0)
				stream << ",\n";
		}

		RG_ERR_CLOSE(stream.str());
	}
}

void GGL::Model::Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim) {
	std::filesystem::path path = GetSavePath(folder);

	if (!std::filesystem::exists(path)) {
		if (allowNotExist) {
			RG_LOG("Warning: Model \"" << modelName << "\" does not exist in " << folder << " and will be reset");
			return;
		} else {
			RG_ERR_CLOSE("Model \"" << modelName << "\" does not exist in " << folder);
		}
	}

	// A model with SlotEncoderConfig::singleEncoding also loads the files of the same model without it, and the other way round.
	// Without it, the first layer has a column for the maximum and for the mean of every group: [global | max, mean | max, mean | ...].
	// With it, there is one column: [global | encoding].
	//
	// For an input with one used slot, in the last group, all other groups are zeros, and maximum and mean of the last
	// group are both that slot's encoding. So the output for such inputs stays the same if
	//	- (to singleEncoding) the one column gets the sum of the last group's two columns
	//	- (from singleEncoding) the last group's two columns get half of the one column each, and the other groups get zeros
	auto& slots = config.slots;
	int64_t encodingSize = slots.IsValid() ? slots.layerSizes.back() : 0;
	int64_t pooledLayerInputs = slots.IsValid() ? (numGlobalInputs + slots.NumPooledOutputs()) : 0;
	int64_t singleLayerInputs = numGlobalInputs + encodingSize;
	enum { NOT_CONVERTED, TO_SINGLE, FROM_SINGLE } conversion = NOT_CONVERTED;

	// What the columns hold: the weights, or the optimizer's averages of their gradients (Adam) and of the squares.
	// The two columns of the last group always got the same gradient as the one column gets.
	enum { WEIGHTS, GRAD_AVG, GRAD_SQ_AVG };
	auto fnToSingle = [&](torch::Tensor pooled, int content) -> torch::Tensor {
		RG_NO_GRAD;
		torch::Tensor lastMax = pooled.narrow(1, pooledLayerInputs - 2 * encodingSize, encodingSize);
		torch::Tensor lastMean = pooled.narrow(1, pooledLayerInputs - encodingSize, encodingSize);
		return torch::cat({ pooled.narrow(1, 0, numGlobalInputs), (content == WEIGHTS) ? (lastMax + lastMean) : lastMean }, 1).contiguous();
	};
	auto fnFromSingle = [&](torch::Tensor single, int content) -> torch::Tensor {
		RG_NO_GRAD;
		torch::Tensor encoding = single.narrow(1, numGlobalInputs, encodingSize);
		std::vector<torch::Tensor> parts = { single.narrow(1, 0, numGlobalInputs) };
		int numGroups = (int)slots.groupSizes.size();
		for (int group = 0; group < numGroups; group++) {
			bool isLast = (group == numGroups - 1);
			for (int i = 0; i < 2; i++) { // Maximum, mean
				if (content == WEIGHTS) {
					parts.push_back(isLast ? (encoding * 0.5f) : torch::zeros_like(encoding));
				} else if (content == GRAD_AVG) {
					parts.push_back(isLast ? encoding.clone() : torch::zeros_like(encoding));
				} else {
					// The columns of the other groups never had a gradient. They get the average of the squares of the
					// last group, which sets Adam's step size: otherwise their first steps would be several times too big
					parts.push_back(encoding.clone());
				}
			}
		}
		return torch::cat(parts, 1).contiguous();
	};

	LoadSeq(seq, path, device, modelName, [&](torch::nn::Sequential& loadedSeq) {
		if (!slots.IsValid() || singleLayerInputs == pooledLayerInputs)
			return;

		auto firstLayer = loadedSeq[0]->as<torch::nn::Linear>();
		if (!firstLayer || firstLayer->weight.dim() != 2)
			return;

		RG_NO_GRAD;
		if (slots.singleEncoding && firstLayer->weight.size(1) == pooledLayerInputs) {
			firstLayer->weight.set_data(fnToSingle(firstLayer->weight, WEIGHTS));
			conversion = TO_SINGLE;
		} else if (!slots.singleEncoding && firstLayer->weight.size(1) == singleLayerInputs) {
			firstLayer->weight.set_data(fnFromSingle(firstLayer->weight, WEIGHTS));
			conversion = FROM_SINGLE;
		}
	});

	if (slots.IsValid()) {
		std::filesystem::path slotPath = GetSlotEncoderSavePath(folder);
		if (!std::filesystem::exists(slotPath))
			RG_ERR_CLOSE("The slot encoder of model \"" << modelName << "\" does not exist in " << folder << " (was this checkpoint made with a different model config?)");
		LoadSeq(slotSeq, slotPath, device, modelName);
	}

	_seqHalfOutdated = true;

	/////////////////////////////

	if (loadOptim) {
		std::filesystem::path optimPath = GetOptimSavePath(folder);

		if (std::filesystem::exists(optimPath)) {
			std::ifstream testStream = std::ifstream(optimPath, std::istream::ate | std::ios::binary);
			if (testStream.tellg() > 0) {
				torch::serialize::InputArchive optimArchive;
				optimArchive.load_from(optimPath.string(), device);
				optim->load(optimArchive);

				if (conversion != NOT_CONVERTED) {
					// The optimizer's state for the first layer has the columns of the file as well
					auto firstLayer = seq[0]->as<torch::nn::Linear>();
					auto& states = optim->state();
					auto itr = states.find(firstLayer->weight.unsafeGetTensorImpl());
					if (itr != states.end()) {
						int64_t fileLayerInputs = (conversion == TO_SINGLE) ? pooledLayerInputs : singleLayerInputs;
						auto fnConvert = [&](torch::Tensor t, int content) {
							return (conversion == TO_SINGLE) ? fnToSingle(t, content) : fnFromSingle(t, content);
						};

						auto adamState = dynamic_cast<torch::optim::AdamParamState*>(itr->second.get());
						if (adamState && adamState->exp_avg().dim() == 2 && adamState->exp_avg().size(1) == fileLayerInputs) {
							adamState->exp_avg(fnConvert(adamState->exp_avg(), GRAD_AVG));
							adamState->exp_avg_sq(fnConvert(adamState->exp_avg_sq(), GRAD_SQ_AVG));
							if (adamState->max_exp_avg_sq().defined())
								adamState->max_exp_avg_sq(fnConvert(adamState->max_exp_avg_sq(), GRAD_SQ_AVG));
						} else {
							states.erase(itr); // Starts again for this layer
						}
					}
				}
			} else {
				RG_LOG("WARNING: Saved optimizer at " << optimPath << " is empty, optimizer will be reset");
			}
		} else {
			RG_LOG("WARNING: No optimizer found at " << optimPath << ", optimizer will be reset");
		}
	}
}

void GGL::Model::CopyInputWeights(int fromInput, int toInput) {
	RG_NO_GRAD;

	auto fnCopyColumn = [this](torch::nn::Sequential& layers, int64_t from, int64_t to) {
		auto firstLayer = layers[0]->as<torch::nn::Linear>();
		RG_ASSERT(firstLayer && from >= 0 && to >= 0 && from < firstLayer->weight.size(1) && to < firstLayer->weight.size(1));
		firstLayer->weight.select(1, to).copy_(firstLayer->weight.select(1, from));

		// The optimizer's state of the column: no average gradient yet, and the step size of the column it now is a copy of
		auto& states = optim->state();
		auto itr = states.find(firstLayer->weight.unsafeGetTensorImpl());
		if (itr != states.end()) {
			if (auto adamState = dynamic_cast<torch::optim::AdamParamState*>(itr->second.get())) {
				adamState->exp_avg().select(1, to).zero_();
				adamState->exp_avg_sq().select(1, to).copy_(adamState->exp_avg_sq().select(1, from));
			}
		}
	};

	if (config.slots.IsValid()) {
		if (fromInput >= numGlobalInputs || toInput >= numGlobalInputs)
			RG_ERR_CLOSE("Model::CopyInputWeights(): Only for inputs that are not part of a slot");

		fnCopyColumn(seq, fromInput, toInput);

		// The slot encoder gets the last contextSize global inputs after the inputs of the slot
		int contextStart = numGlobalInputs - config.slots.contextSize;
		if (fromInput >= contextStart && toInput >= contextStart)
			fnCopyColumn(slotSeq, config.slots.slotSize + (fromInput - contextStart), config.slots.slotSize + (toInput - contextStart));
	} else {
		fnCopyColumn(seq, fromInput, toInput);
	}

	_seqHalfOutdated = true;
}

torch::Tensor GGL::Model::CopyParams() const {
	return torch::nn::utils::parameters_to_vector(parameters()).cpu();
}
