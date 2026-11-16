#pragma once

#include "ModelConfig.h"

#include <filesystem>
#include <utility>
#include <vector>

namespace GGL {
	// Loads the models of a checkpoint (or of a saved policy version) and saves them to another folder.
	//
	// This converts between a model with SlotEncoderConfig::singleEncoding and the same model without it:
	// loading reads the files of either kind (see Model::Load), and saving writes the kind of the configs given here.
	//
	//	withCriticAndOptims: true for a checkpoint (policy, critic, shared head and their optimizers),
	//		false for a saved policy version (policy and shared head only)
	//	inputWeightCopies: pairs (from, to) of model inputs. Each "to" input gets the weights of the "from" input,
	//		in the model that the inputs go into (see Model::CopyInputWeights)
	RG_IMEXPORT void ConvertModels(
		std::filesystem::path srcFolder, std::filesystem::path dstFolder,
		int obsSize, int numActions,
		PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
		bool withCriticAndOptims,
		const std::vector<std::pair<int, int>>& inputWeightCopies = {}
	);
}
