#include "ModelConvert.h"

#include <GigaLearnCPP/Util/Models.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>

void GGL::ConvertModels(
	std::filesystem::path srcFolder, std::filesystem::path dstFolder,
	int obsSize, int numActions,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
	bool withCriticAndOptims,
	const std::vector<std::pair<int, int>>& inputWeightCopies) {

	ModelSet models = {};
	try {
		PPOLearner::MakeModels(withCriticAndOptims, obsSize, numActions, sharedHeadConfig, policyConfig, criticConfig, torch::kCPU, models);
		models.Load(srcFolder, false, withCriticAndOptims);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("ConvertModels: Exception when loading the models from " << srcFolder << ": " << e.what());
	}

	// The model that gets the inputs
	Model* inputModel = sharedHeadConfig.IsValid() ? models["shared_head"] : NULL;
	for (auto& pair : inputWeightCopies) {
		if (inputModel) {
			inputModel->CopyInputWeights(pair.first, pair.second);
		} else {
			models["policy"]->CopyInputWeights(pair.first, pair.second);
			if (withCriticAndOptims)
				models["critic"]->CopyInputWeights(pair.first, pair.second);
		}
	}

	std::filesystem::create_directories(dstFolder);
	models.Save(dstFolder, withCriticAndOptims);
	models.Free();
}
