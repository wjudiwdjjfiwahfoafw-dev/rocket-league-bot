#pragma once

#include "ModelConfig.h"

namespace GGL {
	struct RG_IMEXPORT InferUnit {
		int obsSize;
		RLGC::ObsBuilder* obsBuilder;
		RLGC::ActionParser* actionParser;
		struct ModelSet* models;
		bool useGPU;

		// NOTE: Reset() will never be called on your obs 
		InferUnit(
			RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig,
			std::filesystem::path modelsFolder, bool useGPU);


		// Sets the mirrored view: 0 = off, 1 = right, -1 = left (see RLGC::ObsBuilder::mirrorXTeam).
		// A policy has to play with the view it was trained with.
		void SetMirrorX(int mirrorX);

		RLGC::Action InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature = 1);
		std::vector<RLGC::Action> BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature = 1);

		// Sets the seed of the random numbers that non-deterministic actions are chosen with (for all InferUnits)
		static void SetRandomSeed(uint64_t seed);

		// TODO: Add deconstructor (make sure to free models too)
	};
}