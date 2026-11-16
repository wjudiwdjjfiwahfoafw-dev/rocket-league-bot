#pragma once

#include "../Framework.h"

namespace GGL {
	enum class ModelOptimType {
		ADAM,
		ADAMW,
		ADAGRAD,
		RMSPROP,
		MAGSGD
	};

	enum class ModelActivationType {
		RELU,
		LEAKY_RELU,
		SIGMOID,
		TANH
	};

	// Optional first stage of a model, for inputs that end with a fixed number of equally shaped "slots"
	//	(e.g. one slot for each other player of the game, where slots can be empty):
	//
	//	Model input:  [global inputs][slot 0][slot 1]...[slot N-1]
	//
	// Every used slot is run through the same small network (the slot encoder), and the results are combined within groups
	//	of slots (e.g. teammates, opponents) by taking the maximum and the mean over the group's used slots.
	// The model's layers then get: [global inputs][max and mean of group 0][max and mean of group 1]...
	//
	// Compared to giving the slots to the layers directly, this means:
	//	- The order of the slots within a group doesn't matter, and empty slots have no effect at all
	//	- What is learned about one slot applies to all of them, and to every number of used slots
	struct SlotEncoderConfig {
		// Number of inputs of one slot. The first input of every slot must be 1 if the slot is used, and 0 if it is empty.
		int slotSize = 0;

		// Number of slots in each group, in the order of the input. E.g. { 2, 3 } for 2 teammate slots followed by 3 opponent slots.
		std::vector<int> groupSizes = {};

		// The last contextSize global inputs are given to the slot encoder together with each slot.
		// This lets it relate the slot to them (e.g. the other player to the own car and the ball).
		int contextSize = 0;

		// Layer sizes of the slot encoder. It uses the layer norm setting of the model, and always ReLU.
		std::vector<int> layerSizes = {};

		// false: the model's layers get the maximum and the mean of every group (2 * layerSizes.back() values per group).
		// true:  they get one encoding, the sum over all used slots of all groups (layerSizes.back() values).
		//
		// For inputs that only ever have one used slot (1v1: the one other player) the sum is that slot's encoding,
		// the same information in fewer values. With 2 groups the first layer of the model has 3 * layerSizes.back() fewer
		// inputs, which makes a small model noticeably faster. Such a model can't tell teammates from opponents by the pooling.
		//
		// A model with this setting loads the saved files of the same model without it (Model::Load converts the first
		// layer). The converted model gives the same output for inputs with exactly one used slot, in the last group.
		bool singleEncoding = false;

		bool IsValid() const {
			return slotSize > 0 && !groupSizes.empty() && !layerSizes.empty();
		}

		int NumSlots() const {
			int numSlots = 0;
			for (int groupSize : groupSizes)
				numSlots += groupSize;
			return numSlots;
		}

		// Number of model inputs that the slots take up
		int NumSlotInputs() const {
			return NumSlots() * slotSize;
		}

		// Number of values that replace the slot inputs for the model's layers
		int NumOutputs() const {
			return singleEncoding ? layerSizes.back() : NumPooledOutputs();
		}

		// Number of values with maximum and mean of every group (singleEncoding = false)
		int NumPooledOutputs() const {
			return (int)groupSizes.size() * 2 * layerSizes.back();
		}
	};

	// Doesn't include inputs or outputs
	struct PartialModelConfig {
		std::vector<int> layerSizes = {};
		ModelActivationType activationType = ModelActivationType::RELU;
		ModelOptimType optimType = ModelOptimType::ADAM;
		bool addLayerNorm = true;
		bool addOutputLayer = true;

		// Not used unless set (see SlotEncoderConfig)
		SlotEncoderConfig slots = {};

		bool IsValid() const {
			return !layerSizes.empty();
		}
	};

	struct ModelConfig : PartialModelConfig {
		int numInputs = -1;
		int numOutputs = -1;

		bool IsValid() const {
			return PartialModelConfig::IsValid() && numInputs > 0 && (numOutputs > 0 || !addOutputLayer);
		}

		ModelConfig(const PartialModelConfig& partialConfig) : PartialModelConfig(partialConfig) {}
	};
}