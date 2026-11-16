#pragma once
#include "../FrameworkTorch.h"

namespace GGL {

	// The two steps around the slot encoder of a model (see SlotEncoderConfig), as single passes over the data (CPU, float32 only).
	//
	// Why: Written with torch operations (index_select, cat, index_copy, amax, sum, ...), picking the used slots and pooling 
	//	their encodings takes about 25 operations, each of which makes its own pass over the data and its own output tensor,
	//	and about as many again for the backward pass. For small models that cost more than the model's layers themselves.
	//
	// The result is the same as Model::EncodeSlots() computes with torch operations (see tests/TestMultiMode.cpp).
	namespace SlotPool {

		// Makes the input of the slot encoder: one row for every used slot, [slot inputs][context inputs of its sample].
		//	input: [N, numInputs], laid out as [numGlobalInputs global inputs][numSlots slots of slotSize inputs].
		//		A slot is used if its first input is > 0.5. The context are the last contextSize global inputs.
		//	outUsedSlots: for every row of the result, (index of the sample) * numSlots + (index of the slot)
		// No gradients (the input is data).
		torch::Tensor GatherUsedSlots(
			const torch::Tensor& input, int64_t numGlobalInputs, int64_t numSlots, int64_t slotSize, int64_t contextSize,
			std::vector<int64_t>& outUsedSlots
		);

		// Makes the input of the model's layers: [N, numGlobalInputs + groupSizes.size() * 2 * E].
		//	For every sample: its global inputs, then for every group of slots the maximum and the mean 
		//	of the encodings of its used slots (zeros if it has none).
		//	encodings: [numUsedSlots, E], must not be negative. usedSlots: as returned by GatherUsedSlots().
		// Supports autograd (first order) for the encodings.
		torch::Tensor Pool(
			const torch::Tensor& input, int64_t numGlobalInputs,
			const torch::Tensor& encodings, const std::vector<int64_t>& usedSlots, const std::vector<int>& groupSizes
		);
	}
}
