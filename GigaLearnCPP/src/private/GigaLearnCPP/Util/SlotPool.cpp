#include "SlotPool.h"

#include <torch/autograd.h>
#include <torch/csrc/autograd/custom_function.h>

#include <cstring>

#if defined(_MSC_VER)
#define GGL_RESTRICT __restrict
#else
#define GGL_RESTRICT __restrict__
#endif

namespace {
	// Index of the group that each slot belongs to
	std::vector<int> GetGroupOfSlots(const std::vector<int>& groupSizes) {
		std::vector<int> result = {};
		for (int groupIdx = 0; groupIdx < groupSizes.size(); groupIdx++)
			for (int i = 0; i < groupSizes[groupIdx]; i++)
				result.push_back(groupIdx);
		return result;
	}

	// out: [N, numGlobal + numGroups * 2 * E], outCounts: [N * numGroups] (number of used slots of each group)
	void RunPoolKernel(
		const float* input, int64_t inputStride, int64_t batchSize, int64_t numGlobal,
		const float* encodings, int64_t encodingSize, const int64_t* usedSlots, int64_t numUsed,
		const std::vector<int>& groupOfSlots, int64_t numGroups,
		float* out, float* outCounts) {

		int64_t numSlots = groupOfSlots.size();
		int64_t pooledSize = numGroups * 2 * encodingSize;
		int64_t outStride = numGlobal + pooledSize;

		for (int64_t n = 0; n < batchSize; n++) {
			float* outRow = out + n * outStride;
			memcpy(outRow, input + n * inputStride, numGlobal * sizeof(float));
			memset(outRow + numGlobal, 0, pooledSize * sizeof(float));
		}
		memset(outCounts, 0, batchSize * numGroups * sizeof(float));

		// Maximum (starting from zero) and sum
		for (int64_t r = 0; r < numUsed; r++) {
			int64_t n = usedSlots[r] / numSlots;
			int64_t groupIdx = groupOfSlots[usedSlots[r] % numSlots];

			const float* GGL_RESTRICT enc = encodings + r * encodingSize;
			float* GGL_RESTRICT outMax = out + n * outStride + numGlobal + groupIdx * 2 * encodingSize;
			float* GGL_RESTRICT outSum = outMax + encodingSize;
			for (int64_t j = 0; j < encodingSize; j++) {
				outMax[j] = (enc[j] > outMax[j]) ? enc[j] : outMax[j];
				outSum[j] += enc[j];
			}
			outCounts[n * numGroups + groupIdx] += 1;
		}

		// Sum -> mean
		for (int64_t n = 0; n < batchSize; n++) {
			for (int64_t groupIdx = 0; groupIdx < numGroups; groupIdx++) {
				float count = outCounts[n * numGroups + groupIdx];
				if (count > 1) {
					float scale = 1 / count;
					float* GGL_RESTRICT outMean = out + n * outStride + numGlobal + groupIdx * 2 * encodingSize + encodingSize;
					for (int64_t j = 0; j < encodingSize; j++)
						outMean[j] *= scale;
				}
			}
		}
	}

	// dEncodings: [numUsed, E]
	// The gradient of a maximum goes to the first slot that has that value (and to none if the maximum is the zero of an empty group)
	void RunPoolBackwardKernel(
		const float* dOut, const float* out, const float* counts, int64_t numGlobal,
		const float* encodings, int64_t encodingSize, const int64_t* usedSlots, int64_t numUsed,
		const std::vector<int>& groupOfSlots, int64_t numGroups,
		float* dEncodings) {

		int64_t numSlots = groupOfSlots.size();
		int64_t outStride = numGlobal + numGroups * 2 * encodingSize;

		// 1 where the maximum of the current group has already been given to a slot
		std::vector<float> takenBuffer(encodingSize);
		float* GGL_RESTRICT taken = takenBuffer.data();
		int64_t lastGroupKey = -1;

		for (int64_t r = 0; r < numUsed; r++) {
			int64_t n = usedSlots[r] / numSlots;
			int64_t groupIdx = groupOfSlots[usedSlots[r] % numSlots];

			// The used slots are sorted by sample and slot, so the slots of one group of one sample follow each other
			int64_t groupKey = n * numGroups + groupIdx;
			if (groupKey != lastGroupKey) {
				lastGroupKey = groupKey;
				memset(taken, 0, encodingSize * sizeof(float));
			}

			int64_t pooledOffset = n * outStride + numGlobal + groupIdx * 2 * encodingSize;
			const float* GGL_RESTRICT enc = encodings + r * encodingSize;
			const float* GGL_RESTRICT outMax = out + pooledOffset;
			const float* GGL_RESTRICT dMax = dOut + pooledOffset;
			const float* GGL_RESTRICT dMean = dMax + encodingSize;
			float* GGL_RESTRICT dEnc = dEncodings + r * encodingSize;
			float invCount = 1 / counts[groupKey];

			for (int64_t j = 0; j < encodingSize; j++) {
				float isMax = (enc[j] == outMax[j] && enc[j] > 0 && taken[j] == 0) ? 1.f : 0.f;
				dEnc[j] = dMean[j] * invCount + dMax[j] * isMax;
				taken[j] += isMax;
			}
		}
	}

	struct SlotPoolFunction : public torch::autograd::Function<SlotPoolFunction> {

		static torch::Tensor forward(
			torch::autograd::AutogradContext* ctx,
			torch::Tensor encodings, torch::Tensor input, torch::Tensor usedSlots, int64_t numGlobal, torch::Tensor groupOfSlotsTensor, int64_t numGroups) {

			int64_t batchSize = input.size(0), encodingSize = encodings.size(1);
			auto groupOfSlots = std::vector<int>(groupOfSlotsTensor.data_ptr<int>(), groupOfSlotsTensor.data_ptr<int>() + groupOfSlotsTensor.numel());

			torch::Tensor out = torch::empty({ batchSize, numGlobal + numGroups * 2 * encodingSize });
			torch::Tensor counts = torch::empty({ batchSize * numGroups });
			RunPoolKernel(
				input.data_ptr<float>(), input.size(1), batchSize, numGlobal,
				encodings.data_ptr<float>(), encodingSize, usedSlots.data_ptr<int64_t>(), usedSlots.numel(),
				groupOfSlots, numGroups,
				out.data_ptr<float>(), counts.data_ptr<float>()
			);

			ctx->save_for_backward({ encodings, out });
			ctx->saved_data["counts"] = counts;
			ctx->saved_data["usedSlots"] = usedSlots;
			ctx->saved_data["groupOfSlots"] = groupOfSlotsTensor;
			ctx->saved_data["numGlobal"] = numGlobal;
			ctx->saved_data["numGroups"] = numGroups;
			return out;
		}

		static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list gradOutputs) {
			auto saved = ctx->get_saved_variables();
			torch::Tensor encodings = saved[0], out = saved[1];
			torch::Tensor counts = ctx->saved_data["counts"].toTensor();
			torch::Tensor usedSlots = ctx->saved_data["usedSlots"].toTensor();
			torch::Tensor groupOfSlotsTensor = ctx->saved_data["groupOfSlots"].toTensor();
			int64_t numGlobal = ctx->saved_data["numGlobal"].toInt();
			int64_t numGroups = ctx->saved_data["numGroups"].toInt();
			auto groupOfSlots = std::vector<int>(groupOfSlotsTensor.data_ptr<int>(), groupOfSlotsTensor.data_ptr<int>() + groupOfSlotsTensor.numel());

			torch::Tensor dOut = gradOutputs[0].contiguous();
			torch::Tensor dEncodings = torch::empty_like(encodings);
			RunPoolBackwardKernel(
				dOut.data_ptr<float>(), out.data_ptr<float>(), counts.data_ptr<float>(), numGlobal,
				encodings.data_ptr<float>(), encodings.size(1), usedSlots.data_ptr<int64_t>(), usedSlots.numel(),
				groupOfSlots, numGroups,
				dEncodings.data_ptr<float>()
			);

			// One entry per argument of forward()
			return { dEncodings, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor() };
		}
	};
}

torch::Tensor GGL::SlotPool::GatherUsedSlots(
	const torch::Tensor& input, int64_t numGlobalInputs, int64_t numSlots, int64_t slotSize, int64_t contextSize,
	std::vector<int64_t>& outUsedSlots) {

	RG_NO_GRAD;
	RG_ASSERT(input.dim() == 2 && input.is_contiguous() && input.scalar_type() == torch::kFloat32 && input.device().is_cpu());

	int64_t batchSize = input.size(0), inputStride = input.size(1);
	const float* data = input.data_ptr<float>();

	outUsedSlots.clear();
	for (int64_t n = 0; n < batchSize; n++) {
		const float* slots = data + n * inputStride + numGlobalInputs;
		for (int64_t s = 0; s < numSlots; s++)
			if (slots[s * slotSize] > 0.5f)
				outUsedSlots.push_back(n * numSlots + s);
	}

	int64_t rowSize = slotSize + contextSize;
	torch::Tensor result = torch::empty({ (int64_t)outUsedSlots.size(), rowSize });
	float* out = result.data_ptr<float>();
	for (size_t r = 0; r < outUsedSlots.size(); r++) {
		int64_t n = outUsedSlots[r] / numSlots, s = outUsedSlots[r] % numSlots;
		const float* sample = data + n * inputStride;
		memcpy(out + r * rowSize, sample + numGlobalInputs + s * slotSize, slotSize * sizeof(float));
		memcpy(out + r * rowSize + slotSize, sample + numGlobalInputs - contextSize, contextSize * sizeof(float));
	}

	return result;
}

torch::Tensor GGL::SlotPool::Pool(
	const torch::Tensor& input, int64_t numGlobalInputs,
	const torch::Tensor& encodings, const std::vector<int64_t>& usedSlots, const std::vector<int>& groupSizes) {

	RG_ASSERT(input.dim() == 2 && input.is_contiguous() && input.scalar_type() == torch::kFloat32 && input.device().is_cpu());
	RG_ASSERT(encodings.dim() == 2 && encodings.size(0) == usedSlots.size());

	std::vector<int> groupOfSlots = GetGroupOfSlots(groupSizes);
	int64_t numGroups = groupSizes.size();
	torch::Tensor contiguousEncodings = encodings.contiguous();

	if (torch::GradMode::is_enabled() && contiguousEncodings.requires_grad()) {
		return SlotPoolFunction::apply(
			contiguousEncodings, input,
			torch::tensor(usedSlots, torch::kInt64), numGlobalInputs,
			torch::tensor(groupOfSlots, torch::kInt32), numGroups
		);
	} else {
		RG_NO_GRAD;
		int64_t batchSize = input.size(0), encodingSize = contiguousEncodings.size(1);
		torch::Tensor out = torch::empty({ batchSize, numGlobalInputs + numGroups * 2 * encodingSize });
		std::vector<float> counts(batchSize * numGroups);
		RunPoolKernel(
			input.data_ptr<float>(), input.size(1), batchSize, numGlobalInputs,
			contiguousEncodings.data_ptr<float>(), encodingSize, usedSlots.data(), usedSlots.size(),
			groupOfSlots, numGroups,
			out.data_ptr<float>(), counts.data()
		);
		return out;
	}
}
