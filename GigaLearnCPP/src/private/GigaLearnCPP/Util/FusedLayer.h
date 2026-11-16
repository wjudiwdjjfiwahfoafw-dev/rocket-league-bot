#pragma once
#include "../FrameworkTorch.h"

namespace GGL {

	// Linear -> LayerNorm -> ReLU as a single operation with hand-written kernels (CPU, float32 only).
	//
	// Why: On a CPU the matrix multiplications of a layer run at the hardware limit (MKL), 
	//	but torch's LayerNorm, ReLU, bias and their backward passes each make their own pass(es) over the whole activation tensor.
	// In a profile of the example bot they took about 40% as long as the matrix multiplications themselves.
	// Here, everything except the multiplications is done row by row while the row is in the L1 cache.
	//
	// The result is the same as the three torch modules up to float32 rounding
	//	(see tests/TestLearnerMath.cpp, which compares both against a float64 reference).
	namespace FusedLayer {

		// If false, the kernels would not be vectorized in this build, and the regular torch modules should be used instead
		bool IsSupported();

		// Can be turned off at runtime (also with the environment variable GGL_NO_FUSED_LAYERS=1)
		bool IsEnabled();
		void SetEnabled(bool enabled);

		// input: [N, numInputs], weight: [numOutputs, numInputs], bias/lnWeight/lnBias: [numOutputs]
		// All tensors must be float32 on the CPU.
		// Supports autograd (first order) for the input and all parameters.
		torch::Tensor Forward(
			const torch::Tensor& input,
			const torch::Tensor& weight, const torch::Tensor& bias,
			const torch::Tensor& lnWeight, const torch::Tensor& lnBias, double lnEps
		);
	}
}
