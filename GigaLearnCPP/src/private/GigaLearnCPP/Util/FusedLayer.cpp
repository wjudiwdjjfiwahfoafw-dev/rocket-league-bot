#include "FusedLayer.h"

#include <torch/autograd.h>
#include <torch/csrc/autograd/custom_function.h>
#include <ATen/Parallel.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

// The kernels below are written with the vector extensions of GCC and Clang (16 floats per vector).
// With other compilers the fused layer reports itself as unsupported, and the regular torch modules are used.
#if defined(__GNUC__) || defined(__clang__)
#define GGL_FUSED_SUPPORTED true
#else
#define GGL_FUSED_SUPPORTED false
#endif

// Compile the kernels for several instruction sets and pick the best one for the CPU at runtime,
//	so that their speed does not depend on building with -march=native.
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12) && defined(__x86_64__) && defined(__linux__)
#define GGL_KERNEL_TARGETS __attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
#else
#define GGL_KERNEL_TARGETS
#endif

namespace {

#if GGL_FUSED_SUPPORTED

	constexpr int VEC_SIZE = 16;
	typedef float FVec __attribute__((vector_size(VEC_SIZE * sizeof(float))));
	typedef float FVecUnaligned __attribute__((vector_size(VEC_SIZE * sizeof(float)), aligned(4), may_alias));

// These are macros and not functions, as the kernels are compiled for several instruction sets (functions may then not get inlined)
#define VEC_LOAD(ptr) (*(const FVecUnaligned*)(ptr))
#define VEC_STORE(ptr, vec) (*(FVecUnaligned*)(ptr) = (vec))

// Sum of the elements of a vector (added up pairwise)
#define VEC_SUM(vec) ({ \
		float _elems[VEC_SIZE]; \
		FVec _vec = (vec); \
		memcpy(_elems, &_vec, sizeof(_elems)); \
		for (int _n = VEC_SIZE; _n > 1; _n /= 2) \
			for (int _k = 0; _k < _n / 2; _k++) \
				_elems[_k] = _elems[2 * _k] + _elems[2 * _k + 1]; \
		_elems[0]; \
	})

	// Turns the result of (input x weight^T) into the layer's output, for the rows [rowBegin, rowEnd).
	//	z: [n, d], in: the matrix product, out: the normalized values before the layer norm's weight and bias ("zHat", needed for the backward pass)
	//	a: [n, d], out: the output of the layer. Can be the same memory as z if zHat is not needed.
	//	rstd: [n] or NULL, out: 1 / standard deviation of each row
	// Sums over a row use several partial sums, so they are both fast and accurate.
	GGL_KERNEL_TARGETS
	void ForwardKernel(
		float* z, float* a, float* rstd,
		const float* bias, const float* gamma, const float* beta,
		int64_t rowBegin, int64_t rowEnd, int64_t d, float eps) {

		const float invD = 1.0f / d;
		const FVec zero = {};

		for (int64_t i = rowBegin; i < rowEnd; i++) {
			float* zRow = z + i * d;
			float* aRow = a + i * d;

			// Add the bias, and sum up the row
			FVec sum0 = {}, sum1 = {}, sum2 = {}, sum3 = {};
			int64_t j = 0;
			for (; j + 4 * VEC_SIZE <= d; j += 4 * VEC_SIZE) {
				FVec v0 = VEC_LOAD(zRow + j) + VEC_LOAD(bias + j);
				FVec v1 = VEC_LOAD(zRow + j + VEC_SIZE) + VEC_LOAD(bias + j + VEC_SIZE);
				FVec v2 = VEC_LOAD(zRow + j + 2 * VEC_SIZE) + VEC_LOAD(bias + j + 2 * VEC_SIZE);
				FVec v3 = VEC_LOAD(zRow + j + 3 * VEC_SIZE) + VEC_LOAD(bias + j + 3 * VEC_SIZE);
				VEC_STORE(zRow + j, v0);
				VEC_STORE(zRow + j + VEC_SIZE, v1);
				VEC_STORE(zRow + j + 2 * VEC_SIZE, v2);
				VEC_STORE(zRow + j + 3 * VEC_SIZE, v3);
				sum0 += v0; sum1 += v1; sum2 += v2; sum3 += v3;
			}
			for (; j + VEC_SIZE <= d; j += VEC_SIZE) {
				FVec v = VEC_LOAD(zRow + j) + VEC_LOAD(bias + j);
				VEC_STORE(zRow + j, v);
				sum0 += v;
			}
			float sum = VEC_SUM((sum0 + sum1) + (sum2 + sum3));
			for (; j < d; j++) {
				float v = zRow[j] + bias[j];
				zRow[j] = v;
				sum += v;
			}
			const float mean = sum * invD;

			// Center the row, and sum up the squares
			sum0 = sum1 = sum2 = sum3 = zero;
			j = 0;
			for (; j + 4 * VEC_SIZE <= d; j += 4 * VEC_SIZE) {
				FVec c0 = VEC_LOAD(zRow + j) - mean;
				FVec c1 = VEC_LOAD(zRow + j + VEC_SIZE) - mean;
				FVec c2 = VEC_LOAD(zRow + j + 2 * VEC_SIZE) - mean;
				FVec c3 = VEC_LOAD(zRow + j + 3 * VEC_SIZE) - mean;
				VEC_STORE(zRow + j, c0);
				VEC_STORE(zRow + j + VEC_SIZE, c1);
				VEC_STORE(zRow + j + 2 * VEC_SIZE, c2);
				VEC_STORE(zRow + j + 3 * VEC_SIZE, c3);
				sum0 += c0 * c0; sum1 += c1 * c1; sum2 += c2 * c2; sum3 += c3 * c3;
			}
			for (; j + VEC_SIZE <= d; j += VEC_SIZE) {
				FVec c = VEC_LOAD(zRow + j) - mean;
				VEC_STORE(zRow + j, c);
				sum0 += c * c;
			}
			float sumSq = VEC_SUM((sum0 + sum1) + (sum2 + sum3));
			for (; j < d; j++) {
				float c = zRow[j] - mean;
				zRow[j] = c;
				sumSq += c * c;
			}
			const float rowRstd = 1.0f / std::sqrt(sumSq * invD + eps);
			if (rstd)
				rstd[i] = rowRstd;

			// Normalize, apply the layer norm's weight and bias, then the ReLU
			j = 0;
			for (; j + VEC_SIZE <= d; j += VEC_SIZE) {
				FVec zHat = VEC_LOAD(zRow + j) * rowRstd;
				FVec y = zHat * VEC_LOAD(gamma + j) + VEC_LOAD(beta + j);
				VEC_STORE(zRow + j, zHat);
				VEC_STORE(aRow + j, (y > zero) ? y : zero);
			}
			for (; j < d; j++) {
				float zHat = zRow[j] * rowRstd;
				float y = zHat * gamma[j] + beta[j];
				zRow[j] = zHat;
				aRow[j] = y > 0 ? y : 0;
			}
		}
	}

	// From the gradient of the layer's output (dA), computes the gradient of the matrix product (dZ) for the rows [rowBegin, rowEnd),
	//	and adds these rows' share of the gradients of the bias and of the layer norm's weight (gamma) and bias (beta) to dGamma/dBeta/dBias.
	GGL_KERNEL_TARGETS
	void BackwardKernel(
		const float* dA, const float* a, const float* zHat, const float* rstd, const float* gamma,
		float* dZ, float* dGamma, float* dBeta, float* dBias,
		int64_t rowBegin, int64_t rowEnd, int64_t d) {

		const float invD = 1.0f / d;
		const FVec zero = {};

		for (int64_t i = rowBegin; i < rowEnd; i++) {
			const float* dARow = dA + i * d;
			const float* aRow = a + i * d;
			const float* zHatRow = zHat + i * d;
			float* dZRow = dZ + i * d;

			// The ReLU (dY), then the layer norm's weight and bias (g = gradient of zHat)
			FVec sumG0 = {}, sumG1 = {}, sumGZ0 = {}, sumGZ1 = {};
			int64_t j = 0;
			for (; j + 2 * VEC_SIZE <= d; j += 2 * VEC_SIZE) {
				FVec zHat0 = VEC_LOAD(zHatRow + j), zHat1 = VEC_LOAD(zHatRow + j + VEC_SIZE);
				FVec dY0 = (VEC_LOAD(aRow + j) > zero) ? VEC_LOAD(dARow + j) : zero;
				FVec dY1 = (VEC_LOAD(aRow + j + VEC_SIZE) > zero) ? VEC_LOAD(dARow + j + VEC_SIZE) : zero;
				VEC_STORE(dGamma + j, VEC_LOAD(dGamma + j) + dY0 * zHat0);
				VEC_STORE(dGamma + j + VEC_SIZE, VEC_LOAD(dGamma + j + VEC_SIZE) + dY1 * zHat1);
				VEC_STORE(dBeta + j, VEC_LOAD(dBeta + j) + dY0);
				VEC_STORE(dBeta + j + VEC_SIZE, VEC_LOAD(dBeta + j + VEC_SIZE) + dY1);
				FVec g0 = dY0 * VEC_LOAD(gamma + j), g1 = dY1 * VEC_LOAD(gamma + j + VEC_SIZE);
				VEC_STORE(dZRow + j, g0);
				VEC_STORE(dZRow + j + VEC_SIZE, g1);
				sumG0 += g0; sumG1 += g1;
				sumGZ0 += g0 * zHat0; sumGZ1 += g1 * zHat1;
			}
			for (; j + VEC_SIZE <= d; j += VEC_SIZE) {
				FVec zHat0 = VEC_LOAD(zHatRow + j);
				FVec dY0 = (VEC_LOAD(aRow + j) > zero) ? VEC_LOAD(dARow + j) : zero;
				VEC_STORE(dGamma + j, VEC_LOAD(dGamma + j) + dY0 * zHat0);
				VEC_STORE(dBeta + j, VEC_LOAD(dBeta + j) + dY0);
				FVec g0 = dY0 * VEC_LOAD(gamma + j);
				VEC_STORE(dZRow + j, g0);
				sumG0 += g0;
				sumGZ0 += g0 * zHat0;
			}
			float sumG = VEC_SUM(sumG0 + sumG1), sumGZ = VEC_SUM(sumGZ0 + sumGZ1);
			for (; j < d; j++) {
				float dY = aRow[j] > 0 ? dARow[j] : 0;
				dGamma[j] += dY * zHatRow[j];
				dBeta[j] += dY;
				float g = dY * gamma[j];
				dZRow[j] = g;
				sumG += g;
				sumGZ += g * zHatRow[j];
			}
			const float meanG = sumG * invD;
			const float meanGZ = sumGZ * invD;
			const float rowRstd = rstd[i];

			// The normalization: dZ = rstd * (g - mean(g) - zHat * mean(g * zHat))
			j = 0;
			for (; j + VEC_SIZE <= d; j += VEC_SIZE) {
				FVec dz = (VEC_LOAD(dZRow + j) - meanG - VEC_LOAD(zHatRow + j) * meanGZ) * rowRstd;
				VEC_STORE(dZRow + j, dz);
				VEC_STORE(dBias + j, VEC_LOAD(dBias + j) + dz);
			}
			for (; j < d; j++) {
				float dz = rowRstd * (dZRow[j] - meanG - zHatRow[j] * meanGZ);
				dZRow[j] = dz;
				dBias[j] += dz;
			}
		}
	}

#else
	// Never called (IsSupported() is false), these only exist so that the code below compiles
	void ForwardKernel(float*, float*, float*, const float*, const float*, const float*, int64_t, int64_t, int64_t, float) {}
	void BackwardKernel(const float*, const float*, const float*, const float*, const float*, float*, float*, float*, float*, int64_t, int64_t, int64_t) {}
#endif

	// Rows per task when a kernel is spread over torch's threads (a task is at least as much work as torch's own grain size for element-wise operations)
	constexpr int64_t GRAIN_ELEMENTS = 32768;
	int64_t GetGrainRows(int64_t d) {
		return RS_MAX((int64_t)1, GRAIN_ELEMENTS / RS_MAX(d, (int64_t)1));
	}

	void RunForwardKernel(
		float* z, float* a, float* rstd,
		const float* bias, const float* gamma, const float* beta,
		int64_t n, int64_t d, float eps) {

		at::parallel_for(0, n, GetGrainRows(d), [&](int64_t rowBegin, int64_t rowEnd) {
			ForwardKernel(z, a, rstd, bias, gamma, beta, rowBegin, rowEnd, d, eps);
		});
	}

	void RunBackwardKernel(
		const float* dA, const float* a, const float* zHat, const float* rstd, const float* gamma,
		float* dZ, float* dGamma, float* dBeta, float* dBias,
		int64_t n, int64_t d) {

		// Every thread adds up its rows' gradients of gamma/beta/bias in its own block: [thread][gamma|beta|bias][d]
		int numThreads = RS_MAX(at::get_num_threads(), 1);
		std::vector<float> sums((size_t)numThreads * 3 * d, 0.f);

		at::parallel_for(0, n, GetGrainRows(d), [&](int64_t rowBegin, int64_t rowEnd) {
			int threadIdx = RS_CLAMP(at::get_thread_num(), 0, numThreads - 1);
			float* threadSums = sums.data() + (size_t)threadIdx * 3 * d;
			BackwardKernel(dA, a, zHat, rstd, gamma, dZ, threadSums, threadSums + d, threadSums + 2 * d, rowBegin, rowEnd, d);
		});

		for (int64_t j = 0; j < d; j++)
			dGamma[j] = dBeta[j] = dBias[j] = 0;
		for (int t = 0; t < numThreads; t++) {
			const float* threadSums = sums.data() + (size_t)t * 3 * d;
			for (int64_t j = 0; j < d; j++) {
				dGamma[j] += threadSums[j];
				dBeta[j] += threadSums[d + j];
				dBias[j] += threadSums[2 * d + j];
			}
		}
	}

	struct FusedLayerFunction : public torch::autograd::Function<FusedLayerFunction> {

		static torch::Tensor forward(
			torch::autograd::AutogradContext* ctx,
			torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor gamma, torch::Tensor beta, double eps) {

			int64_t n = input.size(0), d = weight.size(0);

			torch::Tensor z = at::mm(input, weight.t()); // Becomes zHat
			torch::Tensor a = torch::empty({ n, d });
			torch::Tensor rstd = torch::empty({ n });
			RunForwardKernel(
				z.data_ptr<float>(), a.data_ptr<float>(), rstd.data_ptr<float>(),
				bias.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
				n, d, (float)eps
			);

			ctx->save_for_backward({ input, weight, gamma, a });
			ctx->saved_data["zHat"] = z;
			ctx->saved_data["rstd"] = rstd;
			ctx->saved_data["inputNeedsGrad"] = input.requires_grad();
			return a;
		}

		static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list gradOutputs) {
			auto saved = ctx->get_saved_variables();
			torch::Tensor input = saved[0], weight = saved[1], gamma = saved[2], a = saved[3];
			torch::Tensor zHat = ctx->saved_data["zHat"].toTensor();
			torch::Tensor rstd = ctx->saved_data["rstd"].toTensor();
			bool inputNeedsGrad = ctx->saved_data["inputNeedsGrad"].toBool();

			torch::Tensor dA = gradOutputs[0].contiguous();
			int64_t n = a.size(0), d = a.size(1);

			torch::Tensor dZ = torch::empty({ n, d });
			torch::Tensor dGamma = torch::empty({ d }), dBeta = torch::empty({ d }), dBias = torch::empty({ d });
			RunBackwardKernel(
				dA.data_ptr<float>(), a.data_ptr<float>(), zHat.data_ptr<float>(), rstd.data_ptr<float>(), gamma.data_ptr<float>(),
				dZ.data_ptr<float>(), dGamma.data_ptr<float>(), dBeta.data_ptr<float>(), dBias.data_ptr<float>(),
				n, d
			);

			torch::Tensor dWeight = at::mm(dZ.t(), input);
			torch::Tensor dInput;
			if (inputNeedsGrad)
				dInput = at::mm(dZ, weight);

			// One entry per argument of forward() (the last one is eps)
			return { dInput, dWeight, dBias, dGamma, dBeta, torch::Tensor() };
		}
	};

	std::atomic<bool> g_Enabled = (getenv("GGL_NO_FUSED_LAYERS") == NULL);
}

bool GGL::FusedLayer::IsSupported() {
	return GGL_FUSED_SUPPORTED;
}

bool GGL::FusedLayer::IsEnabled() {
	return GGL_FUSED_SUPPORTED && g_Enabled;
}

void GGL::FusedLayer::SetEnabled(bool enabled) {
	g_Enabled = enabled;
}

torch::Tensor GGL::FusedLayer::Forward(
	const torch::Tensor& input,
	const torch::Tensor& weight, const torch::Tensor& bias,
	const torch::Tensor& lnWeight, const torch::Tensor& lnBias, double lnEps) {

	torch::Tensor contiguousInput = input.contiguous();

	bool needsGrad = torch::GradMode::is_enabled() && (
		contiguousInput.requires_grad() || weight.requires_grad() || bias.requires_grad() ||
		lnWeight.requires_grad() || lnBias.requires_grad());

	if (needsGrad) {
		return FusedLayerFunction::apply(contiguousInput, weight, bias, lnWeight, lnBias, lnEps);
	} else {
		// Inference: nothing has to be kept for a backward pass, so the output can overwrite the matrix product
		RG_NO_GRAD;
		torch::Tensor z = at::mm(contiguousInput, weight.t());
		RunForwardKernel(
			z.data_ptr<float>(), z.data_ptr<float>(), NULL,
			bias.data_ptr<float>(), lnWeight.data_ptr<float>(), lnBias.data_ptr<float>(),
			z.size(0), z.size(1), (float)lnEps
		);
		return z;
	}
}
