// Checks that the optimized learning code computes the same things as the original, straightforward formulation.
//
// The PPO learner was changed for speed in three ways:
//	1. The shared head is run once per minibatch instead of once for the policy and once for the critic
//	2. Hidden layers (linear -> layer norm -> ReLU) are run by hand-written kernels on the CPU (Util/FusedLayer)
//	3. A batch is split into minibatches that several threads work on at once, and the gradients are added up
// None of these should change the result beyond float32 rounding. This test measures that:
//	it computes the gradients of a batch with the original formulation (plain torch modules, whole batch at once, shared head twice),
//	in float32 and in float64, and compares the optimized code against both.
//
// Build and run (from the repo root):
//	cmake --build build --target GigaLearnTests
//	./build/GigaLearnTests [checkpoint folder]

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/GAE.h>
#include <GigaLearnCPP/Util/FusedLayer.h>

#include <torch/nn/utils/convert_parameters.h>

using namespace GGL;

static constexpr int OBS_SIZE = 109, NUM_ACTIONS = 90;
static constexpr int64_t BATCH_SIZE = 6000;

struct RefNets {
	torch::nn::Sequential sharedHead, policy, critic;

	std::vector<torch::Tensor> Params() {
		std::vector<torch::Tensor> result;
		for (auto& seq : { sharedHead, policy, critic })
			for (auto& param : seq->parameters())
				result.push_back(param);
		return result;
	}
};

// Copies of the learner's models as plain torch modules of the given type
static RefNets MakeRefNets(PPOLearner& ppo, torch::Dtype dtype) {
	auto fnClone = [&](Model* model) {
		auto clone = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(model->seq->clone());
		clone->to(dtype);
		return torch::nn::Sequential(clone);
	};

	RefNets nets;
	nets.sharedHead = fnClone(ppo.models["shared_head"]);
	nets.policy = fnClone(ppo.models["policy"]);
	nets.critic = fnClone(ppo.models["critic"]);
	return nets;
}

// The loss exactly as the original PPOLearner::Learn() computed it on the CPU: the whole batch at once, 
//	with the shared head evaluated once for the policy and once more for the critic.
static torch::Tensor OriginalLoss(RefNets& nets, const ExperienceTensors& batch, const PPOLearnerConfig& config, torch::Dtype dtype) {
	auto obs = batch.states.to(dtype);
	auto oldProbs = batch.logProbs.to(dtype);
	auto advantages = batch.advantages.to(dtype);
	auto targetValues = batch.targetValues.to(dtype);
	auto actionMasks = batch.actionMasks.to(torch::kBool);

	// InferPolicyProbsFromModels()
	auto logits = nets.policy->forward(nets.sharedHead->forward(obs)) / config.policyTemperature;
	auto probs = torch::softmax(logits + (-1e10f) * actionMasks.logical_not(), -1).view({ -1, NUM_ACTIONS }).clamp(1e-11f, 1);

	auto logProbs = probs.log().gather(-1, batch.actions.unsqueeze(-1));
	auto entropy = (-(probs.log() * probs).sum(-1) / logf(NUM_ACTIONS)).mean(); // ComputeEntropy() with maskEntropy = false
	logProbs = logProbs.view_as(oldProbs);

	auto ratio = torch::exp(logProbs - oldProbs);
	auto clipped = torch::clamp(ratio, 1 - config.clipRange, 1 + config.clipRange);
	auto policyLoss = -torch::min(ratio * advantages, clipped * advantages).mean();
	auto ppoLoss = policyLoss - entropy * config.entropyScale;

	// InferCritic()
	auto vals = nets.critic->forward(nets.sharedHead->forward(obs)).flatten().view_as(targetValues);
	auto criticLoss = torch::mse_loss(vals, targetValues);

	return ppoLoss + criticLoss;
}

static std::vector<torch::Tensor> GradsOf(const std::vector<torch::Tensor>& params) {
	std::vector<torch::Tensor> result;
	for (auto& param : params)
		result.push_back(param.grad().detach().clone().to(torch::kFloat64));
	return result;
}

static double RelativeDiff(const std::vector<torch::Tensor>& a, const std::vector<torch::Tensor>& b) {
	double diffSq = 0, normSq = 0;
	for (size_t i = 0; i < a.size(); i++) {
		diffSq += (a[i] - b[i]).pow(2).sum().item<double>();
		normSq += b[i].pow(2).sum().item<double>();
	}
	return sqrt(diffSq / normSq);
}

static std::vector<torch::Tensor> LearnerParams(PPOLearner& ppo) {
	// Same order as RefNets::Params()
	std::vector<torch::Tensor> result;
	for (const char* name : { "shared_head", "policy", "critic" })
		for (auto& param : ppo.models[name]->parameters())
			result.push_back(param);
	return result;
}

static void ZeroGrads(PPOLearner& ppo) {
	for (auto& param : LearnerParams(ppo))
		if (param.grad().defined())
			param.mutable_grad() = torch::Tensor();
}

static int g_NumFailed = 0;
static void Check(bool condition, const std::string& what) {
	std::cout << (condition ? "  [ok]     " : "  [FAILED] ") << what << std::endl;
	g_NumFailed += !condition;
}

int main(int argc, char* argv[]) {
	torch::manual_seed(123);

	{ // Adjacent trajectories must not borrow each other's bootstrap values (or cross a true goal).
		auto rewards = torch::tensor({1.f, 2.f, 3.f, 4.f});
		auto terminals = torch::tensor({(int)RLGC::TerminalType::NOT_TERMINAL, (int)RLGC::TerminalType::TRUNCATED,
			(int)RLGC::TerminalType::NORMAL, (int)RLGC::TerminalType::TRUNCATED}).to(torch::kInt8);
		torch::Tensor adv, targets, returns;
		float clipped;
		GAE::Compute(rewards, terminals, torch::zeros_like(rewards), torch::tensor({10.f, 40.f}),
			adv, targets, returns, clipped, 1.f, 1.f);
		Check(torch::equal(targets, torch::tensor({13.f, 12.f, 3.f, 44.f})), "GAE uses the matching bootstrap at each truncation");
	}

	PPOLearnerConfig config = {};
	config.sharedHead.layerSizes = { 256, 256 };
	config.policy.layerSizes = { 256, 256, 256 };
	config.critic.layerSizes = { 256, 256, 256 };
	config.batchSize = BATCH_SIZE;
	config.tsPerItr = BATCH_SIZE;
	config.entropyScale = 0.035f;
	config.policyLR = config.criticLR = 2e-4f;

	PPOLearner ppo = PPOLearner(OBS_SIZE, NUM_ACTIONS, config, torch::Device(torch::kCPU));
	if (argc > 1) {
		std::cout << "Loading models from " << argv[1] << std::endl;
		ppo.LoadFrom(argv[1]);
	} else {
		std::cout << "Using freshly initialized models (pass a checkpoint folder to test trained ones)" << std::endl;
	}

	if (!FusedLayer::IsSupported())
		std::cout << "NOTE: Fused layers are not supported in this build, so they are not tested" << std::endl;

	// A batch of made-up experience
	ExperienceTensors batch;
	{
		RG_NO_GRAD;
		batch.states = torch::randn({ BATCH_SIZE, OBS_SIZE });
		batch.actionMasks = (torch::rand({ BATCH_SIZE, NUM_ACTIONS }) < 0.4).to(torch::kUInt8);
		batch.actionMasks.select(1, 0).fill_(1);
		auto probs = PPOLearner::InferPolicyProbsFromModels(ppo.models, batch.states, batch.actionMasks, 1, false);
		batch.actions = torch::multinomial(probs, 1, true).flatten();
		batch.logProbs = (probs.log().gather(-1, batch.actions.unsqueeze(-1)).flatten() + 0.05f * torch::randn({ BATCH_SIZE })).contiguous();
		batch.advantages = torch::randn({ BATCH_SIZE });
		batch.targetValues = torch::randn({ BATCH_SIZE });
	}

	//////////////////////////////////////////////////////////////////////////////////////
	std::cout << "\nOutputs of the models, fused layers compared to the torch modules (largest difference / largest value):" << std::endl;
	if (FusedLayer::IsSupported()) {
		RG_NO_GRAD;
		for (const char* name : { "shared_head", "policy", "critic" }) {
			Model* model = ppo.models[name];
			torch::Tensor input = (std::string(name) == "shared_head") ? batch.states : torch::randn({ BATCH_SIZE, 256 }).relu();

			FusedLayer::SetEnabled(false);
			auto outTorch = model->Forward(input, false);
			auto outSeq = model->seq->forward(input);
			FusedLayer::SetEnabled(true);
			auto outFused = model->Forward(input, false);

			double scale = outTorch.abs().max().item<double>();
			double diff = (outFused - outTorch).abs().max().item<double>() / scale;
			bool sameAsSeq = torch::equal(outTorch, outSeq);
			Check(diff < 1e-4 && sameAsSeq, RS_STR(name << ": " << std::scientific << std::setprecision(2) << diff));
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////
	std::cout << "\nGradients of one batch of " << BATCH_SIZE << " samples (relative difference of all " << "parameters' gradients together):" << std::endl;

	// Reference in float64: the original formulation with (nearly) no rounding error
	std::vector<torch::Tensor> grads64;
	{
		RefNets nets = MakeRefNets(ppo, torch::kFloat64);
		OriginalLoss(nets, batch, config, torch::kFloat64).backward();
		grads64 = GradsOf(nets.Params());
	}

	// The original formulation in float32, as it used to run
	std::vector<torch::Tensor> gradsOriginal;
	{
		RefNets nets = MakeRefNets(ppo, torch::kFloat32);
		OriginalLoss(nets, batch, config, torch::kFloat32).backward();
		gradsOriginal = GradsOf(nets.Params());
	}
	double originalError = RelativeDiff(gradsOriginal, grads64);
	std::cout << "  Original code (float32) vs exact (float64): " << std::scientific << std::setprecision(2) << originalError << "   <- float32 rounding, the yardstick for the rest" << std::endl;

	auto fnTestLearner = [&](const std::string& name, bool fused, int numThreads, int64_t miniBatchSize) -> std::vector<torch::Tensor> {
		FusedLayer::SetEnabled(fused);
		ppo.numLearnThreads = numThreads;
		ppo.miniBatchSize = miniBatchSize;
		ZeroGrads(ppo);

		PPOLearner::LearnStats stats = {};
		ppo.AccumulateBatchGrads(batch, true, true, stats);
		auto grads = GradsOf(LearnerParams(ppo));
		ZeroGrads(ppo);

		double error = RelativeDiff(grads, grads64);
		double diffToOriginal = RelativeDiff(grads, gradsOriginal);
		Check(
			error < RS_MAX(originalError * 3, 1e-6),
			RS_STR(name << ": " << std::scientific << std::setprecision(2) << error << " vs exact, " << diffToOriginal << " vs original code")
		);
		return grads;
	};

	fnTestLearner("Shared head once, whole batch at once, torch modules  ", false, 1, BATCH_SIZE);
	fnTestLearner("Shared head once, minibatches of 1024, torch modules  ", false, 1, 1024);
	if (FusedLayer::IsSupported()) {
		auto single = fnTestLearner("Fused layers, minibatches of 1024, 1 thread           ", true, 1, 1024);
		auto twoThreads = fnTestLearner("Fused layers, minibatches of 1024, 2 threads          ", true, 2, 1024);
		auto fourThreads = fnTestLearner("Fused layers, minibatches of 1024, 4 threads          ", true, 4, 1024);
		fnTestLearner("Fused layers, minibatches of 700 (uneven), 3 threads  ", true, 3, 700);

		std::cout << "  (2 threads vs 1 thread: " << RelativeDiff(twoThreads, single) << ", 4 threads vs 1 thread: " << RelativeDiff(fourThreads, single) << ")" << std::endl;
	} else {
		fnTestLearner("Torch modules, minibatches of 1024, 2 threads         ", false, 2, 1024);
	}

	//////////////////////////////////////////////////////////////////////////////////////
	std::cout << "\nValue predictions in parts and with threads, compared to all at once:" << std::endl;
	{
		RG_NO_GRAD;
		FusedLayer::SetEnabled(false);
		ppo.numLearnThreads = 1;
		ppo.miniBatchSize = BATCH_SIZE;
		auto reference = ppo.InferCritic(batch.states);

		FusedLayer::SetEnabled(true);
		ppo.numLearnThreads = 3;
		ppo.miniBatchSize = 700;
		auto inParts = ppo.InferCriticInParts(batch.states);

		double diff = (inParts - reference).abs().max().item<double>() / reference.abs().max().item<double>();
		Check(diff < 1e-4, RS_STR("largest difference / largest value: " << std::scientific << std::setprecision(2) << diff));
	}

	//////////////////////////////////////////////////////////////////////////////////////
	std::cout << "\nA full Learn() call (2 epochs) with the optimized code:" << std::endl;
	{
		FusedLayer::SetEnabled(true);
		ppo.numLearnThreads = 2;
		ppo.miniBatchSize = 1024;

		ExperienceBuffer experience = ExperienceBuffer(123, torch::kCPU);
		experience.data = batch;

		auto before = torch::nn::utils::parameters_to_vector(LearnerParams(ppo)).clone();
		Report report = {};
		ppo.Learn(experience, report, false);
		auto after = torch::nn::utils::parameters_to_vector(LearnerParams(ppo));

		bool finite = torch::isfinite(after).all().item<bool>();
		double change = (after - before).norm().item<double>();
		Check(finite && change > 0, RS_STR("parameters changed by " << std::scientific << std::setprecision(2) << change << " and are finite"));
		Check(
			report.Has("Policy Entropy") && report.Has("Mean KL Divergence") && report.Has("Critic Loss"),
			RS_STR("reported entropy " << std::fixed << std::setprecision(4) << report["Policy Entropy"] << ", KL " << report["Mean KL Divergence"] << ", critic loss " << report["Critic Loss"])
		);
	}

	std::cout << std::endl << (g_NumFailed ? "FAILED: " : "All checks passed") << (g_NumFailed ? std::to_string(g_NumFailed) + " check(s)" : "") << std::endl;
	return g_NumFailed ? 1 : 0;
}
