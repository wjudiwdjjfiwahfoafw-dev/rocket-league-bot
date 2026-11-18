// Checks of the observation and the network for 1v1, 2v2 and 3v3:
//	- MultiModeObs: same size in all modes, game mode and "slot is used" inputs, empty slots, same view for both teams
//	- The slot encoder of Model (see SlotEncoderConfig): compared against a simple reference implementation in float64,
//		order of the slots doesn't matter, content of empty slots doesn't matter, gradients, saving and loading
//	- The PPO learner with such a model: threads and minibatches give the same gradients as the whole batch at once,
//		and actions inferred on several threads are the same as on one
//
// Build and run (from the repo root):
//	cmake --build build --target GigaLearnMultiModeTests
//	./build/GigaLearnMultiModeTests

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/Util/FusedLayer.h>
#include <GigaLearnCPP/Util/Models.h>

#include <RLGymCPP/ObsBuilders/MultiModeObs.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/Gamestates/StateUtil.h>
#include <RLGymCPP/EnvSet/EnvSet.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>

#include <torch/autograd.h>

#include <algorithm>
#include <filesystem>
#include <random>

using namespace GGL;
using namespace RLGC;

typedef MultiModeObs Obs;

static int g_NumFailed = 0;
static void Check(bool condition, const std::string& what) {
	std::cout << (condition ? "  [ok]     " : "  [FAILED] ") << what << std::endl;
	g_NumFailed += !condition;
}

static std::string Sci(double val) {
	std::stringstream stream;
	stream << std::scientific << std::setprecision(2) << val;
	return stream.str();
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Random game states

static std::mt19937 g_Rand(42);

static float Rand(float min, float max) {
	return min + (max - min) * (g_Rand() / (float)g_Rand.max());
}

static Vec RandVec(float range) {
	return Vec(Rand(-range, range), Rand(-range, range), Rand(-range, range));
}

static GameState MakeRandomState(int teamSize) {
	GameState state = {};
	for (int i = 0; i < teamSize * 2; i++) {
		Player player = {};
		player.index = i;
		player.carId = i + 1;
		player.team = (i % 2 == 0) ? Team::BLUE : Team::ORANGE;

		player.pos = Vec(Rand(-4000, 4000), Rand(-5000, 5000), Rand(17, 1900));
		player.rotMat = Angle(Rand(-3.1f, 3.1f), Rand(-1.5f, 1.5f), Rand(-3.1f, 3.1f)).ToRotMat();
		player.vel = RandVec(2000);
		player.angVel = RandVec(5);

		player.boost = Rand(0, 100);
		player.isOnGround = g_Rand() % 2;
		player.hasJumped = g_Rand() % 2;
		player.hasDoubleJumped = g_Rand() % 2;
		player.isDemoed = (g_Rand() % 8) == 0;
		player.isSupersonic = g_Rand() % 2;
		for (int j = 0; j < Action::ELEM_AMOUNT; j++)
			player.prevAction[j] = Rand(-1, 1);

		state.players.push_back(player);
	}

	state.ball.pos = Vec(Rand(-4000, 4000), Rand(-5000, 5000), Rand(93, 1900));
	state.ball.vel = RandVec(3000);
	state.ball.angVel = RandVec(6);

	int numPads = CommonValues::BOOST_LOCATIONS_AMOUNT;
	for (int i = 0; i < numPads; i++) {
		bool available = g_Rand() % 2;
		float timer = available ? 0 : Rand(0, 10);
		state.boostPads[i] = available;
		state.boostPadTimers[i] = timer;
		state.boostPadsInv[numPads - i - 1] = available;
		state.boostPadTimersInv[numPads - i - 1] = timer;
	}

	return state;
}

// The same situation with the teams swapped: everything is rotated by 180 degrees, and every player is on the other team
static GameState MirrorState(const GameState& state) {
	GameState mirrored = state;
	for (auto& player : mirrored.players) {
		static_cast<PhysState&>(player) = InvertPhys(player);
		player.team = (player.team == Team::BLUE) ? Team::ORANGE : Team::BLUE;
	}
	static_cast<PhysState&>(mirrored.ball) = InvertPhys(state.ball);
	std::swap(mirrored.boostPads, mirrored.boostPadsInv);
	std::swap(mirrored.boostPadTimers, mirrored.boostPadTimersInv);
	return mirrored;
}

///////////////////////////////////////////////////////////////////////////////////////////////

static void TestObs() {
	std::cout << "\nMultiModeObs (" << Obs::OBS_SIZE << " inputs: " << Obs::GLOBAL_SIZE << " global, " << Obs::NUM_SLOTS << " slots of " << Obs::SLOT_SIZE << "):" << std::endl;

	Obs obsBuilder = Obs();

	for (int teamSize = 1; teamSize <= Obs::MAX_TEAM_SIZE; teamSize++) {
		bool sizeOk = true, modeOk = true, usedOk = true, emptyOk = true, relationOk = true, mirrorOk = true, finiteOk = true;
		float maxAbs = 0;

		for (int trial = 0; trial < 200; trial++) {
			GameState state = MakeRandomState(teamSize);
			GameState mirrored = MirrorState(state);

			for (int playerIdx = 0; playerIdx < state.players.size(); playerIdx++) {
				auto& player = state.players[playerIdx];
				FList obs = obsBuilder.BuildObs(player, state);

				sizeOk &= obs.size() == Obs::OBS_SIZE;
				if (obs.size() != Obs::OBS_SIZE)
					continue;

				for (float val : obs) {
					finiteOk &= std::isfinite(val);
					maxAbs = RS_MAX(maxAbs, fabsf(val));
				}

				// Game mode
				for (int i = 0; i < Obs::NUM_MODE_INPUTS; i++)
					modeOk &= obs[Obs::NUM_PAD_INPUTS + i] == ((i == teamSize - 1) ? 1.f : 0.f);

				// Slots: teamSize - 1 teammates, teamSize opponents, the rest is all zeros
				int numUsed[2] = {};
				for (int slotIdx = 0; slotIdx < Obs::NUM_SLOTS; slotIdx++) {
					const float* slot = obs.data() + Obs::GLOBAL_SIZE + slotIdx * Obs::SLOT_SIZE;
					bool teammateSlot = slotIdx < Obs::NUM_TEAMMATE_SLOTS;
					if (slot[0] == 1) {
						numUsed[teammateSlot ? 0 : 1]++;
						usedOk &= slot[1] == (teammateSlot ? 1.f : 0.f);
					} else {
						for (int i = 0; i < Obs::SLOT_SIZE; i++)
							emptyOk &= slot[i] == 0;
					}
				}
				usedOk &= (numUsed[0] == teamSize - 1) && (numUsed[1] == teamSize);

				// Relation inputs of the first opponent slot (for a blue player, whose view is not rotated)
				if (player.team == Team::BLUE) {
					const Player* opponent = NULL;
					for (auto& other : state.players) {
						if (other.team != player.team) {
							opponent = &other;
							break;
						}
					}

					const float* slot = obs.data() + Obs::GLOBAL_SIZE + Obs::NUM_TEAMMATE_SLOTS * Obs::SLOT_SIZE;
					const float* rel = slot + 2 + Obs::NUM_PLAYER_INPUTS;
					Vec relPos = opponent->pos - player.pos;
					Vec relVel = opponent->vel - player.vel;
					Vec localRelPos = player.rotMat.Dot(relPos);
					float distDiff = (state.ball.pos - opponent->pos).Length() - (state.ball.pos - player.pos).Length();

					float expected[10] = {
						relPos.x * Obs::POS_COEF, relPos.y * Obs::POS_COEF, relPos.z * Obs::POS_COEF,
						relVel.x * Obs::VEL_COEF, relVel.y * Obs::VEL_COEF, relVel.z * Obs::VEL_COEF,
						localRelPos.x * Obs::POS_COEF, localRelPos.y * Obs::POS_COEF, localRelPos.z * Obs::POS_COEF,
						distDiff * Obs::POS_COEF
					};
					for (int i = 0; i < 10; i++)
						relationOk &= fabsf(rel[i] - expected[i]) < 1e-5f;

					// The slot's player inputs start with the opponent's position
					relationOk &= fabsf(slot[2] - opponent->pos.x * Obs::POS_COEF) < 1e-6f;
				}

				// The player sees the same thing when everything is on the other side of the field
				FList mirroredObs = obsBuilder.BuildObs(mirrored.players[playerIdx], mirrored);
				mirrorOk &= (obs == mirroredObs);
			}
		}

		std::string mode = std::to_string(teamSize) + "v" + std::to_string(teamSize);
		Check(sizeOk && finiteOk, mode + ": every observation has " + std::to_string(Obs::OBS_SIZE) + " finite inputs (largest value: " + std::to_string(maxAbs) + ")");
		Check(modeOk, mode + ": game mode input");
		Check(usedOk, mode + ": " + std::to_string(teamSize - 1) + " teammate and " + std::to_string(teamSize) + " opponent slots are used");
		Check(emptyOk, mode + ": empty slots are all zeros");
		Check(relationOk, mode + ": inputs that relate a slot's player to the observing player");
		Check(mirrorOk, mode + ": identical observation when the teams are swapped");
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////

// Observations of random game states of all three modes: [numSamples, OBS_SIZE]
static torch::Tensor MakeObsBatch(int numSamples) {
	Obs obsBuilder = Obs();
	std::vector<float> data = {};
	int count = 0;
	while (count < numSamples) {
		GameState state = MakeRandomState(1 + (g_Rand() % Obs::MAX_TEAM_SIZE));
		for (auto& player : state.players) {
			if (count >= numSamples)
				break;
			data += obsBuilder.BuildObs(player, state);
			count++;
		}
	}
	return torch::tensor(data).reshape({ (int64_t)numSamples, (int64_t)Obs::OBS_SIZE });
}

static SlotEncoderConfig MakeSlotConfig(std::vector<int> layerSizes) {
	SlotEncoderConfig slots = {};
	slots.slotSize = Obs::SLOT_SIZE;
	slots.groupSizes = { Obs::NUM_TEAMMATE_SLOTS, Obs::NUM_OPPONENT_SLOTS };
	slots.contextSize = Obs::CONTEXT_SIZE;
	slots.layerSizes = layerSizes;
	return slots;
}

static torch::nn::Sequential CloneSeq(torch::nn::Sequential& seq, torch::Dtype dtype) {
	auto clone = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(seq->clone());
	clone->to(dtype);
	return torch::nn::Sequential(clone);
}

// What a model with a slot encoder computes, written in the most direct way: every slot is encoded (empty or not),
//	then the encodings of empty slots are set to zero, and maximum and mean are taken per group.
static torch::Tensor ReferenceForward(Model& model, torch::nn::Sequential& slotSeq, torch::nn::Sequential& seq, torch::Tensor input) {
	auto& slots = model.config.slots;
	int64_t numGlobal = model.numGlobalInputs;

	torch::Tensor globalInputs = input.narrow(1, 0, numGlobal);
	torch::Tensor context = globalInputs.narrow(1, numGlobal - slots.contextSize, slots.contextSize);

	std::vector<torch::Tensor> parts = { globalInputs };
	int slotIdx = 0;
	for (int groupSize : slots.groupSizes) {
		torch::Tensor maxEncoding, sumEncoding, usedCount;
		for (int i = 0; i < groupSize; i++, slotIdx++) {
			torch::Tensor slot = input.narrow(1, numGlobal + slotIdx * slots.slotSize, slots.slotSize);
			torch::Tensor used = (slot.narrow(1, 0, 1) > 0.5).to(input.scalar_type());
			torch::Tensor encoding = slotSeq->forward(torch::cat({ slot, context }, 1)) * used;

			maxEncoding = (i == 0) ? encoding : torch::max(maxEncoding, encoding);
			sumEncoding = (i == 0) ? encoding : (sumEncoding + encoding);
			usedCount = (i == 0) ? used : (usedCount + used);
		}
		parts.push_back(maxEncoding);
		parts.push_back(sumEncoding / usedCount.clamp_min(1));
	}

	return seq->forward(torch::cat(parts, 1));
}

static double MaxRelDiff(torch::Tensor a, torch::Tensor b) {
	a = a.to(torch::kFloat64);
	b = b.to(torch::kFloat64);
	return (a - b).abs().max().item<double>() / b.abs().max().item<double>();
}

static double RelativeDiff(const std::vector<torch::Tensor>& a, const std::vector<torch::Tensor>& b) {
	double diffSq = 0, normSq = 0;
	for (size_t i = 0; i < a.size(); i++) {
		diffSq += (a[i].to(torch::kFloat64) - b[i].to(torch::kFloat64)).pow(2).sum().item<double>();
		normSq += b[i].to(torch::kFloat64).pow(2).sum().item<double>();
	}
	return sqrt(diffSq / normSq);
}

static void TestSlotEncoder() {
	std::cout << "\nSlot encoder (a model with " << Obs::OBS_SIZE << " inputs, compared to a direct implementation in float64):" << std::endl;

	PartialModelConfig partialConfig = {};
	partialConfig.layerSizes = { 192, 128 };
	partialConfig.addOutputLayer = false;
	partialConfig.slots = MakeSlotConfig({ 96, 64 });

	ModelConfig config = partialConfig;
	config.numInputs = Obs::OBS_SIZE;
	config.numOutputs = 0;

	Model model = Model("shared_head", config, torch::Device(torch::kCPU));
	{ // Layer norm weights and biases start as ones and zeros, make them something else so that they matter
		RG_NO_GRAD;
		for (auto& param : model.parameters())
			if (param.dim() == 1)
				param.add_(torch::randn_like(param) * 0.3f);
	}

	const int64_t NUM_SAMPLES = 1500;
	torch::Tensor input = MakeObsBatch(NUM_SAMPLES);

	auto slotSeq64 = CloneSeq(model.slotSeq, torch::kFloat64);
	auto seq64 = CloneSeq(model.seq, torch::kFloat64);
	torch::Tensor input64 = input.to(torch::kFloat64);

	torch::Tensor reference;
	{
		RG_NO_GRAD;
		reference = ReferenceForward(model, slotSeq64, seq64, input64);
	}

	Check(model.numGlobalInputs == Obs::GLOBAL_SIZE && reference.size(1) == 128, "sizes: " + std::to_string(model.numGlobalInputs) + " global inputs, " + std::to_string(config.slots.NumOutputs()) + " values from the slots");

	// Outputs
	for (int fused = 0; fused < (FusedLayer::IsSupported() ? 2 : 1); fused++) {
		RG_NO_GRAD;
		FusedLayer::SetEnabled(fused);
		double diff = MaxRelDiff(model.Forward(input, false), reference);
		Check(diff < 1e-5, std::string("output with ") + (fused ? "fused layers" : "torch layers") + ": differs by " + Sci(diff));
	}

	// Gradients of all parameters
	{
		std::vector<torch::Tensor> params64 = seq64->parameters();
		for (auto& param : slotSeq64->parameters())
			params64.push_back(param);
		torch::Tensor target64 = torch::randn_like(reference);
		auto grads64 = torch::autograd::grad({ (ReferenceForward(model, slotSeq64, seq64, input64) - target64).pow(2).mean() }, params64);

		for (int fused = 0; fused < (FusedLayer::IsSupported() ? 2 : 1); fused++) {
			FusedLayer::SetEnabled(fused);
			auto grads = torch::autograd::grad({ (model.Forward(input, false) - target64.to(torch::kFloat32)).pow(2).mean() }, model.parameters());
			double diff = RelativeDiff(grads, grads64);
			Check(grads.size() == grads64.size() && diff < 1e-5, std::string("gradients with ") + (fused ? "fused layers" : "torch layers") + ": differ by " + Sci(diff));
		}
	}
	FusedLayer::SetEnabled(true);

	{
		RG_NO_GRAD;
		torch::Tensor output = model.Forward(input, false);

		// Order of the slots within their group
		{
			std::vector<int64_t> columns = {};
			for (int i = 0; i < Obs::GLOBAL_SIZE; i++)
				columns.push_back(i);
			for (int slotIdx : { 1, 0, 4, 2, 3 }) // Teammates swapped, opponents rotated
				for (int i = 0; i < Obs::SLOT_SIZE; i++)
					columns.push_back(Obs::GLOBAL_SIZE + slotIdx * Obs::SLOT_SIZE + i);

			torch::Tensor permuted = input.index_select(1, torch::tensor(columns));
			double diff = MaxRelDiff(model.Forward(permuted, false), output);
			Check(!torch::equal(permuted, input) && diff < 1e-5, "slots in a different order within their group: output differs by " + Sci(diff));
		}

		// Content of empty slots
		{
			torch::Tensor garbage = input.clone();
			int64_t numEmpty = 0;
			for (int slotIdx = 0; slotIdx < Obs::NUM_SLOTS; slotIdx++) {
				torch::Tensor slot = garbage.narrow(1, Obs::GLOBAL_SIZE + slotIdx * Obs::SLOT_SIZE, Obs::SLOT_SIZE);
				torch::Tensor empty = (slot.narrow(1, 0, 1) < 0.5).to(torch::kFloat32);
				numEmpty += (int64_t)empty.sum().item<double>();
				slot.narrow(1, 1, Obs::SLOT_SIZE - 1).add_(torch::randn({ NUM_SAMPLES, Obs::SLOT_SIZE - 1 }) * 10 * empty);
			}
			torch::Tensor garbageOutput = model.Forward(garbage, false);
			Check(numEmpty > 0 && !torch::equal(garbage, input) && torch::equal(garbageOutput, output), "random values in the " + std::to_string(numEmpty) + " empty slots: output is identical");
		}

		// The slots do matter
		{
			torch::Tensor changed = input.clone();
			changed.narrow(1, Obs::GLOBAL_SIZE + Obs::NUM_TEAMMATE_SLOTS * Obs::SLOT_SIZE + 2, 3).add_(0.2f); // Position of the first opponent
			double diff = MaxRelDiff(model.Forward(changed, false), output);
			Check(diff > 1e-3, "moving an opponent changes the output (by " + Sci(diff) + ")");
		}

		// No used slot at all
		{
			torch::Tensor noSlots = input.narrow(0, 0, 4).clone();
			noSlots.narrow(1, Obs::GLOBAL_SIZE, Obs::NUM_SLOTS * Obs::SLOT_SIZE).zero_();
			torch::Tensor out = model.Forward(noSlots, false);
			double diff = MaxRelDiff(out, ReferenceForward(model, slotSeq64, seq64, noSlots.to(torch::kFloat64)));
			Check(diff < 1e-5, "a batch without any used slot: differs by " + Sci(diff));
		}

		// Saving and loading
		{
			auto folder = std::filesystem::temp_directory_path() / "ggl_multimode_test";
			std::filesystem::create_directories(folder);
			model.Save(folder);

			Model loaded = Model("shared_head", config, torch::Device(torch::kCPU));
			bool differentBefore = !torch::equal(loaded.Forward(input, false), output);
			loaded.Load(folder, false, true);
			Check(differentBefore && torch::equal(loaded.Forward(input, false), output), "saved and loaded model gives identical output");

			Model* clone = model.MakeClone();
			Check(torch::equal(clone->Forward(input, false), output), "cloned model gives identical output");
			delete clone;
			std::filesystem::remove_all(folder);
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////

static std::vector<torch::Tensor> LearnerParams(PPOLearner& ppo) {
	std::vector<torch::Tensor> result;
	for (const char* name : { "shared_head", "policy", "critic" })
		for (auto& param : ppo.models[name]->parameters())
			result.push_back(param);
	return result;
}

static void TestLearner() {
	const int NUM_ACTIONS = 90;
	const int64_t BATCH_SIZE = 3000;

	std::cout << "\nPPO learner with a slot encoder (gradients of one batch of " << BATCH_SIZE << " samples):" << std::endl;

	PPOLearnerConfig config = {};
	config.sharedHead.layerSizes = { 192, 128 };
	config.sharedHead.slots = MakeSlotConfig({ 96, 64 });
	config.policy.layerSizes = { 128, 96 };
	config.critic.layerSizes = { 128, 96 };
	config.batchSize = BATCH_SIZE;
	config.tsPerItr = BATCH_SIZE;
	config.entropyScale = 0.035f;
	config.policyLR = config.criticLR = 2e-4f;

	PPOLearner ppo = PPOLearner(Obs::OBS_SIZE, NUM_ACTIONS, config, torch::Device(torch::kCPU));

	ExperienceTensors batch;
	{
		RG_NO_GRAD;
		batch.states = MakeObsBatch(BATCH_SIZE);
		batch.actionMasks = (torch::rand({ BATCH_SIZE, NUM_ACTIONS }) < 0.4).to(torch::kUInt8);
		batch.actionMasks.select(1, 0).fill_(1);
		auto probs = PPOLearner::InferPolicyProbsFromModels(ppo.models, batch.states, batch.actionMasks, 1, false);
		batch.actions = torch::multinomial(probs, 1, true).flatten();
		batch.logProbs = (probs.log().gather(-1, batch.actions.unsqueeze(-1)).flatten() + 0.05f * torch::randn({ BATCH_SIZE })).contiguous();
		batch.advantages = torch::randn({ BATCH_SIZE });
		batch.targetValues = torch::randn({ BATCH_SIZE });
	}

	auto fnGrads = [&](bool fused, int numThreads, int64_t miniBatchSize) {
		FusedLayer::SetEnabled(fused);
		ppo.numLearnThreads = numThreads;
		ppo.miniBatchSize = miniBatchSize;

		for (auto& param : LearnerParams(ppo))
			if (param.grad().defined())
				param.mutable_grad() = torch::Tensor();

		PPOLearner::LearnStats stats = {};
		ppo.AccumulateBatchGrads(batch, true, true, stats);

		std::vector<torch::Tensor> grads;
		for (auto& param : LearnerParams(ppo))
			grads.push_back(param.grad().detach().clone());
		return grads;
	};

	auto reference = fnGrads(false, 1, BATCH_SIZE);
	bool allDefined = true;
	for (auto& grad : reference)
		allDefined &= grad.abs().sum().item<double>() > 0;
	Check(allDefined, "every parameter gets a gradient (" + std::to_string(reference.size()) + " parameter tensors)");

	double diff = RelativeDiff(fnGrads(false, 1, 512), reference);
	Check(diff < 1e-5, "minibatches of 512, torch layers, 1 thread vs the whole batch at once: differ by " + Sci(diff));

	if (FusedLayer::IsSupported()) {
		diff = RelativeDiff(fnGrads(true, 1, 512), reference);
		Check(diff < 1e-5, "minibatches of 512, fused layers, 1 thread: differ by " + Sci(diff));
		diff = RelativeDiff(fnGrads(true, 2, 512), reference);
		Check(diff < 1e-5, "minibatches of 512, fused layers, 2 threads: differ by " + Sci(diff));
		diff = RelativeDiff(fnGrads(true, 3, 700), reference);
		Check(diff < 1e-5, "minibatches of 700 (uneven), fused layers, 3 threads: differ by " + Sci(diff));
	}

	{ // Value predictions in parts
		RG_NO_GRAD;
		FusedLayer::SetEnabled(false);
		ppo.numLearnThreads = 1;
		ppo.miniBatchSize = BATCH_SIZE;
		auto whole = ppo.InferCritic(batch.states);

		FusedLayer::SetEnabled(true);
		ppo.numLearnThreads = 2;
		ppo.miniBatchSize = 512;
		diff = MaxRelDiff(ppo.InferCriticInParts(batch.states), whole);
		Check(diff < 1e-4, "value predictions in parts on 2 threads vs all at once: differ by " + Sci(diff));
	}

	{ // Actions chosen from probabilities that several threads computed
		RG_NO_GRAD;
		FusedLayer::SetEnabled(true);
		ppo.numInferThreads = 3;
		torch::Tensor actions, logProbs, values;
		ppo.InferActions(batch.states, batch.actionMasks, &actions, &logProbs, nullptr, &values);
		Check(MaxRelDiff(values, ppo.InferCriticInParts(batch.states)) < 1e-4,
			"critic predictions from the joint rollout pass match separate threaded inference");

		auto probs = PPOLearner::InferPolicyProbsFromModels(ppo.models, batch.states, batch.actionMasks, 1, false);
		auto expectedLogProbs = probs.log().gather(-1, actions.unsqueeze(-1)).flatten();
		double logProbDiff = (logProbs - expectedLogProbs).abs().max().item<double>();
		bool allValid = batch.actionMasks.to(torch::kBool).gather(-1, actions.unsqueeze(-1)).all().item<bool>();
		Check(
			actions.size(0) == BATCH_SIZE && allValid && logProbDiff < 1e-4,
			"actions inferred on 3 threads: all allowed, their probabilities differ from one thread by " + Sci(logProbDiff)
		);
		ppo.numInferThreads = 1;
	}

	{ // A full learn call
		ppo.numLearnThreads = 2;
		ppo.miniBatchSize = 512;
		auto before = ppo.models["shared_head"]->CopyParams();

		ExperienceBuffer experience = ExperienceBuffer(123, torch::kCPU);
		experience.data = batch;

		Report report = {};
		ppo.Learn(experience, report, false);

		auto after = ppo.models["shared_head"]->CopyParams();
		bool finite = after.isfinite().all().item<bool>();
		double change = (after - before).abs().sum().item<double>();
		Check(finite && change > 0, "a full Learn() call changes the shared layers and the slot encoder, all values finite");
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////

// The 1v1-only version of a model (SlotEncoderConfig::singleEncoding) takes over the saved files of the full model
static void TestSingleEncoding() {
	std::cout << "\nThe 1v1-only version of a model (one encoding instead of maximum and mean per group):" << std::endl;

	PartialModelConfig partialConfig = {};
	partialConfig.layerSizes = { 192, 128 };
	partialConfig.addOutputLayer = false;
	partialConfig.slots = MakeSlotConfig({ 96, 64 });

	ModelConfig config = partialConfig;
	config.numInputs = Obs::OBS_SIZE;
	config.numOutputs = 0;

	ModelConfig singleConfig = config;
	singleConfig.slots.singleEncoding = true;

	auto device = torch::Device(torch::kCPU);
	Model full = Model("shared_head", config, device);
	{
		RG_NO_GRAD;
		for (auto& param : full.parameters())
			if (param.dim() == 1)
				param.add_(torch::randn_like(param) * 0.3f);
	}

	// 1v1 inputs
	constexpr int NUM_SAMPLES = 96;
	std::vector<float> obsData = {};
	Obs obs = {};
	for (int i = 0; i < NUM_SAMPLES; i++) {
		GameState state = MakeRandomState(1);
		obsData += obs.BuildObs(state.players[g_Rand() % 2], state);
	}
	torch::Tensor input = torch::tensor(obsData).view({ NUM_SAMPLES, Obs::OBS_SIZE });
	torch::Tensor target = torch::randn({ NUM_SAMPLES, 128 });

	auto fnTrainStep = [&](Model& model) {
		(model.Forward(input, false) - target).square().mean().backward();
		model.StepOptim();
	};

	// A few training steps, so that the optimizer has a state that needs converting as well
	full.SetOptimLR(1e-3f);
	for (int i = 0; i < 3; i++)
		fnTrainStep(full);

	auto folder = std::filesystem::temp_directory_path() / "ggl_single_encoding_test";
	std::filesystem::create_directories(folder);
	full.Save(folder);

	Model single = Model("shared_head", singleConfig, device);
	single.SetOptimLR(1e-3f);
	single.Load(folder, false, true);

	auto fnFirstLayerInputs = [](Model& model) {
		return model.seq[0]->as<torch::nn::Linear>()->weight.size(1);
	};
	int numGlobal = Obs::GLOBAL_SIZE;
	Check(fnFirstLayerInputs(full) == numGlobal + 4 * 64 && fnFirstLayerInputs(single) == numGlobal + 64,
		RS_STR("its first layer has " << fnFirstLayerInputs(single) << " inputs instead of " << fnFirstLayerInputs(full)));

	{
		RG_NO_GRAD;
		double diff = MaxRelDiff(single.Forward(input, false), full.Forward(input, false));
		Check(diff < 1e-5, RS_STR("loaded from the files of the full model, it gives the same output for 1v1 inputs (relative difference " << Sci(diff) << ")"));
	}

	{ // Round trip through its own files
		auto folder2 = std::filesystem::temp_directory_path() / "ggl_single_encoding_test2";
		std::filesystem::create_directories(folder2);
		single.Save(folder2);

		Model reloaded = Model("shared_head", singleConfig, device);
		reloaded.Load(folder2, false, true);

		{
			RG_NO_GRAD;
			double diff = MaxRelDiff(reloaded.Forward(input, false), single.Forward(input, false));
			Check(diff < 1e-6, RS_STR("saved and loaded again, the output is the same (relative difference " << Sci(diff) << ")"));
		}

		// ... and back: the full model loads the files of the 1v1-only version
		Model expanded = Model("shared_head", config, device);
		expanded.SetOptimLR(1e-3f);
		expanded.Load(folder2, false, true);
		{
			RG_NO_GRAD;
			double diff = MaxRelDiff(expanded.Forward(input, false), single.Forward(input, false));
			Check(fnFirstLayerInputs(expanded) == fnFirstLayerInputs(full) && diff < 1e-5,
				RS_STR("the full model loads the files of the 1v1-only version, with the same output for 1v1 inputs (relative difference " << Sci(diff) << ")"));

			// The teammates' columns are zeros, so a teammate does not change the output yet
			auto weight = expanded.seq[0]->as<torch::nn::Linear>()->weight;
			double teammateWeights = weight.narrow(1, numGlobal, 2 * 64).abs().max().item<double>();
			double opponentWeights = weight.narrow(1, numGlobal + 2 * 64, 2 * 64).abs().max().item<double>();
			Check(teammateWeights == 0 && opponentWeights > 0, "its weights for the teammates start at zero");
		}
		{
			fnTrainStep(expanded);
			RG_NO_GRAD;
			bool finite = true;
			for (auto& param : expanded.parameters())
				finite &= param.isfinite().all().item<bool>();
			Check(finite, "a training step of the converted full model gives finite weights");
		}

		{ // The input of a game mode that was never played gets the weights of the 1v1 input
			RG_NO_GRAD;
			int modeInput = Obs::NUM_PAD_INPUTS;
			torch::Tensor input2v2 = input.clone();
			input2v2.select(1, modeInput).fill_(0);
			input2v2.select(1, modeInput + 1).fill_(1);

			torch::Tensor outBefore = expanded.Forward(input, false).clone();
			double diffBefore = MaxRelDiff(expanded.Forward(input2v2, false), outBefore);
			expanded.CopyInputWeights(modeInput, modeInput + 1);
			double diffAfter = MaxRelDiff(expanded.Forward(input2v2, false), outBefore);
			double diffSame = MaxRelDiff(expanded.Forward(input, false), outBefore);
			Check(diffBefore > 1e-3 && diffAfter < 1e-5 && diffSame < 1e-6,
				RS_STR("with the weights of the 1v1 input copied to the 2v2 input, the same situation gives the same output in both modes (relative difference " <<
					Sci(diffAfter) << ", before the copy " << Sci(diffBefore) << ")"));
		}
		std::filesystem::remove_all(folder2);
	}

	{ // The optimizer's state came along: one more training step changes all other weights exactly as in the full model
		fnTrainStep(full);
		fnTrainStep(single);

		RG_NO_GRAD;
		double diff = 0;
		auto fullSlotParams = full.slotSeq->parameters(), singleSlotParams = single.slotSeq->parameters();
		for (size_t i = 0; i < fullSlotParams.size(); i++)
			diff = RS_MAX(diff, MaxRelDiff(singleSlotParams[i], fullSlotParams[i]));

		auto fullParams = full.seq->parameters(), singleParams = single.seq->parameters();
		for (size_t i = 1; i < fullParams.size(); i++)
			diff = RS_MAX(diff, MaxRelDiff(singleParams[i], fullParams[i]));

		// First layer: the columns of the global inputs
		diff = RS_MAX(diff, MaxRelDiff(singleParams[0].narrow(1, 0, numGlobal), fullParams[0].narrow(1, 0, numGlobal)));

		Check(diff < 1e-4, RS_STR("with the converted optimizer state, a training step moves the other weights as in the full model (relative difference " << Sci(diff) << ")"));
	}

	std::filesystem::remove_all(folder);
}

///////////////////////////////////////////////////////////////////////////////////////////////

static double MaxAbsDiff(const FList& a, const FList& b) {
	if (a.size() != b.size())
		return 1e9;
	double result = 0;
	for (size_t i = 0; i < a.size(); i++)
		result = RS_MAX(result, std::abs((double)a[i] - (double)b[i]));
	return result;
}

// Part of TestMirror(): pairs of games with teams of teamSize.
// wrongActions: the second game gets the actions without mirroring the steering (to show that the check notices a mistake)
struct MirrorGamesResult {
	int numGames = 0, numAgreeEarly = 0, numAgreeEnd = 0;
	double maxDiffStart = 0, minDiffOff = 1e9, medianDiffEnd = 0;
	int numMirrored[2] = {}, numObs = 0, numPadPickups = 0, numTouches = 0, numSideChanges = 0;
};

static MirrorGamesResult RunMirrorGames(int teamSize, bool wrongActions) {
	using namespace RLGC::CommonValues;
	constexpr int MAX_CARS = 4, NUM_TRIALS = 60, NUM_STEPS = 30, EARLY_STEP = 4, TICK_SKIP = 8;
	constexpr double AGREE_DIFF = 1e-3;
	int numCars = teamSize * 2;

	Arena* arenas[2];
	Car* cars[2][MAX_CARS];
	for (int k = 0; k < 2; k++) {
		arenas[k] = Arena::Create(GameMode::SOCCAR);
		for (int i = 0; i < numCars; i++)
			cars[k][i] = arenas[k]->AddCar((i % 2 == 0) ? Team::BLUE : Team::ORANGE);
	}

	DefaultAction actionParser = {};
	Obs obsOn = {}, obsOff = {};

	MirrorGamesResult result = {};
	std::vector<double> endDiffs = {};

	for (int trial = 0; trial < NUM_TRIALS; trial++) {
		// Both views take turns: the network sees its car on the right half (1) or on the left half (-1)
		obsOn.mirrorXTeam[0] = obsOn.mirrorXTeam[1] = ((trial / 2) % 2 == 0) ? 1 : -1;

		// The same kickoff in both games gives both the same internal state (jump and flip timers etc.)
		for (int k = 0; k < 2; k++)
			arenas[k]->ResetToRandomKickoff(1000 + trial);

		bool onGround = (trial % 2) == 1;
		bool onPad = (trial % 6) == 1;
		for (int i = 0; i < numCars; i++) {
			PhysState phys = {};
			if (onGround) {
				// Driving straight
				phys.pos = Vec(Rand(-3800, 3800), Rand(-4800, 4800), 17.01f);
				phys.rotMat = Angle(Rand(-3.1f, 3.1f), 0, 0).ToRotMat();
				phys.vel = phys.rotMat.forward * Rand(0, 1800);
				if (onPad && i == 0)
					phys.pos = Vec(BOOST_LOCATIONS[3].x, BOOST_LOCATIONS[3].y, 17.01f); // A big pad in the corner
			} else {
				phys.pos = Vec(Rand(-3500, 3500), Rand(-4500, 4500), Rand(300, 1500));
				phys.rotMat = Angle(Rand(-3.1f, 3.1f), Rand(-1.5f, 1.5f), Rand(-3.1f, 3.1f)).ToRotMat();
				phys.vel = RandVec(1500);
				phys.angVel = RandVec(4);
			}
			PhysState mirrored = MirrorPhysX(phys);

			CarState carState = cars[0][i]->GetState();
			carState.boost = (onPad && i == 0) ? 10 : Rand(0, 100);

			carState.pos = phys.pos; carState.rotMat = phys.rotMat; carState.vel = phys.vel; carState.angVel = phys.angVel;
			cars[0][i]->SetState(carState);
			carState.pos = mirrored.pos; carState.rotMat = mirrored.rotMat; carState.vel = mirrored.vel; carState.angVel = mirrored.angVel;
			cars[1][i]->SetState(carState);
		}

		{ // The ball: in half of the games right where the first car is heading, so that it gets hit
			PhysState phys = {};
			phys.pos = Vec(Rand(-3500, 3500), Rand(-4500, 4500), Rand(93, 1200));
			phys.vel = RandVec(1500);
			phys.angVel = RandVec(4);
			if (trial % 4 < 2) {
				CarState carState = cars[0][0]->GetState();
				phys.pos = carState.pos + carState.vel * 0.3f + carState.rotMat.forward * 250;
				phys.pos.z = RS_MAX(phys.pos.z, 93.f);
				phys.vel = Vec(0, 0, 0);
			}
			PhysState mirrored = MirrorPhysX(phys);

			BallState ballState = arenas[0]->ball->GetState();
			ballState.pos = phys.pos; ballState.vel = phys.vel; ballState.angVel = phys.angVel;
			arenas[0]->ball->SetState(ballState);
			ballState.pos = mirrored.pos; ballState.vel = mirrored.vel; ballState.angVel = mirrored.angVel;
			arenas[1]->ball->SetState(ballState);
		}

		GameState states[2] = { GameState(arenas[0]), GameState(arenas[1]) };
		// Arena stores cars in a pointer-hashed unordered_set: match by stable ID,
		// not allocator-dependent iteration order across two independent arenas.
		auto canonicalize = [](GameState& state) {
			std::sort(state.players.begin(), state.players.end(), [](const Player& a, const Player& b) { return a.carId < b.carId; });
			for (int i = 0; i < state.players.size(); i++) state.players[i].index = i;
		};
		canonicalize(states[0]);
		canonicalize(states[1]);
		std::vector<Action> actions[2] = { std::vector<Action>(numCars), std::vector<Action>(numCars) };
		bool startSide[MAX_CARS] = {}, sideChanged[MAX_CARS] = {};
		for (int i = 0; i < numCars; i++)
			startSide[i] = states[0].players[i].pos.x < 0;

		for (int step = 0; step <= NUM_STEPS; step++) {
			double diff = 0;
			for (int i = 0; i < numCars; i++) {
				const Player& playerA = states[0].players[i];
				const Player& playerB = states[1].players[i];

				diff = RS_MAX(diff, MaxAbsDiff(obsOn.BuildObs(playerA, states[0]), obsOn.BuildObs(playerB, states[1])));
				result.minDiffOff = RS_MIN(result.minDiffOff, MaxAbsDiff(obsOff.BuildObs(playerA, states[0]), obsOff.BuildObs(playerB, states[1])));

				result.numMirrored[0] += obsOn.IsMirroredX(playerA);
				result.numMirrored[1] += obsOn.IsMirroredX(playerB);
				result.numObs++;
				result.numTouches += playerA.ballTouchedStep;
				sideChanged[i] |= (playerA.pos.x < 0) != startSide[i];
			}
			if (step == 0)
				result.maxDiffStart = RS_MAX(result.maxDiffStart, diff);
			if (step == EARLY_STEP)
				result.numAgreeEarly += diff < AGREE_DIFF;
			if (step == NUM_STEPS) {
				result.numAgreeEnd += diff < AGREE_DIFF;
				endDiffs.push_back(diff);
				break;
			}

			// New random actions now and then for the first game, and the mirrored actions for the second
			for (int i = 0; i < numCars; i++) {
				if (step == 0 || (g_Rand() % 3) == 0) {
					auto mask = actionParser.GetActionMask(states[0].players[i], states[0]);
					int actionIdx;
					do {
						actionIdx = g_Rand() % mask.size();
					} while (!mask[actionIdx]);
					actions[0][i] = actionParser.ParseAction(actionIdx, states[0].players[i], states[0]);
				}
				actions[1][i] = actions[0][i];
				Obs::MirrorActionX(actions[1][i]);
				if (wrongActions) {
					actions[1][i].steer = actions[0][i].steer;
					actions[1][i].yaw = actions[0][i].yaw;
				}
			}

			for (int k = 0; k < 2; k++) {
				for (int i = 0; i < numCars; i++)
					cars[k][i]->controls = (CarControls)actions[k][i];
				arenas[k]->Step(TICK_SKIP);
				IList unused;
				std::vector<Action> nativeActions;
				for (auto car : arenas[k]->_cars) {
					for (int i = 0; i < numCars; i++) {
						if (cars[k][i]->id == car->id) nativeActions.push_back(actions[k][i]);
					}
				}
				states[k].UpdateFromArena(arenas[k], nativeActions, NULL);
				canonicalize(states[k]);
			}
		}

		for (bool padActive : states[0].boostPads)
			result.numPadPickups += !padActive;
		for (int i = 0; i < numCars; i++)
			result.numSideChanges += sideChanged[i];
	}

	std::sort(endDiffs.begin(), endDiffs.end());
	result.medianDiffEnd = endDiffs[endDiffs.size() / 2];
	result.numGames = NUM_TRIALS;

	for (int k = 0; k < 2; k++)
		delete arenas[k];
	return result;
}

// The mirrored view (ObsBuilder::mirrorXTeam, the option --mirror), checked against the simulation itself.
//
// Two games are simulated. The second one starts as the left-right mirror image of the first, and its cars get the
// mirrored actions. If "mirroring" is done right (positions, rotations, spin, boost pads, actions), the second game stays
// the mirror image of the first, and with the mirrored view on, a player gets the same observation in both games:
// in one of the two its car is on the left half of the field, and that one is mirrored.
// Any mistake (a wrong sign for the spin, the wrong pad, the wrong action element) makes the two differ.
static void TestMirror(std::filesystem::path meshesPath) {
	using namespace RLGC::CommonValues;
	std::cout << "\nMirrored view:" << std::endl;

	{ // Boost pads
		bool ok = true;
		int numOwnPartner = 0;
		for (int i = 0; i < BOOST_LOCATIONS_AMOUNT; i++) {
			int m = Obs::GetMirroredPadIndex(i);
			Vec a = BOOST_LOCATIONS[i], b = BOOST_LOCATIONS[m];
			ok &= (Obs::GetMirroredPadIndex(m) == i) && std::abs(a.x + b.x) < 1 && std::abs(a.y - b.y) < 3 && a.z == b.z;
			numOwnPartner += (m == i);
		}
		Check(ok && numOwnPartner == 6, "every boost pad has its partner on the other side (the 6 pads on the middle line are their own)");
	}

	if (!std::filesystem::exists(meshesPath / "soccar")) {
		std::cout << "NOTE: No collision meshes at " << meshesPath << ", so the check against the simulation is skipped." << std::endl;
		std::cout << "      Run this from the repository folder, or give the path of collision_meshes as the first argument." << std::endl;
		return;
	}
	RocketSim::Init(meshesPath, true);

	for (int teamSize = 1; teamSize <= 2; teamSize++) {
		MirrorGamesResult r = RunMirrorGames(teamSize, false);
		std::string mode = RS_STR(teamSize << "v" << teamSize);

		std::cout << "  " << mode << ": " << r.numGames << " pairs of games of 2 seconds with random actions (" <<
			r.numTouches << " ball touches, " << r.numPadPickups << " boost pads taken, " << r.numSideChanges << " cars changed the half of the field)" << std::endl;

		Check(r.numMirrored[0] + r.numMirrored[1] == r.numObs && r.numMirrored[0] > r.numObs / 4 && r.numMirrored[1] > r.numObs / 4,
			RS_STR(mode << ": in every pair of observations exactly one is mirrored (" << r.numMirrored[0] << " + " << r.numMirrored[1] << " of " << r.numObs << ")"));
		Check(r.maxDiffStart < 1e-5, RS_STR(mode << ": at the start, the mirrored view gives the same observation in the game and in its mirror image (max difference " << Sci(r.maxDiffStart) << ")"));
		Check(r.minDiffOff > 1e-3, RS_STR(mode << ": without the mirrored view the two always differ (smallest difference " << Sci(r.minDiffOff) << ")"));
		Check(r.numTouches >= 5 && r.numPadPickups >= 5 && r.numSideChanges >= 5, mode + ": the games contain ball touches, boost pickups and cars that change sides");

		// The simulation is not mirror-symmetric to the last digit: where cars land, jump off or collide, the two games
		// get slightly different spins, and from there they drift apart. So not every pair stays together for 2 seconds,
		// and fewer do with more cars. (With a mistake in the mirroring none do, see the last check.)
		// Four-car collisions amplify floating-point/order differences sooner than 1v1.
		// Keep exact initial-observation equality and the 2-second check above/below;
		// tolerate up to 20% early divergence only in 2v2, versus 10% in 1v1.
		int minEarly = r.numGames * (teamSize == 1 ? 9 : 8) / 10;
		Check(r.numAgreeEarly >= minEarly && r.numAgreeEnd >= r.numGames * 4 / 10,
			RS_STR(mode << ": with mirrored actions the observations stay the same: in " << r.numAgreeEarly << " of " << r.numGames << " games after 0.27 seconds, in " <<
				r.numAgreeEnd << " after 2 seconds (median difference " << Sci(r.medianDiffEnd) << ")"));
	}

	{ // The same check with a mistake built in (the steering is not mirrored) has to fail
		MirrorGamesResult r = RunMirrorGames(1, true);
		Check(r.numAgreeEnd <= r.numGames / 10,
			RS_STR("with the steering not mirrored (a deliberate mistake), only " << r.numAgreeEnd << " of " << r.numGames << " games stay the same for 2 seconds"));
	}
}

static void TestRenderBehavior() {
	auto logits = torch::tensor({{1.f, 3.f, 10.f}, {2.f, 1.f, -1.f}});
	auto masks = torch::tensor({{1, 1, 0}, {1, 1, 1}}, torch::kUInt8);
	auto probs = PPOLearner::PolicyProbsFromLogits(logits, masks, 1.f);
	bool deterministic = true;
	for (int i = 0; i < 50; i++) {
		torch::Tensor actions;
		PPOLearner::ChooseActions(probs, true, &actions, nullptr);
		deterministic &= torch::equal(actions, torch::tensor({1, 0}, torch::kInt64));
	}
	Check(deterministic, "render action selection is masked argmax, not multinomial sampling");
	auto sharpened = PPOLearner::PolicyProbsFromLogits(logits, masks, 0.4f);
	Check(torch::equal(probs.argmax(1), sharpened.argmax(1)),
		"positive output temperature does not change deterministic actions");
	torch::Tensor sampled;
	PPOLearner::ChooseActions(torch::full({100, 2}, 0.5f), false, &sampled, nullptr);
	Check(sampled.min().item<int64_t>() == 0 && sampled.max().item<int64_t>() == 1,
		"normal training action selection remains stochastic");

	auto obs = std::make_unique<MultiModeObs>(false);
	auto parser = std::make_unique<DefaultAction>();
	auto setter = std::make_unique<KickoffState>();
	auto terminal = std::make_unique<GoalScoreCondition>();
	EnvSetConfig cfg{};
	cfg.numArenas = 1;
	cfg.tickSkip = 8;
	cfg.actionDelay = 7;
	cfg.envCreateFn = [&](int) {
		EnvCreateResult result{};
		result.arena = Arena::Create(GameMode::SOCCAR);
		result.arena->AddCar(Team::BLUE);
		result.arena->AddCar(Team::ORANGE);
		result.obsBuilder = obs.get();
		result.actionParser = parser.get();
		result.stateSetter = setter.get();
		result.terminalConditions = {terminal.get()};
		return result;
	};
	EnvSet env(cfg);
	auto arena = env.arenas[0];
	env.SetMirrorX(-1, -1);
	for (int size : {3, 1, 2, 2, 3, 1}) {
		env.ResetSingleArenaWithTeamSize(size);
		bool valid = env.arenas.size() == 1 && env.arenas[0] == arena &&
			env.state.numPlayers == 2 * size && env.state.gameStates[0].players.size() == 2 * size &&
			env.state.arenaPlayerStartIdx == std::vector<int>{0} &&
			env.state.obs.size[0] == 2 * size && env.state.actionMasks.size[0] == 2 * size &&
			env.state.rewards.size() == 2 * size && env.state.terminals[0] == 0;
		Check(valid, RS_STR("single render arena resizes player buffers safely to " << size << "v" << size));
		env.StepFirstHalf(false);
		env.StepSecondHalf(IList(size * 2, 0), false);
	}
}

int main(int argc, char* argv[]) {
	torch::manual_seed(123);

	if (!FusedLayer::IsSupported())
		std::cout << "NOTE: Fused layers are not supported in this build, so they are not tested" << std::endl;

	TestObs();
	TestSlotEncoder();
	TestLearner();
	TestSingleEncoding();
	TestMirror((argc > 1) ? argv[1] : "collision_meshes");
	TestRenderBehavior();

	std::cout << std::endl << (g_NumFailed ? RS_STR(g_NumFailed << " CHECK(S) FAILED") : std::string("All checks passed")) << std::endl;
	return g_NumFailed ? EXIT_FAILURE : EXIT_SUCCESS;
}
