#include "MultiModeObs.h"
#include <RLGymCPP/Gamestates/StateUtil.h>
#include "../Math.h"

void RLGC::MultiModeObs::AddPlayerInputs(FList& obs, const Player& player, const PhysState& phys, const PhysState& ball) {
	obs += phys.pos * POS_COEF;
	obs += phys.rotMat.forward;
	obs += phys.rotMat.up;
	obs += phys.vel * VEL_COEF;
	obs += phys.angVel * ANG_VEL_COEF;
	obs += phys.rotMat.Dot(phys.angVel) * ANG_VEL_COEF;

	obs += phys.rotMat.Dot(ball.pos - phys.pos) * POS_COEF;
	obs += phys.rotMat.Dot(ball.vel - phys.vel) * VEL_COEF;

	obs += player.boost / 100;
	obs += (float)player.isOnGround;
	obs += (float)player.HasFlipOrJump();
	obs += (float)player.isDemoed;
	obs += (float)player.hasJumped; // Allows detecting flip resets
	obs += (float)player.isSupersonic;
}

int RLGC::MultiModeObs::GetMirroredPadIndex(int padIndex) {
	struct Table {
		int indices[CommonValues::BOOST_LOCATIONS_AMOUNT];

		Table() {
			for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
				Vec target = CommonValues::BOOST_LOCATIONS[i] * Vec(-1, 1, 1);

				// The nearest pad to the mirrored position (the game's pad positions are symmetric to within 2 units)
				int best = 0;
				for (int j = 1; j < CommonValues::BOOST_LOCATIONS_AMOUNT; j++)
					if (CommonValues::BOOST_LOCATIONS[j].DistSq(target) < CommonValues::BOOST_LOCATIONS[best].DistSq(target))
						best = j;
				indices[i] = best;
			}
		}
	};
	static const Table table = {};
	return table.indices[padIndex];
}

RLGC::FList RLGC::MultiModeObs::BuildObs(const Player& player, const GameState& state) {
	FList obs = {};
	obs.reserve(OBS_SIZE);

	bool inv = player.team == Team::ORANGE;

	// Mirrored view: everything below is built from the game with left and right swapped (see ObsBuilder::mirrorXTeam)
	bool mirror = IsMirroredX(player);

	auto ball = MirrorPhysX(InvertPhys(state.ball, inv), mirror);
	auto& pads = state.GetBoostPads(inv);
	auto& padTimers = state.GetBoostPadTimers(inv);

	for (int i = 0; i < NUM_PAD_INPUTS; i++) {
		int padIdx = mirror ? GetMirroredPadIndex(i) : i;
		obs += pads[padIdx] ? 1.f : (1.f / (1.f + padTimers[padIdx])); // Approaches 1 as the pad becomes available
	}

	{ // Game mode
		int teamSizes[2] = {};
		for (auto& otherPlayer : state.players)
			teamSizes[(int)otherPlayer.team]++;

		int modeIdx = GetModeIndex(teamSizes[0], teamSizes[1]);
		for (int i = 0; i < NUM_MODE_INPUTS; i++)
			obs += (float)(i == modeIdx);
	}

	obs += ball.pos * POS_COEF;
	obs += ball.vel * VEL_COEF;
	obs += ball.angVel * ANG_VEL_COEF;

	auto selfPhys = MirrorPhysX(InvertPhys(player, inv), mirror);
	AddPlayerInputs(obs, player, selfPhys, ball);
	{
		Action prevAction = player.prevAction;
		if (mirror)
			MirrorActionX(prevAction);
		for (int i = 0; i < NUM_PREV_ACTION_INPUTS; i++)
			obs += prevAction[i];
	}

	// Player slots
	float selfBallDist = (ball.pos - selfPhys.pos).Length();
	// Order pointers, not 42-float temporary vectors. This keeps BuildObs to one allocation.
	std::array<const Player*, NUM_SLOTS> ordered = {};
	int counts[2] = {};
	for (auto& other : state.players) {
		if (other.carId == player.carId)
			continue;
		int group = other.team == player.team ? 0 : 1;
		if (counts[group] >= (group ? NUM_OPPONENT_SLOTS : NUM_TEAMMATE_SLOTS))
			RG_ERR_CLOSE("MultiModeObs: too many players for the available slots");
		ordered[(group ? NUM_TEAMMATE_SLOTS : 0) + counts[group]++] = &other;
	}
	if (shuffleSlots) {
		auto middle = ordered.begin() + NUM_TEAMMATE_SLOTS;
		std::shuffle(ordered.begin(), middle, ::Math::GetRandEngine());
		std::shuffle(middle, ordered.end(), ::Math::GetRandEngine());
	}
	for (const Player* other : ordered) {
		if (!other) {
			obs.resize(obs.size() + SLOT_SIZE, 0.f);
			continue;
		}
		auto phys = MirrorPhysX(InvertPhys(*other, inv), mirror);
		obs += 1.f;
		obs += (float)(other->team == player.team);
		AddPlayerInputs(obs, *other, phys, ball);
		Vec relPos = phys.pos - selfPhys.pos;
		obs += relPos * POS_COEF;
		obs += (phys.vel - selfPhys.vel) * VEL_COEF;
		obs += selfPhys.rotMat.Dot(relPos) * POS_COEF;
		obs += ((ball.pos - phys.pos).Length() - selfBallDist) * POS_COEF;
	}

	return obs;
}
