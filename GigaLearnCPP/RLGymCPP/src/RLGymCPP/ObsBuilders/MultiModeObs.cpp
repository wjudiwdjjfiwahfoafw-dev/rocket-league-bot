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
	const int groupSlots[2] = { NUM_TEAMMATE_SLOTS, NUM_OPPONENT_SLOTS };
	std::vector<FList> groups[2] = {};

	for (auto& otherPlayer : state.players) {
		if (otherPlayer.carId == player.carId)
			continue;

		bool isTeammate = otherPlayer.team == player.team;
		auto& group = groups[isTeammate ? 0 : 1];
		if (group.size() >= groupSlots[isTeammate ? 0 : 1]) {
			RG_ERR_CLOSE(
				"MultiModeObs: Too many " << (isTeammate ? "teammates" : "opponents") << ", the observation has slots for " <<
				NUM_TEAMMATE_SLOTS << " teammates and " << NUM_OPPONENT_SLOTS << " opponents (" << state.players.size() << " players are in this game)"
			);
		}

		auto phys = MirrorPhysX(InvertPhys(otherPlayer, inv), mirror);

		FList slot = {};
		slot.reserve(SLOT_SIZE);
		slot += 1.f;
		slot += (float)isTeammate;
		AddPlayerInputs(slot, otherPlayer, phys, ball);

		Vec relPos = phys.pos - selfPhys.pos;
		slot += relPos * POS_COEF;
		slot += (phys.vel - selfPhys.vel) * VEL_COEF;
		slot += selfPhys.rotMat.Dot(relPos) * POS_COEF;
		slot += ((ball.pos - phys.pos).Length() - selfBallDist) * POS_COEF;

		group.push_back(slot);
	}

	for (int i = 0; i < 2; i++) {
		auto& group = groups[i];
		group.resize(groupSlots[i], FList(SLOT_SIZE, 0));

		if (shuffleSlots)
			std::shuffle(group.begin(), group.end(), ::Math::GetRandEngine());

		for (auto& slot : group)
			obs += slot;
	}

	return obs;
}
