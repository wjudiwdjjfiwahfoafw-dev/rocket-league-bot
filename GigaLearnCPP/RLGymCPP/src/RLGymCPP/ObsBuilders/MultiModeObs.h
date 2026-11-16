#pragma once

#include "ObsBuilder.h"

namespace RLGC {
	// An observation with one fixed size and layout for 1v1, 2v2 and 3v3, so that one network can play all three.
	//
	// Layout (294 inputs). Everything is seen from the observing player's side of the field
	//	(for orange players the field is rotated by 180 degrees, so the own goal is always at -Y):
	//
	//	Global inputs (84)
	//		[  0.. 33]  Boost pads: 1 if available, otherwise 1/(1 + seconds until it is back)
	//		[ 34.. 36]  Game mode, one-hot: 1v1, 2v2, 3v3 (by the size of the bigger team)
	//		[ 37.. 45]  Ball: position, velocity, angular velocity
	//		[ 46.. 75]  The observing player (see "Player inputs" below)
	//		[ 76.. 83]  The observing player's previous action (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
	//
	//	5 player slots of 42 inputs each: 2 for teammates, then 3 for opponents.
	//	Slots without a player are all zeros. Used slots are filled from the front of their group, in no meaningful order.
	//		[0]         1 = this slot is used
	//		[1]         1 = teammate, 0 = opponent
	//		[2..31]     Player inputs
	//		[32..34]    Position relative to the observing player
	//		[35..37]    Velocity relative to the observing player
	//		[38..40]    Position relative to the observing player, in the observing player's car frame (forward, right, up)
	//		[41]        Distance to the ball, minus the observing player's distance to the ball (negative = this player is closer)
	//
	//	Player inputs (30):
	//		position, forward direction, up direction, velocity, angular velocity,
	//		angular velocity / ball position / ball velocity in the car's own frame,
	//		boost, on ground, has a flip or jump left, is demolished, has jumped, is supersonic
	//
	// Every input can also be read from the game through RLBot.
	//
	// For networks: [GLOBAL_SIZE global inputs][NUM_SLOTS slots of SLOT_SIZE]. The last CONTEXT_SIZE global inputs (mode, ball,
	//	own car) describe the observing player's situation, which is what a slot has to be compared with.
	class MultiModeObs : public ObsBuilder {
	public:
		constexpr static float
			POS_COEF = 1 / 5000.f,
			VEL_COEF = 1 / 2300.f,
			ANG_VEL_COEF = 1 / 3.f;

		constexpr static int
			MAX_TEAM_SIZE = 3,
			NUM_TEAMMATE_SLOTS = MAX_TEAM_SIZE - 1,
			NUM_OPPONENT_SLOTS = MAX_TEAM_SIZE,
			NUM_SLOTS = NUM_TEAMMATE_SLOTS + NUM_OPPONENT_SLOTS,

			NUM_PAD_INPUTS = CommonValues::BOOST_LOCATIONS_AMOUNT,
			NUM_MODE_INPUTS = MAX_TEAM_SIZE,
			NUM_BALL_INPUTS = 9,
			NUM_PLAYER_INPUTS = 30,
			NUM_PREV_ACTION_INPUTS = (int)Action::ELEM_AMOUNT,

			GLOBAL_SIZE = NUM_PAD_INPUTS + NUM_MODE_INPUTS + NUM_BALL_INPUTS + NUM_PLAYER_INPUTS + NUM_PREV_ACTION_INPUTS,
			CONTEXT_SIZE = GLOBAL_SIZE - NUM_PAD_INPUTS,

			NUM_SLOT_EXTRA_INPUTS = 12,
			SLOT_SIZE = NUM_PLAYER_INPUTS + NUM_SLOT_EXTRA_INPUTS,

			OBS_SIZE = GLOBAL_SIZE + NUM_SLOTS * SLOT_SIZE;

		// Puts the players of each group into their slots in a random order (empty slots anywhere).
		// Only needed for networks that treat every slot differently (a plain MLP on the whole observation),
		//	so that they can't get used to a slot order. Networks that pool over the slots don't need it.
		bool shuffleSlots;

		MultiModeObs(bool shuffleSlots = false) : shuffleSlots(shuffleSlots) {}

		// Index of the game mode input that is set for a game with these team sizes (0 = 1v1, 1 = 2v2, 2 = 3v3)
		static int GetModeIndex(int teamSizeA, int teamSizeB) {
			return RS_CLAMP(RS_MAX(teamSizeA, teamSizeB), 1, MAX_TEAM_SIZE) - 1;
		}

		virtual FList BuildObs(const Player& player, const GameState& state) override;

		// The mirrored view (see ObsBuilder::mirrorXTeam) is implemented
		virtual bool SupportsMirrorX() const override { return true; }

		// Index of the boost pad on the other side of the field (x -> -x)
		static int GetMirroredPadIndex(int padIndex);

	protected:
		// Adds the NUM_PLAYER_INPUTS inputs of a player. "phys" and "ball" must already be from the observing player's side
		static void AddPlayerInputs(FList& obs, const Player& player, const PhysState& phys, const PhysState& ball);
	};
}
