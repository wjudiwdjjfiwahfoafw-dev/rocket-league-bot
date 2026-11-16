#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"
#include "../BasicTypes/Lists.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/obs_builders/obs_builder.py
namespace RLGC {
	class ObsBuilder {
	public:
		// Mirrored view, per team (index 0 = blue, 1 = orange): 0 = off, 1 = right, -1 = left.
		// Set through EnvSet::SetMirrorX() or InferUnit::SetMirrorX().
		//
		// The field, the cars and the rules are the same when left and right are swapped. With the mirrored view "right",
		// a player whose car is on the left half of the field (seen from its own goal) gets the whole game mirrored,
		// so that the network only ever sees its own car on the right half. It then has to learn every situation once
		// instead of twice. The action that the network picks for the mirrored game is mirrored back before it is applied.
		// "left" is the same with the halves swapped.
		//
		// A policy has to be used with the view it was trained with, so this is a property of a checkpoint.
		// Only obs builders that return true from SupportsMirrorX() implement it.
		int mirrorXTeam[2] = { 0, 0 };

		virtual bool SupportsMirrorX() const { return false; }

		// True if this player currently gets the mirrored game
		bool IsMirroredX(const Player& player) const {
			int side = mirrorXTeam[(int)player.team];
			if (side == 0)
				return false;

			// Orange players see the field rotated by 180 degrees, which swaps left and right as well
			float x = (player.team == Team::ORANGE) ? -player.pos.x : player.pos.x;

			// Mirrored if the car is on the other half than the one the network is to see it on
			return x * side < 0;
		}

		// Mirrors an action left-to-right, in place
		static void MirrorActionX(Action& action) {
			action.steer = -action.steer;
			action.yaw = -action.yaw;
			action.roll = -action.roll;
		}

		virtual void Reset(const GameState& initialState) {}

		// NOTE: May be called once during environment initialization to determine policy neuron size
		virtual FList BuildObs(const Player& player, const GameState& state) = 0;
	};
}