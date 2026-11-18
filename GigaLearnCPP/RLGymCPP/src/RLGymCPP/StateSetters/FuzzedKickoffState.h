#pragma once
#include "StateSetter.h"

namespace RLGC {
	// Like KickoffState, but very slightly randomizes the cars
	//	to prevent deterministic agents from repeating gameplay.
	class FuzzedKickoffState : public StateSetter {
		bool seeded = false;
		std::minstd_rand rng;
	public:
		// A private stream per rating arena: thread scheduling must not select its kickoffs.
		void SetSeed(uint32_t seed) { rng.seed(seed); seeded = true; }

		constexpr static float
			FUZZ_POS_RANGE = 0.1f;

		static_assert(
			(float)(3e4f + (FUZZ_POS_RANGE / 100.f)) != 3e4f,
			"FuzzedKickoffState::FUZZ_POS_RANGE range is too small to survive float rounding"
		);

		void ResetArena(Arena* arena) {
			arena->ResetToRandomKickoff(seeded ? (int)rng() : -1);

			for (auto& car : arena->_cars) {
				auto state = car->GetState();
				for (int i = 0; i < 3; i++)
					state.pos[i] += seeded ? std::uniform_real_distribution<float>(-FUZZ_POS_RANGE, FUZZ_POS_RANGE)(rng) :
						RocketSim::Math::RandFloat(-FUZZ_POS_RANGE, FUZZ_POS_RANGE);
				car->SetState(state);
			}
		}
	};
}