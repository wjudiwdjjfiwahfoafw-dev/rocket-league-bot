#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"

#ifdef __GNUG__
#include <cxxabi.h> // For demangling type names
#endif

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/reward_function.py
namespace RLGC {
	class Reward {
	private:
		std::string _cachedName = {};

	public:
		virtual void Reset(const GameState& initialState) {}

		virtual void PreStep(const GameState& state) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			throw std::runtime_error("GetReward() is unimplemented");
			return 0;
		}

		// Get all rewards for all players
		virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) {

			std::vector<float> rewards = std::vector<float>(state.players.size());
			for (int i = 0; i < state.players.size(); i++) {
				rewards[i] = GetReward(state.players[i], state, isFinal);
			}

			return rewards;
		}

		virtual std::string GetName() {

			if (!_cachedName.empty())
				return _cachedName;

			std::string rewardName = typeid(*this).name();

#ifdef __GNUG__
			// GCC/Clang give mangled type names (e.g. "N4RLGC9AirRewardE"), so demangle and simplify them:
			//	"RLGC::AirReward" -> "AirReward"
			//	"RLGC::PlayerDataEventReward<&RLGC::PlayerEventState::bump, false>" -> "PlayerDataEventReward<bump,false>"
			{
				int status = 0;
				char* demangled = abi::__cxa_demangle(rewardName.c_str(), NULL, NULL, &status);
				if (demangled) {
					if (status == 0)
						rewardName = demangled;
					free(demangled);
				}

				std::string simplified = {};
				std::string curIdent = {};
				for (size_t i = 0; i <= rewardName.size(); i++) {
					char c = (i < rewardName.size()) ? rewardName[i] : 0;
					if (isalnum((unsigned char)c) || c == '_') {
						curIdent += c;
						continue;
					}

					bool isQualifier = (c == ':' && (i + 1) < rewardName.size() && rewardName[i + 1] == ':');
					if (isQualifier) {
						i++; // Skip the namespace/class qualifier and its "::"
					} else {
						simplified += curIdent;
						if (c != 0 && c != ' ' && c != '&')
							simplified += c;
					}
					curIdent.clear();
				}

				_cachedName = simplified;
				return simplified;
			}
#endif

			// Trim the string to after cetain keys
			{
				constexpr const char* TRIM_KEYS[] = {
					"::", // Namespace separator
					" " // Any spaces
				};
				for (const char* key : TRIM_KEYS) {
					size_t idx = rewardName.rfind(key);
					if (idx == std::string::npos)
						continue;

					rewardName.erase(rewardName.begin(), rewardName.begin() + idx + strlen(key));
				}
			}

			_cachedName = rewardName;
			return rewardName;
		}

		virtual ~Reward() {};
	};

	struct WeightedReward {
		Reward* reward;
		float weight;

		WeightedReward(Reward* reward, float scale) : reward(reward), weight(scale) {}
		WeightedReward(Reward* reward, int scale) : reward(reward), weight(scale) {}
	};
}