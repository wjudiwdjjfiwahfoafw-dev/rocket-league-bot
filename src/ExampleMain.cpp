// GigaLearnCPP example bot: trains, evaluates and runs a Rocket League PPO agent.
//
// This is the original ExampleMain.cpp (same environment, rewards and actions), extended with:
//	 - bots for 1v1, 2v2 and 3v3 at once (--model s/m/l/xl), next to the original 1v1 example bot (--model example)
//	 - command-line options instead of hard-coded values (run with --help)
//	 - automatic device selection, so it also runs on machines without a CUDA GPU
//	 - optional time/timestep limits and a graceful Ctrl+C (the learner saves a checkpoint before stopping)
//	 - an evaluation mode (--eval) that plays a checkpoint against random actions or another checkpoint
//	 - reward weights and the no-touch timeout as options (--rewards, --no-touch-timeout), to compare training setups
//	 - Weights & Biases logging (--wandb)
//	 - GigaLearn's skill tracker and policy versions as options (--skill-tracker, --ts-per-version, --train-vs-old):
//	   the bot regularly plays against saved old versions of itself, which gives it an Elo rating
//	 - a rating mode (--rate-rounds) that rates a checkpoint with the skill tracker, without training it
//	 - episodes that start from random states with a moving ball (--random-state-prob, --random-ball-speed)
//	 - an RLBot mode (--rlbot) that runs a checkpoint in the real game through the RLBot framework

#include <GigaLearnCPP/Learner.h>
#include <GigaLearnCPP/Util/InferUnit.h>
#include <GigaLearnCPP/Util/ModelConvert.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/DefaultObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ObsBuilders/MultiModeObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include "RLBotClient.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <random>

using namespace GGL; // GigaLearn
using namespace RLGC; // RLGymCPP

///////////////////////////////////////////////////////////////////////////////////////////////
// Things that define the bot.
// The training, evaluation and RLBot modes all use these, so they always stay in sync.
// NOTE: If you change the obs builder, action parser or model sizes, old checkpoints will no longer load.
///////////////////////////////////////////////////////////////////////////////////////////////

constexpr int TICK_SKIP = 8;
constexpr int ACTION_DELAY = TICK_SKIP - 1; // Normal value in other RLGym frameworks

// Seconds without a ball touch before a training episode is cut off, 0 = never (set with --no-touch-timeout)
static float g_NoTouchTimeout = 10;

// For kickoffs: while the ball still lies at rest on the centre spot, +1 per step for a player who is closer to it
// than every opponent, -1 for one who is further away (so it sums to zero over both players of a 1v1).
// Self-play alone can drift to kickoffs that arrive late, because the copy on the other side does the same and nobody
// is punished for it. Older versions that rush the ball then win the kickoffs (observed in earlier kickoff experiments).
class KickoffProximityReward : public Reward {
public:
	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
		if (state.ball.pos.x != 0 || state.ball.pos.y != 0 || state.ball.vel.LengthSq() != 0)
			return 0; // Not a kickoff, or the ball has been touched

		// The nearest player of each team. The whole team gets the reward, so that one player going for the
		// kickoff is enough (with a reward per player, both players of a 2v2 team would be paid to race for the ball)
		float ownDist = -1, oppDist = -1;
		for (auto& other : state.players) {
			float dist = (other.pos - state.ball.pos).Length();
			float& teamDist = (other.team == player.team) ? ownDist : oppDist;
			if (teamDist < 0 || dist < teamDist)
				teamDist = dist;
		}
		if (oppDist < 0)
			return 0;
		return (ownDist < oppDist) ? 1.f : ((ownDist > oppDist) ? -1.f : 0.f);
	}
};

// Rewards a player for staying on the defensive side of the ball (between own goal and the ball).
// Encourages proper rotation and prevents overcommitting ahead of the play.
class BehindBallReward : public Reward {
public:
	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
		float sign = (player.team == Team::BLUE) ? 1.f : -1.f;
		// Positive when player is closer to own goal than the ball is
		float diff = (state.ball.pos.y - player.pos.y) * sign;
		return std::clamp(diff / 1000.f, -0.5f, 1.f);
	}
};

// Penalizes crowding teammates in 2v2 and 3v3 so players spread out for passes and defense.
// Returns 0 in 1v1 (no teammates).
class TeamSpacingReward : public Reward {
public:
	static constexpr float MIN_DIST = 1500.f;
	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
		float minTeammateDist = 1e9f;
		bool hasTeammate = false;
		for (auto& other : state.players) {
			if (other.team == player.team && other.carId != player.carId && !other.isDemoed) {
				hasTeammate = true;
				float d = (other.pos - player.pos).Length();
				if (d < minTeammateDist)
					minTeammateDist = d;
			}
		}
		if (!hasTeammate || minTeammateDist >= MIN_DIST)
			return 0.f;
		return -(1.f - minTeammateDist / MIN_DIST);
	}
};

// VelocityPlayerToBallReward for the player of each team that is nearest to the ball, nothing for its teammates.
// In 1v1 it is the same reward. With teammates, the reward per player pays all of them to drive at the ball at once,
// this one leaves the others free to take a position.
class NearestVelocityToBallReward : public Reward {
public:
	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
		float ownDistSq = (state.ball.pos - player.pos).LengthSq();
		for (auto& other : state.players)
			if (other.team == player.team && other.carId != player.carId && (state.ball.pos - other.pos).LengthSq() < ownDistSq)
				return 0;

		Vec dirToBall = (state.ball.pos - player.pos).Normalized();
		Vec normVel = player.vel / CommonValues::CAR_MAX_SPEED;
		return dirToBall.Dot(normVel);
	}
};

// VelocityPlayerToBallReward of the team's nearest player to the ball, given to every player of the team.
// In 1v1 it is the same reward. With teammates, nobody gains by being the nearest one: the team is paid once,
// whoever goes for the ball. (NearestVelocityToBallReward pays the nearest player only, which makes teammates race each other.)
class TeamVelocityToBallReward : public Reward {
public:
	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
		const Player* nearest = &player;
		float nearestDistSq = (state.ball.pos - player.pos).LengthSq();
		for (auto& other : state.players) {
			if (other.team != player.team)
				continue;
			float distSq = (state.ball.pos - other.pos).LengthSq();
			if (distSq < nearestDistSq) {
				nearestDistSq = distSq;
				nearest = &other;
			}
		}

		Vec dirToBall = (state.ball.pos - nearest->pos).Normalized();
		Vec normVel = nearest->vel / CommonValues::CAR_MAX_SPEED;
		return dirToBall.Dot(normVel);
	}
};

static inline bool IsUntouchedKickoffBall(const BallState& ball) {
	return fabsf(ball.pos.x) < 1.f && fabsf(ball.pos.y) < 1.f && ball.vel.LengthSq() < 1.f;
}

// Exact conservative potential-derivative ball velocity reward toward the opponent goal mouth.
// Projects onto the convex scoring opening [-760, 760] x {goalY} x [110, 540] (rewarding corner
// placement and centering passes, not just dead-center hits) with smooth urgency amplification inside
// 2800 uu of both nets. Any closed ball loop integrates to zero, preventing backboard/corner farming.
class ObjectiveBallVelReward : public Reward {
public:
	static constexpr float GOAL_HALF_X   = 760.f;
	static constexpr float GOAL_MIN_Z    = 110.f;
	static constexpr float GOAL_MAX_Z    = 540.f;
	static constexpr float GOAL_TARGET_Y = CommonValues::BACK_WALL_Y + CommonValues::BALL_RADIUS;
	static constexpr float DANGER_RADIUS = 2800.f;
	static constexpr float DANGER_BONUS  = 0.85f;

	static float GoalFlow(const BallState& ball, bool orangeGoal) {
		Vec closest(
			std::clamp(ball.pos.x, -GOAL_HALF_X, GOAL_HALF_X),
			orangeGoal ? GOAL_TARGET_Y : -GOAL_TARGET_Y,
			std::clamp(ball.pos.z, GOAL_MIN_Z, GOAL_MAX_Z)
		);
		Vec toGoal = closest - ball.pos;
		float dist = toGoal.Length();
		if (dist < 1e-3f)
			return 0.f;
		float urgency = 1.f;
		if (dist < DANGER_RADIUS)
			urgency += DANGER_BONUS * (1.f - dist / DANGER_RADIUS);
		return urgency * (toGoal * (1.f / dist)).Dot(ball.vel) * (1.f / CommonValues::BALL_MAX_SPEED);
	}

	virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
		// Exact zero-sum across Blue and Orange; computed once per arena step.
		float blueVal = GoalFlow(state.ball, true) - GoalFlow(state.ball, false);
		std::vector<float> out(state.players.size());
		for (size_t i = 0; i < state.players.size(); i++)
			out[i] = (state.players[i].team == Team::BLUE) ? blueVal : -blueVal;
		return out;
	}

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		float blueVal = GoalFlow(state.ball, true) - GoalFlow(state.ball, false);
		return (player.team == Team::BLUE) ? blueVal : -blueVal;
	}
};

// Goal-directed touch impulse reward: rewards strong ball acceleration toward the opponent goal mouth
// (on attack/midfield) or safely radially away from own goal (deep on defense), boosts productive aerial
// touches, and penalizes hard touches driven toward own net. Used inside ZeroSumReward so winning 50/50s
// is directly rewarded while own-zone panic hits are penalized.
class DirectionalTouchReward : public Reward {
public:
	static constexpr float MIN_DELTA_VEL  = 220.f;
	static constexpr float NORM_DELTA_VEL = RLGC::Math::KPHToVel(105);

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep || !state.prev)
			return 0.f;

		Vec deltaVel = state.ball.vel - state.prev->ball.vel;
		if (deltaVel.LengthSq() < MIN_DELTA_VEL * MIN_DELTA_VEL)
			return 0.f;

		float sign = (player.team == Team::BLUE) ? 1.f : -1.f;
		float goalLineY = CommonValues::BACK_WALL_Y + CommonValues::BALL_RADIUS;
		Vec oppGoal(
			std::clamp(state.ball.pos.x, -760.f, 760.f),
			sign * goalLineY,
			std::clamp(state.ball.pos.z, 110.f, 540.f)
		);
		Vec ownGoal(
			std::clamp(state.ball.pos.x, -850.f, 850.f),
			-sign * goalLineY,
			std::clamp(state.ball.pos.z, 93.f, 600.f)
		);

		Vec toOpp = oppGoal - state.ball.pos;
		float dOpp = toOpp.Length();
		Vec dirAttack = (dOpp > 1e-3f) ? (toOpp * (1.f / dOpp)) : Vec(0.f, sign, 0.f);

		Vec fromOwn = state.ball.pos - ownGoal;
		float dOwn = fromOwn.Length();
		Vec dirClear = (dOwn > 1e-3f) ? (fromOwn * (1.f / dOwn)) : Vec(0.f, sign, 0.f);

		// Near own goal, blend in radial escape so safe corner/upfield clears are rewarded and
		// centering the ball across own goal crease is discouraged.
		float clearBlend = std::clamp(1.f - dOwn / 2600.f, 0.f, 0.7f);
		float directed = (1.f - clearBlend) * deltaVel.Dot(dirAttack) + clearBlend * deltaVel.Dot(dirClear);
		float score = std::clamp(directed / NORM_DELTA_VEL, -0.6f, 1.1f);

		// Reward productive aerial touches only in team modes (2v2/3v3) where a teammate covers recovery;
		// in 1v1 risky aerial commits without scoring leave an open net.
		if (score > 0.f && state.players.size() > 2 && state.ball.pos.z > 300.f) {
			float airBonus = 1.f + 0.35f * std::clamp((state.ball.pos.z - 300.f) / 700.f, 0.f, 1.f);
			score *= airBonus;
		}
		return score;
	}
};

// Zero-sum bonus on goals proportional to scoring shot speed (fast shots beat defenders reliably).
class GoalSpeedReward : public Reward {
public:
	static constexpr float REF_SPEED = RLGC::Math::KPHToVel(120);

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (!state.goalScored)
			return 0.f;
		float frac = std::clamp(state.ball.vel.Length() / REF_SPEED, 0.f, 1.2f);
		bool scored = (player.team != RS_TEAM_FROM_Y(state.ball.pos.y));
		return scored ? frac : -frac;
	}
};

// Direct finishing & playmaking credit on goals in team modes (2v2/3v3: scorer +1.0, assist +0.75).
// Returns 0 in 1v1 where GoalReward already provides exact individual credit without team dilution.
class ScorerAssistReward : public Reward {
public:
	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (state.players.size() <= 2)
			return 0.f;
		if (player.eventState.goal)
			return 1.f;
		if (player.eventState.assist)
			return 0.75f;
		return 0.f;
	}
};

// Credits ballistic on-target shots and setup passes tracked by RocketSim's GameEventTracker (1s cooldown).
class ShotAndPassReward : public Reward {
public:
	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (player.eventState.shot)
			return 1.f;
		if (player.eventState.shotPass)
			return 0.6f;
		return 0.f;
	}
};

// Credits defensive saves: official event saves and genuine close-range goal-bound shot denials,
// guarded by both a post-kickoff grace window (preventing fake-kickoff save farming) and a per-team cooldown.
class ClutchSaveReward : public Reward {
	float cooldown[2] = { 0.f, 0.f };
	float sinceKickoff = 10.f;
public:
	static constexpr float SAVE_COOLDOWN    = 2.0f;
	static constexpr float KICKOFF_NO_SAVE  = 2.5f;

	virtual void Reset(const GameState& initialState) override {
		cooldown[0] = cooldown[1] = 0.f;
		sinceKickoff = IsUntouchedKickoffBall(initialState.ball) ? 0.f : 10.f;
	}

	virtual void PreStep(const GameState& state) override {
		for (int t = 0; t < 2; t++)
			cooldown[t] = std::max(0.f, cooldown[t] - state.deltaTime);
		if (IsUntouchedKickoffBall(state.ball))
			sinceKickoff = 0.f;
		else
			sinceKickoff += state.deltaTime;
	}

	static bool BallHeadingIntoGoal(const BallState& b, Team goalTeam) {
		float sign = (goalTeam == Team::ORANGE) ? 1.f : -1.f;
		float vyToward = b.vel.y * sign;
		if (vyToward < 1000.f)
			return false;
		float distY = (sign * CommonValues::BACK_WALL_Y - b.pos.y) * sign;
		if (distY <= 0.f || distY > 2600.f)
			return false;
		float t = distY / vyToward;
		if (t > 1.5f)
			return false;
		float xAt = b.pos.x + b.vel.x * t;
		float zAt = b.pos.z + b.vel.z * t + 0.5f * CommonValues::GRAVITY_Z * t * t;
		return fabsf(xAt) <= CommonValues::GOAL_WIDTH_FROM_CENTER &&
		       zAt <= CommonValues::GOAL_HEIGHT;
	}

	virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
		std::vector<float> out(state.players.size(), 0.f);
		if (state.goalScored || sinceKickoff < KICKOFF_NO_SAVE)
			return out;
		for (size_t i = 0; i < state.players.size(); i++) {
			const auto& p = state.players[i];
			int t = (int)p.team;
			if (cooldown[t] > 0.f)
				continue;
			float val = 0.f;
			if (p.eventState.save) {
				val = 1.f;
			} else if (p.ballTouchedStep && state.prev &&
			           BallHeadingIntoGoal(state.prev->ball, p.team) &&
			           !BallHeadingIntoGoal(state.ball, p.team)) {
				val = std::clamp(fabsf(state.prev->ball.vel.y) / 2000.f, 0.5f, 1.f);
			}
			if (val > 0.f) {
				out[i] = val;
				cooldown[t] = SAVE_COOLDOWN;
			}
		}
		return out;
	}

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		return (sinceKickoff >= KICKOFF_NO_SAVE && player.eventState.save) ? 1.f : 0.f;
	}
};

// Fake-kickoff-proof exact zero-sum kickoff reward:
// 1) While the kickoff ball sits untouched on the center spot, compares closing velocity of each team's
//    closest kickoff car(s) (leaving back teammates free to cheat up, guard net, or grab boost).
// 2) On the step the kickoff ball is first touched, awards a crisp zero-sum bonus for winning the
//    first touch (0 if both teams make simultaneous 50/50 contact) plus directional kickoff outcome
//    into the opponent's half — preventing the braking-before-contact fake-kickoff exploit.
class KickoffSpeedReward : public Reward {
	bool kickoffPending = false;
public:
	static constexpr float FIRST_TOUCH_BONUS = 6.0f;
	static constexpr float OUTCOME_VEL_BONUS = 4.0f;

	virtual void Reset(const GameState& initialState) override {
		kickoffPending = IsUntouchedKickoffBall(initialState.ball);
	}

	virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
		std::vector<float> out(state.players.size(), 0.f);
		if (IsUntouchedKickoffBall(state.ball)) {
			kickoffPending = true;

			float minDist[2] = { 1e9f, 1e9f };
			for (const auto& p : state.players) {
				if (p.isDemoed)
					continue;
				float d = (state.ball.pos - p.pos).Length();
				int t = (int)p.team;
				if (d < minDist[t])
					minDist[t] = d;
			}

			float bestClosing[2] = { -1.f, -1.f };
			bool hasCar[2] = { false, false };
			for (const auto& p : state.players) {
				if (p.isDemoed)
					continue;
				int t = (int)p.team;
				Vec toBall = state.ball.pos - p.pos;
				float d = toBall.Length();
				if (d > minDist[t] + 150.f || d < 1e-3f)
					continue;
				float c = (toBall * (1.f / d)).Dot(p.vel) * (1.f / CommonValues::CAR_MAX_SPEED);
				if (!hasCar[t] || c > bestClosing[t]) {
					bestClosing[t] = c;
					hasCar[t] = true;
				}
			}
			if (!hasCar[0] || !hasCar[1])
				return out;

			float blueAdv = bestClosing[0] - bestClosing[1];
			for (size_t i = 0; i < state.players.size(); i++)
				out[i] = (state.players[i].team == Team::BLUE) ? blueAdv : -blueAdv;
			return out;
		}

		if (kickoffPending) {
			kickoffPending = false;
			bool touched[2] = { false, false };
			for (const auto& p : state.players)
				if (p.ballTouchedStep)
					touched[(int)p.team] = true;

			float blueAdv = 0.f;
			if (touched[0] != touched[1])
				blueAdv += touched[0] ? FIRST_TOUCH_BONUS : -FIRST_TOUCH_BONUS;
			// Also reward winning the initial 50/50 velocity toward the opponent half.
			blueAdv += OUTCOME_VEL_BONUS * std::clamp(state.ball.vel.y / 2200.f, -1.f, 1.f);
			for (size_t i = 0; i < state.players.size(); i++)
				out[i] = (state.players[i].team == Team::BLUE) ? blueAdv : -blueAdv;
		}
		return out;
	}

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		return GetAllRewards(state, isFinal)[player.index];
	}
};

// Exact zero-sum first-man challenge pressure in team modes (2v2/3v3): compares non-negative closing
// speed of each team's nearest alive player toward the ball during live play. Cures passive 2v2/3v3
// play without inducing reckless unbacked dives in 1v1 (returns 0 in 1v1) or punishing shadow defense.
class TeamPressureReward : public Reward {
public:
	virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
		std::vector<float> out(state.players.size(), 0.f);
		if (state.players.size() <= 2 || IsUntouchedKickoffBall(state.ball))
			return out;

		float bestDistSq[2] = { 1e18f, 1e18f };
		float press[2] = { 0.f, 0.f };
		for (const auto& p : state.players) {
			if (p.isDemoed)
				continue;
			int t = (int)p.team;
			Vec toBall = state.ball.pos - p.pos;
			float dSq = toBall.LengthSq();
			if (dSq < bestDistSq[t] && dSq > 1.f) {
				bestDistSq[t] = dSq;
				float closing = (toBall * (1.f / sqrtf(dSq))).Dot(p.vel) * (1.f / CommonValues::CAR_MAX_SPEED);
				press[t] = std::max(0.f, closing);
			}
		}

		float blueAdv = press[0] - press[1];
		for (size_t i = 0; i < state.players.size(); i++)
			out[i] = (state.players[i].team == Team::BLUE) ? blueAdv : -blueAdv;
		return out;
	}

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		return GetAllRewards(state, isFinal)[player.index];
	}
};

// Exact zero-sum 1v1 discipline & shadow-positioning reward (0 in 2v2/3v3): rewards keeping own car
// goal-side when the ball heads toward own net and staying grounded when far from aerial balls so the
// bot never leaves an open net in 1v1 from aimless high-air commits.
class SoloDisciplineReward : public Reward {
public:
	virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
		std::vector<float> out(state.players.size(), 0.f);
		if (state.players.size() != 2 || IsUntouchedKickoffBall(state.ball))
			return out;

		float score[2] = { 0.f, 0.f };
		Vec ball2D(state.ball.pos.x, state.ball.pos.y, 0.f);
		for (const auto& p : state.players) {
			if (p.isDemoed) {
				score[(int)p.team] = -0.5f;
				continue;
			}
			float sign = (p.team == Team::BLUE) ? 1.f : -1.f;
			float s = 0.f;
			// Penalize aimless high-air float when not near an aerial ball (major cause of 1v1 open nets).
			if (!p.isOnGround && p.pos.z > 240.f && !p.ballTouchedStep) {
				float ballDist = (state.ball.pos - p.pos).Length();
				if (ballDist > 550.f || state.ball.pos.z < 180.f)
					s -= 0.45f * std::clamp((p.pos.z - 240.f) / 500.f, 0.25f, 1.f);
			}
			// Reward goal-side recovery when the ball is moving toward own goal.
			float vyOwn = -state.ball.vel.y * sign;
			if (vyOwn > 350.f) {
				Vec ownGoal(0.f, -sign * CommonValues::BACK_WALL_Y, 0.f);
				float margin = (ball2D - ownGoal).Length() - (Vec(p.pos.x, p.pos.y, 0.f) - ownGoal).Length();
				float urgency = std::clamp(vyOwn / 2200.f, 0.15f, 1.f);
				s += 0.55f * urgency * std::clamp(margin / 1400.f, -1.f, 0.5f);
			}
			score[(int)p.team] = s;
		}
		float blueAdv = score[0] - score[1];
		for (size_t i = 0; i < state.players.size(); i++)
			out[i] = (state.players[i].team == Team::BLUE) ? blueAdv : -blueAdv;
		return out;
	}

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		return GetAllRewards(state, isFinal)[player.index];
	}
};

// Exact zero-sum 2v2/3v3 last-man rotation reward. Checks whether each team keeps at least one alive
// defender closer (in 2D Euclidean distance) to own goal than the ball is, scaled by threat level.
// Leaves 1st/2nd man free to attack, pass and challenge while eliminating full-team overcommits.
// Returns 0 in 1v1.
class DefensiveCoverReward : public Reward {
public:
	virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
		std::vector<float> out(state.players.size(), 0.f);
		if (state.players.size() <= 2)
			return out;

		float cover[2] = { 0.f, 0.f };
		Vec ball2D(state.ball.pos.x, state.ball.pos.y, 0.f);
		for (int t = 0; t < 2; t++) {
			float sign = (t == 0) ? 1.f : -1.f;
			float ballY = state.ball.pos.y * sign;
			float ballVY = state.ball.vel.y * sign;
			float posThreat = std::clamp((1500.f - ballY) / 5500.f, 0.15f, 1.f);
			float velThreat = std::clamp(-ballVY / 3500.f, 0.f, 0.5f);
			float threat = std::min(1.f, posThreat + velThreat);

			Vec ownGoal(0.f, -sign * CommonValues::BACK_WALL_Y, 0.f);
			float ballDistGoal = (ball2D - ownGoal).Length();

			float bestMargin = -1.f;
			bool anyAlive = false;
			for (const auto& p : state.players) {
				if ((int)p.team != t || p.isDemoed)
					continue;
				anyAlive = true;
				float carDistGoal = (Vec(p.pos.x, p.pos.y, 0.f) - ownGoal).Length();
				float s = std::clamp((ballDistGoal - carDistGoal) / 1200.f, -1.f, 0.6f);
				if (s > bestMargin)
					bestMargin = s;
			}
			cover[t] = (anyAlive ? bestMargin : -0.5f) * threat;
		}

		float blueAdv = cover[0] - cover[1];
		for (size_t i = 0; i < state.players.size(); i++)
			out[i] = (state.players[i].team == Team::BLUE) ? blueAdv : -blueAdv;
		return out;
	}

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		return GetAllRewards(state, isFinal)[player.index];
	}
};

// Penalizes crowding teammates and double-committing on the ball in 2v2/3v3 (stronger on the later/farther
// committer so the 2nd man rotates off cleanly), while relaxing inside own goal crease for saves. Zero in 1v1.
class DoubleCommitReward : public Reward {
public:
	static constexpr float SPACE_DIST       = 1350.f;
	static constexpr float BALL_COMMIT_DIST = 1450.f;

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (player.isDemoed || state.players.size() <= 2)
			return 0.f;

		float ownBallDist = (state.ball.pos - player.pos).Length();
		float sign = (player.team == Team::BLUE) ? 1.f : -1.f;
		bool inOwnCrease = (player.pos.y * sign < -4300.f && fabsf(player.pos.x) < 1200.f);

		float penalty = 0.f;
		for (const auto& other : state.players) {
			if (other.team != player.team || other.carId == player.carId || other.isDemoed)
				continue;
			float d = (other.pos - player.pos).Length();
			if (d >= SPACE_DIST)
				continue;
			float close = 1.f - d / SPACE_DIST;
			float otherBallDist = (state.ball.pos - other.pos).Length();
			float mult = 0.5f;
			if (!inOwnCrease && ownBallDist < BALL_COMMIT_DIST && otherBallDist < BALL_COMMIT_DIST)
				mult = (ownBallDist > otherBallDist) ? 1.6f : 0.9f;
			penalty = std::min(penalty, -close * mult);
		}
		return penalty;
	}
};

// The rewards of the training. --rewards changes the weights, a weight of 0 turns a reward off.
struct RewardDef {
	const char* name;
	float weight; // Default weight
	Reward* (*make)();
	const char* description;
};

static const std::vector<RewardDef> REWARD_DEFS = {
	// These are ok rewards that will produce a scoring bot in ~100m steps

	// Movement
	{ "Air", 0.1f, []() -> Reward* { return new AirReward(); }, "every step in the air" },

	// Player-ball
	{ "FaceBall", 0.25f, []() -> Reward* { return new FaceBallReward(); }, "facing the ball" },
	{ "VelocityPlayerToBall", 4.f, []() -> Reward* { return new VelocityPlayerToBallReward(); }, "speed towards the ball" },
	{ "StrongTouch", 75, []() -> Reward* { return new StrongTouchReward(20, 100); }, "hitting the ball hard (20-100 km/h change)" },

	// Ball-goal
	{ "VelocityBallToGoal", 3.f, []() -> Reward* { return new ZeroSumReward(new VelocityBallToGoalReward(), 1); }, "ball speed towards the opponent goal (zero-sum)" },

	// Positioning & spacing
	{ "BehindBall", 0.5f, []() -> Reward* { return new BehindBallReward(); }, "being between own goal and the ball" },
	{ "TeamSpacing", 1.f, []() -> Reward* { return new TeamSpacingReward(); }, "spacing out from teammates in 2v2/3v3" },

	// Boost
	{ "PickupBoost", 10.f, []() -> Reward* { return new PickupBoostReward(); }, "collecting boost" },
	{ "SaveBoost", 0.2f, []() -> Reward* { return new SaveBoostReward(); }, "having boost" },

	// Game events
	{ "Bump", 20, []() -> Reward* { return new ZeroSumReward(new BumpReward(), 0.5f); }, "bumping an opponent (zero-sum)" },
	{ "Demo", 80, []() -> Reward* { return new ZeroSumReward(new DemoReward(), 0.5f); }, "demolishing an opponent (zero-sum)" },
	{ "Goal", 200, []() -> Reward* { return new GoalReward(); }, "goals: plus for scoring, minus for conceding" },

	// Not used by default
	{ "TouchAccel", 15.f, []() -> Reward* { return new TouchAccelReward(); }, "speeding the ball up with a touch (up to 110 km/h)" },
	{ "TouchBall", 0, []() -> Reward* { return new TouchBallReward(); }, "every step with a ball touch" },
	{ "Speed", 0, []() -> Reward* { return new SpeedReward(); }, "own speed" },
	{ "KickoffProximity", 0, []() -> Reward* { return new KickoffProximityReward(); }, "kickoffs: the team's nearest player being closer to the ball than the opponents' while it still lies on the centre spot (zero-sum)" },
	{ "NearestVelocityToBall", 0, []() -> Reward* { return new NearestVelocityToBallReward(); }, "speed towards the ball, only for the player of each team that is nearest to the ball (in 1v1 the same as VelocityPlayerToBall)" },
	{ "TeamVelocityToBall", 0, []() -> Reward* { return new TeamVelocityToBallReward(); }, "speed towards the ball of the team's nearest player to the ball, for every player of the team (in 1v1 the same as VelocityPlayerToBall)" },

	// Competitive zero-sum multi-mode rewards (disabled by default unless selected in --rewards)
	{ "ObjectiveBallVel", 0, []() -> Reward* { return new ObjectiveBallVelReward(); }, "exact potential-derivative ball velocity toward goal mouth with box urgency (zero-sum)" },
	{ "DirectionalTouch", 0, []() -> Reward* { return new ZeroSumReward(new DirectionalTouchReward(), 0.35f); }, "goal-directed power touches, aerial hits, 50/50 wins and safe defensive clears (zero-sum)" },
	{ "GoalSpeed", 0, []() -> Reward* { return new GoalSpeedReward(); }, "bonus scaled by shot speed when a goal is scored (zero-sum)" },
	{ "ScorerAssist", 0, []() -> Reward* { return new ZeroSumReward(new ScorerAssistReward(), 0.f); }, "direct scorer and passing assist credit on goals (zero-sum)" },
	{ "Shot", 0, []() -> Reward* { return new ZeroSumReward(new ShotAndPassReward(), 0.25f); }, "verified high-speed shots on target and shot-passes (zero-sum)" },
	{ "Save", 0, []() -> Reward* { return new ZeroSumReward(new ClutchSaveReward(), 0.25f); }, "verified saves and goal-bound shot blocks with cooldown (zero-sum)" },
	{ "KickoffSpeed", 0, []() -> Reward* { return new KickoffSpeedReward(); }, "closing speed, first-touch win and directional 50/50 outcome on kickoffs (zero-sum)" },
	{ "TeamPressure", 0, []() -> Reward* { return new TeamPressureReward(); }, "non-negative closing speed of each team's first-man challenger in 2v2/3v3 (zero-sum, 0 in 1v1)" },
	{ "SoloDiscipline", 0, []() -> Reward* { return new SoloDisciplineReward(); }, "1v1 shadow recovery and anti-overcommit air discipline (zero-sum, 0 in 2v2/3v3)" },
	{ "DefensiveCover", 0, []() -> Reward* { return new DefensiveCoverReward(); }, "last-man 2D goal-side rotation cover in 2v2/3v3 (zero-sum, 0 in 1v1)" },
	{ "DoubleCommit", 0, []() -> Reward* { return new ZeroSumReward(new DoubleCommitReward(), 0.3f); }, "penalizes teammate crowding and ball double-commits in 2v2/3v3 (zero-sum, 0 in 1v1)" },
	{ "PickupBoostZS", 0, []() -> Reward* { return new ZeroSumReward(new PickupBoostReward(), 0.2f); }, "collecting boost pads and starving opponents (zero-sum)" },
	{ "DemoZS", 0, []() -> Reward* { return new ZeroSumReward(new DemoReward(), 0.5f); }, "demolishing opponents while avoiding being demolished (zero-sum)" },
};

// Named reward presets for --rewards. Every preset lists all weights explicitly, so a preset never
// silently inherits a changed default. "v3" is the recipe the 66-cycle campaign was trained with;
// "v4" restores the dense per-step shaping that RLGym ships by default (player-to-ball speed, touch
// acceleration, air/face micro-shaping, team spacing) which v3 had switched off, and rebalances the
// event rewards around it.
static const std::map<std::string, std::string> REWARD_PRESETS = {
	{ "v3", "Air=0,FaceBall=0,VelocityPlayerToBall=0,NearestVelocityToBall=0,TeamVelocityToBall=0,StrongTouch=0,"
	        "TouchAccel=0,VelocityBallToGoal=0,BehindBall=0,TeamSpacing=0,PickupBoost=0,Demo=0,Goal=750,"
	        "GoalSpeed=120,ScorerAssist=85,Save=28,Shot=18,DirectionalTouch=16,ObjectiveBallVel=6.5,"
	        "KickoffSpeed=5.5,TeamPressure=1.6,SoloDiscipline=0.55,DefensiveCover=0.45,DoubleCommit=0.9,"
	        "PickupBoostZS=3.5,SaveBoost=0.06,DemoZS=25,Bump=4" },
	// v4: the event rewards of v3 (Goal stays the anchor at 750) plus the dense RLGym-style shaping that v3 had
	// turned off, at weights below the REWARD_DEFS suggestions because those suggestions assume no event rewards:
	// Air 0.05 (vs 0.1) and FaceBall 0.1 (vs 0.25) pay for aerial play, TouchAccel 10 (vs 15) for flicks and power
	// shots, TeamSpacing 0.75 (vs 1) for rotation, NearestVelocityToBall 0.5 for the nearest player of each team
	// driving at the ball (0.5, not RLGym's 1.0, because DirectionalTouch/ObjectiveBallVel already pay for it).
	{ "v4", "Air=0.05,FaceBall=0.1,VelocityPlayerToBall=0,NearestVelocityToBall=0.5,TouchBall=0,Speed=0,"
	        "TeamVelocityToBall=0,StrongTouch=0,TouchAccel=10,VelocityBallToGoal=0,BehindBall=0,TeamSpacing=0.75,"
	        "KickoffProximity=0,PickupBoost=0,Demo=0,Goal=750,GoalSpeed=110,ScorerAssist=75,Save=30,Shot=20,"
	        "DirectionalTouch=16,ObjectiveBallVel=10,KickoffSpeed=6,TeamPressure=2,SoloDiscipline=0.7,"
	        "DefensiveCover=0.6,DoubleCommit=1,PickupBoostZS=4,SaveBoost=0.05,DemoZS=15,Bump=3" },
	// v5: v4 with FaceBall removed. FaceBallReward is rotMat.forward . normalize(ball - player), paid every step
	// regardless of what the car is doing, so at 0.1 it paid ~0.032/step (~19-22 per 600-step episode, more than a
	// whole Shot=20 event) for pointing the nose at the ball. By 1795 Elo that is a learned reflex, and the term's
	// cheapest optimum is hovering near the ball facing it instead of committing to a play.
	{ "v5", "Air=0.05,FaceBall=0,VelocityPlayerToBall=0,NearestVelocityToBall=0.5,TouchBall=0,Speed=0,"
	        "TeamVelocityToBall=0,StrongTouch=0,TouchAccel=10,VelocityBallToGoal=0,BehindBall=0,TeamSpacing=0.75,"
	        "KickoffProximity=0,PickupBoost=0,Demo=0,Goal=750,GoalSpeed=110,ScorerAssist=75,Save=30,Shot=20,"
	        "DirectionalTouch=16,ObjectiveBallVel=10,KickoffSpeed=6,TeamPressure=2,SoloDiscipline=0.7,"
	        "DefensiveCover=0.6,DoubleCommit=1,PickupBoostZS=4,SaveBoost=0.05,DemoZS=15,Bump=3" },
};

// Weights that differ from the defaults (--rewards)
static std::map<std::string, float> g_RewardWeights = {};
static std::string g_RewardPreset = {};

float GetRewardWeight(const RewardDef& def) {
	auto itr = g_RewardWeights.find(def.name);
	return (itr != g_RewardWeights.end()) ? itr->second : def.weight;
}

// "VelocityBallToGoal=6,Air=0" -> g_RewardWeights; a bare preset name ("v4") expands to its weights
void ParseRewardWeights(std::string str) {
	if (!str.empty() && str.find('=') == std::string::npos) {
		auto preset = REWARD_PRESETS.find(str);
		if (preset == REWARD_PRESETS.end()) {
			std::string names = {};
			for (auto& pair : REWARD_PRESETS)
				names += (names.empty() ? "" : ", ") + pair.first;
			RG_ERR_CLOSE("Option --rewards: \"" << str << "\" is neither Name=weight nor a preset (available: " << names << ")");
		}
		g_RewardPreset = str;
		str = preset->second;
	}
	g_RewardWeights.clear();

	std::stringstream stream(str);
	std::string item;
	while (std::getline(stream, item, ',')) {
		size_t eqPos = item.find('=');
		std::string name = item.substr(0, eqPos);
		const RewardDef* found = NULL;
		for (auto& def : REWARD_DEFS)
			if (name == def.name)
				found = &def;

		float weight = 0;
		bool valid = found && eqPos != std::string::npos;
		if (valid) {
			try {
				size_t numParsed = 0;
				std::string weightStr = item.substr(eqPos + 1);
				weight = std::stof(weightStr, &numParsed);
				valid = (numParsed == weightStr.size());
			} catch (std::exception&) {
				valid = false;
			}
		}

		if (!valid) {
			std::string names = {};
			for (auto& def : REWARD_DEFS)
				names += (names.empty() ? "" : ", ") + std::string(def.name);
			RG_ERR_CLOSE("Option --rewards: \"" << item << "\" is not valid, expected Name=weight with one of the names: " << names);
		}
		g_RewardWeights[name] = weight;
	}
}

// All rewards that are in use with their weights, e.g. "Air=0.25, FaceBall=0.25, ..."
std::string GetRewardSummary() {
	std::stringstream result;
	for (auto& def : REWARD_DEFS) {
		float weight = GetRewardWeight(def);
		if (weight != 0)
			result << (result.tellp() > 0 ? ", " : "") << def.name << "=" << weight;
	}
	return result.str();
}

// Portion of episodes that start from a random state instead of a kickoff (set with --random-state-prob)
static float g_RandomStateProb = 0;

// Random states: top speed of the ball (uu/s), and how the cars start (set with --random-ball-speed, --random-car-speed, --random-cars-in-air)
static float g_RandomBallSpeed = 2000;
static bool g_RandomCarSpeed = false, g_RandomCarsInAir = false;
static float g_TacticalStateProb = 0;
static bool g_FuzzedKickoffs = false;

// Clamps a 2D point comfortably inside Soccar arena side/back walls and 45-degree corner chamfers.
static Vec ClampArenaXY(float x, float y, float z = 17.f) {
	x = std::clamp(x, -3450.f, 3450.f);
	y = std::clamp(y, -4550.f, 4550.f);
	constexpr float CORNER_SUM = 7150.f; // Real corner wall satisfies |x| + |y| ~= 8064
	float s = fabsf(x) + fabsf(y);
	if (s > CORNER_SUM) {
		float scale = CORNER_SUM / s;
		x *= scale;
		y *= scale;
	}
	return Vec(x, y, z);
}

// Symmetric game-realistic tactical state setter: practices clutch saves & shot-finishing,
// corner/backboard clears & centering passes, high-speed open-field counterattacks, and contested
// midfield ground/aerial 50/50s — always with aligned forward car velocities, realistic boost,
// and exact 50/50 symmetry between Blue and Orange.
class TacticalStateSetter : public StateSetter {
	static CarState MakeGroundCar(Vec pos, float yaw, float forwardSpeed, float boost) {
		CarState cs = {};
		cs.pos = ClampArenaXY(pos.x, pos.y, 17.f);
		cs.isOnGround = true;
		Angle ang(yaw, 0.f, 0.f);
		cs.rotMat = ang.ToRotMat();
		cs.vel = cs.rotMat.forward * forwardSpeed;
		cs.isSupersonic = (forwardSpeed >= CommonValues::SUPERSONIC_THRESHOLD);
		cs.boost = std::clamp(boost, 0.f, 100.f);
		return cs;
	}

	static float YawToward(const Vec& from, const Vec& to, float jitter = 0.35f) {
		float base = atan2f(to.y - from.y, to.x - from.x);
		return base + RocketSim::Math::RandFloat(-jitter, jitter);
	}

public:
	virtual void ResetArena(Arena* arena) override {
		using RocketSim::Math::RandFloat;
		using RocketSim::Math::RandInt;

		arena->ResetToRandomKickoff();

		// Randomly pick which side starts under pressure (exact 50/50 Blue/Orange symmetry)
		float attackSign = (RandInt(0, 2) == 0) ? 1.f : -1.f; // +1: Blue attacks +Y, -1: Orange attacks -Y
		float mirrorSign = (RandInt(0, 2) == 0) ? 1.f : -1.f; // exact 50/50 left/right symmetry
		int scenario = RandInt(0, 4);

		BallState bs = {};
		std::vector<Car*> attackers, defenders;
		for (Car* car : arena->_cars) {
			bool isAtt = ((car->team == Team::BLUE) == (attackSign > 0.f));
			(isAtt ? attackers : defenders).push_back(car);
		}

		// Ensure no two cars overlap when placed
		std::vector<Vec> placed;
		auto placeSafe = [&](Vec p) -> Vec {
			p = ClampArenaXY(p.x, p.y, 17.f);
			for (int attempt = 0; attempt < 8; attempt++) {
				bool ok = (p.DistSq2D(bs.pos) > 260.f * 260.f);
				for (const Vec& q : placed)
					if (p.DistSq2D(q) < 320.f * 320.f)
						ok = false;
				if (ok)
					break;
				p = ClampArenaXY(p.x + RandFloat(-500.f, 500.f), p.y + RandFloat(-500.f, 500.f), 17.f);
			}
			placed.push_back(p);
			return p;
		};

		if (scenario == 0) {
			// Scenario 0: Shot on target / clutch goal-line & shadow save drill
			float bx = mirrorSign * RandFloat(-200.f, 2200.f);
			float by = attackSign * RandFloat(1200.f, 3400.f);
			float bz = RandFloat(95.f, 520.f);
			bs.pos = ClampArenaXY(bx, by, bz);

			Vec goalTarget(RandFloat(-780.f, 780.f), attackSign * 5200.f, RandFloat(110.f, 560.f));
			Vec toGoal = goalTarget - bs.pos;
			float speed = RandFloat(1250.f, 2850.f);
			float flightT = std::max(0.35f, toGoal.Length() / speed);
			bs.vel = toGoal.Normalized() * speed;
			// Compensate partial gravity so aerial shots actually reach the goal frame
			bs.vel.z = std::clamp((goalTarget.z - bs.pos.z - 0.5f * CommonValues::GRAVITY_Z * flightT * flightT) / flightT, -600.f, 1100.f);
			bs.angVel = RLGC::Math::RandVec(Vec(-3, -3, -3), Vec(3, 3, 3));

			for (size_t i = 0; i < defenders.size(); i++) {
				Vec pos = (i == 0)
					? Vec(-mirrorSign * RandFloat(100.f, 1100.f), attackSign * RandFloat(3900.f, 4520.f), 17.f)
					: Vec(RandFloat(-2600.f, 2600.f), attackSign * RandFloat(1800.f, 4200.f), 17.f);
				pos = placeSafe(pos);
				Vec aim(RandFloat(-400.f, 400.f), attackSign * 4900.f, 17.f);
				float yaw = YawToward(pos, (i == 0) ? bs.pos : aim, 0.45f);
				defenders[i]->SetState(MakeGroundCar(pos, yaw, RandFloat(400.f, 1650.f), RandFloat(15.f, 80.f)));
			}
			for (size_t i = 0; i < attackers.size(); i++) {
				Vec pos = (i == 0)
					? Vec(bs.pos.x + RandFloat(-700.f, 700.f), bs.pos.y - attackSign * RandFloat(450.f, 1300.f), 17.f)
					: Vec(RandFloat(-2800.f, 2800.f), attackSign * RandFloat(-800.f, 2400.f), 17.f);
				pos = placeSafe(pos);
				float yaw = YawToward(pos, bs.pos, 0.4f);
				attackers[i]->SetState(MakeGroundCar(pos, yaw, RandFloat(600.f, 1850.f), RandFloat(10.f, 75.f)));
			}
		} else if (scenario == 1) {
			// Scenario 1: Corner / backboard centering pass vs back-post clear
			float bx = mirrorSign * RandFloat(1400.f, 3300.f);
			float by = attackSign * RandFloat(3100.f, 4450.f);
			float bz = RandFloat(95.f, 780.f);
			bs.pos = ClampArenaXY(bx, by, bz);

			Vec boxTarget(mirrorSign * RandFloat(-600.f, 700.f), attackSign * RandFloat(3800.f, 4850.f), RandFloat(120.f, 550.f));
			bs.vel = (boxTarget - bs.pos).Normalized() * RandFloat(900.f, 2300.f);
			bs.angVel = RLGC::Math::RandVec(Vec(-4, -4, -4), Vec(4, 4, 4));

			for (size_t i = 0; i < defenders.size(); i++) {
				Vec pos = (i == 0)
					? Vec(-mirrorSign * RandFloat(-200.f, 950.f), attackSign * RandFloat(4100.f, 4520.f), 17.f)
					: Vec(mirrorSign * RandFloat(-1800.f, 2200.f), attackSign * RandFloat(2500.f, 4300.f), 17.f);
				pos = placeSafe(pos);
				defenders[i]->SetState(MakeGroundCar(pos, YawToward(pos, bs.pos, 0.5f), RandFloat(300.f, 1500.f), RandFloat(12.f, 85.f)));
			}
			for (size_t i = 0; i < attackers.size(); i++) {
				Vec pos = (i == 0)
					? Vec(RandFloat(-1500.f, 1500.f), attackSign * RandFloat(1800.f, 3500.f), 17.f)
					: Vec(RandFloat(-2800.f, 2800.f), attackSign * RandFloat(-500.f, 2600.f), 17.f);
				pos = placeSafe(pos);
				attackers[i]->SetState(MakeGroundCar(pos, YawToward(pos, boxTarget, 0.45f), RandFloat(500.f, 1750.f), RandFloat(15.f, 85.f)));
			}
		} else if (scenario == 2) {
			// Scenario 2: Fast counterattack / open-field transition with moving cars
			bs.pos = ClampArenaXY(RandFloat(-2600.f, 2600.f), attackSign * RandFloat(-1500.f, 2000.f), RandFloat(95.f, 460.f));
			Vec upfield(RandFloat(-1800.f, 1800.f), attackSign * 5000.f, RandFloat(100.f, 450.f));
			bs.vel = (upfield - bs.pos).Normalized() * RandFloat(700.f, 2300.f);
			bs.angVel = RLGC::Math::RandVec(Vec(-4, -4, -4), Vec(4, 4, 4));

			for (size_t i = 0; i < attackers.size(); i++) {
				float yOff = (i == 0) ? RandFloat(450.f, 1500.f) : RandFloat(1200.f, 3200.f);
				Vec pos = placeSafe(Vec(bs.pos.x + RandFloat(-1400.f, 1400.f), bs.pos.y - attackSign * yOff, 17.f));
				attackers[i]->SetState(MakeGroundCar(pos, YawToward(pos, bs.pos, 0.35f), RandFloat(800.f, 2100.f), RandFloat(20.f, 95.f)));
			}
			for (size_t i = 0; i < defenders.size(); i++) {
				Vec pos = placeSafe(Vec(RandFloat(-2800.f, 2800.f), bs.pos.y + attackSign * RandFloat(-400.f, 2600.f), 17.f));
				Vec ownNet(0.f, attackSign * 5000.f, 17.f);
				defenders[i]->SetState(MakeGroundCar(pos, YawToward(pos, (i == 0) ? bs.pos : ownNet, 0.5f), RandFloat(700.f, 2050.f), RandFloat(10.f, 80.f)));
			}
		} else {
			// Scenario 3: Contested midfield 50/50 & bouncing aerial ball
			bs.pos = ClampArenaXY(RandFloat(-2900.f, 2900.f), RandFloat(-2200.f, 2200.f), RandFloat(95.f, 920.f));
			bs.vel = Vec(RandFloat(-1600.f, 1600.f), RandFloat(-1600.f, 1600.f), RandFloat(-300.f, 750.f));
			if (bs.vel.LengthSq() < 200.f * 200.f)
				bs.vel.y = attackSign * 500.f;
			bs.angVel = RLGC::Math::RandVec(Vec(-4, -4, -4), Vec(4, 4, 4));

			auto placeTeam = [&](std::vector<Car*>& cars, float teamHomeSign) {
				for (size_t i = 0; i < cars.size(); i++) {
					float backDist = (i == 0) ? RandFloat(650.f, 1700.f) : RandFloat(1500.f, 3400.f);
					Vec pos = placeSafe(Vec(bs.pos.x + RandFloat(-1600.f, 1600.f), bs.pos.y + teamHomeSign * backDist, 17.f));
					cars[i]->SetState(MakeGroundCar(pos, YawToward(pos, bs.pos, 0.45f), RandFloat(500.f, 1950.f), RandFloat(15.f, 90.f)));
				}
			};
			placeTeam(attackers, -attackSign);
			placeTeam(defenders, attackSign);
		}

		arena->ball->SetState(bs);
	}
};

// RLGym's RandomState (random positions for the ball and the cars), with the top speed of the ball as a setting.
// NOTE: The ball needs some speed. RocketSim leaves a ball that is exactly at rest alone until something touches it
//	(as the game does with the kickoff ball), so a ball that is placed in the air with zero velocity hangs there:
//	out of reach for a bot on the ground, for the whole episode.
class RandomStateWithBallSpeed : public StateSetter {
public:
	RandomState randomState;
	float maxBallSpeed;

	RandomStateWithBallSpeed(float maxBallSpeed, bool randCarSpeed, bool carsOnGround) :
		randomState(maxBallSpeed > 0, randCarSpeed, carsOnGround), maxBallSpeed(maxBallSpeed) {
	}

	virtual void ResetArena(Arena* arena) {
		randomState.ResetArena(arena);

		if (maxBallSpeed > 0) {
			constexpr float RANDOM_STATE_MAX_BALL_SPEED = 4000; // RandomState picks a speed between 0 and this
			BallState ballState = arena->ball->GetState();
			ballState.vel *= maxBallSpeed / RANDOM_STATE_MAX_BALL_SPEED;
			arena->ball->SetState(ballState);
		}
	}
};

// Which observation and network the bot uses (set with --model)
struct BotSpec {
	std::string name;
	const char* description;

	// false: The original example, AdvancedObs and a plain network. Its observation grows with the number of players, so it only plays 1v1.
	// true:  MultiModeObs, which has the same size in 1v1, 2v2 and 3v3, so one network plays all three.
	bool multiMode;

	// Layer sizes. The shared head feeds both the policy and the critic.
	std::vector<int> slotEncoderSizes, sharedHeadSizes, policySizes, criticSizes;

	// (For comparisons) false: no slot encoder, the player slots go to the shared head like every other input
	bool useSlotEncoder = true;

	// Not empty: this is the 1v1-only version of that multi-mode model (SlotEncoderConfig::singleEncoding).
	// The shared layers get the one other player's encoding directly, instead of maximum and mean over the teammates and
	// over the opponents, three quarters of which are zeros or copies in 1v1. It takes over the checkpoints of that model.
	std::string only1v1Of = {};
};

static const std::vector<BotSpec> BOT_PRESETS = {
	{ "example", "the original example bot, 1v1 only",                         false, {},           { 256, 256 },          { 256, 256, 256 }, { 256, 256, 256 } },
	{ "s",       "1v1/2v2/3v3, small (for tests and weak CPUs)",               true,  { 128, 128 }, { 256, 256 },          { 256, 256 },      { 256, 256 } },
	{ "s-1v1",   "1v1 only: \"s\" without the parts that 1v1 does not use, faster. Continues checkpoints of \"s\"", true, { 128, 128 }, { 256, 256 }, { 256, 256 }, { 256, 256 }, true, "s" },
	{ "m",       "1v1/2v2/3v3, medium",                                        true,  { 256, 256 }, { 512, 512, 512 },     { 512, 512 },      { 512, 512 } },
	{ "l",       "1v1/2v2/3v3, large (the size this project trains)",          true,  { 256, 256 }, { 1024, 1024, 1024 },  { 512, 512 },      { 512, 512 } },
	{ "xl",      "1v1/2v2/3v3, extra large (needs a GPU)",                     true,  { 512, 512 }, { 2048, 2048, 1024 },  { 1024, 1024 },    { 1024, 1024 } },
};

static BotSpec g_Bot = BOT_PRESETS[0];

// Team sizes of the game modes the bot is trained in, and the team size of each game (see SetTrainModes())
static std::vector<int> g_TrainTeamSizes = { 1 };
static std::vector<int> g_TeamSizePattern = { 1 };
static bool g_RenderOnly = false;
static std::mt19937 g_RenderModeRng;

static int ChooseRenderTeamSize() {
	std::uniform_int_distribution<size_t> choose(0, g_TrainTeamSizes.size() - 1);
	int size = g_TrainTeamSizes[choose(g_RenderModeRng)];
	RG_LOG("Render episode: " << size << "v" << size << " (one arena, deterministic argmax)");
	return size;
}

// "s", "m-flat", ... -> BotSpec. The suffix "-flat" removes the slot encoder (for comparisons)
BotSpec FindBotPreset(std::string name) {
	const std::string FLAT_SUFFIX = "-flat";
	bool flat = name.size() > FLAT_SUFFIX.size() && name.compare(name.size() - FLAT_SUFFIX.size(), FLAT_SUFFIX.size(), FLAT_SUFFIX) == 0;
	std::string baseName = flat ? name.substr(0, name.size() - FLAT_SUFFIX.size()) : name;

	for (auto& preset : BOT_PRESETS) {
		if (preset.name == baseName && (!flat || preset.multiMode)) {
			BotSpec result = preset;
			result.name = name;
			result.useSlotEncoder = !flat;
			return result;
		}
	}

	std::string names = {};
	for (auto& preset : BOT_PRESETS)
		names += (names.empty() ? "" : ", ") + preset.name;
	RG_ERR_CLOSE("Unknown model \"" << name << "\", expected one of: " << names);
	return {};
}

ObsBuilder* MakeObsBuilder() {
	if (g_Bot.multiMode) {
		// Without a slot encoder the network sees every slot separately, so the slots are shuffled to keep it from relying on their order
		return new MultiModeObs(!g_Bot.useSlotEncoder);
	} else {
		return new AdvancedObs();
	}
}

ActionParser* MakeActionParser() {
	return new DefaultAction();
}

Arena* MakeArena(int teamSize) {
	auto arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < teamSize; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}
	return arena;
}

void SetModelConfig(PPOLearnerConfig& ppo) {
	ppo.sharedHead.layerSizes = g_Bot.sharedHeadSizes;
	ppo.policy.layerSizes = g_Bot.policySizes;
	ppo.critic.layerSizes = g_Bot.criticSizes;

	if (g_Bot.multiMode && g_Bot.useSlotEncoder) {
		// Every other player is run through the same small network together with the own car and the ball,
		//	and the results are pooled over the teammates and over the opponents (see SlotEncoderConfig).
		auto& slots = ppo.sharedHead.slots;
		slots.slotSize = MultiModeObs::SLOT_SIZE;
		slots.groupSizes = { MultiModeObs::NUM_TEAMMATE_SLOTS, MultiModeObs::NUM_OPPONENT_SLOTS };
		slots.contextSize = MultiModeObs::CONTEXT_SIZE;
		slots.layerSizes = g_Bot.slotEncoderSizes;
		slots.singleEncoding = !g_Bot.only1v1Of.empty();
	}

	auto optim = ModelOptimType::ADAM;
	ppo.policy.optimType = optim;
	ppo.critic.optimType = optim;
	ppo.sharedHead.optimType = optim;

	auto activation = ModelActivationType::RELU;
	ppo.policy.activationType = activation;
	ppo.critic.activationType = activation;
	ppo.sharedHead.activationType = activation;

	bool addLayerNorm = true;
	ppo.policy.addLayerNorm = addLayerNorm;
	ppo.critic.addLayerNorm = addLayerNorm;
	ppo.sharedHead.addLayerNorm = addLayerNorm;
}

// Sets the game modes to train in (--modes).
// Equal arenas (legacy) or equal player experience: for 1v1/2v2/3v3 a 6:3:2 arena mix
// gives each mode 6 players per side. Multiples of 11 arenas implement that mix exactly.
void SetTrainModes(const std::vector<int>& teamSizes, bool balancePlayers) {
	g_TrainTeamSizes = teamSizes;
	g_TeamSizePattern.clear();

	std::vector<int> gamesOfMode = {}, placedOfMode(teamSizes.size(), 0);
	int patternSize = 0;
	for (int teamSize : teamSizes) {
		int games = balancePlayers ? 6 / teamSize : 1;
		gamesOfMode.push_back(games);
		patternSize += games;
	}

	// Spread every mode evenly over the pattern
	for (int i = 0; i < patternSize; i++) {
		int bestMode = 0;
		float bestLag = -FLT_MAX;
		for (int mode = 0; mode < gamesOfMode.size(); mode++) {
			float lag = gamesOfMode[mode] * (i + 1) / (float)patternSize - placedOfMode[mode];
			if (lag > bestLag) {
				bestLag = lag;
				bestMode = mode;
			}
		}
		placedOfMode[bestMode]++;
		g_TeamSizePattern.push_back(teamSizes[bestMode]);
	}
}

// Team size of each game of the training
int GetTeamSizeOfGame(int index) {
	return g_TeamSizePattern[index % g_TeamSizePattern.size()];
}

// Create the RLGymCPP environment for each of our games
EnvCreateResult EnvCreateFunc(int index) {
	std::vector<WeightedReward> rewards = {};
	for (auto& def : REWARD_DEFS) {
		float weight = GetRewardWeight(def);
		if (weight != 0)
			rewards.push_back({ def.make(), weight });
	}

	std::vector<TerminalCondition*> terminalConditions = {};
	if (g_NoTouchTimeout > 0)
		terminalConditions.push_back(new NoTouchCondition(g_NoTouchTimeout));
	terminalConditions.push_back(new GoalScoreCondition());

	EnvCreateResult result = {};
	result.actionParser = MakeActionParser();
	result.obsBuilder = MakeObsBuilder();

	auto fnMakeKickoff = []() -> StateSetter* {
		return g_FuzzedKickoffs ? (StateSetter*)new FuzzedKickoffState() : (StateSetter*)new KickoffState();
	};
	float kickProb = std::max(0.f, 1.f - g_RandomStateProb - g_TacticalStateProb);
	if (g_RandomStateProb <= 0.f && g_TacticalStateProb <= 0.f) {
		result.stateSetter = fnMakeKickoff();
	} else {
		std::vector<std::pair<StateSetter*, float>> setters = {};
		if (kickProb > 0.f)
			setters.push_back({ fnMakeKickoff(), kickProb });
		if (g_RandomStateProb > 0.f)
			setters.push_back({ new RandomStateWithBallSpeed(g_RandomBallSpeed, g_RandomCarSpeed, !g_RandomCarsInAir), g_RandomStateProb });
		if (g_TacticalStateProb > 0.f)
			setters.push_back({ new TacticalStateSetter(), g_TacticalStateProb });
		result.stateSetter = new CombinedState(setters);
	}

	result.terminalConditions = terminalConditions;
	result.rewards = rewards;

	result.arena = MakeArena(g_RenderOnly ? ChooseRenderTeamSize() : GetTeamSizeOfGame(index));

	return result;
}

// Size of the observation of one player (works without RocketSim being initialized)
int GetObsSize() {
	if (g_Bot.multiMode)
		return MultiModeObs::OBS_SIZE;

	GameState state = {};
	for (int i = 0; i < 2; i++) {
		Player player = {};
		player.index = i;
		player.carId = i + 1;
		player.team = (i % 2 == 0) ? Team::BLUE : Team::ORANGE;
		state.players.push_back(player);
	}

	ObsBuilder* obsBuilder = MakeObsBuilder();
	obsBuilder->Reset(state);
	return (int)obsBuilder->BuildObs(state.players[0], state).size();
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Command-line options
///////////////////////////////////////////////////////////////////////////////////////////////

struct OptionDef {
	const char* name;
	const char* defaultVal; // NULL = this option is a flag without a value
	const char* help;
};

static const OptionDef OPTION_DEFS[] = {
	{ "help", NULL, "Show this help" },

	{ "meshes", "", "Folder with the RocketSim collision meshes (default: look for \"collision_meshes\")" },
	{ "checkpoints", "checkpoints", "Checkpoint folder. Training resumes from the newest checkpoint in it" },

	// The bot
	{ "model", "auto", "Which bot (observation and network sizes), see the list below. auto = the one the checkpoint was made with, or \"example\" for a new one" },
	{ "modes", "auto", "Game modes to train in, e.g. \"1v1,2v2,3v3\". auto = all three for the 1v1/2v2/3v3 bots" },

	// Training (the defaults are the values of the original example)
	{ "device", "auto", "auto, cpu or cuda" },
	{ "games", "256", "Number of games simulated at once" },
	{ "ts-per-itr", "50000", "Timesteps collected per iteration" },
	{ "mode-balance", "games", "Allocate arenas equally by game (games) or by player timesteps (players)" },
	{ "fixed-rollout", NULL, "Bootstrap unfinished trajectories at each update; keep playing the same games with the new policy" },
	{ "lean-metrics", NULL, "Skip per-player and reward-component diagnostics; keep PPO, throughput and skill ratings" },
	{ "player-metrics", NULL, "Keep sampled per-player diagnostics even when --lean-metrics is enabled" },
	{ "reward-metrics", NULL, "Keep sampled reward-component diagnostics even when --lean-metrics is enabled" },
	{ "batch-size", "0", "PPO batch size, 0 = same as --ts-per-itr. Smaller = more updates per iteration" },
	{ "minibatch-size", "0", "Samples sent through the network at once while learning (the gradients are added up, so the result stays the same). 0 = automatic: the whole batch on a GPU, about 1024 on a CPU" },
	{ "threads", "0", "CPU threads used for learning on the CPU, 0 = automatic" },
	{ "collection-infer-threads", "0", "CPU threads for action inference during rollouts, 0 = automatic" },
	{ "epochs", "1", "PPO epochs per iteration" },
	{ "lr", "1.5e-4", "Learning rate of the policy and the critic" },
	{ "critic-lr", "0", "Separate critic learning rate, 0 = same as --lr" },
	{ "entropy-scale", "0.035", "Scale of the (normalized) entropy bonus" },
	{ "mask-entropy", NULL, "Normalize policy entropy over the valid actions of each state" },
	{ "norm-advantages", "off", "Normalize PPO policy advantages: off, all or mode (per 1v1/2v2/3v3)" },
	{ "target-kl", "0", "Stop remaining PPO batches in an iteration if batch KL exceeds 1.5x this threshold (0 = off)" },
	{ "gamma", "0.99", "Reward discount" },
	{ "gae-lambda", "0.95", "GAE lambda" },
	{ "return-stat-max-count", "0", "Cap effective sample count of return standardizer so reward changes adapt smoothly (0 = off)" },
	{ "random-state-prob", "0", "Portion of episodes (0-1) starting from a random state instead of a kickoff" },
	{ "tactical-state-prob", "0", "Portion of episodes (0-1) starting from symmetric tactical save/shot/pass/transition states" },
	{ "fuzzed-kickoffs", NULL, "Apply tiny position fuzz on training kickoffs (matches anchored evaluation kickoffs)" },
	{ "pfsp", NULL, "Use Prioritized Fictitious Self-Play weighting when sampling saved policy opponents" },
	{ "random-ball-speed", "2000", "Random states: the ball starts with a random speed of up to this (uu/s, at most 6000). With 0 it hangs in the air until a car touches it" },
	{ "random-car-speed", NULL, "Random states: the cars start with a random speed and spin, instead of at rest" },
	{ "random-cars-in-air", NULL, "Random states: half of the cars start in the air with a random orientation, instead of all on the ground" },
	{ "no-touch-timeout", "10", "Seconds without a ball touch before an episode is cut off, 0 = never" },
	{ "rewards", "", "Reward weights that differ from the defaults, e.g. \"VelocityBallToGoal=6,Air=0\" (0 = off), see the list below — or a named preset: v3 (the 66-cycle campaign), v4 (dense RLGym-style shaping restored)" },
	{ "seed", "123", "Random seed of the training or the evaluation, -1 = use the current time" },
	{ "ts-per-save", "1000000", "Timesteps between checkpoints" },
	{ "keep-checkpoints", "8", "Number of checkpoints kept before the oldest are deleted, -1 = keep all" },
	{ "max-timesteps", "0", "Save and stop once this many total timesteps are reached, 0 = no limit" },
	{ "max-minutes", "0", "Save and stop after this many minutes, 0 = no limit" },
	{ "run-name", "gigalearncpp-run", "Run name for the metrics receiver" },
	{ "no-metrics", NULL, "Don't send metrics to the Python metrics receiver" },
	{ "wandb", NULL, "Also log the metrics to Weights & Biases (needs the API key in WANDB_API_KEY, or a previous `wandb login`)" },
	{ "wandb-project", "", "W&B project (default: $WANDB_PROJECT, or \"gigalearncpp\")" },
	{ "wandb-entity", "", "W&B user or team (default: $WANDB_ENTITY, or the default of the API key)" },
	{ "wandb-group", "", "W&B group of the run (default: $WANDB_RUN_GROUP, or \"unnamed-runs\")" },
	{ "new-run", NULL, "Start a new metrics run instead of continuing the one of the checkpoint (for a run that branches off another run)" },

	// GigaLearn's policy versions and skill tracker (the defaults are the ones of the framework)
	{ "skill-tracker", NULL, "Rate the bot with GigaLearn's skill tracker: every few iterations it plays against a saved old version of itself, and every goal moves its Elo rating (metric \"Rating/1v1\" etc.)" },
	{ "skill-interval", "16", "Skill tracker: iterations between rating games" },
	{ "skill-minutes", "0", "Skill tracker: rating sessions instead, for the least run time spent on rating. Every this many minutes, --skill-rounds rounds of rating games are played in a row, and nothing but training happens in between. Policy versions are then saved right after a session. 0 = off" },
	{ "skill-rounds", "10", "Skill tracker: rounds of rating games per session (with --skill-minutes). 10 rounds of 32 games take about 12 seconds and rate the bot to about +-17 points, 60 rounds to about +-12" },
	{ "skill-arenas", "16", "Skill tracker: rating games played at once" },
	{ "skill-sim-time", "45", "Skill tracker: seconds of game time per round of rating games" },
	{ "skill-max-sim-time", "240", "Skill tracker: seconds of game time after which the opponent version is changed, even without enough goals" },
	{ "skill-tournament-elo-budget", "100", "Skill tracker: maximum rating change of the current policy per tournament (a session or a --rate-rounds run), per game mode. Each round may move the rating by budget / rounds, however many goals are scored" },
	{ "skill-initial-rating", "0", "Skill tracker: rating of the first version" },
	{ "skill-deterministic", NULL, "Skill tracker: both sides pick their most likely action (like the bot does in the real game) instead of sampling" },
	{ "ts-per-version", "25m", "Timesteps between saved policy versions (kept in <checkpoints>/policy_versions). They are the opponents of the skill tracker and of --train-vs-old. With --skill-minutes: a version is saved after the first rating session that comes this many timesteps after the newest version" },
	{ "max-versions", "32", "Number of policy versions kept, the oldest are dropped" },
	{ "mirror", "auto", "Mirrored view: off, right, left or auto. With \"right\", a car on the left half of the field sees the whole game mirrored (and its action is mirrored back), so the network only ever sees its own car on the right half and learns every situation once instead of twice. \"left\" is the same with the halves swapped. auto = as the checkpoint was trained, and off for a new bot. Needs a multi-mode model" },
	{ "train-vs-old", "0", "Portion of iterations (0-1) in which the bot trains against a saved old version instead of itself (the framework suggests 0.15)" },
	{ "rate-rounds", "0", "Don't train, rate the newest checkpoint in --checkpoints with GigaLearn's skill tracker instead: it plays this many rounds of rating games against saved policy versions. Nothing is saved" },
	{ "rate-versions", "", "With --rate-rounds: folder of the policy versions to play against (default: policy_versions in --checkpoints). Give the same folder for checkpoints of different runs to rate them against the same opponents" },
	{ "rate-anchored", NULL, "With --rate-rounds and --rate-versions: fix opponent Elo, initialize each mode at its pool mean and cycle opponents on both colors. Rounds must be a multiple of twice the pool size" },
	{ "rate-out", "", "With --rate-rounds: also write the result to this JSON file" },
	{ "expand-1v1", "", "Don't train: copy the newest checkpoint of a 1v1-only model (s-1v1) from this folder to --checkpoints as a checkpoint of the full model (s), together with its policy versions. The copy plays 1v1 exactly as before, and starts 2v2 and 3v3 with what it does in 1v1, so it can be trained in more modes" },
	{ "render", NULL, "Don't train: one arena, deterministic argmax actions; randomly choose from --modes at every episode reset" },

	// Evaluation
	{ "eval", "", "Don't train, evaluate this checkpoint instead (a checkpoint folder, or a folder of checkpoints)" },
	{ "eval-mirror", "auto", "Evaluation: mirrored view of the --eval checkpoint (off, right, left, or auto = as it was trained, see --mirror)" },
	{ "vs-mirror", "auto", "Evaluation: mirrored view of the --vs checkpoint (off, right, left, or auto = as it was trained)" },
	{ "vs", "random", "Evaluation opponent: \"random\" (random actions) or another checkpoint" },
	{ "eval-minutes", "60", "Game time to simulate for the evaluation, in minutes (per game mode)" },
	{ "eval-modes", "auto", "Game modes to evaluate in, e.g. \"1v1,3v3\". auto = all modes the bot can play" },
	{ "eval-arenas", "16", "Number of games simulated at once for the evaluation" },
	{ "eval-no-touch-timeout", "10", "Evaluation: restart with a kickoff after this many seconds without a ball touch, 0 = never" },
	{ "eval-max-seconds", "0", "Evaluation: restart with a kickoff after this many seconds without a goal, 0 = never" },
	{ "eval-finish-games", NULL, "Evaluation: when the game time is up, let the running kickoffs finish instead of dropping them (needs one of the two limits above)" },
	{ "eval-out", "", "Also write the evaluation results to this JSON file" },
	{ "deterministic", NULL, "Evaluation: always pick the most likely action instead of sampling" },

	// RLBot
	{ "rlbot", "", "Don't train, run this checkpoint in Rocket League through RLBot instead" },
	{ "rlbot-port", "42653", "RLBot port, must match rlbot/port.cfg" },
};

struct Options {
	std::map<std::string, std::string> values;
	std::set<std::string> flags;

	static const OptionDef* FindDef(const std::string& name) {
		for (auto& def : OPTION_DEFS)
			if (name == def.name)
				return &def;
		return NULL;
	}

	static void PrintHelp() {
		std::cout << "Usage: GigaLearnBot [options]\n\n";
		std::cout << "Trains a bot by default. Use --eval, --rlbot or --render for the other modes.\n";
		std::cout << "Numbers can be written like 50000, 50k, 2.5m or 1b.\n\nOptions:\n";
		for (auto& def : OPTION_DEFS) {
			std::string left = std::string("  --") + def.name + (def.defaultVal ? " <value>" : "");
			std::cout << left << std::string(left.size() < 30 ? 30 - left.size() : 1, ' ') << def.help;
			if (def.defaultVal && def.defaultVal[0])
				std::cout << " [" << def.defaultVal << "]";
			std::cout << "\n";
		}

		auto fnSizes = [](const std::vector<int>& sizes) {
			std::string str = {};
			for (int size : sizes)
				str += (str.empty() ? "" : "-") + std::to_string(size);
			return str;
		};
		std::cout << "\nModels (--model):\n";
		for (auto& preset : BOT_PRESETS) {
			std::cout << "  " << preset.name << std::string(9 - preset.name.size(), ' ') << preset.description << ". ";
			if (preset.multiMode)
				std::cout << "Player encoder " << fnSizes(preset.slotEncoderSizes) << ", shared layers ";
			else
				std::cout << "Shared layers ";
			std::cout << fnSizes(preset.sharedHeadSizes) << ", policy " << fnSizes(preset.policySizes) << ", critic " << fnSizes(preset.criticSizes) << "\n";
		}

		std::cout << "\nRewards (--rewards Name=weight,...):\n";
		for (auto& def : REWARD_DEFS) {
			std::string name = def.name;
			std::cout << "  " << name << std::string(name.size() < 22 ? 22 - name.size() : 1, ' ') << def.description << " [" << def.weight << "]\n";
		}
		std::cout << std::flush;
	}

	// Returns false if the arguments are invalid
	bool Parse(int argc, char* argv[]) {
		for (auto& def : OPTION_DEFS)
			if (def.defaultVal)
				values[def.name] = def.defaultVal;

		for (int i = 1; i < argc; i++) {
			std::string arg = argv[i];
			if (arg == "-h")
				arg = "--help";

			const OptionDef* def = (arg.rfind("--", 0) == 0) ? FindDef(arg.substr(2)) : NULL;
			if (!def) {
				std::cout << "Unknown option \"" << arg << "\", use --help to see all options." << std::endl;
				return false;
			}

			if (def->defaultVal) {
				if (i + 1 >= argc) {
					std::cout << "Option \"" << arg << "\" needs a value." << std::endl;
					return false;
				}
				values[def->name] = argv[++i];
			} else {
				flags.insert(def->name);
			}
		}
		return true;
	}

	bool Has(const std::string& name) const {
		return flags.count(name) > 0;
	}

	std::string Str(const std::string& name) const {
		return values.at(name);
	}

	// Parses numbers, including ones like "50k", "2.5m", "1b" and "1,000,000"
	double Num(const std::string& name) const {
		std::string str = {};
		for (char c : values.at(name))
			if (c != ',' && c != '_' && c != '\'')
				str += c;

		double scale = 1;
		if (!str.empty()) {
			char last = (char)tolower((unsigned char)str.back());
			if (last == 'k') scale = 1e3;
			if (last == 'm') scale = 1e6;
			if (last == 'b') scale = 1e9;
			if (scale != 1)
				str.pop_back();
		}

		try {
			size_t numParsed = 0;
			double val = std::stod(str, &numParsed);
			if (numParsed != str.size())
				throw std::invalid_argument(str);
			return val * scale;
		} catch (std::exception&) {
			RG_ERR_CLOSE("Option --" << name << " has the invalid value \"" << values.at(name) << "\", expected a number");
		}
		return 0;
	}

	int64_t Int(const std::string& name) const {
		return (int64_t)llround(Num(name));
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////////////////////

std::filesystem::path GetExeDir(const char* argv0) {
	std::error_code ec;
#ifdef __linux__
	auto linkPath = std::filesystem::read_symlink("/proc/self/exe", ec);
	if (!ec)
		return linkPath.parent_path();
#endif
	auto path = std::filesystem::absolute(argv0, ec);
	return ec ? std::filesystem::current_path() : path.parent_path();
}

// Sets an environment variable of this process (the Python metrics receiver reads its settings from them)
void SetEnv(const char* name, const std::string& value) {
#ifdef _WIN32
	_putenv_s(name, value.c_str());
#else
	setenv(name, value.c_str(), 1);
#endif
}

// The first of: the value of an option, the value of an environment variable, a default
std::string OptionOrEnv(const std::string& optionValue, const char* envName, const char* defaultValue) {
	if (!optionValue.empty())
		return optionValue;
	const char* envValue = std::getenv(envName);
	return (envValue && envValue[0]) ? envValue : defaultValue;
}

// Finds the collision meshes folder and initializes RocketSim with it
void InitRocketSim(const Options& opt, std::filesystem::path exeDir) {
	std::vector<std::filesystem::path> candidates = {};
	if (!opt.Str("meshes").empty()) {
		candidates = { opt.Str("meshes") };
	} else {
		candidates = {
			"collision_meshes",
			exeDir / "collision_meshes",
			exeDir.parent_path() / "collision_meshes"
		};
	}

	for (auto& path : candidates) {
		if (std::filesystem::is_directory(path / "soccar")) {
			RG_LOG("Using collision meshes from " << std::filesystem::absolute(path));
			RocketSim::Init(path);
			return;
		}
	}

	std::stringstream tried;
	for (auto& path : candidates)
		tried << "\n\t" << std::filesystem::absolute(path);
	RG_ERR_CLOSE(
		"Couldn't find the RocketSim collision meshes (a folder that contains \"soccar/mesh_0.cmf\" etc.), tried:" << tried.str() <<
		"\nUse --meshes <folder> to say where they are."
	);
}

// The embedded Python interpreter needs to find the "python_scripts" folder (the metrics/render receivers).
// CMake copies that folder next to the executable, so we make Python look there no matter the working directory.
void AddToPythonPath(std::filesystem::path dir) {
	if (!std::filesystem::is_directory(dir / "python_scripts"))
		return;

	std::string newVal = dir.string();
	const char* oldVal = std::getenv("PYTHONPATH");
#ifdef _WIN32
	if (oldVal && oldVal[0])
		newVal += std::string(";") + oldVal;
	_putenv_s("PYTHONPATH", newVal.c_str());
#else
	if (oldVal && oldVal[0])
		newVal += std::string(":") + oldVal;
	setenv("PYTHONPATH", newVal.c_str(), 1);
#endif
}

// Accepts either a checkpoint folder, or a folder of timestep-numbered checkpoints (in which case the newest is used)
std::filesystem::path ResolveCheckpoint(std::filesystem::path path) {
	if (!std::filesystem::is_directory(path))
		RG_ERR_CLOSE("Checkpoint folder " << path << " does not exist");

	if (std::filesystem::exists(path / "POLICY.lt"))
		return path;

	int64_t newest = -1;
	for (auto& entry : std::filesystem::directory_iterator(path)) {
		std::string name = entry.path().filename().string();
		bool isNumber = !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char c) { return isdigit(c); });
		if (entry.is_directory() && isNumber && name.size() < 18)
			newest = RS_MAX(newest, std::stoll(name));
	}

	if (newest == -1)
		RG_ERR_CLOSE("Folder " << path << " is not a checkpoint (no POLICY.lt) and doesn't contain numbered checkpoints either");

	return path / std::to_string(newest);
}

// Path of the newest timestep-numbered checkpoint in a folder, or an empty path if there is none
std::filesystem::path FindNewestCheckpoint(std::filesystem::path folder) {
	std::error_code ec;
	if (!std::filesystem::is_directory(folder, ec))
		return {};

	int64_t newest = -1;
	for (auto& entry : std::filesystem::directory_iterator(folder)) {
		std::string name = entry.path().filename().string();
		bool isNumber = !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char c) { return isdigit(c); });
		if (entry.is_directory() && isNumber && name.size() < 18)
			newest = RS_MAX(newest, std::stoll(name));
	}

	return (newest == -1) ? std::filesystem::path() : (folder / std::to_string(newest));
}

// The mirrored view (--mirror) that the checkpoint or saved policy version in this folder was trained with:
// 0 = off, 1 = right, -1 = left
int ReadMirrorX(std::filesystem::path checkpointPath) {
	for (const char* fileName : { "RUNNING_STATS.json", "STATS.json" }) {
		std::ifstream fIn(checkpointPath / fileName);
		if (!fIn.good())
			continue;

		try {
			auto j = nlohmann::json::parse(fIn);
			if (j.contains("mirror_x")) {
				int mirrorX = j["mirror_x"].get<int>();
				return (mirrorX > 0) - (mirrorX < 0);
			}
		} catch (std::exception& e) {
			RG_ERR_CLOSE("Can't read " << (checkpointPath / fileName) << ": " << e.what());
		}
	}
	return 0;
}

const char* MirrorName(int mirrorX) {
	return (mirrorX == 0) ? "off" : ((mirrorX > 0) ? "right" : "left");
}

// Value of a --mirror style option: off, right, left, or auto (= as the checkpoint was trained)
int ResolveMirrorX(const Options& opt, const char* optionName, std::filesystem::path checkpointPath) {
	std::string val = opt.Str(optionName);
	if (val == "off")
		return 0;
	if (val == "right")
		return 1;
	if (val == "left")
		return -1;
	if (val != "auto")
		RG_ERR_CLOSE("Option --" << optionName << ": expected off, right, left or auto, got \"" << val << "\"");
	return checkpointPath.empty() ? 0 : ReadMirrorX(checkpointPath);
}

// The notes that training saved with a checkpoint (INFO.json), e.g. which model it was made with. Empty if there are none.
std::map<std::string, std::string> ReadCheckpointInfo(std::filesystem::path checkpointPath) {
	std::map<std::string, std::string> result = {};
	std::ifstream inFile(checkpointPath / "INFO.json");
	if (!inFile.good())
		return result;

	try {
		auto j = nlohmann::json::parse(inFile);
		for (auto& pair : j.items())
			if (pair.value().is_string())
				result[pair.key()] = pair.value().get<std::string>();
	} catch (std::exception& e) {
		RG_LOG("WARNING: Can't read " << (checkpointPath / "INFO.json") << ": " << e.what());
	}
	return result;
}

// Sets g_Bot from --model and the checkpoint that is going to be used (an empty path if there is none)
void SelectBot(const Options& opt, std::filesystem::path checkpointPath, bool ignoreModelOption = false) {
	std::string wanted = ignoreModelOption ? "auto" : opt.Str("model");

	std::string saved = {};
	if (!checkpointPath.empty()) {
		auto info = ReadCheckpointInfo(checkpointPath);
		if (info.count("model"))
			saved = info["model"];
	}

	if (wanted == "auto") {
		// Checkpoints from before there were several models have no notes, they are from the example bot
		wanted = saved.empty() ? "example" : saved;
	} else if (!saved.empty() && saved != wanted && FindBotPreset(wanted).only1v1Of != saved && FindBotPreset(saved).only1v1Of != wanted) {
		// (A model and its 1v1-only version take over each other's checkpoints, see BotSpec::only1v1Of)
		RG_ERR_CLOSE("The checkpoint " << checkpointPath << " was made with --model " << saved << ", it can't be used with --model " << wanted);
	}

	g_Bot = FindBotPreset(wanted);
}

// "1v1,2v2,3v3" (or "1,2,3") -> team sizes. "auto" = all modes the bot can play
std::vector<int> ParseModes(std::string str, const char* optionName) {
	std::vector<int> allTeamSizes = {};
	for (int teamSize = 1; teamSize <= ((g_Bot.multiMode && g_Bot.only1v1Of.empty()) ? MultiModeObs::MAX_TEAM_SIZE : 1); teamSize++)
		allTeamSizes.push_back(teamSize);

	if (str == "auto")
		return allTeamSizes;

	std::set<int> teamSizes = {};
	std::stringstream stream(str);
	std::string item;
	while (std::getline(stream, item, ',')) {
		int teamSize = 0;
		if (item.size() == 1 && isdigit((unsigned char)item[0])) {
			teamSize = item[0] - '0';
		} else if (item.size() == 3 && isdigit((unsigned char)item[0]) && item[1] == 'v' && item[2] == item[0]) {
			teamSize = item[0] - '0';
		}

		if (std::find(allTeamSizes.begin(), allTeamSizes.end(), teamSize) == allTeamSizes.end()) {
			RG_ERR_CLOSE(
				"Option --" << optionName << ": \"" << item << "\" is not a game mode that the model \"" << g_Bot.name << "\" can play " <<
				((g_Bot.multiMode && g_Bot.only1v1Of.empty()) ? "(expected a list like 1v1,2v2,3v3)" : "(it only plays 1v1, the models s, m, l and xl play 1v1, 2v2 and 3v3)")
			);
		}
		teamSizes.insert(teamSize);
	}

	if (teamSizes.empty())
		RG_ERR_CLOSE("Option --" << optionName << " needs at least one game mode");
	return std::vector<int>(teamSizes.begin(), teamSizes.end());
}

std::string ModeName(int teamSize) {
	return std::to_string(teamSize) + "v" + std::to_string(teamSize);
}

std::string ModeNames(const std::vector<int>& teamSizes) {
	std::string result = {};
	for (int teamSize : teamSizes)
		result += (result.empty() ? "" : ",") + ModeName(teamSize);
	return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Training
///////////////////////////////////////////////////////////////////////////////////////////////

static std::atomic<int> g_StopSignals = 0;
static std::chrono::steady_clock::time_point g_TrainStartTime = {};
static double g_MaxTrainSeconds = 0;
static uint64_t g_MaxTimesteps = 0;
static bool g_LeanMetrics = false;
static bool g_PlayerMetrics = false;

// First Ctrl+C (or kill): finish the current iteration, save, and stop. Second one: exit immediately.
void OnStopSignal(int signalNum) {
	if (++g_StopSignals > 1)
		std::_Exit(130);
	std::signal(signalNum, OnStopSignal);
}

double GetTrainSeconds() {
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_TrainStartTime).count();
}

void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {

	// Check if we should stop training
	// The learner then finishes the current iteration, saves a checkpoint, and returns from Start()
	if (!learner->quitRequested && !learner->config.renderMode) {
		const char* stopReason = NULL;
		if (g_StopSignals > 0) {
			stopReason = "interrupted";
		} else if (g_MaxTrainSeconds > 0 && GetTrainSeconds() >= g_MaxTrainSeconds) {
			stopReason = "time limit reached";
		}

		if (stopReason) {
			RG_LOG("Stopping after this iteration (" << stopReason << ")...");
			// A planned end gets a final rating session (if rating sessions are on), an interrupt saves and exits right away
			learner->rateBeforeQuit = (g_StopSignals == 0);
			learner->quitRequested = true;
		}
	}

	if (g_LeanMetrics && !g_PlayerMetrics) return; // PPO, timings, step reward and all ratings are still reported

	// Keep expensive diagnostics lightweight but evenly sample a quarter of steps; the metrics
	// receiver averages the skipped iteration windows so the published series are less noisy.
	static uint64_t metricSampleCounter = 0;
	bool doExpensiveMetrics = (metricSampleCounter++ % 4) == 0;

	// Add our metrics
	for (auto& state : states) {
		if (doExpensiveMetrics) {
			for (auto& player : state.players) {
				report.AddAvg("Player/In Air Ratio", !player.isOnGround);
				report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
				report.AddAvg("Player/Demoed Ratio", player.isDemoed);

				report.AddAvg("Player/Speed", player.vel.Length());
				Vec dirToBall = (state.ball.pos - player.pos).Normalized();
				report.AddAvg("Player/Speed Towards Ball", RS_MAX(0, player.vel.Dot(dirToBall)));
				report.AddAvg("Player/Ball Distance", player.pos.Dist(state.ball.pos));

				report.AddAvg("Player/Boost", player.boost);

				if (player.ballTouchedStep)
					report.AddAvg("Player/Touch Height", state.ball.pos.z);
			}
		}

		// Portion of game steps that end with a goal (multiply by steps per second for goals per second)
		report.AddAvg("Game/Goal Ratio", state.goalScored);

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}

	// The same for each game mode, if there are several
	if (g_TrainTeamSizes.size() > 1 && doExpensiveMetrics && !learner->config.renderMode) {
		static const std::string
			TOUCH_NAMES[] = { "Mode 1v1/Ball Touch Ratio", "Mode 2v2/Ball Touch Ratio", "Mode 3v3/Ball Touch Ratio" },
			DIST_NAMES[] = { "Mode 1v1/Ball Distance", "Mode 2v2/Ball Distance", "Mode 3v3/Ball Distance" },
			REWARD_NAMES[] = { "Mode 1v1/Step Reward", "Mode 2v2/Step Reward", "Mode 3v3/Step Reward" },
			GOAL_NAMES[] = { "Mode 1v1/Goal Ratio", "Mode 2v2/Goal Ratio", "Mode 3v3/Goal Ratio" };

		auto& envState = learner->envSet->state;
		for (int arenaIdx = 0; arenaIdx < states.size(); arenaIdx++) {
			auto& state = states[arenaIdx];
			int modeIdx = RS_CLAMP((int)state.players.size() / 2, 1, 3) - 1;
			int playerStartIdx = envState.arenaPlayerStartIdx[arenaIdx];

			for (int i = 0; i < state.players.size(); i++) {
				auto& player = state.players[i];
				report.AddAvg(TOUCH_NAMES[modeIdx], player.ballTouchedStep);
				report.AddAvg(DIST_NAMES[modeIdx], player.pos.Dist(state.ball.pos));
				report.AddAvg(REWARD_NAMES[modeIdx], envState.rewards[playerStartIdx + i]);
			}
			report.AddAvg(GOAL_NAMES[modeIdx], state.goalScored);
		}
	}
}

// Rates a checkpoint with GigaLearn's skill tracker, without training it (--rate-rounds).
// The loaded policy plays the tracker's matches against saved policy versions, and the tournament moves its own rating as during training:
// it starts from the Elo the checkpoint carries, and the result changes that policy, not the versions it played.
// With the versions of one run as common opponents (--rate-versions), checkpoints of different runs get ratings on the same scale.
int RunRating(Learner* learner, const Options& opt, int numRounds) {
	struct OpponentStats {
		double goalsFor = 0, goalsAgainst = 0;
		int rounds = 0;
		std::map<std::string, double> ratings;
	};
	std::map<uint64_t, OpponentStats> opponents;
	std::map<std::string, std::vector<double>> ratingSeries;
	std::map<std::string, std::pair<int, int>> modeGoals;
	std::vector<double> rankingHistory;
	std::map<std::string, float> startRatings = learner->GetSkillRatings();
	double goalsFor = 0, goalsAgainst = 0, gameMinutes = 0;
	double kickoffsFor = 0, kickoffsAgainst = 0;
	int roundsPlayed = 0;

	const std::string RATING_PREFIX = "Rating/", OPPONENT_RATING_PREFIX = "Skill Matches/Opponent Rating ";

	auto startTime = std::chrono::steady_clock::now();
	learner->StartSkillMatches(numRounds, [&](int round, const Report& report) {
		roundsPlayed = round + 1;
		if (g_StopSignals > 0)
			learner->quitRequested = true; // Ctrl+C: stop after this round and report what has been played
		double roundGoalsFor = report["Skill Matches/Goals For"], roundGoalsAgainst = report["Skill Matches/Goals Against"];
		goalsFor += roundGoalsFor;
		goalsAgainst += roundGoalsAgainst;
		gameMinutes += report["Skill Matches/Game Minutes"];
		kickoffsFor += report["Skill Matches/Kickoff First Touches For"];
		kickoffsAgainst += report["Skill Matches/Kickoff First Touches Against"];

		auto& opponent = opponents[(uint64_t)report["Skill Matches/Opponent Timesteps"]];
		opponent.goalsFor += roundGoalsFor;
		opponent.goalsAgainst += roundGoalsAgainst;
		opponent.rounds++;

		double sum = 0;
		int ratedModes = 0;
		for (int size : g_TrainTeamSizes) {
			auto name = ModeName(size);
			for (bool scored : {true, false}) {
				auto key = std::string(scored ? "Skill Matches/Goals For " : "Skill Matches/Goals Against ") + name;
				if (report.Has(key)) (scored ? modeGoals[name].first : modeGoals[name].second) += (int)report[key];
			}
			if (report.Has(RATING_PREFIX + name)) {
				double value = report[RATING_PREFIX + name];
				ratingSeries[name].push_back(value);
				sum += value;
				ratedModes++;
			}
			if (report.Has(OPPONENT_RATING_PREFIX + name))
				opponent.ratings[name] = report[OPPONENT_RATING_PREFIX + name];
		}
		if (ratedModes) rankingHistory.push_back(sum / ratedModes);
	});
	double realSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();

	auto& tracker = learner->config.skillTracker;
	std::filesystem::path versionsFolder = learner->config.policyVersionsFolder.empty() ?
		(learner->config.checkpointFolder / "policy_versions") : learner->config.policyVersionsFolder;

	nlohmann::json j = {};
	j["checkpoints"] = learner->config.checkpointFolder.string();
	j["checkpoint_timesteps"] = learner->totalTimesteps;
	j["versions_folder"] = versionsFolder.string();
	j["rounds"] = roundsPlayed;
	j["anchored"] = tracker.anchored;
	j["seed"] = learner->config.randomSeed;
	j["goals_for"] = goalsFor;
	for (auto& [name, goals] : modeGoals)
		j["mode_goals"][name] = { { "for", goals.first }, { "against", goals.second } };
	j["kickoff_first_touches_for"] = kickoffsFor;
	j["kickoff_first_touches_against"] = kickoffsAgainst;
	j["goals_against"] = goalsAgainst;
	j["game_minutes"] = gameMinutes;
	j["real_seconds"] = realSeconds;
	j["settings"] = {
		{ "arenas", tracker.numArenas }, { "sim_time", tracker.simTime }, { "max_sim_time", tracker.maxSimTime },
		{ "tournament_elo_budget", tracker.tournamentEloBudget }, { "initial_rating", tracker.initialRating },
		{ "deterministic", tracker.deterministic },
		{ "policy_temperature", learner->config.ppo.policyTemperature },
		// Skill matches always install RLGC::FuzzedKickoffState, so every round of a tournament starts from its
		//own kickoff: sampled actions at temperature 1 and these kickoffs are what keep rounds from repeating.
		{ "fuzzed_kickoffs", true }
	};

	RG_LOG(RG_DIVIDER);
	RG_LOG("Rating of the checkpoint with " << learner->totalTimesteps << " timesteps, by GigaLearn's skill tracker:");
	RG_LOG("\tOpponents: " << opponents.size() << " policy versions played, from " << versionsFolder);
	RG_LOG("\tRounds: " << roundsPlayed << " (" << tracker.numArenas << " games of up to " << tracker.simTime << "s each), " << (int)gameMinutes << " game minutes in " << (int)realSeconds << " seconds");
	if (goalsFor + goalsAgainst > 0)
		RG_LOG("\tGoals: " << goalsFor << ":" << goalsAgainst << " (" << (100 * goalsFor / (goalsFor + goalsAgainst)) << "% scored by the checkpoint)");
		if (kickoffsFor + kickoffsAgainst > 0)
			RG_LOG("\tFirst touch after the kickoff: " << kickoffsFor << ":" << kickoffsAgainst << " (" << (100 * kickoffsFor / (kickoffsFor + kickoffsAgainst)) << "% by the checkpoint)");

	j["ratings"] = nlohmann::json::object();
	for (auto& pair : ratingSeries) {
		auto& series = pair.second;

		// A tournament spreads one fixed budget over its rounds, so a single reading still carries the noise of
		// its goal share. The average leaves out the first quarter of the rounds, in which the rating still moves from its starting value to its level against these opponents.
		size_t avgStart = series.size() / 4;
		double avg = 0;
		for (size_t i = avgStart; i < series.size(); i++)
			avg += series[i];
		avg /= RS_MAX(series.size() - avgStart, (size_t)1);

		double start = startRatings.contains(pair.first) ? startRatings[pair.first] : tracker.initialRating;
		RG_LOG("\tRating/" << pair.first << ": " << avg << " on average over the last " << (series.size() - avgStart) << " rounds (start " << start << ", after the last round " << series.back() << ")");

		j["ratings"][pair.first] = { { "average", avg }, { "average_rounds", series.size() - avgStart }, { "start", start }, { "end", series.back() }, { "series", series } };
	}

	if (!ratingSeries.empty()) {
		double avgAll = 0, startSum = 0;
		for (auto& pair : ratingSeries) {
			avgAll += j["ratings"][pair.first]["average"].get<double>();
			startSum += j["ratings"][pair.first]["start"].get<double>();
		}
		avgAll /= ratingSeries.size();
		RG_LOG("\tRating/Ranking: " << avgAll);
		j["ratings"]["Ranking"] = { { "average", avgAll }, { "start", startSum / ratingSeries.size() },
			{ "end", rankingHistory.back() }, { "series", rankingHistory },
			{ "average_rounds", rankingHistory.size() - rankingHistory.size() / 4 } };
		j["ratings"]["Average"] = { { "average", avgAll } };
	}

	j["opponents"] = nlohmann::json::array();
	RG_LOG("\tPer opponent (timesteps of the version: goals of the checkpoint : goals of the version, rating of the version):");
	for (auto& pair : opponents) {
		std::stringstream ratingsStr;
		for (auto& ratingPair : pair.second.ratings)
			ratingsStr << ", " << ratingPair.first << " " << ratingPair.second;
		RG_LOG("\t\t" << pair.first << ": " << pair.second.goalsFor << ":" << pair.second.goalsAgainst << ratingsStr.str());

		j["opponents"].push_back({
			{ "timesteps", pair.first }, { "goals_for", pair.second.goalsFor }, { "goals_against", pair.second.goalsAgainst },
			{ "rounds", pair.second.rounds }, { "ratings", pair.second.ratings }
		});
	}

	if (!opt.Str("rate-out").empty()) {
		std::ofstream outFile(opt.Str("rate-out"));
		if (!outFile.good())
			RG_ERR_CLOSE("Can't write the rating to \"" << opt.Str("rate-out") << "\"");
		outFile << j.dump(1, '\t') << std::endl;
		RG_LOG("Result written to \"" << opt.Str("rate-out") << "\"");
	}

	std::cout << std::flush;
	return EXIT_SUCCESS;
}

int RunTraining(const Options& opt, std::filesystem::path exeDir) {
	InitRocketSim(opt, exeDir);
	AddToPythonPath(exeDir);

	// Which bot. If training continues from a checkpoint, it is the one that the checkpoint was made with
	SelectBot(opt, FindNewestCheckpoint(opt.Str("checkpoints")));
	std::string balance = opt.Str("mode-balance");
	if (balance != "games" && balance != "players")
		RG_ERR_CLOSE("--mode-balance must be games or players");
	SetTrainModes(ParseModes(opt.Str("modes"), "modes"), balance == "players");

	// Make configuration for the learner
	LearnerConfig cfg = {};

	// The environment
	g_RandomStateProb = RS_CLAMP((float)opt.Num("random-state-prob"), 0.f, 1.f);
	g_TacticalStateProb = RS_CLAMP((float)opt.Num("tactical-state-prob"), 0.f, 1.f);
	if (g_RandomStateProb + g_TacticalStateProb > 1.f)
		RG_ERR_CLOSE("--random-state-prob + --tactical-state-prob cannot exceed 1");
	g_FuzzedKickoffs = opt.Has("fuzzed-kickoffs");
	g_RandomBallSpeed = RS_CLAMP((float)opt.Num("random-ball-speed"), 0.f, 6000.f);
	g_RandomCarSpeed = opt.Has("random-car-speed");
	g_RandomCarsInAir = opt.Has("random-cars-in-air");
	if (g_RandomStateProb > 0 && g_RandomBallSpeed <= 0)
		RG_LOG("WARNING: --random-ball-speed 0 leaves the ball of every random state hanging in the air until a car touches it");
	g_NoTouchTimeout = RS_MAX((float)opt.Num("no-touch-timeout"), 0.f);
	ParseRewardWeights(opt.Str("rewards"));

	// Saved with every checkpoint, so that --eval and --rlbot know what they are loading
	cfg.checkpointInfo["model"] = g_Bot.name;
	cfg.checkpointInfo["modes"] = ModeNames(g_TrainTeamSizes);
	cfg.checkpointInfo["obs"] = g_Bot.multiMode ? "MultiModeObs" : "AdvancedObs";
	cfg.checkpointInfo["rewards"] = GetRewardSummary();

	std::string device = opt.Str("device");
	if (device == "auto") {
		cfg.deviceType = LearnerDeviceType::AUTO; // Uses your CUDA GPU if there is one
	} else if (device == "cpu") {
		cfg.deviceType = LearnerDeviceType::CPU;
	} else if (device == "cuda" || device == "gpu") {
		cfg.deviceType = LearnerDeviceType::GPU_CUDA;
	} else {
		RG_ERR_CLOSE("Unknown device \"" << device << "\", expected auto, cpu or cuda");
	}

	cfg.tickSkip = TICK_SKIP;
	cfg.actionDelay = ACTION_DELAY;

	// Play around with this to see what the optimal is for your machine, more games will consume more RAM
	cfg.numGames = opt.Has("render") ? 1 : (int)opt.Int("games");

	// The random seed can have a strong effect on the outcome of a run
	cfg.randomSeed = opt.Int("seed");

	g_LeanMetrics = opt.Has("lean-metrics");
	g_PlayerMetrics = opt.Has("player-metrics");
	cfg.addRewardsToMetrics = !g_LeanMetrics || opt.Has("reward-metrics");
	cfg.ppo.fixedRollout = opt.Has("fixed-rollout");
	cfg.ppo.tsPerItr = opt.Int("ts-per-itr");
	cfg.ppo.batchSize = opt.Int("batch-size");
	if (cfg.ppo.batchSize <= 0)
		cfg.ppo.batchSize = cfg.ppo.tsPerItr;
	cfg.ppo.miniBatchSize = opt.Int("minibatch-size"); // Lower this if too much VRAM is being allocated
	cfg.learnThreads = (int)opt.Int("threads");
	cfg.collectionInferThreads = (int)opt.Int("collection-infer-threads");

	if (cfg.ppo.batchSize > cfg.ppo.tsPerItr)
		RG_ERR_CLOSE("--batch-size can't be bigger than --ts-per-itr (no learning would happen)");

	// Using 2 epochs seems pretty optimal when comparing time training to skill
	// Perhaps 1 or 3 is better for you, test and find out!
	cfg.ppo.epochs = (int)opt.Int("epochs");

	// This scales differently than "ent_coef" in other frameworks
	// This is the scale for normalized entropy, which means you won't have to change it if you add more actions
	cfg.ppo.entropyScale = (float)opt.Num("entropy-scale");
	cfg.ppo.maskEntropy = opt.Has("mask-entropy");

	{
		std::string normAdv = opt.Str("norm-advantages");
		if (normAdv == "off") {
			cfg.ppo.normalizeAdvantages = AdvantageNormMode::OFF;
		} else if (normAdv == "all") {
			cfg.ppo.normalizeAdvantages = AdvantageNormMode::ALL;
		} else if (normAdv == "mode") {
			cfg.ppo.normalizeAdvantages = AdvantageNormMode::PER_MODE;
		} else {
			RG_ERR_CLOSE("Option --norm-advantages: expected off, all or mode, got \"" << normAdv << "\"");
		}
	}
	cfg.ppo.targetKL = RS_MAX((float)opt.Num("target-kl"), 0.f);

	// Rate of reward decay
	// Starting low tends to work out
	cfg.ppo.gaeGamma = (float)opt.Num("gamma");
	cfg.ppo.gaeLambda = RS_CLAMP((float)opt.Num("gae-lambda"), 0.f, 1.f);
	cfg.maxReturnStatCount = RS_MAX(opt.Int("return-stat-max-count"), (int64_t)0);

	// Good learning rate to start
	cfg.ppo.policyLR = (float)opt.Num("lr");
	float criticLR = (float)opt.Num("critic-lr");
	cfg.ppo.criticLR = (criticLR > 0.f) ? criticLR : cfg.ppo.policyLR;

	SetModelConfig(cfg.ppo);

	cfg.checkpointFolder = opt.Str("checkpoints");
	cfg.tsPerSave = opt.Int("ts-per-save");
	cfg.checkpointsToKeep = (int)opt.Int("keep-checkpoints");

	cfg.sendMetrics = !opt.Has("no-metrics"); // Send metrics
	cfg.metricsProjectName = OptionOrEnv(opt.Str("wandb-project"), "WANDB_PROJECT", "gigalearncpp");
	cfg.metricsGroupName = OptionOrEnv(opt.Str("wandb-group"), "WANDB_RUN_GROUP", "unnamed-runs");
	cfg.metricsRunName = opt.Str("run-name");
	cfg.renderMode = opt.Has("render");
	g_RenderOnly = cfg.renderMode;
	if (cfg.renderMode) {
		cfg.ppo.deterministic = true;
		cfg.renderChooseTeamSize = ChooseRenderTeamSize;
		g_RenderModeRng.seed(static_cast<uint32_t>(cfg.randomSeed));
	}

	// GigaLearn's policy versions: copies of the policy, saved every tsPerVersion timesteps into <checkpoints>/policy_versions.
	// The skill tracker lets the current policy play against them and keeps an Elo rating from the goals,
	//	and training itself can use them as opponents.
	cfg.skillTracker.enabled = opt.Has("skill-tracker");
	cfg.skillTracker.updateInterval = (int)RS_MAX(opt.Int("skill-interval"), (int64_t)1);
	cfg.skillTracker.updateIntervalMinutes = (float)RS_MAX(opt.Num("skill-minutes"), 0.0);
	cfg.skillTracker.sessionRounds = (int)RS_MAX(opt.Int("skill-rounds"), (int64_t)1);
	cfg.skillTracker.numArenas = (int)RS_MAX(opt.Int("skill-arenas"), (int64_t)1);
	cfg.skillTracker.simTime = (float)opt.Num("skill-sim-time");
	cfg.skillTracker.maxSimTime = (float)opt.Num("skill-max-sim-time");
	cfg.skillTracker.tournamentEloBudget = (float)RS_MAX(opt.Num("skill-tournament-elo-budget"), 0.0);
	cfg.skillTracker.initialRating = (float)opt.Num("skill-initial-rating");
	cfg.skillTracker.deterministic = opt.Has("skill-deterministic");

	cfg.tsPerVersion = RS_MAX(opt.Int("ts-per-version"), (int64_t)1);
	cfg.maxOldVersions = (int)RS_MAX(opt.Int("max-versions"), (int64_t)1);
	cfg.trainAgainstOldChance = RS_CLAMP((float)opt.Num("train-vs-old"), 0.f, 1.f);
	cfg.trainAgainstOldVersions = cfg.trainAgainstOldChance > 0;
	cfg.pfspOpponents = opt.Has("pfsp");

	cfg.mirrorX = ResolveMirrorX(opt, "mirror", FindNewestCheckpoint(opt.Str("checkpoints")));
	cfg.mirrorXFromCheckpoint = false; // Already looked up, and the option can change it
	if (cfg.mirrorX && !g_Bot.multiMode)
		RG_ERR_CLOSE("Option --mirror: the mirrored view needs a multi-mode model (--model s, m, l or xl)");

	// Rating only: no training, just the skill tracker's matches against saved policy versions (see RunRating())
	int rateRounds = (int)opt.Int("rate-rounds");
	cfg.skillTracker.anchored = opt.Has("rate-anchored");
	if (cfg.skillTracker.anchored && (rateRounds <= 0 || opt.Str("rate-versions").empty()))
		RG_ERR_CLOSE("--rate-anchored needs --rate-rounds and --rate-versions");
	if (rateRounds > 0) {
		cfg.skillTracker.enabled = true;
		cfg.trainAgainstOldVersions = false;
		cfg.sendMetrics = false;
		if (!opt.Str("rate-versions").empty()) {
			cfg.policyVersionsFolder = opt.Str("rate-versions");
			if (!std::filesystem::is_directory(cfg.policyVersionsFolder))
				RG_ERR_CLOSE("--rate-versions: " << cfg.policyVersionsFolder << " is not a folder");
		}
	}

	bool usesVersions = cfg.skillTracker.enabled || cfg.trainAgainstOldVersions;

	// Settings of the Python metrics receiver (python_scripts/metric_receiver.py), it reads them from the environment
	if (cfg.sendMetrics && !cfg.renderMode) {
		if (opt.Has("wandb"))
			SetEnv("GGL_WANDB", "1");
		if (!opt.Str("wandb-entity").empty())
			SetEnv("WANDB_ENTITY", opt.Str("wandb-entity"));
		if (opt.Has("new-run"))
			SetEnv("GGL_NEW_RUN", "1");

		// The settings of the run, shown as the run config in wandb
		nlohmann::json j = {};
		j["model"] = g_Bot.name;
		j["obs"] = g_Bot.multiMode ? "MultiModeObs" : "AdvancedObs";
		j["obs_size"] = GetObsSize();
		j["modes"] = ModeNames(g_TrainTeamSizes);
		j["device"] = opt.Str("device");
		j["games"] = cfg.numGames;
		j["mode_balance"] = balance;
		j["fixed_rollout"] = cfg.ppo.fixedRollout;
		j["lean_metrics"] = g_LeanMetrics;
		j["ts_per_itr"] = cfg.ppo.tsPerItr;
		j["batch_size"] = cfg.ppo.batchSize;
		j["epochs"] = cfg.ppo.epochs;
		j["lr"] = cfg.ppo.policyLR;
		j["critic_lr"] = cfg.ppo.criticLR;
		j["entropy_scale"] = cfg.ppo.entropyScale;
		j["mask_entropy"] = cfg.ppo.maskEntropy;
		j["norm_advantages"] = opt.Str("norm-advantages");
		j["target_kl"] = cfg.ppo.targetKL;
		j["gamma"] = cfg.ppo.gaeGamma;
		j["gae_lambda"] = cfg.ppo.gaeLambda;
		j["random_state_prob"] = g_RandomStateProb;
		j["tactical_state_prob"] = g_TacticalStateProb;
		j["fuzzed_kickoffs"] = g_FuzzedKickoffs;
		j["pfsp"] = cfg.pfspOpponents;
		if (g_RandomStateProb > 0) {
			j["random_ball_speed"] = g_RandomBallSpeed;
			j["random_car_speed"] = g_RandomCarSpeed;
			j["random_cars_in_air"] = g_RandomCarsInAir;
		}
		j["no_touch_timeout"] = g_NoTouchTimeout;
		j["seed"] = cfg.randomSeed;
		j["tick_skip"] = TICK_SKIP;
		j["action_delay"] = ACTION_DELAY;
		j["ts_per_save"] = cfg.tsPerSave;
		j["rewards"] = GetRewardSummary();
		j["reward_changes"] = opt.Str("rewards");
		j["reward_preset"] = g_RewardPreset;
		for (auto& def : REWARD_DEFS)
			j["reward_weights"][def.name] = GetRewardWeight(def);
		j["skill_tracker"] = cfg.skillTracker.enabled;
		if (cfg.skillTracker.enabled) {
			j["skill_interval"] = cfg.skillTracker.updateInterval;
			j["skill_minutes"] = cfg.skillTracker.updateIntervalMinutes;
			j["skill_rounds"] = cfg.skillTracker.sessionRounds;
			j["skill_arenas"] = cfg.skillTracker.numArenas;
			j["skill_sim_time"] = cfg.skillTracker.simTime;
			j["skill_max_sim_time"] = cfg.skillTracker.maxSimTime;
			j["skill_tournament_elo_budget"] = cfg.skillTracker.tournamentEloBudget;
			j["skill_initial_rating"] = cfg.skillTracker.initialRating;
			j["skill_deterministic"] = cfg.skillTracker.deterministic;
		}
		if (usesVersions) {
			j["ts_per_version"] = cfg.tsPerVersion;
			j["max_versions"] = cfg.maxOldVersions;
		}
		j["train_vs_old"] = cfg.trainAgainstOldChance;
		j["mirror"] = MirrorName(cfg.mirrorX);
		j["checkpoints"] = cfg.checkpointFolder.string();
		{
			auto startCheckpoint = FindNewestCheckpoint(cfg.checkpointFolder);
			j["start_checkpoint"] = startCheckpoint.empty() ? "" : startCheckpoint.filename().string();
		}
		SetEnv("GGL_RUN_CONFIG", j.dump());
	}

	g_MaxTimesteps = (uint64_t)RS_MAX(opt.Int("max-timesteps"), (int64_t)0);
	cfg.maxTimesteps = g_MaxTimesteps;
	g_MaxTrainSeconds = opt.Num("max-minutes") * 60;

	RG_LOG("Training settings:");
	RG_LOG("\tModel: " << g_Bot.name << " (" << g_Bot.description << "), " << GetObsSize() << " inputs");
	{
		std::stringstream modeStr;
		for (int teamSize : g_TrainTeamSizes) {
			int numGames = 0;
			for (int i = 0; i < cfg.numGames; i++)
				numGames += (GetTeamSizeOfGame(i) == teamSize);
			modeStr << (modeStr.str().empty() ? "" : ", ") << numGames << " x " << ModeName(teamSize) << " (" << (numGames * teamSize * 2) << " players)";
		}
		if (cfg.renderMode) {
			RG_LOG("\tRender modes: " << ModeNames(g_TrainTeamSizes) << "; one arena, random mode each episode, deterministic argmax");
		} else {
			RG_LOG("\tGame modes: " << modeStr.str());
		}
	}
	RG_LOG("\tGames: " << cfg.numGames << ", timesteps per iteration: " << cfg.ppo.tsPerItr << ", batch size: " << cfg.ppo.batchSize << ", epochs: " << cfg.ppo.epochs);
	RG_LOG("\tLearning rate: " << cfg.ppo.policyLR << ", entropy scale: " << cfg.ppo.entropyScale << ", gamma: " << cfg.ppo.gaeGamma);
	RG_LOG("\tRandom state portion: " << g_RandomStateProb << ", no-touch timeout: " << g_NoTouchTimeout << "s, seed: " << cfg.randomSeed);
	if (g_RandomStateProb > 0) {
		RG_LOG(
			"\tRandom states: ball speed up to " << g_RandomBallSpeed << ", cars " << (g_RandomCarSpeed ? "moving" : "at rest") <<
			", " << (g_RandomCarsInAir ? "half of them in the air" : "on the ground"));
	}
	RG_LOG("\tRewards" << (g_RewardPreset.empty() ? "" : " (preset " + g_RewardPreset + ")") << ": " << GetRewardSummary());
	bool ratingSessions = cfg.skillTracker.enabled && cfg.skillTracker.updateIntervalMinutes > 0;
	if (cfg.skillTracker.enabled) {
		std::string when = ratingSessions ?
			RS_STR("a session of " << cfg.skillTracker.sessionRounds << " rounds every " << cfg.skillTracker.updateIntervalMinutes << " minutes") :
			RS_STR("every " << cfg.skillTracker.updateInterval << " iterations");
		RG_LOG(
			"\tSkill tracker: " << when << ", " << cfg.skillTracker.numArenas << " games of " <<
			cfg.skillTracker.simTime << "s, at most " << cfg.skillTracker.tournamentEloBudget << " rating points per tournament, " <<
			(cfg.skillTracker.deterministic ? "most likely" : "sampled") << " actions"
		);
	}
	if (usesVersions) {
		std::string when = ratingSessions ?
			RS_STR("one after a rating session, every " << cfg.tsPerVersion << " timesteps or more") :
			RS_STR("one per " << cfg.tsPerVersion << " timesteps");
		RG_LOG("\tPolicy versions: " << when << ", " << cfg.maxOldVersions << " kept, training against them in " << (cfg.trainAgainstOldChance * 100) << "% of the iterations");
	}
	if (g_MaxTimesteps > 0)
		RG_LOG("\tTimestep limit: " << g_MaxTimesteps);
	if (g_MaxTrainSeconds > 0)
		RG_LOG("\tTime limit: " << (g_MaxTrainSeconds / 60) << " minutes");

	// Make the learner with the environment creation function and the config we just made
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);
	learner->shouldAbortRating = []() { return g_StopSignals > 0; };

	if (rateRounds > 0) {
		if (learner->totalTimesteps == 0)
			RG_ERR_CLOSE("--rate-rounds: there is no checkpoint to rate in " << cfg.checkpointFolder);
		std::signal(SIGINT, OnStopSignal);
		std::signal(SIGTERM, OnStopSignal);
		return RunRating(learner, opt, rateRounds);
	}

	if (g_MaxTimesteps > 0 && learner->totalTimesteps >= g_MaxTimesteps) {
		RG_LOG("The newest checkpoint already has " << learner->totalTimesteps << " timesteps, which reaches the timestep limit. Nothing to do.");
		return EXIT_SUCCESS;
	}

	// NOTE: Must be done after the learner is created, as starting Python replaces the interrupt handler
	if (cfg.renderMode) {
		// Render mode runs forever and has nothing to save, so Ctrl+C should just end the program
		std::signal(SIGINT, SIG_DFL);
	} else {
		std::signal(SIGINT, OnStopSignal);
		std::signal(SIGTERM, OnStopSignal);
	}

	uint64_t startTimesteps = learner->totalTimesteps;
	g_TrainStartTime = std::chrono::steady_clock::now();

	// Start learning!
	learner->Start();

	// Upload the remaining metrics and close the run (wandb would otherwise show it as crashed)
	if (learner->metricSender)
		learner->metricSender->Finish();

	double seconds = GetTrainSeconds();
	uint64_t trainedTimesteps = learner->totalTimesteps - startTimesteps;
	RG_LOG(RG_DIVIDER);
	RG_LOG("Training stopped.");
	RG_LOG("\tThis session: " << trainedTimesteps << " timesteps in " << (seconds / 60) << " minutes (" << (int64_t)(trainedTimesteps / RS_MAX(seconds, 1e-3)) << " steps/second)");
	RG_LOG("\tTotal: " << learner->totalTimesteps << " timesteps, " << learner->totalIterations << " iterations");
	RG_LOG("\tNewest checkpoint: " << (cfg.checkpointFolder / std::to_string(learner->totalTimesteps)));

	return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Evaluation: plays a checkpoint (A) against random actions or another checkpoint (B)
///////////////////////////////////////////////////////////////////////////////////////////////

struct EvalSide {
	std::string name;
	uint64_t goals = 0, touches = 0, kickoffFirstTouches = 0, samples = 0, airSamples = 0;
	double speedSum = 0, ballDistSum = 0, boostSum = 0;
};

struct EvalArena {
	Arena* arena = NULL;
	GameState state = {}, prevState = {};
	std::vector<Action> actions = {};
	Team teamA = Team::BLUE;
	float timeSinceTouch = 0, timeSinceKickoff = 0;
	bool ballTouchedSinceKickoff = false;
	bool needsReset = true;
	bool active = true; // false: this game is over and no new one is started (see finishGames)
};

struct EvalResult {
	int teamSize = 1;
	int numArenas = 0;
	EvalSide sideA = {}, sideB = {};
	uint64_t numEpisodes = 0, numTimeouts = 0, numMaxTimeEnds = 0;
	double simMinutes = 0, realSeconds = 0;
};

// Loads the policy of a checkpoint, with the observation and the network of the model that the checkpoint was made with.
// g_Bot is that model afterwards.
// mirrorOption: name of the option that says whether the policy plays with the mirrored view (default: as it was trained)
InferUnit* MakeInferUnit(const Options& opt, std::filesystem::path checkpointPath, bool ignoreModelOption = false, const char* mirrorOption = NULL) {
	SelectBot(opt, checkpointPath, ignoreModelOption);

	PPOLearnerConfig ppoConfig = {};
	SetModelConfig(ppoConfig);
	InferUnit* unit = new InferUnit(MakeObsBuilder(), GetObsSize(), MakeActionParser(), ppoConfig.sharedHead, ppoConfig.policy, checkpointPath, false);

	int mirrorX = mirrorOption ? ResolveMirrorX(opt, mirrorOption, checkpointPath) : ReadMirrorX(checkpointPath);
	if (mirrorX) {
		if (!g_Bot.multiMode)
			RG_ERR_CLOSE("The mirrored view needs a multi-mode model, " << checkpointPath << " is not one");
		unit->SetMirrorX(mirrorX);
		RG_LOG("Mirrored view: " << MirrorName(mirrorX) << " for " << checkpointPath);
	}
	return unit;
}

// Plays one game mode. unitB = NULL means random actions.
// Every game starts with a kickoff and ends with a goal, or without a result after noTouchTimeout seconds without
//	a ball touch or maxSeconds seconds of play (0 = that limit is off).
// finishGames: once the game time is up no new kickoffs are started, but the running ones are played to their end.
//	Without it they are dropped, which wastes their time and never lets a game count that is longer than
//	the time each arena gets.
EvalResult EvaluateMode(
	int teamSize, InferUnit* unitA, InferUnit* unitB, ActionParser* actionParser,
	int numArenas, double gameMinutes, bool deterministic, float noTouchTimeout, float maxSeconds, bool finishGames, std::mt19937& randEngine) {

	if (noTouchTimeout <= 0 && maxSeconds <= 0)
		finishGames = false; // Games might never end

	EvalResult result = {};
	result.teamSize = teamSize;
	result.numArenas = numArenas;
	EvalSide& sideA = result.sideA;
	EvalSide& sideB = result.sideB;

	std::vector<EvalArena> arenas(numArenas);
	for (int i = 0; i < numArenas; i++) {
		arenas[i].arena = MakeArena(teamSize);
		arenas[i].teamA = (i % 2 == 0) ? Team::BLUE : Team::ORANGE;
	}

	const double stepsPerSecond = 120.0 / TICK_SKIP;
	uint64_t targetGameSteps = (uint64_t)(gameMinutes * 60 * stepsPerSecond);
	uint64_t gameSteps = 0;

	auto startTime = std::chrono::steady_clock::now();

	int numActive = numArenas;
	while (numActive > 0) {
		bool timeUp = (gameSteps >= targetGameSteps);
		if (timeUp && !finishGames)
			break;

		// Reset finished games back to a kickoff
		for (auto& ea : arenas) {
			if (!ea.needsReset || !ea.active)
				continue;

			if (timeUp) {
				ea.active = false;
				numActive--;
				continue;
			}

			ea.arena->ResetToRandomKickoff((int)(randEngine() & 0x7FFFFFFF));

			// Move the cars a tiny bit (up to 1 of the ~118 units a car is long).
			// Otherwise bots that always pick their most likely action would repeat the same few games.
			for (auto car : ea.arena->_cars) {
				auto carState = car->GetState();
				for (int axis = 0; axis < 2; axis++)
					carState.pos[axis] += std::uniform_real_distribution<float>(-1, 1)(randEngine);
				car->SetState(carState);
			}

			ea.state = GameState(ea.arena);
			ea.prevState.MakeEmpty();
			ea.timeSinceTouch = 0;
			ea.timeSinceKickoff = 0;
			ea.ballTouchedSinceKickoff = false;
			ea.needsReset = false;
			result.numEpisodes++;
		}

		// Collect the players of both sides
		std::vector<Player> players[2];
		std::vector<GameState> states[2];
		std::vector<std::pair<int, int>> slots[2]; // (arena index, player index)
		if (numActive == 0)
			break;

		for (int i = 0; i < numArenas; i++) {
			if (!arenas[i].active)
				continue;

			auto& state = arenas[i].state;
			arenas[i].actions.assign(state.players.size(), Action(0, 0, 0, 0, 0, 0, 0, 0));
			for (int j = 0; j < state.players.size(); j++) {
				int side = (state.players[j].team == arenas[i].teamA) ? 0 : 1;
				players[side].push_back(state.players[j]);
				states[side].push_back(state);
				slots[side].push_back({ i, j });
			}
		}

		// Choose the actions of both sides
		for (int side = 0; side < 2; side++) {
			InferUnit* unit = (side == 0) ? unitA : unitB;

			std::vector<Action> actions;
			if (unit) {
				actions = unit->BatchInferActions(players[side], states[side], deterministic);
			} else {
				// Random (but valid) actions
				for (int k = 0; k < players[side].size(); k++) {
					auto mask = actionParser->GetActionMask(players[side][k], states[side][k]);
					std::vector<int> validActions = {};
					for (int a = 0; a < mask.size(); a++)
						if (mask[a])
							validActions.push_back(a);
					RG_ASSERT(!validActions.empty());

					int actionIdx = validActions[randEngine() % validActions.size()];
					actions.push_back(actionParser->ParseAction(actionIdx, players[side][k], states[side][k]));
				}
			}

			for (int k = 0; k < actions.size(); k++)
				arenas[slots[side][k].first].actions[slots[side][k].second] = actions[k];
		}

		// Step the games, exactly like the training environment does
		auto fnStepArena = [&](int arenaIdx) {
			auto& ea = arenas[arenaIdx];
			if (!ea.active)
				return;

			ea.prevState = ea.state;
			ea.state.ResetBeforeStep();

			// The previous actions stay active for the action delay
			ea.arena->Step(ACTION_DELAY);

			auto carItr = ea.arena->_cars.begin();
			for (int j = 0; j < ea.actions.size(); j++, carItr++)
				(*carItr)->controls = (CarControls)ea.actions[j];

			ea.arena->Step(TICK_SKIP - ACTION_DELAY);
			ea.state.UpdateFromArena(ea.arena, ea.actions, &ea.prevState);
		};
		g_ThreadPool.StartBatchedJobs(fnStepArena, numArenas, false);

		// Update the statistics, and check if games are over
		for (auto& ea : arenas) {
			if (!ea.active)
				continue;

			auto& state = ea.state;

			bool touched[2] = {};
			for (auto& player : state.players) {
				int side = (player.team == ea.teamA) ? 0 : 1;
				EvalSide& stats = (side == 0) ? sideA : sideB;

				stats.samples++;
				stats.airSamples += !player.isOnGround;
				stats.speedSum += player.vel.Length();
				stats.ballDistSum += player.pos.Dist(state.ball.pos);
				stats.boostSum += player.boost;

				if (player.ballTouchedStep) {
					stats.touches++;
					touched[side] = true;
				}
			}

			if (touched[0] || touched[1]) {
				if (!ea.ballTouchedSinceKickoff) {
					ea.ballTouchedSinceKickoff = true;
					sideA.kickoffFirstTouches += touched[0];
					sideB.kickoffFirstTouches += touched[1];
				}
				ea.timeSinceTouch = 0;
			} else {
				ea.timeSinceTouch += state.deltaTime;
			}

			ea.timeSinceKickoff += state.deltaTime;

			if (state.goalScored) {
				// The ball is in the goal of the team that got scored on
				Team scoredOnTeam = RS_TEAM_FROM_Y(state.ball.pos.y);
				((scoredOnTeam != ea.teamA) ? sideA : sideB).goals++;
				ea.needsReset = true;
			} else if (noTouchTimeout > 0 && ea.timeSinceTouch >= noTouchTimeout) {
				result.numTimeouts++;
				ea.needsReset = true;
			} else if (maxSeconds > 0 && ea.timeSinceKickoff >= maxSeconds) {
				result.numMaxTimeEnds++;
				ea.needsReset = true;
			}
		}

		gameSteps += numActive;
	}

	for (auto& ea : arenas)
		delete ea.arena;

	result.realSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
	result.simMinutes = gameSteps / stepsPerSecond / 60;
	return result;
}

int RunEvaluation(const Options& opt, std::filesystem::path exeDir) {
	InitRocketSim(opt, exeDir);

	auto pathA = ResolveCheckpoint(opt.Str("eval"));
	std::filesystem::path pathB = {};
	if (opt.Str("vs") != "random")
		pathB = ResolveCheckpoint(opt.Str("vs"));

	bool deterministic = opt.Has("deterministic");
	float noTouchTimeout = RS_MAX((float)opt.Num("eval-no-touch-timeout"), 0.f);
	float maxSeconds = RS_MAX((float)opt.Num("eval-max-seconds"), 0.f);
	bool finishGames = opt.Has("eval-finish-games");
	int numArenas = RS_MAX((int)opt.Int("eval-arenas"), 2);
	numArenas += numArenas % 2; // Even amount, so both sides play as blue and as orange equally often
	double gameMinutes = opt.Num("eval-minutes");

	// The same seed gives the same kickoffs, the same random actions and the same sampled actions
	int64_t seed = opt.Int("seed");
	if (seed == -1)
		seed = (int64_t)RS_CUR_MS();
	std::mt19937 randEngine((uint32_t)seed);
	InferUnit::SetRandomSeed((uint64_t)seed);

	// The two sides can be different models, each one gets its own observation and network
	std::string nameA = pathA.string(), nameB = "random actions";
	std::string modelA = {}, modelB = "random";
	ActionParser* actionParser = MakeActionParser();

	InferUnit* unitB = NULL; // NULL = random actions
	bool multiModeB = true;
	if (!pathB.empty()) {
		nameB = pathB.string();
		unitB = MakeInferUnit(opt, pathB, true, "vs-mirror");
		modelB = g_Bot.name;
		multiModeB = g_Bot.multiMode && g_Bot.only1v1Of.empty();
	}

	InferUnit* unitA = MakeInferUnit(opt, pathA, false, "eval-mirror");
	modelA = g_Bot.name;

	// The game modes that both sides can play
	std::vector<int> teamSizes = ParseModes(opt.Str("eval-modes"), "eval-modes");
	if (!multiModeB) {
		if (opt.Str("eval-modes") != "auto" && teamSizes != std::vector<int>{ 1 })
			RG_ERR_CLOSE("Option --eval-modes: the model \"" << modelB << "\" of the opponent only plays 1v1");
		teamSizes = { 1 };
	}

	RG_LOG("Evaluating:");
	RG_LOG("\tA: " << nameA << " (model \"" << modelA << "\")");
	RG_LOG("\tB: " << nameB << (unitB ? (" (model \"" + modelB + "\")") : std::string()));
	RG_LOG("\t" << gameMinutes << " minutes of game time in each of " << ModeNames(teamSizes) << ", " << numArenas << " games at once, " << (deterministic ? "deterministic" : "sampled") << " actions");

	std::vector<EvalResult> results = {};
	for (int teamSize : teamSizes) {
		EvalResult result = EvaluateMode(teamSize, unitA, unitB, actionParser, numArenas, gameMinutes, deterministic, noTouchTimeout, maxSeconds, finishGames, randEngine);
		result.sideA.name = nameA;
		result.sideB.name = nameB;
		results.push_back(result);
	}

	// Show the results
	auto fnAvg = [](double sum, uint64_t count) { return count ? (sum / count) : 0; };

	printf("\n========================= EVALUATION RESULTS =========================\n");
	printf("A: %s\n", nameA.c_str());
	printf("B: %s\n", nameB.c_str());
	for (auto& result : results) {
		auto& sideA = result.sideA;
		auto& sideB = result.sideB;
		auto fnPerMinute = [&](uint64_t count) { return count / result.simMinutes; };

		printf("\n--- %s ---\n", ModeName(result.teamSize).c_str());
		printf("Simulated %.1f minutes of game time in %.1f seconds (%d games at once, sides swapped in half of them)\n", result.simMinutes, result.realSeconds, result.numArenas);
		printf("Kickoffs played: %llu", (unsigned long long)result.numEpisodes);
		if (noTouchTimeout > 0)
			printf(" (%llu ended without a goal after %.0fs with no touch)", (unsigned long long)result.numTimeouts, noTouchTimeout);
		if (maxSeconds > 0)
			printf(" (%llu ended without a goal after %.0fs of play)", (unsigned long long)result.numMaxTimeEnds, maxSeconds);
		printf("\n\n");
		printf("%-36s %14s %14s\n", (result.teamSize > 1) ? "(per team, averages per player)" : "", "A", "B");
		printf("%-36s %14llu %14llu\n", "Goals", (unsigned long long)sideA.goals, (unsigned long long)sideB.goals);
		printf("%-36s %14.2f %14.2f\n", "Goals per hour of game time", fnPerMinute(sideA.goals) * 60, fnPerMinute(sideB.goals) * 60);
		printf("%-36s %14llu %14llu\n", "Ball touches", (unsigned long long)sideA.touches, (unsigned long long)sideB.touches);
		printf("%-36s %14.2f %14.2f\n", "Ball touches per minute", fnPerMinute(sideA.touches), fnPerMinute(sideB.touches));
		printf("%-36s %14llu %14llu\n", "First touch after kickoff", (unsigned long long)sideA.kickoffFirstTouches, (unsigned long long)sideB.kickoffFirstTouches);
		printf("%-36s %14.0f %14.0f\n", "Average distance to ball (uu)", fnAvg(sideA.ballDistSum, sideA.samples), fnAvg(sideB.ballDistSum, sideB.samples));
		printf("%-36s %14.0f %14.0f\n", "Average speed (uu/s)", fnAvg(sideA.speedSum, sideA.samples), fnAvg(sideB.speedSum, sideB.samples));
		printf("%-36s %14.1f %14.1f\n", "Average boost", fnAvg(sideA.boostSum, sideA.samples), fnAvg(sideB.boostSum, sideB.samples));
		printf("%-36s %13.1f%% %13.1f%%\n", "Time in the air", fnAvg(sideA.airSamples, sideA.samples) * 100, fnAvg(sideB.airSamples, sideB.samples) * 100);
	}
	printf("======================================================================\n");
	fflush(stdout);

	if (!opt.Str("eval-out").empty()) {
		using nlohmann::json;

		auto fnResultJSON = [&](const EvalResult& result) {
			auto fnSideJSON = [&](const EvalSide& side) {
				json j = {};
				j["name"] = side.name;
				j["goals"] = side.goals;
				j["ball_touches"] = side.touches;
				j["ball_touches_per_minute"] = side.touches / result.simMinutes;
				j["kickoff_first_touches"] = side.kickoffFirstTouches;
				j["avg_ball_distance"] = fnAvg(side.ballDistSum, side.samples);
				j["avg_speed"] = fnAvg(side.speedSum, side.samples);
				j["avg_boost"] = fnAvg(side.boostSum, side.samples);
				j["air_ratio"] = fnAvg(side.airSamples, side.samples);
				return j;
			};

			json j = {};
			j["mode"] = ModeName(result.teamSize);
			j["a"] = fnSideJSON(result.sideA);
			j["b"] = fnSideJSON(result.sideB);
			j["game_minutes"] = result.simMinutes;
			j["a"]["model"] = modelA;
			j["b"]["model"] = modelB;
			j["kickoffs"] = result.numEpisodes;
			j["no_touch_timeouts"] = result.numTimeouts;
			j["max_seconds_ends"] = result.numMaxTimeEnds;
			j["no_touch_timeout"] = noTouchTimeout;
			j["max_seconds"] = maxSeconds;
			j["real_seconds"] = result.realSeconds;
			j["deterministic"] = deterministic;
			j["seed"] = seed;
			return j;
		};

		// One game mode: the result itself. Several: one result per mode
		json j = {};
		if (results.size() == 1) {
			j = fnResultJSON(results[0]);
		} else {
			j["a_name"] = nameA;
			j["b_name"] = nameB;
			for (auto& result : results)
				j["modes"][ModeName(result.teamSize)] = fnResultJSON(result);
		}

		std::ofstream outFile(opt.Str("eval-out"));
		if (!outFile.good())
			RG_ERR_CLOSE("Can't write the evaluation results to \"" << opt.Str("eval-out") << "\"");
		outFile << j.dump(4) << std::endl;
		RG_LOG("Results written to \"" << opt.Str("eval-out") << "\"");
	}

	return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////
// RLBot: runs a checkpoint in the real game
// NOTE: You need the RLBot framework and Rocket League for this, see rlbot/README.md
///////////////////////////////////////////////////////////////////////////////////////////////

int RunRLBot(const Options& opt) {
	auto path = ResolveCheckpoint(opt.Str("rlbot"));
	InferUnit* inferUnit = MakeInferUnit(opt, path);

	RG_LOG("Starting RLBot client with checkpoint " << path << " (model \"" << g_Bot.name << "\")...");

	RLBotParams params = {};
	params.port = (int)opt.Int("rlbot-port");
	params.tickSkip = TICK_SKIP;
	params.actionDelay = ACTION_DELAY;
	params.inferUnit = inferUnit;

	RLBotClient::Run(params);
	return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////

// --expand-1v1: makes a checkpoint of the full model out of a checkpoint of its 1v1-only version (with the policy versions)
int RunExpand1v1(const Options& opt) {
	std::filesystem::path srcFolder = opt.Str("expand-1v1"), dstFolder = opt.Str("checkpoints");
	std::filesystem::path srcCheckpoint = FindNewestCheckpoint(srcFolder);
	if (srcCheckpoint.empty())
		RG_ERR_CLOSE("--expand-1v1: there is no checkpoint in " << srcFolder);
	if (!FindNewestCheckpoint(dstFolder).empty())
		RG_ERR_CLOSE("--expand-1v1: " << dstFolder << " (--checkpoints) already has a checkpoint, give an empty folder");

	auto info = ReadCheckpointInfo(srcCheckpoint);
	BotSpec srcBot = FindBotPreset(info.count("model") ? info["model"] : "example");
	if (srcBot.only1v1Of.empty())
		RG_ERR_CLOSE("--expand-1v1: " << srcCheckpoint << " is of the model \"" << srcBot.name << "\", which is not a 1v1-only model");
	g_Bot = FindBotPreset(srcBot.only1v1Of);

	PPOLearnerConfig ppoConfig = {};
	SetModelConfig(ppoConfig);
	int numActions = MakeActionParser()->GetActionAmount();

	// The inputs of the other game modes were always zero, and still have the random weights the model started with.
	// They get the weights of the 1v1 input: the bot then starts every mode with what it does in 1v1
	std::vector<std::pair<int, int>> inputCopies = {};
	for (int modeIdx = 1; modeIdx < MultiModeObs::NUM_MODE_INPUTS; modeIdx++)
		inputCopies.push_back({ MultiModeObs::NUM_PAD_INPUTS, MultiModeObs::NUM_PAD_INPUTS + modeIdx });

	auto fnCopyOtherFiles = [](std::filesystem::path src, std::filesystem::path dst) {
		for (auto& entry : std::filesystem::directory_iterator(src))
			if (entry.is_regular_file() && entry.path().extension() == ".json")
				std::filesystem::copy_file(entry.path(), dst / entry.path().filename(), std::filesystem::copy_options::overwrite_existing);
	};

	std::filesystem::path dstCheckpoint = dstFolder / srcCheckpoint.filename();
	ConvertModels(srcCheckpoint, dstCheckpoint, GetObsSize(), numActions, ppoConfig.sharedHead, ppoConfig.policy, ppoConfig.critic, true, inputCopies);
	fnCopyOtherFiles(srcCheckpoint, dstCheckpoint);
	{
		info["model"] = g_Bot.name;
		std::ofstream infoOut(dstCheckpoint / "INFO.json");
		infoOut << nlohmann::json(info).dump(4) << std::endl;
	}
	RG_LOG("Checkpoint " << srcCheckpoint << " (" << srcBot.name << ") -> " << dstCheckpoint << " (" << g_Bot.name << ")");

	int numVersions = 0;
	std::filesystem::path srcVersions = srcFolder / "policy_versions";
	if (std::filesystem::is_directory(srcVersions)) {
		for (auto& entry : std::filesystem::directory_iterator(srcVersions)) {
			if (!entry.is_directory())
				continue;
			std::filesystem::path dstVersion = dstFolder / "policy_versions" / entry.path().filename();
			ConvertModels(entry.path(), dstVersion, GetObsSize(), numActions, ppoConfig.sharedHead, ppoConfig.policy, ppoConfig.critic, false, inputCopies);
			fnCopyOtherFiles(entry.path(), dstVersion);
			numVersions++;
		}
	}
	RG_LOG(numVersions << " policy versions -> " << (dstFolder / "policy_versions"));
	return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
	Options opt = {};
	if (!opt.Parse(argc, argv))
		return EXIT_FAILURE;

	if (opt.Has("help")) {
		Options::PrintHelp();
		return EXIT_SUCCESS;
	}

	std::filesystem::path exeDir = GetExeDir(argv[0]);

	try {
		if (!opt.Str("expand-1v1").empty()) {
			return RunExpand1v1(opt);
		} else if (!opt.Str("eval").empty()) {
			return RunEvaluation(opt, exeDir);
		} else if (!opt.Str("rlbot").empty()) {
			return RunRLBot(opt);
		} else {
			return RunTraining(opt, exeDir);
		}
	} catch (std::exception& e) {
		// NOTE: GigaLearn/RLGym fatal errors are already logged when they are thrown
		std::string message = e.what();
		if (message.find("RG FATAL ERROR") == std::string::npos)
			std::cout << "Stopped because of an error: " << message << std::endl;
		return EXIT_FAILURE;
	}
}
