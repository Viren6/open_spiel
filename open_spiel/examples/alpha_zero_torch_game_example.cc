// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/bots/human/human_bot.h"
#include "open_spiel/games/kuhn_poker/kuhn_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, game, "tic_tac_toe", "The name of the game to play.");
ABSL_FLAG(std::string, player1, "az", "Who controls player1.");
ABSL_FLAG(std::string, player2, "random", "Who controls player2.");
ABSL_FLAG(std::string, az_path, "", "Path to AZ experiment.");
ABSL_FLAG(std::string, az_graph_def, "vpnet.pb",
          "AZ graph definition file name.");
ABSL_FLAG(double, uct_c, 2, "UCT exploration constant.");
ABSL_FLAG(int, rollout_count, 10, "How many rollouts per evaluation.");
ABSL_FLAG(int, max_simulations, 10000, "How many simulations to run.");
ABSL_FLAG(int, num_games, 1, "How many games to play.");
ABSL_FLAG(int, max_memory_mb, 1000,
          "The maximum memory used before cutting the search short.");
ABSL_FLAG(int, az_checkpoint, -1, "Checkpoint of AZ model.");
ABSL_FLAG(int, az_batch_size, 1, "Batch size of AZ inference.");
ABSL_FLAG(int, az_threads, 1, "Number of threads to run for AZ inference.");
ABSL_FLAG(int, az_cache_size, 16384, "Cache size of AZ algorithm.");
ABSL_FLAG(int, az_cache_shards, 1, "Cache shards of AZ algorithm.");
ABSL_FLAG(bool, solve, true, "Whether to use MCTS-Solver.");
ABSL_FLAG(uint_fast32_t, seed, 0, "Seed for MCTS.");
ABSL_FLAG(bool, verbose, false, "Show the MCTS stats of possible moves.");
ABSL_FLAG(bool, quiet, false, "Show the MCTS stats of possible moves.");
ABSL_FLAG(bool, az_use_pcr, false,
          "Use playout cap randomization for the AlphaZero player.");
ABSL_FLAG(double, az_p_full, 0.25,
          "Probability that AZ uses the full PCR search cap.");
ABSL_FLAG(int, az_full_simulations, 1000,
          "Max simulations for full PCR searches when using PCR.");
ABSL_FLAG(int, az_fast_simulations, 200,
          "Max simulations for fast PCR searches when using PCR.");
ABSL_FLAG(bool, az_disable_noise_on_fast, true,
          "Disable exploration noise on fast PCR searches.");

uint_fast32_t Seed() {
  uint_fast32_t seed = absl::GetFlag(FLAGS_seed);
  return seed != 0 ? seed : absl::ToUnixMicros(absl::Now());
}

namespace {

double KuhnPokerMultiplier(const open_spiel::Game& game,
                           const open_spiel::State& state) {
  if (game.GetType().short_name != "kuhn_poker") {
    return 1.0;
  }
  const std::vector<open_spiel::Action>& history = state.History();
  const int num_players = game.NumPlayers();
  if (history.size() <= num_players) {
    return 1.2;
  }
  bool bet_seen = false;
  for (int i = num_players; i < history.size(); ++i) {
    if (history[i] == open_spiel::kuhn_poker::ActionType::kBet) {
      bet_seen = true;
      break;
    }
  }
  if (!bet_seen && history.size() <= num_players + 1) {
    return 1.2;
  }
  if (bet_seen) {
    return 0.7;
  }
  return 1.0;
}

class PCRMCTSBot : public open_spiel::algorithms::MCTSBot {
 public:
  PCRMCTSBot(const open_spiel::Game& game,
             std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator,
             double uct_c, int full_simulations, int fast_simulations,
             int64_t max_memory_mb, bool solve, int seed, bool verbose,
             double policy_alpha, double policy_epsilon, double p_full,
             bool disable_noise_on_fast)
      : open_spiel::algorithms::MCTSBot(
            game, std::move(evaluator), uct_c, full_simulations,
            max_memory_mb, solve, seed, verbose,
            open_spiel::algorithms::ChildSelectionPolicy::PUCT,
            policy_alpha, policy_epsilon,
            /*dont_return_chance_node=*/true),
        game_(game),
        p_full_(p_full),
        full_simulations_(std::max(1, full_simulations)),
        fast_simulations_(std::max(1, fast_simulations)),
        disable_noise_on_fast_(disable_noise_on_fast),
        policy_alpha_(policy_alpha),
        policy_epsilon_(policy_epsilon),
        rng_(seed) {}

  open_spiel::Action Step(const open_spiel::State& state) override {
    bool force_full = p_full_ >= 1.0;
    bool force_fast = p_full_ <= 0.0;
    bool full_search = force_full ||
                       (!force_fast && dist_(rng_) < p_full_);
    if (force_fast) {
      full_search = false;
    }
    double multiplier = KuhnPokerMultiplier(game_, state);
    int full_cap = std::max(
        1, static_cast<int>(std::round(full_simulations_ * multiplier)));
    int fast_cap = std::max(
        1, static_cast<int>(std::round(fast_simulations_ * multiplier)));
    SetMaxSimulations(full_search ? full_cap : fast_cap);
    double epsilon = policy_epsilon_;
    if (!full_search && disable_noise_on_fast_) {
      epsilon = 0.0;
    }
    SetDirichletNoise(policy_alpha_, epsilon);
    return open_spiel::algorithms::MCTSBot::Step(state);
  }

 private:
  const open_spiel::Game& game_;
  double p_full_;
  int full_simulations_;
  int fast_simulations_;
  bool disable_noise_on_fast_;
  double policy_alpha_;
  double policy_epsilon_;
  std::mt19937 rng_;
  absl::uniform_real_distribution<double> dist_{0.0, 1.0};
};

}  // namespace

std::unique_ptr<open_spiel::Bot>
InitBot(std::string type, const open_spiel::Game &game,
        open_spiel::Player player,
        std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator,
        std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator>
            az_evaluator) {
  if (type == "az") {
    if (absl::GetFlag(FLAGS_az_use_pcr)) {
      return std::make_unique<PCRMCTSBot>(
          game, std::move(az_evaluator), absl::GetFlag(FLAGS_uct_c),
          absl::GetFlag(FLAGS_az_full_simulations),
          absl::GetFlag(FLAGS_az_fast_simulations),
          absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve),
          Seed(), absl::GetFlag(FLAGS_verbose),
          /*policy_alpha=*/0.0, /*policy_epsilon=*/0.0,
          absl::GetFlag(FLAGS_az_p_full),
          absl::GetFlag(FLAGS_az_disable_noise_on_fast));
    }
    return std::make_unique<open_spiel::algorithms::MCTSBot>(
        game, std::move(az_evaluator), absl::GetFlag(FLAGS_uct_c),
        absl::GetFlag(FLAGS_max_simulations),
        absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), Seed(),
        absl::GetFlag(FLAGS_verbose),
        open_spiel::algorithms::ChildSelectionPolicy::PUCT, 0, 0,
        /*dont_return_chance_node=*/true);
  }
  if (type == "human") {
    return std::make_unique<open_spiel::HumanBot>();
  }
  if (type == "mcts") {
    return std::make_unique<open_spiel::algorithms::MCTSBot>(
        game, std::move(evaluator), absl::GetFlag(FLAGS_uct_c),
        absl::GetFlag(FLAGS_max_simulations),
        absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), Seed(),
        absl::GetFlag(FLAGS_verbose));
  }
  if (type == "random") {
    return open_spiel::MakeUniformRandomBot(player, Seed());
  }

  open_spiel::SpielFatalError(
      "Bad player type. Known types: az, human, mcts, random");
}

open_spiel::Action GetAction(const open_spiel::State &state,
                             std::string action_str) {
  for (open_spiel::Action action : state.LegalActions()) {
    if (action_str == state.ActionToString(state.CurrentPlayer(), action))
      return action;
  }
  return open_spiel::kInvalidAction;
}

std::pair<std::vector<double>, std::vector<std::string>>
PlayGame(const open_spiel::Game &game,
         std::vector<std::unique_ptr<open_spiel::Bot>> &bots, std::mt19937 &rng,
         const std::vector<std::string> &initial_actions) {
  bool quiet = absl::GetFlag(FLAGS_quiet);
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::vector<std::string> history;

  if (!quiet)
    std::cerr << "Initial state:\n" << state << std::endl;

  // Play the initial actions (if there are any).
  for (const auto &action_str : initial_actions) {
    open_spiel::Player current_player = state->CurrentPlayer();
    open_spiel::Action action = GetAction(*state, action_str);

    if (action == open_spiel::kInvalidAction)
      open_spiel::SpielFatalError(absl::StrCat("Invalid action: ", action_str));

    history.push_back(action_str);
    state->ApplyAction(action);

    if (!quiet) {
      std::cerr << "Player " << current_player
                << " forced action: " << action_str << std::endl;
      std::cerr << "Next state:\n" << state->ToString() << std::endl;
    }
  }

  while (!state->IsTerminal()) {
    open_spiel::Player player = state->CurrentPlayer();

    open_spiel::Action action;
    if (state->IsChanceNode()) {
      // Chance node; sample one according to underlying distribution.
      open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
      action = open_spiel::SampleAction(outcomes, rng).first;
    } else {
      // The state must be a decision node, ask the right bot to make its
      // action.
      action = bots[player]->Step(*state);
    }
    if (!quiet)
      std::cerr << "Player " << player
                << " chose action: " << state->ActionToString(player, action)
                << std::endl;

    // Inform the other bot of the action performed.
    for (open_spiel::Player p = 0; p < bots.size(); ++p) {
      if (p != player) {
        bots[p]->InformAction(*state, player, action);
      }
    }

    // Update history and get the next state.
    history.push_back(state->ActionToString(player, action));
    state->ApplyAction(action);

    if (!quiet)
      std::cerr << "Next state:\n" << state->ToString() << std::endl;
  }

  std::cerr << "Returns: " << absl::StrJoin(state->Returns(), ", ")
            << std::endl;
  std::cerr << "Game actions: " << absl::StrJoin(history, ", ") << std::endl;

  return {state->Returns(), history};
}

int main(int argc, char **argv) {
  std::vector<char *> positional_args = absl::ParseCommandLine(argc, argv);
  std::mt19937 rng(Seed());  // Random number generator.

  // Create the game.
  std::string game_name = absl::GetFlag(FLAGS_game);
  std::cerr << "Game: " << game_name << std::endl;
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(game_name);

  // Ensure the game is AlphaZero-compatible and arguments are compatible.
  open_spiel::GameType game_type = game->GetType();
  if (game->NumPlayers() != 2)
    open_spiel::SpielFatalError("AlphaZero can only handle 2-player games.");
  if (game_type.reward_model != open_spiel::GameType::RewardModel::kTerminal)
    open_spiel::SpielFatalError("Game must have terminal rewards.");
  if (game_type.dynamics != open_spiel::GameType::Dynamics::kSequential)
    open_spiel::SpielFatalError("Game must have sequential turns.");
  if (absl::GetFlag(FLAGS_az_path).empty())
    open_spiel::SpielFatalError("AlphaZero path must be specified.");
  if (absl::GetFlag(FLAGS_player1) != "az" &&
      absl::GetFlag(FLAGS_player2) != "az")
    open_spiel::SpielFatalError("One of the players must be AlphaZero.");

  open_spiel::algorithms::torch_az::DeviceManager device_manager;
  device_manager.AddDevice(open_spiel::algorithms::torch_az::VPNetModel(
      *game, absl::GetFlag(FLAGS_az_path), absl::GetFlag(FLAGS_az_graph_def),
      "/cpu:0"));
  device_manager.Get(0, 0)->LoadCheckpoint(absl::GetFlag(FLAGS_az_checkpoint));
  auto az_evaluator =
      std::make_shared<open_spiel::algorithms::torch_az::VPNetEvaluator>(
          /*device_manager=*/&device_manager,
          /*batch_size=*/absl::GetFlag(FLAGS_az_batch_size),
          /*threads=*/absl::GetFlag(FLAGS_az_threads),
          /*cache_size=*/absl::GetFlag(FLAGS_az_cache_size),
          /*cache_shards=*/absl::GetFlag(FLAGS_az_cache_shards));
  auto evaluator =
      std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
          absl::GetFlag(FLAGS_rollout_count), Seed());

  std::vector<std::unique_ptr<open_spiel::Bot>> bots;
  bots.push_back(
      InitBot(absl::GetFlag(FLAGS_player1), *game, 0, evaluator, az_evaluator));
  bots.push_back(
      InitBot(absl::GetFlag(FLAGS_player2), *game, 1, evaluator, az_evaluator));

  std::vector<std::string> initial_actions;
  for (int i = 1; i < positional_args.size(); ++i) {
    initial_actions.push_back(positional_args[i]);
  }

  std::map<std::string, int> histories;
  std::vector<double> overall_returns(2, 0);
  std::vector<int> overall_wins(2, 0);
  int num_games = absl::GetFlag(FLAGS_num_games);
  for (int game_num = 0; game_num < num_games; ++game_num) {
    auto [returns, history] = PlayGame(*game, bots, rng, initial_actions);
    histories[absl::StrJoin(history, " ")] += 1;
    for (int i = 0; i < returns.size(); ++i) {
      double v = returns[i];
      overall_returns[i] += v;
      if (v > 0) {
        overall_wins[i] += 1;
      }
    }
  }

  std::cerr << "Number of games played: " << num_games << std::endl;
  std::cerr << "Number of distinct games played: " << histories.size()
            << std::endl;
  std::cerr << "Players: " << absl::GetFlag(FLAGS_player1) << ", "
            << absl::GetFlag(FLAGS_player2) << std::endl;
  std::cerr << "Overall wins: " << absl::StrJoin(overall_wins, ", ")
            << std::endl;
  std::cerr << "Overall returns: " << absl::StrJoin(overall_returns, ", ")
            << std::endl;

  return 0;
}
