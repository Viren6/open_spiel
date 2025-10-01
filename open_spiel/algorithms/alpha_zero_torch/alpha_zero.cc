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

#include "open_spiel/algorithms/alpha_zero_torch/alpha_zero.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <numeric>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/random/uniform_real_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/circular_buffer.h"
#include "open_spiel/utils/data_logger.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/logger.h"
#include "open_spiel/utils/lru_cache.h"
#include "open_spiel/utils/serializable_circular_buffer.h"
#include "open_spiel/utils/stats.h"
#include "open_spiel/utils/thread.h"
#include "open_spiel/utils/threaded_queue.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

namespace {

using open_spiel::ActionsAndProbs;
using open_spiel::algorithms::SearchNode;

double HypotheticalPUCTValue(const SearchNode& child, double parent_count,
                             double child_count, double uct_c) {
  if (!child.outcome.empty()) {
    return child.outcome[child.player];
  }
  double average = 0.0;
  if (child_count > 0) {
    average = child.total_reward / child_count;
  }
  double exploration = 0.0;
  if (parent_count > 0) {
    exploration = uct_c * child.prior * std::sqrt(parent_count) /
                  (child_count + 1.0);
  }
  return average + exploration;
}

std::vector<double> AdjustRootVisitCounts(
    const SearchNode& root, double forced_playouts_k,
    double forced_playouts_gamma, bool policy_target_pruning, double uct_c) {
  std::vector<double> visits;
  visits.reserve(root.children.size());
  for (const SearchNode& child : root.children) {
    visits.push_back(child.explore_count);
  }
  double total_visits =
      std::accumulate(visits.begin(), visits.end(), 0.0);
  if (!policy_target_pruning || forced_playouts_k <= 0 ||
      total_visits <= 0 || root.player == open_spiel::kChancePlayerId) {
    return visits;
  }

  int best_index = 0;
  for (int i = 1; i < visits.size(); ++i) {
    if (visits[i] > visits[best_index]) {
      best_index = i;
    }
  }

  std::vector<double> adjusted = visits;
  double adjusted_total = total_visits;
  double forced_scale =
      std::pow(std::max(1.0, total_visits), forced_playouts_gamma);

  for (int i = 0; i < adjusted.size(); ++i) {
    if (i == best_index) {
      continue;
    }
    const SearchNode& child = root.children[i];
    double target =
        forced_playouts_k * child.dirichlet_noise * forced_scale;
    int max_remove = std::min<int>(
        child.forced_playouts,
        static_cast<int>(std::floor(target + 1e-9)));
    max_remove = std::min<int>(max_remove, static_cast<int>(adjusted[i]));
    for (int r = 0; r < max_remove; ++r) {
      if (adjusted[i] <= 0 || adjusted_total <= 1) {
        break;
      }
      double new_count = adjusted[i] - 1;
      double new_total = adjusted_total - 1;
      double child_puct =
          HypotheticalPUCTValue(child, new_total, new_count, uct_c);
      const SearchNode& best_child = root.children[best_index];
      double best_puct = HypotheticalPUCTValue(
          best_child, new_total, adjusted[best_index], uct_c);
      if (child_puct >= best_puct) {
        break;
      }
      adjusted[i] = new_count;
      adjusted_total = new_total;
    }
  }

  for (int i = 0; i < adjusted.size(); ++i) {
    if (i == best_index) {
      continue;
    }
    if (adjusted[i] <= 1.0) {
      adjusted_total -= adjusted[i];
      adjusted[i] = 0.0;
    }
  }

  return adjusted;
}

ActionsAndProbs BuildPolicyFromVisits(const SearchNode& root,
                                      const std::vector<double>& visits,
                                      double root_temperature) {
  ActionsAndProbs policy;
  policy.reserve(root.children.size());
  double temperature = root_temperature > 0 ? root_temperature : 1.0;
  double inv_temp = 1.0 / temperature;
  double sum = 0.0;
  for (int i = 0; i < visits.size() && i < root.children.size(); ++i) {
    double count = visits[i];
    if (count <= 0) {
      continue;
    }
    double weight = inv_temp == 1.0 ? count : std::pow(count, inv_temp);
    if (!std::isfinite(weight) || weight <= 0) {
      continue;
    }
    policy.emplace_back(root.children[i].action, weight);
    sum += weight;
  }

  if (policy.empty() || sum <= 0) {
    policy.clear();
    if (!root.children.empty()) {
      double uniform = 1.0 / static_cast<double>(root.children.size());
      for (const SearchNode& child : root.children) {
        policy.emplace_back(child.action, uniform);
      }
    }
    return policy;
  }

  NormalizePolicy(&policy);
  return policy;
}

ActionsAndProbs ApplyTemperatureToPolicy(const ActionsAndProbs& policy,
                                         double temperature) {
  if (policy.empty() || temperature <= 0 || temperature == 1.0) {
    return policy;
  }
  double inv_temp = 1.0 / temperature;
  ActionsAndProbs adjusted;
  adjusted.reserve(policy.size());
  double sum = 0.0;
  for (const auto& [action, prob] : policy) {
    double clamped = std::max(prob, 1e-12);
    double weight = std::pow(clamped, inv_temp);
    adjusted.emplace_back(action, weight);
    sum += weight;
  }
  if (sum <= 0) {
    return policy;
  }
  NormalizePolicy(&adjusted);
  return adjusted;
}

}  // namespace

struct StartInfo {
  absl::Time start_time;
  int start_step;
  int model_checkpoint_step;
  int64_t total_trajectories;
};

StartInfo StartInfoFromLearnerJson(const std::string& path) {
  StartInfo start_info;
  file::File learner_file(path + "/learner.jsonl", "r");
  std::vector<std::string> learner_lines =
      absl::StrSplit(learner_file.ReadContents(), '\n');
  std::string last_learner_line;

  // Get the last non-empty line in learner.jsonl.
  for (int i = learner_lines.size() - 1; i >= 0; i--) {
    if (!learner_lines[i].empty()) {
      last_learner_line = learner_lines[i];
      break;
    }
  }

  json::Object last_learner_json = json::FromString(
      last_learner_line).value().GetObject();

  start_info.start_time = absl::Now() - absl::Seconds(
      last_learner_json["time_rel"].GetDouble());
  start_info.start_step = last_learner_json["step"].GetInt() + 1;
  start_info.model_checkpoint_step = VPNetModel::kMostRecentCheckpointStep;
  start_info.total_trajectories =
      last_learner_json["total_trajectories"].GetInt();

  return start_info;
}

struct Trajectory {
  struct State {
    std::vector<float> observation;
    open_spiel::Player current_player;
    std::vector<open_spiel::Action> legal_actions;
    open_spiel::Action action;
    open_spiel::ActionsAndProbs policy;
    double value;
  };

  std::vector<State> states;
  std::vector<double> returns;
};

Trajectory PlayGame(Logger* logger, int game_num, const open_spiel::Game& game,
                    std::vector<std::unique_ptr<MCTSBot>>* bots,
                    std::mt19937* rng, double temperature, int temperature_drop,
                    double cutoff_value, const AlphaZeroConfig& config,
                    bool verbose = false) {
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::vector<std::string> history;
  Trajectory trajectory;

  while (true) {
    if (state->IsChanceNode()) {
      open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
      open_spiel::Action action =
          open_spiel::SampleAction(outcomes, *rng).first;
      history.push_back(state->ActionToString(state->CurrentPlayer(), action));
      state->ApplyAction(action);
    } else {
      open_spiel::Player player = state->CurrentPlayer();
      std::unique_ptr<SearchNode> root = (*bots)[player]->MCTSearch(*state);
      std::vector<double> adjusted_visits = AdjustRootVisitCounts(
          *root, config.forced_playouts_k, config.forced_playouts_gamma,
          config.policy_target_pruning, config.uct_c);
      open_spiel::ActionsAndProbs training_policy = BuildPolicyFromVisits(
          *root, adjusted_visits, config.policy_root_temperature);
      open_spiel::ActionsAndProbs sampling_policy = training_policy;
      if (history.size() < temperature_drop) {
        sampling_policy =
            ApplyTemperatureToPolicy(training_policy, temperature);
      }
      open_spiel::Action action;
      if (history.size() >= temperature_drop) {
        action = root->BestChild().action;
      } else if (!sampling_policy.empty()) {
        action = open_spiel::SampleAction(sampling_policy, *rng).first;
      } else {
        action = root->BestChild().action;
      }

      double root_value = root->total_reward / root->explore_count;
      trajectory.states.push_back(Trajectory::State{
          state->ObservationTensor(), player, state->LegalActions(), action,
          std::move(training_policy), root_value});
      std::string action_str = state->ActionToString(player, action);
      history.push_back(action_str);
      state->ApplyAction(action);
      if (verbose) {
        logger->Print("Player: %d, action: %s", player, action_str);
      }
      if (state->IsTerminal()) {
        trajectory.returns = state->Returns();
        break;
      } else if (std::abs(root_value) > cutoff_value) {
        trajectory.returns.resize(2);
        trajectory.returns[player] = root_value;
        trajectory.returns[1 - player] = -root_value;
        break;
      }
    }
  }

  logger->Print("Game %d: Returns: %s; Actions: %s", game_num,
                absl::StrJoin(trajectory.returns, " "),
                absl::StrJoin(history, " "));
  return trajectory;
}

std::unique_ptr<MCTSBot> InitAZBot(const AlphaZeroConfig& config,
                                   const open_spiel::Game& game,
                                   std::shared_ptr<Evaluator> evaluator,
                                   bool evaluation) {
  double dirichlet_alpha =
      config.policy_use_dynamic_alpha ? 0.0 : config.policy_alpha;
  auto bot = std::make_unique<MCTSBot>(
      game, std::move(evaluator), config.uct_c, config.max_simulations,
      /*max_memory_mb=*/10,
      /*solve=*/false,
      /*seed=*/0,
      /*verbose=*/false, ChildSelectionPolicy::PUCT,
      evaluation ? 0 : dirichlet_alpha,
      evaluation ? 0 : config.policy_epsilon,
      /*dont_return_chance_node*/ true);

  if (config.policy_use_dynamic_alpha && config.policy_alpha_base > 0) {
    bot->SetDynamicRootExploration(config.policy_alpha_base,
                                   config.policy_alpha_reference);
  } else {
    bot->ClearDynamicRootExploration();
  }

  if (evaluation) {
    bot->SetRootPolicyTemperature(1.0);
    bot->SetForcedPlayoutConfig(0.0, config.forced_playouts_gamma);
    bot->SetPolicyTargetPruning(false);
  } else {
    bot->SetRootPolicyTemperature(config.policy_root_temperature);
    bot->SetForcedPlayoutConfig(config.forced_playouts_k,
                                config.forced_playouts_gamma);
    bot->SetPolicyTargetPruning(config.policy_target_pruning);
  }

  return bot;
}

// An actor thread runner that generates games and returns trajectories.
void actor(const open_spiel::Game& game, const AlphaZeroConfig& config, int num,
           ThreadedQueue<Trajectory>* trajectory_queue,
           std::shared_ptr<VPNetEvaluator> vp_eval, StopToken* stop) {
  std::unique_ptr<Logger> logger;
  if (num < 20) {  // Limit the number of open files.
    logger.reset(new FileLogger(config.path, absl::StrCat("actor-", num)));
  } else {
    logger.reset(new NoopLogger());
  }
  std::mt19937 rng(absl::ToUnixNanos(absl::Now()));
  absl::uniform_real_distribution<double> dist(0.0, 1.0);
  std::vector<std::unique_ptr<MCTSBot>> bots;
  bots.reserve(2);
  for (int player = 0; player < 2; player++) {
    bots.push_back(InitAZBot(config, game, vp_eval, false));
  }
  for (int game_num = 1; !stop->StopRequested(); ++game_num) {
    double cutoff =
        (dist(rng) < config.cutoff_probability ? config.cutoff_value
                                               : game.MaxUtility() + 1);
    if (!trajectory_queue->Push(
            PlayGame(logger.get(), game_num, game, &bots, &rng,
                     config.temperature, config.temperature_drop, cutoff,
                     config),
            absl::Seconds(10))) {
      logger->Print("Failed to push a trajectory after 10 seconds.");
    }
  }
  logger->Print("Got a quit.");
}

class EvalResults {
 public:
  explicit EvalResults(int count, int evaluation_window) {
    results_.reserve(count);
    for (int i = 0; i < count; ++i) {
      results_.emplace_back(evaluation_window);
    }
  }

  // How many evals per difficulty.
  int EvalCount() {
    absl::MutexLock lock(&m_);
    return eval_num_ / results_.size();
  }

  // Which eval to do next: difficulty, player0.
  std::pair<int, bool> Next() {
    absl::MutexLock lock(&m_);
    int next = eval_num_ % (results_.size() * 2);
    eval_num_ += 1;
    return {next / 2, next % 2};
  }

  void Add(int i, double value) {
    absl::MutexLock lock(&m_);
    results_[i].Add(value);
  }

  std::vector<double> AvgResults() {
    absl::MutexLock lock(&m_);
    std::vector<double> out;
    out.reserve(results_.size());
    for (const auto& result : results_) {
      out.push_back(result.Empty() ? 0
                                   : (absl::c_accumulate(result.Data(), 0.0) /
                                      result.Size()));
    }
    return out;
  }

 private:
  std::vector<CircularBuffer<double>> results_;
  int eval_num_ = 0;
  absl::Mutex m_;
};

// A thread that plays vs standard MCTS.
void evaluator(const open_spiel::Game& game, const AlphaZeroConfig& config,
               int num, EvalResults* results,
               std::shared_ptr<VPNetEvaluator> vp_eval, StopToken* stop) {
  FileLogger logger(config.path, absl::StrCat("evaluator-", num));
  std::mt19937 rng;
  auto rand_evaluator = std::make_shared<RandomRolloutEvaluator>(1, num);

  for (int game_num = 1; !stop->StopRequested(); ++game_num) {
    auto [difficulty, first] = results->Next();
    int az_player = first ? 0 : 1;
    int rand_max_simulations =
        config.max_simulations * std::pow(10, difficulty / 2.0);
    std::vector<std::unique_ptr<MCTSBot>> bots;
    bots.reserve(2);
    bots.push_back(InitAZBot(config, game, vp_eval, true));
    bots.push_back(std::make_unique<MCTSBot>(
        game, rand_evaluator, config.uct_c, rand_max_simulations,
        /*max_memory_mb=*/1000,
        /*solve=*/true,
        /*seed=*/num * 1000 + game_num,
        /*verbose=*/false, ChildSelectionPolicy::UCT,
        /*dirichlet_alpha=*/0,
        /*dirichlet_epsilon=*/0,
        /*dont_return_chance_node=*/true));
    if (az_player == 1) {
      std::swap(bots[0], bots[1]);
    }

    logger.Print("Running MCTS with %d simulations", rand_max_simulations);
    Trajectory trajectory = PlayGame(
        &logger, game_num, game, &bots, &rng, /*temperature=*/1,
        /*temperature_drop=*/0, /*cutoff_value=*/game.MaxUtility() + 1,
        config);

    results->Add(difficulty, trajectory.returns[az_player]);
    logger.Print("Game %d: AZ: %5.2f, MCTS: %5.2f, MCTS-sims: %d, length: %d",
                 game_num, trajectory.returns[az_player],
                 trajectory.returns[1 - az_player], rand_max_simulations,
                 trajectory.states.size());
  }
  logger.Print("Got a quit.");
}

void learner(const open_spiel::Game& game, const AlphaZeroConfig& config,
             DeviceManager* device_manager,
             std::shared_ptr<VPNetEvaluator> eval,
             ThreadedQueue<Trajectory>* trajectory_queue,
             EvalResults* eval_results, StopToken* stop,
             const StartInfo& start_info) {
  FileLogger logger(config.path, "learner", "a");
  DataLoggerJsonLines data_logger(
      config.path, "learner", true, "a", start_info.start_time);
  std::mt19937 rng;

  int device_id = 0;  // Do not change, the first device is the learner.
  logger.Print("Running the learner on device %d: %s", device_id,
               device_manager->Get(0, device_id)->Device());

  SerializableCircularBuffer<VPNetModel::TrainInputs> replay_buffer(
      config.replay_buffer_size);
  if (start_info.start_step > 1) {
    replay_buffer.LoadBuffer(config.path + "/replay_buffer.data");
  }
  int learn_rate = config.replay_buffer_size / config.replay_buffer_reuse;
  int64_t total_trajectories = start_info.total_trajectories;

  const int stage_count = 7;
  std::vector<open_spiel::BasicStats> value_accuracies(stage_count);
  std::vector<open_spiel::BasicStats> value_predictions(stage_count);
  open_spiel::BasicStats game_lengths;
  open_spiel::HistogramNumbered game_lengths_hist(game.MaxGameLength() + 1);

  open_spiel::HistogramNamed outcomes({"Player1", "Player2", "Draw"});
  // Actor threads have likely been contributing for a while, so put `last` in
  // the past to avoid a giant spike on the first step.
  absl::Time last = absl::Now() - absl::Seconds(60);
  for (int step = start_info.start_step;
       !stop->StopRequested() &&
           (config.max_steps == 0 || step <= config.max_steps);
       ++step) {
    outcomes.Reset();
    game_lengths.Reset();
    game_lengths_hist.Reset();
    for (auto& value_accuracy : value_accuracies) {
      value_accuracy.Reset();
    }
    for (auto& value_prediction : value_predictions) {
      value_prediction.Reset();
    }

    // Collect trajectories
    int queue_size = trajectory_queue->Size();
    int num_states = 0;
    int num_trajectories = 0;
    while (!stop->StopRequested() && num_states < learn_rate) {
      absl::optional<Trajectory> trajectory = trajectory_queue->Pop();
      if (trajectory) {
        num_trajectories += 1;
        total_trajectories += 1;
        game_lengths.Add(trajectory->states.size());
        game_lengths_hist.Add(trajectory->states.size());

        double p1_outcome = trajectory->returns[0];
        outcomes.Add(p1_outcome > 0 ? 0 : (p1_outcome < 0 ? 1 : 2));

        for (const Trajectory::State& state : trajectory->states) {
          replay_buffer.Add(VPNetModel::TrainInputs{state.legal_actions,
                                                    state.observation,
                                                    state.policy, p1_outcome});
          num_states += 1;
        }

        for (int stage = 0; stage < stage_count; ++stage) {
          // Scale for the length of the game
          int index = (trajectory->states.size() - 1) *
                      static_cast<double>(stage) / (stage_count - 1);
          const Trajectory::State& s = trajectory->states[index];
          value_accuracies[stage].Add(
              (s.value >= 0) == (trajectory->returns[s.current_player] >= 0));
          value_predictions[stage].Add(abs(s.value));
        }
      }
    }
    absl::Time now = absl::Now();
    double seconds = absl::ToDoubleSeconds(now - last);

    logger.Print("Step: %d", step);
    logger.Print(
        "Collected %5d states from %3d games, %.1f states/s; "
        "%.1f states/(s*actor), game length: %.1f",
        num_states, num_trajectories, num_states / seconds,
        num_states / (config.actors * seconds),
        static_cast<double>(num_states) / num_trajectories);
    logger.Print("Queue size: %d. Buffer size: %d. States seen: %d", queue_size,
                 replay_buffer.Size(), replay_buffer.TotalAdded());

    if (stop->StopRequested()) {
      break;
    }

    last = now;

    replay_buffer.SaveBuffer(config.path + "/replay_buffer.data");

    VPNetModel::LossInfo losses;
    {  // Extra scope to return the device for use for inference asap.
      DeviceManager::DeviceLoan learn_model =
          device_manager->Get(config.train_batch_size, device_id);

      // Let the device manager know that the first device is now
      // off-limits for inference and should only be used for learning
      // (if config.explicit_learning == true).
      device_manager->SetLearning(config.explicit_learning);

      // Learn from them.
      for (int i = 0; i < replay_buffer.Size() / config.train_batch_size; i++) {
        losses += learn_model->Learn(
            replay_buffer.Sample(&rng, config.train_batch_size));
      }

      // The device manager can now once again use the first device for
      // inference (if it could not before).
      device_manager->SetLearning(false);
    }

    // Always save a checkpoint, either for keeping or for loading the weights
    // to the other sessions. It only allows numbers, so use -1 as "latest".
    std::string checkpoint_path = device_manager->Get(0, device_id)
        ->SaveCheckpoint(VPNetModel::kMostRecentCheckpointStep);
    if (step % config.checkpoint_freq == 0) {
      device_manager->Get(0, device_id)->SaveCheckpoint(step);
    }
    if (device_manager->Count() > 0) {
      for (int i = 0; i < device_manager->Count(); ++i) {
        if (i != device_id) {
          device_manager->Get(0, i)->LoadCheckpoint(checkpoint_path);
        }
      }
    }
    logger.Print("Checkpoint saved: %s", checkpoint_path);

    DataLogger::Record record = {
        {"step", step},
        {"total_states", replay_buffer.TotalAdded()},
        {"states_per_s", num_states / seconds},
        {"states_per_s_actor", num_states / (config.actors * seconds)},
        {"total_trajectories", total_trajectories},
        {"trajectories_per_s", num_trajectories / seconds},
        {"queue_size", queue_size},
        {"game_length", game_lengths.ToJson()},
        {"game_length_hist", game_lengths_hist.ToJson()},
        {"outcomes", outcomes.ToJson()},
        {"value_accuracy",
         json::TransformToArray(value_accuracies,
                                [](auto v) { return v.ToJson(); })},
        {"value_prediction",
         json::TransformToArray(value_predictions,
                                [](auto v) { return v.ToJson(); })},
        {"eval", json::Object({
                     {"count", eval_results->EvalCount()},
                     {"results", json::CastToArray(eval_results->AvgResults())},
                 })},
        {"batch_size", eval->BatchSizeStats().ToJson()},
        {"batch_size_hist", eval->BatchSizeHistogram().ToJson()},
        {"loss", json::Object({
                     {"policy", losses.Policy()},
                     {"value", losses.Value()},
                     {"l2reg", losses.L2()},
                     {"sum", losses.Total()},
                 })},
    };
    eval->ResetBatchSizeStats();
    logger.Print("Losses: policy: %.4f, value: %.4f, l2: %.4f, sum: %.4f",
                 losses.Policy(), losses.Value(), losses.L2(), losses.Total());

    LRUCacheInfo cache_info = eval->CacheInfo();
    if (cache_info.size > 0) {
      logger.Print(absl::StrFormat(
          "Cache size: %d/%d: %.1f%%, hits: %d, misses: %d, hit rate: %.3f%%",
          cache_info.size, cache_info.max_size, 100.0 * cache_info.Usage(),
          cache_info.hits, cache_info.misses, 100.0 * cache_info.HitRate()));
      eval->ClearCache();
    }
    record.emplace("cache",
                   json::Object({
                       {"size", cache_info.size},
                       {"max_size", cache_info.max_size},
                       {"usage", cache_info.Usage()},
                       {"requests", cache_info.Total()},
                       {"requests_per_s", cache_info.Total() / seconds},
                       {"hits", cache_info.hits},
                       {"misses", cache_info.misses},
                       {"misses_per_s", cache_info.misses / seconds},
                       {"hit_rate", cache_info.HitRate()},
                   }));

    data_logger.Write(record);
    logger.Print("");
  }
}

bool AlphaZero(AlphaZeroConfig config, StopToken* stop, bool resuming) {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(config.game);

  open_spiel::GameType game_type = game->GetType();
  if (game->NumPlayers() != 2)
    open_spiel::SpielFatalError("AlphaZero can only handle 2-player games.");
  if (game_type.reward_model != open_spiel::GameType::RewardModel::kTerminal)
    open_spiel::SpielFatalError("Game must have terminal rewards.");
  if (game_type.dynamics != open_spiel::GameType::Dynamics::kSequential)
    open_spiel::SpielFatalError("Game must have sequential turns.");

  file::Mkdirs(config.path);
  if (!file::IsDirectory(config.path)) {
    std::cerr << config.path << " is not a directory." << std::endl;
    return false;
  }

  std::cout << "Logging directory: " << config.path << std::endl;

  if (config.graph_def.empty()) {
    config.graph_def = "vpnet.pb";
    std::string model_path = absl::StrCat(config.path, "/", config.graph_def);
    if (file::Exists(model_path)) {
      std::cout << "Overwriting existing model: " << model_path << std::endl;
    } else {
      std::cout << "Creating model: " << model_path << std::endl;
    }
    SPIEL_CHECK_TRUE(CreateGraphDef(
        *game, config.learning_rate, config.weight_decay, config.path,
        config.graph_def, config.nn_model, config.nn_width, config.nn_depth));
  } else {
    std::string model_path = absl::StrCat(config.path, "/", config.graph_def);
    if (file::Exists(model_path)) {
      std::cout << "Using existing model: " << model_path << std::endl;
    } else {
      std::cout << "Model not found: " << model_path << std::endl;
    }
  }

  std::cout << "Playing game: " << config.game << std::endl;

  config.inference_batch_size = std::max(
      1,
      std::min(config.inference_batch_size, config.actors + config.evaluators));

  config.inference_threads =
      std::max(1, std::min(config.inference_threads,
                           (1 + config.actors + config.evaluators) / 2));

  {
    file::File fd(config.path + "/config.json", "w");
    fd.Write(json::ToString(config.ToJson(), true) + "\n");
  }

  StartInfo start_info = {/*start_time=*/absl::Now(),
                          /*start_step=*/1,
                          /*model_checkpoint_step=*/0,
                          /*total_trajectories=*/0};
  if (resuming) {
    start_info = StartInfoFromLearnerJson(config.path);
  }

  DeviceManager device_manager;
  for (const absl::string_view& device : absl::StrSplit(config.devices, ',')) {
    device_manager.AddDevice(
        VPNetModel(*game, config.path, config.graph_def, std::string(device)));
  }

  if (device_manager.Count() == 0) {
    std::cerr << "No devices specified?" << std::endl;
    return false;
  }

  // The explicit_learning option should only be used when multiple
  // devices are available (so that inference can continue while
  // also undergoing learning).
  if (device_manager.Count() <= 1 && config.explicit_learning) {
    std::cerr << "Explicit learning can only be used with multiple devices."
              << std::endl;
    return false;
  }

  std::cerr << "Loading model from step " << start_info.model_checkpoint_step
            << std::endl;
  {  // Make sure they're all in sync.
    if (!resuming) {
      device_manager.Get(0)->SaveCheckpoint(start_info.model_checkpoint_step);
    }
    for (int i = 0; i < device_manager.Count(); ++i) {
      device_manager.Get(0, i)->LoadCheckpoint(
          start_info.model_checkpoint_step);
    }
  }

  auto eval = std::make_shared<VPNetEvaluator>(
      &device_manager, config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  ThreadedQueue<Trajectory> trajectory_queue(config.replay_buffer_size /
                                             config.replay_buffer_reuse);

  EvalResults eval_results(config.eval_levels, config.evaluation_window);

  std::vector<Thread> actors;
  actors.reserve(config.actors);
  for (int i = 0; i < config.actors; ++i) {
    actors.emplace_back(
        [&, i]() { actor(*game, config, i, &trajectory_queue, eval, stop); });
  }
  std::vector<Thread> evaluators;
  evaluators.reserve(config.evaluators);
  for (int i = 0; i < config.evaluators; ++i) {
    evaluators.emplace_back(
        [&, i]() { evaluator(*game, config, i, &eval_results, eval, stop); });
  }
  learner(*game, config, &device_manager, eval, &trajectory_queue,
          &eval_results, stop, start_info);

  if (!stop->StopRequested()) {
    stop->Stop();
  }

  // Empty the queue so that the actors can exit.
  trajectory_queue.BlockNewValues();
  trajectory_queue.Clear();

  std::cout << "Joining all the threads." << std::endl;
  for (auto& t : actors) {
    t.join();
  }
  for (auto& t : evaluators) {
    t.join();
  }
  std::cout << "Exiting cleanly." << std::endl;
  return true;
}

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel
