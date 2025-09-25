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

#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#include "open_spiel/abseil-cpp/absl/hash/hash.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/utils/stats.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {
namespace {

template <typename T>
std::string FormatVectorPrefix(const std::vector<T>& values, int max_elems) {
  std::ostringstream oss;
  oss << std::setprecision(9);
  oss << "[";
  const int count = std::min<int>(values.size(), max_elems);
  for (int i = 0; i < count; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << values[i];
  }
  if (values.size() > static_cast<size_t>(max_elems)) {
    oss << ", ...";
  }
  oss << "]";
  return oss.str();
}

template <typename T>
std::string DescribeVectorDiff(const std::vector<T>& requested,
                               const std::vector<T>& cached) {
  std::ostringstream oss;
  oss << std::setprecision(9);
  oss << "{requested_size=" << requested.size()
      << ", cached_size=" << cached.size();
  const size_t limit = std::min(requested.size(), cached.size());
  size_t mismatch_index = 0;
  while (mismatch_index < limit &&
         requested[mismatch_index] == cached[mismatch_index]) {
    ++mismatch_index;
  }
  if (mismatch_index < limit) {
    oss << ", first_mismatch_index=" << mismatch_index
        << ", requested_value=" << requested[mismatch_index]
        << ", cached_value=" << cached[mismatch_index];
  } else if (requested.size() != cached.size()) {
    oss << ", mismatch_due_to_size=true";
  } else {
    oss << ", mismatch_not_identified=true";
  }
  oss << ", requested_prefix=" << FormatVectorPrefix(requested, 10)
      << ", cached_prefix=" << FormatVectorPrefix(cached, 10) << "}";
  return oss.str();
}

void LogCacheCollision(uint64_t key, int shard,
                       const VPNetModel::InferenceInputs& requested,
                       const VPNetModel::InferenceInputs& cached) {
  std::ostringstream oss;
  oss << std::setprecision(9);
  oss << "AlphaZero Torch inference cache collision detected"
      << " (thread=" << std::this_thread::get_id()
      << ", shard=" << shard << ", key=" << key << ")";
  oss << ". Legal actions diff "
      << DescribeVectorDiff(requested.legal_actions, cached.legal_actions);
  oss << "; Observation diff "
      << DescribeVectorDiff(requested.observations, cached.observations);
  std::cerr << oss.str() << std::endl;
}

}  // namespace

VPNetEvaluator::VPNetEvaluator(DeviceManager* device_manager, int batch_size,
                               int threads, int cache_size, int cache_shards)
    : device_manager_(*device_manager),
      batch_size_(batch_size),
      queue_(batch_size * threads * 4),
      batch_size_hist_(batch_size + 1) {
  cache_shards = std::max(1, cache_shards);
  cache_.reserve(cache_shards);
  for (int i = 0; i < cache_shards; ++i) {
    cache_.push_back(std::make_unique<LRUCache<uint64_t, CacheValue>>(
        cache_size / cache_shards));
  }
  if (batch_size_ <= 1) {
    threads = 0;
  }
  inference_threads_.reserve(threads);
  for (int i = 0; i < threads; ++i) {
    inference_threads_.emplace_back([this]() { this->Runner(); });
  }
}

VPNetEvaluator::~VPNetEvaluator() {
  stop_.Stop();
  queue_.BlockNewValues();
  queue_.Clear();
  for (auto& t : inference_threads_) {
    t.join();
  }
}

void VPNetEvaluator::ClearCache() {
  for (auto& c : cache_) {
    c->Clear();
  }
}

LRUCacheInfo VPNetEvaluator::CacheInfo() {
  LRUCacheInfo info;
  for (auto& c : cache_) {
    info += c->Info();
  }
  return info;
}

std::vector<double> VPNetEvaluator::Evaluate(const State& state) {
  // TODO(author5): currently assumes zero-sum.
  double p0value = Inference(state).value;
  return {p0value, -p0value};
}

open_spiel::ActionsAndProbs VPNetEvaluator::Prior(const State& state) {
  if (state.IsChanceNode()) {
    return state.ChanceOutcomes();
  } else {
    return Inference(state).policy;
  }
}

VPNetModel::InferenceOutputs VPNetEvaluator::Inference(const State& state) {
  VPNetModel::InferenceInputs inputs = {state.LegalActions(),
                                        state.ObservationTensor()};

  uint64_t key = 0;
  int cache_shard = 0;
  if (!cache_.empty()) {
    key = absl::Hash<VPNetModel::InferenceInputs>{}(inputs);
    cache_shard = key % cache_.size();
    absl::optional<const CacheValue> opt_value =
        cache_[cache_shard]->Get(key);
    if (opt_value) {
      if (opt_value->inputs == inputs) {
        return opt_value->outputs;
      }
      LogCacheCollision(key, cache_shard, inputs, opt_value->inputs);
    }
  }
  VPNetModel::InferenceOutputs outputs;
  if (batch_size_ <= 1) {
    outputs = device_manager_.Get(1)->Inference(std::vector{inputs})[0];
  } else {
    std::promise<VPNetModel::InferenceOutputs> prom;
    std::future<VPNetModel::InferenceOutputs> fut = prom.get_future();
    queue_.Push(QueueItem{inputs, &prom});
    outputs = fut.get();
  }
  if (!cache_.empty()) {
    cache_[cache_shard]->Set(key, CacheValue{inputs, outputs});
  }
  return outputs;
}

void VPNetEvaluator::Runner() {
  std::vector<VPNetModel::InferenceInputs> inputs;
  std::vector<std::promise<VPNetModel::InferenceOutputs>*> promises;
  inputs.reserve(batch_size_);
  promises.reserve(batch_size_);
  while (!stop_.StopRequested()) {
    {
      // Only one thread at a time should be listening to the queue to maximize
      // batch size and minimize latency.
      absl::MutexLock lock(&inference_queue_m_);
      absl::Time deadline = absl::InfiniteFuture();
      for (int i = 0; i < batch_size_; ++i) {
        absl::optional<QueueItem> item = queue_.Pop(deadline);
        if (!item) {  // Hit the deadline.
          break;
        }
        if (inputs.empty()) {
          deadline = absl::Now() + absl::Milliseconds(1);
        }
        inputs.push_back(item->inputs);
        promises.push_back(item->prom);
      }
    }

    if (inputs.empty()) {  // Almost certainly StopRequested.
      continue;
    }

    {
      absl::MutexLock lock(&stats_m_);
      batch_size_stats_.Add(inputs.size());
      batch_size_hist_.Add(inputs.size());
    }

    std::vector<VPNetModel::InferenceOutputs> outputs =
        device_manager_.Get(inputs.size())->Inference(inputs);
    for (int i = 0; i < promises.size(); ++i) {
      promises[i]->set_value(outputs[i]);
    }
    inputs.clear();
    promises.clear();
  }
}

void VPNetEvaluator::ResetBatchSizeStats() {
  absl::MutexLock lock(&stats_m_);
  batch_size_stats_.Reset();
  batch_size_hist_.Reset();
}

open_spiel::BasicStats VPNetEvaluator::BatchSizeStats() {
  absl::MutexLock lock(&stats_m_);
  return batch_size_stats_;
}

open_spiel::HistogramNumbered VPNetEvaluator::BatchSizeHistogram() {
  absl::MutexLock lock(&stats_m_);
  return batch_size_hist_;
}

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel
