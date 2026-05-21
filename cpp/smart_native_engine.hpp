#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace smart_native {

struct NativeSearchConfig {
  std::size_t max_step = 0;
  std::size_t mcts_iter = 0;
  std::size_t num_action_scale = 2;
  double action_unit = 0.01;
  double cover_penalty = 100.0;
  double pen_rate = 1.0;
  double exp_weight = 0.001;
  double gamma = 1.0;
  double action_prior_weight = 0.0;
  double puct_prior_weight = 0.0;
  double action_value_weight = 0.0;
  std::uint64_t seed = 0;
  bool stateful_union_cache = true;
  bool transposition_table = false;
  bool native_recenter = false;
  std::size_t cache_capacity = 65536;
  std::size_t transposition_table_size = 8192;
  std::size_t action_prior_top_k = 0;
  std::string volume_method = "mesh";
  std::vector<double> action_prior_logits;
  std::vector<double> action_value_logits;
};

struct NativeSearchResult {
  std::string status = "success";
  std::string backend = "smart-cpp-native";
  std::string command;
  std::string output_dir;
  std::string output_path;
  bool axis_only = true;
  std::size_t steps = 0;
  std::size_t iterations_run = 0;
  std::size_t node_count = 0;
  std::size_t exported_boxes = 0;
  std::size_t initial_partition_count = 0;
  std::size_t active_partition_count = 0;
  std::size_t adjacency_pair_count = 0;
  std::size_t candidate_inserts = 0;
  std::size_t candidate_erases = 0;
  std::size_t candidate_queries = 0;
  std::size_t action_prior_logits = 0;
  std::size_t action_value_logits = 0;
  std::size_t action_prior_top_k = 0;
  std::size_t transposition_table_size = 0;
  std::size_t transposition_hits = 0;
  std::size_t transposition_stores = 0;
  std::size_t recenter_applies = 0;
  std::size_t recenter_invalid = 0;
  double best_reward = 0.0;
  double initial_bbox_score = 0.0;
  double last_bbox_score = 0.0;
  double elapsed_sec = 0.0;
};

struct NativeMergeConfig {
  double merge_eps = 0.02;
  std::size_t final_k = 0;
  bool tilted = true;
  bool only_nearby = true;
};

struct NativeRefineMctsResult {
  std::string status = "success";
  std::string backend = "smart-cpp-native";
  NativeSearchResult refine;
  NativeSearchResult mcts;
  double elapsed_sec = 0.0;
};

NativeSearchResult run_merge_files(const std::string& msh_path,
                                   const std::string& partitions_path,
                                   const std::string& output_segment_path,
                                   const NativeMergeConfig& config);

NativeSearchResult run_refine_files(const std::string& msh_path,
                                    const std::string& bbox_params_path,
                                    const std::string& output_dir,
                                    const NativeSearchConfig& config);

NativeSearchResult run_mcts_files(const std::string& msh_path,
                                  const std::string& bbox_params_path,
                                  const std::string& output_dir,
                                  const NativeSearchConfig& config);

NativeRefineMctsResult run_refine_mcts_files(
    const std::string& msh_path,
    const std::string& bbox_params_path,
    const std::string& refine_output_dir,
    const std::string& mcts_output_dir,
    const NativeSearchConfig& refine_config,
    const NativeSearchConfig& mcts_config);

std::string result_json(const NativeSearchResult& result);
std::string result_json(const NativeRefineMctsResult& result);

}  // namespace smart_native
