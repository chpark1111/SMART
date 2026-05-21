#include "smart_native_core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

namespace py = pybind11;

extern "C" float smart_manifold_cube_volume(float x, float y, float z);
extern "C" void* smart_manifold_mesh_new(const float* vertices,
                                          std::size_t n_vertices,
                                          const std::uint32_t* faces,
                                          std::size_t n_faces);
extern "C" void smart_manifold_delete(void* handle);
extern "C" double smart_manifold_handle_volume(void* handle);
extern "C" double smart_manifold_handle_volume_properties(void* handle);
extern "C" double smart_manifold_residual_volume_for_boxes(
    void* handle, const float* box_vertices, std::size_t n_boxes);
extern "C" double smart_manifold_residual_volume_for_boxes_properties(
    void* handle, const float* box_vertices, std::size_t n_boxes);
extern "C" int smart_manifold_residual_volume_for_boxes_pair(
    void* handle, const float* box_vertices, std::size_t n_boxes,
    double* out_mesh_volume, double* out_properties_volume);
extern "C" int smart_manifold_best_axis_actions_for_mask(
    void* handle, const double* bounds, const double* rotations,
    const std::uint8_t* bbox_mask, std::size_t n_boxes,
    std::size_t num_action_scale, double action_unit, double volume_sum,
    double last_bbox_score, double cover_penalty, double pen_rate,
    double initial_best, const double* action_scales,
    std::intptr_t* out_actions, double* out_rewards, int volume_method);
extern "C" float smart_manifold_mesh_volume(const float* vertices,
                                             std::size_t n_vertices,
                                             const std::uint32_t* faces,
                                             std::size_t n_faces);
extern "C" float smart_manifold_axis_box_intersection_volume(
    const float* vertices, std::size_t n_vertices, const std::uint32_t* faces,
    std::size_t n_faces, float lx, float ly, float lz, float rx, float ry,
    float rz);
extern "C" void* smart_manifold_state_new(
    const float* vertices, std::size_t n_vertices, const std::uint32_t* faces,
    std::size_t n_faces, const double* bounds, const double* rotations,
    std::size_t n_boxes, double volume_sum, double last_bbox_score,
    int stateful_union_cache, std::size_t cache_capacity, int volume_method);
extern "C" void smart_manifold_state_delete(void* handle);
extern "C" int smart_manifold_state_reset(void* handle, const double* bounds,
                                           const double* rotations,
                                           std::size_t n_boxes,
                                           double last_bbox_score);
extern "C" std::size_t smart_manifold_state_num_boxes(void* handle);
extern "C" int smart_manifold_state_copy(void* handle, double* out_bounds,
                                          double* out_rotations);
extern "C" int smart_manifold_state_copy_bbox(void* handle,
                                               std::size_t bbox_idx,
                                               double* out_bounds,
                                               double* out_rotation);
extern "C" double smart_manifold_state_last_bbox_score(void* handle);
extern "C" double smart_manifold_state_total_bbox_volume(void* handle);
extern "C" std::size_t smart_manifold_state_valid_count(void* handle);
extern "C" int smart_manifold_state_cache_stats(void* handle,
                                                 std::uint64_t* out_values);
extern "C" double smart_manifold_state_covered(void* handle);
extern "C" double smart_manifold_state_score(void* handle,
                                             double cover_penalty,
                                             double pen_rate);
extern "C" double smart_manifold_state_score_axis_action(
    void* handle, std::intptr_t action, std::size_t num_action_scale,
    double action_unit, double cover_penalty, double pen_rate,
    const double* action_scales);
extern "C" double smart_manifold_state_score_replacement(
    void* handle, std::size_t bbox_idx, const double* candidate_bounds,
    const double* candidate_rotation, double cover_penalty, double pen_rate);
extern "C" double smart_manifold_state_apply_replacement(
    void* handle, std::size_t bbox_idx, const double* candidate_bounds,
    const double* candidate_rotation, double cover_penalty, double pen_rate);
extern "C" int smart_manifold_state_select_replacements_for_mask(
    void* handle, const std::uint8_t* bbox_mask,
    const double* candidate_bounds, const double* candidate_rotations,
    std::size_t num_action_scale, double cover_penalty, double pen_rate,
    std::intptr_t* out_actions, double* out_rewards);
extern "C" int smart_manifold_state_best_axis_actions_for_mask(
    void* handle, const std::uint8_t* bbox_mask,
    std::size_t num_action_scale, double action_unit, double cover_penalty,
    double pen_rate, double initial_best, const double* action_scales,
    std::intptr_t* out_actions, double* out_rewards);
extern "C" double smart_manifold_state_apply_axis_action(
    void* handle, std::intptr_t action, std::size_t num_action_scale,
    double action_unit, double cover_penalty, double pen_rate,
    const double* action_scales);
extern "C" int smart_manifold_state_greedy_axis_refine_segment(
    void* handle, std::size_t num_action_scale, double action_unit,
    double cover_penalty, double pen_rate, std::size_t max_steps,
    const double* action_scales, std::intptr_t* out_actions,
    double* out_rewards, std::size_t* out_steps, double* out_last_score);
extern "C" int smart_manifold_state_greedy_axis_rollout_step(
    void* handle, const std::uint8_t* bbox_mask,
    std::size_t num_action_scale, double action_unit, double cover_penalty,
    double pen_rate, const double* action_scales, std::uint8_t* out_next_mask,
    std::intptr_t* out_action, double* out_best_reward,
    double* out_applied_reward, double* out_last_score);
extern "C" int smart_manifold_state_greedy_axis_rollout_segment(
    void* handle, const std::uint8_t* bbox_mask,
    std::size_t num_action_scale, double action_unit, double cover_penalty,
    double pen_rate, std::size_t max_steps, const double* action_scales,
    std::intptr_t* out_actions, double* out_best_rewards,
    double* out_applied_rewards, std::uint8_t* out_next_mask,
    std::size_t* out_steps, double* out_last_score);
extern "C" int smart_manifold_state_rollback(void* handle);

namespace {

void require_status(int status, const char* message) {
  if (status == 0) {
    throw std::runtime_error(message);
  }
}

double finite_or_throw(double value, const char* message) {
  if (!std::isfinite(value)) {
    throw std::runtime_error(message);
  }
  return value;
}

std::string float_bits_key(double value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(value));
  std::ostringstream stream;
  stream << std::hex << std::setw(16) << std::setfill('0') << bits;
  return stream.str();
}

void check_row_width(const std::vector<std::vector<double>>& rows,
                     std::size_t width, const char* label) {
  for (const auto& row : rows) {
    if (row.size() != width) {
      throw std::runtime_error(std::string(label) + " rows must have width " +
                               std::to_string(width));
    }
  }
}

void check_row_width(const std::vector<std::vector<std::size_t>>& rows,
                     std::size_t width, const char* label) {
  for (const auto& row : rows) {
    if (row.size() != width) {
      throw std::runtime_error(std::string(label) + " rows must have width " +
                               std::to_string(width));
    }
  }
}

std::vector<double> flatten_double_rows(
    const std::vector<std::vector<double>>& rows, std::size_t width,
    const char* label) {
  check_row_width(rows, width, label);
  std::vector<double> out;
  out.reserve(rows.size() * width);
  for (const auto& row : rows) {
    out.insert(out.end(), row.begin(), row.end());
  }
  return out;
}

std::vector<std::vector<double>> unflatten_double_rows(
    const std::vector<double>& flat, std::size_t width) {
  std::vector<std::vector<double>> out;
  out.reserve(flat.size() / width);
  for (std::size_t idx = 0; idx < flat.size(); idx += width) {
    out.emplace_back(flat.begin() + static_cast<std::ptrdiff_t>(idx),
                     flat.begin() + static_cast<std::ptrdiff_t>(idx + width));
  }
  return out;
}

std::vector<float> flatten_vertices_float(
    const std::vector<std::vector<double>>& vertices) {
  check_row_width(vertices, 3, "vertices");
  std::vector<float> out;
  out.reserve(vertices.size() * 3);
  for (const auto& row : vertices) {
    for (double value : row) {
      out.push_back(static_cast<float>(value));
    }
  }
  return out;
}

std::vector<std::uint32_t> flatten_faces_uint32(
    const std::vector<std::vector<std::size_t>>& faces) {
  std::vector<std::uint32_t> out;
  out.reserve(faces.size() * 3);
  for (const auto& row : faces) {
    if (row.size() != 3) {
      throw std::runtime_error("faces rows must have width 3");
    }
    for (std::size_t value : row) {
      out.push_back(static_cast<std::uint32_t>(value));
    }
  }
  return out;
}

std::vector<std::size_t> flatten_size_t_rows(
    const std::vector<std::vector<std::size_t>>& rows, std::size_t width,
    const char* label) {
  std::vector<std::size_t> out;
  out.reserve(rows.size() * width);
  for (const auto& row : rows) {
    if (row.size() != width) {
      throw std::runtime_error(std::string(label) + " rows must have width " +
                               std::to_string(width));
    }
    out.insert(out.end(), row.begin(), row.end());
  }
  return out;
}

std::vector<std::vector<std::size_t>> unflatten_size_t_rows(
    const std::vector<std::size_t>& flat, std::size_t width) {
  std::vector<std::vector<std::size_t>> out;
  out.reserve(flat.size() / width);
  for (std::size_t idx = 0; idx < flat.size(); idx += width) {
    out.emplace_back(flat.begin() + static_cast<std::ptrdiff_t>(idx),
                     flat.begin() + static_cast<std::ptrdiff_t>(idx + width));
  }
  return out;
}

std::vector<std::uint8_t> bool_mask_to_u8(const std::vector<bool>& mask) {
  std::vector<std::uint8_t> out;
  out.reserve(mask.size());
  for (bool value : mask) {
    out.push_back(value ? 1u : 0u);
  }
  return out;
}

std::vector<float> flatten_bridge_box_vertices(
    const std::vector<std::vector<std::vector<double>>>& box_vertices) {
  std::vector<float> out;
  out.reserve(box_vertices.size() * 8 * 3);
  for (const auto& box : box_vertices) {
    if (box.size() != 8) {
      throw std::runtime_error("each bridge box must contain exactly 8 vertices");
    }
    for (const auto& point : box) {
      if (point.size() != 3) {
        throw std::runtime_error("bridge box vertices must be [x, y, z] rows");
      }
      out.push_back(static_cast<float>(point[0]));
      out.push_back(static_cast<float>(point[1]));
      out.push_back(static_cast<float>(point[2]));
    }
  }
  return out;
}

std::vector<float> flatten_bridge_oriented_box_vertices(
    const std::vector<std::vector<double>>& bounds,
    const std::vector<std::vector<double>>& rotations) {
  if (bounds.size() != rotations.size()) {
    throw std::runtime_error("bounds and rotations must have the same length");
  }
  check_row_width(bounds, 6, "bounds");
  check_row_width(rotations, 9, "rotations");
  std::vector<float> out;
  out.reserve(bounds.size() * 8 * 3);
  for (std::size_t box_idx = 0; box_idx < bounds.size(); ++box_idx) {
    const auto& row = bounds[box_idx];
    const auto& rot = rotations[box_idx];
    const double lengths[3] = {
        row[3] - row[0],
        row[4] - row[1],
        row[5] - row[2],
    };
    const double base[3] = {
        row[0] * rot[0] + row[1] * rot[3] + row[2] * rot[6],
        row[0] * rot[1] + row[1] * rot[4] + row[2] * rot[7],
        row[0] * rot[2] + row[1] * rot[5] + row[2] * rot[8],
    };
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {
          const double point[3] = {
              base[0] + rot[0] * i * lengths[0] +
                            rot[3] * j * lengths[1] +
                            rot[6] * k * lengths[2],
              base[1] + rot[1] * i * lengths[0] +
                            rot[4] * j * lengths[1] +
                            rot[7] * k * lengths[2],
              base[2] + rot[2] * i * lengths[0] +
                            rot[5] * j * lengths[1] +
                            rot[8] * k * lengths[2],
          };
          out.push_back(static_cast<float>(point[0]));
          out.push_back(static_cast<float>(point[1]));
          out.push_back(static_cast<float>(point[2]));
        }
      }
    }
  }
  return out;
}

py::tuple native_box_mesh_py(double x, double y, double z, double lx,
                             double ly, double lz,
                             const std::vector<double>& rotation) {
  if (rotation.size() != 9) {
    throw std::runtime_error("rotation must have 9 values");
  }
  const double base[3] = {x, y, z};
  const double lengths[3] = {lx, ly, lz};
  std::vector<std::vector<double>> vertices;
  vertices.reserve(8);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        vertices.push_back({
            base[0] + rotation[0] * i * lengths[0] +
                          rotation[3] * j * lengths[1] +
                          rotation[6] * k * lengths[2],
            base[1] + rotation[1] * i * lengths[0] +
                          rotation[4] * j * lengths[1] +
                          rotation[7] * k * lengths[2],
            base[2] + rotation[2] * i * lengths[0] +
                          rotation[5] * j * lengths[1] +
                          rotation[8] * k * lengths[2],
        });
      }
    }
  }
  const std::vector<std::vector<std::size_t>> faces = {
      {1, 3, 0}, {1, 5, 7}, {4, 6, 7}, {0, 2, 6},
      {2, 3, 7}, {0, 5, 1}, {3, 2, 0}, {1, 7, 3},
      {4, 7, 5}, {0, 6, 4}, {2, 7, 6}, {0, 4, 5},
  };
  return py::make_tuple(vertices, faces);
}

int parse_volume_method(const std::string& volume_method) {
  if (volume_method == "mesh") {
    return 0;
  }
  if (volume_method == "properties" || volume_method == "get_properties") {
    return 1;
  }
  throw std::runtime_error("volume_method must be 'mesh' or 'properties'");
}

std::vector<double> native_action_scales_py(std::size_t num_action_scale) {
  std::vector<double> out(num_action_scale);
  require_status(smart_native_action_scales(num_action_scale, out.data()),
                 "native action-scale generation failed");
  return out;
}

std::vector<std::vector<std::size_t>> native_action_indices_py(
    std::size_t num_bbox, std::size_t num_action_scale) {
  const std::size_t n_actions =
      smart_native_action_count(num_bbox, num_action_scale);
  std::vector<std::size_t> flat(n_actions * 3);
  require_status(
      smart_native_action_indices(num_bbox, num_action_scale, flat.data()),
      "native action-index generation failed");
  std::vector<std::vector<std::size_t>> out;
  out.reserve(n_actions);
  for (std::size_t idx = 0; idx < n_actions; ++idx) {
    out.push_back({flat[idx * 3], flat[idx * 3 + 1], flat[idx * 3 + 2]});
  }
  return out;
}

std::vector<std::size_t> native_opposite_actions_py(
    std::size_t num_bbox, std::size_t num_action_scale) {
  const std::size_t n_actions =
      smart_native_action_count(num_bbox, num_action_scale);
  std::vector<std::size_t> out(n_actions);
  require_status(
      smart_native_opposite_actions(num_bbox, num_action_scale, out.data()),
      "native opposite-action generation failed");
  return out;
}

std::vector<bool> native_child_action_mask_py(
    std::size_t num_actions, std::size_t action, std::size_t num_action_scale,
    py::object parent_mask) {
  std::vector<std::uint8_t> parent;
  const std::uint8_t* parent_ptr = nullptr;
  if (!parent_mask.is_none()) {
    const std::vector<bool> parent_bool = parent_mask.cast<std::vector<bool>>();
    if (parent_bool.size() != num_actions) {
      throw std::runtime_error("parent_mask length must match num_actions");
    }
    parent.reserve(parent_bool.size());
    for (bool value : parent_bool) {
      parent.push_back(value ? 1u : 0u);
    }
    parent_ptr = parent.data();
  }
  std::vector<std::uint8_t> raw(num_actions);
  require_status(smart_native_child_action_mask(num_actions, action,
                                                num_action_scale, parent_ptr,
                                                raw.data()),
                 "native child-action mask failed");
  std::vector<bool> out;
  out.reserve(raw.size());
  for (std::uint8_t value : raw) {
    out.push_back(value != 0);
  }
  return out;
}

double native_discounted_reward_py(const std::vector<double>& rewards,
                                   double gamma) {
  return smart_native_discounted_reward(rewards.data(), rewards.size(), gamma);
}

std::size_t native_ucb_best_count_py(
    std::size_t parent_visits, const std::vector<double>& child_qs,
    const std::vector<std::size_t>& child_visits, double exp_weight) {
  if (child_qs.size() != child_visits.size()) {
    throw std::runtime_error("child_qs and child_visits must have same length");
  }
  std::size_t out = 0;
  require_status(smart_native_ucb_best_count(
                     parent_visits, child_qs.data(), child_visits.data(),
                     child_qs.size(), exp_weight, &out),
                 "native UCB best-count failed");
  return out;
}

std::size_t native_best_ucb_child_py(
    std::size_t parent_visits, const std::vector<double>& child_qs,
    const std::vector<std::size_t>& child_visits, double exp_weight,
    std::size_t tie_pick) {
  if (child_qs.size() != child_visits.size()) {
    throw std::runtime_error("child_qs and child_visits must have same length");
  }
  std::size_t out = 0;
  require_status(smart_native_best_ucb_child(
                     parent_visits, child_qs.data(), child_visits.data(),
                     child_qs.size(), exp_weight, tie_pick, &out),
                 "native UCB selection failed");
  return out;
}

double native_prob_skip_exploration_py(
    double parent_reward, const std::vector<double>& child_rewards,
    const std::vector<double>& child_qs, double best_reward, double skip_rate) {
  if (child_rewards.size() != child_qs.size()) {
    throw std::runtime_error("child_rewards and child_qs must have same length");
  }
  double out = 0.0;
  require_status(smart_native_prob_skip_exploration(
                     parent_reward, child_rewards.data(), child_qs.data(),
                     child_qs.size(), best_reward, skip_rate, &out),
                 "native PNS skip probability failed");
  return out;
}

std::vector<double> native_softmax_scaled_py(const std::vector<double>& values,
                                             double scale) {
  std::vector<double> out(values.size());
  require_status(
      smart_native_softmax_scaled(values.data(), values.size(), scale, out.data()),
      "native softmax failed");
  return out;
}

std::vector<double> native_weighted_action_scores_py(
    const std::vector<double>& base_rewards,
    const std::vector<double>& prior_logits,
    const std::vector<double>& value_logits,
    double base_scale,
    double prior_weight,
    double value_weight) {
  const std::size_t n = base_rewards.size();
  if (prior_logits.size() != n || value_logits.size() != n) {
    throw std::runtime_error(
        "base_rewards, prior_logits, and value_logits must have same length");
  }
  std::vector<double> out;
  out.reserve(n);
  for (std::size_t idx = 0; idx < n; ++idx) {
    double score = base_rewards[idx] * base_scale +
                   prior_logits[idx] * prior_weight +
                   value_logits[idx] * value_weight;
    if (std::isnan(score)) {
      score = -std::numeric_limits<double>::infinity();
    }
    out.push_back(score);
  }
  return out;
}

std::vector<std::size_t> native_top_k_actions_py(
    const std::vector<std::size_t>& actions,
    const std::vector<double>& scores,
    std::size_t top_k) {
  if (actions.size() != scores.size()) {
    throw std::runtime_error("actions and scores must have same length");
  }
  std::vector<std::size_t> order(actions.size());
  for (std::size_t idx = 0; idx < order.size(); ++idx) {
    order[idx] = idx;
  }
  std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    const double l_score = scores[left];
    const double r_score = scores[right];
    if (l_score != r_score) {
      return l_score > r_score;
    }
    return actions[left] < actions[right];
  });
  if (top_k > order.size()) {
    top_k = order.size();
  }
  std::vector<std::size_t> out;
  out.reserve(top_k);
  for (std::size_t idx = 0; idx < top_k; ++idx) {
    out.push_back(actions[order[idx]]);
  }
  return out;
}

std::size_t native_best_score_action_py(const std::vector<std::size_t>& actions,
                                        const std::vector<double>& scores,
                                        std::size_t tie_pick) {
  if (actions.empty()) {
    throw std::runtime_error("actions must not be empty");
  }
  if (actions.size() != scores.size()) {
    throw std::runtime_error("actions and scores must have same length");
  }
  double best = -std::numeric_limits<double>::infinity();
  for (double score : scores) {
    if (score > best) {
      best = score;
    }
  }
  std::vector<std::size_t> ties;
  ties.reserve(actions.size());
  for (std::size_t idx = 0; idx < actions.size(); ++idx) {
    if (scores[idx] == best) {
      ties.push_back(actions[idx]);
    }
  }
  if (ties.empty()) {
    return actions[0];
  }
  return ties[tie_pick % ties.size()];
}

std::pair<std::size_t, std::size_t> action_trace_fields_cpp(
    std::size_t action, std::size_t num_action_scale) {
  const std::size_t per_bbox = 6 * num_action_scale + 1;
  const std::size_t bbox_idx = action / per_bbox;
  const std::size_t local = action % per_bbox;
  if (local == per_bbox - 1) {
    return {bbox_idx, 6};
  }
  return {bbox_idx, local / num_action_scale};
}

std::vector<std::size_t> native_diverse_escape_actions_py(
    const std::vector<std::size_t>& actions,
    const std::vector<double>& scores,
    const std::vector<std::size_t>& primary_keep,
    std::size_t num_action_scale,
    std::size_t escape_top_k) {
  if (actions.size() != scores.size()) {
    throw std::runtime_error("actions and scores must have same length");
  }
  if (num_action_scale == 0 || escape_top_k == 0) {
    return {};
  }
  std::vector<std::size_t> order(actions.size());
  for (std::size_t idx = 0; idx < order.size(); ++idx) {
    order[idx] = idx;
  }
  std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    if (scores[left] != scores[right]) {
      return scores[left] > scores[right];
    }
    return actions[left] < actions[right];
  });

  std::set<std::size_t> primary(primary_keep.begin(), primary_keep.end());
  std::set<std::size_t> selected;
  std::set<std::size_t> used_bboxes;
  std::set<std::size_t> used_coords;
  for (std::size_t action : primary_keep) {
    const auto fields = action_trace_fields_cpp(action, num_action_scale);
    used_bboxes.insert(fields.first);
    used_coords.insert(fields.second);
  }

  std::vector<std::size_t> out;
  out.reserve(escape_top_k);
  for (std::size_t pos : order) {
    const std::size_t action = actions[pos];
    if (primary.count(action) != 0) {
      continue;
    }
    const auto fields = action_trace_fields_cpp(action, num_action_scale);
    if (used_bboxes.count(fields.first) != 0 &&
        used_coords.count(fields.second) != 0 &&
        out.size() + 1 < escape_top_k) {
      continue;
    }
    out.push_back(action);
    selected.insert(action);
    used_bboxes.insert(fields.first);
    used_coords.insert(fields.second);
    if (out.size() >= escape_top_k) {
      return out;
    }
  }
  for (std::size_t pos : order) {
    const std::size_t action = actions[pos];
    if (primary.count(action) != 0 || selected.count(action) != 0) {
      continue;
    }
    out.push_back(action);
    selected.insert(action);
    if (out.size() >= escape_top_k) {
      break;
    }
  }
  return out;
}

std::vector<double> native_add_puct_prior_py(
    const std::vector<double>& uct_scores,
    const std::vector<double>& prior_logits,
    const std::vector<std::size_t>& child_visits,
    std::size_t parent_visits,
    double prior_weight) {
  const std::size_t n = uct_scores.size();
  if (prior_logits.size() != n || child_visits.size() != n) {
    throw std::runtime_error(
        "uct_scores, prior_logits, and child_visits must have same length");
  }
  std::vector<double> out = uct_scores;
  if (n == 0 || parent_visits == 0 || prior_weight == 0.0) {
    return out;
  }
  double max_logit = -std::numeric_limits<double>::infinity();
  for (double value : prior_logits) {
    if (value > max_logit) {
      max_logit = value;
    }
  }
  std::vector<double> probs(n, 0.0);
  double total = 0.0;
  for (std::size_t idx = 0; idx < n; ++idx) {
    const double shifted = prior_logits[idx] - max_logit;
    const double prob = std::exp(shifted);
    probs[idx] = prob;
    total += prob;
  }
  if (total <= 0.0) {
    return out;
  }
  const double sqrt_parent = std::sqrt(static_cast<double>(parent_visits));
  for (std::size_t idx = 0; idx < n; ++idx) {
    const double prob = probs[idx] / total;
    out[idx] += prior_weight * prob * sqrt_parent /
                (1.0 + static_cast<double>(child_visits[idx]));
  }
  return out;
}

bool dict_contains_key(const py::dict& context, const char* key) {
  return context.contains(py::str(key));
}

py::object dict_get_object(const py::dict& context, const char* key) {
  if (!dict_contains_key(context, key)) {
    return py::none();
  }
  return py::reinterpret_borrow<py::object>(context[py::str(key)]);
}

bool object_truthy(const py::object& value) {
  if (value.is_none()) {
    return false;
  }
  return PyObject_IsTrue(value.ptr()) == 1;
}

double context_double_or(const py::dict& context,
                         const char* key,
                         double default_value) {
  py::object value = dict_get_object(context, key);
  if (!object_truthy(value)) {
    return default_value;
  }
  return value.cast<double>();
}

double context_double_default(const py::dict& context,
                              const char* key,
                              double default_value) {
  py::object value = dict_get_object(context, key);
  if (value.is_none()) {
    return default_value;
  }
  return value.cast<double>();
}

bool context_bool_fallback(const py::dict& context,
                           const char* key,
                           const char* fallback_key,
                           bool default_value) {
  py::object value = dict_get_object(context, key);
  if (value.is_none()) {
    value = dict_get_object(context, fallback_key);
  }
  if (value.is_none()) {
    return default_value;
  }
  return object_truthy(value);
}

std::string context_string_or(const py::dict& context, const char* key) {
  py::object value = dict_get_object(context, key);
  if (value.is_none()) {
    return std::string();
  }
  return py::str(value).cast<std::string>();
}

std::vector<double> linear_features_from_context_cpp(
    const py::dict& context,
    const std::vector<std::string>& categories) {
  const double bvs = context_double_or(context, "bvs", 1.0);
  const double step = context_double_or(context, "step", 0.0);
  const bool has_max_step = dict_contains_key(context, "max_step");
  const double max_step =
      std::max(context_double_or(context, "max_step", 0.0), 1.0);
  const double step_fraction = has_max_step ? step / max_step : step / 150.0;
  const double num_bbox = context_double_or(context, "num_bbox", 0.0);
  const double action_unit = context_double_or(context, "action_unit", 0.0);
  const double cover_penalty =
      context_double_or(context, "cover_penalty", 100.0);
  const double pen_rate = context_double_or(context, "pen_rate", 1.0);
  py::object not_updated_obj = dict_get_object(context, "mcts_not_updated");
  if (not_updated_obj.is_none()) {
    not_updated_obj = dict_get_object(context, "not_updated");
  }
  const double mcts_not_updated =
      object_truthy(not_updated_obj) ? not_updated_obj.cast<double>() : 0.0;
  py::object best_reward_obj = dict_get_object(context, "mcts_best_reward");
  if (best_reward_obj.is_none()) {
    best_reward_obj = dict_get_object(context, "best_reward");
  }
  const double mcts_best_reward =
      object_truthy(best_reward_obj) ? best_reward_obj.cast<double>() : 0.0;
  const double mcts_escape_active =
      context_bool_fallback(context, "mcts_escape_active", "escape_active",
                            false)
          ? 1.0
          : 0.0;
  const std::string category = context_string_or(context, "category");
  std::vector<double> features = {
      1.0,
      bvs,
      bvs - 1.0,
      std::abs(bvs - 1.0),
      step_fraction,
      action_unit,
      num_bbox / 32.0,
      cover_penalty / 100.0,
      pen_rate,
      mcts_not_updated / std::max(max_step, 1.0),
      mcts_best_reward,
      mcts_escape_active,
  };
  features.reserve(features.size() + categories.size());
  for (const std::string& item : categories) {
    features.push_back(category == item ? 1.0 : 0.0);
  }
  return features;
}

double action_scale_value_cpp(std::size_t scale_idx,
                              std::size_t num_action_scale) {
  num_action_scale = std::max<std::size_t>(num_action_scale, 1);
  const std::size_t half = std::max<std::size_t>(num_action_scale / 2, 1);
  if (scale_idx < half) {
    return -std::pow(2.0, static_cast<double>(half - 1 - scale_idx));
  }
  return std::pow(2.0, static_cast<double>(scale_idx - half));
}

std::vector<double> bbox_bounds_from_context_cpp(const py::dict& context,
                                                 std::size_t bbox_idx,
                                                 bool* found) {
  static const char* keys[6] = {
      "bbox_min_x", "bbox_min_y", "bbox_min_z",
      "bbox_max_x", "bbox_max_y", "bbox_max_z",
  };
  bool has_direct = true;
  for (const char* key : keys) {
    has_direct = has_direct && dict_contains_key(context, key);
  }
  if (has_direct) {
    if (found != nullptr) {
      *found = true;
    }
    return {
        context_double_or(context, "bbox_min_x", 0.0),
        context_double_or(context, "bbox_min_y", 0.0),
        context_double_or(context, "bbox_min_z", 0.0),
        context_double_or(context, "bbox_max_x", 0.0),
        context_double_or(context, "bbox_max_y", 0.0),
        context_double_or(context, "bbox_max_z", 0.0),
    };
  }

  py::object bounds_obj = dict_get_object(context, "bbox_bounds");
  if (!bounds_obj.is_none() && py::isinstance<py::sequence>(bounds_obj) &&
      !py::isinstance<py::str>(bounds_obj)) {
    py::sequence bounds_seq = py::reinterpret_borrow<py::sequence>(bounds_obj);
    if (bbox_idx < static_cast<std::size_t>(py::len(bounds_seq))) {
      py::object row_obj =
          py::reinterpret_borrow<py::object>(bounds_seq[bbox_idx]);
      if (py::isinstance<py::sequence>(row_obj) &&
          !py::isinstance<py::str>(row_obj)) {
        py::sequence row = py::reinterpret_borrow<py::sequence>(row_obj);
        if (py::len(row) >= 6) {
          if (found != nullptr) {
            *found = true;
          }
          std::vector<double> out;
          out.reserve(6);
          for (std::size_t idx = 0; idx < 6; ++idx) {
            out.push_back(
                py::reinterpret_borrow<py::object>(row[idx]).cast<double>());
          }
          return out;
        }
      }
    }
  }
  if (found != nullptr) {
    *found = false;
  }
  return {};
}

std::vector<double> bbox_action_geometry_features_cpp(
    const py::dict& context,
    std::size_t bbox_idx,
    std::size_t coord_idx,
    double signed_delta) {
  bool has_bounds = false;
  const std::vector<double> bounds =
      bbox_bounds_from_context_cpp(context, bbox_idx, &has_bounds);
  const double volume_sum =
      std::max(context_double_or(context, "volume_sum", 0.0), 1.0e-12);
  std::vector<double> dims(3, 0.0);
  std::vector<double> center(3, 0.0);
  double bbox_volume = 0.0;
  double valid = 0.0;
  double new_volume = 0.0;
  double invalid_after = 0.0;
  if (!has_bounds) {
    dims = {
        context_double_or(context, "bbox_dim_x", 0.0),
        context_double_or(context, "bbox_dim_y", 0.0),
        context_double_or(context, "bbox_dim_z", 0.0),
    };
    center = {
        context_double_or(context, "bbox_center_x", 0.0),
        context_double_or(context, "bbox_center_y", 0.0),
        context_double_or(context, "bbox_center_z", 0.0),
    };
    bbox_volume = context_double_or(context, "bbox_volume", 0.0);
    valid = bbox_volume > 0.0 && dims[0] > 0.0 && dims[1] > 0.0 &&
                    dims[2] > 0.0
                ? 1.0
                : 0.0;
    new_volume = context_double_default(
        context, "action_new_bbox_volume", bbox_volume);
    invalid_after =
        context_bool_fallback(context, "action_invalid_after",
                              "action_invalid_after", false)
            ? 1.0
            : 0.0;
  } else {
    dims = {
        std::max(0.0, bounds[3] - bounds[0]),
        std::max(0.0, bounds[4] - bounds[1]),
        std::max(0.0, bounds[5] - bounds[2]),
    };
    center = {
        0.5 * (bounds[0] + bounds[3]),
        0.5 * (bounds[1] + bounds[4]),
        0.5 * (bounds[2] + bounds[5]),
    };
    bbox_volume = dims[0] * dims[1] * dims[2];
    valid = bbox_volume > 0.0 ? 1.0 : 0.0;
    std::vector<double> candidate = bounds;
    if (coord_idx < 6) {
      candidate[coord_idx] += signed_delta;
    }
    const double new_dims[3] = {
        candidate[3] - candidate[0],
        candidate[4] - candidate[1],
        candidate[5] - candidate[2],
    };
    invalid_after =
        (new_dims[0] <= 0.0 || new_dims[1] <= 0.0 || new_dims[2] <= 0.0)
            ? 1.0
            : 0.0;
    new_volume =
        invalid_after != 0.0 ? 0.0 : new_dims[0] * new_dims[1] * new_dims[2];
  }
  const double min_dim = std::min({dims[0], dims[1], dims[2]});
  const double max_dim = std::max({dims[0], dims[1], dims[2]});
  const double extent_ratio = min_dim / std::max(max_dim, 1.0e-12);
  const double bbox_volume_ratio = bbox_volume / volume_sum;
  const double new_volume_ratio = new_volume / volume_sum;
  const double action_delta = coord_idx < 6 ? signed_delta : 0.0;
  double shrinks = 0.0;
  double expands = 0.0;
  if (coord_idx < 3) {
    shrinks = action_delta > 0.0 ? 1.0 : 0.0;
    expands = action_delta < 0.0 ? 1.0 : 0.0;
  } else if (coord_idx < 6) {
    shrinks = action_delta < 0.0 ? 1.0 : 0.0;
    expands = action_delta > 0.0 ? 1.0 : 0.0;
  }
  return {
      coord_idx < 3 ? 1.0 : 0.0,
      coord_idx >= 3 && coord_idx < 6 ? 1.0 : 0.0,
      valid,
      dims[0],
      dims[1],
      dims[2],
      bbox_volume_ratio,
      std::log1p(std::max(bbox_volume_ratio, 0.0)),
      center[0],
      center[1],
      center[2],
      extent_ratio,
      action_delta,
      std::abs(action_delta),
      shrinks,
      expands,
      new_volume_ratio,
      new_volume_ratio - bbox_volume_ratio,
      invalid_after,
  };
}

std::vector<double> action_features_from_context_cpp(
    const py::dict& context,
    std::size_t action,
    std::size_t action_num_action_scale,
    std::size_t model_num_action_scale,
    const std::vector<std::string>& categories) {
  action_num_action_scale =
      std::max<std::size_t>(action_num_action_scale, 1);
  model_num_action_scale =
      std::max<std::size_t>(model_num_action_scale, 1);
  const std::size_t per_bbox = 6 * action_num_action_scale + 1;
  const std::size_t bbox_idx = action / per_bbox;
  const std::size_t local = action % per_bbox;
  const bool is_recenter = local == per_bbox - 1;
  const std::size_t coord_idx =
      is_recenter ? 6 : local / action_num_action_scale;
  const std::size_t scale_idx =
      is_recenter ? 0 : local % action_num_action_scale;
  const std::size_t record_num_bbox =
      static_cast<std::size_t>(std::max(context_double_or(context, "num_bbox", 0.0), 0.0));
  const std::size_t num_bbox =
      std::max<std::size_t>({record_num_bbox, bbox_idx + 1, 1});
  const double signed_scale =
      coord_idx < 6 ? action_scale_value_cpp(scale_idx, action_num_action_scale)
                    : 0.0;
  const double max_scale =
      std::max(std::abs(action_scale_value_cpp(0, action_num_action_scale)),
               1.0);
  const double scale_idx_norm =
      action_num_action_scale > 1
          ? static_cast<double>(scale_idx) /
                static_cast<double>(action_num_action_scale - 1)
          : 0.0;
  std::vector<double> features =
      linear_features_from_context_cpp(context, categories);
  const double action_unit = context_double_or(context, "action_unit", 0.0);
  const double signed_delta = signed_scale * action_unit;
  const std::vector<double> geo = bbox_action_geometry_features_cpp(
      context, bbox_idx, coord_idx, signed_delta);
  features.reserve(features.size() + 30 + 3 + 7 + model_num_action_scale);
  features.insert(features.end(),
                  {
                      static_cast<double>(bbox_idx) /
                          static_cast<double>(num_bbox),
                      (static_cast<double>(bbox_idx) -
                       static_cast<double>(num_bbox - 1) * 0.5) /
                          std::max(static_cast<double>(num_bbox), 1.0),
                      static_cast<double>(num_bbox - 1 - bbox_idx) /
                          static_cast<double>(num_bbox),
                      static_cast<double>(local) /
                          static_cast<double>(std::max<std::size_t>(
                              per_bbox - 1, 1)),
                      coord_idx < 6 ? 1.0 : 0.0,
                      coord_idx >= 6 ? 1.0 : 0.0,
                      coord_idx < 6 && coord_idx % 2 == 0 ? 1.0 : 0.0,
                      coord_idx < 6 && coord_idx % 2 == 1 ? 1.0 : 0.0,
                      signed_scale / max_scale,
                      std::abs(signed_scale) / max_scale,
                      scale_idx_norm,
                  });
  features.insert(features.end(), geo.begin(), geo.end());
  const std::intptr_t axis_idx =
      coord_idx < 6 ? static_cast<std::intptr_t>(coord_idx / 2) : -1;
  for (std::size_t idx = 0; idx < 3; ++idx) {
    features.push_back(axis_idx == static_cast<std::intptr_t>(idx) ? 1.0
                                                                   : 0.0);
  }
  for (std::size_t idx = 0; idx < 7; ++idx) {
    features.push_back(coord_idx == idx ? 1.0 : 0.0);
  }
  for (std::size_t idx = 0; idx < model_num_action_scale; ++idx) {
    features.push_back(scale_idx == idx && coord_idx < 6 ? 1.0 : 0.0);
  }
  return features;
}

std::vector<double> action_mlp_hidden_cpp(
    const std::vector<double>& features,
    const std::vector<std::vector<double>>& input_weights,
    const std::vector<double>& hidden_bias) {
  std::vector<double> hidden = hidden_bias;
  for (std::size_t feat_idx = 0;
       feat_idx < features.size() && feat_idx < input_weights.size();
       ++feat_idx) {
    const auto& row = input_weights[feat_idx];
    for (std::size_t hidden_idx = 0;
         hidden_idx < hidden.size() && hidden_idx < row.size(); ++hidden_idx) {
      hidden[hidden_idx] += features[feat_idx] * row[hidden_idx];
    }
  }
  for (double& value : hidden) {
    value = std::tanh(value);
  }
  return hidden;
}

py::tuple native_action_mlp_logits_values_py(
    const std::vector<std::size_t>& actions,
    std::size_t action_num_action_scale,
    std::size_t model_num_action_scale,
    const py::dict& context,
    const std::vector<std::string>& categories,
    const std::vector<std::vector<double>>& action_input_weights,
    const std::vector<double>& action_hidden_bias,
    const std::vector<double>& action_output_weights,
    double action_output_bias,
    const std::vector<double>& action_value_output_weights,
    double action_value_output_bias) {
  if (action_input_weights.empty() || action_hidden_bias.empty() ||
      action_output_weights.empty()) {
    throw std::runtime_error(
        "action-level MLP weights are required for native inference");
  }
  const std::size_t model_scale =
      std::max<std::size_t>(model_num_action_scale, action_num_action_scale);
  std::vector<double> logits;
  std::vector<double> values;
  logits.reserve(actions.size());
  values.reserve(actions.size());
  for (std::size_t action : actions) {
    const std::vector<double> features = action_features_from_context_cpp(
        context, action, action_num_action_scale, model_scale, categories);
    const std::vector<double> hidden = action_mlp_hidden_cpp(
        features, action_input_weights, action_hidden_bias);
    double logit = action_output_bias;
    for (std::size_t idx = 0;
         idx < hidden.size() && idx < action_output_weights.size(); ++idx) {
      logit += hidden[idx] * action_output_weights[idx];
    }
    logits.push_back(logit);
    double value = action_value_output_bias;
    if (action_value_output_weights.empty()) {
      value = 0.0;
    } else {
      for (std::size_t idx = 0;
           idx < hidden.size() && idx < action_value_output_weights.size();
           ++idx) {
        value += hidden[idx] * action_value_output_weights[idx];
      }
    }
    values.push_back(value);
  }
  return py::make_tuple(logits, values);
}

class SmartCppActionMlpPolicy {
 public:
  SmartCppActionMlpPolicy(
      std::vector<std::string> categories,
      std::vector<std::vector<double>> action_input_weights,
      std::vector<double> action_hidden_bias,
      std::vector<double> action_output_weights,
      double action_output_bias,
      std::vector<double> action_value_output_weights,
      double action_value_output_bias)
      : categories_(std::move(categories)),
        action_input_weights_(std::move(action_input_weights)),
        action_hidden_bias_(std::move(action_hidden_bias)),
        action_output_weights_(std::move(action_output_weights)),
        action_output_bias_(action_output_bias),
        action_value_output_weights_(std::move(action_value_output_weights)),
        action_value_output_bias_(action_value_output_bias) {
    if (action_input_weights_.empty() || action_hidden_bias_.empty() ||
        action_output_weights_.empty()) {
      throw std::runtime_error(
          "action-level MLP weights are required for ActionMlpPolicy");
    }
  }

  py::tuple logits_values(const std::vector<std::size_t>& actions,
                          std::size_t action_num_action_scale,
                          std::size_t model_num_action_scale,
                          const py::dict& context) const {
    return native_action_mlp_logits_values_py(
        actions, action_num_action_scale, model_num_action_scale, context,
        categories_, action_input_weights_, action_hidden_bias_,
        action_output_weights_, action_output_bias_,
        action_value_output_weights_, action_value_output_bias_);
  }

  std::size_t feature_dim() const { return action_input_weights_.size(); }

  std::size_t hidden_size() const { return action_hidden_bias_.size(); }

  bool has_value_head() const { return !action_value_output_weights_.empty(); }

 private:
  std::vector<std::string> categories_;
  std::vector<std::vector<double>> action_input_weights_;
  std::vector<double> action_hidden_bias_;
  std::vector<double> action_output_weights_;
  double action_output_bias_ = 0.0;
  std::vector<double> action_value_output_weights_;
  double action_value_output_bias_ = 0.0;
};

std::size_t opposite_action_py(std::size_t action, std::size_t num_action_scale) {
  if (num_action_scale == 0) {
    throw std::runtime_error("num_action_scale must be positive");
  }
  const std::size_t per_bbox = 6 * num_action_scale + 1;
  const std::size_t local = action % per_bbox;
  const std::size_t bbox_idx = action / per_bbox;
  if (local == per_bbox - 1) {
    return action;
  }
  const std::size_t coord_idx = local / num_action_scale;
  const std::size_t scale_idx = local % num_action_scale;
  return bbox_idx * per_bbox + coord_idx * num_action_scale +
         (num_action_scale - 1 - scale_idx);
}

std::vector<bool> opposite_action_mask_py(std::size_t action,
                                          std::size_t num_bbox,
                                          std::size_t num_action_scale) {
  const std::size_t total =
      smart_native_action_count(num_bbox, num_action_scale);
  if (action >= total) {
    throw std::runtime_error("action is out of range");
  }
  std::vector<bool> out(total, false);
  out[opposite_action_py(action, num_action_scale)] = true;
  return out;
}

std::vector<std::size_t> untried_actions_py(const std::vector<bool>& action_mask) {
  std::vector<std::size_t> out;
  for (std::size_t idx = 0; idx < action_mask.size(); ++idx) {
    if (!action_mask[idx]) {
      out.push_back(idx);
    }
  }
  return out;
}

std::vector<bool> single_untried_action_mask_py(std::size_t total_actions,
                                                std::size_t action) {
  if (action >= total_actions) {
    throw std::runtime_error("action is out of range");
  }
  std::vector<bool> out(total_actions, true);
  out[action] = false;
  return out;
}

std::vector<double> ucb_scores_py(std::size_t parent_visits,
                                  const std::vector<double>& child_qs,
                                  const std::vector<std::size_t>& child_visits,
                                  double exp_weight) {
  if (child_qs.size() != child_visits.size()) {
    throw std::runtime_error("child_qs and child_visits must have same length");
  }
  std::vector<double> out;
  out.reserve(child_qs.size());
  if (parent_visits == 0) {
    out.assign(child_qs.size(), std::numeric_limits<double>::infinity());
    return out;
  }
  const double log_parent = std::log(static_cast<double>(parent_visits));
  for (std::size_t idx = 0; idx < child_qs.size(); ++idx) {
    if (child_visits[idx] == 0) {
      out.push_back(std::numeric_limits<double>::infinity());
    } else {
      out.push_back(child_qs[idx] + exp_weight *
                                      std::sqrt(2.0 * log_parent /
                                                static_cast<double>(child_visits[idx])));
    }
  }
  return out;
}

std::vector<std::size_t> ucb_best_indices_py(
    std::size_t parent_visits, const std::vector<double>& child_qs,
    const std::vector<std::size_t>& child_visits, double exp_weight) {
  const std::vector<double> scores =
      ucb_scores_py(parent_visits, child_qs, child_visits, exp_weight);
  if (scores.empty()) {
    return {};
  }
  const double best = *std::max_element(scores.begin(), scores.end());
  std::vector<std::size_t> out;
  for (std::size_t idx = 0; idx < scores.size(); ++idx) {
    if (scores[idx] == best) {
      out.push_back(idx);
    }
  }
  return out;
}

double manifold_mesh_volume_py(
    const std::vector<std::vector<double>>& vertices,
    const std::vector<std::vector<std::size_t>>& faces) {
  std::vector<float> flat_vertices = flatten_vertices_float(vertices);
  std::vector<std::uint32_t> flat_faces = flatten_faces_uint32(faces);
  return static_cast<double>(smart_manifold_mesh_volume(
      flat_vertices.data(), vertices.size(), flat_faces.data(), faces.size()));
}

double manifold_axis_box_intersection_volume_py(
    const std::vector<std::vector<double>>& vertices,
    const std::vector<std::vector<std::size_t>>& faces,
    const std::vector<double>& bounds) {
  if (bounds.size() != 6) {
    throw std::runtime_error("bounds must have length 6");
  }
  std::vector<float> flat_vertices = flatten_vertices_float(vertices);
  std::vector<std::uint32_t> flat_faces = flatten_faces_uint32(faces);
  float flat_bounds[6];
  for (std::size_t idx = 0; idx < 6; ++idx) {
    flat_bounds[idx] = static_cast<float>(bounds[idx]);
  }
  return static_cast<double>(smart_manifold_axis_box_intersection_volume(
      flat_vertices.data(), vertices.size(), flat_faces.data(), faces.size(),
      flat_bounds[0], flat_bounds[1], flat_bounds[2], flat_bounds[3],
      flat_bounds[4], flat_bounds[5]));
}

std::vector<double> native_bbox_volumes_py(
    const std::vector<std::vector<double>>& bounds);
std::vector<bool> native_bbox_valid_mask_py(
    const std::vector<std::vector<double>>& bounds);
std::vector<std::vector<double>> native_apply_axis_action_py(
    const std::vector<std::vector<double>>& bounds, std::size_t action,
    std::size_t num_action_scale, double action_unit);
std::vector<double> native_action_upper_rewards_py(
    const std::vector<std::vector<double>>& bounds, std::size_t num_action_scale,
    double action_unit, double volume_sum, double last_bbox_score);
std::vector<double> native_bbox_action_upper_rewards_py(
    const std::vector<std::vector<double>>& bounds, std::size_t bbox_idx,
    std::size_t num_action_scale, double action_unit, double volume_sum,
    double last_bbox_score);

class SmartCppBBoxState {
 public:
  SmartCppBBoxState(const std::vector<std::vector<double>>& bounds,
                    std::size_t num_action_scale, double action_unit,
                    double volume_sum, double last_bbox_score)
      : bounds_(bounds),
        num_action_scale_(num_action_scale),
        action_unit_(action_unit),
        volume_sum_(volume_sum),
        last_bbox_score_(last_bbox_score) {
    if (volume_sum <= 0.0) {
      throw std::runtime_error("volume_sum must be positive");
    }
    check_row_width(bounds_, 6, "bounds");
    refresh();
  }

  std::size_t num_bbox() const { return bounds_.size(); }

  std::size_t num_actions() const {
    return smart_native_action_count(bounds_.size(), num_action_scale_);
  }

  std::vector<std::vector<double>> bounds() const { return bounds_; }

  std::vector<double> volumes() const { return volumes_; }

  double total_volume() const { return total_volume_; }

  double bvs() const { return total_volume_ / volume_sum_; }

  std::vector<bool> valid_mask() const {
    return native_bbox_valid_mask_py(bounds_);
  }

  std::size_t valid_count() const {
    const std::vector<bool> mask = valid_mask();
    return static_cast<std::size_t>(
        std::count(mask.begin(), mask.end(), true));
  }

  double last_bbox_score() const { return last_bbox_score_; }

  void set_last_bbox_score(double last_bbox_score) {
    last_bbox_score_ = last_bbox_score;
  }

  SmartCppBBoxState with_last_bbox_score(double last_bbox_score) const {
    return SmartCppBBoxState(bounds_, num_action_scale_, action_unit_,
                             volume_sum_, last_bbox_score);
  }

  std::string state_key() const {
    std::string out;
    for (const auto& row : bounds_) {
      for (double value : row) {
        out += std::to_string(value);
        out += ",";
      }
      out += "|";
    }
    return out;
  }

  std::vector<double> action_upper_rewards() const {
    return native_action_upper_rewards_py(bounds_, num_action_scale_,
                                          action_unit_, volume_sum_,
                                          last_bbox_score_);
  }

  std::vector<double> bbox_action_upper_rewards(std::size_t bbox_idx) const {
    return native_bbox_action_upper_rewards_py(bounds_, bbox_idx,
                                               num_action_scale_, action_unit_,
                                               volume_sum_, last_bbox_score_);
  }

  std::vector<std::vector<double>> apply_axis_action(std::size_t action) const {
    return native_apply_axis_action_py(bounds_, action, num_action_scale_,
                                       action_unit_);
  }

  SmartCppBBoxState after_axis_action(std::size_t action) const {
    return SmartCppBBoxState(apply_axis_action(action), num_action_scale_,
                             action_unit_, volume_sum_, last_bbox_score_);
  }

  void apply_axis_action_in_place(std::size_t action) {
    bounds_ = apply_axis_action(action);
    refresh();
  }

 private:
  void refresh() {
    volumes_ = native_bbox_volumes_py(bounds_);
    total_volume_ = 0.0;
    for (double value : volumes_) {
      total_volume_ += value;
    }
  }

  std::vector<std::vector<double>> bounds_;
  std::size_t num_action_scale_ = 0;
  double action_unit_ = 0.0;
  double volume_sum_ = 0.0;
  double last_bbox_score_ = 0.0;
  std::vector<double> volumes_;
  double total_volume_ = 0.0;
};

std::vector<double> native_bbox_volumes_py(
    const std::vector<std::vector<double>>& bounds) {
  std::vector<double> flat = flatten_double_rows(bounds, 6, "bounds");
  std::vector<double> out(bounds.size());
  require_status(
      smart_native_bbox_volumes(flat.data(), bounds.size(), out.data()),
      "native bbox volume calculation failed");
  return out;
}

std::vector<bool> native_bbox_valid_mask_py(
    const std::vector<std::vector<double>>& bounds) {
  std::vector<double> flat = flatten_double_rows(bounds, 6, "bounds");
  std::vector<std::uint8_t> raw(bounds.size(), 0);
  require_status(
      smart_native_bbox_valid_mask(flat.data(), bounds.size(), raw.data()),
      "native bbox valid-mask calculation failed");
  std::vector<bool> out;
  out.reserve(raw.size());
  for (std::uint8_t value : raw) {
    out.push_back(value != 0);
  }
  return out;
}

double native_total_bbox_volume_py(
    const std::vector<std::vector<double>>& bounds) {
  std::vector<double> flat = flatten_double_rows(bounds, 6, "bounds");
  double out = 0.0;
  require_status(
      smart_native_total_bbox_volume(flat.data(), bounds.size(), &out),
      "native bbox total-volume calculation failed");
  return out;
}

std::vector<double> native_bbox_union_bounds_py(
    const std::vector<std::vector<double>>& bounds) {
  if (bounds.empty()) {
    throw std::runtime_error("bounds must not be empty");
  }
  std::vector<double> flat = flatten_double_rows(bounds, 6, "bounds");
  std::vector<double> out(6, 0.0);
  require_status(
      smart_native_bbox_union_bounds(flat.data(), bounds.size(), out.data()),
      "native bbox union-bounds calculation failed");
  return out;
}

double native_bbox_union_volume_py(
    const std::vector<std::vector<double>>& bounds) {
  if (bounds.empty()) {
    throw std::runtime_error("bounds must not be empty");
  }
  std::vector<double> flat = flatten_double_rows(bounds, 6, "bounds");
  double out = 0.0;
  require_status(
      smart_native_bbox_union_volume(flat.data(), bounds.size(), &out),
      "native bbox union-volume calculation failed");
  return out;
}

std::vector<bool> native_coverage_mask_py(
    const std::vector<std::vector<double>>& points,
    const std::vector<double>& bounds) {
  if (bounds.size() != 6) {
    throw std::runtime_error("bounds must have length 6");
  }
  std::vector<double> flat_points = flatten_double_rows(points, 3, "points");
  std::vector<std::uint8_t> raw(points.size(), 0);
  require_status(smart_native_coverage_mask(flat_points.data(), points.size(),
                                            bounds.data(), raw.data()),
                 "native coverage-mask calculation failed");
  std::vector<bool> out;
  out.reserve(raw.size());
  for (std::uint8_t value : raw) {
    out.push_back(value != 0);
  }
  return out;
}

py::array_t<double> native_recenter_points_for_box_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> voxels,
    py::array_t<double, py::array::c_style | py::array::forcecast> centroids,
    const std::vector<double>& bounds,
    const std::vector<double>& rotation) {
  if (bounds.size() != 6 || rotation.size() != 9) {
    throw std::runtime_error("bounds and rotation must have lengths 6 and 9");
  }
  auto vertex_info = vertices.request();
  auto voxel_info = voxels.request();
  auto centroid_info = centroids.request();
  if (vertex_info.ndim != 2 || vertex_info.shape[1] != 3) {
    throw std::runtime_error("vertices must have shape (N, 3)");
  }
  if (voxel_info.ndim != 2 || voxel_info.shape[1] != 4) {
    throw std::runtime_error("voxels must have shape (M, 4)");
  }
  if (centroid_info.ndim != 2 || centroid_info.shape[1] != 3 ||
      centroid_info.shape[0] != voxel_info.shape[0]) {
    throw std::runtime_error("centroids must have shape (M, 3)");
  }

  const auto* vertex_data = static_cast<const double*>(vertex_info.ptr);
  const auto* voxel_data = static_cast<const std::int64_t*>(voxel_info.ptr);
  const auto* centroid_data = static_cast<const double*>(centroid_info.ptr);
  const std::size_t n_vertices =
      static_cast<std::size_t>(vertex_info.shape[0]);
  const std::size_t n_voxels =
      static_cast<std::size_t>(voxel_info.shape[0]);
  for (std::size_t voxel_idx = 0; voxel_idx < n_voxels; ++voxel_idx) {
    const double* point = centroid_data + voxel_idx * 3;
    const double x = point[0] * rotation[0] + point[1] * rotation[1] +
                     point[2] * rotation[2];
    const double y = point[0] * rotation[3] + point[1] * rotation[4] +
                     point[2] * rotation[5];
    const double z = point[0] * rotation[6] + point[1] * rotation[7] +
                     point[2] * rotation[8];
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      const std::vector<py::ssize_t> empty_shape = {
          static_cast<py::ssize_t>(0), static_cast<py::ssize_t>(3)};
      return py::array_t<double>(empty_shape);
    }
  }
  std::vector<double> selected;
  selected.reserve(n_voxels * 4 * 3);

  for (std::size_t voxel_idx = 0; voxel_idx < n_voxels; ++voxel_idx) {
    const double* point = centroid_data + voxel_idx * 3;
    const double x = point[0] * rotation[0] + point[1] * rotation[1] +
                     point[2] * rotation[2];
    const double y = point[0] * rotation[3] + point[1] * rotation[4] +
                     point[2] * rotation[5];
    const double z = point[0] * rotation[6] + point[1] * rotation[7] +
                     point[2] * rotation[8];
    if (x < bounds[0] || x > bounds[3] || y < bounds[1] ||
        y > bounds[4] || z < bounds[2] || z > bounds[5]) {
      continue;
    }
    const std::int64_t* voxel = voxel_data + voxel_idx * 4;
    for (std::size_t corner = 0; corner < 4; ++corner) {
      if (voxel[corner] < 0 ||
          static_cast<std::size_t>(voxel[corner]) >= n_vertices) {
        throw std::runtime_error("voxel vertex index is out of range");
      }
      const double* vertex =
          vertex_data + static_cast<std::size_t>(voxel[corner]) * 3;
      selected.push_back(vertex[0]);
      selected.push_back(vertex[1]);
      selected.push_back(vertex[2]);
    }
  }

  const std::size_t n_points = selected.size() / 3;
  py::array_t<double> out({n_points, static_cast<std::size_t>(3)});
  auto out_info = out.request();
  auto* out_data = static_cast<double*>(out_info.ptr);
  if (!selected.empty()) {
    std::copy(selected.begin(), selected.end(), out_data);
  }
  return out;
}

std::vector<std::vector<double>> native_apply_axis_action_py(
    const std::vector<std::vector<double>>& bounds, std::size_t action,
    std::size_t num_action_scale, double action_unit) {
  std::vector<double> flat = flatten_double_rows(bounds, 6, "bounds");
  std::vector<double> out(flat.size(), 0.0);
  require_status(smart_native_apply_axis_action(
                     flat.data(), bounds.size(), action, num_action_scale,
                     action_unit, out.data()),
                 "native axis-action apply failed");
  return unflatten_double_rows(out, 6);
}

std::vector<double> native_action_upper_rewards_py(
    const std::vector<std::vector<double>>& bounds, std::size_t num_action_scale,
    double action_unit, double volume_sum, double last_bbox_score) {
  std::vector<double> flat = flatten_double_rows(bounds, 6, "bounds");
  const std::size_t n_actions =
      smart_native_action_count(bounds.size(), num_action_scale);
  std::vector<double> out(n_actions, 0.0);
  require_status(smart_native_action_upper_rewards(
                     flat.data(), bounds.size(), num_action_scale, action_unit,
                     volume_sum, last_bbox_score, out.data()),
                 "native action upper-reward calculation failed");
  return out;
}

std::vector<double> native_bbox_action_upper_rewards_py(
    const std::vector<std::vector<double>>& bounds, std::size_t bbox_idx,
    std::size_t num_action_scale, double action_unit, double volume_sum,
    double last_bbox_score) {
  std::vector<double> flat = flatten_double_rows(bounds, 6, "bounds");
  std::vector<double> out(6 * num_action_scale + 1, 0.0);
  require_status(smart_native_bbox_action_upper_rewards(
                     flat.data(), bounds.size(), bbox_idx, num_action_scale,
                     action_unit, volume_sum, last_bbox_score, out.data()),
                 "native bbox action upper-reward calculation failed");
  return out;
}

std::vector<double> native_bavf_scores_py(const std::vector<double>& part_volumes,
                                          const std::vector<double>& bbox_volumes,
                                          double alpha) {
  if (part_volumes.size() != bbox_volumes.size()) {
    throw std::runtime_error("part_volumes and bbox_volumes must have same length");
  }
  std::vector<double> out(part_volumes.size(), 0.0);
  require_status(smart_native_bavf_scores(part_volumes.data(),
                                          bbox_volumes.data(),
                                          part_volumes.size(), alpha,
                                          out.data()),
                 "native BAVF score calculation failed");
  return out;
}

double native_merge_bavf_reward_py(double prev_bvs, double left_bbox_volume,
                                   double right_bbox_volume,
                                   double merged_bbox_volume,
                                   double shape_volume) {
  double out = 0.0;
  require_status(smart_native_merge_bavf_reward(
                     prev_bvs, left_bbox_volume, right_bbox_volume,
                     merged_bbox_volume, shape_volume, &out),
                 "native BAVF merge-reward calculation failed");
  return out;
}

std::pair<std::vector<std::vector<double>>, std::vector<double>>
native_normalize_vertices_raw_py(
    const std::vector<std::vector<double>>& vertices,
    const std::string& mode, const std::string& center, double target) {
  if (vertices.empty()) {
    throw std::runtime_error("vertices must not be empty");
  }
  int mode_id = -1;
  if (mode == "bbox_diagonal") {
    mode_id = 0;
  } else if (mode == "unit_bbox") {
    mode_id = 1;
  } else if (mode == "unit_sphere") {
    mode_id = 2;
  } else {
    throw std::runtime_error("unsupported normalization mode");
  }
  int center_id = -1;
  if (center == "bbox") {
    center_id = 0;
  } else if (center == "mean") {
    center_id = 1;
  } else {
    throw std::runtime_error("unsupported normalization center");
  }
  std::vector<double> flat = flatten_double_rows(vertices, 3, "vertices");
  std::vector<double> out_vertices(flat.size(), 0.0);
  std::vector<double> out_stats(34, 0.0);
  require_status(smart_native_normalize_vertices(
                     flat.data(), vertices.size(), mode_id, center_id, target,
                     out_vertices.data(), out_stats.data()),
                 "native vertex normalization failed");
  return {unflatten_double_rows(out_vertices, 3), out_stats};
}

py::dict native_vertex_stats_dict(const std::vector<double>& raw,
                                  std::size_t offset) {
  py::dict out;
  out["vertex_count"] = static_cast<std::size_t>(std::llround(raw[offset]));
  out["bbox_min"] = py::cast(std::vector<double>{
      raw[offset + 1], raw[offset + 2], raw[offset + 3]});
  out["bbox_max"] = py::cast(std::vector<double>{
      raw[offset + 4], raw[offset + 5], raw[offset + 6]});
  out["bbox_extent"] = py::cast(std::vector<double>{
      raw[offset + 7], raw[offset + 8], raw[offset + 9]});
  out["bbox_diagonal"] = raw[offset + 10];
  out["bbox_center"] = py::cast(std::vector<double>{
      raw[offset + 11], raw[offset + 12], raw[offset + 13]});
  out["sphere_radius"] = raw[offset + 14];
  return out;
}

py::dict native_normalization_stats_dict(const std::vector<double>& raw) {
  if (raw.size() < 34) {
    throw std::runtime_error("native normalization stats are incomplete");
  }
  py::dict out;
  out["before"] = native_vertex_stats_dict(raw, 0);
  out["center"] = py::cast(std::vector<double>{raw[15], raw[16], raw[17]});
  out["scale"] = raw[18];
  out["after"] = native_vertex_stats_dict(raw, 19);
  return out;
}

py::dict native_normalize_obj_file_py(
    const std::string& input_path, const std::string& output_path,
    const std::string& mode, const std::string& center, double target) {
  int mode_id = -1;
  if (mode == "bbox_diagonal") {
    mode_id = 0;
  } else if (mode == "unit_bbox") {
    mode_id = 1;
  } else if (mode == "unit_sphere") {
    mode_id = 2;
  } else {
    throw std::runtime_error("unsupported normalization mode");
  }
  int center_id = -1;
  if (center == "bbox") {
    center_id = 0;
  } else if (center == "mean") {
    center_id = 1;
  } else {
    throw std::runtime_error("unsupported normalization center");
  }

  std::ifstream input(input_path);
  if (!input) {
    throw std::runtime_error("failed to open OBJ input: " + input_path);
  }
  std::vector<std::string> lines;
  std::vector<double> vertices;
  std::string line;
  while (std::getline(input, line)) {
    lines.push_back(line);
    if (line.rfind("v ", 0) != 0) {
      continue;
    }
    std::istringstream stream(line);
    std::string tag;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    stream >> tag >> x >> y >> z;
    if (!stream || tag != "v") {
      throw std::runtime_error("malformed vertex line in OBJ: " + line);
    }
    vertices.push_back(x);
    vertices.push_back(y);
    vertices.push_back(z);
  }
  if (vertices.empty()) {
    throw std::runtime_error("no vertices found in OBJ: " + input_path);
  }

  std::vector<double> normalized(vertices.size(), 0.0);
  std::vector<double> raw_stats(34, 0.0);
  require_status(smart_native_normalize_vertices(
                     vertices.data(), vertices.size() / 3, mode_id, center_id,
                     target, normalized.data(), raw_stats.data()),
                 "native OBJ normalization failed");

  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("failed to open OBJ output: " + output_path);
  }
  output << std::setprecision(9);
  std::size_t vertex_idx = 0;
  for (const auto& original : lines) {
    if (original.rfind("v ", 0) != 0) {
      output << original << '\n';
      continue;
    }
    std::istringstream stream(original);
    std::string tag;
    double ignored_x = 0.0;
    double ignored_y = 0.0;
    double ignored_z = 0.0;
    stream >> tag >> ignored_x >> ignored_y >> ignored_z;
    std::vector<std::string> extras;
    std::string token;
    while (stream >> token) {
      extras.push_back(token);
    }
    const double* point = normalized.data() + vertex_idx * 3;
    output << "v " << point[0] << ' ' << point[1] << ' ' << point[2];
    for (const auto& extra : extras) {
      output << ' ' << extra;
    }
    output << '\n';
    ++vertex_idx;
  }

  return native_normalization_stats_dict(raw_stats);
}

std::size_t parse_obj_vertex_index(const std::string& token,
                                   std::size_t n_vertices) {
  const std::size_t slash = token.find('/');
  const std::string head = slash == std::string::npos ? token : token.substr(0, slash);
  if (head.empty()) {
    throw std::runtime_error("empty OBJ face vertex index");
  }
  const long long raw = std::stoll(head);
  if (raw == 0) {
    throw std::runtime_error("OBJ face indices are 1-based; got 0");
  }
  long long zero_based = raw > 0
                             ? raw - 1
                             : static_cast<long long>(n_vertices) + raw;
  if (zero_based < 0 ||
      static_cast<std::size_t>(zero_based) >= n_vertices) {
    throw std::runtime_error("OBJ face vertex index out of range");
  }
  return static_cast<std::size_t>(zero_based);
}

py::tuple native_load_obj_mesh_py(const std::string& input_path) {
  std::ifstream input(input_path);
  if (!input) {
    throw std::runtime_error("failed to open OBJ input: " + input_path);
  }
  std::vector<std::vector<double>> vertices;
  std::vector<std::vector<std::size_t>> faces;
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("v ", 0) == 0) {
      std::istringstream stream(line);
      std::string tag;
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      stream >> tag >> x >> y >> z;
      if (!stream || tag != "v") {
        throw std::runtime_error("malformed OBJ vertex line: " + line);
      }
      vertices.push_back({x, y, z});
    } else if (line.rfind("f ", 0) == 0) {
      std::istringstream stream(line);
      std::string tag;
      stream >> tag;
      std::vector<std::size_t> polygon;
      std::string token;
      while (stream >> token) {
        polygon.push_back(parse_obj_vertex_index(token, vertices.size()));
      }
      if (polygon.size() < 3) {
        throw std::runtime_error("OBJ face has fewer than three vertices");
      }
      for (std::size_t idx = 1; idx + 1 < polygon.size(); ++idx) {
        faces.push_back({polygon[0], polygon[idx], polygon[idx + 1]});
      }
    }
  }
  if (vertices.empty()) {
    throw std::runtime_error("no vertices found in OBJ: " + input_path);
  }
  return py::make_tuple(vertices, faces);
}

std::size_t native_save_obj_mesh_py(
    const std::string& output_path,
    const std::vector<std::vector<double>>& vertices,
    const std::vector<std::vector<std::size_t>>& faces) {
  check_row_width(vertices, 3, "vertices");
  check_row_width(faces, 3, "faces");
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("failed to open OBJ output: " + output_path);
  }
  output << std::setprecision(9);
  for (const auto& vertex : vertices) {
    output << "v " << vertex[0] << ' ' << vertex[1] << ' ' << vertex[2] << '\n';
  }
  for (const auto& face : faces) {
    output << "f " << (face[0] + 1) << ' ' << (face[1] + 1) << ' '
           << (face[2] + 1) << '\n';
  }
  return vertices.size();
}

double native_symmetric_chamfer_py(
    const std::vector<std::vector<double>>& left,
    const std::vector<std::vector<double>>& right) {
  std::vector<double> flat_left = flatten_double_rows(left, 3, "left");
  std::vector<double> flat_right = flatten_double_rows(right, 3, "right");
  double out = 0.0;
  require_status(smart_native_symmetric_chamfer(
                     flat_left.data(), left.size(), flat_right.data(),
                     right.size(), &out),
                 "native symmetric Chamfer calculation failed");
  return out;
}

std::vector<double> native_tetra_volumes_py(
    const std::vector<std::vector<double>>& vertices,
    const std::vector<std::vector<std::size_t>>& voxels) {
  std::vector<double> flat_vertices =
      flatten_double_rows(vertices, 3, "vertices");
  std::vector<std::size_t> flat_voxels =
      flatten_size_t_rows(voxels, 4, "voxels");
  std::vector<double> out(voxels.size(), 0.0);
  require_status(smart_native_tetra_volumes(
                     flat_vertices.data(), vertices.size(), flat_voxels.data(),
                     voxels.size(), out.data()),
                 "native tetra volume calculation failed");
  return out;
}

std::vector<double> native_tetra_centroids_py(
    const std::vector<std::vector<double>>& vertices,
    const std::vector<std::vector<std::size_t>>& voxels) {
  std::vector<double> flat_vertices =
      flatten_double_rows(vertices, 3, "vertices");
  std::vector<std::size_t> flat_voxels =
      flatten_size_t_rows(voxels, 4, "voxels");
  std::vector<double> out(voxels.size() * 3, 0.0);
  require_status(smart_native_tetra_centroids(
                     flat_vertices.data(), vertices.size(), flat_voxels.data(),
                     voxels.size(), out.data()),
                 "native tetra centroid calculation failed");
  return out;
}

std::vector<std::vector<std::size_t>> native_tetra_surface_faces_py(
    const std::vector<std::vector<std::size_t>>& voxels) {
  std::vector<std::size_t> flat_voxels =
      flatten_size_t_rows(voxels, 4, "voxels");
  std::vector<std::size_t> flat_faces(voxels.size() * 4 * 3, 0);
  std::size_t n_faces = 0;
  require_status(smart_native_tetra_surface_faces(
                     flat_voxels.data(), voxels.size(), flat_faces.data(),
                     &n_faces),
                 "native tetra surface-face extraction failed");
  flat_faces.resize(n_faces * 3);
  return unflatten_size_t_rows(flat_faces, 3);
}

std::vector<std::vector<std::size_t>> native_tetra_adjacency_py(
    const std::vector<std::vector<std::size_t>>& voxels) {
  std::vector<std::size_t> flat_voxels =
      flatten_size_t_rows(voxels, 4, "voxels");
  std::vector<std::size_t> offsets(voxels.size() + 1, 0);
  std::size_t n_values = 0;
  require_status(smart_native_tetra_adjacency(flat_voxels.data(), voxels.size(),
                                              offsets.data(), nullptr, 0,
                                              &n_values),
                 "native tetra adjacency sizing failed");
  std::vector<std::size_t> values(n_values, 0);
  require_status(smart_native_tetra_adjacency(
                     flat_voxels.data(), voxels.size(), offsets.data(),
                     values.data(), values.size(), &n_values),
                 "native tetra adjacency extraction failed");
  std::vector<std::vector<std::size_t>> out;
  out.reserve(voxels.size());
  for (std::size_t idx = 0; idx < voxels.size(); ++idx) {
    out.emplace_back(values.begin() + static_cast<std::ptrdiff_t>(offsets[idx]),
                     values.begin() + static_cast<std::ptrdiff_t>(offsets[idx + 1]));
  }
  return out;
}

py::tuple native_load_gmsh_py(const std::string& path) {
  std::size_t n_vertices = 0;
  std::size_t n_faces = 0;
  std::size_t n_voxels = 0;
  require_status(smart_native_load_gmsh_counts(path.c_str(), &n_vertices,
                                               &n_faces, &n_voxels),
                 "native Gmsh count failed");
  std::vector<double> vertices(n_vertices * 3, 0.0);
  std::vector<std::size_t> faces(n_faces * 3, 0);
  std::vector<std::size_t> voxels(n_voxels * 4, 0);
  require_status(smart_native_load_gmsh(
                     path.c_str(), vertices.data(), faces.data(),
                     voxels.data(), vertices.size(), faces.size(),
                     voxels.size(), &n_vertices, &n_faces, &n_voxels),
                 "native Gmsh load failed");
  vertices.resize(n_vertices * 3);
  faces.resize(n_faces * 3);
  voxels.resize(n_voxels * 4);
  return py::make_tuple(unflatten_double_rows(vertices, 3),
                        unflatten_size_t_rows(faces, 3),
                        unflatten_size_t_rows(voxels, 4));
}

void native_save_gmsh_py(
    const std::string& path,
    const std::vector<std::vector<double>>& vertices,
    const std::vector<std::vector<std::size_t>>& faces,
    const std::vector<std::vector<std::size_t>>& voxels) {
  std::vector<double> flat_vertices =
      flatten_double_rows(vertices, 3, "vertices");
  std::vector<std::size_t> flat_faces = flatten_size_t_rows(faces, 3, "faces");
  std::vector<std::size_t> flat_voxels =
      flatten_size_t_rows(voxels, 4, "voxels");
  require_status(smart_native_save_gmsh(path.c_str(), flat_vertices.data(),
                                        vertices.size(), flat_faces.data(),
                                        faces.size(), flat_voxels.data(),
                                        voxels.size()),
                 "native Gmsh save failed");
}

std::vector<std::pair<std::size_t, double>>
native_centroid_proxy_axis_rewards_py(
    const std::vector<std::vector<double>>& centroids,
    const std::vector<double>& volumes,
    const std::vector<std::vector<double>>& bounds,
    const std::vector<std::vector<double>>& rotations,
    std::size_t num_action_scale,
    double action_unit,
    double volume_sum,
    double last_bbox_score,
    double cover_penalty,
    double pen_rate) {
  if (centroids.size() != volumes.size()) {
    throw std::runtime_error("centroids and volumes must have same length");
  }
  if (bounds.size() != rotations.size()) {
    throw std::runtime_error("bounds and rotations must have same length");
  }
  std::vector<double> flat_centroids =
      flatten_double_rows(centroids, 3, "centroids");
  std::vector<double> flat_bounds = flatten_double_rows(bounds, 6, "bounds");
  std::vector<double> flat_rotations =
      flatten_double_rows(rotations, 9, "rotations");
  const std::size_t n_actions =
      smart_native_action_count(bounds.size(), num_action_scale);
  std::vector<std::size_t> actions(n_actions, 0);
  std::vector<double> rewards(n_actions, 0.0);
  std::size_t n_rewards = 0;
  require_status(smart_native_centroid_proxy_axis_rewards(
                     flat_centroids.data(), volumes.data(), centroids.size(),
                     flat_bounds.data(), flat_rotations.data(), bounds.size(),
                     num_action_scale, action_unit, volume_sum,
                     last_bbox_score, cover_penalty, pen_rate, actions.data(),
                     rewards.data(), &n_rewards),
                 "native centroid-proxy action reward calculation failed");
  std::vector<std::pair<std::size_t, double>> out;
  out.reserve(n_rewards);
  for (std::size_t idx = 0; idx < n_rewards; ++idx) {
    out.emplace_back(actions[idx], rewards[idx]);
  }
  return out;
}

py::tuple native_partition_summaries_py(
    const std::vector<std::vector<double>>& vertices,
    const std::vector<std::vector<std::size_t>>& voxels,
    const std::vector<double>& volumes,
    const std::vector<std::vector<std::size_t>>& partitions,
    bool unique_points) {
  if (volumes.size() != voxels.size()) {
    throw std::runtime_error("volumes must have same length as voxels");
  }
  std::vector<double> flat_vertices =
      flatten_double_rows(vertices, 3, "vertices");
  std::vector<std::size_t> flat_voxels =
      flatten_size_t_rows(voxels, 4, "voxels");
  std::vector<std::size_t> partition_offsets;
  std::vector<std::size_t> partition_indices;
  partition_offsets.reserve(partitions.size() + 1);
  partition_offsets.push_back(0);
  for (const auto& part : partitions) {
    if (part.empty()) {
      throw std::runtime_error("partition must not be empty");
    }
    partition_indices.insert(partition_indices.end(), part.begin(), part.end());
    partition_offsets.push_back(partition_indices.size());
  }
  std::vector<double> out_volumes(partitions.size(), 0.0);
  std::vector<double> out_bounds(partitions.size() * 6, 0.0);
  std::vector<std::size_t> out_point_offsets(partitions.size() + 1, 0);
  std::vector<double> out_points(partition_indices.size() * 4 * 3, 0.0);
  std::size_t n_points = 0;
  require_status(smart_native_partition_summaries(
                     flat_vertices.data(), vertices.size(), flat_voxels.data(),
                     voxels.size(), volumes.data(), partition_offsets.data(),
                     partition_indices.data(), partitions.size(),
                     partition_indices.size(), unique_points ? 1 : 0,
                     out_volumes.data(), out_bounds.data(),
                     out_point_offsets.data(), out_points.data(), &n_points),
                 "native partition summary calculation failed");
  out_points.resize(n_points);
  std::vector<std::vector<double>> points;
  points.reserve(partitions.size());
  for (std::size_t idx = 0; idx < partitions.size(); ++idx) {
    points.emplace_back(
        out_points.begin() + static_cast<std::ptrdiff_t>(out_point_offsets[idx]),
        out_points.begin() + static_cast<std::ptrdiff_t>(out_point_offsets[idx + 1]));
  }
  return py::make_tuple(out_volumes, unflatten_double_rows(out_bounds, 6),
                        points);
}

using Vec3 = std::array<double, 3>;
using Plane4 = std::array<double, 4>;

struct ConvexInfoCpp {
  std::vector<Plane4> planes;
  Vec3 aabb_min;
  Vec3 aabb_max;
};

constexpr double kPlaneDedupeTol = 1.0e-6;

Vec3 sub3_cpp(const Vec3& left, const Vec3& right) {
  return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

Vec3 scale3_cpp(const Vec3& value, double scale) {
  return {value[0] * scale, value[1] * scale, value[2] * scale};
}

double dot3_cpp(const Vec3& left, const Vec3& right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

Vec3 cross3_cpp(const Vec3& left, const Vec3& right) {
  return {left[1] * right[2] - left[2] * right[1],
          left[2] * right[0] - left[0] * right[2],
          left[0] * right[1] - left[1] * right[0]};
}

double norm3_cpp(const Vec3& value) { return std::sqrt(dot3_cpp(value, value)); }

Vec3 normalize3_cpp(const Vec3& value) {
  const double norm = norm3_cpp(value);
  if (norm <= 1.0e-12) {
    throw std::runtime_error("rotation axis has near-zero length");
  }
  return scale3_cpp(value, 1.0 / norm);
}

double determinant3_cpp(const std::array<Vec3, 3>& matrix) {
  return matrix[0][0] *
             (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
         matrix[0][1] *
             (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
         matrix[0][2] *
             (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

bool solve_3x3_cpp(const std::array<Vec3, 3>& matrix, const Vec3& rhs,
                   Vec3* out) {
  const double det = determinant3_cpp(matrix);
  if (std::abs(det) <= 1.0e-12) {
    return false;
  }
  auto mx = matrix;
  mx[0][0] = rhs[0];
  mx[1][0] = rhs[1];
  mx[2][0] = rhs[2];
  auto my = matrix;
  my[0][1] = rhs[0];
  my[1][1] = rhs[1];
  my[2][1] = rhs[2];
  auto mz = matrix;
  mz[0][2] = rhs[0];
  mz[1][2] = rhs[1];
  mz[2][2] = rhs[2];
  *out = {determinant3_cpp(mx) / det, determinant3_cpp(my) / det,
          determinant3_cpp(mz) / det};
  return true;
}

double tet_volume_cpp(const std::array<Vec3, 4>& points) {
  const Vec3 a = sub3_cpp(points[1], points[0]);
  const Vec3 b = sub3_cpp(points[2], points[0]);
  const Vec3 c = sub3_cpp(points[3], points[0]);
  return std::abs(dot3_cpp(a, cross3_cpp(b, c))) / 6.0;
}

double clean_key_float_cpp(double value) {
  return std::abs(value) < 5.0e-9 ? 0.0 : value;
}

bool points_close_cpp(const Vec3& left, const Vec3& right, double tol) {
  return std::abs(left[0] - right[0]) <= tol &&
         std::abs(left[1] - right[1]) <= tol &&
         std::abs(left[2] - right[2]) <= tol;
}

void dedupe_points_cpp(std::vector<Vec3>* points, double tol = 1.0e-10) {
  std::vector<Vec3> unique;
  unique.reserve(points->size());
  for (const Vec3& point : *points) {
    bool seen = false;
    for (const Vec3& candidate : unique) {
      if (points_close_cpp(
              {clean_key_float_cpp(point[0]), clean_key_float_cpp(point[1]),
               clean_key_float_cpp(point[2])},
              {clean_key_float_cpp(candidate[0]),
               clean_key_float_cpp(candidate[1]),
               clean_key_float_cpp(candidate[2])},
              tol)) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      unique.push_back(point);
    }
  }
  *points = std::move(unique);
}

Vec3 centroid3_cpp(const std::vector<Vec3>& points) {
  Vec3 out = {0.0, 0.0, 0.0};
  for (const Vec3& point : points) {
    out[0] += point[0];
    out[1] += point[1];
    out[2] += point[2];
  }
  const double scale = 1.0 / static_cast<double>(points.size());
  return {out[0] * scale, out[1] * scale, out[2] * scale};
}

bool planes_close_cpp(const Plane4& left, const Plane4& right) {
  return std::abs(left[0] - right[0]) <= kPlaneDedupeTol &&
         std::abs(left[1] - right[1]) <= kPlaneDedupeTol &&
         std::abs(left[2] - right[2]) <= kPlaneDedupeTol &&
         std::abs(left[3] - right[3]) <= kPlaneDedupeTol;
}

void push_unique_plane_cpp(std::vector<Plane4>* planes,
                           const Plane4& candidate) {
  for (const Plane4& plane : *planes) {
    if (planes_close_cpp(plane, candidate)) {
      return;
    }
  }
  planes->push_back(candidate);
}

std::vector<Plane4> convex_hull_planes_cpp(const std::vector<Vec3>& points) {
  std::vector<Plane4> planes;
  const double eps = 1.0e-9;
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (std::size_t j = i + 1; j < points.size(); ++j) {
      for (std::size_t k = j + 1; k < points.size(); ++k) {
        Vec3 normal =
            cross3_cpp(sub3_cpp(points[j], points[i]), sub3_cpp(points[k], points[i]));
        const double norm = norm3_cpp(normal);
        if (norm <= 1.0e-12) {
          continue;
        }
        normal = scale3_cpp(normal, 1.0 / norm);
        double d = -dot3_cpp(normal, points[i]);
        double max_dist = -std::numeric_limits<double>::infinity();
        double min_dist = std::numeric_limits<double>::infinity();
        for (const Vec3& point : points) {
          const double dist = dot3_cpp(normal, point) + d;
          max_dist = std::max(max_dist, dist);
          min_dist = std::min(min_dist, dist);
        }
        if (max_dist <= eps) {
        } else if (min_dist >= -eps) {
          normal = scale3_cpp(normal, -1.0);
          d = -d;
        } else {
          continue;
        }
        push_unique_plane_cpp(&planes, {normal[0], normal[1], normal[2], d});
      }
    }
  }
  return planes;
}

std::vector<Vec3> halfspace_vertices_cpp(const std::vector<Plane4>& planes,
                                         double tol) {
  std::vector<Vec3> points;
  for (std::size_t i = 0; i < planes.size(); ++i) {
    for (std::size_t j = i + 1; j < planes.size(); ++j) {
      for (std::size_t k = j + 1; k < planes.size(); ++k) {
        const std::array<Vec3, 3> matrix = {
            Vec3{planes[i][0], planes[i][1], planes[i][2]},
            Vec3{planes[j][0], planes[j][1], planes[j][2]},
            Vec3{planes[k][0], planes[k][1], planes[k][2]},
        };
        const Vec3 rhs = {-planes[i][3], -planes[j][3], -planes[k][3]};
        Vec3 point;
        if (!solve_3x3_cpp(matrix, rhs, &point)) {
          continue;
        }
        bool inside = true;
        for (const Plane4& plane : planes) {
          if (dot3_cpp({plane[0], plane[1], plane[2]}, point) + plane[3] >
              tol) {
            inside = false;
            break;
          }
        }
        if (inside) {
          points.push_back(point);
        }
      }
    }
  }
  dedupe_points_cpp(&points);
  return points;
}

double convex_volume_from_planes_and_points_cpp(
    const std::vector<Plane4>& planes, const std::vector<Vec3>& points) {
  double total = 0.0;
  for (const Plane4& plane : planes) {
    const Vec3 normal = {plane[0], plane[1], plane[2]};
    std::vector<Vec3> face_points;
    for (const Vec3& point : points) {
      const double dist = dot3_cpp(normal, point) + plane[3];
      if (std::abs(dist) <= 1.0e-9) {
        face_points.push_back(point);
      }
    }
    dedupe_points_cpp(&face_points);
    if (face_points.size() < 3) {
      continue;
    }
    const Vec3 center = centroid3_cpp(face_points);
    Vec3 u = {0.0, 0.0, 0.0};
    for (const Vec3& point : face_points) {
      const Vec3 candidate = sub3_cpp(point, center);
      const double length = norm3_cpp(candidate);
      if (length > 1.0e-12) {
        u = scale3_cpp(candidate, 1.0 / length);
        break;
      }
    }
    if (norm3_cpp(u) <= 1.0e-12) {
      continue;
    }
    const Vec3 v = cross3_cpp(normal, u);
    std::sort(face_points.begin(), face_points.end(),
              [&](const Vec3& left, const Vec3& right) {
                const Vec3 left_vec = sub3_cpp(left, center);
                const Vec3 right_vec = sub3_cpp(right, center);
                const double left_angle =
                    std::atan2(dot3_cpp(left_vec, v), dot3_cpp(left_vec, u));
                const double right_angle =
                    std::atan2(dot3_cpp(right_vec, v), dot3_cpp(right_vec, u));
                return left_angle < right_angle;
              });
    for (std::size_t idx = 1; idx + 1 < face_points.size(); ++idx) {
      const Vec3 p0 = face_points[0];
      Vec3 p1 = face_points[idx];
      Vec3 p2 = face_points[idx + 1];
      const Vec3 tri_normal = cross3_cpp(sub3_cpp(p1, p0), sub3_cpp(p2, p0));
      if (dot3_cpp(tri_normal, normal) < 0.0) {
        std::swap(p1, p2);
      }
      total += dot3_cpp(p0, cross3_cpp(p1, p2)) / 6.0;
    }
  }
  return std::abs(total);
}

double convex_volume_from_planes_cpp(const std::vector<Plane4>& planes) {
  const std::vector<Vec3> points = halfspace_vertices_cpp(planes, 1.0e-9);
  if (points.size() < 4) {
    return 0.0;
  }
  return convex_volume_from_planes_and_points_cpp(planes, points);
}

double convex_hull_volume_cpp(const std::vector<Vec3>& points) {
  if (points.size() < 4) {
    return 0.0;
  }
  return convex_volume_from_planes_and_points_cpp(convex_hull_planes_cpp(points),
                                                 points);
}

ConvexInfoCpp convex_info_from_points_cpp(const std::vector<Vec3>& points) {
  if (points.size() < 4) {
    throw std::runtime_error("convex polyhedron needs at least 4 points");
  }
  std::vector<Plane4> planes = convex_hull_planes_cpp(points);
  if (planes.empty()) {
    throw std::runtime_error("convex polyhedron has no hull planes");
  }
  Vec3 min_point = points[0];
  Vec3 max_point = points[0];
  for (const Vec3& point : points) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      min_point[axis] = std::min(min_point[axis], point[axis]);
      max_point[axis] = std::max(max_point[axis], point[axis]);
    }
  }
  return {std::move(planes), min_point, max_point};
}

double bbox_volume_cpp(const std::vector<double>& row) {
  if (row.size() != 6) {
    throw std::runtime_error("bbox bounds must have six values");
  }
  return std::max(0.0, row[3] - row[0]) *
         std::max(0.0, row[4] - row[1]) *
         std::max(0.0, row[5] - row[2]);
}

Vec3 transform_local_point_cpp(const Vec3& point, const std::array<Vec3, 3>& rot) {
  return {point[0] * rot[0][0] + point[1] * rot[1][0] +
              point[2] * rot[2][0],
          point[0] * rot[0][1] + point[1] * rot[1][1] +
              point[2] * rot[2][1],
          point[0] * rot[0][2] + point[1] * rot[1][2] +
              point[2] * rot[2][2]};
}

std::vector<Vec3> oriented_box_vertices_cpp(const std::vector<double>& bounds,
                                            const std::array<Vec3, 3>& rot) {
  const double xs[2] = {bounds[0], bounds[3]};
  const double ys[2] = {bounds[1], bounds[4]};
  const double zs[2] = {bounds[2], bounds[5]};
  std::vector<Vec3> out;
  out.reserve(8);
  for (double x : xs) {
    for (double y : ys) {
      for (double z : zs) {
        out.push_back(transform_local_point_cpp({x, y, z}, rot));
      }
    }
  }
  return out;
}

ConvexInfoCpp convex_info_from_oriented_box_cpp(
    const std::vector<double>& bounds, const std::vector<double>& rotation) {
  if (bounds.size() != 6) {
    throw std::runtime_error("bounds must have six values");
  }
  if (rotation.size() != 9) {
    throw std::runtime_error("rotation must be a flattened 3x3 row-major matrix");
  }
  const std::array<Vec3, 3> rot = {
      Vec3{rotation[0], rotation[1], rotation[2]},
      Vec3{rotation[3], rotation[4], rotation[5]},
      Vec3{rotation[6], rotation[7], rotation[8]},
  };
  const std::vector<Vec3> vertices = oriented_box_vertices_cpp(bounds, rot);
  Vec3 min_point = vertices[0];
  Vec3 max_point = vertices[0];
  for (const Vec3& point : vertices) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      min_point[axis] = std::min(min_point[axis], point[axis]);
      max_point[axis] = std::max(max_point[axis], point[axis]);
    }
  }
  std::vector<Plane4> planes;
  planes.reserve(6);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const Vec3 normal_max = normalize3_cpp(rot[axis]);
    const Vec3 normal_min = scale3_cpp(normal_max, -1.0);
    push_unique_plane_cpp(
        &planes,
        {normal_max[0], normal_max[1], normal_max[2], -bounds[axis + 3]});
    push_unique_plane_cpp(
        &planes,
        {normal_min[0], normal_min[1], normal_min[2], bounds[axis]});
  }
  return {std::move(planes), min_point, max_point};
}

bool aabb_overlap_cpp(const ConvexInfoCpp& left, const ConvexInfoCpp& right) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (left.aabb_min[axis] > right.aabb_max[axis] + 1.0e-12 ||
        right.aabb_min[axis] > left.aabb_max[axis] + 1.0e-12) {
      return false;
    }
  }
  return true;
}

bool all_aabb_overlap_cpp(const std::vector<const ConvexInfoCpp*>& infos) {
  if (infos.empty()) {
    return false;
  }
  Vec3 mins = infos[0]->aabb_min;
  Vec3 maxs = infos[0]->aabb_max;
  for (std::size_t idx = 1; idx < infos.size(); ++idx) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      mins[axis] = std::max(mins[axis], infos[idx]->aabb_min[axis]);
      maxs[axis] = std::min(maxs[axis], infos[idx]->aabb_max[axis]);
    }
  }
  return mins[0] <= maxs[0] + 1.0e-12 &&
         mins[1] <= maxs[1] + 1.0e-12 &&
         mins[2] <= maxs[2] + 1.0e-12;
}

double intersection_volume_infos_cpp(
    const std::vector<const ConvexInfoCpp*>& infos) {
  if (infos.empty() || !all_aabb_overlap_cpp(infos)) {
    return 0.0;
  }
  std::vector<Plane4> planes;
  for (const ConvexInfoCpp* info : infos) {
    for (const Plane4& plane : info->planes) {
      push_unique_plane_cpp(&planes, plane);
    }
  }
  return convex_volume_from_planes_cpp(planes);
}

std::vector<std::vector<std::size_t>> overlap_components_cpp(
    const std::vector<ConvexInfoCpp>& infos,
    const std::vector<std::size_t>& indices) {
  if (indices.size() <= 1) {
    return {indices};
  }
  std::vector<bool> visited(indices.size(), false);
  std::vector<std::vector<std::size_t>> components;
  for (std::size_t start = 0; start < indices.size(); ++start) {
    if (visited[start]) {
      continue;
    }
    visited[start] = true;
    std::vector<std::size_t> stack = {start};
    std::vector<std::size_t> component;
    while (!stack.empty()) {
      const std::size_t pos = stack.back();
      stack.pop_back();
      component.push_back(indices[pos]);
      for (std::size_t next = 0; next < indices.size(); ++next) {
        if (visited[next]) {
          continue;
        }
        if (aabb_overlap_cpp(infos[indices[pos]], infos[indices[next]])) {
          visited[next] = true;
          stack.push_back(next);
        }
      }
    }
    components.push_back(std::move(component));
  }
  return components;
}

double union_volume_indices_inclusion_cpp(
    const std::vector<ConvexInfoCpp>& box_infos,
    const std::vector<std::size_t>& indices, const ConvexInfoCpp* base_info,
    const std::unordered_map<std::size_t, double>* single_cache) {
  double total = 0.0;
  if (indices.size() >= sizeof(std::size_t) * 8) {
    throw std::runtime_error("too many boxes for inclusion-exclusion");
  }
  const std::size_t subset_count = std::size_t{1} << indices.size();
  for (std::size_t mask = 1; mask < subset_count; ++mask) {
    std::vector<const ConvexInfoCpp*> refs;
    if (base_info != nullptr) {
      refs.push_back(base_info);
    }
    std::size_t bits = 0;
    std::size_t single_idx = 0;
    for (std::size_t pos = 0; pos < indices.size(); ++pos) {
      if ((mask & (std::size_t{1} << pos)) != 0) {
        ++bits;
        single_idx = indices[pos];
        refs.push_back(&box_infos[indices[pos]]);
      }
    }
    double volume = 0.0;
    if (bits == 1 && single_cache != nullptr) {
      const auto found = single_cache->find(single_idx);
      volume = found == single_cache->end()
                   ? intersection_volume_infos_cpp(refs)
                   : found->second;
    } else {
      volume = intersection_volume_infos_cpp(refs);
    }
    total += bits % 2 == 1 ? volume : -volume;
  }
  return std::max(0.0, total);
}

double union_volume_indices_cpp(
    const std::vector<ConvexInfoCpp>& box_infos,
    const std::vector<std::size_t>& indices, const ConvexInfoCpp* base_info,
    const std::unordered_map<std::size_t, double>* single_cache) {
  const auto components = overlap_components_cpp(box_infos, indices);
  if (components.size() > 1) {
    double total = 0.0;
    for (const auto& component : components) {
      total += union_volume_indices_inclusion_cpp(box_infos, component,
                                                  base_info, single_cache);
    }
    return std::max(0.0, total);
  }
  return union_volume_indices_inclusion_cpp(box_infos, indices, base_info,
                                            single_cache);
}

std::vector<Vec3> vertices_to_vec3_cpp(
    const std::vector<std::vector<double>>& vertices) {
  check_row_width(vertices, 3, "vertices");
  std::vector<Vec3> out;
  out.reserve(vertices.size());
  for (const auto& row : vertices) {
    out.push_back({row[0], row[1], row[2]});
  }
  return out;
}

std::vector<std::array<std::size_t, 4>> voxels_to_array4_cpp(
    const std::vector<std::vector<std::size_t>>& voxels,
    std::size_t vertex_count) {
  std::vector<std::array<std::size_t, 4>> out;
  out.reserve(voxels.size());
  for (const auto& voxel : voxels) {
    if (voxel.size() != 4) {
      throw std::runtime_error("voxels must be tetrahedron index rows");
    }
    for (std::size_t index : voxel) {
      if (index >= vertex_count) {
        throw std::runtime_error("voxel index is out of range");
      }
    }
    out.push_back({voxel[0], voxel[1], voxel[2], voxel[3]});
  }
  return out;
}

std::pair<std::vector<ConvexInfoCpp>, std::vector<double>>
tet_infos_and_volumes_cpp(
    const std::vector<Vec3>& vertices,
    const std::vector<std::array<std::size_t, 4>>& voxels) {
  std::vector<ConvexInfoCpp> infos;
  std::vector<double> volumes;
  infos.reserve(voxels.size());
  volumes.reserve(voxels.size());
  for (const auto& voxel : voxels) {
    const std::array<Vec3, 4> points = {vertices[voxel[0]], vertices[voxel[1]],
                                        vertices[voxel[2]], vertices[voxel[3]]};
    std::vector<Vec3> point_vec(points.begin(), points.end());
    infos.push_back(convex_info_from_points_cpp(point_vec));
    volumes.push_back(tet_volume_cpp(points));
  }
  return {std::move(infos), std::move(volumes)};
}

std::unordered_map<std::string, double> tet_clipping_metrics_from_precomputed_cpp(
    const std::vector<ConvexInfoCpp>& tet_infos,
    const std::vector<double>& tet_volumes,
    const std::vector<ConvexInfoCpp>& box_infos,
    const std::vector<double>& box_volumes, double surface_volume) {
  if (tet_infos.size() != tet_volumes.size()) {
    throw std::runtime_error("tet info and volume counts differ");
  }
  const std::vector<std::size_t> box_indices = [&]() {
    std::vector<std::size_t> out(box_infos.size());
    for (std::size_t idx = 0; idx < out.size(); ++idx) {
      out[idx] = idx;
    }
    return out;
  }();
  const double box_union_volume =
      union_volume_indices_cpp(box_infos, box_indices, nullptr, nullptr);
  std::vector<double> per_box_intersections(box_infos.size(), 0.0);
  double shape_box_union_intersection = 0.0;
  double tet_volume_sum = 0.0;

  for (std::size_t tet_idx = 0; tet_idx < tet_infos.size(); ++tet_idx) {
    const ConvexInfoCpp& tet_info = tet_infos[tet_idx];
    const double tet_vol = tet_volumes[tet_idx];
    tet_volume_sum += tet_vol;
    std::vector<std::size_t> overlapping;
    for (std::size_t idx = 0; idx < box_infos.size(); ++idx) {
      if (aabb_overlap_cpp(tet_info, box_infos[idx])) {
        overlapping.push_back(idx);
      }
    }
    if (overlapping.empty()) {
      continue;
    }
    std::unordered_map<std::size_t, double> single_cache;
    for (std::size_t idx : overlapping) {
      const double volume =
          intersection_volume_infos_cpp({&tet_info, &box_infos[idx]});
      single_cache[idx] = volume;
      per_box_intersections[idx] += volume;
    }
    const double tet_union = union_volume_indices_cpp(
        box_infos, overlapping, &tet_info, &single_cache);
    shape_box_union_intersection += std::min(tet_union, tet_vol);
  }

  double mov = 0.0;
  for (std::size_t idx = 0; idx < box_volumes.size(); ++idx) {
    const double intersection = per_box_intersections[idx];
    if (intersection > 1.0e-10) {
      mov = std::max(mov, std::max(0.0, box_volumes[idx] - intersection) /
                              intersection);
    }
  }
  double box_volume_sum = 0.0;
  for (double volume : box_volumes) {
    box_volume_sum += volume;
  }
  const double covered = shape_box_union_intersection / surface_volume;
  const double outside_box_volume =
      std::max(0.0, box_union_volume - shape_box_union_intersection);
  const double tov = covered >= 0.99
                         ? (box_union_volume - surface_volume) / surface_volume
                         : outside_box_volume / surface_volume;
  const double union_volume =
      surface_volume + box_union_volume - shape_box_union_intersection;
  const double viou =
      union_volume <= 0.0 ? 0.0 : shape_box_union_intersection / union_volume;

  return {{"num_box", static_cast<double>(box_infos.size())},
          {"BVS", box_volume_sum / surface_volume},
          {"MOV", mov},
          {"Covered", covered},
          {"TOV", tov},
          {"vIoU", viou},
          {"tet_volume_sum", tet_volume_sum},
          {"surface_volume", surface_volume},
          {"box_union_volume", box_union_volume},
          {"shape_box_union_intersection", shape_box_union_intersection}};
}

double tet_clipping_covered_from_precomputed_cpp(
    const std::vector<ConvexInfoCpp>& tet_infos,
    const std::vector<double>& tet_volumes,
    const std::vector<ConvexInfoCpp>& box_infos, double surface_volume) {
  if (surface_volume <= 0.0) {
    throw std::runtime_error("surface_volume must be positive");
  }
  if (box_infos.empty()) {
    return 0.0;
  }
  double shape_box_union_intersection = 0.0;
  for (std::size_t tet_idx = 0; tet_idx < tet_infos.size(); ++tet_idx) {
    const ConvexInfoCpp& tet_info = tet_infos[tet_idx];
    std::vector<std::size_t> overlapping;
    for (std::size_t idx = 0; idx < box_infos.size(); ++idx) {
      if (aabb_overlap_cpp(tet_info, box_infos[idx])) {
        overlapping.push_back(idx);
      }
    }
    if (overlapping.empty()) {
      continue;
    }
    std::unordered_map<std::size_t, double> single_cache;
    for (std::size_t idx : overlapping) {
      single_cache[idx] =
          intersection_volume_infos_cpp({&tet_info, &box_infos[idx]});
    }
    const double tet_union = union_volume_indices_cpp(
        box_infos, overlapping, &tet_info, &single_cache);
    shape_box_union_intersection += std::min(tet_union, tet_volumes[tet_idx]);
  }
  return shape_box_union_intersection / surface_volume;
}

std::unordered_map<std::string, double> tet_clipping_metrics_py(
    const std::vector<std::vector<double>>& vertices,
    const std::vector<std::vector<std::size_t>>& voxels,
    const std::vector<std::vector<std::vector<double>>>& box_vertices,
    double surface_volume, std::size_t max_boxes, py::object box_volumes_obj) {
  if (surface_volume <= 0.0) {
    throw std::runtime_error("surface_volume must be positive");
  }
  if (box_vertices.empty()) {
    throw std::runtime_error("box_vertices must not be empty");
  }
  if (box_vertices.size() > max_boxes) {
    throw std::runtime_error("box count exceeds max_boxes");
  }
  const std::vector<Vec3> verts3 = vertices_to_vec3_cpp(vertices);
  const auto vox4 = voxels_to_array4_cpp(voxels, verts3.size());
  auto tet_data = tet_infos_and_volumes_cpp(verts3, vox4);
  std::vector<double> explicit_box_volumes;
  const bool has_box_volumes = !box_volumes_obj.is_none();
  if (has_box_volumes) {
    explicit_box_volumes = box_volumes_obj.cast<std::vector<double>>();
    if (explicit_box_volumes.size() != box_vertices.size()) {
      throw std::runtime_error("box_volumes length does not match box count");
    }
  }
  std::vector<ConvexInfoCpp> box_infos;
  std::vector<double> resolved_box_volumes;
  box_infos.reserve(box_vertices.size());
  resolved_box_volumes.reserve(box_vertices.size());
  for (std::size_t idx = 0; idx < box_vertices.size(); ++idx) {
    const std::vector<Vec3> points = vertices_to_vec3_cpp(box_vertices[idx]);
    resolved_box_volumes.push_back(has_box_volumes
                                       ? explicit_box_volumes[idx]
                                       : convex_hull_volume_cpp(points));
    box_infos.push_back(convex_info_from_points_cpp(points));
  }
  return tet_clipping_metrics_from_precomputed_cpp(
      tet_data.first, tet_data.second, box_infos, resolved_box_volumes,
      surface_volume);
}

class TetClippingStateCpp {
 public:
  TetClippingStateCpp(const std::vector<std::vector<double>>& vertices,
                      const std::vector<std::vector<std::size_t>>& voxels,
                      double surface_volume)
      : surface_volume_(surface_volume) {
    if (surface_volume <= 0.0) {
      throw std::runtime_error("surface_volume must be positive");
    }
    const std::vector<Vec3> verts3 = vertices_to_vec3_cpp(vertices);
    const auto vox4 = voxels_to_array4_cpp(voxels, verts3.size());
    auto tet_data = tet_infos_and_volumes_cpp(verts3, vox4);
    tet_infos_ = std::move(tet_data.first);
    tet_volumes_ = std::move(tet_data.second);
  }

  std::unordered_map<std::string, double> metrics(
      const std::vector<std::vector<std::vector<double>>>& box_vertices,
      std::size_t max_boxes, py::object box_volumes_obj) const {
    if (box_vertices.empty()) {
      throw std::runtime_error("box_vertices must not be empty");
    }
    if (box_vertices.size() > max_boxes) {
      throw std::runtime_error("box count exceeds max_boxes");
    }
    std::vector<double> explicit_box_volumes;
    const bool has_box_volumes = !box_volumes_obj.is_none();
    if (has_box_volumes) {
      explicit_box_volumes = box_volumes_obj.cast<std::vector<double>>();
      if (explicit_box_volumes.size() != box_vertices.size()) {
        throw std::runtime_error("box_volumes length does not match box count");
      }
    }
    std::vector<ConvexInfoCpp> box_infos;
    std::vector<double> resolved_box_volumes;
    for (std::size_t idx = 0; idx < box_vertices.size(); ++idx) {
      const std::vector<Vec3> points = vertices_to_vec3_cpp(box_vertices[idx]);
      resolved_box_volumes.push_back(has_box_volumes
                                         ? explicit_box_volumes[idx]
                                         : convex_hull_volume_cpp(points));
      box_infos.push_back(convex_info_from_points_cpp(points));
    }
    return tet_clipping_metrics_from_precomputed_cpp(
        tet_infos_, tet_volumes_, box_infos, resolved_box_volumes,
        surface_volume_);
  }

  std::unordered_map<std::string, double> metrics_for_boxes(
      const std::vector<std::vector<double>>& bounds,
      const std::vector<std::vector<double>>& rotations,
      std::size_t max_boxes) const {
    if (bounds.empty()) {
      throw std::runtime_error("bounds must not be empty");
    }
    if (bounds.size() > max_boxes) {
      throw std::runtime_error("box count exceeds max_boxes");
    }
    if (bounds.size() != rotations.size()) {
      throw std::runtime_error("rotations length must match bounds length");
    }
    std::vector<ConvexInfoCpp> box_infos;
    std::vector<double> box_volumes;
    for (std::size_t idx = 0; idx < bounds.size(); ++idx) {
      box_volumes.push_back(bbox_volume_cpp(bounds[idx]));
      box_infos.push_back(
          convex_info_from_oriented_box_cpp(bounds[idx], rotations[idx]));
    }
    return tet_clipping_metrics_from_precomputed_cpp(
        tet_infos_, tet_volumes_, box_infos, box_volumes, surface_volume_);
  }

  double covered_for_boxes(const std::vector<std::vector<double>>& bounds,
                           const std::vector<std::vector<double>>& rotations,
                           std::size_t max_boxes) const {
    if (bounds.empty()) {
      return 0.0;
    }
    if (bounds.size() > max_boxes) {
      throw std::runtime_error("box count exceeds max_boxes");
    }
    if (bounds.size() != rotations.size()) {
      throw std::runtime_error("rotations length must match bounds length");
    }
    std::vector<ConvexInfoCpp> box_infos;
    for (std::size_t idx = 0; idx < bounds.size(); ++idx) {
      box_infos.push_back(
          convex_info_from_oriented_box_cpp(bounds[idx], rotations[idx]));
    }
    return tet_clipping_covered_from_precomputed_cpp(
        tet_infos_, tet_volumes_, box_infos, surface_volume_);
  }

 private:
  std::vector<ConvexInfoCpp> tet_infos_;
  std::vector<double> tet_volumes_;
  double surface_volume_;
};

bool attr_bool_py(const py::object& obj, const char* name, bool default_value) {
  if (!py::hasattr(obj, name)) {
    return default_value;
  }
  return obj.attr(name).cast<bool>();
}

double attr_double_py(const py::object& obj, const char* name,
                      double default_value) {
  if (!py::hasattr(obj, name)) {
    return default_value;
  }
  return obj.attr(name).cast<double>();
}

std::size_t attr_size_py(const py::object& obj, const char* name,
                         std::size_t default_value) {
  if (!py::hasattr(obj, name)) {
    return default_value;
  }
  return obj.attr(name).cast<std::size_t>();
}

std::string attr_string_py(const py::object& obj, const char* name,
                           const std::string& default_value) {
  if (!py::hasattr(obj, name)) {
    return default_value;
  }
  return py::str(obj.attr(name)).cast<std::string>();
}

class GreedyRefineCppCallbackRunner {
 public:
  GreedyRefineCppCallbackRunner(const py::object& args,
                                const py::object& env)
      : args_(args), env_(env) {
    print_off_ = attr_bool_py(args_, "print_off", false);
    max_step_ = attr_size_py(env_, "max_step", 0);
    env_reset_ = env_.attr("reset");
    env_greedy_sample_ = env_.attr("greedy_sample");
    env_step_ = env_.attr("step");
    if (py::hasattr(env_, "_bridge_axis_refine_segment")) {
      env_bridge_axis_refine_segment_ =
          env_.attr("_bridge_axis_refine_segment");
    } else {
      env_bridge_axis_refine_segment_ = py::none();
    }
    if (py::hasattr(env_, "_bridge_apply_scored_action")) {
      env_bridge_apply_scored_action_ =
          env_.attr("_bridge_apply_scored_action");
    } else {
      env_bridge_apply_scored_action_ = py::none();
    }
    if (py::hasattr(env_, "_bridge_apply_cached_action")) {
      env_bridge_apply_cached_action_ =
          env_.attr("_bridge_apply_cached_action");
    } else {
      env_bridge_apply_cached_action_ = py::none();
    }
  }

  py::tuple run() {
    env_reset_();
    step_count_ = 0;
    done_ = false;
    std::vector<double> rewards;

    while (!done_) {
      const std::size_t remaining = remaining_steps();
      if (remaining > 0 && !env_bridge_axis_refine_segment_.is_none()) {
        py::object segment = env_bridge_axis_refine_segment_(remaining);
        if (!segment.is_none()) {
          py::tuple segment_tuple = segment.cast<py::tuple>();
          std::vector<double> segment_rewards =
              segment_tuple[0].cast<std::vector<double>>();
          std::vector<std::size_t> segment_actions =
              segment_tuple[1].cast<std::vector<std::size_t>>();
          mark_steps(segment_rewards.size(),
                     segment_tuple[2].cast<int>() != 0);
          for (std::size_t idx = 0; idx < segment_rewards.size(); ++idx) {
            if (!print_off_) {
              py::print(segment_actions[idx], segment_rewards[idx]);
            }
            rewards.push_back(segment_rewards[idx]);
          }
          if (done_) {
            break;
          }
          if (!segment_rewards.empty()) {
            continue;
          }
        }
      }

      py::object greedy = env_greedy_sample_(true);
      py::tuple greedy_tuple = greedy.cast<py::tuple>();
      const std::size_t action = greedy_tuple[0].cast<std::size_t>();
      const double candidate_reward = greedy_tuple[1].cast<double>();
      if (candidate_reward <= 0.0) {
        break;
      }

      const double reward = apply_action(action, candidate_reward);
      if (!print_off_) {
        py::print(action, reward);
      }
      rewards.push_back(reward);
    }

    return py::make_tuple(rewards, rewards.size());
  }

 private:
  std::size_t remaining_steps() const {
    if (max_step_ == 0 || max_step_ <= step_count_ + 1) {
      return 0;
    }
    return max_step_ - step_count_ - 1;
  }

  void mark_step(bool done) {
    ++step_count_;
    done_ = done;
  }

  void mark_steps(std::size_t count, bool done) {
    step_count_ += count;
    done_ = done;
  }

  double apply_action(std::size_t action, double candidate_reward) {
    py::object scored = py::none();
    if (!env_bridge_apply_scored_action_.is_none()) {
      scored = env_bridge_apply_scored_action_(action, candidate_reward);
    }
    if (!scored.is_none()) {
      py::tuple scored_tuple = scored.cast<py::tuple>();
      mark_step(scored_tuple[1].cast<int>() != 0);
      return scored_tuple[0].cast<double>();
    }

    py::object cached = py::none();
    if (!env_bridge_apply_cached_action_.is_none()) {
      cached = env_bridge_apply_cached_action_(action);
    }
    if (!cached.is_none()) {
      py::tuple cached_tuple = cached.cast<py::tuple>();
      mark_step(cached_tuple[1].cast<int>() != 0);
      return cached_tuple[0].cast<double>();
    }

    py::object step = env_step_(action);
    py::tuple step_tuple = step.cast<py::tuple>();
    mark_step(step_tuple[2].cast<int>() != 0);
    return step_tuple[0].cast<double>();
  }

  py::object args_;
  py::object env_;
  py::object env_reset_;
  py::object env_greedy_sample_;
  py::object env_step_;
  py::object env_bridge_axis_refine_segment_;
  py::object env_bridge_apply_scored_action_;
  py::object env_bridge_apply_cached_action_;
  std::size_t max_step_ = 0;
  std::size_t step_count_ = 0;
  bool done_ = false;
  bool print_off_ = false;
};

py::tuple run_greedy_refine_callbacks_py(const py::object& args,
                                         const py::object& env) {
  GreedyRefineCppCallbackRunner runner(args, env);
  return runner.run();
}

struct MctsCppNode {
  double q;
  double reward;
  std::size_t num_vis;
  std::string state_key;
  std::vector<std::size_t> child_ids;
  std::vector<std::size_t> child_actions;
  std::vector<bool> action_mask;
  std::vector<std::size_t> untried_actions;

  MctsCppNode() = default;

  MctsCppNode(std::vector<bool> mask, std::string key)
      : q(-std::numeric_limits<double>::max()),
        reward(-std::numeric_limits<double>::max()),
        num_vis(0),
        state_key(std::move(key)),
        action_mask(std::move(mask)) {
    untried_actions = untried_actions_py(action_mask);
  }

  void add_child(std::size_t action, std::size_t child_id) {
    const auto it = std::find(untried_actions.begin(), untried_actions.end(),
                              action);
    if (it != untried_actions.end()) {
      untried_actions.erase(it);
    }
    child_actions.push_back(action);
    child_ids.push_back(child_id);
  }
};

struct MctsCppTranspositionEntry {
  double q = -std::numeric_limits<double>::max();
  double reward = -std::numeric_limits<double>::max();
  std::size_t num_vis = 0;
};

class MctsCppCallbackRunner {
 public:
  MctsCppCallbackRunner(const py::object& args, const py::object& env,
                        std::size_t num_iter,
                        const std::vector<double>& action_prior_logits,
                        const std::vector<double>& action_value_logits)
      : args_(args), env_(env), num_iter_(num_iter) {
    np_random_ = py::module_::import("numpy").attr("random");
    np_random_rand_ = np_random_.attr("rand");
    np_random_randint_ = np_random_.attr("randint");
    env_reset_ = env_.attr("reset");
    env_step_ = env_.attr("step");
    env_render_ = env_.attr("render");
    if (py::hasattr(env_, "current_state_summary")) {
      env_current_state_summary_ = env_.attr("current_state_summary");
    } else {
      env_current_state_summary_ = py::none();
    }
    if (py::hasattr(env_, "_state_cache_key")) {
      env_state_cache_key_ = env_.attr("_state_cache_key");
    } else {
      env_state_cache_key_ = py::none();
    }
    if (py::hasattr(env_, "_bridge_apply_cached_action")) {
      env_bridge_apply_cached_action_ =
          env_.attr("_bridge_apply_cached_action");
    } else {
      env_bridge_apply_cached_action_ = py::none();
    }
    if (py::hasattr(env_, "_bridge_apply_unscored_action")) {
      env_bridge_apply_unscored_action_ =
          env_.attr("_bridge_apply_unscored_action");
    } else {
      env_bridge_apply_unscored_action_ = py::none();
    }
    if (py::hasattr(env_, "_bridge_recenter_bbox_params")) {
      env_bridge_recenter_bbox_params_ =
          env_.attr("_bridge_recenter_bbox_params");
    } else {
      env_bridge_recenter_bbox_params_ = py::none();
    }
    if (py::hasattr(env_, "_bridge_apply_scored_action")) {
      env_bridge_apply_scored_action_ =
          env_.attr("_bridge_apply_scored_action");
    } else {
      env_bridge_apply_scored_action_ = py::none();
    }
    if (py::hasattr(env_, "_bridge_mcts_greedy_rollout_segment")) {
      env_bridge_mcts_greedy_rollout_segment_ =
          env_.attr("_bridge_mcts_greedy_rollout_segment");
    } else {
      env_bridge_mcts_greedy_rollout_segment_ = py::none();
    }
    if (py::hasattr(env_, "_bridge_reset_for_cpp_mcts")) {
      env_bridge_reset_for_cpp_mcts_ =
          env_.attr("_bridge_reset_for_cpp_mcts");
    } else {
      env_bridge_reset_for_cpp_mcts_ = py::none();
    }
    if (py::hasattr(env_, "_ensure_manifold_stateful_state")) {
      env_ensure_manifold_stateful_state_ =
          env_.attr("_ensure_manifold_stateful_state");
    } else {
      env_ensure_manifold_stateful_state_ = py::none();
    }
    if (py::hasattr(env_, "_bridge_sync_axis_deltas")) {
      env_bridge_sync_axis_deltas_ = env_.attr("_bridge_sync_axis_deltas");
    } else {
      env_bridge_sync_axis_deltas_ = py::none();
    }
    if (py::hasattr(env_, "_trace_action")) {
      env_trace_action_ = env_.attr("_trace_action");
    } else {
      env_trace_action_ = py::none();
    }
    if (py::hasattr(env_, "_bridge_mcts_greedy_rollout_step")) {
      env_bridge_mcts_greedy_rollout_step_ =
          env_.attr("_bridge_mcts_greedy_rollout_step");
    } else {
      env_bridge_mcts_greedy_rollout_step_ = py::none();
    }
    if (py::hasattr(env_, "_bridge_greedy_samples_for_mask")) {
      env_bridge_greedy_samples_for_mask_ =
          env_.attr("_bridge_greedy_samples_for_mask");
    } else {
      env_bridge_greedy_samples_for_mask_ = py::none();
    }
    if (py::hasattr(env_, "ith_bbox_greedy_sample")) {
      env_ith_bbox_greedy_sample_ = env_.attr("ith_bbox_greedy_sample");
    } else {
      env_ith_bbox_greedy_sample_ = py::none();
    }
    num_bbox_ = env_.attr("num_bbox").cast<std::size_t>();
    num_action_scale_ = env_.attr("num_action_scale").cast<std::size_t>();
    num_actions_ = smart_native_action_count(num_bbox_, num_action_scale_);
    actions_per_bbox_ = num_bbox_ == 0 ? 0 : num_actions_ / num_bbox_;
    exp_weight_ = attr_double_py(args_, "exp_w", 1.0);
    skip_rate_ = attr_double_py(args_, "skip_rate", 0.7);
    gamma_ = attr_double_py(args_, "gamma", 1.0);
    cover_penalty_ = attr_double_py(args_, "cover_penalty", 100.0);
    max_step_ = attr_size_py(args_, "max_step", 0);
    pns_ = attr_bool_py(args_, "pns", false);
    grdexp_ = attr_bool_py(args_, "grdexp", false);
    mask_prun_ = attr_bool_py(args_, "mask_prun", false);
    skip_summary_metrics_ = attr_bool_py(args_, "skip_summary_metrics", false);
    stateful_unscored_apply_ =
        attr_bool_py(args_, "stateful_unscored_apply", false);
    use_fused_rollout_step_ =
        attr_bool_py(args_, "mcts_fused_rollout_step", false) ||
        attr_bool_py(args_, "mcts_native_axis_rollout_step", false) ||
        attr_bool_py(args_, "mcts_native_axis_rollout_segment", false);
    use_native_axis_rollout_segment_ =
        attr_bool_py(args_, "mcts_native_axis_rollout_segment", false);
    trace_actions_enabled_ =
        !attr_string_py(args_, "trace_actions_path", "").empty();
    use_transposition_table_ =
        attr_bool_py(args_, "transposition_table", false);
    transposition_table_size_ =
        attr_size_py(args_, "transposition_table_size", 8192);
    use_cpp_rng_ = attr_bool_py(args_, "mcts_cpp_rng", false);
    const std::size_t cpp_rng_seed =
        attr_size_py(args_, "mcts_cpp_rng_seed",
                     attr_size_py(args_, "seed", 7777));
    cpp_rng_.seed(static_cast<std::uint64_t>(cpp_rng_seed));
    action_prior_weight_ = attr_double_py(args_, "action_prior_weight", 0.0);
    puct_prior_weight_ = attr_double_py(args_, "puct_prior_weight", 0.0);
    action_value_weight_ = attr_double_py(args_, "action_value_weight", 0.0);
    action_prior_top_k_ = attr_size_py(args_, "action_prior_top_k", 0);
    action_prior_select_ = attr_string_py(args_, "action_prior_select", "legacy");
    if (action_prior_select_ != "legacy" && action_prior_select_ != "best" &&
        action_prior_select_ != "softmax") {
      action_prior_select_ = "legacy";
    }
    action_prior_select_temperature_ = std::max(
        attr_double_py(args_, "action_prior_select_temperature", 1.0),
        1.0e-6);
    escape_policy_ = attr_bool_py(args_, "escape_policy", false);
    escape_after_no_update_ = attr_size_py(args_, "escape_after_no_update", 20);
    escape_action_top_k_ = attr_size_py(args_, "escape_action_top_k", 0);
    escape_probability_ =
        std::clamp(attr_double_py(args_, "escape_probability", 0.5), 0.0, 1.0);
    use_native_action_selection_ = num_actions_ >= 128;
    exp_action_reward_.assign(num_actions_, 0.0);
    exp_action_cnt_.assign(num_actions_, 0);
    action_prior_logits_.assign(num_actions_, 0.0);
    for (std::size_t idx = 0;
         idx < action_prior_logits.size() && idx < action_prior_logits_.size();
         ++idx) {
      action_prior_logits_[idx] = action_prior_logits[idx];
    }
    action_value_logits_.assign(num_actions_, 0.0);
    for (std::size_t idx = 0;
         idx < action_value_logits.size() && idx < action_value_logits_.size();
         ++idx) {
      action_value_logits_[idx] = action_value_logits[idx];
    }
    opposite_actions_ =
        native_opposite_actions_py(num_bbox_, num_action_scale_);
    nodes_.emplace_back(std::vector<bool>(num_actions_, false), std::string());
    prune_node_untried_actions(nodes_.back());
  }

  std::unordered_map<std::string, double> run() {
    reset_env();
    if (use_transposition_table_) {
      nodes_[0].state_key = env_state_key();
    }
    for (std::size_t ith = 0; ith < num_iter_; ++ith) {
      auto selected = select(0);
      auto simulated = simulate(selected.first, selected.second);
      backpropagate(simulated.path, simulated.rewards);
      select_best(simulated.rewards, ith + 1);
      reset_env();
      iterations_run_ = ith + 1;
      if (ith > 100 && best_reward_ < 1.0e-2) {
        break;
      }
      if (not_updated_ > 400) {
        break;
      }
    }
    std::unordered_map<std::string, double> out;
    out["best_reward"] = best_reward_;
    out["iterations_run"] = static_cast<double>(iterations_run_);
    out["node_count"] = static_cast<double>(nodes_.size());
    out["transposition_hits"] = static_cast<double>(transposition_hits_);
    out["transposition_table_size"] =
        static_cast<double>(transposition_table_.size());
    out["native_axis_rollout_segment_enabled"] =
        use_native_axis_rollout_segment_ ? 1.0 : 0.0;
    out["fused_rollout_step_enabled"] = use_fused_rollout_step_ ? 1.0 : 0.0;
    out["cpp_rng_enabled"] = use_cpp_rng_ ? 1.0 : 0.0;
    out["fused_rollout_steps"] = static_cast<double>(fused_rollout_steps_);
    out["direct_stateful_segments"] =
        static_cast<double>(direct_stateful_segments_);
    out["direct_stateful_segment_steps"] =
        static_cast<double>(direct_stateful_segment_steps_);
    out["direct_stateful_axis_applies"] =
        static_cast<double>(direct_stateful_axis_applies_);
    out["direct_stateful_recenter_applies"] =
        static_cast<double>(direct_stateful_recenter_applies_);
    out["cpp_fast_resets"] = static_cast<double>(cpp_fast_resets_);
    out["python_resets"] = static_cast<double>(python_resets_);
    out["prior_pruned_nodes"] = static_cast<double>(prior_pruned_nodes_);
    out["prior_pruned_actions"] = static_cast<double>(prior_pruned_actions_);
    out["prior_kept_actions"] = static_cast<double>(prior_kept_actions_);
    out["puct_prior_selections"] = static_cast<double>(puct_prior_selections_);
    out["escape_pruned_nodes"] = static_cast<double>(escape_pruned_nodes_);
    out["escape_kept_actions"] = static_cast<double>(escape_kept_actions_);
    out["escape_choices"] = static_cast<double>(escape_choices_);
    out["mcts_runner_cpp"] = 1.0;
    add_env_count(out, "_initial_bbox_cache_hits",
                  "initial_bbox_cache_hits");
    add_env_count(out, "_initial_bbox_cache_misses",
                  "initial_bbox_cache_misses");
    return out;
  }

 private:
  struct SimResult {
    std::vector<std::size_t> path;
    std::vector<double> rewards;
  };

  std::pair<std::vector<std::size_t>, std::vector<double>> select(
      std::size_t root_id) {
    std::size_t node_id = root_id;
    std::vector<std::size_t> path;
    std::vector<double> rewards;
    while (true) {
      path.push_back(node_id);
      if (env_done()) {
        return {path, rewards};
      }
      MctsCppNode& node = nodes_[node_id];
      if (node.untried_actions.empty()) {
        const auto selected = ucb_select(node_id);
        rewards.push_back(env_step_reward(selected.second));
        node_id = selected.first;
      } else if (pns_ && rand_f64() < prob_skip_exploration(node_id)) {
        const auto selected = ucb_select(node_id);
        rewards.push_back(env_step_reward(selected.second));
        node_id = selected.first;
      } else {
        const std::size_t action = choose_untried_action(node.untried_actions);
        rewards.push_back(env_step_reward(action));
        std::vector<bool> child_mask = child_action_mask(
            action, mask_prun_ ? &node.action_mask : nullptr);
        const std::size_t child_id = nodes_.size();
        nodes_.emplace_back(std::move(child_mask), env_state_key());
        seed_from_transposition(nodes_.back());
        prune_node_untried_actions(nodes_.back());
        nodes_[node_id].add_child(action, child_id);
        path.push_back(child_id);
        exp_action_ = action;
        return {path, rewards};
      }
    }
  }

  SimResult simulate(std::vector<std::size_t> path,
                     std::vector<double> rewards) {
    std::vector<bool> mask_bbox(num_bbox_, true);
    std::size_t node_id = path.empty() ? 0 : path.back();
    while (!env_done()) {
      if (use_native_axis_rollout_segment_) {
        const std::size_t step_count = env_step_count();
        const std::size_t remaining =
            max_step_ > step_count + 1 ? max_step_ - step_count - 1 : 0;
        py::object segment =
            direct_stateful_rollout_segment(mask_bbox, remaining);
        if (!env_bridge_mcts_greedy_rollout_segment_.is_none()) {
          if (segment.is_none()) {
            segment =
                env_bridge_mcts_greedy_rollout_segment_(mask_bbox, remaining);
          }
        }
        if (!segment.is_none()) {
          py::tuple tup = segment.cast<py::tuple>();
          std::vector<std::size_t> actions =
              tup[0].cast<std::vector<std::size_t>>();
          std::vector<double> segment_rewards =
              tup[1].cast<std::vector<double>>();
          const bool segment_done =
              tup.size() > 3 ? (tup[3].cast<int>() != 0) : false;
          if (actions.empty()) {
            break;
          }
          mark_steps(actions.size(), segment_done);
          for (std::size_t idx = 0;
               idx < actions.size() && idx < segment_rewards.size(); ++idx) {
            const double reward = segment_rewards[idx];
            if (!std::isfinite(reward) || reward <= 0.0) {
              break;
            }
            ++fused_rollout_steps_;
            rewards.push_back(reward);
            if (grdexp_) {
              const std::size_t action = actions[idx];
              nodes_[node_id].untried_actions = {action};
              nodes_[node_id].action_mask =
                  single_untried_action_mask_py(num_actions_, action);
              const std::size_t child_id = nodes_.size();
              nodes_.emplace_back(child_action_mask(action, nullptr),
                                  std::string());
              prune_node_untried_actions(nodes_.back());
              nodes_[node_id].add_child(action, child_id);
              path.push_back(child_id);
              node_id = child_id;
            }
          }
          break;
        }
      }

      if (use_native_action_selection_ && use_fused_rollout_step_) {
        py::object fused = py::none();
        if (!env_bridge_mcts_greedy_rollout_step_.is_none()) {
          fused = env_bridge_mcts_greedy_rollout_step_(mask_bbox);
        }
        if (!fused.is_none()) {
          py::tuple tup = fused.cast<py::tuple>();
          py::object action_value = tup[0].cast<py::object>();
          const double expected_reward = tup[1].cast<double>();
          const double reward = tup[2].cast<double>();
          const bool done = tup[3].cast<int>() != 0;
          mask_bbox = tup[4].cast<std::vector<bool>>();
          if (action_value.is_none() || expected_reward <= 0.0) {
            break;
          }
          const std::size_t action = action_value.cast<std::size_t>();
          if (!std::isfinite(reward) || reward <= 0.0) {
            break;
          }
          mark_step_done(done);
          ++fused_rollout_steps_;
          rewards.push_back(reward);
          if (grdexp_) {
            nodes_[node_id].untried_actions = {action};
            nodes_[node_id].action_mask =
                single_untried_action_mask_py(num_actions_, action);
            const std::size_t child_id = nodes_.size();
            nodes_.emplace_back(child_action_mask(action, nullptr),
                                env_state_key());
            seed_from_transposition(nodes_.back());
            prune_node_untried_actions(nodes_.back());
            nodes_[node_id].add_child(action, child_id);
            path.push_back(child_id);
            node_id = child_id;
          }
          if (done) {
            break;
          }
          continue;
        }
      }

      std::size_t mx_action = 0;
      bool has_action = false;
      double mx_reward = -std::numeric_limits<double>::max();
      py::object batch = py::none();
      std::vector<std::intptr_t> batch_actions;
      std::vector<double> batch_rewards;
      if (use_native_action_selection_) {
        if (!env_bridge_greedy_samples_for_mask_.is_none()) {
          batch = env_bridge_greedy_samples_for_mask_(mask_bbox);
        }
        if (!batch.is_none()) {
          py::tuple tup = batch.cast<py::tuple>();
          batch_actions = tup[0].cast<std::vector<std::intptr_t>>();
          batch_rewards = tup[1].cast<std::vector<double>>();
        }
      }
      for (std::size_t idx = 0; idx < num_bbox_; ++idx) {
        if (!mask_bbox[idx]) {
          continue;
        }
        std::intptr_t action = -1;
        double reward = -std::numeric_limits<double>::max();
        if (!batch.is_none()) {
          if (batch_actions.size() != num_bbox_ ||
              batch_rewards.size() != num_bbox_) {
            throw std::runtime_error(
                "bridge batch greedy result length does not match bbox count");
          }
          action = batch_actions[idx];
          reward = batch_rewards[idx];
        } else {
          if (env_ith_bbox_greedy_sample_.is_none()) {
            throw std::runtime_error("env.ith_bbox_greedy_sample is required");
          }
          py::object result = env_ith_bbox_greedy_sample_(idx);
          py::tuple tup = result.cast<py::tuple>();
          action = tup[0].cast<std::intptr_t>();
          reward = tup[1].cast<double>();
        }
        if (mx_reward < reward) {
          mx_reward = reward;
          if (action >= 0) {
            mx_action = static_cast<std::size_t>(action);
            has_action = true;
          } else {
            has_action = false;
          }
        }
        if (reward < 0.0) {
          mask_bbox[idx] = false;
        }
      }
      if (mx_reward <= 0.0 || !has_action) {
        break;
      }
      const double reward = env_step_scored_reward(mx_action, mx_reward);
      if (!std::isfinite(reward) || reward <= 0.0) {
        break;
      }
      const double reward_to_store =
          std::abs(reward - mx_reward) <= 1.0e-12 + 1.0e-9 * std::abs(mx_reward)
              ? mx_reward
              : reward;
      rewards.push_back(reward_to_store);
      if (grdexp_) {
        nodes_[node_id].untried_actions = {mx_action};
        nodes_[node_id].action_mask =
            single_untried_action_mask_py(num_actions_, mx_action);
        const std::size_t child_id = nodes_.size();
        nodes_.emplace_back(child_action_mask(mx_action, nullptr),
                            env_state_key());
        seed_from_transposition(nodes_.back());
        prune_node_untried_actions(nodes_.back());
        nodes_[node_id].add_child(mx_action, child_id);
        path.push_back(child_id);
        node_id = child_id;
      }
    }
    return {path, rewards};
  }

  void backpropagate(const std::vector<std::size_t>& path,
                     const std::vector<double>& rewards) {
    const double reward_sum = discounted_reward(rewards);
    if (pns_ && exp_action_ < exp_action_reward_.size()) {
      const std::size_t cnt = exp_action_cnt_[exp_action_];
      const double previous = exp_action_reward_[exp_action_];
      exp_action_reward_[exp_action_] =
          previous / static_cast<double>(cnt + 1) * static_cast<double>(cnt) +
          (reward_sum - best_reward_) / static_cast<double>(cnt + 1);
      ++exp_action_cnt_[exp_action_];
    }
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      MctsCppNode& node = nodes_[*it];
      if (node.num_vis == 0) {
        node.reward = reward_sum;
      }
      ++node.num_vis;
      if (reward_sum > node.q) {
        node.q = reward_sum;
      }
      store_transposition(*it);
    }
  }

  void select_best(const std::vector<double>& rewards, std::size_t num_iter) {
    const double reward_sum = discounted_reward(rewards);
    ++not_updated_;
    if (reward_sum > best_reward_) {
      best_reward_ = reward_sum;
      not_updated_ = 0;
      if (!skip_summary_metrics_) {
        if (!env_current_state_summary_.is_none()) {
          env_current_state_summary_();
        }
      }
      env_render_(num_iter);
    }
  }

  std::pair<std::size_t, std::size_t> ucb_select(std::size_t node_id) {
    MctsCppNode& node = nodes_[node_id];
    const std::size_t pos = ucb_select_position(node);
    return {node.child_ids[pos], node.child_actions[pos]};
  }

  std::size_t ucb_select_position(const MctsCppNode& node) {
    if (node.child_ids.empty()) {
      throw std::runtime_error("children must not be empty");
    }
    if (puct_prior_weight_ != 0.0 || action_value_weight_ != 0.0) {
      std::vector<double> scores;
      std::vector<std::size_t> child_visits;
      scores.reserve(node.child_ids.size());
      child_visits.reserve(node.child_ids.size());
      const double log_parent =
          node.num_vis == 0 ? 0.0 : std::log(static_cast<double>(node.num_vis));
      for (std::size_t child_id : node.child_ids) {
        const MctsCppNode& child = nodes_[child_id];
        double score = std::numeric_limits<double>::infinity();
        if (node.num_vis > 0 && child.num_vis > 0) {
          score = child.q +
                  exp_weight_ *
                      std::sqrt(2.0 * log_parent /
                                static_cast<double>(child.num_vis));
        }
        scores.push_back(score);
        child_visits.push_back(child.num_vis);
      }
      if (puct_prior_weight_ != 0.0 && !action_prior_logits_.empty()) {
        if (node.num_vis > 0 && !node.child_actions.empty()) {
          double max_logit = -std::numeric_limits<double>::infinity();
          for (std::size_t action : node.child_actions) {
            const double value = action < action_prior_logits_.size()
                                     ? action_prior_logits_[action]
                                     : 0.0;
            if (value > max_logit) {
              max_logit = value;
            }
          }
          std::vector<double> probs;
          probs.reserve(node.child_actions.size());
          double total = 0.0;
          for (std::size_t action : node.child_actions) {
            const double value = action < action_prior_logits_.size()
                                     ? action_prior_logits_[action]
                                     : 0.0;
            const double prob = std::exp(value - max_logit);
            probs.push_back(prob);
            total += prob;
          }
          if (total > 0.0) {
            const double sqrt_parent =
                std::sqrt(static_cast<double>(node.num_vis));
            for (std::size_t idx = 0; idx < scores.size(); ++idx) {
              scores[idx] += puct_prior_weight_ * (probs[idx] / total) *
                             sqrt_parent /
                             (1.0 + static_cast<double>(child_visits[idx]));
            }
          }
        }
      }
      if (action_value_weight_ != 0.0 && !action_value_logits_.empty()) {
        for (std::size_t idx = 0; idx < node.child_actions.size(); ++idx) {
          const std::size_t action = node.child_actions[idx];
          const double value = action < action_value_logits_.size()
                                   ? action_value_logits_[action]
                                   : 0.0;
          scores[idx] += action_value_weight_ * value;
        }
      }
      const double best = *std::max_element(scores.begin(), scores.end());
      std::vector<std::size_t> best_positions;
      for (std::size_t idx = 0; idx < scores.size(); ++idx) {
        if (scores[idx] == best) {
          best_positions.push_back(idx);
        }
      }
      if (puct_prior_weight_ != 0.0) {
        ++puct_prior_selections_;
      }
      return best_positions[random_randint(best_positions.size())];
    }
    std::vector<std::size_t> best_positions;
    best_positions.reserve(node.child_ids.size());
    double best_score = -std::numeric_limits<double>::infinity();
    const double log_parent =
        node.num_vis == 0 ? 0.0 : std::log(static_cast<double>(node.num_vis));
    for (std::size_t idx = 0; idx < node.child_ids.size(); ++idx) {
      const MctsCppNode& child = nodes_[node.child_ids[idx]];
      double score = std::numeric_limits<double>::infinity();
      if (node.num_vis > 0 && child.num_vis > 0) {
        score = child.q +
                exp_weight_ *
                    std::sqrt(2.0 * log_parent /
                              static_cast<double>(child.num_vis));
      }
      if (score > best_score) {
        best_score = score;
        best_positions.clear();
        best_positions.push_back(idx);
      } else if (score == best_score) {
        best_positions.push_back(idx);
      }
    }
    if (best_positions.empty()) {
      throw std::runtime_error("no UCB best position found");
    }
    return best_positions[random_randint(best_positions.size())];
  }

  double prob_skip_exploration(std::size_t node_id) const {
    const MctsCppNode& node = nodes_[node_id];
    if (node.child_ids.empty()) {
      return 0.0;
    }
    double max_q = 0.0;
    for (std::size_t child_id : node.child_ids) {
      const MctsCppNode& child = nodes_[child_id];
      if (child.reward > node.reward && child.q > max_q) {
        max_q = child.q;
      }
    }
    const double probability = max_q / (best_reward_ + 1.0e-9);
    return std::min(std::max(probability, 0.0), skip_rate_);
  }

  std::vector<double> exp_prob(const std::vector<std::size_t>& actions,
                               double scale) const {
    if (actions.empty()) {
      return {};
    }
    std::vector<double> values;
    values.reserve(actions.size());
    for (std::size_t action : actions) {
      const double prior = action < action_prior_logits_.size()
                               ? action_prior_logits_[action]
                               : 0.0;
      values.push_back(exp_action_reward_[action] * scale +
                       action_prior_weight_ * prior);
    }
    return softmax_scaled(values, 1.0);
  }

  std::vector<double> action_proposal_scores(
      const std::vector<std::size_t>& actions) const {
    std::vector<double> scores;
    scores.reserve(actions.size());
    const double prior_scale =
        action_prior_weight_ != 0.0 ? action_prior_weight_ : puct_prior_weight_;
    for (std::size_t action : actions) {
      const double prior = action < action_prior_logits_.size()
                               ? action_prior_logits_[action]
                               : 0.0;
      const double value = action < action_value_logits_.size()
                               ? action_value_logits_[action]
                               : 0.0;
      scores.push_back(prior_scale * prior + action_value_weight_ * value);
    }
    return scores;
  }

  std::size_t choose_untried_action(const std::vector<std::size_t>& actions) {
    if (actions.empty()) {
      throw std::runtime_error("untried action list must not be empty");
    }
    if (escape_active() && actions.size() > action_prior_top_k_ &&
        rand_f64() < escape_probability_) {
      const std::size_t first_escape =
          std::min(action_prior_top_k_, actions.size());
      const std::size_t offset =
          random_randint(actions.size() - first_escape);
      ++escape_choices_;
      return actions[first_escape + offset];
    }
    if (action_prior_select_active()) {
      const std::vector<double> scores = action_proposal_scores(actions);
      if (action_prior_select_ == "best") {
        return best_score_action(actions, scores);
      }
      if (action_prior_select_ == "softmax") {
        std::vector<double> scaled;
        scaled.reserve(scores.size());
        for (double score : scores) {
          scaled.push_back(score / action_prior_select_temperature_);
        }
        return random_choice(actions, softmax_scaled(scaled, 1.0));
      }
    }
    if (pns_) {
      return random_choice(actions, exp_prob(actions, 100.0));
    }
    return actions[random_randint(actions.size())];
  }

  void prune_node_untried_actions(MctsCppNode& node) {
    if (action_prior_top_k_ == 0 ||
        (action_prior_logits_.empty() && action_value_logits_.empty()) ||
        (action_prior_weight_ == 0.0 && puct_prior_weight_ == 0.0 &&
         action_value_weight_ == 0.0) ||
        node.untried_actions.size() <= action_prior_top_k_) {
      return;
    }
    std::vector<std::size_t> actions = node.untried_actions;
    const std::vector<double> scores = action_proposal_scores(actions);
    std::vector<std::size_t> order(actions.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t left_idx, std::size_t right_idx) {
                const double left_score = scores[left_idx];
                const double right_score = scores[right_idx];
                if (left_score != right_score) {
                  return left_score > right_score;
                }
                return actions[left_idx] < actions[right_idx];
              });
    std::vector<std::size_t> keep;
    keep.reserve(action_prior_top_k_ + escape_action_top_k_);
    for (std::size_t idx = 0;
         idx < order.size() && keep.size() < action_prior_top_k_; ++idx) {
      keep.push_back(actions[order[idx]]);
    }
    if (escape_active()) {
      const std::vector<std::size_t> escape =
          diverse_escape_actions(actions, scores, keep);
      for (std::size_t action : escape) {
        if (std::find(keep.begin(), keep.end(), action) == keep.end()) {
          keep.push_back(action);
        }
      }
      if (!escape.empty()) {
        ++escape_pruned_nodes_;
        escape_kept_actions_ += escape.size();
      }
    }
    const std::size_t pruned =
        node.untried_actions.size() > keep.size()
            ? node.untried_actions.size() - keep.size()
            : 0;
    node.untried_actions = keep;
    node.action_mask.assign(num_actions_, true);
    for (std::size_t action : node.untried_actions) {
      if (action < node.action_mask.size()) {
        node.action_mask[action] = false;
      }
    }
    ++prior_pruned_nodes_;
    prior_pruned_actions_ += pruned;
    prior_kept_actions_ += node.untried_actions.size();
  }

  bool action_prior_select_active() const {
    return action_prior_select_ != "legacy" &&
           (action_prior_weight_ != 0.0 || puct_prior_weight_ != 0.0 ||
            action_value_weight_ != 0.0);
  }

  bool escape_active() const {
    return escape_policy_ && escape_action_top_k_ > 0 &&
           not_updated_ >= escape_after_no_update_ &&
           action_prior_top_k_ > 0 &&
           (action_prior_weight_ != 0.0 || puct_prior_weight_ != 0.0 ||
            action_value_weight_ != 0.0);
  }

  std::array<std::size_t, 3> action_trace_fields(std::size_t action) const {
    const std::size_t per_bbox = 6 * num_action_scale_ + 1;
    const std::size_t bbox_idx = action / per_bbox;
    const std::size_t local = action % per_bbox;
    if (local == per_bbox - 1) {
      return {bbox_idx, 6, 0};
    }
    return {bbox_idx, local / num_action_scale_, local % num_action_scale_};
  }

  std::vector<std::size_t> diverse_escape_actions(
      const std::vector<std::size_t>& actions,
      const std::vector<double>& scores,
      const std::vector<std::size_t>& primary_keep) const {
    if (!escape_active()) {
      return {};
    }
    std::vector<std::size_t> order(actions.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t left_idx, std::size_t right_idx) {
                const double left_score = scores[left_idx];
                const double right_score = scores[right_idx];
                if (left_score != right_score) {
                  return left_score > right_score;
                }
                return actions[left_idx] < actions[right_idx];
              });
    std::vector<std::size_t> out;
    out.reserve(escape_action_top_k_);
    std::vector<std::size_t> used_bboxes;
    std::vector<std::size_t> used_coords;
    for (std::size_t action : primary_keep) {
      const auto fields = action_trace_fields(action);
      used_bboxes.push_back(fields[0]);
      used_coords.push_back(fields[1]);
    }
    auto contains = [](const std::vector<std::size_t>& values,
                       std::size_t value) {
      return std::find(values.begin(), values.end(), value) != values.end();
    };
    for (std::size_t idx : order) {
      const std::size_t action = actions[idx];
      if (contains(primary_keep, action)) {
        continue;
      }
      const auto fields = action_trace_fields(action);
      if (contains(used_bboxes, fields[0]) &&
          contains(used_coords, fields[1]) &&
          out.size() + 1 < escape_action_top_k_) {
        continue;
      }
      out.push_back(action);
      used_bboxes.push_back(fields[0]);
      used_coords.push_back(fields[1]);
      if (out.size() >= escape_action_top_k_) {
        return out;
      }
    }
    for (std::size_t idx : order) {
      const std::size_t action = actions[idx];
      if (contains(primary_keep, action) || contains(out, action)) {
        continue;
      }
      out.push_back(action);
      if (out.size() >= escape_action_top_k_) {
        break;
      }
    }
    return out;
  }

  std::size_t best_score_action(const std::vector<std::size_t>& actions,
                                const std::vector<double>& scores) const {
    if (actions.empty() || scores.size() != actions.size()) {
      throw std::runtime_error("best-score action inputs are invalid");
    }
    const double best = *std::max_element(scores.begin(), scores.end());
    std::vector<std::size_t> best_actions;
    for (std::size_t idx = 0; idx < actions.size(); ++idx) {
      if (scores[idx] == best) {
        best_actions.push_back(actions[idx]);
      }
    }
    return best_actions[random_randint(best_actions.size())];
  }

  std::vector<double> softmax_scaled(const std::vector<double>& values,
                                     double scale) const {
    std::vector<double> out(values.size(), 0.0);
    if (values.empty()) {
      return out;
    }
    double max_value = -std::numeric_limits<double>::infinity();
    for (double value : values) {
      max_value = std::max(max_value, value * scale);
    }
    double total = 0.0;
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
      out[idx] = std::exp(values[idx] * scale - max_value);
      total += out[idx];
    }
    if (total == 0.0) {
      const double uniform = 1.0 / static_cast<double>(values.size());
      std::fill(out.begin(), out.end(), uniform);
      return out;
    }
    for (double& value : out) {
      value /= total;
    }
    return out;
  }

  std::vector<bool> child_action_mask(
      std::size_t action, const std::vector<bool>* parent_mask) const {
    std::vector<bool> out =
        parent_mask == nullptr ? std::vector<bool>(num_actions_, false)
                               : *parent_mask;
    out[opposite_actions_[action]] = true;
    return out;
  }

  double discounted_reward(const std::vector<double>& rewards) const {
    double total = 0.0;
    double scale = 1.0;
    for (double reward : rewards) {
      total += reward * scale;
      scale *= gamma_;
    }
    return total;
  }

  double env_pen_rate() const {
    if (py::hasattr(env_, "pen_rate")) {
      try {
        return env_.attr("pen_rate").cast<double>();
      } catch (...) {
        return 1.0;
      }
    }
    return 1.0;
  }

  py::object direct_stateful_rollout_segment(
      const std::vector<bool>& mask_bbox,
      std::size_t remaining) {
    if (remaining == 0 || env_ensure_manifold_stateful_state_.is_none() ||
        env_bridge_sync_axis_deltas_.is_none()) {
      return py::none();
    }
    py::object state = env_ensure_manifold_stateful_state_();
    if (state.is_none() ||
        !py::hasattr(state, "greedy_axis_rollout_segment_delta")) {
      return py::none();
    }
    py::object result =
        state.attr("greedy_axis_rollout_segment_delta")(
            mask_bbox, cover_penalty_, env_pen_rate(), remaining);
    if (result.is_none()) {
      return py::none();
    }
    py::tuple tup = result.cast<py::tuple>();
    if (tup.size() < 8) {
      return py::none();
    }
    std::vector<std::size_t> actions =
        tup[0].cast<std::vector<std::size_t>>();
    std::vector<double> applied_rewards =
        tup[2].cast<std::vector<double>>();
    std::vector<bool> next_mask = tup[3].cast<std::vector<bool>>();
    std::vector<std::size_t> touched_indices =
        tup[4].cast<std::vector<std::size_t>>();
    std::vector<std::vector<double>> touched_bounds =
        tup[5].cast<std::vector<std::vector<double>>>();
    std::vector<std::vector<double>> touched_rotations =
        tup[6].cast<std::vector<std::vector<double>>>();
    const double next_score = tup[7].cast<double>();
    if (!actions.empty()) {
      env_bridge_sync_axis_deltas_(
          touched_indices, touched_bounds, touched_rotations, next_score,
          actions.size(), py::arg("state_already_current") = true);
      ++direct_stateful_segments_;
      direct_stateful_segment_steps_ += actions.size();
      if (trace_actions_enabled_ && !env_trace_action_.is_none()) {
        for (std::size_t idx = 0;
             idx < actions.size() && idx < applied_rewards.size(); ++idx) {
          env_trace_action_("manifold_stateful_direct_segment", actions[idx],
                            applied_rewards[idx]);
        }
      }
    }
    const bool done =
        max_step_ > 0 && runner_step_cnt_ + actions.size() >= max_step_ - 1;
    return py::make_tuple(actions, applied_rewards, next_mask,
                          done ? 1 : 0);
  }

  py::object direct_stateful_axis_apply(std::size_t action) {
    if (env_ensure_manifold_stateful_state_.is_none() ||
        env_bridge_sync_axis_deltas_.is_none() || actions_per_bbox_ == 0 ||
        action >= num_actions_ ||
        action % actions_per_bbox_ == actions_per_bbox_ - 1) {
      return py::none();
    }
    py::object state = env_ensure_manifold_stateful_state_();
    if (state.is_none() || !py::hasattr(state, "apply_axis_action_delta")) {
      return py::none();
    }
    py::object result = state.attr("apply_axis_action_delta")(
        action, cover_penalty_, env_pen_rate());
    if (result.is_none()) {
      return py::none();
    }
    py::tuple tup = result.cast<py::tuple>();
    if (tup.size() < 5) {
      return py::none();
    }
    const double reward = tup[0].cast<double>();
    const std::size_t bbox_idx = tup[1].cast<std::size_t>();
    std::vector<double> bounds = tup[2].cast<std::vector<double>>();
    std::vector<double> rotation = tup[3].cast<std::vector<double>>();
    const double next_score = tup[4].cast<double>();
    env_bridge_sync_axis_deltas_(
        std::vector<std::size_t>{bbox_idx},
        std::vector<std::vector<double>>{bounds},
        std::vector<std::vector<double>>{rotation},
        next_score,
        1,
        py::arg("state_already_current") = true);
    ++direct_stateful_axis_applies_;
    if (trace_actions_enabled_ && !env_trace_action_.is_none()) {
      env_trace_action_("manifold_stateful_direct_axis_apply", action, reward);
    }
    const bool done =
        max_step_ > 0 && runner_step_cnt_ + 1 >= max_step_ - 1;
    return py::make_tuple(reward, done ? 1 : 0);
  }

  py::object direct_stateful_recenter_apply(std::size_t action) {
    if (env_ensure_manifold_stateful_state_.is_none() ||
        env_bridge_sync_axis_deltas_.is_none() ||
        env_bridge_recenter_bbox_params_.is_none() ||
        actions_per_bbox_ == 0 || action >= num_actions_ ||
        action % actions_per_bbox_ != actions_per_bbox_ - 1) {
      return py::none();
    }
    const std::size_t bbox_idx = action / actions_per_bbox_;
    py::object state = env_ensure_manifold_stateful_state_();
    if (state.is_none() || !py::hasattr(state, "apply_replacement_delta")) {
      return py::none();
    }
    py::object candidate = env_bridge_recenter_bbox_params_(bbox_idx);
    if (candidate.is_none()) {
      return py::none();
    }
    py::tuple candidate_tup = candidate.cast<py::tuple>();
    if (candidate_tup.size() < 2) {
      return py::none();
    }
    std::vector<double> candidate_bounds =
        candidate_tup[0].cast<std::vector<double>>();
    std::vector<double> candidate_rotation =
        candidate_tup[1].cast<std::vector<double>>();
    py::object result = state.attr("apply_replacement_delta")(
        bbox_idx, candidate_bounds, candidate_rotation, cover_penalty_,
        env_pen_rate());
    if (result.is_none()) {
      return py::none();
    }
    py::tuple tup = result.cast<py::tuple>();
    if (tup.size() < 5) {
      return py::none();
    }
    const double reward = tup[0].cast<double>();
    const std::size_t updated_bbox_idx = tup[1].cast<std::size_t>();
    std::vector<double> bounds = tup[2].cast<std::vector<double>>();
    std::vector<double> rotation = tup[3].cast<std::vector<double>>();
    const double next_score = tup[4].cast<double>();
    env_bridge_sync_axis_deltas_(
        std::vector<std::size_t>{updated_bbox_idx},
        std::vector<std::vector<double>>{bounds},
        std::vector<std::vector<double>>{rotation},
        next_score,
        1,
        py::arg("state_already_current") = true);
    ++direct_stateful_recenter_applies_;
    if (trace_actions_enabled_ && !env_trace_action_.is_none()) {
      env_trace_action_("manifold_stateful_direct_recenter_apply", action,
                        reward);
    }
    const bool done =
        max_step_ > 0 && runner_step_cnt_ + 1 >= max_step_ - 1;
    return py::make_tuple(reward, done ? 1 : 0);
  }

  void reset_env() {
    bool reset_done = false;
    if (!env_bridge_reset_for_cpp_mcts_.is_none()) {
      py::object fast = env_bridge_reset_for_cpp_mcts_();
      if (!fast.is_none() && fast.cast<bool>()) {
        reset_done = true;
        ++cpp_fast_resets_;
      }
    }
    if (!reset_done) {
      env_reset_();
      ++python_resets_;
    }
    runner_done_ = false;
    runner_step_cnt_ = 0;
  }

  void mark_step_done(bool done) {
    ++runner_step_cnt_;
    runner_done_ = done;
  }

  void mark_steps(std::size_t count, bool done) {
    runner_step_cnt_ += count;
    runner_done_ = done;
  }

  bool env_done() const { return runner_done_; }

  std::size_t env_step_count() const {
    return runner_step_cnt_;
  }

  double env_step_reward(std::size_t action) {
    py::object cached = py::none();
    if (!env_bridge_apply_cached_action_.is_none()) {
      cached = env_bridge_apply_cached_action_(action);
    }
    if (!cached.is_none()) {
      py::tuple tup = cached.cast<py::tuple>();
      mark_step_done(tup.size() > 1 ? (tup[1].cast<int>() != 0) : false);
      return tup[0].cast<double>();
    }
    if (stateful_unscored_apply_) {
      py::object direct = direct_stateful_axis_apply(action);
      if (!direct.is_none()) {
        py::tuple tup = direct.cast<py::tuple>();
        mark_step_done(tup.size() > 1 ? (tup[1].cast<int>() != 0) : false);
        return tup[0].cast<double>();
      }
      direct = direct_stateful_recenter_apply(action);
      if (!direct.is_none()) {
        py::tuple tup = direct.cast<py::tuple>();
        mark_step_done(tup.size() > 1 ? (tup[1].cast<int>() != 0) : false);
        return tup[0].cast<double>();
      }
      if (!env_bridge_apply_unscored_action_.is_none()) {
        py::object applied = env_bridge_apply_unscored_action_(action);
        if (!applied.is_none()) {
          py::tuple tup = applied.cast<py::tuple>();
          mark_step_done(tup.size() > 1 ? (tup[1].cast<int>() != 0) : false);
          return tup[0].cast<double>();
        }
      }
    }
    py::object result = env_step_(action);
    py::tuple tup = result.cast<py::tuple>();
    mark_step_done(tup.size() > 2 ? (tup[2].cast<int>() != 0) : false);
    return tup[0].cast<double>();
  }

  double env_step_scored_reward(std::size_t action, double expected_reward) {
    py::object cached = py::none();
    if (!env_bridge_apply_cached_action_.is_none()) {
      cached = env_bridge_apply_cached_action_(action);
    }
    if (!cached.is_none()) {
      py::tuple tup = cached.cast<py::tuple>();
      mark_step_done(tup.size() > 1 ? (tup[1].cast<int>() != 0) : false);
      return tup[0].cast<double>();
    }
    if (!env_bridge_apply_scored_action_.is_none()) {
      py::object scored =
          env_bridge_apply_scored_action_(action, expected_reward);
      if (!scored.is_none()) {
        py::tuple tup = scored.cast<py::tuple>();
        mark_step_done(tup.size() > 1 ? (tup[1].cast<int>() != 0) : false);
        return tup[0].cast<double>();
      }
    }
    return env_step_reward(action);
  }

  std::string env_state_key() const {
    if (!use_transposition_table_) {
      return std::string();
    }
    if (env_state_cache_key_.is_none()) {
      return std::string();
    }
    py::object value = env_state_cache_key_();
    if (value.is_none()) {
      return std::string();
    }
    return py::str(value).cast<std::string>();
  }

  void seed_from_transposition(MctsCppNode& node) {
    if (!use_transposition_table_ || node.state_key.empty()) {
      return;
    }
    const auto found = transposition_table_.find(node.state_key);
    if (found == transposition_table_.end()) {
      return;
    }
    node.q = found->second.q;
    node.reward = found->second.reward;
    node.num_vis = found->second.num_vis;
    ++transposition_hits_;
  }

  void store_transposition(std::size_t node_id) {
    if (!use_transposition_table_ || transposition_table_size_ == 0) {
      return;
    }
    const MctsCppNode& node = nodes_[node_id];
    if (node.num_vis == 0 || node.state_key.empty()) {
      return;
    }
    transposition_table_[node.state_key] =
        MctsCppTranspositionEntry{node.q, node.reward, node.num_vis};
    transposition_order_.push_back(node.state_key);
    while (transposition_table_.size() > transposition_table_size_) {
      const std::string old_key = transposition_order_.front();
      transposition_order_.pop_front();
      if (std::find(transposition_order_.begin(), transposition_order_.end(),
                    old_key) == transposition_order_.end()) {
        transposition_table_.erase(old_key);
      }
    }
  }

  double rand_f64() const {
    if (use_cpp_rng_) {
      std::uniform_real_distribution<double> dist(0.0, 1.0);
      return dist(cpp_rng_);
    }
    return np_random_rand_().cast<double>();
  }

  std::size_t random_randint(std::size_t upper) const {
    if (upper == 0) {
      throw std::runtime_error("randint upper bound must be positive");
    }
    if (use_cpp_rng_) {
      std::uniform_int_distribution<std::size_t> dist(0, upper - 1);
      return dist(cpp_rng_);
    }
    py::object value = np_random_randint_(upper, py::arg("size") = 1);
    return value.attr("__getitem__")(0).cast<std::size_t>();
  }

  std::size_t random_choice(const std::vector<std::size_t>& actions,
                            const std::vector<double>& probs) const {
    if (actions.empty()) {
      throw std::runtime_error("choice actions must not be empty");
    }
    if (probs.size() != actions.size()) {
      throw std::runtime_error("choice probability length must match actions");
    }
    double total = 0.0;
    for (double prob : probs) {
      if (std::isfinite(prob) && prob > 0.0) {
        total += prob;
      }
    }
    if (total <= 0.0) {
      return actions[random_randint(actions.size())];
    }
    const double target = rand_f64() * total;
    double cumulative = 0.0;
    for (std::size_t idx = 0; idx < actions.size(); ++idx) {
      const double prob =
          std::isfinite(probs[idx]) && probs[idx] > 0.0 ? probs[idx] : 0.0;
      cumulative += prob;
      if (target <= cumulative) {
        return actions[idx];
      }
    }
    return actions.back();
  }

  void add_env_count(std::unordered_map<std::string, double>& out,
                     const char* attr, const char* key) const {
    if (py::hasattr(env_, attr)) {
      out[key] = env_.attr(attr).cast<double>();
    }
  }

  py::object args_;
  py::object env_;
  py::object np_random_;
  py::object np_random_rand_;
  py::object np_random_randint_;
  py::object env_reset_;
  py::object env_step_;
  py::object env_render_;
  py::object env_current_state_summary_;
  py::object env_state_cache_key_;
  py::object env_bridge_apply_cached_action_;
  py::object env_bridge_apply_unscored_action_;
  py::object env_bridge_recenter_bbox_params_;
  py::object env_bridge_apply_scored_action_;
  py::object env_bridge_mcts_greedy_rollout_segment_;
  py::object env_bridge_reset_for_cpp_mcts_;
  py::object env_ensure_manifold_stateful_state_;
  py::object env_bridge_sync_axis_deltas_;
  py::object env_trace_action_;
  py::object env_bridge_mcts_greedy_rollout_step_;
  py::object env_bridge_greedy_samples_for_mask_;
  py::object env_ith_bbox_greedy_sample_;
  std::size_t num_iter_ = 0;
  std::vector<MctsCppNode> nodes_;
  std::size_t num_bbox_ = 0;
  std::size_t num_action_scale_ = 0;
  std::size_t num_actions_ = 0;
  std::size_t actions_per_bbox_ = 0;
  double exp_weight_ = 1.0;
  double skip_rate_ = 0.7;
  double gamma_ = 1.0;
  double cover_penalty_ = 100.0;
  std::size_t max_step_ = 0;
  bool pns_ = false;
  bool grdexp_ = false;
  bool mask_prun_ = false;
  bool skip_summary_metrics_ = false;
  bool stateful_unscored_apply_ = false;
  bool use_fused_rollout_step_ = false;
  bool use_native_axis_rollout_segment_ = false;
  bool use_transposition_table_ = false;
  bool use_native_action_selection_ = false;
  bool use_cpp_rng_ = false;
  bool trace_actions_enabled_ = false;
  bool runner_done_ = false;
  std::size_t runner_step_cnt_ = 0;
  std::size_t direct_stateful_segments_ = 0;
  std::size_t direct_stateful_segment_steps_ = 0;
  std::size_t direct_stateful_axis_applies_ = 0;
  std::size_t direct_stateful_recenter_applies_ = 0;
  std::size_t cpp_fast_resets_ = 0;
  std::size_t python_resets_ = 0;
  mutable std::mt19937_64 cpp_rng_;
  std::size_t transposition_table_size_ = 8192;
  std::size_t transposition_hits_ = 0;
  std::unordered_map<std::string, MctsCppTranspositionEntry>
      transposition_table_;
  std::deque<std::string> transposition_order_;
  std::vector<double> exp_action_reward_;
  std::vector<std::size_t> exp_action_cnt_;
  double action_prior_weight_ = 0.0;
  double puct_prior_weight_ = 0.0;
  double action_value_weight_ = 0.0;
  std::size_t action_prior_top_k_ = 0;
  std::string action_prior_select_ = "legacy";
  double action_prior_select_temperature_ = 1.0;
  bool escape_policy_ = false;
  std::size_t escape_after_no_update_ = 20;
  std::size_t escape_action_top_k_ = 0;
  double escape_probability_ = 0.5;
  std::vector<double> action_prior_logits_;
  std::vector<double> action_value_logits_;
  std::size_t exp_action_ = 0;
  std::vector<std::size_t> opposite_actions_;
  double best_reward_ = 0.0;
  std::size_t not_updated_ = 0;
  std::size_t iterations_run_ = 0;
  std::size_t fused_rollout_steps_ = 0;
  std::size_t prior_pruned_nodes_ = 0;
  std::size_t prior_pruned_actions_ = 0;
  std::size_t prior_kept_actions_ = 0;
  std::size_t puct_prior_selections_ = 0;
  std::size_t escape_pruned_nodes_ = 0;
  std::size_t escape_kept_actions_ = 0;
  std::size_t escape_choices_ = 0;
};

std::unordered_map<std::string, double> run_mcts_callbacks_py(
    const py::object& args, const py::object& env, std::size_t num_iter,
    const std::vector<double>& action_prior_logits,
    const std::vector<double>& action_value_logits) {
  MctsCppCallbackRunner runner(args, env, num_iter, action_prior_logits,
                               action_value_logits);
  return runner.run();
}

std::string bbox_rot_state_key_py(
    const std::vector<std::vector<double>>& bounds,
    const std::vector<std::vector<double>>& rotations) {
  if (bounds.size() != rotations.size()) {
    throw std::runtime_error("bounds and rotations must have the same length");
  }
  check_row_width(bounds, 6, "bounds");
  check_row_width(rotations, 9, "rotations");
  std::string out;
  for (std::size_t idx = 0; idx < bounds.size(); ++idx) {
    if (idx > 0) {
      out.push_back('|');
    }
    out.push_back('b');
    for (std::size_t value_idx = 0; value_idx < bounds[idx].size(); ++value_idx) {
      if (value_idx > 0) {
        out.push_back(',');
      }
      out += float_bits_key(bounds[idx][value_idx]);
    }
    out.push_back('r');
    for (std::size_t value_idx = 0; value_idx < rotations[idx].size(); ++value_idx) {
      if (value_idx > 0) {
        out.push_back(',');
      }
      out += float_bits_key(rotations[idx][value_idx]);
    }
  }
  return out;
}

class SmartCppManifoldBridgeMesh {
 public:
  SmartCppManifoldBridgeMesh(
      const std::vector<std::vector<double>>& vertices,
      const std::vector<std::vector<std::size_t>>& faces) {
    std::vector<float> flat_vertices = flatten_vertices_float(vertices);
    std::vector<std::uint32_t> flat_faces = flatten_faces_uint32(faces);
    ptr_ = smart_manifold_mesh_new(flat_vertices.data(), vertices.size(),
                                   flat_faces.data(), faces.size());
    if (ptr_ == nullptr) {
      throw std::runtime_error("Manifold bridge mesh construction failed");
    }
  }

  SmartCppManifoldBridgeMesh(const SmartCppManifoldBridgeMesh&) = delete;
  SmartCppManifoldBridgeMesh& operator=(const SmartCppManifoldBridgeMesh&) = delete;

  ~SmartCppManifoldBridgeMesh() {
    if (ptr_ != nullptr) {
      smart_manifold_delete(ptr_);
      ptr_ = nullptr;
    }
  }

  double volume() const {
    return finite_or_throw(smart_manifold_handle_volume(ptr_),
                           "Manifold bridge mesh volume failed");
  }

  double volume_properties() const {
    return finite_or_throw(smart_manifold_handle_volume_properties(ptr_),
                           "Manifold bridge properties volume failed");
  }

  double residual_volume_for_boxes(
      const std::vector<std::vector<std::vector<double>>>& box_vertices) const {
    const std::vector<float> flat = flatten_bridge_box_vertices(box_vertices);
    return finite_or_throw(
        smart_manifold_residual_volume_for_boxes(ptr_, flat.data(),
                                                 box_vertices.size()),
        "Manifold bridge residual volume evaluation failed");
  }

  double residual_volume_for_boxes_properties(
      const std::vector<std::vector<std::vector<double>>>& box_vertices) const {
    const std::vector<float> flat = flatten_bridge_box_vertices(box_vertices);
    return finite_or_throw(
        smart_manifold_residual_volume_for_boxes_properties(
            ptr_, flat.data(), box_vertices.size()),
        "Manifold bridge properties residual volume evaluation failed");
  }

  std::pair<double, double> residual_volume_for_boxes_pair(
      const std::vector<std::vector<std::vector<double>>>& box_vertices) const {
    const std::vector<float> flat = flatten_bridge_box_vertices(box_vertices);
    double mesh_volume = 0.0;
    double properties_volume = 0.0;
    require_status(smart_manifold_residual_volume_for_boxes_pair(
                       ptr_, flat.data(), box_vertices.size(), &mesh_volume,
                       &properties_volume),
                   "Manifold bridge residual volume pair evaluation failed");
    return {mesh_volume, properties_volume};
  }

  double residual_volume_for_box_params(
      const std::vector<std::vector<double>>& bounds,
      const std::vector<std::vector<double>>& rotations) const {
    const std::vector<float> flat =
        flatten_bridge_oriented_box_vertices(bounds, rotations);
    return finite_or_throw(
        smart_manifold_residual_volume_for_boxes(ptr_, flat.data(),
                                                 bounds.size()),
        "Manifold bridge residual volume evaluation failed");
  }

  double residual_volume_for_box_params_properties(
      const std::vector<std::vector<double>>& bounds,
      const std::vector<std::vector<double>>& rotations) const {
    const std::vector<float> flat =
        flatten_bridge_oriented_box_vertices(bounds, rotations);
    return finite_or_throw(
        smart_manifold_residual_volume_for_boxes_properties(
            ptr_, flat.data(), bounds.size()),
        "Manifold bridge properties residual volume evaluation failed");
  }

  std::pair<double, double> residual_volume_for_box_params_pair(
      const std::vector<std::vector<double>>& bounds,
      const std::vector<std::vector<double>>& rotations) const {
    const std::vector<float> flat =
        flatten_bridge_oriented_box_vertices(bounds, rotations);
    double mesh_volume = 0.0;
    double properties_volume = 0.0;
    require_status(smart_manifold_residual_volume_for_boxes_pair(
                       ptr_, flat.data(), bounds.size(), &mesh_volume,
                       &properties_volume),
                   "Manifold bridge residual volume pair evaluation failed");
    return {mesh_volume, properties_volume};
  }

  double covered_for_bounds(
      const std::vector<std::vector<double>>& bounds,
      const std::vector<std::vector<double>>& rotations,
      double volume_sum,
      const std::string& volume_method = "mesh") const {
    if (volume_sum <= 0.0) {
      throw std::runtime_error("volume_sum must be positive");
    }
    const int method = parse_volume_method(volume_method);
    std::vector<std::vector<double>> valid_bounds;
    std::vector<std::vector<double>> valid_rotations;
    for (std::size_t idx = 0; idx < bounds.size(); ++idx) {
      if (bounds[idx].size() != 6 || rotations[idx].size() != 9) {
        throw std::runtime_error("invalid bridge bbox params");
      }
      if (bounds[idx][0] < bounds[idx][3] && bounds[idx][1] < bounds[idx][4] &&
          bounds[idx][2] < bounds[idx][5]) {
        valid_bounds.push_back(bounds[idx]);
        valid_rotations.push_back(rotations[idx]);
      }
    }
    if (valid_bounds.empty()) {
      return 0.0;
    }
    const double residual =
        method == 0
            ? residual_volume_for_box_params(valid_bounds, valid_rotations)
            : residual_volume_for_box_params_properties(valid_bounds,
                                                        valid_rotations);
    return 1.0 - residual / volume_sum;
  }

  py::tuple best_axis_actions_for_mask(
      const std::vector<std::vector<double>>& bounds,
      const std::vector<std::vector<double>>& rotations,
      const std::vector<bool>& bbox_mask,
      std::size_t num_action_scale,
      double action_unit,
      double volume_sum,
      double last_bbox_score,
      double cover_penalty,
      double pen_rate,
      double initial_best,
      const std::string& volume_method = "mesh") const {
    if (bounds.size() != rotations.size() || bounds.size() != bbox_mask.size()) {
      throw std::runtime_error("bounds, rotations, and bbox_mask must have the same length");
    }
    std::vector<double> flat_bounds = flatten_double_rows(bounds, 6, "bounds");
    std::vector<double> flat_rotations =
        flatten_double_rows(rotations, 9, "rotations");
    std::vector<std::uint8_t> flat_mask = bool_mask_to_u8(bbox_mask);
    std::vector<double> action_scales = native_action_scales_py(num_action_scale);
    std::vector<std::intptr_t> actions(bounds.size(), -1);
    std::vector<double> rewards(bounds.size(), initial_best);
    require_status(smart_manifold_best_axis_actions_for_mask(
                       ptr_, flat_bounds.data(), flat_rotations.data(),
                       flat_mask.data(), bounds.size(), num_action_scale,
                       action_unit, volume_sum, last_bbox_score, cover_penalty,
                       pen_rate, initial_best, action_scales.data(),
                       actions.data(), rewards.data(),
                       parse_volume_method(volume_method)),
                   "Manifold bridge batch action scoring failed");
    return py::make_tuple(actions, rewards);
  }

  py::tuple best_axis_action(
      const std::vector<std::vector<double>>& bounds,
      const std::vector<std::vector<double>>& rotations,
      std::intptr_t bbox_idx,
      std::size_t num_action_scale,
      double action_unit,
      double volume_sum,
      double last_bbox_score,
      double cover_penalty,
      double pen_rate,
      double initial_best,
      const std::string& volume_method = "mesh") const {
    std::vector<bool> mask(bounds.size(), bbox_idx < 0);
    if (bbox_idx >= 0) {
      if (static_cast<std::size_t>(bbox_idx) >= bounds.size()) {
        throw std::runtime_error("bbox_idx is out of range");
      }
      mask[static_cast<std::size_t>(bbox_idx)] = true;
    }
    auto result = best_axis_actions_for_mask(
        bounds, rotations, mask, num_action_scale, action_unit, volume_sum,
        last_bbox_score, cover_penalty, pen_rate, initial_best, volume_method);
    const auto actions = result[0].cast<std::vector<std::intptr_t>>();
    const auto rewards = result[1].cast<std::vector<double>>();
    std::intptr_t best_action = -1;
    double best_reward = initial_best;
    for (std::size_t idx = 0; idx < actions.size(); ++idx) {
      if (best_reward < rewards[idx]) {
        best_reward = rewards[idx];
        best_action = actions[idx];
      }
    }
    return py::make_tuple(best_action, best_reward);
  }

  py::tuple greedy_axis_refine_segment(
      std::vector<std::vector<double>> bounds,
      const std::vector<std::vector<double>>& rotations,
      std::size_t num_action_scale,
      double action_unit,
      double volume_sum,
      double last_bbox_score,
      double cover_penalty,
      double pen_rate,
      std::size_t max_steps) const {
    double current_score = last_bbox_score;
    std::vector<double> rewards;
    std::vector<std::size_t> actions;
    for (std::size_t step = 0; step < max_steps; ++step) {
      const double bvs = native_total_bbox_volume_py(bounds) / volume_sum;
      const double bvs_reward = -std::abs(bvs - 1.0) - current_score;
      auto best = best_axis_action(bounds, rotations, -1, num_action_scale,
                                   action_unit, volume_sum, current_score,
                                   cover_penalty, pen_rate, bvs_reward, "mesh");
      const std::intptr_t action = best[0].cast<std::intptr_t>();
      const double reward = best[1].cast<double>();
      if (action < 0 || reward <= 0.0) {
        break;
      }
      bounds = native_apply_axis_action_py(
          bounds, static_cast<std::size_t>(action), num_action_scale,
          action_unit);
      current_score += reward;
      rewards.push_back(reward);
      actions.push_back(static_cast<std::size_t>(action));
    }
    return py::make_tuple(bounds, rotations, rewards, actions, current_score);
  }

 private:
  void* ptr_ = nullptr;
};

class SmartCppCandidateBitsetState {
 public:
  SmartCppCandidateBitsetState(
      const std::vector<std::vector<double>>& centroids,
      const std::vector<double>& volumes,
      double volume_sum)
      : centroids_(centroids), volumes_(volumes), volume_sum_(volume_sum) {
    if (centroids_.size() != volumes_.size()) {
      throw std::runtime_error("centroids and volumes must have the same length");
    }
    if (volume_sum_ <= 0.0) {
      throw std::runtime_error("volume_sum must be positive");
    }
    check_row_width(centroids_, 3, "centroids");
  }

  std::size_t num_centroids() const { return centroids_.size(); }

  double volume_sum() const { return volume_sum_; }

  std::vector<std::pair<std::size_t, double>> axis_rewards(
      const std::vector<std::vector<double>>& bounds,
      const std::vector<std::vector<double>>& rotations,
      std::size_t num_action_scale,
      double action_unit,
      double last_bbox_score,
      double cover_penalty,
      double pen_rate) const {
    return native_centroid_proxy_axis_rewards_py(
        centroids_, volumes_, bounds, rotations, num_action_scale, action_unit,
        volume_sum_, last_bbox_score, cover_penalty, pen_rate);
  }

  std::vector<std::pair<std::size_t, double>> topk_axis_actions(
      const std::vector<std::vector<double>>& bounds,
      const std::vector<std::vector<double>>& rotations,
      std::size_t num_action_scale,
      double action_unit,
      double last_bbox_score,
      double cover_penalty,
      double pen_rate,
      std::intptr_t bbox_idx,
      std::size_t top_k) const {
    if (top_k == 0) {
      return {};
    }
    std::vector<std::pair<std::size_t, double>> records = axis_rewards(
        bounds, rotations, num_action_scale, action_unit, last_bbox_score,
        cover_penalty, pen_rate);
    if (bbox_idx >= 0) {
      if (static_cast<std::size_t>(bbox_idx) >= bounds.size()) {
        throw std::runtime_error("bbox_idx is out of range");
      }
      const std::size_t actions_per_bbox = 6 * num_action_scale + 1;
      const std::size_t start = static_cast<std::size_t>(bbox_idx) * actions_per_bbox;
      const std::size_t end = start + 6 * num_action_scale;
      records.erase(
          std::remove_if(records.begin(), records.end(),
                         [start, end](const auto& item) {
                           return item.first < start || item.first >= end;
                         }),
          records.end());
    }
    records.erase(
        std::remove_if(records.begin(), records.end(),
                       [](const auto& item) { return !std::isfinite(item.second); }),
        records.end());
    std::sort(records.begin(), records.end(), [](const auto& left, const auto& right) {
      if (left.second == right.second) {
        return left.first < right.first;
      }
      return left.second > right.second;
    });
    if (records.size() > top_k) {
      records.resize(top_k);
    }
    return records;
  }

 private:
  std::vector<std::vector<double>> centroids_;
  std::vector<double> volumes_;
  double volume_sum_ = 0.0;
};

class SmartCppManifoldState {
 public:
  struct NativeAxisApply {
    double reward = -std::numeric_limits<double>::infinity();
    std::size_t bbox_idx = 0;
    double last_score = -std::numeric_limits<double>::infinity();
  };

  struct NativeReplacementApply {
    double reward = -std::numeric_limits<double>::infinity();
    std::size_t bbox_idx = 0;
    double last_score = -std::numeric_limits<double>::infinity();
  };

  struct NativeGreedySegment {
    std::vector<std::size_t> actions;
    std::vector<double> best_rewards;
    std::vector<double> applied_rewards;
    std::vector<bool> next_mask;
    double last_score = -std::numeric_limits<double>::infinity();
  };

  struct NativeBestAxisBatch {
    std::vector<std::intptr_t> actions;
    std::vector<double> rewards;
  };

  SmartCppManifoldState(const std::vector<std::vector<double>>& vertices,
                        const std::vector<std::vector<std::size_t>>& faces,
                        const std::vector<std::vector<double>>& bounds,
                        const std::vector<std::vector<double>>& rotations,
                        std::size_t num_action_scale, double action_unit,
                        double volume_sum, double last_bbox_score,
                        bool stateful_union_cache,
                        std::size_t cache_capacity,
                        const std::string& volume_method)
      : num_action_scale_(num_action_scale),
        action_unit_(action_unit),
        volume_sum_(volume_sum),
        action_scales_(native_action_scales_py(num_action_scale)),
        initial_bounds_(bounds),
        initial_rotations_(rotations),
        initial_last_bbox_score_(last_bbox_score) {
    if (volume_sum <= 0.0) {
      throw std::runtime_error("volume_sum must be positive");
    }
    if (num_action_scale == 0 || num_action_scale % 2 != 0) {
      throw std::runtime_error("num_action_scale must be a positive even number");
    }
    if (bounds.size() != rotations.size()) {
      throw std::runtime_error("bounds and rotations must have the same length");
    }
    std::vector<float> flat_vertices = flatten_vertices_float(vertices);
    std::vector<std::uint32_t> flat_faces = flatten_faces_uint32(faces);
    std::vector<double> flat_bounds = flatten_double_rows(bounds, 6, "bounds");
    std::vector<double> flat_rotations =
        flatten_double_rows(rotations, 9, "rotations");
    ptr_ = smart_manifold_state_new(
        flat_vertices.data(), vertices.size(), flat_faces.data(), faces.size(),
        flat_bounds.data(), flat_rotations.data(), bounds.size(), volume_sum,
        last_bbox_score, stateful_union_cache ? 1 : 0,
        std::max<std::size_t>(1, cache_capacity),
        parse_volume_method(volume_method));
    if (ptr_ == nullptr) {
      throw std::runtime_error("Manifold state construction failed");
    }
  }

  SmartCppManifoldState(const SmartCppManifoldState&) = delete;
  SmartCppManifoldState& operator=(const SmartCppManifoldState&) = delete;

  ~SmartCppManifoldState() {
    if (ptr_ != nullptr) {
      smart_manifold_state_delete(ptr_);
      ptr_ = nullptr;
    }
  }

  std::size_t num_boxes() const {
    return smart_manifold_state_num_boxes(ptr_);
  }

  void reset_to_state(const std::vector<std::vector<double>>& bounds,
                      const std::vector<std::vector<double>>& rotations,
                      double last_bbox_score) {
    if (bounds.size() != rotations.size()) {
      throw std::runtime_error("bounds and rotations must have the same length");
    }
    std::vector<double> flat_bounds = flatten_double_rows(bounds, 6, "bounds");
    std::vector<double> flat_rotations =
        flatten_double_rows(rotations, 9, "rotations");
    require_status(smart_manifold_state_reset(ptr_, flat_bounds.data(),
                                              flat_rotations.data(),
                                              bounds.size(), last_bbox_score),
                   "Manifold state reset failed");
  }

  void reset_to_initial() {
    reset_to_state(initial_bounds_, initial_rotations_,
                   initial_last_bbox_score_);
  }

  void commit_current_as_initial() {
    auto copied = copy_bounds_rotations();
    initial_bounds_ = std::move(copied.first);
    initial_rotations_ = std::move(copied.second);
    initial_last_bbox_score_ = last_bbox_score();
  }

  std::pair<std::vector<std::vector<double>>,
            std::vector<std::vector<double>>>
  copy_bounds_rotations() const {
    const std::size_t n_boxes = num_boxes();
    std::vector<double> bounds(n_boxes * 6);
    std::vector<double> rotations(n_boxes * 9);
    require_status(
        smart_manifold_state_copy(ptr_, bounds.data(), rotations.data()),
        "Manifold state copy failed");
    return {unflatten_double_rows(bounds, 6),
            unflatten_double_rows(rotations, 9)};
  }

  std::vector<std::vector<double>> bounds() const {
    return copy_bounds_rotations().first;
  }

  std::vector<std::vector<double>> rotations() const {
    return copy_bounds_rotations().second;
  }

  py::tuple state() const {
    auto copied = copy_bounds_rotations();
    return py::make_tuple(copied.first, copied.second, last_bbox_score());
  }

  std::pair<std::vector<double>, std::vector<double>> bbox_params(
      std::size_t bbox_idx) const {
    if (bbox_idx >= num_boxes()) {
      throw std::runtime_error("bbox_idx is out of range");
    }
    std::vector<double> bounds(6);
    std::vector<double> rotation(9);
    require_status(smart_manifold_state_copy_bbox(
                       ptr_, bbox_idx, bounds.data(), rotation.data()),
                   "Manifold state bbox copy failed");
    return {bounds, rotation};
  }

  double last_bbox_score() const {
    return finite_or_throw(smart_manifold_state_last_bbox_score(ptr_),
                           "Manifold state last score failed");
  }

  double total_volume() const {
    return finite_or_throw(smart_manifold_state_total_bbox_volume(ptr_),
                           "Manifold state total volume failed");
  }

  double bvs() const { return total_volume() / volume_sum_; }

  std::size_t valid_count() const {
    return smart_manifold_state_valid_count(ptr_);
  }

  std::unordered_map<std::string, std::uint64_t> cache_stats() const {
    std::uint64_t values[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    require_status(smart_manifold_state_cache_stats(ptr_, values),
                   "Manifold state cache stats failed");
    return {
        {"reward_cache_size", values[0]},
        {"reward_cache_hits", values[1]},
        {"reward_cache_misses", values[2]},
        {"version", values[3]},
        {"state_hash", values[4]},
        {"except_union_builds", values[5]},
        {"except_union_cache_hits", values[6]},
        {"ordered_prefix_builds", values[7]},
        {"ordered_prefix_cache_hits", values[8]},
    };
  }

  std::uint64_t state_hash() const {
    std::uint64_t values[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    require_status(smart_manifold_state_cache_stats(ptr_, values),
                   "Manifold state cache stats failed");
    return values[4];
  }

  std::string state_key() const {
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0')
           << state_hash();
    return stream.str();
  }

  double covered() const {
    return finite_or_throw(smart_manifold_state_covered(ptr_),
                           "Manifold state coverage failed");
  }

  double score(double cover_penalty, double pen_rate) const {
    return finite_or_throw(
        smart_manifold_state_score(ptr_, cover_penalty, pen_rate),
        "Manifold state score failed");
  }

  double score_axis_action(std::intptr_t action, double cover_penalty,
                           double pen_rate) const {
    return finite_or_throw(
        smart_manifold_state_score_axis_action(
            ptr_, action, num_action_scale_, action_unit_, cover_penalty,
            pen_rate, action_scales_.data()),
        "Manifold state action score failed");
  }

  double score_axis_action_reward(std::intptr_t action, double cover_penalty,
                                  double pen_rate) const {
    return score_axis_action(action, cover_penalty, pen_rate) -
           last_bbox_score();
  }

  double score_replacement(std::size_t bbox_idx,
                           const std::vector<double>& candidate_bounds,
                           const std::vector<double>& candidate_rotation,
                           double cover_penalty, double pen_rate) const {
    if (candidate_bounds.size() != 6 || candidate_rotation.size() != 9) {
      throw std::runtime_error(
          "candidate bounds/rotation must have lengths 6 and 9");
    }
    return finite_or_throw(
        smart_manifold_state_score_replacement(
            ptr_, bbox_idx, candidate_bounds.data(), candidate_rotation.data(),
            cover_penalty, pen_rate),
        "Manifold state replacement score failed");
  }

  py::tuple apply_replacement_delta(
      std::size_t bbox_idx, const std::vector<double>& candidate_bounds,
      const std::vector<double>& candidate_rotation, double cover_penalty,
      double pen_rate) {
    NativeReplacementApply native = apply_replacement_delta_native(
        bbox_idx, candidate_bounds, candidate_rotation, cover_penalty,
        pen_rate);
    auto params = bbox_params(native.bbox_idx);
    return py::make_tuple(native.reward, native.bbox_idx, params.first,
                          params.second, native.last_score);
  }

  NativeReplacementApply apply_replacement_delta_native(
      std::size_t bbox_idx, const std::vector<double>& candidate_bounds,
      const std::vector<double>& candidate_rotation, double cover_penalty,
      double pen_rate) {
    if (candidate_bounds.size() != 6 || candidate_rotation.size() != 9) {
      throw std::runtime_error(
          "candidate bounds/rotation must have lengths 6 and 9");
    }
    const double reward = finite_or_throw(
        smart_manifold_state_apply_replacement(
            ptr_, bbox_idx, candidate_bounds.data(), candidate_rotation.data(),
            cover_penalty, pen_rate),
        "Manifold state replacement apply failed");
    return NativeReplacementApply{reward, bbox_idx, last_bbox_score()};
  }

  py::tuple score_action_batch(const std::vector<bool>& bbox_mask,
                               double cover_penalty, double pen_rate,
                               double initial_best) const {
    NativeBestAxisBatch native = best_axis_actions_native(
        bbox_mask, cover_penalty, pen_rate, initial_best);
    return py::make_tuple(native.actions, native.rewards);
  }

  NativeBestAxisBatch best_axis_actions_native(
      const std::vector<bool>& bbox_mask,
      double cover_penalty,
      double pen_rate,
      double initial_best) const {
    const std::size_t n_boxes = num_boxes();
    if (bbox_mask.size() != n_boxes) {
      throw std::runtime_error("bbox_mask length must match bbox count");
    }
    std::vector<std::uint8_t> flat_mask = bool_mask_to_u8(bbox_mask);
    std::vector<std::intptr_t> actions(n_boxes, -1);
    std::vector<double> rewards(n_boxes, initial_best);
    require_status(smart_manifold_state_best_axis_actions_for_mask(
                       ptr_, flat_mask.data(), num_action_scale_, action_unit_,
                       cover_penalty, pen_rate, initial_best,
                       action_scales_.data(), actions.data(), rewards.data()),
                   "Manifold state batch action scoring failed");
    return NativeBestAxisBatch{actions, rewards};
  }

  py::tuple select_replacement_batch(
      const std::vector<bool>& bbox_mask,
      const std::vector<std::vector<double>>& candidate_bounds,
      const std::vector<std::vector<double>>& candidate_rotations,
      std::vector<std::intptr_t> actions, std::vector<double> rewards,
      double cover_penalty, double pen_rate) const {
    const std::size_t n_boxes = num_boxes();
    if (bbox_mask.size() != n_boxes || candidate_bounds.size() != n_boxes ||
        candidate_rotations.size() != n_boxes || actions.size() != n_boxes ||
        rewards.size() != n_boxes) {
      throw std::runtime_error(
          "replacement batch lengths must match bbox count");
    }
    std::vector<std::uint8_t> flat_mask = bool_mask_to_u8(bbox_mask);
    std::vector<double> flat_bounds =
        flatten_double_rows(candidate_bounds, 6, "candidate_bounds");
    std::vector<double> flat_rotations =
        flatten_double_rows(candidate_rotations, 9, "candidate_rotations");
    require_status(smart_manifold_state_select_replacements_for_mask(
                       ptr_, flat_mask.data(), flat_bounds.data(),
                       flat_rotations.data(), num_action_scale_, cover_penalty,
                       pen_rate, actions.data(), rewards.data()),
                   "Manifold state replacement batch selection failed");
    return py::make_tuple(actions, rewards);
  }

  double apply_axis_action(std::intptr_t action, double cover_penalty,
                           double pen_rate) {
    return finite_or_throw(
        smart_manifold_state_apply_axis_action(
            ptr_, action, num_action_scale_, action_unit_, cover_penalty,
            pen_rate, action_scales_.data()),
        "Manifold state apply failed");
  }

  py::tuple apply_axis_action_delta(std::intptr_t action,
                                    double cover_penalty, double pen_rate) {
    NativeAxisApply native =
        apply_axis_action_delta_native(action, cover_penalty, pen_rate);
    auto params = bbox_params(native.bbox_idx);
    return py::make_tuple(native.reward, native.bbox_idx, params.first,
                          params.second, native.last_score);
  }

  NativeAxisApply apply_axis_action_delta_native(std::intptr_t action,
                                                 double cover_penalty,
                                                 double pen_rate) {
    const std::size_t actions_per_bbox = 6 * num_action_scale_ + 1;
    if (action < 0 ||
        static_cast<std::size_t>(action) >= num_boxes() * actions_per_bbox ||
        static_cast<std::size_t>(action) % actions_per_bbox ==
            actions_per_bbox - 1) {
      throw std::runtime_error("action is out of range for axis delta apply");
    }
    const std::size_t bbox_idx =
        static_cast<std::size_t>(action) / actions_per_bbox;
    const double reward = apply_axis_action(action, cover_penalty, pen_rate);
    return NativeAxisApply{reward, bbox_idx, last_bbox_score()};
  }

  py::tuple greedy_axis_rollout_step(const std::vector<bool>& bbox_mask,
                                     double cover_penalty, double pen_rate) {
    const std::size_t n_boxes = num_boxes();
    if (bbox_mask.size() != n_boxes) {
      throw std::runtime_error("bbox_mask length must match bbox count");
    }
    std::vector<std::uint8_t> flat_mask = bool_mask_to_u8(bbox_mask);
    std::vector<std::uint8_t> next_mask_raw(n_boxes, 0);
    std::intptr_t action = -1;
    double best_reward = -std::numeric_limits<double>::infinity();
    double applied_reward = 0.0;
    double last_score = 0.0;
    require_status(smart_manifold_state_greedy_axis_rollout_step(
                       ptr_, flat_mask.data(), num_action_scale_, action_unit_,
                       cover_penalty, pen_rate, action_scales_.data(),
                       next_mask_raw.data(), &action, &best_reward,
                       &applied_reward, &last_score),
                   "Manifold state native rollout step failed");
    std::vector<bool> next_mask;
    next_mask.reserve(next_mask_raw.size());
    for (std::uint8_t value : next_mask_raw) {
      next_mask.push_back(value != 0);
    }
    if (action < 0 || applied_reward <= 0.0) {
      return py::make_tuple(action, best_reward, applied_reward, next_mask,
                            last_score, -1, std::vector<double>{},
                            std::vector<double>{});
    }
    const std::size_t bbox_idx =
        static_cast<std::size_t>(action) / (6 * num_action_scale_ + 1);
    auto params = bbox_params(bbox_idx);
    return py::make_tuple(action, best_reward, applied_reward, next_mask,
                          last_score, static_cast<std::intptr_t>(bbox_idx),
                          params.first, params.second);
  }

  py::tuple greedy_axis_rollout_segment(const std::vector<bool>& bbox_mask,
                                        double cover_penalty, double pen_rate,
                                        std::size_t max_steps) {
    const std::size_t n_boxes = num_boxes();
    if (bbox_mask.size() != n_boxes) {
      throw std::runtime_error("bbox_mask length must match bbox count");
    }
    std::vector<std::uint8_t> flat_mask = bool_mask_to_u8(bbox_mask);
    std::vector<std::intptr_t> actions(max_steps, -1);
    std::vector<double> best_rewards(max_steps, 0.0);
    std::vector<double> applied_rewards(max_steps, 0.0);
    std::vector<std::uint8_t> next_mask_raw(n_boxes, 0);
    std::size_t steps = 0;
    double last_score = 0.0;
    require_status(smart_manifold_state_greedy_axis_rollout_segment(
                       ptr_, flat_mask.data(), num_action_scale_, action_unit_,
                       cover_penalty, pen_rate, max_steps,
                       action_scales_.data(), actions.data(),
                       best_rewards.data(), applied_rewards.data(),
                       next_mask_raw.data(), &steps, &last_score),
                   "Manifold state native rollout segment failed");
    actions.resize(steps);
    best_rewards.resize(steps);
    applied_rewards.resize(steps);
    std::vector<bool> next_mask;
    next_mask.reserve(next_mask_raw.size());
    for (std::uint8_t value : next_mask_raw) {
      next_mask.push_back(value != 0);
    }
    auto copied = copy_bounds_rotations();
    return py::make_tuple(actions, best_rewards, applied_rewards, next_mask,
                          copied.first, copied.second, last_score);
  }

  py::tuple greedy_axis_rollout_segment_delta(
      const std::vector<bool>& bbox_mask,
      double cover_penalty,
      double pen_rate,
      std::size_t max_steps) {
    NativeGreedySegment native = greedy_axis_rollout_segment_native(
        bbox_mask, cover_penalty, pen_rate, max_steps);
    auto touched = touched_bbox_params(native.actions);
    return py::make_tuple(native.actions, native.best_rewards,
                          native.applied_rewards, native.next_mask,
                          std::get<0>(touched), std::get<1>(touched),
                          std::get<2>(touched), native.last_score);
  }

  NativeGreedySegment greedy_axis_rollout_segment_native(
      const std::vector<bool>& bbox_mask,
      double cover_penalty,
      double pen_rate,
      std::size_t max_steps) {
    const std::size_t n_boxes = num_boxes();
    if (bbox_mask.size() != n_boxes) {
      throw std::runtime_error("bbox_mask length must match bbox count");
    }
    std::vector<std::uint8_t> flat_mask = bool_mask_to_u8(bbox_mask);
    std::vector<std::intptr_t> actions(max_steps, -1);
    std::vector<double> best_rewards(max_steps, 0.0);
    std::vector<double> applied_rewards(max_steps, 0.0);
    std::vector<std::uint8_t> next_mask_raw(n_boxes, 0);
    std::size_t steps = 0;
    double last_score = 0.0;
    require_status(smart_manifold_state_greedy_axis_rollout_segment(
                       ptr_, flat_mask.data(), num_action_scale_, action_unit_,
                       cover_penalty, pen_rate, max_steps,
                       action_scales_.data(), actions.data(),
                       best_rewards.data(), applied_rewards.data(),
                       next_mask_raw.data(), &steps, &last_score),
                   "Manifold state native rollout segment failed");
    actions.resize(steps);
    best_rewards.resize(steps);
    applied_rewards.resize(steps);
    std::vector<std::size_t> typed_actions;
    typed_actions.reserve(actions.size());
    for (std::intptr_t action : actions) {
      if (action < 0) {
        break;
      }
      typed_actions.push_back(static_cast<std::size_t>(action));
    }
    std::vector<bool> next_mask;
    next_mask.reserve(next_mask_raw.size());
    for (std::uint8_t value : next_mask_raw) {
      next_mask.push_back(value != 0);
    }
    return NativeGreedySegment{typed_actions, best_rewards, applied_rewards,
                               next_mask, last_score};
  }

  void rollback() {
    require_status(smart_manifold_state_rollback(ptr_),
                   "Manifold state rollback failed");
  }

  py::tuple greedy_axis_refine_segment(double cover_penalty, double pen_rate,
                                       std::size_t max_steps) {
    std::vector<std::intptr_t> raw_actions(max_steps, -1);
    std::vector<double> rewards(max_steps, 0.0);
    std::size_t steps = 0;
    double last_score = 0.0;
    require_status(smart_manifold_state_greedy_axis_refine_segment(
                       ptr_, num_action_scale_, action_unit_, cover_penalty,
                       pen_rate, max_steps, action_scales_.data(),
                       raw_actions.data(), rewards.data(), &steps,
                       &last_score),
                   "Manifold state native greedy segment failed");
    raw_actions.resize(steps);
    rewards.resize(steps);
    std::vector<std::size_t> actions;
    actions.reserve(raw_actions.size());
    for (std::intptr_t action : raw_actions) {
      if (action < 0) {
        throw std::runtime_error("selected action is out of range");
      }
      actions.push_back(static_cast<std::size_t>(action));
    }
    auto copied = copy_bounds_rotations();
    return py::make_tuple(copied.first, copied.second, rewards, actions,
                          last_score);
  }

  py::tuple greedy_axis_refine_segment_delta(double cover_penalty,
                                             double pen_rate,
                                             std::size_t max_steps) {
    NativeGreedySegment native =
        greedy_axis_refine_segment_native(cover_penalty, pen_rate, max_steps);
    auto touched = touched_bbox_params(native.actions);
    return py::make_tuple(native.actions, native.applied_rewards,
                          std::get<0>(touched), std::get<1>(touched),
                          std::get<2>(touched), native.last_score);
  }

  NativeGreedySegment greedy_axis_refine_segment_native(double cover_penalty,
                                                        double pen_rate,
                                                        std::size_t max_steps) {
    std::vector<std::intptr_t> raw_actions(max_steps, -1);
    std::vector<double> rewards(max_steps, 0.0);
    std::size_t steps = 0;
    double last_score = 0.0;
    require_status(smart_manifold_state_greedy_axis_refine_segment(
                       ptr_, num_action_scale_, action_unit_, cover_penalty,
                       pen_rate, max_steps, action_scales_.data(),
                       raw_actions.data(), rewards.data(), &steps,
                       &last_score),
                   "Manifold state native greedy segment failed");
    raw_actions.resize(steps);
    rewards.resize(steps);
    std::vector<std::size_t> actions;
    actions.reserve(raw_actions.size());
    for (std::intptr_t action : raw_actions) {
      if (action < 0) {
        throw std::runtime_error("selected action is out of range");
      }
      actions.push_back(static_cast<std::size_t>(action));
    }
    return NativeGreedySegment{actions, rewards, rewards, {}, last_score};
  }

 private:
  std::tuple<std::vector<std::size_t>, std::vector<std::vector<double>>,
             std::vector<std::vector<double>>>
  touched_bbox_params(const std::vector<std::intptr_t>& actions) const {
    const std::size_t actions_per_bbox = 6 * num_action_scale_ + 1;
    std::vector<std::size_t> touched_indices;
    for (std::intptr_t action : actions) {
      if (action < 0) {
        continue;
      }
      const std::size_t bbox_idx =
          static_cast<std::size_t>(action) / actions_per_bbox;
      if (bbox_idx >= num_boxes()) {
        throw std::runtime_error("selected action is out of range");
      }
      if (std::find(touched_indices.begin(), touched_indices.end(),
                    bbox_idx) == touched_indices.end()) {
        touched_indices.push_back(bbox_idx);
      }
    }
    std::vector<std::vector<double>> touched_bounds;
    std::vector<std::vector<double>> touched_rotations;
    touched_bounds.reserve(touched_indices.size());
    touched_rotations.reserve(touched_indices.size());
    for (std::size_t bbox_idx : touched_indices) {
      auto params = bbox_params(bbox_idx);
      touched_bounds.push_back(std::move(params.first));
      touched_rotations.push_back(std::move(params.second));
    }
    return {touched_indices, touched_bounds, touched_rotations};
  }

  std::tuple<std::vector<std::size_t>, std::vector<std::vector<double>>,
             std::vector<std::vector<double>>>
  touched_bbox_params(const std::vector<std::size_t>& actions) const {
    std::vector<std::intptr_t> typed;
    typed.reserve(actions.size());
    for (std::size_t action : actions) {
      typed.push_back(static_cast<std::intptr_t>(action));
    }
    return touched_bbox_params(typed);
  }

  void* ptr_ = nullptr;
  std::size_t num_action_scale_ = 0;
  double action_unit_ = 0.0;
  double volume_sum_ = 0.0;
  std::vector<double> action_scales_;
  std::vector<std::vector<double>> initial_bounds_;
  std::vector<std::vector<double>> initial_rotations_;
  double initial_last_bbox_score_ = 0.0;
};

class NativeSmartEngine {
 public:
  struct NativeBoxFit {
    bool valid = false;
    std::vector<double> bounds;
    std::vector<double> rotation;
    double volume = 0.0;
  };

  NativeSmartEngine(const std::vector<std::vector<double>>& vertices,
                    const std::vector<std::vector<std::size_t>>& faces,
                    const std::vector<std::vector<std::size_t>>& voxels,
                    const std::vector<double>& tet_volumes,
                    const std::vector<std::vector<double>>& centroids,
                    const std::vector<std::vector<double>>& bounds,
                    const std::vector<std::vector<double>>& rotations,
                    const std::string& category,
                    std::size_t num_action_scale,
                    double action_unit,
                    double volume_sum,
                    double last_bbox_score,
                    bool stateful_union_cache,
                    std::size_t cache_capacity,
                    const std::string& volume_method)
      : vertices_(vertices),
        faces_(faces),
        voxels_(voxels),
        tet_volumes_(tet_volumes),
        centroids_(centroids),
        category_(category),
        num_action_scale_(num_action_scale),
        action_unit_(action_unit),
        volume_sum_(resolve_volume_sum(volume_sum, tet_volumes)),
        stateful_union_cache_(stateful_union_cache),
        cache_capacity_(std::max<std::size_t>(1, cache_capacity)),
        volume_method_(volume_method) {
    check_row_width(vertices_, 3, "vertices");
    check_row_width(faces_, 3, "faces");
    check_row_width(bounds, 6, "bounds");
    check_row_width(rotations, 9, "rotations");
    if (!voxels_.empty()) {
      check_row_width(voxels_, 4, "voxels");
    }
    if (!centroids_.empty()) {
      check_row_width(centroids_, 3, "centroids");
    }
    if (volume_sum_ <= 0.0) {
      throw std::runtime_error("NativeSmartEngine volume_sum must be positive");
    }
    state_ = std::make_unique<SmartCppManifoldState>(
        vertices_, faces_, bounds, rotations, num_action_scale_, action_unit_,
        volume_sum_, last_bbox_score, stateful_union_cache_, cache_capacity_,
        volume_method_);
    actions_per_bbox_ = 6 * num_action_scale_ + 1;
    num_actions_ = bounds.size() * actions_per_bbox_;
    opposite_actions_ = native_opposite_actions_py(bounds.size(), num_action_scale_);
    native_recenter_enabled_ =
        !vertices_.empty() && !voxels_.empty() && !centroids_.empty() &&
        voxels_.size() == centroids_.size();
    snapshot_best_from_state();
    stats_["engine_constructed"] = 1.0;
    stats_["cpp_native"] = 1.0;
    stats_["native_recenter_enabled"] = native_recenter_enabled_ ? 1.0 : 0.0;
  }

  py::tuple boxes() const {
    auto copied = state_->copy_bounds_rotations();
    return py::make_tuple(copied.first, copied.second,
                          state_->last_bbox_score());
  }

  py::tuple best_boxes() const {
    return py::make_tuple(best_bounds_, best_rotations_, best_score_);
  }

  double recompute_score(double cover_penalty, double pen_rate) {
    const double score = state_->score(cover_penalty, pen_rate);
    auto copied = state_->copy_bounds_rotations();
    state_->reset_to_state(copied.first, copied.second, score);
    state_->commit_current_as_initial();
    snapshot_best_from_state();
    stats_["native_recompute_score_runs"] += 1.0;
    return score;
  }

  std::unordered_map<std::string, double> stats() const {
    std::unordered_map<std::string, double> out = stats_;
    out["last_bbox_score"] = state_->last_bbox_score();
    out["best_bbox_score"] = best_score_;
    out["num_boxes"] = static_cast<double>(state_->num_boxes());
    out["volume_sum"] = volume_sum_;
    out["native_recenter_enabled"] = native_recenter_enabled_ ? 1.0 : 0.0;
    out["native_recenter_applies"] =
        static_cast<double>(native_recenter_applies_);
    out["native_recenter_invalid"] =
        static_cast<double>(native_recenter_invalid_);
    for (const auto& item : state_->cache_stats()) {
      out["manifold_" + item.first] = static_cast<double>(item.second);
    }
    return out;
  }

  py::dict run_refine(std::size_t max_steps,
                      double cover_penalty,
                      double pen_rate) {
    py::dict out;
    std::vector<std::size_t> actions;
    std::vector<double> rewards;
    double score = state_->last_bbox_score();
    if (native_recenter_enabled_) {
      for (std::size_t step = 0; step < max_steps; ++step) {
        const std::size_t n_boxes = state_->num_boxes();
        const double initial_best =
            -std::abs(state_->bvs() - 1.0) - state_->last_bbox_score();
        std::vector<bool> bbox_mask(n_boxes, true);
        SmartCppManifoldState::NativeBestAxisBatch axis_batch =
            state_->best_axis_actions_native(bbox_mask, cover_penalty,
                                             pen_rate, initial_best);
        std::intptr_t best_action = -1;
        double best_reward = initial_best;
        for (std::size_t idx = 0; idx < axis_batch.actions.size(); ++idx) {
          if (axis_batch.actions[idx] >= 0 &&
              axis_batch.rewards[idx] > best_reward) {
            best_reward = axis_batch.rewards[idx];
            best_action = axis_batch.actions[idx];
          }
        }
        for (std::size_t bbox_idx = 0; bbox_idx < n_boxes; ++bbox_idx) {
          RecenterCandidate candidate = recenter_candidate_for_bbox(bbox_idx);
          if (!candidate.valid) {
            continue;
          }
          const double candidate_score = state_->score_replacement(
              bbox_idx, candidate.bounds, candidate.rotation,
              cover_penalty, pen_rate);
          const double reward = candidate_score - state_->last_bbox_score();
          if (reward > best_reward) {
            best_reward = reward;
            best_action = static_cast<std::intptr_t>(
                bbox_idx * actions_per_bbox_ + (actions_per_bbox_ - 1));
          }
        }
        if (best_action < 0 || !std::isfinite(best_reward) ||
            best_reward <= 0.0) {
          break;
        }
        const double applied_reward = apply_native_action(
            static_cast<std::size_t>(best_action), cover_penalty, pen_rate);
        if (!std::isfinite(applied_reward) || applied_reward <= 0.0) {
          break;
        }
        actions.push_back(static_cast<std::size_t>(best_action));
        rewards.push_back(applied_reward);
        score = state_->last_bbox_score();
      }
      stats_["native_refine_recenter_enabled"] = 1.0;
    } else {
      SmartCppManifoldState::NativeGreedySegment result =
          state_->greedy_axis_refine_segment_native(cover_penalty, pen_rate,
                                                    max_steps);
      actions = result.actions;
      rewards = result.applied_rewards;
      score = result.last_score;
      stats_["native_refine_recenter_enabled"] = 0.0;
    }
    if (score >= best_score_) {
      snapshot_best_from_state();
    }
    stats_["native_refine_runs"] += 1.0;
    stats_["native_refine_steps"] += static_cast<double>(actions.size());
    stats_["native_refine_axis_only"] = native_recenter_enabled_ ? 0.0 : 1.0;
    out["actions"] = py::cast(actions);
    out["rewards"] = py::cast(rewards);
    out["last_bbox_score"] = py::float_(state_->last_bbox_score());
    out["best_bbox_score"] = py::float_(best_score_);
    out["axis_only"] = py::bool_(!native_recenter_enabled_);
    out["recenter_enabled"] = py::bool_(native_recenter_enabled_);
    return out;
  }

  py::dict run_mcts(std::size_t num_iter,
                    std::size_t max_step,
                    double cover_penalty,
                    double pen_rate,
                    double exp_weight,
                    double gamma,
                    std::uint64_t seed,
                    const std::vector<double>& action_prior_logits,
                    const std::vector<double>& action_value_logits,
                    double prior_weight,
                    double value_weight,
                    bool transposition_table = false,
                    std::size_t transposition_table_size = 8192,
                    std::size_t action_prior_top_k = 0) {
    std::mt19937_64 rng(seed);
    std::vector<MctsCppNode> nodes;
    std::unordered_map<std::string, MctsCppTranspositionEntry> transpositions;
    std::deque<std::string> transposition_order;
    std::size_t transposition_hits = 0;
    std::size_t prior_pruned_nodes = 0;
    std::size_t prior_pruned_actions = 0;
    std::size_t prior_kept_actions = 0;
    auto prune_node = [&](MctsCppNode& node) {
      if (action_prior_top_k == 0 ||
          (prior_weight == 0.0 && value_weight == 0.0) ||
          (action_prior_logits.empty() && action_value_logits.empty()) ||
          node.untried_actions.size() <= action_prior_top_k) {
        return;
      }
      std::vector<std::size_t> actions = node.untried_actions;
      std::sort(actions.begin(), actions.end(),
                [&](std::size_t left, std::size_t right) {
                  const double left_score = native_action_static_score(
                      left, action_prior_logits, action_value_logits,
                      prior_weight, value_weight);
                  const double right_score = native_action_static_score(
                      right, action_prior_logits, action_value_logits,
                      prior_weight, value_weight);
                  if (left_score != right_score) {
                    return left_score > right_score;
                  }
                  return left < right;
                });
      if (actions.size() > action_prior_top_k) {
        actions.resize(action_prior_top_k);
      }
      const std::size_t pruned =
          node.untried_actions.size() > actions.size()
              ? node.untried_actions.size() - actions.size()
              : 0;
      node.untried_actions = std::move(actions);
      node.action_mask.assign(num_actions_, true);
      for (std::size_t action : node.untried_actions) {
        if (action < node.action_mask.size()) {
          node.action_mask[action] = false;
        }
      }
      ++prior_pruned_nodes;
      prior_pruned_actions += pruned;
      prior_kept_actions += node.untried_actions.size();
    };
    auto seed_from_transposition = [&](MctsCppNode& node) {
      if (!transposition_table || node.state_key.empty()) {
        return;
      }
      auto found = transpositions.find(node.state_key);
      if (found == transpositions.end()) {
        return;
      }
      node.q = found->second.q;
      node.reward = found->second.reward;
      node.num_vis = found->second.num_vis;
      ++transposition_hits;
    };
    auto store_transposition = [&](std::size_t node_id) {
      if (!transposition_table || transposition_table_size == 0 ||
          node_id >= nodes.size()) {
        return;
      }
      const MctsCppNode& node = nodes[node_id];
      if (node.state_key.empty() || node.num_vis == 0) {
        return;
      }
      transpositions[node.state_key] =
          MctsCppTranspositionEntry{node.q, node.reward, node.num_vis};
      transposition_order.push_back(node.state_key);
      while (transpositions.size() > transposition_table_size &&
             !transposition_order.empty()) {
        const std::string old_key = transposition_order.front();
        transposition_order.pop_front();
        if (std::find(transposition_order.begin(), transposition_order.end(),
                      old_key) == transposition_order.end()) {
          transpositions.erase(old_key);
        }
      }
    };
    nodes.emplace_back(root_action_mask(), state_->state_key());
    prune_node(nodes.back());
    double best_return = -std::numeric_limits<double>::infinity();
    std::vector<double> best_rewards;
    std::vector<std::size_t> best_actions;
    std::size_t iterations_run = 0;
    for (std::size_t iter = 0; iter < num_iter; ++iter) {
      state_->reset_to_initial();
      std::vector<double> rewards;
      std::vector<std::size_t> path;
      std::vector<std::size_t> rollout_actions;
      std::size_t node_id = 0;
      std::size_t steps = 0;
      bool expanded = false;
      while (steps < max_step) {
        path.push_back(node_id);
        MctsCppNode& node = nodes[node_id];
        if (!node.untried_actions.empty()) {
          const std::size_t action = choose_native_untried_action(
              node.untried_actions, action_prior_logits, action_value_logits,
              prior_weight, value_weight, rng);
          const double reward =
              apply_native_action(action, cover_penalty, pen_rate);
          rewards.push_back(reward);
          rollout_actions.push_back(action);
          ++steps;
          const std::size_t child_id = nodes.size();
          nodes.emplace_back(child_action_mask(action, &node.action_mask),
                             state_->state_key());
          seed_from_transposition(nodes.back());
          prune_node(nodes.back());
          nodes[node_id].add_child(action, child_id);
          path.push_back(child_id);
          expanded = true;
          if (!std::isfinite(reward) || reward <= 0.0 || steps >= max_step) {
            break;
          }
          const std::size_t remaining = max_step - steps;
          if (remaining > 0) {
            std::vector<bool> mask(state_->num_boxes(), true);
            SmartCppManifoldState::NativeGreedySegment segment =
                state_->greedy_axis_rollout_segment_native(
                    mask, cover_penalty, pen_rate, remaining);
            const std::vector<std::size_t>& seg_actions = segment.actions;
            const std::vector<double>& seg_rewards = segment.applied_rewards;
            for (std::size_t idx = 0;
                 idx < seg_actions.size() && idx < seg_rewards.size(); ++idx) {
              if (!std::isfinite(seg_rewards[idx]) || seg_rewards[idx] <= 0.0) {
                break;
              }
              rollout_actions.push_back(seg_actions[idx]);
              rewards.push_back(seg_rewards[idx]);
              ++steps;
              if (steps >= max_step) {
                break;
              }
            }
          }
          break;
        }
        if (node.child_ids.empty()) {
          break;
        }
        const std::size_t pos = native_ucb_select_position(
            nodes, node, exp_weight, action_prior_logits, action_value_logits,
            prior_weight, value_weight, rng);
        const std::size_t action = node.child_actions[pos];
        const double reward =
            apply_native_action(action, cover_penalty, pen_rate);
        rewards.push_back(reward);
        rollout_actions.push_back(action);
        ++steps;
        node_id = node.child_ids[pos];
        if (!std::isfinite(reward) || reward <= 0.0) {
          break;
        }
      }
      const double total_return = discounted_rewards(rewards, gamma);
      backprop_native(nodes, path, total_return);
      for (std::size_t node_in_path : path) {
        store_transposition(node_in_path);
      }
      if (total_return > best_return) {
        best_return = total_return;
        best_rewards = rewards;
        best_actions = rollout_actions;
        snapshot_best_from_state();
      }
      if (!expanded && rewards.empty()) {
        break;
      }
      iterations_run = iter + 1;
    }
    state_->reset_to_state(best_bounds_, best_rotations_, best_score_);
    stats_["native_mcts_runs"] += 1.0;
    stats_["native_mcts_iterations"] += static_cast<double>(iterations_run);
    stats_["native_mcts_axis_only"] = native_recenter_enabled_ ? 0.0 : 1.0;
    stats_["native_mcts_tree"] = 1.0;
    stats_["native_mcts_nodes"] = static_cast<double>(nodes.size());
    stats_["native_mcts_transposition_table"] =
        transposition_table ? 1.0 : 0.0;
    stats_["native_mcts_transposition_hits"] =
        static_cast<double>(transposition_hits);
    stats_["native_mcts_transposition_table_size"] =
        static_cast<double>(transpositions.size());
    stats_["native_mcts_action_prior_top_k"] =
        static_cast<double>(action_prior_top_k);
    stats_["native_mcts_prior_pruned_nodes"] =
        static_cast<double>(prior_pruned_nodes);
    stats_["native_mcts_prior_pruned_actions"] =
        static_cast<double>(prior_pruned_actions);
    stats_["native_mcts_prior_kept_actions"] =
        static_cast<double>(prior_kept_actions);
    py::dict out;
    out["best_reward"] = py::float_(best_return);
    out["iterations_run"] = py::int_(iterations_run);
    out["node_count"] = py::int_(nodes.size());
    out["actions"] = py::cast(best_actions);
    out["rewards"] = py::cast(best_rewards);
    out["last_bbox_score"] = py::float_(state_->last_bbox_score());
    out["best_bbox_score"] = py::float_(best_score_);
    out["axis_only"] = py::bool_(!native_recenter_enabled_);
    out["recenter_enabled"] = py::bool_(native_recenter_enabled_);
    out["tree"] = py::bool_(true);
    out["transposition_hits"] = py::int_(transposition_hits);
    out["transposition_table_size"] = py::int_(transpositions.size());
    return out;
  }

  py::dict run_refine_then_mcts(std::size_t refine_max_steps,
                                std::size_t mcts_iter,
                                std::size_t mcts_max_step,
                                double cover_penalty,
                                double pen_rate,
                                double exp_weight,
                                double gamma,
                                std::uint64_t seed,
                                const std::vector<double>& action_prior_logits,
                                const std::vector<double>& action_value_logits,
                                double prior_weight,
                                double value_weight,
                                bool transposition_table = false,
                                std::size_t transposition_table_size = 8192,
                                std::size_t action_prior_top_k = 0) {
    py::dict refine = run_refine(refine_max_steps, cover_penalty, pen_rate);
    state_->commit_current_as_initial();
    snapshot_best_from_state();
    py::dict mcts = run_mcts(mcts_iter, mcts_max_step, cover_penalty, pen_rate,
                             exp_weight, gamma, seed, action_prior_logits,
                             action_value_logits, prior_weight, value_weight,
                             transposition_table, transposition_table_size,
                             action_prior_top_k);
    stats_["native_refine_then_mcts_runs"] += 1.0;
    stats_["native_refine_then_mcts_single_state_bridge"] = 1.0;
    py::dict out;
    out["command"] = py::str("refine-mcts");
    out["single_state_bridge"] = py::bool_(true);
    out["single_engine_state"] = py::bool_(true);
    out["refine"] = refine;
    out["mcts"] = mcts;
    out["last_bbox_score"] = py::float_(state_->last_bbox_score());
    out["best_bbox_score"] = py::float_(best_score_);
    out["stats"] = py::cast(stats());
    return out;
  }

  py::dict run_merge(
      const std::vector<std::vector<std::size_t>>& adjacency_pairs,
      double merge_eps,
      double shape_volume,
      std::size_t final_k = 0) {
    check_row_width(adjacency_pairs, 2, "adjacency_pairs");
    auto copied = state_->copy_bounds_rotations();
    std::vector<std::vector<double>> bounds = copied.first;
    const std::size_t n = bounds.size();
    const double denom = shape_volume > 0.0 ? shape_volume : volume_sum_;
    std::vector<double> volumes = native_bbox_volumes_py(bounds);
    std::vector<std::set<std::size_t>> neighbors(n);
    for (const auto& pair : adjacency_pairs) {
      const std::size_t a = pair[0];
      const std::size_t b = pair[1];
      if (a < n && b < n && a != b) {
        neighbors[a].insert(b);
        neighbors[b].insert(a);
      }
    }
    std::vector<std::uint8_t> active(n, 1);
    std::vector<std::uint64_t> versions(n, 0);
    std::vector<std::pair<std::size_t, std::size_t>> merges;
    std::vector<double> merge_rewards;
    std::size_t active_count = n;
    double active_volume_total = active_total_volume(bounds, active);
    std::set<MergeDeltaKey> ordered_candidates;
    std::map<std::pair<std::size_t, std::size_t>, MergeCandidate> candidates;
    std::size_t candidate_inserts = 0;
    std::size_t candidate_erases = 0;
    std::size_t candidate_queries = 0;
    auto pair_key = [](std::size_t a, std::size_t b) {
      if (a > b) {
        std::swap(a, b);
      }
      return std::make_pair(a, b);
    };
    auto erase_candidate = [&](std::size_t a, std::size_t b) {
      auto key = pair_key(a, b);
      auto found = candidates.find(key);
      if (found == candidates.end()) {
        return;
      }
      ordered_candidates.erase(MergeDeltaKey::from_candidate(found->second));
      candidates.erase(found);
      ++candidate_erases;
    };
    auto insert_candidate = [&](std::size_t a, std::size_t b) {
      if (a == b || a >= n || b >= n || !active[a] || !active[b]) {
        return;
      }
      auto key = pair_key(a, b);
      erase_candidate(key.first, key.second);
      MergeCandidate candidate = score_merge_candidate_delta(
          key.first, key.second, bounds, volumes, versions, denom);
      candidates.emplace(key, candidate);
      ordered_candidates.insert(MergeDeltaKey::from_candidate(candidate));
      ++candidate_inserts;
    };
    auto best_candidate = [&]() -> std::optional<MergeCandidate> {
      if (ordered_candidates.empty()) {
        return std::nullopt;
      }
      ++candidate_queries;
      const double prev_bvs = active_volume_total / denom;
      const double target_delta = 1.0 - prev_bvs;
      std::vector<MergeDeltaKey> probes;
      auto upper = ordered_candidates.lower_bound(
          MergeDeltaKey::target(target_delta));
      if (upper != ordered_candidates.end()) {
        probes.push_back(*upper);
      }
      if (upper != ordered_candidates.begin()) {
        probes.push_back(*std::prev(upper));
      }
      if (probes.empty()) {
        return std::nullopt;
      }
      MergeCandidate best;
      double best_reward = -std::numeric_limits<double>::infinity();
      bool have_best = false;
      for (const MergeDeltaKey& probe : probes) {
        auto found = candidates.find(pair_key(probe.left, probe.right));
        if (found == candidates.end()) {
          continue;
        }
        MergeCandidate candidate = found->second;
        candidate.reward = reward_for_merge_delta(prev_bvs, candidate.delta);
        if (!have_best || candidate.reward > best_reward ||
            (candidate.reward == best_reward &&
             std::tie(candidate.left, candidate.right) <
                 std::tie(best.left, best.right))) {
          best = candidate;
          best_reward = candidate.reward;
          have_best = true;
        }
      }
      if (!have_best) {
        return std::nullopt;
      }
      return best;
    };
    for (std::size_t a = 0; a < n; ++a) {
      if (!active[a]) {
        continue;
      }
      for (std::size_t b : neighbors[a]) {
        if (a < b && active[b]) {
          insert_candidate(a, b);
        }
      }
    }
    const double reward_threshold = -std::abs(merge_eps);
    while (active_count > 1) {
      if (final_k > 0 && active_count <= final_k) {
        break;
      }
      std::optional<MergeCandidate> selected = best_candidate();
      if (!selected.has_value()) {
        break;
      }
      MergeCandidate top = selected.value();
      if (final_k == 0 && !(top.reward > reward_threshold)) {
        break;
      }
      const std::size_t keep = top.left;
      const std::size_t drop = top.right;
      if (!active[keep] || !active[drop] ||
          versions[keep] != top.left_version ||
          versions[drop] != top.right_version) {
        erase_candidate(keep, drop);
        continue;
      }
      for (std::size_t nb : neighbors[keep]) {
        erase_candidate(keep, nb);
      }
      for (std::size_t nb : neighbors[drop]) {
        erase_candidate(drop, nb);
      }
      const double previous_keep_volume = volumes[keep];
      const double previous_drop_volume = volumes[drop];
      bounds[keep] = top.merged_bounds;
      volumes[keep] = bbox_volume_row(bounds[keep]);
      active_volume_total +=
          volumes[keep] - previous_keep_volume - previous_drop_volume;
      active[drop] = 0;
      active_count -= 1;
      versions[keep] += 1;
      versions[drop] += 1;
      std::set<std::size_t> merged_neighbors = neighbors[keep];
      merged_neighbors.insert(neighbors[drop].begin(), neighbors[drop].end());
      merged_neighbors.erase(keep);
      merged_neighbors.erase(drop);
      neighbors[keep] = merged_neighbors;
      for (std::size_t nb : merged_neighbors) {
        neighbors[nb].erase(drop);
        if (active[nb]) {
          neighbors[nb].insert(keep);
        }
      }
      neighbors[drop].clear();
      merges.push_back({keep, drop});
      merge_rewards.push_back(top.reward);
      for (std::size_t nb : neighbors[keep]) {
        if (active[nb]) {
          insert_candidate(keep, nb);
        }
      }
    }
    std::vector<std::size_t> active_indices;
    std::vector<std::vector<double>> active_bounds;
    for (std::size_t idx = 0; idx < n; ++idx) {
      if (active[idx]) {
        active_indices.push_back(idx);
        active_bounds.push_back(bounds[idx]);
      }
    }
    stats_["native_merge_runs"] += 1.0;
    stats_["native_merge_steps"] += static_cast<double>(merges.size());
    stats_["native_merge_heap"] = 1.0;
    stats_["native_merge_ordered_delta"] = 1.0;
    stats_["native_merge_candidate_inserts"] =
        static_cast<double>(candidate_inserts);
    stats_["native_merge_candidate_erases"] =
        static_cast<double>(candidate_erases);
    stats_["native_merge_candidate_queries"] =
        static_cast<double>(candidate_queries);
    py::dict out;
    out["merges"] = py::cast(merges);
    out["rewards"] = py::cast(merge_rewards);
    out["active_indices"] = py::cast(active_indices);
    out["bounds"] = py::cast(active_bounds);
    out["heap"] = py::bool_(true);
    out["ordered_delta_queue"] = py::bool_(true);
    out["candidate_inserts"] = py::int_(candidate_inserts);
    out["candidate_erases"] = py::int_(candidate_erases);
    out["candidate_queries"] = py::int_(candidate_queries);
    return out;
  }

  std::vector<std::vector<std::size_t>> partition_adjacency_pairs(
      const std::vector<std::vector<std::size_t>>& partitions,
      bool only_nearby) const {
    return build_partition_adjacency_pairs(partitions, only_nearby);
  }

  py::dict run_partition_merge_auto_adjacency(
      const std::vector<std::vector<std::size_t>>& partitions,
      bool only_nearby,
      double merge_eps,
      double shape_volume,
      std::size_t final_k = 0,
      bool tilted = false) {
    const auto adjacency_pairs =
        build_partition_adjacency_pairs(partitions, only_nearby);
    py::dict out = run_partition_merge(partitions, adjacency_pairs, merge_eps,
                                       shape_volume, final_k, tilted);
    out["adjacency_pairs"] = py::cast(adjacency_pairs);
    out["adjacency_pair_count"] = py::int_(adjacency_pairs.size());
    out["adjacency_only_nearby"] = py::bool_(only_nearby);
    return out;
  }

  py::dict run_partition_merge(
      const std::vector<std::vector<std::size_t>>& partitions,
      const std::vector<std::vector<std::size_t>>& adjacency_pairs,
      double merge_eps,
      double shape_volume,
      std::size_t final_k = 0,
      bool tilted = false) {
    check_row_width(adjacency_pairs, 2, "adjacency_pairs");
    const std::size_t n = partitions.size();
    const double denom = shape_volume > 0.0 ? shape_volume : volume_sum_;
    if (denom <= 0.0) {
      throw std::runtime_error("shape_volume must be positive");
    }

    struct PartitionCandidate {
      double reward = -std::numeric_limits<double>::infinity();
      double delta = 0.0;
      std::size_t left = 0;
      std::size_t right = 0;
      std::uint64_t left_version = 0;
      std::uint64_t right_version = 0;
      NativeBoxFit fit;
      std::vector<std::array<double, 3>> points;
      std::vector<std::size_t> partition;
    };

    std::vector<std::vector<std::array<double, 3>>> points(n);
    std::vector<std::vector<std::size_t>> partition_state = partitions;
    std::vector<std::vector<double>> bounds(n);
    std::vector<std::vector<double>> rotations(n);
    std::vector<double> volumes(n, 0.0);
    for (std::size_t idx = 0; idx < n; ++idx) {
      points[idx] = points_for_partition(partitions[idx]);
      NativeBoxFit fit = box_fit_from_points(points[idx], tilted);
      if (!fit.valid) {
        throw std::runtime_error("partition has no valid bbox fit");
      }
      bounds[idx] = fit.bounds;
      rotations[idx] = fit.rotation;
      volumes[idx] = fit.volume;
    }

    std::vector<std::set<std::size_t>> neighbors(n);
    for (const auto& pair : adjacency_pairs) {
      const std::size_t a = pair[0];
      const std::size_t b = pair[1];
      if (a < n && b < n && a != b) {
        neighbors[a].insert(b);
        neighbors[b].insert(a);
      }
    }
    std::vector<std::uint8_t> active(n, 1);
    std::vector<std::uint64_t> versions(n, 0);
    std::vector<std::pair<std::size_t, std::size_t>> merges;
    std::vector<double> merge_rewards;
    std::size_t active_count = n;
    double active_volume_total = 0.0;
    for (double volume : volumes) {
      active_volume_total += volume;
    }
    std::set<MergeDeltaKey> ordered_candidates;
    std::map<std::pair<std::size_t, std::size_t>, PartitionCandidate>
        candidates;
    std::size_t candidate_inserts = 0;
    std::size_t candidate_erases = 0;
    std::size_t candidate_queries = 0;
    auto pair_key = [](std::size_t a, std::size_t b) {
      if (a > b) {
        std::swap(a, b);
      }
      return std::make_pair(a, b);
    };
    auto key_from_candidate = [](const PartitionCandidate& candidate) {
      return MergeDeltaKey{candidate.delta, candidate.left, candidate.right};
    };
    auto erase_candidate = [&](std::size_t a, std::size_t b) {
      auto key = pair_key(a, b);
      auto found = candidates.find(key);
      if (found == candidates.end()) {
        return;
      }
      ordered_candidates.erase(key_from_candidate(found->second));
      candidates.erase(found);
      ++candidate_erases;
    };
    auto score_candidate = [&](std::size_t left, std::size_t right) {
      PartitionCandidate candidate;
      candidate.left = left;
      candidate.right = right;
      candidate.left_version = versions[left];
      candidate.right_version = versions[right];
      candidate.points = unique_point_union(points[left], points[right]);
      candidate.partition = partition_state[left];
      candidate.partition.insert(candidate.partition.end(),
                                 partition_state[right].begin(),
                                 partition_state[right].end());
      std::sort(candidate.partition.begin(), candidate.partition.end());
      candidate.fit = box_fit_from_points(candidate.points, tilted);
      candidate.delta =
          (candidate.fit.volume - volumes[left] - volumes[right]) / denom;
      return candidate;
    };
    auto insert_candidate = [&](std::size_t a, std::size_t b) {
      if (a == b || a >= n || b >= n || !active[a] || !active[b]) {
        return;
      }
      auto key = pair_key(a, b);
      erase_candidate(key.first, key.second);
      PartitionCandidate candidate = score_candidate(key.first, key.second);
      if (!candidate.fit.valid) {
        return;
      }
      candidates.emplace(key, candidate);
      ordered_candidates.insert(key_from_candidate(candidate));
      ++candidate_inserts;
    };
    auto best_candidate = [&]() -> std::optional<PartitionCandidate> {
      if (ordered_candidates.empty()) {
        return std::nullopt;
      }
      ++candidate_queries;
      const double prev_bvs = active_volume_total / denom;
      const double target_delta = 1.0 - prev_bvs;
      std::vector<MergeDeltaKey> probes;
      auto upper = ordered_candidates.lower_bound(
          MergeDeltaKey::target(target_delta));
      if (upper != ordered_candidates.end()) {
        probes.push_back(*upper);
      }
      if (upper != ordered_candidates.begin()) {
        probes.push_back(*std::prev(upper));
      }
      PartitionCandidate best;
      double best_reward = -std::numeric_limits<double>::infinity();
      bool have_best = false;
      for (const MergeDeltaKey& probe : probes) {
        auto found = candidates.find(pair_key(probe.left, probe.right));
        if (found == candidates.end()) {
          continue;
        }
        PartitionCandidate candidate = found->second;
        candidate.reward = reward_for_merge_delta(prev_bvs, candidate.delta);
        if (!have_best || candidate.reward > best_reward ||
            (candidate.reward == best_reward &&
             std::tie(candidate.left, candidate.right) <
                 std::tie(best.left, best.right))) {
          best = std::move(candidate);
          best_reward = best.reward;
          have_best = true;
        }
      }
      if (!have_best) {
        return std::nullopt;
      }
      return best;
    };
    for (std::size_t a = 0; a < n; ++a) {
      for (std::size_t b : neighbors[a]) {
        if (a < b) {
          insert_candidate(a, b);
        }
      }
    }

    const double reward_threshold = -std::abs(merge_eps);
    while (active_count > 1) {
      if (final_k > 0 && active_count <= final_k) {
        break;
      }
      std::optional<PartitionCandidate> selected = best_candidate();
      if (!selected.has_value()) {
        break;
      }
      PartitionCandidate top = selected.value();
      if (final_k == 0 && !(top.reward > reward_threshold)) {
        break;
      }
      const std::size_t keep = top.left;
      const std::size_t drop = top.right;
      if (!active[keep] || !active[drop] ||
          versions[keep] != top.left_version ||
          versions[drop] != top.right_version) {
        erase_candidate(keep, drop);
        continue;
      }
      for (std::size_t nb : neighbors[keep]) {
        erase_candidate(keep, nb);
      }
      for (std::size_t nb : neighbors[drop]) {
        erase_candidate(drop, nb);
      }
      const double previous_keep_volume = volumes[keep];
      const double previous_drop_volume = volumes[drop];
      points[keep] = std::move(top.points);
      partition_state[keep] = std::move(top.partition);
      bounds[keep] = std::move(top.fit.bounds);
      rotations[keep] = std::move(top.fit.rotation);
      volumes[keep] = top.fit.volume;
      active_volume_total +=
          volumes[keep] - previous_keep_volume - previous_drop_volume;
      active[drop] = 0;
      --active_count;
      ++versions[keep];
      ++versions[drop];
      std::set<std::size_t> merged_neighbors = neighbors[keep];
      merged_neighbors.insert(neighbors[drop].begin(), neighbors[drop].end());
      merged_neighbors.erase(keep);
      merged_neighbors.erase(drop);
      neighbors[keep] = merged_neighbors;
      for (std::size_t nb : merged_neighbors) {
        neighbors[nb].erase(drop);
        if (active[nb]) {
          neighbors[nb].insert(keep);
        }
      }
      neighbors[drop].clear();
      partition_state[drop].clear();
      points[drop].clear();
      merges.push_back({keep, drop});
      merge_rewards.push_back(top.reward);
      for (std::size_t nb : neighbors[keep]) {
        if (active[nb]) {
          insert_candidate(keep, nb);
        }
      }
    }

    std::vector<std::size_t> active_indices;
    std::vector<std::vector<std::size_t>> active_partitions;
    std::vector<std::vector<double>> active_bounds;
    std::vector<std::vector<double>> active_rotations;
    for (std::size_t idx = 0; idx < n; ++idx) {
      if (!active[idx]) {
        continue;
      }
      active_indices.push_back(idx);
      active_partitions.push_back(partition_state[idx]);
      active_bounds.push_back(bounds[idx]);
      active_rotations.push_back(rotations[idx]);
    }
    stats_["native_partition_merge_runs"] += 1.0;
    stats_["native_partition_merge_steps"] += static_cast<double>(merges.size());
    stats_["native_partition_merge_tilted"] = tilted ? 1.0 : 0.0;
    py::dict out;
    out["merges"] = py::cast(merges);
    out["rewards"] = py::cast(merge_rewards);
    out["active_indices"] = py::cast(active_indices);
    out["partitions"] = py::cast(active_partitions);
    out["bounds"] = py::cast(active_bounds);
    out["rotations"] = py::cast(active_rotations);
    out["tilted"] = py::bool_(tilted);
    out["ordered_delta_queue"] = py::bool_(true);
    out["candidate_inserts"] = py::int_(candidate_inserts);
    out["candidate_erases"] = py::int_(candidate_erases);
    out["candidate_queries"] = py::int_(candidate_queries);
    return out;
  }

  std::size_t export_obj(const std::string& path) const {
    auto copied = state_->copy_bounds_rotations();
    const std::vector<float> flat =
        flatten_bridge_oriented_box_vertices(copied.first, copied.second);
    std::ofstream output(path);
    if (!output) {
      throw std::runtime_error("failed to open native SMART OBJ output");
    }
    static const int faces[12][3] = {
        {0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6},
        {0, 1, 5}, {0, 5, 4}, {2, 6, 7}, {2, 7, 3},
        {0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5},
    };
    std::size_t vertex_offset = 0;
    for (std::size_t box_idx = 0; box_idx < copied.first.size(); ++box_idx) {
      output << "o bbox_" << box_idx << "\n";
      const std::size_t base = box_idx * 8 * 3;
      for (std::size_t idx = 0; idx < 8; ++idx) {
        output << "v " << flat[base + idx * 3] << " "
               << flat[base + idx * 3 + 1] << " "
               << flat[base + idx * 3 + 2] << "\n";
      }
      for (const auto& face : faces) {
        output << "f " << vertex_offset + face[0] + 1 << " "
               << vertex_offset + face[1] + 1 << " "
               << vertex_offset + face[2] + 1 << "\n";
      }
      vertex_offset += 8;
    }
    return copied.first.size();
  }

  std::size_t export_bbox_dir(const std::string& directory) const {
    auto copied = state_->copy_bounds_rotations();
    const std::vector<float> flat =
        flatten_bridge_oriented_box_vertices(copied.first, copied.second);
    ensure_directories(directory);
    static const int faces[12][3] = {
        {0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6},
        {0, 1, 5}, {0, 5, 4}, {2, 6, 7}, {2, 7, 3},
        {0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5},
    };
    for (std::size_t box_idx = 0; box_idx < copied.first.size(); ++box_idx) {
      const std::string path =
          join_path(directory, "bbox" + std::to_string(box_idx) + ".obj");
      std::ofstream output(path);
      if (!output) {
        throw std::runtime_error("failed to open native bbox OBJ output");
      }
      output << "o bbox_" << box_idx << "\n";
      const std::size_t base = box_idx * 8 * 3;
      for (std::size_t idx = 0; idx < 8; ++idx) {
        output << "v " << flat[base + idx * 3] << " "
               << flat[base + idx * 3 + 1] << " "
               << flat[base + idx * 3 + 2] << "\n";
      }
      for (const auto& face : faces) {
        output << "f " << face[0] + 1 << " " << face[1] + 1 << " "
               << face[2] + 1 << "\n";
      }
    }
    write_bbox_params_json(directory, copied.first, copied.second);
    return copied.first.size();
  }

 private:
  struct MergeCandidate {
    double reward = -std::numeric_limits<double>::infinity();
    double delta = 0.0;
    std::size_t left = 0;
    std::size_t right = 0;
    std::uint64_t left_version = 0;
    std::uint64_t right_version = 0;
    std::vector<double> merged_bounds;

    bool operator<(const MergeCandidate& other) const {
      if (reward != other.reward) {
        return reward < other.reward;
      }
      if (left != other.left) {
        return left > other.left;
      }
      return right > other.right;
    }
  };

  struct MergeDeltaKey {
    double delta = 0.0;
    std::size_t left = 0;
    std::size_t right = 0;

    static MergeDeltaKey from_candidate(const MergeCandidate& candidate) {
      return MergeDeltaKey{candidate.delta, candidate.left, candidate.right};
    }

    static MergeDeltaKey target(double delta) {
      return MergeDeltaKey{delta, 0, 0};
    }

    bool operator<(const MergeDeltaKey& other) const {
      if (delta != other.delta) {
        return delta < other.delta;
      }
      if (left != other.left) {
        return left < other.left;
      }
      return right < other.right;
    }
  };

  struct FaceKey {
    std::array<std::size_t, 3> vertices;

    bool operator==(const FaceKey& other) const {
      return vertices == other.vertices;
    }
  };

  struct FaceKeyHash {
    std::size_t operator()(const FaceKey& key) const {
      std::size_t hash = 1469598103934665603ull;
      for (std::size_t value : key.vertices) {
        hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };

  static FaceKey make_face_key(std::size_t a,
                               std::size_t b,
                               std::size_t c) {
    std::array<std::size_t, 3> vertices = {a, b, c};
    std::sort(vertices.begin(), vertices.end());
    return FaceKey{vertices};
  }

  std::vector<std::vector<std::size_t>> build_partition_adjacency_pairs(
      const std::vector<std::vector<std::size_t>>& partitions,
      bool only_nearby) const {
    const std::size_t n = partitions.size();
    std::set<std::pair<std::size_t, std::size_t>> pairs;
    if (n < 2) {
      return {};
    }
    if (!only_nearby) {
      for (std::size_t left = 0; left < n; ++left) {
        if (partitions[left].empty()) {
          continue;
        }
        for (std::size_t right = left + 1; right < n; ++right) {
          if (!partitions[right].empty()) {
            pairs.emplace(left, right);
          }
        }
      }
    } else {
      const std::size_t invalid = std::numeric_limits<std::size_t>::max();
      std::vector<std::size_t> voxel_partition(voxels_.size(), invalid);
      for (std::size_t part_idx = 0; part_idx < n; ++part_idx) {
        for (std::size_t voxel_idx : partitions[part_idx]) {
          if (voxel_idx < voxel_partition.size()) {
            voxel_partition[voxel_idx] = part_idx;
          }
        }
      }
      std::unordered_map<FaceKey, std::size_t, FaceKeyHash> owner_by_face;
      static const std::size_t local_faces[4][3] = {
          {0, 1, 2},
          {0, 1, 3},
          {0, 2, 3},
          {1, 2, 3},
      };
      for (std::size_t voxel_idx = 0; voxel_idx < voxels_.size(); ++voxel_idx) {
        const std::size_t part_idx = voxel_partition[voxel_idx];
        if (part_idx == invalid || voxels_[voxel_idx].size() != 4) {
          continue;
        }
        for (const auto& face : local_faces) {
          FaceKey key = make_face_key(voxels_[voxel_idx][face[0]],
                                      voxels_[voxel_idx][face[1]],
                                      voxels_[voxel_idx][face[2]]);
          auto inserted = owner_by_face.emplace(key, part_idx);
          if (!inserted.second && inserted.first->second != part_idx) {
            std::size_t left = inserted.first->second;
            std::size_t right = part_idx;
            if (left > right) {
              std::swap(left, right);
            }
            pairs.emplace(left, right);
          }
        }
      }
    }
    std::vector<std::vector<std::size_t>> out;
    out.reserve(pairs.size());
    for (const auto& pair : pairs) {
      out.push_back({pair.first, pair.second});
    }
    return out;
  }

  static double resolve_volume_sum(double volume_sum,
                                   const std::vector<double>& tet_volumes) {
    if (volume_sum > 0.0) {
      return volume_sum;
    }
    double total = 0.0;
    for (double value : tet_volumes) {
      total += value;
    }
    return total;
  }

  static double bbox_volume_row(const std::vector<double>& bounds) {
    if (bounds.size() != 6 || bounds[0] >= bounds[3] ||
        bounds[1] >= bounds[4] || bounds[2] >= bounds[5]) {
      return 0.0;
    }
    return (bounds[3] - bounds[0]) * (bounds[4] - bounds[1]) *
           (bounds[5] - bounds[2]);
  }

  static double active_total_volume(
      const std::vector<std::vector<double>>& bounds,
      const std::vector<std::uint8_t>& active) {
    double total = 0.0;
    for (std::size_t idx = 0; idx < bounds.size(); ++idx) {
      if (active[idx]) {
        total += bbox_volume_row(bounds[idx]);
      }
    }
    return total;
  }

  static double reward_for_merge_delta(double prev_bvs, double delta) {
    return -std::abs(prev_bvs + delta - 1.0) + std::abs(prev_bvs - 1.0);
  }

  static void write_json_number_array(std::ofstream& output,
                                      const std::vector<double>& values) {
    output << "[";
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
      if (idx > 0) {
        output << ",";
      }
      output << std::setprecision(17) << values[idx];
    }
    output << "]";
  }

  static void write_bbox_params_json(
      const std::string& directory,
      const std::vector<std::vector<double>>& bounds,
      const std::vector<std::vector<double>>& rotations) {
    const std::string path = join_path(directory, "bbox_params.json");
    std::ofstream output(path);
    if (!output) {
      throw std::runtime_error("failed to open native bbox params output");
    }
    output << "{\n";
    output << "  \"schema_version\": 1,\n";
    output << "  \"source\": \"smart._cpp.NativeSmartEngine\",\n";
    output << "  \"boxes\": [\n";
    for (std::size_t idx = 0; idx < bounds.size(); ++idx) {
      output << "    {\"index\": " << idx << ", \"bounds\": ";
      write_json_number_array(output, bounds[idx]);
      output << ", \"rotation\": ";
      write_json_number_array(output, rotations[idx]);
      output << "}";
      if (idx + 1 < bounds.size()) {
        output << ",";
      }
      output << "\n";
    }
    output << "  ]\n";
    output << "}\n";
  }

  static std::string join_path(const std::string& directory,
                               const std::string& filename) {
    if (directory.empty()) {
      return filename;
    }
    const char last = directory[directory.size() - 1];
    if (last == '/' || last == '\\') {
      return directory + filename;
    }
    return directory + "/" + filename;
  }

  static void ensure_directories(const std::string& directory) {
    if (directory.empty()) {
      return;
    }
    std::string current;
    current.reserve(directory.size());
    for (std::size_t idx = 0; idx < directory.size(); ++idx) {
      const char ch = directory[idx];
      current.push_back(ch);
      if (ch != '/' || current.size() == 1) {
        continue;
      }
      mkdir_if_missing(current);
    }
    mkdir_if_missing(current);
  }

  static void mkdir_if_missing(const std::string& path) {
    if (path.empty()) {
      return;
    }
    if (::mkdir(path.c_str(), 0777) == 0) {
      return;
    }
    if (errno == EEXIST) {
      return;
    }
    throw std::runtime_error("failed to create native bbox output directory: " +
                             path);
  }

  static MergeCandidate score_merge_candidate_delta(
      std::size_t left,
      std::size_t right,
      const std::vector<std::vector<double>>& bounds,
      const std::vector<double>& volumes,
      const std::vector<std::uint64_t>& versions,
      double shape_volume) {
    const std::vector<double> merged_bounds =
        native_bbox_union_bounds_py({bounds[left], bounds[right]});
    const double merged_volume = bbox_volume_row(merged_bounds);
    const double delta =
        (merged_volume - volumes[left] - volumes[right]) / shape_volume;
    return MergeCandidate{-std::numeric_limits<double>::infinity(),
                          delta,
                          left,
                          right,
                          versions[left],
                          versions[right],
                          merged_bounds};
  }

  std::vector<std::array<double, 3>> points_for_partition(
      const std::vector<std::size_t>& partition) const {
    std::set<std::array<double, 3>> unique;
    for (std::size_t voxel_idx : partition) {
      if (voxel_idx >= voxels_.size()) {
        continue;
      }
      for (std::size_t vertex_idx : voxels_[voxel_idx]) {
        if (vertex_idx >= vertices_.size() || vertices_[vertex_idx].size() != 3) {
          continue;
        }
        const auto& vertex = vertices_[vertex_idx];
        if (!std::isfinite(vertex[0]) || !std::isfinite(vertex[1]) ||
            !std::isfinite(vertex[2])) {
          continue;
        }
        unique.insert({vertex[0], vertex[1], vertex[2]});
      }
    }
    return {unique.begin(), unique.end()};
  }

  static std::vector<std::array<double, 3>> unique_point_union(
      const std::vector<std::array<double, 3>>& left,
      const std::vector<std::array<double, 3>>& right) {
    std::set<std::array<double, 3>> unique;
    unique.insert(left.begin(), left.end());
    unique.insert(right.begin(), right.end());
    return {unique.begin(), unique.end()};
  }

  static NativeBoxFit axis_fit_from_points(
      const std::vector<std::array<double, 3>>& points) {
    NativeBoxFit fit;
    if (points.empty()) {
      return fit;
    }
    std::array<double, 3> mn = points.front();
    std::array<double, 3> mx = points.front();
    for (const auto& point : points) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        mn[axis] = std::min(mn[axis], point[axis]);
        mx[axis] = std::max(mx[axis], point[axis]);
      }
    }
    fit.bounds = {mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]};
    fit.rotation = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    fit.volume = bbox_volume_row(fit.bounds);
    fit.valid = fit.volume > 0.0;
    return fit;
  }

  static NativeBoxFit box_fit_from_points(
      const std::vector<std::array<double, 3>>& points,
      bool tilted) {
    if (!tilted || !recenter_points_have_area(points)) {
      return axis_fit_from_points(points);
    }
    const auto rot_rows = pca_rotation_rows(
        points, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    std::array<double, 3> mn = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    std::array<double, 3> mx = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (const auto& point : points) {
      const std::array<double, 3> local = {
          dot3_array(point, rot_rows[0]),
          dot3_array(point, rot_rows[1]),
          dot3_array(point, rot_rows[2]),
      };
      for (std::size_t axis = 0; axis < 3; ++axis) {
        mn[axis] = std::min(mn[axis], local[axis]);
        mx[axis] = std::max(mx[axis], local[axis]);
      }
    }
    NativeBoxFit fit;
    fit.bounds = {mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]};
    fit.rotation = flatten_rotation_rows(rot_rows);
    fit.volume = bbox_volume_row(fit.bounds);
    fit.valid = fit.volume > 0.0;
    if (!fit.valid) {
      return axis_fit_from_points(points);
    }
    return fit;
  }

  std::vector<std::size_t> axis_actions() const {
    const std::size_t per_bbox = 6 * num_action_scale_ + 1;
    std::vector<std::size_t> actions;
    actions.reserve(state_->num_boxes() * 6 * num_action_scale_);
    for (std::size_t bbox_idx = 0; bbox_idx < state_->num_boxes(); ++bbox_idx) {
      const std::size_t start = bbox_idx * per_bbox;
      for (std::size_t local = 0; local < 6 * num_action_scale_; ++local) {
        actions.push_back(start + local);
      }
    }
    return actions;
  }

  struct RecenterCandidate {
    bool valid = false;
    std::size_t bbox_idx = 0;
    std::vector<double> bounds;
    std::vector<double> rotation;
  };

  static bool finite_point3(const std::vector<double>& point) {
    return point.size() == 3 && std::isfinite(point[0]) &&
           std::isfinite(point[1]) && std::isfinite(point[2]);
  }

  static bool point_in_oriented_bounds_cpp(const std::vector<double>& point,
                                           const std::vector<double>& bounds,
                                           const std::vector<double>& rotation) {
    const double x = point[0] * rotation[0] + point[1] * rotation[1] +
                     point[2] * rotation[2];
    const double y = point[0] * rotation[3] + point[1] * rotation[4] +
                     point[2] * rotation[5];
    const double z = point[0] * rotation[6] + point[1] * rotation[7] +
                     point[2] * rotation[8];
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z) &&
           bounds[0] <= x && x <= bounds[3] && bounds[1] <= y &&
           y <= bounds[4] && bounds[2] <= z && z <= bounds[5];
  }

  static double dot3_array(const std::array<double, 3>& left,
                           const std::array<double, 3>& right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
  }

  static std::array<double, 3> cross3_array(
      const std::array<double, 3>& left,
      const std::array<double, 3>& right) {
    return {left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0]};
  }

  static bool normalize3_array(std::array<double, 3>& value) {
    const double norm = std::sqrt(dot3_array(value, value));
    if (!std::isfinite(norm) || norm <= 1e-12) {
      return false;
    }
    value[0] /= norm;
    value[1] /= norm;
    value[2] /= norm;
    return true;
  }

  static void canonicalize_axis_sign(std::array<double, 3>& axis) {
    std::size_t max_idx = 0;
    double max_abs = std::abs(axis[0]);
    for (std::size_t idx = 1; idx < 3; ++idx) {
      const double abs_value = std::abs(axis[idx]);
      if (abs_value > max_abs) {
        max_abs = abs_value;
        max_idx = idx;
      }
    }
    if (axis[max_idx] < 0.0) {
      axis[0] *= -1.0;
      axis[1] *= -1.0;
      axis[2] *= -1.0;
    }
  }

  static std::vector<std::array<double, 3>> recenter_points_from_tets(
      const std::vector<std::vector<double>>& vertices,
      const std::vector<std::vector<std::size_t>>& voxels,
      const std::vector<std::vector<double>>& centroids,
      const std::vector<double>& bounds,
      const std::vector<double>& rotation) {
    std::vector<std::array<double, 3>> selected;
    if (bounds.size() != 6 || rotation.size() != 9 ||
        voxels.size() != centroids.size()) {
      return selected;
    }
    selected.reserve(voxels.size() * 4);
    for (std::size_t voxel_idx = 0; voxel_idx < voxels.size(); ++voxel_idx) {
      if (!finite_point3(centroids[voxel_idx]) ||
          !point_in_oriented_bounds_cpp(centroids[voxel_idx], bounds,
                                        rotation)) {
        continue;
      }
      for (std::size_t vertex_idx : voxels[voxel_idx]) {
        if (vertex_idx >= vertices.size() || !finite_point3(vertices[vertex_idx])) {
          continue;
        }
        const auto& vertex = vertices[vertex_idx];
        selected.push_back({vertex[0], vertex[1], vertex[2]});
      }
    }
    return selected;
  }

  static bool recenter_points_have_area(
      const std::vector<std::array<double, 3>>& points) {
    if (points.size() < 4) {
      return false;
    }
    std::array<double, 3> mn = points.front();
    std::array<double, 3> mx = points.front();
    for (const auto& point : points) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        mn[axis] = std::min(mn[axis], point[axis]);
        mx[axis] = std::max(mx[axis], point[axis]);
      }
    }
    std::size_t varying_axes = 0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (mx[axis] - mn[axis] > 1e-9) {
        ++varying_axes;
      }
    }
    return varying_axes >= 2;
  }

  static std::array<std::array<double, 3>, 3> pca_rotation_rows(
      const std::vector<std::array<double, 3>>& points,
      const std::vector<double>& fallback_rotation) {
    std::array<std::array<double, 3>, 3> rows = {{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    if (fallback_rotation.size() == 9) {
      rows = {{
          {fallback_rotation[0], fallback_rotation[1], fallback_rotation[2]},
          {fallback_rotation[3], fallback_rotation[4], fallback_rotation[5]},
          {fallback_rotation[6], fallback_rotation[7], fallback_rotation[8]},
      }};
    }
    if (!recenter_points_have_area(points)) {
      return rows;
    }

    std::array<double, 3> mean = {0.0, 0.0, 0.0};
    for (const auto& point : points) {
      mean[0] += point[0];
      mean[1] += point[1];
      mean[2] += point[2];
    }
    const double inv_n = 1.0 / static_cast<double>(points.size());
    mean[0] *= inv_n;
    mean[1] *= inv_n;
    mean[2] *= inv_n;

    double a[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    for (const auto& point : points) {
      const double d[3] = {point[0] - mean[0], point[1] - mean[1],
                           point[2] - mean[2]};
      for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = i; j < 3; ++j) {
          a[i][j] += d[i] * d[j] * inv_n;
        }
      }
    }
    a[1][0] = a[0][1];
    a[2][0] = a[0][2];
    a[2][1] = a[1][2];
    double v[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    for (std::size_t iter = 0; iter < 32; ++iter) {
      std::size_t p = 0;
      std::size_t q = 1;
      double max_offdiag = std::abs(a[0][1]);
      if (std::abs(a[0][2]) > max_offdiag) {
        p = 0;
        q = 2;
        max_offdiag = std::abs(a[0][2]);
      }
      if (std::abs(a[1][2]) > max_offdiag) {
        p = 1;
        q = 2;
        max_offdiag = std::abs(a[1][2]);
      }
      if (max_offdiag < 1e-12) {
        break;
      }
      const double app = a[p][p];
      const double aqq = a[q][q];
      const double apq = a[p][q];
      const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
      const double c = std::cos(phi);
      const double s = std::sin(phi);
      for (std::size_t k = 0; k < 3; ++k) {
        const double aik = a[p][k];
        const double aqk = a[q][k];
        a[p][k] = c * aik - s * aqk;
        a[q][k] = s * aik + c * aqk;
      }
      for (std::size_t k = 0; k < 3; ++k) {
        const double akp = a[k][p];
        const double akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (std::size_t k = 0; k < 3; ++k) {
        const double vip = v[k][p];
        const double viq = v[k][q];
        v[k][p] = c * vip - s * viq;
        v[k][q] = s * vip + c * viq;
      }
    }

    std::array<std::size_t, 3> order = {0, 1, 2};
    std::sort(order.begin(), order.end(), [&](std::size_t left,
                                               std::size_t right) {
      return a[left][left] > a[right][right];
    });
    for (std::size_t row = 0; row < 3; ++row) {
      const std::size_t col = order[row];
      rows[row] = {v[0][col], v[1][col], v[2][col]};
      if (!normalize3_array(rows[row])) {
        return {{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
        }};
      }
      canonicalize_axis_sign(rows[row]);
    }
    rows[2] = cross3_array(rows[0], rows[1]);
    if (!normalize3_array(rows[2])) {
      return {{
          {1.0, 0.0, 0.0},
          {0.0, 1.0, 0.0},
          {0.0, 0.0, 1.0},
      }};
    }
    rows[1] = cross3_array(rows[2], rows[0]);
    normalize3_array(rows[1]);
    canonicalize_axis_sign(rows[0]);
    canonicalize_axis_sign(rows[1]);
    canonicalize_axis_sign(rows[2]);
    return rows;
  }

  static std::vector<double> flatten_rotation_rows(
      const std::array<std::array<double, 3>, 3>& rows) {
    return {rows[0][0], rows[0][1], rows[0][2],
            rows[1][0], rows[1][1], rows[1][2],
            rows[2][0], rows[2][1], rows[2][2]};
  }

  static RecenterCandidate recenter_candidate_from_points(
      std::size_t bbox_idx,
      const std::vector<double>& bounds,
      const std::vector<double>& rotation,
      const std::vector<std::array<double, 3>>& points) {
    RecenterCandidate candidate;
    candidate.bbox_idx = bbox_idx;
    if (bounds.size() != 6 || rotation.size() != 9 ||
        !recenter_points_have_area(points)) {
      return candidate;
    }
    const auto rot_rows = pca_rotation_rows(points, rotation);
    std::array<double, 3> mn = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    std::array<double, 3> mx = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (const auto& point : points) {
      const std::array<double, 3> local = {
          dot3_array(point, rot_rows[0]),
          dot3_array(point, rot_rows[1]),
          dot3_array(point, rot_rows[2]),
      };
      for (std::size_t axis = 0; axis < 3; ++axis) {
        mn[axis] = std::min(mn[axis], local[axis]);
        mx[axis] = std::max(mx[axis], local[axis]);
      }
    }
    std::vector<double> out_bounds = bounds;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double center = 0.5 * (mn[axis] + mx[axis]);
      const double current_center =
          0.5 * (bounds[axis] + bounds[axis + 3]);
      const double shift = center - current_center;
      out_bounds[axis] += shift;
      out_bounds[axis + 3] += shift;
      if (!std::isfinite(out_bounds[axis]) ||
          !std::isfinite(out_bounds[axis + 3]) ||
          out_bounds[axis + 3] - out_bounds[axis] <= 1e-9) {
        return candidate;
      }
    }
    candidate.valid = true;
    candidate.bounds = std::move(out_bounds);
    candidate.rotation = flatten_rotation_rows(rot_rows);
    return candidate;
  }

  RecenterCandidate recenter_candidate_for_bbox(std::size_t bbox_idx) const {
    RecenterCandidate candidate;
    candidate.bbox_idx = bbox_idx;
    if (!native_recenter_enabled_ || bbox_idx >= state_->num_boxes()) {
      return candidate;
    }
    auto params = state_->bbox_params(bbox_idx);
    std::vector<std::array<double, 3>> points = recenter_points_from_tets(
        vertices_, voxels_, centroids_, params.first, params.second);
    return recenter_candidate_from_points(bbox_idx, params.first,
                                          params.second, points);
  }

  bool is_recenter_action(std::size_t action) const {
    return actions_per_bbox_ > 0 &&
           action < num_actions_ &&
           action % actions_per_bbox_ == actions_per_bbox_ - 1;
  }

  std::vector<bool> root_action_mask() const {
    std::vector<bool> mask(num_actions_, false);
    if (!native_recenter_enabled_) {
      mask_recenter_actions(mask);
    }
    return mask;
  }

  void mask_recenter_actions(std::vector<bool>& mask) const {
    if (actions_per_bbox_ == 0) {
      return;
    }
    for (std::size_t start = actions_per_bbox_ - 1; start < mask.size();
         start += actions_per_bbox_) {
      mask[start] = true;
    }
  }

  std::vector<bool> child_action_mask(
      std::size_t action, const std::vector<bool>* parent_mask) const {
    std::vector<bool> mask =
        parent_mask == nullptr ? std::vector<bool>(num_actions_, false)
                               : *parent_mask;
    if (is_recenter_action(action)) {
      mask[action] = true;
    } else if (action < opposite_actions_.size()) {
      mask[opposite_actions_[action]] = true;
    }
    if (!native_recenter_enabled_) {
      mask_recenter_actions(mask);
    }
    return mask;
  }

  double apply_native_recenter_action(std::size_t action,
                                      double cover_penalty,
                                      double pen_rate) {
    if (!is_recenter_action(action) || !native_recenter_enabled_) {
      return -std::numeric_limits<double>::infinity();
    }
    const std::size_t bbox_idx = action / actions_per_bbox_;
    RecenterCandidate candidate = recenter_candidate_for_bbox(bbox_idx);
    if (!candidate.valid) {
      ++native_recenter_invalid_;
      return -std::numeric_limits<double>::infinity();
    }
    try {
      SmartCppManifoldState::NativeReplacementApply applied =
          state_->apply_replacement_delta_native(
              bbox_idx, candidate.bounds, candidate.rotation,
              cover_penalty, pen_rate);
      ++native_recenter_applies_;
      return applied.reward;
    } catch (const std::exception&) {
      ++native_recenter_invalid_;
      return -std::numeric_limits<double>::infinity();
    }
  }

  double apply_native_action(std::size_t action,
                             double cover_penalty,
                             double pen_rate) {
    if (is_recenter_action(action)) {
      return apply_native_recenter_action(action, cover_penalty, pen_rate);
    }
    return apply_native_axis_action(action, cover_penalty, pen_rate);
  }

  double apply_native_axis_action(std::size_t action,
                                  double cover_penalty,
                                  double pen_rate) {
    try {
      SmartCppManifoldState::NativeAxisApply applied =
          state_->apply_axis_action_delta_native(
              static_cast<std::intptr_t>(action), cover_penalty, pen_rate);
      return applied.reward;
    } catch (const std::exception&) {
      return -std::numeric_limits<double>::infinity();
    }
  }

  static double native_action_static_score(
      std::size_t action,
      const std::vector<double>& prior_logits,
      const std::vector<double>& value_logits,
      double prior_weight,
      double value_weight) {
    double score = 0.0;
    if (action < prior_logits.size()) {
      score += prior_weight * prior_logits[action];
    }
    if (action < value_logits.size()) {
      score += value_weight * value_logits[action];
    }
    return score;
  }

  std::size_t choose_native_untried_action(
      const std::vector<std::size_t>& actions,
      const std::vector<double>& prior_logits,
      const std::vector<double>& value_logits,
      double prior_weight,
      double value_weight,
      std::mt19937_64& rng) const {
    if (actions.empty()) {
      throw std::runtime_error("native MCTS untried action list is empty");
    }
    if (prior_weight == 0.0 && value_weight == 0.0) {
      std::uniform_int_distribution<std::size_t> dist(0, actions.size() - 1);
      return actions[dist(rng)];
    }
    double best_score = -std::numeric_limits<double>::infinity();
    std::vector<std::size_t> best_actions;
    for (std::size_t action : actions) {
      const double score = native_action_static_score(
          action, prior_logits, value_logits, prior_weight, value_weight);
      if (score > best_score) {
        best_score = score;
        best_actions.clear();
        best_actions.push_back(action);
      } else if (score == best_score) {
        best_actions.push_back(action);
      }
    }
    std::uniform_int_distribution<std::size_t> dist(0, best_actions.size() - 1);
    return best_actions[dist(rng)];
  }

  std::size_t native_ucb_select_position(
      const std::vector<MctsCppNode>& nodes,
      const MctsCppNode& node,
      double exp_weight,
      const std::vector<double>& prior_logits,
      const std::vector<double>& value_logits,
      double prior_weight,
      double value_weight,
      std::mt19937_64& rng) const {
    if (node.child_ids.empty()) {
      throw std::runtime_error("native MCTS child list is empty");
    }
    std::vector<std::size_t> best_positions;
    double best_score = -std::numeric_limits<double>::infinity();
    const double log_parent =
        node.num_vis == 0 ? 0.0 : std::log(static_cast<double>(node.num_vis));
    for (std::size_t idx = 0; idx < node.child_ids.size(); ++idx) {
      const MctsCppNode& child = nodes[node.child_ids[idx]];
      double score = std::numeric_limits<double>::infinity();
      if (node.num_vis > 0 && child.num_vis > 0) {
        score = child.q +
                exp_weight *
                    std::sqrt(2.0 * log_parent /
                              static_cast<double>(child.num_vis));
      }
      const std::size_t action = node.child_actions[idx];
      if (action < prior_logits.size()) {
        score += prior_weight * prior_logits[action];
      }
      if (action < value_logits.size()) {
        score += value_weight * value_logits[action];
      }
      if (score > best_score) {
        best_score = score;
        best_positions.clear();
        best_positions.push_back(idx);
      } else if (score == best_score) {
        best_positions.push_back(idx);
      }
    }
    std::uniform_int_distribution<std::size_t> dist(0,
                                                    best_positions.size() - 1);
    return best_positions[dist(rng)];
  }

  static void backprop_native(std::vector<MctsCppNode>& nodes,
                              const std::vector<std::size_t>& path,
                              double total_return) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      if (*it >= nodes.size()) {
        continue;
      }
      MctsCppNode& node = nodes[*it];
      if (node.num_vis == 0) {
        node.reward = total_return;
      }
      ++node.num_vis;
      if (total_return > node.q) {
        node.q = total_return;
      }
    }
  }

  static double discounted_rewards(const std::vector<double>& rewards,
                                   double gamma) {
    double total = 0.0;
    double scale = 1.0;
    for (double reward : rewards) {
      total += reward * scale;
      scale *= gamma;
    }
    return total;
  }

  std::size_t choose_mcts_root_action(
      const std::vector<std::size_t>& actions,
      const std::vector<std::size_t>& visits,
      const std::vector<double>& q,
      std::size_t parent_visits,
      double exp_weight,
      const std::vector<double>& prior_logits,
      const std::vector<double>& value_logits,
      double prior_weight,
      double value_weight,
      std::mt19937_64& rng) const {
    std::vector<std::size_t> best_positions;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t idx = 0; idx < actions.size(); ++idx) {
      double score = std::numeric_limits<double>::infinity();
      if (visits[idx] > 0) {
        score = q[idx] +
                exp_weight *
                    std::sqrt(std::log(static_cast<double>(parent_visits) + 1.0) /
                              static_cast<double>(visits[idx]));
      }
      const std::size_t action = actions[idx];
      if (action < prior_logits.size()) {
        score += prior_weight * prior_logits[action];
      }
      if (action < value_logits.size()) {
        score += value_weight * value_logits[action];
      }
      if (score > best_score) {
        best_score = score;
        best_positions.clear();
        best_positions.push_back(idx);
      } else if (score == best_score) {
        best_positions.push_back(idx);
      }
    }
    if (best_positions.empty()) {
      throw std::runtime_error("native MCTS has no candidate actions");
    }
    std::uniform_int_distribution<std::size_t> dist(0,
                                                    best_positions.size() - 1);
    return best_positions[dist(rng)];
  }

  void snapshot_best_from_state() {
    auto copied = state_->copy_bounds_rotations();
    best_bounds_ = std::move(copied.first);
    best_rotations_ = std::move(copied.second);
    best_score_ = state_->last_bbox_score();
  }

  std::vector<std::vector<double>> vertices_;
  std::vector<std::vector<std::size_t>> faces_;
  std::vector<std::vector<std::size_t>> voxels_;
  std::vector<double> tet_volumes_;
  std::vector<std::vector<double>> centroids_;
  std::string category_;
  std::size_t num_action_scale_ = 0;
  double action_unit_ = 0.0;
  double volume_sum_ = 0.0;
  std::size_t num_actions_ = 0;
  std::size_t actions_per_bbox_ = 0;
  bool stateful_union_cache_ = true;
  bool native_recenter_enabled_ = false;
  std::size_t native_recenter_applies_ = 0;
  std::size_t native_recenter_invalid_ = 0;
  std::size_t cache_capacity_ = 65536;
  std::string volume_method_ = "mesh";
  std::unique_ptr<SmartCppManifoldState> state_;
  std::vector<std::size_t> opposite_actions_;
  std::vector<std::vector<double>> best_bounds_;
  std::vector<std::vector<double>> best_rotations_;
  double best_score_ = -std::numeric_limits<double>::infinity();
  std::unordered_map<std::string, double> stats_;
};

std::unique_ptr<NativeSmartEngine> native_smart_engine_from_gmsh_py(
    const std::string& msh_path,
    const std::vector<std::vector<double>>& bounds,
    const std::vector<std::vector<double>>& rotations,
    const std::string& category,
    std::size_t num_action_scale,
    double action_unit,
    double volume_sum,
    double last_bbox_score,
    bool stateful_union_cache,
    std::size_t cache_capacity,
    const std::string& volume_method) {
  std::size_t n_vertices = 0;
  std::size_t n_faces = 0;
  std::size_t n_voxels = 0;
  require_status(smart_native_load_gmsh_counts(
                     msh_path.c_str(), &n_vertices, &n_faces, &n_voxels),
                 "native SMART engine Gmsh count failed");
  std::vector<double> flat_vertices(n_vertices * 3, 0.0);
  std::vector<std::size_t> flat_faces(n_faces * 3, 0);
  std::vector<std::size_t> flat_voxels(n_voxels * 4, 0);
  require_status(smart_native_load_gmsh(
                     msh_path.c_str(), flat_vertices.data(),
                     flat_faces.data(), flat_voxels.data(),
                     flat_vertices.size(), flat_faces.size(),
                     flat_voxels.size(), &n_vertices, &n_faces, &n_voxels),
                 "native SMART engine Gmsh load failed");
  flat_vertices.resize(n_vertices * 3);
  flat_faces.resize(n_faces * 3);
  flat_voxels.resize(n_voxels * 4);
  std::vector<double> tet_volumes(n_voxels, 0.0);
  std::vector<double> flat_centroids(n_voxels * 3, 0.0);
  if (n_voxels > 0) {
    require_status(smart_native_tetra_volumes(
                       flat_vertices.data(), n_vertices, flat_voxels.data(),
                       n_voxels, tet_volumes.data()),
                   "native SMART engine tetra volume calculation failed");
    require_status(smart_native_tetra_centroids(
                       flat_vertices.data(), n_vertices, flat_voxels.data(),
                       n_voxels, flat_centroids.data()),
                   "native SMART engine tetra centroid calculation failed");
  }
  return std::make_unique<NativeSmartEngine>(
      unflatten_double_rows(flat_vertices, 3),
      unflatten_size_t_rows(flat_faces, 3),
      unflatten_size_t_rows(flat_voxels, 4), tet_volumes,
      unflatten_double_rows(flat_centroids, 3), bounds, rotations, category,
      num_action_scale, action_unit, volume_sum, last_bbox_score,
      stateful_union_cache, cache_capacity, volume_method);
}

}  // namespace

PYBIND11_MODULE(_cpp, module) {
  module.doc() = "SMART native C++ core and fixed-Manifold bridge";

  module.def("native_core_available", []() { return true; });
  module.def("manifold_bridge_available", []() { return true; });
  module.def("manifold_cube_volume", &smart_manifold_cube_volume);
  module.def("manifold_mesh_volume", &manifold_mesh_volume_py);
  module.def("manifold_axis_box_intersection_volume",
             &manifold_axis_box_intersection_volume_py);
  module.def("native_action_count", &smart_native_action_count);
  module.def("native_action_scales", &native_action_scales_py);
  module.def("native_action_indices", &native_action_indices_py);
  module.def("native_opposite_actions", &native_opposite_actions_py);
  module.def("native_child_action_mask", &native_child_action_mask_py,
             py::arg("num_actions"), py::arg("action"),
             py::arg("num_action_scale"), py::arg("parent_mask") = py::none());
  module.def("native_discounted_reward", &native_discounted_reward_py);
  module.def("native_ucb_best_count", &native_ucb_best_count_py);
  module.def("native_best_ucb_child", &native_best_ucb_child_py);
  module.def("native_prob_skip_exploration",
             &native_prob_skip_exploration_py);
  module.def("native_softmax_scaled", &native_softmax_scaled_py,
             py::arg("values"), py::arg("scale") = 100.0);
  module.def("native_weighted_action_scores",
             &native_weighted_action_scores_py,
             py::arg("base_rewards"), py::arg("prior_logits"),
             py::arg("value_logits"), py::arg("base_scale"),
             py::arg("prior_weight"), py::arg("value_weight"));
  module.def("native_top_k_actions", &native_top_k_actions_py,
             py::arg("actions"), py::arg("scores"), py::arg("top_k"));
  module.def("native_best_score_action", &native_best_score_action_py,
             py::arg("actions"), py::arg("scores"), py::arg("tie_pick"));
  module.def("native_diverse_escape_actions",
             &native_diverse_escape_actions_py,
             py::arg("actions"), py::arg("scores"),
             py::arg("primary_keep"), py::arg("num_action_scale"),
             py::arg("escape_top_k"));
  module.def("native_add_puct_prior", &native_add_puct_prior_py,
             py::arg("uct_scores"), py::arg("prior_logits"),
             py::arg("child_visits"), py::arg("parent_visits"),
             py::arg("prior_weight"));
  module.def("native_action_mlp_logits_values",
             &native_action_mlp_logits_values_py,
             py::arg("actions"), py::arg("action_num_action_scale"),
             py::arg("model_num_action_scale"), py::arg("context"),
             py::arg("categories"), py::arg("action_input_weights"),
             py::arg("action_hidden_bias"), py::arg("action_output_weights"),
             py::arg("action_output_bias"),
             py::arg("action_value_output_weights") = std::vector<double>{},
             py::arg("action_value_output_bias") = 0.0);
  module.def("opposite_action", &opposite_action_py);
  module.def("opposite_action_mask", &opposite_action_mask_py);
  module.def("untried_actions", &untried_actions_py);
  module.def("single_untried_action_mask", &single_untried_action_mask_py);
  module.def("ucb_scores", &ucb_scores_py);
  module.def("ucb_best_indices", &ucb_best_indices_py);
  module.def("native_bbox_volumes", &native_bbox_volumes_py);
  module.def("native_bbox_valid_mask", &native_bbox_valid_mask_py);
  module.def("native_total_bbox_volume", &native_total_bbox_volume_py);
  module.def("native_bbox_union_bounds", &native_bbox_union_bounds_py);
  module.def("native_bbox_union_volume", &native_bbox_union_volume_py);
  module.def("native_box_mesh", &native_box_mesh_py, py::arg("x"),
             py::arg("y"), py::arg("z"), py::arg("lx"), py::arg("ly"),
             py::arg("lz"), py::arg("rotation"));
  module.def("native_coverage_mask", &native_coverage_mask_py);
  module.def("native_recenter_points_for_box",
             &native_recenter_points_for_box_py,
             py::arg("vertices"), py::arg("voxels"), py::arg("centroids"),
             py::arg("bounds"), py::arg("rotation"));
  module.def("native_apply_axis_action", &native_apply_axis_action_py);
  module.def("native_action_upper_rewards", &native_action_upper_rewards_py);
  module.def("native_bbox_action_upper_rewards",
             &native_bbox_action_upper_rewards_py);
  module.def("native_bavf_scores", &native_bavf_scores_py,
             py::arg("part_volumes"), py::arg("bbox_volumes"),
             py::arg("alpha") = 100.0);
  module.def("native_merge_bavf_reward", &native_merge_bavf_reward_py);
  module.def("native_incremental_average",
             &smart_native_incremental_average);
  module.def("native_normalize_vertices_raw",
             &native_normalize_vertices_raw_py);
  module.def("native_normalize_obj_file",
             &native_normalize_obj_file_py,
             py::arg("input_path"), py::arg("output_path"), py::arg("mode"),
             py::arg("center"), py::arg("target"));
  module.def("native_load_obj_mesh", &native_load_obj_mesh_py,
             py::arg("input_path"));
  module.def("native_save_obj_mesh", &native_save_obj_mesh_py,
             py::arg("output_path"), py::arg("vertices"), py::arg("faces"));
  module.def("native_symmetric_chamfer", &native_symmetric_chamfer_py);
  module.def("native_tetra_volumes", &native_tetra_volumes_py);
  module.def("native_tetra_centroids", &native_tetra_centroids_py);
  module.def("native_tetra_surface_faces", &native_tetra_surface_faces_py);
  module.def("native_tetra_adjacency", &native_tetra_adjacency_py);
  module.def("native_load_gmsh", &native_load_gmsh_py);
  module.def("native_save_gmsh", &native_save_gmsh_py);
  module.def("native_centroid_proxy_axis_rewards",
             &native_centroid_proxy_axis_rewards_py);
  module.def("native_partition_summaries", &native_partition_summaries_py,
             py::arg("vertices"), py::arg("voxels"), py::arg("volumes"),
             py::arg("partitions"), py::arg("unique_points") = false);
  module.def("tet_clipping_metrics", &tet_clipping_metrics_py,
             py::arg("vertices"), py::arg("voxels"), py::arg("box_vertices"),
             py::arg("surface_volume"), py::arg("max_boxes"),
             py::arg("box_volumes") = py::none());
  module.def("bbox_rot_state_key", &bbox_rot_state_key_py);
  module.def("run_mcts_callbacks", &run_mcts_callbacks_py,
             py::arg("args"), py::arg("env"), py::arg("num_iter"),
             py::arg("action_prior_logits") = std::vector<double>{},
             py::arg("action_value_logits") = std::vector<double>{});
  module.def("run_greedy_refine_callbacks",
             &run_greedy_refine_callbacks_py);

  py::class_<SmartCppActionMlpPolicy>(module, "ActionMlpPolicy")
      .def(py::init<std::vector<std::string>,
                    std::vector<std::vector<double>>,
                    std::vector<double>, std::vector<double>, double,
                    std::vector<double>, double>(),
           py::arg("categories"), py::arg("action_input_weights"),
           py::arg("action_hidden_bias"), py::arg("action_output_weights"),
           py::arg("action_output_bias"),
           py::arg("action_value_output_weights") = std::vector<double>{},
           py::arg("action_value_output_bias") = 0.0)
      .def("logits_values", &SmartCppActionMlpPolicy::logits_values,
           py::arg("actions"), py::arg("action_num_action_scale"),
           py::arg("model_num_action_scale"), py::arg("context"))
      .def("feature_dim", &SmartCppActionMlpPolicy::feature_dim)
      .def("hidden_size", &SmartCppActionMlpPolicy::hidden_size)
      .def("has_value_head", &SmartCppActionMlpPolicy::has_value_head);

  py::class_<SmartCppBBoxState>(module, "BBoxState")
      .def(py::init<const std::vector<std::vector<double>>&, std::size_t,
                    double, double, double>(),
           py::arg("bounds"), py::arg("num_action_scale"),
           py::arg("action_unit"), py::arg("volume_sum"),
           py::arg("last_bbox_score"))
      .def("num_bbox", &SmartCppBBoxState::num_bbox)
      .def("num_actions", &SmartCppBBoxState::num_actions)
      .def("bounds", &SmartCppBBoxState::bounds)
      .def("volumes", &SmartCppBBoxState::volumes)
      .def("total_volume", &SmartCppBBoxState::total_volume)
      .def("bvs", &SmartCppBBoxState::bvs)
      .def("valid_mask", &SmartCppBBoxState::valid_mask)
      .def("valid_count", &SmartCppBBoxState::valid_count)
      .def("last_bbox_score", &SmartCppBBoxState::last_bbox_score)
      .def("set_last_bbox_score", &SmartCppBBoxState::set_last_bbox_score)
      .def("with_last_bbox_score", &SmartCppBBoxState::with_last_bbox_score)
      .def("state_key", &SmartCppBBoxState::state_key)
      .def("action_upper_rewards", &SmartCppBBoxState::action_upper_rewards)
      .def("bbox_action_upper_rewards",
           &SmartCppBBoxState::bbox_action_upper_rewards)
      .def("apply_axis_action", &SmartCppBBoxState::apply_axis_action)
      .def("after_axis_action", &SmartCppBBoxState::after_axis_action)
      .def("apply_axis_action_in_place",
           &SmartCppBBoxState::apply_axis_action_in_place);

  py::class_<SmartCppManifoldBridgeMesh>(module, "ManifoldBridgeMesh")
      .def(py::init<const std::vector<std::vector<double>>&,
                    const std::vector<std::vector<std::size_t>>&>(),
           py::arg("vertices"), py::arg("faces"))
      .def("volume", &SmartCppManifoldBridgeMesh::volume)
      .def("volume_properties", &SmartCppManifoldBridgeMesh::volume_properties)
      .def("residual_volume_for_boxes",
           &SmartCppManifoldBridgeMesh::residual_volume_for_boxes)
      .def("residual_volume_for_boxes_properties",
           &SmartCppManifoldBridgeMesh::residual_volume_for_boxes_properties)
      .def("residual_volume_for_boxes_pair",
           &SmartCppManifoldBridgeMesh::residual_volume_for_boxes_pair)
      .def("residual_volume_for_box_params",
           &SmartCppManifoldBridgeMesh::residual_volume_for_box_params)
      .def("residual_volume_for_box_params_properties",
           &SmartCppManifoldBridgeMesh::residual_volume_for_box_params_properties)
      .def("residual_volume_for_box_params_pair",
           &SmartCppManifoldBridgeMesh::residual_volume_for_box_params_pair)
      .def("covered_for_bounds", &SmartCppManifoldBridgeMesh::covered_for_bounds,
           py::arg("bounds"), py::arg("rotations"), py::arg("volume_sum"),
           py::arg("volume_method") = "mesh")
      .def("best_axis_action", &SmartCppManifoldBridgeMesh::best_axis_action,
           py::arg("bounds"), py::arg("rotations"), py::arg("bbox_idx"),
           py::arg("num_action_scale"), py::arg("action_unit"),
           py::arg("volume_sum"), py::arg("last_bbox_score"),
           py::arg("cover_penalty"), py::arg("pen_rate"),
           py::arg("initial_best"), py::arg("volume_method") = "mesh")
      .def("best_axis_actions_for_mask",
           &SmartCppManifoldBridgeMesh::best_axis_actions_for_mask,
           py::arg("bounds"), py::arg("rotations"), py::arg("bbox_mask"),
           py::arg("num_action_scale"), py::arg("action_unit"),
           py::arg("volume_sum"), py::arg("last_bbox_score"),
           py::arg("cover_penalty"), py::arg("pen_rate"),
           py::arg("initial_best"), py::arg("volume_method") = "mesh")
      .def("greedy_axis_refine_segment",
           &SmartCppManifoldBridgeMesh::greedy_axis_refine_segment,
           py::arg("bounds"), py::arg("rotations"),
           py::arg("num_action_scale"), py::arg("action_unit"),
           py::arg("volume_sum"), py::arg("last_bbox_score"),
           py::arg("cover_penalty"), py::arg("pen_rate"),
           py::arg("max_steps"));

  py::class_<SmartCppCandidateBitsetState>(module, "CandidateBitsetState")
      .def(py::init<const std::vector<std::vector<double>>&,
                    const std::vector<double>&, double>(),
           py::arg("centroids"), py::arg("volumes"), py::arg("volume_sum"))
      .def("num_centroids", &SmartCppCandidateBitsetState::num_centroids)
      .def("volume_sum", &SmartCppCandidateBitsetState::volume_sum)
      .def("axis_rewards", &SmartCppCandidateBitsetState::axis_rewards,
           py::arg("bounds"), py::arg("rotations"),
           py::arg("num_action_scale"), py::arg("action_unit"),
           py::arg("last_bbox_score"), py::arg("cover_penalty"),
           py::arg("pen_rate"))
      .def("topk_axis_actions",
           &SmartCppCandidateBitsetState::topk_axis_actions,
           py::arg("bounds"), py::arg("rotations"),
           py::arg("num_action_scale"), py::arg("action_unit"),
           py::arg("last_bbox_score"), py::arg("cover_penalty"),
           py::arg("pen_rate"), py::arg("bbox_idx"), py::arg("top_k"));

  py::class_<TetClippingStateCpp>(module, "TetClippingState")
      .def(py::init<const std::vector<std::vector<double>>&,
                    const std::vector<std::vector<std::size_t>>&, double>(),
           py::arg("vertices"), py::arg("voxels"),
           py::arg("surface_volume"))
      .def("metrics", &TetClippingStateCpp::metrics,
           py::arg("box_vertices"), py::arg("max_boxes"),
           py::arg("box_volumes") = py::none())
      .def("metrics_for_boxes", &TetClippingStateCpp::metrics_for_boxes,
           py::arg("bounds"), py::arg("rotations"), py::arg("max_boxes"))
      .def("covered_for_boxes", &TetClippingStateCpp::covered_for_boxes,
           py::arg("bounds"), py::arg("rotations"), py::arg("max_boxes"));

  py::class_<NativeSmartEngine>(module, "NativeSmartEngine")
      .def(py::init<const std::vector<std::vector<double>>&,
                    const std::vector<std::vector<std::size_t>>&,
                    const std::vector<std::vector<std::size_t>>&,
                    const std::vector<double>&,
                    const std::vector<std::vector<double>>&,
                    const std::vector<std::vector<double>>&,
                    const std::vector<std::vector<double>>&,
                    const std::string&, std::size_t, double, double, double,
                    bool, std::size_t, const std::string&>(),
           py::arg("vertices"), py::arg("faces"), py::arg("voxels"),
           py::arg("tet_volumes"), py::arg("centroids"), py::arg("bounds"),
           py::arg("rotations"), py::arg("category") = "",
           py::arg("num_action_scale") = 2, py::arg("action_unit") = 0.01,
           py::arg("volume_sum") = 0.0, py::arg("last_bbox_score") = 0.0,
           py::arg("stateful_union_cache") = true,
           py::arg("cache_capacity") = 65536,
           py::arg("volume_method") = "mesh")
      .def("boxes", &NativeSmartEngine::boxes)
      .def("best_boxes", &NativeSmartEngine::best_boxes)
      .def("stats", &NativeSmartEngine::stats)
      .def("recompute_score", &NativeSmartEngine::recompute_score,
           py::arg("cover_penalty"), py::arg("pen_rate") = 1.0)
      .def("run_refine", &NativeSmartEngine::run_refine,
           py::arg("max_steps"), py::arg("cover_penalty"),
           py::arg("pen_rate") = 1.0)
      .def("run_mcts", &NativeSmartEngine::run_mcts,
           py::arg("num_iter"), py::arg("max_step"),
           py::arg("cover_penalty"), py::arg("pen_rate") = 1.0,
           py::arg("exp_weight") = 0.001, py::arg("gamma") = 1.0,
           py::arg("seed") = 7777,
           py::arg("action_prior_logits") = std::vector<double>{},
           py::arg("action_value_logits") = std::vector<double>{},
           py::arg("prior_weight") = 0.0,
           py::arg("value_weight") = 0.0,
           py::arg("transposition_table") = false,
           py::arg("transposition_table_size") = 8192,
           py::arg("action_prior_top_k") = 0)
      .def("run_refine_then_mcts", &NativeSmartEngine::run_refine_then_mcts,
           py::arg("refine_max_steps"), py::arg("mcts_iter"),
           py::arg("mcts_max_step"), py::arg("cover_penalty"),
           py::arg("pen_rate") = 1.0, py::arg("exp_weight") = 0.001,
           py::arg("gamma") = 1.0, py::arg("seed") = 7777,
           py::arg("action_prior_logits") = std::vector<double>{},
           py::arg("action_value_logits") = std::vector<double>{},
           py::arg("prior_weight") = 0.0,
           py::arg("value_weight") = 0.0,
           py::arg("transposition_table") = false,
           py::arg("transposition_table_size") = 8192,
           py::arg("action_prior_top_k") = 0)
      .def("run_merge", &NativeSmartEngine::run_merge,
           py::arg("adjacency_pairs"), py::arg("merge_eps") = 0.0,
           py::arg("shape_volume") = 0.0, py::arg("final_k") = 0)
      .def("run_partition_merge", &NativeSmartEngine::run_partition_merge,
           py::arg("partitions"), py::arg("adjacency_pairs"),
           py::arg("merge_eps") = 0.0, py::arg("shape_volume") = 0.0,
           py::arg("final_k") = 0, py::arg("tilted") = false)
      .def("partition_adjacency_pairs",
           &NativeSmartEngine::partition_adjacency_pairs,
           py::arg("partitions"), py::arg("only_nearby") = true)
      .def("run_partition_merge_auto_adjacency",
           &NativeSmartEngine::run_partition_merge_auto_adjacency,
           py::arg("partitions"), py::arg("only_nearby") = true,
           py::arg("merge_eps") = 0.0, py::arg("shape_volume") = 0.0,
           py::arg("final_k") = 0, py::arg("tilted") = false)
      .def("export_obj", &NativeSmartEngine::export_obj,
           py::arg("path"))
      .def("export_bbox_dir", &NativeSmartEngine::export_bbox_dir,
           py::arg("directory"));
  module.def("native_smart_engine_from_gmsh",
             &native_smart_engine_from_gmsh_py,
             py::arg("msh_path"), py::arg("bounds"), py::arg("rotations"),
             py::arg("category") = "", py::arg("num_action_scale") = 2,
             py::arg("action_unit") = 0.01, py::arg("volume_sum") = 0.0,
             py::arg("last_bbox_score") = 0.0,
             py::arg("stateful_union_cache") = true,
             py::arg("cache_capacity") = 65536,
             py::arg("volume_method") = "mesh");

  py::class_<SmartCppManifoldState>(module, "ManifoldState")
      .def(py::init<const std::vector<std::vector<double>>&,
                    const std::vector<std::vector<std::size_t>>&,
                    const std::vector<std::vector<double>>&,
                    const std::vector<std::vector<double>>&, std::size_t,
                    double, double, double, bool, std::size_t,
                    const std::string&>(),
           py::arg("vertices"), py::arg("faces"), py::arg("bounds"),
           py::arg("rotations"), py::arg("num_action_scale"),
           py::arg("action_unit"), py::arg("volume_sum"),
           py::arg("last_bbox_score"),
           py::arg("stateful_union_cache") = true,
           py::arg("cache_capacity") = 65536,
           py::arg("volume_method") = "mesh")
      .def("num_boxes", &SmartCppManifoldState::num_boxes)
      .def("reset_to_state", &SmartCppManifoldState::reset_to_state)
      .def("reset_to_initial", &SmartCppManifoldState::reset_to_initial)
      .def("bounds", &SmartCppManifoldState::bounds)
      .def("rotations", &SmartCppManifoldState::rotations)
      .def("state", &SmartCppManifoldState::state)
      .def("bbox_params", &SmartCppManifoldState::bbox_params)
      .def("last_bbox_score", &SmartCppManifoldState::last_bbox_score)
      .def("total_volume", &SmartCppManifoldState::total_volume)
      .def("bvs", &SmartCppManifoldState::bvs)
      .def("valid_count", &SmartCppManifoldState::valid_count)
      .def("cache_stats", &SmartCppManifoldState::cache_stats)
      .def("state_hash", &SmartCppManifoldState::state_hash)
      .def("state_key", &SmartCppManifoldState::state_key)
      .def("covered", &SmartCppManifoldState::covered)
      .def("score", &SmartCppManifoldState::score,
           py::arg("cover_penalty"), py::arg("pen_rate"))
      .def("score_axis_action", &SmartCppManifoldState::score_axis_action,
           py::arg("action"), py::arg("cover_penalty"), py::arg("pen_rate"))
      .def("score_axis_action_reward",
           &SmartCppManifoldState::score_axis_action_reward,
           py::arg("action"), py::arg("cover_penalty"), py::arg("pen_rate"))
      .def("score_replacement", &SmartCppManifoldState::score_replacement,
           py::arg("bbox_idx"), py::arg("candidate_bounds"),
           py::arg("candidate_rotation"), py::arg("cover_penalty"),
           py::arg("pen_rate"))
      .def("apply_replacement_delta",
           &SmartCppManifoldState::apply_replacement_delta,
           py::arg("bbox_idx"), py::arg("candidate_bounds"),
           py::arg("candidate_rotation"), py::arg("cover_penalty"),
           py::arg("pen_rate"))
      .def("score_action_batch", &SmartCppManifoldState::score_action_batch,
           py::arg("bbox_mask"), py::arg("cover_penalty"),
           py::arg("pen_rate"), py::arg("initial_best"))
      .def("select_replacement_batch",
           &SmartCppManifoldState::select_replacement_batch,
           py::arg("bbox_mask"), py::arg("candidate_bounds"),
           py::arg("candidate_rotations"), py::arg("actions"),
           py::arg("rewards"), py::arg("cover_penalty"), py::arg("pen_rate"))
      .def("apply_axis_action", &SmartCppManifoldState::apply_axis_action,
           py::arg("action"), py::arg("cover_penalty"), py::arg("pen_rate"))
      .def("apply_axis_action_delta",
           &SmartCppManifoldState::apply_axis_action_delta,
           py::arg("action"), py::arg("cover_penalty"), py::arg("pen_rate"))
      .def("greedy_axis_rollout_step",
           &SmartCppManifoldState::greedy_axis_rollout_step,
           py::arg("bbox_mask"), py::arg("cover_penalty"),
           py::arg("pen_rate"))
      .def("greedy_axis_rollout_segment",
           &SmartCppManifoldState::greedy_axis_rollout_segment,
           py::arg("bbox_mask"), py::arg("cover_penalty"),
           py::arg("pen_rate"), py::arg("max_steps"))
      .def("greedy_axis_rollout_segment_delta",
           &SmartCppManifoldState::greedy_axis_rollout_segment_delta,
           py::arg("bbox_mask"), py::arg("cover_penalty"),
           py::arg("pen_rate"), py::arg("max_steps"))
      .def("rollback", &SmartCppManifoldState::rollback)
      .def("greedy_axis_refine_segment",
           &SmartCppManifoldState::greedy_axis_refine_segment,
           py::arg("cover_penalty"), py::arg("pen_rate"),
           py::arg("max_steps"))
      .def("greedy_axis_refine_segment_delta",
           &SmartCppManifoldState::greedy_axis_refine_segment_delta,
           py::arg("cover_penalty"), py::arg("pen_rate"),
           py::arg("max_steps"));
}
