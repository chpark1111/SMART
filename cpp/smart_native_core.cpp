#include "smart_native_core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {
double action_scale_at(std::size_t scale_idx, std::size_t num_action_scale) {
  const std::size_t half = num_action_scale / 2;
  if (scale_idx < half) {
    return -std::pow(2.0, static_cast<double>(half - 1 - scale_idx));
  }
  return std::pow(2.0, static_cast<double>(scale_idx - half));
}

bool valid_action_scale_count(std::size_t num_action_scale) {
  return num_action_scale > 0 && num_action_scale % 2 == 0;
}

double bbox_volume_at(const double* bounds, std::size_t idx) {
  const double* row = bounds + idx * 6;
  const double dx = std::max(0.0, row[3] - row[0]);
  const double dy = std::max(0.0, row[4] - row[1]);
  const double dz = std::max(0.0, row[5] - row[2]);
  return dx * dy * dz;
}

bool bbox_valid_at(const double* bounds, std::size_t idx) {
  const double* row = bounds + idx * 6;
  return row[0] < row[3] && row[1] < row[4] && row[2] < row[5];
}

bool point_in_oriented_bounds(const double* point,
                              const double* bounds,
                              const double* rotation) {
  const double x = point[0] * rotation[0] + point[1] * rotation[1] +
                   point[2] * rotation[2];
  const double y = point[0] * rotation[3] + point[1] * rotation[4] +
                   point[2] * rotation[5];
  const double z = point[0] * rotation[6] + point[1] * rotation[7] +
                   point[2] * rotation[8];
  return bounds[0] <= x && x <= bounds[3] && bounds[1] <= y &&
         y <= bounds[4] && bounds[2] <= z && z <= bounds[5];
}

double distance3(const double* point, const double* center) {
  const double dx = point[0] - center[0];
  const double dy = point[1] - center[1];
  const double dz = point[2] - center[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double dot3(const double* left, const double* right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

void cross3(const double* left, const double* right, double* out) {
  out[0] = left[1] * right[2] - left[2] * right[1];
  out[1] = left[2] * right[0] - left[0] * right[2];
  out[2] = left[0] * right[1] - left[1] * right[0];
}

bool voxel_indices_valid(const std::size_t* voxel, std::size_t n_vertices) {
  return voxel[0] < n_vertices && voxel[1] < n_vertices &&
         voxel[2] < n_vertices && voxel[3] < n_vertices;
}

std::array<std::size_t, 3> sorted_face(std::array<std::size_t, 3> face) {
  std::sort(face.begin(), face.end());
  return face;
}

std::array<std::array<std::size_t, 3>, 4> tet_faces(const std::size_t* voxel) {
  return {{
      {voxel[0], voxel[1], voxel[2]},
      {voxel[0], voxel[1], voxel[3]},
      {voxel[0], voxel[2], voxel[3]},
      {voxel[1], voxel[2], voxel[3]},
  }};
}

std::array<std::array<std::size_t, 3>, 4> tet_faces(
    const std::array<std::size_t, 4>& voxel) {
  return {{
      {voxel[0], voxel[1], voxel[2]},
      {voxel[0], voxel[1], voxel[3]},
      {voxel[0], voxel[2], voxel[3]},
      {voxel[1], voxel[2], voxel[3]},
  }};
}

struct ParsedGmsh {
  std::vector<std::array<double, 3>> vertices;
  std::vector<std::array<std::size_t, 3>> faces;
  std::vector<std::array<std::size_t, 4>> voxels;
};

bool parse_first_size(const std::string& line, std::size_t* out) {
  if (out == nullptr) {
    return false;
  }
  std::istringstream stream(line);
  stream >> *out;
  return !stream.fail();
}

std::vector<std::array<std::size_t, 3>> surface_faces_from_voxels(
    const std::vector<std::array<std::size_t, 4>>& voxels) {
  std::map<std::array<std::size_t, 3>, std::size_t> key_to_index;
  std::vector<std::pair<std::array<std::size_t, 3>, bool>> ordered_faces;
  ordered_faces.reserve(voxels.size() * 4);
  for (const auto& voxel : voxels) {
    for (const auto& face : tet_faces(voxel)) {
      const auto key = sorted_face(face);
      auto iter = key_to_index.find(key);
      if (iter == key_to_index.end()) {
        key_to_index.emplace(key, ordered_faces.size());
        ordered_faces.emplace_back(face, true);
      } else {
        ordered_faces[iter->second].second = false;
      }
    }
  }
  std::vector<std::array<std::size_t, 3>> faces;
  faces.reserve(ordered_faces.size());
  for (const auto& entry : ordered_faces) {
    if (entry.second) {
      faces.push_back(entry.first);
    }
  }
  return faces;
}

bool parse_gmsh_file(const char* path, ParsedGmsh* out) {
  if (path == nullptr || out == nullptr) {
    return false;
  }
  std::ifstream input(path);
  if (!input) {
    return false;
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(input, line)) {
    lines.push_back(line);
  }

  std::size_t nodes_marker = lines.size();
  std::size_t elements_marker = lines.size();
  for (std::size_t idx = 0; idx < lines.size(); ++idx) {
    if (lines[idx] == "$Nodes") {
      nodes_marker = idx;
    } else if (lines[idx] == "$Elements") {
      elements_marker = idx;
    }
  }
  if (nodes_marker >= lines.size() || elements_marker >= lines.size()) {
    return false;
  }

  const std::size_t nodes_start = nodes_marker + 1;
  if (nodes_start >= lines.size()) {
    return false;
  }
  std::size_t node_count = 0;
  if (!parse_first_size(lines[nodes_start], &node_count)) {
    return false;
  }
  std::map<long long, std::size_t> node_id_to_index;
  out->vertices.clear();
  out->vertices.reserve(node_count);
  for (std::size_t offset = 0; offset < node_count; ++offset) {
    const std::size_t line_idx = nodes_start + 1 + offset;
    if (line_idx >= lines.size()) {
      return false;
    }
    std::istringstream stream(lines[line_idx]);
    long long node_id = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    stream >> node_id >> x >> y >> z;
    if (stream.fail()) {
      return false;
    }
    node_id_to_index[node_id] = offset;
    out->vertices.push_back({x, y, z});
  }

  const std::size_t elements_start = elements_marker + 1;
  if (elements_start >= lines.size()) {
    return false;
  }
  std::size_t element_count = 0;
  if (!parse_first_size(lines[elements_start], &element_count)) {
    return false;
  }
  out->faces.clear();
  out->voxels.clear();
  for (std::size_t offset = 0; offset < element_count; ++offset) {
    const std::size_t line_idx = elements_start + 1 + offset;
    if (line_idx >= lines.size()) {
      return false;
    }
    std::istringstream stream(lines[line_idx]);
    long long element_id = 0;
    long long element_type = 0;
    std::size_t tag_count = 0;
    stream >> element_id >> element_type >> tag_count;
    if (stream.fail()) {
      return false;
    }
    for (std::size_t tag_idx = 0; tag_idx < tag_count; ++tag_idx) {
      long long ignored_tag = 0;
      stream >> ignored_tag;
      if (stream.fail()) {
        return false;
      }
    }
    std::vector<std::size_t> ids;
    long long node_id = 0;
    while (stream >> node_id) {
      auto iter = node_id_to_index.find(node_id);
      if (iter == node_id_to_index.end()) {
        return false;
      }
      ids.push_back(iter->second);
    }
    if (element_type == 2 && ids.size() >= 3) {
      out->faces.push_back({ids[0], ids[1], ids[2]});
    } else if (element_type == 4 && ids.size() >= 4) {
      out->voxels.push_back({ids[0], ids[1], ids[2], ids[3]});
    }
  }
  if (out->faces.empty() && !out->voxels.empty()) {
    out->faces = surface_faces_from_voxels(out->voxels);
  }
  return true;
}

bool count_gmsh_file(const char* path,
                     std::size_t* out_n_vertices,
                     std::size_t* out_n_faces,
                     std::size_t* out_n_voxels) {
  if (path == nullptr || out_n_vertices == nullptr || out_n_faces == nullptr ||
      out_n_voxels == nullptr) {
    return false;
  }
  std::ifstream input(path);
  if (!input) {
    return false;
  }

  std::string line;
  bool saw_nodes = false;
  bool saw_elements = false;
  *out_n_vertices = 0;
  *out_n_faces = 0;
  *out_n_voxels = 0;

  while (std::getline(input, line)) {
    if (line == "$Nodes") {
      if (!std::getline(input, line) ||
          !parse_first_size(line, out_n_vertices)) {
        return false;
      }
      saw_nodes = true;
      for (std::size_t idx = 0; idx < *out_n_vertices; ++idx) {
        if (!std::getline(input, line)) {
          return false;
        }
      }
      continue;
    }
    if (line == "$Elements") {
      std::size_t element_count = 0;
      if (!std::getline(input, line) ||
          !parse_first_size(line, &element_count)) {
        return false;
      }
      saw_elements = true;
      for (std::size_t idx = 0; idx < element_count; ++idx) {
        if (!std::getline(input, line)) {
          return false;
        }
        std::istringstream stream(line);
        long long element_id = 0;
        long long element_type = 0;
        std::size_t tag_count = 0;
        stream >> element_id >> element_type >> tag_count;
        if (stream.fail()) {
          return false;
        }
        if (element_type == 2) {
          *out_n_faces += 1;
        } else if (element_type == 4) {
          *out_n_voxels += 1;
        }
      }
    }
  }

  if (!saw_nodes || !saw_elements) {
    return false;
  }
  if (*out_n_faces == 0 && *out_n_voxels > 0) {
    *out_n_faces = *out_n_voxels * 4;
  }
  return true;
}

std::uint64_t double_bits(double value) {
  std::uint64_t out = 0;
  std::memcpy(&out, &value, sizeof(double));
  return out;
}

void fill_vertex_stats(const double* vertices,
                       std::size_t n_vertices,
                       double* out_stats) {
  double min_v[3] = {vertices[0], vertices[1], vertices[2]};
  double max_v[3] = {vertices[0], vertices[1], vertices[2]};
  for (std::size_t idx = 1; idx < n_vertices; ++idx) {
    const double* point = vertices + idx * 3;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      min_v[axis] = std::min(min_v[axis], point[axis]);
      max_v[axis] = std::max(max_v[axis], point[axis]);
    }
  }
  double extent[3] = {
      max_v[0] - min_v[0],
      max_v[1] - min_v[1],
      max_v[2] - min_v[2],
  };
  double bbox_center[3] = {
      (min_v[0] + max_v[0]) / 2.0,
      (min_v[1] + max_v[1]) / 2.0,
      (min_v[2] + max_v[2]) / 2.0,
  };
  double sphere_radius = 0.0;
  for (std::size_t idx = 0; idx < n_vertices; ++idx) {
    sphere_radius = std::max(
        sphere_radius,
        distance3(vertices + idx * 3, bbox_center));
  }

  out_stats[0] = static_cast<double>(n_vertices);
  out_stats[1] = min_v[0];
  out_stats[2] = min_v[1];
  out_stats[3] = min_v[2];
  out_stats[4] = max_v[0];
  out_stats[5] = max_v[1];
  out_stats[6] = max_v[2];
  out_stats[7] = extent[0];
  out_stats[8] = extent[1];
  out_stats[9] = extent[2];
  out_stats[10] = std::sqrt(extent[0] * extent[0] + extent[1] * extent[1] +
                            extent[2] * extent[2]);
  out_stats[11] = bbox_center[0];
  out_stats[12] = bbox_center[1];
  out_stats[13] = bbox_center[2];
  out_stats[14] = sphere_radius;
}
}  // namespace

extern "C" std::size_t smart_native_action_count(std::size_t num_bbox,
                                                  std::size_t num_action_scale) {
  return num_bbox * (6 * num_action_scale + 1);
}

extern "C" int smart_native_action_scales(std::size_t num_action_scale,
                                           double* out_scales) {
  if (out_scales == nullptr || !valid_action_scale_count(num_action_scale)) {
    return 0;
  }
  for (std::size_t idx = 0; idx < num_action_scale; ++idx) {
    out_scales[idx] = action_scale_at(idx, num_action_scale);
  }
  return 1;
}

extern "C" int smart_native_action_indices(std::size_t num_bbox,
                                            std::size_t num_action_scale,
                                            std::size_t* out_triplets) {
  if (out_triplets == nullptr || !valid_action_scale_count(num_action_scale)) {
    return 0;
  }
  std::size_t write_idx = 0;
  for (std::size_t bbox_idx = 0; bbox_idx < num_bbox; ++bbox_idx) {
    for (std::size_t coord_idx = 0; coord_idx < 6; ++coord_idx) {
      for (std::size_t scale_idx = 0; scale_idx < num_action_scale; ++scale_idx) {
        out_triplets[write_idx++] = bbox_idx;
        out_triplets[write_idx++] = coord_idx;
        out_triplets[write_idx++] = scale_idx;
      }
    }
    out_triplets[write_idx++] = bbox_idx;
    out_triplets[write_idx++] = 6;
    out_triplets[write_idx++] = 0;
  }
  return 1;
}

extern "C" int smart_native_opposite_actions(std::size_t num_bbox,
                                              std::size_t num_action_scale,
                                              std::size_t* out_actions) {
  if (out_actions == nullptr || !valid_action_scale_count(num_action_scale)) {
    return 0;
  }
  const std::size_t actions_per_bbox = 6 * num_action_scale + 1;
  const std::size_t num_actions =
      smart_native_action_count(num_bbox, num_action_scale);
  for (std::size_t action = 0; action < num_actions; ++action) {
    const std::size_t bbox_idx = action / actions_per_bbox;
    const std::size_t local = action % actions_per_bbox;
    if (local == actions_per_bbox - 1) {
      out_actions[action] = action;
      continue;
    }
    const std::size_t coord_idx = local / num_action_scale;
    const std::size_t scale_idx = local % num_action_scale;
    const std::size_t opposite_scale = num_action_scale - 1 - scale_idx;
    out_actions[action] =
        bbox_idx * actions_per_bbox + coord_idx * num_action_scale +
        opposite_scale;
  }
  return 1;
}

extern "C" int smart_native_child_action_mask(std::size_t num_actions,
                                               std::size_t action,
                                               std::size_t num_action_scale,
                                               const std::uint8_t* parent_mask,
                                               std::uint8_t* out_mask) {
  if (out_mask == nullptr || !valid_action_scale_count(num_action_scale) ||
      action >= num_actions) {
    return 0;
  }
  if (parent_mask == nullptr) {
    std::fill(out_mask, out_mask + num_actions, 0);
  } else {
    std::copy(parent_mask, parent_mask + num_actions, out_mask);
  }

  const std::size_t actions_per_bbox = 6 * num_action_scale + 1;
  const std::size_t bbox_idx = action / actions_per_bbox;
  const std::size_t local = action % actions_per_bbox;
  std::size_t opposite = action;
  if (local != actions_per_bbox - 1) {
    const std::size_t coord_idx = local / num_action_scale;
    const std::size_t scale_idx = local % num_action_scale;
    const std::size_t opposite_scale = num_action_scale - 1 - scale_idx;
    opposite = bbox_idx * actions_per_bbox + coord_idx * num_action_scale +
               opposite_scale;
  }
  if (opposite >= num_actions) {
    return 0;
  }
  out_mask[opposite] = 1;
  return 1;
}

extern "C" double smart_native_discounted_reward(const double* rewards,
                                                  std::size_t n_rewards,
                                                  double gamma) {
  if (rewards == nullptr && n_rewards > 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double total = 0.0;
  double scale = 1.0;
  for (std::size_t idx = 0; idx < n_rewards; ++idx) {
    total += rewards[idx] * scale;
    scale *= gamma;
  }
  return total;
}

extern "C" int smart_native_best_ucb_child(std::size_t parent_visits,
                                            const double* child_qs,
                                            const std::size_t* child_visits,
                                            std::size_t n_children,
                                            double exp_weight,
                                            std::size_t tie_pick,
                                            std::size_t* out_position) {
  if (out_position == nullptr || child_qs == nullptr ||
      child_visits == nullptr || n_children == 0) {
    return 0;
  }

  std::vector<std::size_t> best_positions;
  best_positions.reserve(n_children);
  double best_score = -std::numeric_limits<double>::infinity();
  const double log_parent =
      parent_visits == 0 ? 0.0 : std::log(static_cast<double>(parent_visits));

  for (std::size_t idx = 0; idx < n_children; ++idx) {
    double score = std::numeric_limits<double>::infinity();
    if (parent_visits > 0 && child_visits[idx] > 0) {
      score = child_qs[idx] +
              exp_weight *
                  std::sqrt(2.0 * log_parent /
                            static_cast<double>(child_visits[idx]));
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
    return 0;
  }
  *out_position = best_positions[tie_pick % best_positions.size()];
  return 1;
}

extern "C" int smart_native_ucb_best_count(std::size_t parent_visits,
                                            const double* child_qs,
                                            const std::size_t* child_visits,
                                            std::size_t n_children,
                                            double exp_weight,
                                            std::size_t* out_count) {
  if (out_count == nullptr || child_qs == nullptr ||
      child_visits == nullptr || n_children == 0) {
    return 0;
  }

  std::size_t best_count = 0;
  double best_score = -std::numeric_limits<double>::infinity();
  const double log_parent =
      parent_visits == 0 ? 0.0 : std::log(static_cast<double>(parent_visits));

  for (std::size_t idx = 0; idx < n_children; ++idx) {
    double score = std::numeric_limits<double>::infinity();
    if (parent_visits > 0 && child_visits[idx] > 0) {
      score = child_qs[idx] +
              exp_weight *
                  std::sqrt(2.0 * log_parent /
                            static_cast<double>(child_visits[idx]));
    }
    if (score > best_score) {
      best_score = score;
      best_count = 1;
    } else if (score == best_score) {
      best_count += 1;
    }
  }

  if (best_count == 0) {
    return 0;
  }
  *out_count = best_count;
  return 1;
}

extern "C" int smart_native_prob_skip_exploration(double parent_reward,
                                                   const double* child_rewards,
                                                   const double* child_qs,
                                                   std::size_t n_children,
                                                   double best_reward,
                                                   double skip_rate,
                                                   double* out_probability) {
  if (out_probability == nullptr || child_rewards == nullptr ||
      child_qs == nullptr) {
    return 0;
  }

  double max_q = 0.0;
  for (std::size_t idx = 0; idx < n_children; ++idx) {
    if (child_rewards[idx] > parent_reward && child_qs[idx] > max_q) {
      max_q = child_qs[idx];
    }
  }
  const double probability = max_q / (best_reward + 1e-9);
  *out_probability = std::min(std::max(probability, 0.0), skip_rate);
  return 1;
}

extern "C" int smart_native_bbox_volumes(const double* bounds,
                                          std::size_t n_boxes,
                                          double* out_volumes) {
  if (bounds == nullptr || out_volumes == nullptr) {
    return 0;
  }
  for (std::size_t idx = 0; idx < n_boxes; ++idx) {
    out_volumes[idx] = bbox_volume_at(bounds, idx);
  }
  return 1;
}

extern "C" int smart_native_bbox_valid_mask(const double* bounds,
                                             std::size_t n_boxes,
                                             std::uint8_t* out_mask) {
  if (bounds == nullptr || out_mask == nullptr) {
    return 0;
  }
  for (std::size_t idx = 0; idx < n_boxes; ++idx) {
    out_mask[idx] = bbox_valid_at(bounds, idx) ? 1 : 0;
  }
  return 1;
}

extern "C" int smart_native_total_bbox_volume(const double* bounds,
                                               std::size_t n_boxes,
                                               double* out_total_volume) {
  if (bounds == nullptr || out_total_volume == nullptr) {
    return 0;
  }
  double total = 0.0;
  for (std::size_t idx = 0; idx < n_boxes; ++idx) {
    total += bbox_volume_at(bounds, idx);
  }
  *out_total_volume = total;
  return 1;
}

extern "C" int smart_native_bbox_union_bounds(const double* bounds,
                                               std::size_t n_boxes,
                                               double* out_bounds) {
  if (bounds == nullptr || out_bounds == nullptr || n_boxes == 0) {
    return 0;
  }
  out_bounds[0] = bounds[0];
  out_bounds[1] = bounds[1];
  out_bounds[2] = bounds[2];
  out_bounds[3] = bounds[3];
  out_bounds[4] = bounds[4];
  out_bounds[5] = bounds[5];
  for (std::size_t idx = 1; idx < n_boxes; ++idx) {
    const double* row = bounds + idx * 6;
    out_bounds[0] = std::min(out_bounds[0], row[0]);
    out_bounds[1] = std::min(out_bounds[1], row[1]);
    out_bounds[2] = std::min(out_bounds[2], row[2]);
    out_bounds[3] = std::max(out_bounds[3], row[3]);
    out_bounds[4] = std::max(out_bounds[4], row[4]);
    out_bounds[5] = std::max(out_bounds[5], row[5]);
  }
  return 1;
}

extern "C" int smart_native_bbox_union_volume(const double* bounds,
                                               std::size_t n_boxes,
                                               double* out_volume) {
  if (out_volume == nullptr) {
    return 0;
  }
  double union_bounds[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (!smart_native_bbox_union_bounds(bounds, n_boxes, union_bounds)) {
    return 0;
  }
  const double dx = std::max(0.0, union_bounds[3] - union_bounds[0]);
  const double dy = std::max(0.0, union_bounds[4] - union_bounds[1]);
  const double dz = std::max(0.0, union_bounds[5] - union_bounds[2]);
  *out_volume = dx * dy * dz;
  return 1;
}

extern "C" int smart_native_coverage_mask(const double* points,
                                           std::size_t n_points,
                                           const double* bounds,
                                           std::uint8_t* out_mask) {
  if ((n_points > 0 && points == nullptr) || bounds == nullptr ||
      out_mask == nullptr) {
    return 0;
  }
  for (std::size_t idx = 0; idx < n_points; ++idx) {
    const double* point = points + idx * 3;
    out_mask[idx] = (bounds[0] <= point[0] && point[0] <= bounds[3] &&
                     bounds[1] <= point[1] && point[1] <= bounds[4] &&
                     bounds[2] <= point[2] && point[2] <= bounds[5])
                        ? 1
                        : 0;
  }
  return 1;
}

extern "C" int smart_native_apply_axis_action(const double* bounds,
                                               std::size_t n_boxes,
                                               std::size_t action,
                                               std::size_t num_action_scale,
                                               double action_unit,
                                               double* out_bounds) {
  if (bounds == nullptr || out_bounds == nullptr ||
      !valid_action_scale_count(num_action_scale)) {
    return 0;
  }
  const std::size_t actions_per_bbox = 6 * num_action_scale + 1;
  const std::size_t bbox_idx = action / actions_per_bbox;
  if (bbox_idx >= n_boxes) {
    return 0;
  }
  std::copy(bounds, bounds + n_boxes * 6, out_bounds);
  const std::size_t local = action % actions_per_bbox;
  if (local == actions_per_bbox - 1) {
    return 1;
  }
  const std::size_t coord_idx = local / num_action_scale;
  const std::size_t scale_idx = local % num_action_scale;
  if (coord_idx >= 6 || scale_idx >= num_action_scale) {
    return 0;
  }
  out_bounds[bbox_idx * 6 + coord_idx] +=
      action_scale_at(scale_idx, num_action_scale) * action_unit;
  return 1;
}

extern "C" int smart_native_normalize_vertices(const double* vertices,
                                                std::size_t n_vertices,
                                                int mode,
                                                int center_mode,
                                                double target,
                                                double* out_vertices,
                                                double* out_stats) {
  if (vertices == nullptr || out_vertices == nullptr || out_stats == nullptr ||
      n_vertices == 0) {
    return 0;
  }

  fill_vertex_stats(vertices, n_vertices, out_stats);
  double center[3] = {out_stats[11], out_stats[12], out_stats[13]};
  if (center_mode == 1) {
    center[0] = 0.0;
    center[1] = 0.0;
    center[2] = 0.0;
    for (std::size_t idx = 0; idx < n_vertices; ++idx) {
      const double* point = vertices + idx * 3;
      center[0] += point[0];
      center[1] += point[1];
      center[2] += point[2];
    }
    center[0] /= static_cast<double>(n_vertices);
    center[1] /= static_cast<double>(n_vertices);
    center[2] /= static_cast<double>(n_vertices);
  } else if (center_mode != 0) {
    return 0;
  }

  double denominator = 0.0;
  if (mode == 0) {
    denominator = out_stats[10];
  } else if (mode == 1) {
    denominator = std::max(out_stats[7], std::max(out_stats[8], out_stats[9]));
  } else if (mode == 2) {
    for (std::size_t idx = 0; idx < n_vertices; ++idx) {
      denominator = std::max(denominator, distance3(vertices + idx * 3, center));
    }
  } else {
    return 0;
  }
  if (!(denominator > 0.0)) {
    return 0;
  }

  const double scale = target / denominator;
  for (std::size_t idx = 0; idx < n_vertices; ++idx) {
    const double* point = vertices + idx * 3;
    double* out_point = out_vertices + idx * 3;
    out_point[0] = (point[0] - center[0]) * scale;
    out_point[1] = (point[1] - center[1]) * scale;
    out_point[2] = (point[2] - center[2]) * scale;
  }

  out_stats[15] = center[0];
  out_stats[16] = center[1];
  out_stats[17] = center[2];
  out_stats[18] = scale;
  fill_vertex_stats(out_vertices, n_vertices, out_stats + 19);
  return 1;
}

extern "C" int smart_native_tetra_volumes(const double* vertices,
                                           std::size_t n_vertices,
                                           const std::size_t* voxels,
                                           std::size_t n_voxels,
                                           double* out_volumes) {
  if (vertices == nullptr || voxels == nullptr || out_volumes == nullptr) {
    return 0;
  }
  for (std::size_t voxel_idx = 0; voxel_idx < n_voxels; ++voxel_idx) {
    const std::size_t* voxel = voxels + voxel_idx * 4;
    if (!voxel_indices_valid(voxel, n_vertices)) {
      return 0;
    }
    const double* p0 = vertices + voxel[0] * 3;
    const double* p1 = vertices + voxel[1] * 3;
    const double* p2 = vertices + voxel[2] * 3;
    const double* p3 = vertices + voxel[3] * 3;
    const double a[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    const double b[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    const double c[3] = {p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]};
    double bx_c[3] = {0.0, 0.0, 0.0};
    cross3(b, c, bx_c);
    out_volumes[voxel_idx] = std::abs(dot3(a, bx_c)) / 6.0;
  }
  return 1;
}

extern "C" int smart_native_tetra_centroids(const double* vertices,
                                             std::size_t n_vertices,
                                             const std::size_t* voxels,
                                             std::size_t n_voxels,
                                             double* out_centroids) {
  if (vertices == nullptr || voxels == nullptr || out_centroids == nullptr) {
    return 0;
  }
  for (std::size_t voxel_idx = 0; voxel_idx < n_voxels; ++voxel_idx) {
    const std::size_t* voxel = voxels + voxel_idx * 4;
    if (!voxel_indices_valid(voxel, n_vertices)) {
      return 0;
    }
    double* out = out_centroids + voxel_idx * 3;
    out[0] = 0.0;
    out[1] = 0.0;
    out[2] = 0.0;
    for (std::size_t corner = 0; corner < 4; ++corner) {
      const double* point = vertices + voxel[corner] * 3;
      out[0] += point[0];
      out[1] += point[1];
      out[2] += point[2];
    }
    out[0] /= 4.0;
    out[1] /= 4.0;
    out[2] /= 4.0;
  }
  return 1;
}

extern "C" int smart_native_tetra_surface_faces(const std::size_t* voxels,
                                                 std::size_t n_voxels,
                                                 std::size_t* out_faces,
                                                 std::size_t* out_n_faces) {
  if (voxels == nullptr || out_faces == nullptr || out_n_faces == nullptr) {
    return 0;
  }
  std::map<std::array<std::size_t, 3>, std::size_t> key_to_index;
  std::vector<std::pair<std::array<std::size_t, 3>, bool>> ordered_faces;
  ordered_faces.reserve(n_voxels * 4);
  for (std::size_t voxel_idx = 0; voxel_idx < n_voxels; ++voxel_idx) {
    const std::size_t* voxel = voxels + voxel_idx * 4;
    for (const auto& face : tet_faces(voxel)) {
      const auto key = sorted_face(face);
      auto iter = key_to_index.find(key);
      if (iter == key_to_index.end()) {
        key_to_index.emplace(key, ordered_faces.size());
        ordered_faces.emplace_back(face, true);
      } else {
        ordered_faces[iter->second].second = false;
      }
    }
  }

  std::size_t write_idx = 0;
  for (const auto& entry : ordered_faces) {
    if (!entry.second) {
      continue;
    }
    out_faces[write_idx++] = entry.first[0];
    out_faces[write_idx++] = entry.first[1];
    out_faces[write_idx++] = entry.first[2];
  }
  *out_n_faces = write_idx / 3;
  return 1;
}

extern "C" int smart_native_tetra_adjacency(const std::size_t* voxels,
                                             std::size_t n_voxels,
                                             std::size_t* out_offsets,
                                             std::size_t* out_values,
                                             std::size_t values_capacity,
                                             std::size_t* out_n_values) {
  if (out_offsets == nullptr || out_n_values == nullptr ||
      (n_voxels > 0 && voxels == nullptr)) {
    return 0;
  }

  std::map<std::array<std::size_t, 3>, std::vector<std::size_t>> face_to_voxels;
  std::vector<std::array<std::size_t, 3>> ordered_faces;
  for (std::size_t voxel_idx = 0; voxel_idx < n_voxels; ++voxel_idx) {
    const std::size_t* voxel = voxels + voxel_idx * 4;
    for (const auto& face : tet_faces(voxel)) {
      const auto key = sorted_face(face);
      if (face_to_voxels.find(key) == face_to_voxels.end()) {
        ordered_faces.push_back(key);
      }
      face_to_voxels[key].push_back(voxel_idx);
    }
  }

  std::vector<std::set<std::size_t>> adjacency(n_voxels);
  for (const auto& key : ordered_faces) {
    const auto& owners = face_to_voxels[key];
    if (owners.size() < 2) {
      continue;
    }
    for (const std::size_t owner : owners) {
      for (const std::size_t other : owners) {
        if (other != owner) {
          adjacency[owner].insert(other);
        }
      }
    }
  }

  std::size_t total = 0;
  out_offsets[0] = 0;
  for (std::size_t idx = 0; idx < n_voxels; ++idx) {
    total += adjacency[idx].size();
    out_offsets[idx + 1] = total;
  }
  *out_n_values = total;

  if (out_values == nullptr) {
    return 1;
  }
  if (values_capacity < total) {
    return 0;
  }
  std::size_t cursor = 0;
  for (const auto& row : adjacency) {
    for (const std::size_t value : row) {
      out_values[cursor++] = value;
    }
  }
  return 1;
}

extern "C" int smart_native_load_gmsh_counts(const char* path,
                                              std::size_t* out_n_vertices,
                                              std::size_t* out_n_faces,
                                              std::size_t* out_n_voxels) {
  if (out_n_vertices == nullptr || out_n_faces == nullptr ||
      out_n_voxels == nullptr) {
    return 0;
  }
  if (!count_gmsh_file(path, out_n_vertices, out_n_faces, out_n_voxels)) {
    return 0;
  }
  return 1;
}

extern "C" int smart_native_load_gmsh(const char* path,
                                       double* out_vertices,
                                       std::size_t* out_faces,
                                       std::size_t* out_voxels,
                                       std::size_t vertex_capacity,
                                       std::size_t face_capacity,
                                       std::size_t voxel_capacity,
                                       std::size_t* out_n_vertices,
                                       std::size_t* out_n_faces,
                                       std::size_t* out_n_voxels) {
  if (out_vertices == nullptr || out_faces == nullptr ||
      out_voxels == nullptr || out_n_vertices == nullptr ||
      out_n_faces == nullptr || out_n_voxels == nullptr) {
    return 0;
  }
  ParsedGmsh parsed;
  if (!parse_gmsh_file(path, &parsed)) {
    return 0;
  }
  if (vertex_capacity < parsed.vertices.size() * 3 ||
      face_capacity < parsed.faces.size() * 3 ||
      voxel_capacity < parsed.voxels.size() * 4) {
    return 0;
  }

  for (std::size_t idx = 0; idx < parsed.vertices.size(); ++idx) {
    out_vertices[idx * 3 + 0] = parsed.vertices[idx][0];
    out_vertices[idx * 3 + 1] = parsed.vertices[idx][1];
    out_vertices[idx * 3 + 2] = parsed.vertices[idx][2];
  }
  for (std::size_t idx = 0; idx < parsed.faces.size(); ++idx) {
    out_faces[idx * 3 + 0] = parsed.faces[idx][0];
    out_faces[idx * 3 + 1] = parsed.faces[idx][1];
    out_faces[idx * 3 + 2] = parsed.faces[idx][2];
  }
  for (std::size_t idx = 0; idx < parsed.voxels.size(); ++idx) {
    out_voxels[idx * 4 + 0] = parsed.voxels[idx][0];
    out_voxels[idx * 4 + 1] = parsed.voxels[idx][1];
    out_voxels[idx * 4 + 2] = parsed.voxels[idx][2];
    out_voxels[idx * 4 + 3] = parsed.voxels[idx][3];
  }
  *out_n_vertices = parsed.vertices.size();
  *out_n_faces = parsed.faces.size();
  *out_n_voxels = parsed.voxels.size();
  return 1;
}

extern "C" int smart_native_save_gmsh(const char* path,
                                       const double* vertices,
                                       std::size_t n_vertices,
                                       const std::size_t* faces,
                                       std::size_t n_faces,
                                       const std::size_t* voxels,
                                       std::size_t n_voxels) {
  if (path == nullptr || (n_vertices > 0 && vertices == nullptr) ||
      (n_faces > 0 && faces == nullptr) ||
      (n_voxels > 0 && voxels == nullptr)) {
    return 0;
  }
  std::ofstream output(path);
  if (!output) {
    return 0;
  }

  output << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
  output << "$Nodes\n" << n_vertices << "\n";
  output << std::setprecision(17);
  for (std::size_t idx = 0; idx < n_vertices; ++idx) {
    const double* vertex = vertices + idx * 3;
    output << (idx + 1) << " " << vertex[0] << " " << vertex[1] << " "
           << vertex[2] << "\n";
  }
  output << "$EndNodes\n";
  output << "$Elements\n" << (n_faces + n_voxels) << "\n";
  std::size_t element_id = 1;
  for (std::size_t idx = 0; idx < n_faces; ++idx) {
    const std::size_t* face = faces + idx * 3;
    if (face[0] >= n_vertices || face[1] >= n_vertices ||
        face[2] >= n_vertices) {
      return 0;
    }
    output << element_id++ << " 2 0 " << (face[0] + 1) << " "
           << (face[1] + 1) << " " << (face[2] + 1) << "\n";
  }
  for (std::size_t idx = 0; idx < n_voxels; ++idx) {
    const std::size_t* voxel = voxels + idx * 4;
    if (voxel[0] >= n_vertices || voxel[1] >= n_vertices ||
        voxel[2] >= n_vertices || voxel[3] >= n_vertices) {
      return 0;
    }
    output << element_id++ << " 4 0 " << (voxel[0] + 1) << " "
           << (voxel[1] + 1) << " " << (voxel[2] + 1) << " "
           << (voxel[3] + 1) << "\n";
  }
  output << "$EndElements\n";
  return output.good() ? 1 : 0;
}

extern "C" int smart_native_symmetric_chamfer(const double* left,
                                               std::size_t n_left,
                                               const double* right,
                                               std::size_t n_right,
                                               double* out_distance) {
  if (left == nullptr || right == nullptr || out_distance == nullptr ||
      n_left == 0 || n_right == 0) {
    return 0;
  }

  auto mean_nearest = [](const double* source, std::size_t n_source,
                         const double* target, std::size_t n_target) -> double {
    double total = 0.0;
    for (std::size_t src_idx = 0; src_idx < n_source; ++src_idx) {
      const double* src = source + src_idx * 3;
      double best = std::numeric_limits<double>::infinity();
      for (std::size_t dst_idx = 0; dst_idx < n_target; ++dst_idx) {
        const double* dst = target + dst_idx * 3;
        const double dx = src[0] - dst[0];
        const double dy = src[1] - dst[1];
        const double dz = src[2] - dst[2];
        best = std::min(best, dx * dx + dy * dy + dz * dz);
      }
      total += best;
    }
    return total / static_cast<double>(n_source);
  };

  *out_distance = mean_nearest(right, n_right, left, n_left) +
                  mean_nearest(left, n_left, right, n_right);
  return 1;
}

extern "C" int smart_native_centroid_proxy_axis_rewards(
    const double* centroids,
    const double* volumes,
    std::size_t n_points,
    const double* bounds,
    const double* rotations,
    std::size_t n_boxes,
    std::size_t num_action_scale,
    double action_unit,
    double volume_sum,
    double last_bbox_score,
    double cover_penalty,
    double pen_rate,
    std::size_t* out_actions,
    double* out_rewards,
    std::size_t* out_n_rewards) {
  if ((n_points > 0 && (centroids == nullptr || volumes == nullptr)) ||
      (n_boxes > 0 && (bounds == nullptr || rotations == nullptr)) ||
      out_actions == nullptr || out_rewards == nullptr ||
      out_n_rewards == nullptr || volume_sum <= 0.0 ||
      !valid_action_scale_count(num_action_scale)) {
    return 0;
  }

  const std::size_t word_count = (n_points + 63) / 64;
  std::vector<std::vector<std::uint64_t>> base_masks(
      n_boxes, std::vector<std::uint64_t>(word_count, 0));
  std::vector<double> bbox_volumes(n_boxes, 0.0);
  double total_bbox_volume = 0.0;
  for (std::size_t bbox_idx = 0; bbox_idx < n_boxes; ++bbox_idx) {
    const double* bbox = bounds + bbox_idx * 6;
    const double* rotation = rotations + bbox_idx * 9;
    if (bbox_valid_at(bounds, bbox_idx)) {
      const double volume = bbox_volume_at(bounds, bbox_idx);
      bbox_volumes[bbox_idx] = volume;
      total_bbox_volume += volume;
      for (std::size_t point_idx = 0; point_idx < n_points; ++point_idx) {
        if (point_in_oriented_bounds(centroids + point_idx * 3, bbox,
                                     rotation)) {
          base_masks[bbox_idx][point_idx / 64] |=
              static_cast<std::uint64_t>(1) << (point_idx % 64);
        }
      }
    }
  }

  const std::size_t actions_per_bbox = 6 * num_action_scale + 1;
  std::size_t cursor = 0;
  for (std::size_t bbox_idx = 0; bbox_idx < n_boxes; ++bbox_idx) {
    const double* base_bbox = bounds + bbox_idx * 6;
    const double* rotation = rotations + bbox_idx * 9;
    for (std::size_t coord_idx = 0; coord_idx < 6; ++coord_idx) {
      for (std::size_t scale_idx = 0; scale_idx < num_action_scale;
           ++scale_idx) {
        const std::size_t action =
            bbox_idx * actions_per_bbox + coord_idx * num_action_scale +
            scale_idx;
        out_actions[cursor] = action;

        double candidate[6] = {base_bbox[0], base_bbox[1], base_bbox[2],
                               base_bbox[3], base_bbox[4], base_bbox[5]};
        candidate[coord_idx] +=
            action_scale_at(scale_idx, num_action_scale) * action_unit;
        if (!(candidate[0] < candidate[3] && candidate[1] < candidate[4] &&
              candidate[2] < candidate[5])) {
          out_rewards[cursor++] = -std::numeric_limits<double>::infinity();
          continue;
        }

        std::vector<std::uint64_t> union_mask(word_count, 0);
        for (std::size_t point_idx = 0; point_idx < n_points; ++point_idx) {
          if (point_in_oriented_bounds(centroids + point_idx * 3, candidate,
                                       rotation)) {
            union_mask[point_idx / 64] |=
                static_cast<std::uint64_t>(1) << (point_idx % 64);
          }
        }
        for (std::size_t other_idx = 0; other_idx < n_boxes; ++other_idx) {
          if (other_idx == bbox_idx) {
            continue;
          }
          for (std::size_t word_idx = 0; word_idx < word_count; ++word_idx) {
            union_mask[word_idx] |= base_masks[other_idx][word_idx];
          }
        }

        double covered_volume = 0.0;
        for (std::size_t word_idx = 0; word_idx < word_count; ++word_idx) {
          std::uint64_t remaining = union_mask[word_idx];
          while (remaining != 0) {
            const std::size_t bit =
                static_cast<std::size_t>(__builtin_ctzll(remaining));
            const std::size_t point_idx = word_idx * 64 + bit;
            if (point_idx < n_points) {
              covered_volume += volumes[point_idx];
            }
            remaining &= remaining - 1;
          }
        }

        const double covered = covered_volume / volume_sum;
        const double candidate_volume =
            std::max(0.0, candidate[3] - candidate[0]) *
            std::max(0.0, candidate[4] - candidate[1]) *
            std::max(0.0, candidate[5] - candidate[2]);
        const double new_total =
            total_bbox_volume - bbox_volumes[bbox_idx] + candidate_volume;
        const double bvs = new_total / volume_sum;
        const double proxy_score =
            -std::abs(bvs - 1.0) -
            (1.0 - covered) * pen_rate * cover_penalty;
        out_rewards[cursor++] = proxy_score - last_bbox_score;
      }
    }
  }
  *out_n_rewards = cursor;
  return 1;
}

extern "C" int smart_native_partition_summaries(
    const double* vertices,
    std::size_t n_vertices,
    const std::size_t* voxels,
    std::size_t n_voxels,
    const double* volumes,
    const std::size_t* partition_offsets,
    const std::size_t* partition_indices,
    std::size_t n_partitions,
    std::size_t n_partition_indices,
    int unique_points,
    double* out_volumes,
    double* out_bounds,
    std::size_t* out_point_offsets,
    double* out_points,
    std::size_t* out_n_points) {
  if (vertices == nullptr || voxels == nullptr || volumes == nullptr ||
      partition_offsets == nullptr || partition_indices == nullptr ||
      out_volumes == nullptr || out_bounds == nullptr ||
      out_point_offsets == nullptr || out_points == nullptr ||
      out_n_points == nullptr) {
    return 0;
  }

  std::size_t point_write = 0;
  out_point_offsets[0] = 0;
  for (std::size_t part_idx = 0; part_idx < n_partitions; ++part_idx) {
    const std::size_t start = partition_offsets[part_idx];
    const std::size_t end = partition_offsets[part_idx + 1];
    if (start >= end || end > n_partition_indices) {
      return 0;
    }
    const std::size_t first_tet_idx = partition_indices[start];
    if (first_tet_idx >= n_voxels ||
        !voxel_indices_valid(voxels + first_tet_idx * 4, n_vertices)) {
      return 0;
    }
    const double* first_vertex =
        vertices + voxels[first_tet_idx * 4] * 3;
    double min_v[3] = {first_vertex[0], first_vertex[1], first_vertex[2]};
    double max_v[3] = {first_vertex[0], first_vertex[1], first_vertex[2]};
    double volume = 0.0;
    std::set<std::array<std::uint64_t, 3>> seen_points;

    for (std::size_t idx = start; idx < end; ++idx) {
      const std::size_t tet_idx = partition_indices[idx];
      if (tet_idx >= n_voxels) {
        return 0;
      }
      const std::size_t* voxel = voxels + tet_idx * 4;
      if (!voxel_indices_valid(voxel, n_vertices)) {
        return 0;
      }
      volume += volumes[tet_idx];
      for (std::size_t corner = 0; corner < 4; ++corner) {
        const double* point = vertices + voxel[corner] * 3;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          min_v[axis] = std::min(min_v[axis], point[axis]);
          max_v[axis] = std::max(max_v[axis], point[axis]);
        }
        bool keep = true;
        if (unique_points) {
          std::array<std::uint64_t, 3> key = {
              double_bits(point[0]),
              double_bits(point[1]),
              double_bits(point[2]),
          };
          keep = seen_points.insert(key).second;
        }
        if (keep) {
          out_points[point_write++] = point[0];
          out_points[point_write++] = point[1];
          out_points[point_write++] = point[2];
        }
      }
    }

    out_volumes[part_idx] = volume;
    double* bounds = out_bounds + part_idx * 6;
    bounds[0] = min_v[0];
    bounds[1] = min_v[1];
    bounds[2] = min_v[2];
    bounds[3] = max_v[0];
    bounds[4] = max_v[1];
    bounds[5] = max_v[2];
    out_point_offsets[part_idx + 1] = point_write;
  }
  *out_n_points = point_write;
  return 1;
}

namespace {
void append_action_upper_rewards_for_box(const double* row,
                                         double old_volume,
                                         double total_volume,
                                         std::size_t num_action_scale,
                                         double action_unit,
                                         double volume_sum,
                                         double last_bbox_score,
                                         double* out_rewards,
                                         std::size_t* write_idx) {
  for (std::size_t coord_idx = 0; coord_idx < 6; ++coord_idx) {
    for (std::size_t scale_idx = 0; scale_idx < num_action_scale; ++scale_idx) {
      double candidate[6] = {
          row[0], row[1], row[2], row[3], row[4], row[5],
      };
      candidate[coord_idx] += action_scale_at(scale_idx, num_action_scale) * action_unit;
      double new_volume = 0.0;
      if (candidate[0] < candidate[3] && candidate[1] < candidate[4] &&
          candidate[2] < candidate[5]) {
        const double dx = std::max(0.0, candidate[3] - candidate[0]);
        const double dy = std::max(0.0, candidate[4] - candidate[1]);
        const double dz = std::max(0.0, candidate[5] - candidate[2]);
        new_volume = dx * dy * dz;
      }
      const double new_total = total_volume - old_volume + new_volume;
      const double bvs = new_total / volume_sum;
      out_rewards[(*write_idx)++] = -std::abs(bvs - 1.0) - last_bbox_score;
    }
  }
  const double bvs = total_volume / volume_sum;
  out_rewards[(*write_idx)++] = -std::abs(bvs - 1.0) - last_bbox_score;
}
}  // namespace

extern "C" int smart_native_action_upper_rewards(const double* bounds,
                                                  std::size_t n_boxes,
                                                  std::size_t num_action_scale,
                                                  double action_unit,
                                                  double volume_sum,
                                                  double last_bbox_score,
                                                  double* out_rewards) {
  if (bounds == nullptr || out_rewards == nullptr ||
      !valid_action_scale_count(num_action_scale) || !(volume_sum > 0.0)) {
    return 0;
  }
  std::vector<double> volumes(n_boxes, 0.0);
  double total_volume = 0.0;
  for (std::size_t idx = 0; idx < n_boxes; ++idx) {
    volumes[idx] = bbox_volume_at(bounds, idx);
    total_volume += volumes[idx];
  }
  std::size_t write_idx = 0;
  for (std::size_t idx = 0; idx < n_boxes; ++idx) {
    append_action_upper_rewards_for_box(
        bounds + idx * 6,
        volumes[idx],
        total_volume,
        num_action_scale,
        action_unit,
        volume_sum,
        last_bbox_score,
        out_rewards,
        &write_idx);
  }
  return 1;
}

extern "C" int smart_native_bbox_action_upper_rewards(const double* bounds,
                                                       std::size_t n_boxes,
                                                       std::size_t bbox_idx,
                                                       std::size_t num_action_scale,
                                                       double action_unit,
                                                       double volume_sum,
                                                       double last_bbox_score,
                                                       double* out_rewards) {
  if (bounds == nullptr || out_rewards == nullptr ||
      !valid_action_scale_count(num_action_scale) || !(volume_sum > 0.0) ||
      bbox_idx >= n_boxes) {
    return 0;
  }
  std::vector<double> volumes(n_boxes, 0.0);
  double total_volume = 0.0;
  for (std::size_t idx = 0; idx < n_boxes; ++idx) {
    volumes[idx] = bbox_volume_at(bounds, idx);
    total_volume += volumes[idx];
  }
  std::size_t write_idx = 0;
  append_action_upper_rewards_for_box(
      bounds + bbox_idx * 6,
      volumes[bbox_idx],
      total_volume,
      num_action_scale,
      action_unit,
      volume_sum,
      last_bbox_score,
      out_rewards,
      &write_idx);
  return 1;
}

extern "C" int smart_native_bavf_scores(const double* part_volumes,
                                         const double* bbox_volumes,
                                         std::size_t n_items,
                                         double alpha,
                                         double* out_scores) {
  if (part_volumes == nullptr || bbox_volumes == nullptr || out_scores == nullptr) {
    return 0;
  }
  for (std::size_t idx = 0; idx < n_items; ++idx) {
    out_scores[idx] =
        bbox_volumes[idx] > 0.0 ? alpha * part_volumes[idx] / bbox_volumes[idx] : 0.0;
  }
  return 1;
}

extern "C" int smart_native_merge_bavf_reward(double prev_bvs,
                                               double left_bbox_volume,
                                               double right_bbox_volume,
                                               double merged_bbox_volume,
                                               double shape_volume,
                                               double* out_reward) {
  if (out_reward == nullptr || !(shape_volume > 0.0)) {
    return 0;
  }
  const double new_bvs =
      (prev_bvs * shape_volume - left_bbox_volume - right_bbox_volume +
       merged_bbox_volume) /
      shape_volume;
  *out_reward = -std::abs(new_bvs - 1.0) + std::abs(prev_bvs - 1.0);
  return 1;
}

extern "C" int smart_native_softmax_scaled(const double* values,
                                            std::size_t n_values,
                                            double scale,
                                            double* out_probs) {
  if (values == nullptr || out_probs == nullptr) {
    return 0;
  }
  if (n_values == 0) {
    return 1;
  }
  double max_value = -std::numeric_limits<double>::infinity();
  for (std::size_t idx = 0; idx < n_values; ++idx) {
    max_value = std::max(max_value, values[idx] * scale);
  }
  double total = 0.0;
  for (std::size_t idx = 0; idx < n_values; ++idx) {
    out_probs[idx] = std::exp(values[idx] * scale - max_value);
    total += out_probs[idx];
  }
  if (total == 0.0) {
    const double uniform = 1.0 / static_cast<double>(n_values);
    for (std::size_t idx = 0; idx < n_values; ++idx) {
      out_probs[idx] = uniform;
    }
    return 1;
  }
  for (std::size_t idx = 0; idx < n_values; ++idx) {
    out_probs[idx] /= total;
  }
  return 1;
}

extern "C" double smart_native_incremental_average(double previous,
                                                    std::size_t count,
                                                    double value) {
  return previous / static_cast<double>(count + 1) * static_cast<double>(count) +
         value / static_cast<double>(count + 1);
}
