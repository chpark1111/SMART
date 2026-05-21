#include <manifold.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace {
struct SmartManifoldHandle {
  manifold::Manifold manifold;
};

struct SmartManifoldSnapshot {
  std::vector<double> bounds;
  std::vector<double> rotations;
  std::vector<manifold::Manifold> boxes;
  std::vector<std::uint8_t> valid;
  std::vector<double> volumes;
  double total_volume = 0.0;
  double last_bbox_score = 0.0;
  std::uint64_t state_hash = 0;
};

struct SmartManifoldState {
  manifold::Manifold surface;
  std::vector<double> bounds;
  std::vector<double> rotations;
  std::vector<manifold::Manifold> boxes;
  std::vector<std::uint8_t> valid;
  std::vector<double> volumes;
  double total_volume = 0.0;
  double volume_sum = 1.0;
  double last_bbox_score = 0.0;
  bool stateful_union_cache = true;
  bool use_properties_volume = false;
  std::size_t cache_capacity = 65536;
  std::unordered_map<std::uint64_t, double> reward_cache;
  std::vector<SmartManifoldSnapshot> history;
  bool residual_cache_valid = false;
  double residual_cache = 0.0;
  bool except_union_cache_valid = false;
  std::vector<manifold::Manifold> except_unions;
  std::vector<std::uint8_t> except_union_valid;
  std::vector<std::uint8_t> except_union_built;
  bool ordered_prefix_cache_valid = false;
  std::vector<manifold::Manifold> ordered_prefix_unions;
  std::vector<std::uint8_t> ordered_prefix_valid;
  std::uint64_t version = 0;
  std::uint64_t state_hash = 0;
  std::uint64_t reward_cache_hits = 0;
  std::uint64_t reward_cache_misses = 0;
  std::uint64_t except_union_builds = 0;
  std::uint64_t except_union_cache_hits = 0;
  std::uint64_t ordered_prefix_builds = 0;
  std::uint64_t ordered_prefix_cache_hits = 0;
};

manifold::Mesh make_mesh(const float* vertices, std::size_t n_vertices,
                         const std::uint32_t* faces, std::size_t n_faces) {
  manifold::Mesh mesh;
  mesh.vertPos.reserve(n_vertices);
  for (std::size_t i = 0; i < n_vertices; ++i) {
    mesh.vertPos.push_back(glm::vec3(vertices[3 * i], vertices[3 * i + 1],
                                     vertices[3 * i + 2]));
  }
  mesh.triVerts.reserve(n_faces);
  for (std::size_t i = 0; i < n_faces; ++i) {
    mesh.triVerts.push_back(glm::ivec3(faces[3 * i], faces[3 * i + 1],
                                       faces[3 * i + 2]));
  }
  return mesh;
}

double signed_mesh_volume(const manifold::Mesh& mesh) {
  double total = 0.0;
  for (const auto& tri : mesh.triVerts) {
    const glm::vec3& p0f = mesh.vertPos[tri.x];
    const glm::vec3& p1f = mesh.vertPos[tri.y];
    const glm::vec3& p2f = mesh.vertPos[tri.z];
    const glm::dvec3 p0(p0f.x, p0f.y, p0f.z);
    const glm::dvec3 p1(p1f.x, p1f.y, p1f.z);
    const glm::dvec3 p2(p2f.x, p2f.y, p2f.z);
    const glm::dvec3 v0 = p1 - p0;
    const glm::dvec3 v1 = p2 - p1;
    const glm::dvec3 cross = glm::cross(v0, v1);
    const glm::dvec3 f1 = p0 + p1 + p2;
    total += cross.x * f1.x / 6.0;
  }
  return total;
}

double mesh_output_volume(const manifold::Manifold& manifold) {
  return signed_mesh_volume(manifold.GetMesh());
}

double properties_volume(const manifold::Manifold& manifold) {
  return manifold.GetProperties().volume;
}

double selected_volume(const manifold::Manifold& manifold,
                       bool use_properties_volume) {
  return use_properties_volume ? properties_volume(manifold)
                               : mesh_output_volume(manifold);
}

bool bbox_is_valid(const double* bounds) {
  return bounds[0] < bounds[3] && bounds[1] < bounds[4] &&
         bounds[2] < bounds[5];
}

double bbox_volume(const double* bounds) {
  if (!bbox_is_valid(bounds)) {
    return 0.0;
  }
  return (bounds[3] - bounds[0]) * (bounds[4] - bounds[1]) *
         (bounds[5] - bounds[2]);
}

std::uint64_t double_bits(double value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "double size mismatch");
  std::memcpy(&bits, &value, sizeof(value));
  return bits;
}

std::uint64_t hash_combine(std::uint64_t seed, std::uint64_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

std::uint64_t score_cache_key(std::uint64_t version, std::intptr_t action,
                              std::size_t num_action_scale,
                              double action_unit, double cover_penalty,
                              double pen_rate) {
  std::uint64_t key = version;
  key = hash_combine(key, static_cast<std::uint64_t>(action));
  key = hash_combine(key, static_cast<std::uint64_t>(num_action_scale));
  key = hash_combine(key, double_bits(action_unit));
  key = hash_combine(key, double_bits(cover_penalty));
  key = hash_combine(key, double_bits(pen_rate));
  return key;
}

std::uint64_t replacement_cache_key(std::uint64_t state_hash,
                                    std::size_t bbox_idx,
                                    const double* bounds,
                                    const double* rotation,
                                    double cover_penalty,
                                    double pen_rate) {
  std::uint64_t key = state_hash;
  key = hash_combine(key, static_cast<std::uint64_t>(bbox_idx));
  for (std::size_t idx = 0; idx < 6; ++idx) {
    key = hash_combine(key, double_bits(bounds[idx]));
  }
  for (std::size_t idx = 0; idx < 9; ++idx) {
    key = hash_combine(key, double_bits(rotation[idx]));
  }
  key = hash_combine(key, double_bits(cover_penalty));
  key = hash_combine(key, double_bits(pen_rate));
  return key;
}

std::uint64_t state_geometry_hash(const std::vector<double>& bounds,
                                  const std::vector<double>& rotations,
                                  double volume_sum) {
  std::uint64_t key = 0xcbf29ce484222325ULL;
  key = hash_combine(key, static_cast<std::uint64_t>(bounds.size()));
  key = hash_combine(key, static_cast<std::uint64_t>(rotations.size()));
  key = hash_combine(key, double_bits(volume_sum));
  for (double value : bounds) {
    key = hash_combine(key, double_bits(value));
  }
  for (double value : rotations) {
    key = hash_combine(key, double_bits(value));
  }
  return key;
}

constexpr std::uint32_t kBoxFaces[36] = {
    1, 3, 0, 1, 5, 7, 4, 6, 7, 0, 2, 6, 2, 3, 7, 0, 5, 1,
    3, 2, 0, 1, 7, 3, 4, 7, 5, 0, 6, 4, 2, 7, 6, 0, 4, 5,
};

manifold::Manifold box_from_vertices(const float* box_vertices) {
  manifold::Mesh mesh = make_mesh(box_vertices, 8, kBoxFaces, 12);
  if (signed_mesh_volume(mesh) < 0.0) {
    for (auto& tri : mesh.triVerts) {
      const int tmp = tri.y;
      tri.y = tri.z;
      tri.z = tmp;
    }
  }
  return manifold::Manifold(mesh);
}

void box_vertices_from_params(const double* bounds, const double* rotation,
                              float* out_vertices) {
  const double lengths[3] = {bounds[3] - bounds[0], bounds[4] - bounds[1],
                             bounds[5] - bounds[2]};
  const double base[3] = {
      bounds[0] * rotation[0] + bounds[1] * rotation[3] +
          bounds[2] * rotation[6],
      bounds[0] * rotation[1] + bounds[1] * rotation[4] +
          bounds[2] * rotation[7],
      bounds[0] * rotation[2] + bounds[1] * rotation[5] +
          bounds[2] * rotation[8],
  };

  std::size_t offset = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        out_vertices[offset++] = static_cast<float>(
            base[0] + rotation[0] * i * lengths[0] +
            rotation[3] * j * lengths[1] + rotation[6] * k * lengths[2]);
        out_vertices[offset++] = static_cast<float>(
            base[1] + rotation[1] * i * lengths[0] +
            rotation[4] * j * lengths[1] + rotation[7] * k * lengths[2]);
        out_vertices[offset++] = static_cast<float>(
            base[2] + rotation[2] * i * lengths[0] +
            rotation[5] * j * lengths[1] + rotation[8] * k * lengths[2]);
      }
    }
  }
}

manifold::Manifold box_from_params(const double* bounds,
                                   const double* rotation) {
  float vertices[8 * 3];
  box_vertices_from_params(bounds, rotation, vertices);
  return box_from_vertices(vertices);
}

bool append_ordered_box(std::vector<manifold::Manifold>* boxes,
                        const std::vector<manifold::Manifold>& current_boxes,
                        const std::vector<std::uint8_t>& current_valid,
                        const manifold::Manifold* candidate_box,
                        bool candidate_valid, std::size_t candidate_idx,
                        std::size_t idx) {
  if (idx == candidate_idx) {
    if (candidate_valid) {
      boxes->push_back(*candidate_box);
      return true;
    }
    return false;
  }
  if (idx < current_valid.size() && current_valid[idx]) {
    boxes->push_back(current_boxes[idx]);
    return true;
  }
  return false;
}

manifold::Manifold union_ordered_boxes(
    const std::vector<manifold::Manifold>& boxes) {
  manifold::Manifold merged = boxes[0];
  for (std::size_t idx = 1; idx < boxes.size(); ++idx) {
    merged = merged + boxes[idx];
  }
  return merged;
}

manifold::Manifold residual_manifold_for_box_set(
    const manifold::Manifold& surface,
    const std::vector<manifold::Manifold>& current_boxes,
    const std::vector<std::uint8_t>& current_valid,
    const manifold::Manifold* candidate_box, bool candidate_valid,
    std::size_t candidate_idx) {
  std::vector<manifold::Manifold> boxes;
  boxes.reserve(current_boxes.size());

  for (std::size_t i = 0; i < current_boxes.size(); ++i) {
    append_ordered_box(&boxes, current_boxes, current_valid, candidate_box,
                       candidate_valid, candidate_idx, i);
  }

  if (boxes.empty()) {
    return surface;
  }
  const manifold::Manifold merged = union_ordered_boxes(boxes);
  return surface - merged;
}

double residual_volume_for_box_set(
    const manifold::Manifold& surface,
    const std::vector<manifold::Manifold>& current_boxes,
    const std::vector<std::uint8_t>& current_valid,
    const manifold::Manifold* candidate_box, bool candidate_valid,
    std::size_t candidate_idx, double empty_residual,
    bool use_properties_volume) {
  bool has_box = false;
  for (std::size_t i = 0; i < current_boxes.size(); ++i) {
    if (i == candidate_idx) {
      has_box = candidate_valid;
    } else if (i < current_valid.size()) {
      has_box = current_valid[i] != 0;
    }
    if (has_box) {
      break;
    }
  }
  if (!has_box) {
    return empty_residual;
  }
  const manifold::Manifold residual = residual_manifold_for_box_set(
      surface, current_boxes, current_valid, candidate_box, candidate_valid,
      candidate_idx);
  return selected_volume(residual, use_properties_volume);
}

manifold::Manifold residual_manifold_for_box_vertices(
    const manifold::Manifold& surface, const float* box_vertices,
    std::size_t n_boxes) {
  if (n_boxes == 0) {
    return surface;
  }
  std::vector<manifold::Manifold> boxes;
  boxes.reserve(n_boxes);
  for (std::size_t i = 0; i < n_boxes; ++i) {
    boxes.push_back(box_from_vertices(box_vertices + i * 8 * 3));
  }
  const manifold::Manifold merged = union_ordered_boxes(boxes);
  return surface - merged;
}

void state_clear_caches(SmartManifoldState* state) {
  state->residual_cache_valid = false;
  state->except_union_cache_valid = false;
  state->except_unions.clear();
  state->except_union_valid.clear();
  state->except_union_built.clear();
  state->ordered_prefix_cache_valid = false;
  state->ordered_prefix_unions.clear();
  state->ordered_prefix_valid.clear();
  state->version += 1;
}

void state_rebuild_boxes(SmartManifoldState* state) {
  const std::size_t n_boxes = state->bounds.size() / 6;
  state->boxes.clear();
  state->boxes.resize(n_boxes);
  state->valid.assign(n_boxes, 0);
  state->volumes.assign(n_boxes, 0.0);
  state->total_volume = 0.0;

  for (std::size_t idx = 0; idx < n_boxes; ++idx) {
    const double* box = state->bounds.data() + idx * 6;
    if (!bbox_is_valid(box)) {
      continue;
    }
    state->valid[idx] = 1;
    state->volumes[idx] = bbox_volume(box);
    state->total_volume += state->volumes[idx];
    state->boxes[idx] = box_from_params(box, state->rotations.data() + idx * 9);
  }
  state->state_hash =
      state_geometry_hash(state->bounds, state->rotations, state->volume_sum);
  state_clear_caches(state);
}

double state_current_residual(SmartManifoldState* state) {
  if (state->stateful_union_cache && state->residual_cache_valid) {
    return state->residual_cache;
  }
  const double residual = residual_volume_for_box_set(
      state->surface, state->boxes, state->valid, nullptr, false,
      std::numeric_limits<std::size_t>::max(), state->volume_sum,
      state->use_properties_volume);
  if (state->stateful_union_cache) {
    state->residual_cache = residual;
    state->residual_cache_valid = true;
  }
  return residual;
}

void state_prepare_except_unions(SmartManifoldState* state) {
  if (state->except_union_cache_valid) {
    return;
  }
  const std::size_t n_boxes = state->bounds.size() / 6;
  state->except_unions.clear();
  state->except_unions.resize(n_boxes);
  state->except_union_valid.assign(n_boxes, 0);
  state->except_union_built.assign(n_boxes, 0);
  state->except_union_cache_valid = true;
}

void state_ensure_except_union(SmartManifoldState* state,
                               std::size_t skip_idx) {
  state_prepare_except_unions(state);
  const std::size_t n_boxes = state->bounds.size() / 6;
  if (skip_idx >= n_boxes) {
    return;
  }
  if (state->except_union_built[skip_idx]) {
    state->except_union_cache_hits += 1;
    return;
  }

  std::vector<manifold::Manifold> boxes;
  boxes.reserve(n_boxes > 0 ? n_boxes - 1 : 0);
  for (std::size_t idx = 0; idx < n_boxes; ++idx) {
    if (idx == skip_idx || !state->valid[idx]) {
      continue;
    }
    boxes.push_back(state->boxes[idx]);
  }
  state->except_union_built[skip_idx] = 1;
  state->except_union_builds += 1;
  if (!boxes.empty()) {
    state->except_union_valid[skip_idx] = 1;
    state->except_unions[skip_idx] =
        boxes.size() == 1
            ? boxes[0]
            : manifold::Manifold::BatchBoolean(
                  boxes, manifold::Manifold::OpType::Add);
  }
}

void state_prepare_ordered_prefix_unions(SmartManifoldState* state) {
  if (state->ordered_prefix_cache_valid) {
    state->ordered_prefix_cache_hits += 1;
    return;
  }
  const std::size_t n_boxes = state->bounds.size() / 6;
  state->ordered_prefix_unions.clear();
  state->ordered_prefix_unions.resize(n_boxes);
  state->ordered_prefix_valid.assign(n_boxes, 0);

  bool has_prefix = false;
  manifold::Manifold prefix;
  for (std::size_t idx = 0; idx < n_boxes; ++idx) {
    if (has_prefix) {
      state->ordered_prefix_unions[idx] = prefix;
      state->ordered_prefix_valid[idx] = 1;
    }
    if (idx < state->valid.size() && state->valid[idx]) {
      prefix = has_prefix ? prefix + state->boxes[idx] : state->boxes[idx];
      has_prefix = true;
    }
  }
  state->ordered_prefix_cache_valid = true;
  state->ordered_prefix_builds += 1;
}

double state_residual_for_candidate_ordered_prefix(
    SmartManifoldState* state, const manifold::Manifold* candidate_box,
    bool candidate_valid, std::size_t candidate_idx) {
  const std::size_t n_boxes = state->bounds.size() / 6;
  if (candidate_idx >= n_boxes) {
    return residual_volume_for_box_set(
        state->surface, state->boxes, state->valid, candidate_box,
        candidate_valid, candidate_idx, state->volume_sum,
        state->use_properties_volume);
  }

  state_prepare_ordered_prefix_unions(state);

  bool has_merged = false;
  manifold::Manifold merged;
  if (state->ordered_prefix_valid[candidate_idx]) {
    merged = state->ordered_prefix_unions[candidate_idx];
    has_merged = true;
  }
  if (candidate_valid) {
    merged = has_merged ? merged + *candidate_box : *candidate_box;
    has_merged = true;
  }
  for (std::size_t idx = candidate_idx + 1; idx < n_boxes; ++idx) {
    if (idx < state->valid.size() && state->valid[idx]) {
      merged = has_merged ? merged + state->boxes[idx] : state->boxes[idx];
      has_merged = true;
    }
  }
  if (!has_merged) {
    return state->volume_sum;
  }
  const manifold::Manifold residual = state->surface - merged;
  return selected_volume(residual, state->use_properties_volume);
}

double state_residual_for_candidate(SmartManifoldState* state,
                                    const manifold::Manifold* candidate_box,
                                    bool candidate_valid,
                                    std::size_t candidate_idx) {
  if (!state->stateful_union_cache) {
    return state_residual_for_candidate_ordered_prefix(
        state, candidate_box, candidate_valid, candidate_idx);
  }

  state_ensure_except_union(state, candidate_idx);
  const bool has_except = state->except_union_valid[candidate_idx] != 0;
  if (!candidate_valid && !has_except) {
    return state->volume_sum;
  }

  manifold::Manifold merged;
  if (candidate_valid && has_except) {
    merged = state->except_unions[candidate_idx] + *candidate_box;
  } else if (candidate_valid) {
    merged = *candidate_box;
  } else {
    merged = state->except_unions[candidate_idx];
  }
  const manifold::Manifold residual = state->surface - merged;
  return selected_volume(residual, state->use_properties_volume);
}

double state_score_axis_action_no_cache(SmartManifoldState* state,
                                        std::intptr_t action,
                                        std::size_t num_action_scale,
                                        double action_unit,
                                        double cover_penalty,
                                        double pen_rate,
                                        const double* action_scales) {
  if (state->volume_sum <= 0.0 || num_action_scale == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::size_t n_boxes = state->bounds.size() / 6;
  const std::size_t actions_per_bbox = 6 * num_action_scale + 1;
  if (action < 0 ||
      static_cast<std::size_t>(action) >= n_boxes * actions_per_bbox) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::size_t bbox_idx =
      static_cast<std::size_t>(action) / actions_per_bbox;
  const std::size_t local_action =
      static_cast<std::size_t>(action) % actions_per_bbox;
  if (local_action == actions_per_bbox - 1) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::size_t coord_idx = local_action / num_action_scale;
  const std::size_t scale_idx = local_action % num_action_scale;

  double candidate_bounds[6];
  const double* current_bounds = state->bounds.data() + bbox_idx * 6;
  std::copy(current_bounds, current_bounds + 6, candidate_bounds);
  candidate_bounds[coord_idx] += action_scales[scale_idx] * action_unit;

  const bool candidate_valid = bbox_is_valid(candidate_bounds);
  const double candidate_volume = bbox_volume(candidate_bounds);
  const double bvs =
      (state->total_volume - state->volumes[bbox_idx] + candidate_volume) /
      state->volume_sum;

  manifold::Manifold candidate_manifold;
  if (candidate_valid) {
    candidate_manifold =
        box_from_params(candidate_bounds, state->rotations.data() + bbox_idx * 9);
  }
  const double residual = state_residual_for_candidate(
      state, candidate_valid ? &candidate_manifold : nullptr, candidate_valid,
      bbox_idx);
  const double covered = 1.0 - residual / state->volume_sum;
  return -std::abs(bvs - 1.0) - (1.0 - covered) * pen_rate * cover_penalty;
}

double state_score_axis_action(SmartManifoldState* state, std::intptr_t action,
                               std::size_t num_action_scale,
                               double action_unit, double cover_penalty,
                               double pen_rate,
                               const double* action_scales) {
  const std::uint64_t state_action_key =
      score_cache_key(state->state_hash, action, num_action_scale, action_unit,
                      cover_penalty, pen_rate);
  const auto found = state->reward_cache.find(state_action_key);
  if (found != state->reward_cache.end()) {
    state->reward_cache_hits += 1;
    return found->second;
  }
  state->reward_cache_misses += 1;
  const double score = state_score_axis_action_no_cache(
      state, action, num_action_scale, action_unit, cover_penalty, pen_rate,
      action_scales);
  if (score == score) {
    if (state->reward_cache.size() >= state->cache_capacity) {
      state->reward_cache.clear();
    }
    state->reward_cache[state_action_key] = score;
  }
  return score;
}

double state_score_replacement(SmartManifoldState* state, std::size_t bbox_idx,
                               const double* candidate_bounds,
                               const double* candidate_rotation,
                               double cover_penalty, double pen_rate) {
  if (state->volume_sum <= 0.0 || bbox_idx >= state->bounds.size() / 6) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::uint64_t cache_key =
      replacement_cache_key(state->state_hash, bbox_idx, candidate_bounds,
                            candidate_rotation, cover_penalty, pen_rate);
  const auto found = state->reward_cache.find(cache_key);
  if (found != state->reward_cache.end()) {
    state->reward_cache_hits += 1;
    return found->second;
  }
  state->reward_cache_misses += 1;

  const bool candidate_valid = bbox_is_valid(candidate_bounds);
  const double candidate_volume = bbox_volume(candidate_bounds);
  const double bvs =
      (state->total_volume - state->volumes[bbox_idx] + candidate_volume) /
      state->volume_sum;

  manifold::Manifold candidate_manifold;
  if (candidate_valid) {
    candidate_manifold =
        box_from_params(candidate_bounds, candidate_rotation);
  }
  const double residual = state_residual_for_candidate(
      state, candidate_valid ? &candidate_manifold : nullptr, candidate_valid,
      bbox_idx);
  const double covered = 1.0 - residual / state->volume_sum;
  const double score =
      -std::abs(bvs - 1.0) - (1.0 - covered) * pen_rate * cover_penalty;
  if (score == score) {
    if (state->reward_cache.size() >= state->cache_capacity) {
      state->reward_cache.clear();
    }
    state->reward_cache[cache_key] = score;
  }
  return score;
}

void state_snapshot(SmartManifoldState* state) {
  SmartManifoldSnapshot snapshot;
  snapshot.bounds = state->bounds;
  snapshot.rotations = state->rotations;
  snapshot.boxes = state->boxes;
  snapshot.valid = state->valid;
  snapshot.volumes = state->volumes;
  snapshot.total_volume = state->total_volume;
  snapshot.last_bbox_score = state->last_bbox_score;
  snapshot.state_hash = state->state_hash;
  state->history.push_back(std::move(snapshot));
}
}  // namespace

extern "C" float smart_manifold_cube_volume(float x, float y, float z) {
  const manifold::Manifold cube =
      manifold::Manifold::Cube(glm::vec3(x, y, z), false);
  return cube.GetProperties().volume;
}

extern "C" void* smart_manifold_mesh_new(const float* vertices,
                                          std::size_t n_vertices,
                                          const std::uint32_t* faces,
                                          std::size_t n_faces) {
  try {
    manifold::Mesh mesh = make_mesh(vertices, n_vertices, faces, n_faces);
    return new SmartManifoldHandle{manifold::Manifold(mesh)};
  } catch (...) {
    return nullptr;
  }
}

extern "C" void smart_manifold_delete(void* handle) {
  delete static_cast<SmartManifoldHandle*>(handle);
}

extern "C" double smart_manifold_handle_volume(void* handle) {
  try {
    const auto* state = static_cast<SmartManifoldHandle*>(handle);
    return mesh_output_volume(state->manifold);
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" double smart_manifold_handle_volume_properties(void* handle) {
  try {
    const auto* state = static_cast<SmartManifoldHandle*>(handle);
    return properties_volume(state->manifold);
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" double smart_manifold_residual_volume_for_boxes(
    void* handle, const float* box_vertices, std::size_t n_boxes) {
  try {
    const auto* state = static_cast<SmartManifoldHandle*>(handle);
    if (n_boxes == 0) {
      return mesh_output_volume(state->manifold);
    }
    const manifold::Manifold residual =
        residual_manifold_for_box_vertices(state->manifold, box_vertices, n_boxes);
    return mesh_output_volume(residual);
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" double smart_manifold_residual_volume_for_boxes_properties(
    void* handle, const float* box_vertices, std::size_t n_boxes) {
  try {
    const auto* state = static_cast<SmartManifoldHandle*>(handle);
    if (n_boxes == 0) {
      return properties_volume(state->manifold);
    }
    const manifold::Manifold residual =
        residual_manifold_for_box_vertices(state->manifold, box_vertices, n_boxes);
    return properties_volume(residual);
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" int smart_manifold_residual_volume_for_boxes_pair(
    void* handle, const float* box_vertices, std::size_t n_boxes,
    double* out_mesh_volume, double* out_properties_volume) {
  try {
    const auto* state = static_cast<SmartManifoldHandle*>(handle);
    const manifold::Manifold residual =
        residual_manifold_for_box_vertices(state->manifold, box_vertices, n_boxes);
    *out_mesh_volume = mesh_output_volume(residual);
    *out_properties_volume = properties_volume(residual);
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" int smart_manifold_best_axis_actions_for_mask(
    void* handle, const double* bounds, const double* rotations,
    const std::uint8_t* bbox_mask, std::size_t n_boxes,
    std::size_t num_action_scale, double action_unit, double volume_sum,
    double last_bbox_score, double cover_penalty, double pen_rate,
    double initial_best, const double* action_scales, std::intptr_t* out_actions,
    double* out_rewards, int volume_method) {
  try {
    const auto* state = static_cast<SmartManifoldHandle*>(handle);
    const std::size_t actions_per_bbox = 6 * num_action_scale + 1;

    std::vector<double> old_volumes(n_boxes, 0.0);
    std::vector<std::uint8_t> current_valid(n_boxes, 0);
    std::vector<manifold::Manifold> current_boxes(n_boxes);
    double total_volume = 0.0;

    for (std::size_t idx = 0; idx < n_boxes; ++idx) {
      const double* box = bounds + idx * 6;
      out_actions[idx] = -1;
      out_rewards[idx] = initial_best;
      if (!bbox_is_valid(box)) {
        continue;
      }
      old_volumes[idx] = bbox_volume(box);
      total_volume += old_volumes[idx];
      current_valid[idx] = 1;
      current_boxes[idx] = box_from_params(box, rotations + idx * 9);
    }

    std::vector<double> candidate_bounds(n_boxes * 6);
    std::copy(bounds, bounds + n_boxes * 6, candidate_bounds.begin());

    for (std::size_t idx = 0; idx < n_boxes; ++idx) {
      if (!bbox_mask[idx]) {
        continue;
      }
      double best_reward = initial_best;
      std::intptr_t best_action = -1;
      double* candidate_box = candidate_bounds.data() + idx * 6;

      for (std::size_t coord_idx = 0; coord_idx < 6; ++coord_idx) {
        for (std::size_t scale_idx = 0; scale_idx < num_action_scale;
             ++scale_idx) {
          const std::intptr_t action = static_cast<std::intptr_t>(
              idx * actions_per_bbox + coord_idx * num_action_scale +
              scale_idx);
          const double original_value = candidate_box[coord_idx];
          candidate_box[coord_idx] += action_scales[scale_idx] * action_unit;

          const bool candidate_valid = bbox_is_valid(candidate_box);
          const double candidate_volume = bbox_volume(candidate_box);
          const double bvs =
              (total_volume - old_volumes[idx] + candidate_volume) /
              volume_sum;
          const double upper_reward = -std::abs(bvs - 1.0) - last_bbox_score;
          if (upper_reward <= best_reward) {
            candidate_box[coord_idx] = original_value;
            continue;
          }

          manifold::Manifold candidate_manifold;
          if (candidate_valid) {
            candidate_manifold =
                box_from_params(candidate_box, rotations + idx * 9);
          }
          const double residual = residual_volume_for_box_set(
              state->manifold, current_boxes, current_valid,
              candidate_valid ? &candidate_manifold : nullptr, candidate_valid,
              idx, volume_sum, volume_method != 0);
          const double covered = 1.0 - residual / volume_sum;
          const double score =
              -std::abs(bvs - 1.0) - (1.0 - covered) * pen_rate * cover_penalty;
          const double reward = score - last_bbox_score;
          if (best_reward < reward) {
            best_reward = reward;
            best_action = action;
          }
          candidate_box[coord_idx] = original_value;
        }
      }

      out_actions[idx] = best_action;
      out_rewards[idx] = best_reward;
    }
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" void* smart_manifold_state_new(
    const float* vertices, std::size_t n_vertices, const std::uint32_t* faces,
    std::size_t n_faces, const double* bounds, const double* rotations,
    std::size_t n_boxes, double volume_sum, double last_bbox_score,
    int stateful_union_cache, std::size_t cache_capacity, int volume_method) {
  try {
    if (volume_sum <= 0.0) {
      return nullptr;
    }
    manifold::Mesh mesh = make_mesh(vertices, n_vertices, faces, n_faces);
    auto* state = new SmartManifoldState();
    state->surface = manifold::Manifold(mesh);
    state->bounds.assign(bounds, bounds + n_boxes * 6);
    state->rotations.assign(rotations, rotations + n_boxes * 9);
    state->volume_sum = volume_sum;
    state->last_bbox_score = last_bbox_score;
    state->stateful_union_cache = stateful_union_cache != 0;
    state->use_properties_volume = volume_method != 0;
    state->cache_capacity = std::max<std::size_t>(1, cache_capacity);
    state_rebuild_boxes(state);
    return state;
  } catch (...) {
    return nullptr;
  }
}

extern "C" void smart_manifold_state_delete(void* handle) {
  delete static_cast<SmartManifoldState*>(handle);
}

extern "C" int smart_manifold_state_reset(
    void* handle, const double* bounds, const double* rotations,
    std::size_t n_boxes, double last_bbox_score) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    state->bounds.assign(bounds, bounds + n_boxes * 6);
    state->rotations.assign(rotations, rotations + n_boxes * 9);
    state->last_bbox_score = last_bbox_score;
    state->history.clear();
    state_rebuild_boxes(state);
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" std::size_t smart_manifold_state_num_boxes(void* handle) {
  const auto* state = static_cast<SmartManifoldState*>(handle);
  return state->bounds.size() / 6;
}

extern "C" int smart_manifold_state_copy(
    void* handle, double* out_bounds, double* out_rotations) {
  try {
    const auto* state = static_cast<SmartManifoldState*>(handle);
    std::copy(state->bounds.begin(), state->bounds.end(), out_bounds);
    std::copy(state->rotations.begin(), state->rotations.end(), out_rotations);
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" int smart_manifold_state_copy_bbox(
    void* handle, std::size_t bbox_idx, double* out_bounds,
    double* out_rotation) {
  try {
    const auto* state = static_cast<SmartManifoldState*>(handle);
    const std::size_t n_boxes = state->bounds.size() / 6;
    if (bbox_idx >= n_boxes) {
      return 0;
    }
    const double* bounds = state->bounds.data() + bbox_idx * 6;
    const double* rotation = state->rotations.data() + bbox_idx * 9;
    std::copy(bounds, bounds + 6, out_bounds);
    std::copy(rotation, rotation + 9, out_rotation);
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" double smart_manifold_state_last_bbox_score(void* handle) {
  try {
    const auto* state = static_cast<SmartManifoldState*>(handle);
    return state->last_bbox_score;
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" double smart_manifold_state_total_bbox_volume(void* handle) {
  try {
    const auto* state = static_cast<SmartManifoldState*>(handle);
    return state->total_volume;
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" std::size_t smart_manifold_state_valid_count(void* handle) {
  const auto* state = static_cast<SmartManifoldState*>(handle);
  std::size_t count = 0;
  for (std::uint8_t valid : state->valid) {
    if (valid) {
      ++count;
    }
  }
  return count;
}

extern "C" int smart_manifold_state_cache_stats(void* handle,
                                                 std::uint64_t* out_values) {
  try {
    const auto* state = static_cast<SmartManifoldState*>(handle);
    out_values[0] = static_cast<std::uint64_t>(state->reward_cache.size());
    out_values[1] = state->reward_cache_hits;
    out_values[2] = state->reward_cache_misses;
    out_values[3] = state->version;
    out_values[4] = state->state_hash;
    out_values[5] = state->except_union_builds;
    out_values[6] = state->except_union_cache_hits;
    out_values[7] = state->ordered_prefix_builds;
    out_values[8] = state->ordered_prefix_cache_hits;
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" double smart_manifold_state_covered(void* handle) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    const double residual = state_current_residual(state);
    return 1.0 - residual / state->volume_sum;
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" double smart_manifold_state_score(void* handle,
                                             double cover_penalty,
                                             double pen_rate) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    const double bvs = state->total_volume / state->volume_sum;
    const double covered = smart_manifold_state_covered(handle);
    return -std::abs(bvs - 1.0) -
           (1.0 - covered) * pen_rate * cover_penalty;
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" double smart_manifold_state_score_axis_action(
    void* handle, std::intptr_t action, std::size_t num_action_scale,
    double action_unit, double cover_penalty, double pen_rate,
    const double* action_scales) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    return state_score_axis_action(state, action, num_action_scale, action_unit,
                                   cover_penalty, pen_rate, action_scales);
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" double smart_manifold_state_score_replacement(
    void* handle, std::size_t bbox_idx, const double* candidate_bounds,
    const double* candidate_rotation, double cover_penalty, double pen_rate) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    return state_score_replacement(state, bbox_idx, candidate_bounds,
                                   candidate_rotation, cover_penalty,
                                   pen_rate);
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" double smart_manifold_state_apply_replacement(
    void* handle, std::size_t bbox_idx, const double* candidate_bounds,
    const double* candidate_rotation, double cover_penalty, double pen_rate) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    const std::size_t n_boxes = state->bounds.size() / 6;
    if (bbox_idx >= n_boxes || candidate_bounds == nullptr ||
        candidate_rotation == nullptr) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double score = state_score_replacement(
        state, bbox_idx, candidate_bounds, candidate_rotation, cover_penalty,
        pen_rate);
    if (!(score == score)) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    state_snapshot(state);
    const double reward = score - state->last_bbox_score;
    double* target_bounds = state->bounds.data() + bbox_idx * 6;
    double* target_rotation = state->rotations.data() + bbox_idx * 9;
    const double old_volume = state->volumes[bbox_idx];
    std::copy(candidate_bounds, candidate_bounds + 6, target_bounds);
    std::copy(candidate_rotation, candidate_rotation + 9, target_rotation);
    state->last_bbox_score = score;
    const bool is_valid = bbox_is_valid(target_bounds);
    state->valid[bbox_idx] = is_valid ? 1 : 0;
    state->volumes[bbox_idx] = is_valid ? bbox_volume(target_bounds) : 0.0;
    state->total_volume += state->volumes[bbox_idx] - old_volume;
    if (is_valid) {
      state->boxes[bbox_idx] = box_from_params(target_bounds, target_rotation);
    } else {
      state->boxes[bbox_idx] = manifold::Manifold();
    }
    state->state_hash =
        state_geometry_hash(state->bounds, state->rotations, state->volume_sum);
    state_clear_caches(state);
    return reward;
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" int smart_manifold_state_select_replacements_for_mask(
    void* handle, const std::uint8_t* bbox_mask,
    const double* candidate_bounds, const double* candidate_rotations,
    std::size_t num_action_scale, double cover_penalty, double pen_rate,
    std::intptr_t* out_actions, double* out_rewards) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    if (bbox_mask == nullptr || candidate_bounds == nullptr ||
        candidate_rotations == nullptr || out_actions == nullptr ||
        out_rewards == nullptr || state->volume_sum <= 0.0 ||
        num_action_scale == 0) {
      return 0;
    }
    const std::size_t n_boxes = state->bounds.size() / 6;
    const std::size_t actions_per_bbox = 6 * num_action_scale + 1;
    const double current_bvs_reward =
        -std::abs(state->total_volume / state->volume_sum - 1.0) -
        state->last_bbox_score;

    for (std::size_t idx = 0; idx < n_boxes; ++idx) {
      if (!bbox_mask[idx] || current_bvs_reward <= out_rewards[idx]) {
        continue;
      }

      const double* replacement_bounds = candidate_bounds + idx * 6;
      const double* replacement_rotation = candidate_rotations + idx * 9;
      const double* current_bounds = state->bounds.data() + idx * 6;
      const double* current_rotation = state->rotations.data() + idx * 9;
      bool unchanged = true;
      for (std::size_t value_idx = 0; value_idx < 6; ++value_idx) {
        unchanged = unchanged &&
                    replacement_bounds[value_idx] == current_bounds[value_idx];
      }
      for (std::size_t value_idx = 0; value_idx < 9; ++value_idx) {
        unchanged = unchanged &&
                    replacement_rotation[value_idx] == current_rotation[value_idx];
      }

      const double score =
          unchanged ? state->last_bbox_score
                    : state_score_replacement(state, idx, replacement_bounds,
                                              replacement_rotation, cover_penalty,
                                              pen_rate);
      if (!(score == score)) {
        continue;
      }
      const double reward = score - state->last_bbox_score;
      if (out_rewards[idx] < reward) {
        out_rewards[idx] = reward;
        out_actions[idx] = static_cast<std::intptr_t>(
            idx * actions_per_bbox + (actions_per_bbox - 1));
      }
    }
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" int smart_manifold_state_best_axis_actions_for_mask(
    void* handle, const std::uint8_t* bbox_mask,
    std::size_t num_action_scale, double action_unit, double cover_penalty,
    double pen_rate, double initial_best, const double* action_scales,
    std::intptr_t* out_actions, double* out_rewards) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    const std::size_t n_boxes = state->bounds.size() / 6;
    const std::size_t actions_per_bbox = 6 * num_action_scale + 1;
    for (std::size_t idx = 0; idx < n_boxes; ++idx) {
      out_actions[idx] = -1;
      out_rewards[idx] = initial_best;
      if (!bbox_mask[idx]) {
        continue;
      }

      double best_reward = initial_best;
      std::intptr_t best_action = -1;
      for (std::size_t coord_idx = 0; coord_idx < 6; ++coord_idx) {
        for (std::size_t scale_idx = 0; scale_idx < num_action_scale;
             ++scale_idx) {
          const std::intptr_t action = static_cast<std::intptr_t>(
              idx * actions_per_bbox + coord_idx * num_action_scale +
              scale_idx);
          double candidate_bounds[6];
          const double* current_bounds = state->bounds.data() + idx * 6;
          std::copy(current_bounds, current_bounds + 6, candidate_bounds);
          candidate_bounds[coord_idx] += action_scales[scale_idx] * action_unit;
          const double candidate_volume = bbox_volume(candidate_bounds);
          const double bvs =
              (state->total_volume - state->volumes[idx] + candidate_volume) /
              state->volume_sum;
          const double upper_reward =
              -std::abs(bvs - 1.0) - state->last_bbox_score;
          if (upper_reward <= best_reward) {
            continue;
          }
          const double score = state_score_axis_action(
              state, action, num_action_scale, action_unit, cover_penalty,
              pen_rate, action_scales);
          if (!(score == score)) {
            continue;
          }
          const double reward = score - state->last_bbox_score;
          if (best_reward < reward) {
            best_reward = reward;
            best_action = action;
          }
        }
      }
      out_actions[idx] = best_action;
      out_rewards[idx] = best_reward;
    }
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" double smart_manifold_state_apply_axis_action(
    void* handle, std::intptr_t action, std::size_t num_action_scale,
    double action_unit, double cover_penalty, double pen_rate,
    const double* action_scales) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    const std::size_t n_boxes = state->bounds.size() / 6;
    const std::size_t actions_per_bbox = 6 * num_action_scale + 1;
    if (action < 0 ||
        static_cast<std::size_t>(action) >= n_boxes * actions_per_bbox) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const std::size_t bbox_idx =
        static_cast<std::size_t>(action) / actions_per_bbox;
    const std::size_t local_action =
        static_cast<std::size_t>(action) % actions_per_bbox;
    if (local_action == actions_per_bbox - 1) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const std::size_t coord_idx = local_action / num_action_scale;
    const std::size_t scale_idx = local_action % num_action_scale;
    const double score = state_score_axis_action(
        state, action, num_action_scale, action_unit, cover_penalty, pen_rate,
        action_scales);
    if (!(score == score)) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    state_snapshot(state);
    const double reward = score - state->last_bbox_score;
    state->bounds[bbox_idx * 6 + coord_idx] +=
        action_scales[scale_idx] * action_unit;
    state->last_bbox_score = score;
    const double* updated_bounds = state->bounds.data() + bbox_idx * 6;
    const double old_volume = state->volumes[bbox_idx];
    const bool is_valid = bbox_is_valid(updated_bounds);
    state->valid[bbox_idx] = is_valid ? 1 : 0;
    state->volumes[bbox_idx] = is_valid ? bbox_volume(updated_bounds) : 0.0;
    state->total_volume += state->volumes[bbox_idx] - old_volume;
    if (is_valid) {
      state->boxes[bbox_idx] =
          box_from_params(updated_bounds, state->rotations.data() + bbox_idx * 9);
    } else {
      state->boxes[bbox_idx] = manifold::Manifold();
    }
    state->state_hash =
        state_geometry_hash(state->bounds, state->rotations, state->volume_sum);
    state_clear_caches(state);
    return reward;
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

extern "C" int smart_manifold_state_greedy_axis_refine_segment(
    void* handle, std::size_t num_action_scale, double action_unit,
    double cover_penalty, double pen_rate, std::size_t max_steps,
    const double* action_scales, std::intptr_t* out_actions,
    double* out_rewards, std::size_t* out_steps, double* out_last_score) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    if (out_steps == nullptr || out_last_score == nullptr ||
        (max_steps > 0 && (out_actions == nullptr || out_rewards == nullptr)) ||
        state->volume_sum <= 0.0 || num_action_scale == 0) {
      return 0;
    }

    std::size_t steps = 0;
    for (; steps < max_steps; ++steps) {
      const std::size_t n_boxes = state->bounds.size() / 6;
      const std::size_t actions_per_bbox = 6 * num_action_scale + 1;
      const double bvs = state->total_volume / state->volume_sum;
      const double bvs_reward = -std::abs(bvs - 1.0) - state->last_bbox_score;
      double best_reward = bvs_reward;
      std::intptr_t best_action = -1;

      for (std::size_t bbox_idx = 0; bbox_idx < n_boxes; ++bbox_idx) {
        for (std::size_t coord_idx = 0; coord_idx < 6; ++coord_idx) {
          for (std::size_t scale_idx = 0; scale_idx < num_action_scale;
               ++scale_idx) {
            const std::intptr_t action = static_cast<std::intptr_t>(
                bbox_idx * actions_per_bbox +
                coord_idx * num_action_scale + scale_idx);

            double candidate_bounds[6];
            const double* current_bounds =
                state->bounds.data() + bbox_idx * 6;
            std::copy(current_bounds, current_bounds + 6, candidate_bounds);
            candidate_bounds[coord_idx] +=
                action_scales[scale_idx] * action_unit;

            const double candidate_volume = bbox_volume(candidate_bounds);
            const double upper_bvs =
                (state->total_volume - state->volumes[bbox_idx] +
                 candidate_volume) /
                state->volume_sum;
            const double upper_reward =
                -std::abs(upper_bvs - 1.0) - state->last_bbox_score;
            if (upper_reward <= best_reward) {
              continue;
            }

            const double score = state_score_axis_action(
                state, action, num_action_scale, action_unit, cover_penalty,
                pen_rate, action_scales);
            if (!(score == score)) {
              continue;
            }
            const double reward = score - state->last_bbox_score;
            if (best_reward < reward) {
              best_reward = reward;
              best_action = action;
            }
          }
        }
      }

      if (best_action < 0 || best_reward <= 0.0) {
        break;
      }
      const double reward = smart_manifold_state_apply_axis_action(
          handle, best_action, num_action_scale, action_unit, cover_penalty,
          pen_rate, action_scales);
      if (!(reward == reward) || reward <= 0.0) {
        break;
      }
      out_actions[steps] = best_action;
      out_rewards[steps] = reward;
    }

    *out_steps = steps;
    *out_last_score = state->last_bbox_score;
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" int smart_manifold_state_greedy_axis_rollout_step(
    void* handle, const std::uint8_t* bbox_mask,
    std::size_t num_action_scale, double action_unit, double cover_penalty,
    double pen_rate, const double* action_scales, std::uint8_t* out_next_mask,
    std::intptr_t* out_action, double* out_best_reward,
    double* out_applied_reward, double* out_last_score) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    if (bbox_mask == nullptr || out_next_mask == nullptr ||
        out_action == nullptr || out_best_reward == nullptr ||
        out_applied_reward == nullptr || out_last_score == nullptr ||
        state->volume_sum <= 0.0 || num_action_scale == 0) {
      return 0;
    }

    const std::size_t n_boxes = state->bounds.size() / 6;
    const std::size_t actions_per_bbox = 6 * num_action_scale + 1;
    std::intptr_t best_action = -1;
    double best_reward = -std::numeric_limits<double>::max();

    for (std::size_t bbox_idx = 0; bbox_idx < n_boxes; ++bbox_idx) {
      out_next_mask[bbox_idx] = 0;
      if (!bbox_mask[bbox_idx]) {
        continue;
      }

      std::intptr_t bbox_best_action = -1;
      double bbox_best_reward = -std::numeric_limits<double>::max();
      for (std::size_t coord_idx = 0; coord_idx < 6; ++coord_idx) {
        for (std::size_t scale_idx = 0; scale_idx < num_action_scale;
             ++scale_idx) {
          const std::intptr_t action = static_cast<std::intptr_t>(
              bbox_idx * actions_per_bbox +
              coord_idx * num_action_scale + scale_idx);

          double candidate_bounds[6];
          const double* current_bounds = state->bounds.data() + bbox_idx * 6;
          std::copy(current_bounds, current_bounds + 6, candidate_bounds);
          candidate_bounds[coord_idx] +=
              action_scales[scale_idx] * action_unit;
          const double candidate_volume = bbox_volume(candidate_bounds);
          const double bvs =
              (state->total_volume - state->volumes[bbox_idx] +
               candidate_volume) /
              state->volume_sum;
          const double upper_reward =
              -std::abs(bvs - 1.0) - state->last_bbox_score;
          if (upper_reward <= bbox_best_reward) {
            continue;
          }

          const double score = state_score_axis_action(
              state, action, num_action_scale, action_unit, cover_penalty,
              pen_rate, action_scales);
          if (!(score == score)) {
            continue;
          }
          const double reward = score - state->last_bbox_score;
          if (bbox_best_reward < reward) {
            bbox_best_reward = reward;
            bbox_best_action = action;
          }
        }
      }

      if (bbox_best_reward >= 0.0) {
        out_next_mask[bbox_idx] = 1;
      }
      if (best_reward < bbox_best_reward) {
        best_reward = bbox_best_reward;
        best_action = bbox_best_action;
      }
    }

    double applied_reward = 0.0;
    if (best_action >= 0 && best_reward > 0.0) {
      applied_reward = smart_manifold_state_apply_axis_action(
          handle, best_action, num_action_scale, action_unit, cover_penalty,
          pen_rate, action_scales);
      if (!(applied_reward == applied_reward)) {
        applied_reward = std::numeric_limits<double>::quiet_NaN();
      }
    }

    *out_action = best_action;
    *out_best_reward = best_reward;
    *out_applied_reward = applied_reward;
    *out_last_score = state->last_bbox_score;
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" int smart_manifold_state_greedy_axis_rollout_segment(
    void* handle, const std::uint8_t* bbox_mask,
    std::size_t num_action_scale, double action_unit, double cover_penalty,
    double pen_rate, std::size_t max_steps, const double* action_scales,
    std::intptr_t* out_actions, double* out_best_rewards,
    double* out_applied_rewards, std::uint8_t* out_next_mask,
    std::size_t* out_steps, double* out_last_score) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    if (bbox_mask == nullptr || out_next_mask == nullptr ||
        out_steps == nullptr || out_last_score == nullptr ||
        (max_steps > 0 &&
         (out_actions == nullptr || out_best_rewards == nullptr ||
          out_applied_rewards == nullptr)) ||
        state->volume_sum <= 0.0 || num_action_scale == 0) {
      return 0;
    }

    const std::size_t n_boxes = state->bounds.size() / 6;
    std::vector<std::uint8_t> current_mask(bbox_mask, bbox_mask + n_boxes);
    std::vector<std::uint8_t> next_mask(n_boxes, 0);
    std::size_t steps = 0;

    for (; steps < max_steps; ++steps) {
      std::intptr_t action = -1;
      double best_reward = -std::numeric_limits<double>::max();
      double applied_reward = 0.0;
      double last_score = state->last_bbox_score;
      const int ok = smart_manifold_state_greedy_axis_rollout_step(
          handle, current_mask.data(), num_action_scale, action_unit,
          cover_penalty, pen_rate, action_scales, next_mask.data(), &action,
          &best_reward, &applied_reward, &last_score);
      if (ok != 1) {
        return 0;
      }
      if (action < 0 || best_reward <= 0.0 || !(applied_reward == applied_reward) ||
          applied_reward <= 0.0) {
        current_mask = next_mask;
        break;
      }
      out_actions[steps] = action;
      out_best_rewards[steps] = best_reward;
      out_applied_rewards[steps] = applied_reward;
      current_mask = next_mask;
    }

    std::copy(current_mask.begin(), current_mask.end(), out_next_mask);
    *out_steps = steps;
    *out_last_score = state->last_bbox_score;
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" int smart_manifold_state_rollback(void* handle) {
  try {
    auto* state = static_cast<SmartManifoldState*>(handle);
    if (state->history.empty()) {
      return 0;
    }
    SmartManifoldSnapshot snapshot = std::move(state->history.back());
    state->history.pop_back();
    state->bounds = std::move(snapshot.bounds);
    state->rotations = std::move(snapshot.rotations);
    state->boxes = std::move(snapshot.boxes);
    state->valid = std::move(snapshot.valid);
    state->volumes = std::move(snapshot.volumes);
    state->total_volume = snapshot.total_volume;
    state->last_bbox_score = snapshot.last_bbox_score;
    state->state_hash = snapshot.state_hash;
    state_clear_caches(state);
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" float smart_manifold_mesh_volume(const float* vertices,
                                             std::size_t n_vertices,
                                             const std::uint32_t* faces,
                                             std::size_t n_faces) {
  try {
    manifold::Mesh mesh = make_mesh(vertices, n_vertices, faces, n_faces);
    const manifold::Manifold man(mesh);
    return signed_mesh_volume(man.GetMesh());
  } catch (...) {
    return std::numeric_limits<float>::quiet_NaN();
  }
}

extern "C" float smart_manifold_axis_box_intersection_volume(
    const float* vertices, std::size_t n_vertices, const std::uint32_t* faces,
    std::size_t n_faces, float lx, float ly, float lz, float rx, float ry,
    float rz) {
  try {
    manifold::Mesh mesh = make_mesh(vertices, n_vertices, faces, n_faces);
    const manifold::Manifold man(mesh);
    const manifold::Manifold box =
        manifold::Manifold::Cube(glm::vec3(rx - lx, ry - ly, rz - lz), false)
            .Translate(glm::vec3(lx, ly, lz));
    const manifold::Manifold intersection = box ^ man;
    return signed_mesh_volume(intersection.GetMesh());
  } catch (...) {
    return std::numeric_limits<float>::quiet_NaN();
  }
}
