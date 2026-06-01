#include "smart_native_engine.hpp"

#include "smart_native_core.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <sys/stat.h>

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
extern "C" double smart_manifold_state_last_bbox_score(void* handle);
extern "C" double smart_manifold_state_score(void* handle,
                                             double cover_penalty,
                                             double pen_rate);
extern "C" double smart_manifold_state_apply_axis_action(
    void* handle, std::intptr_t action, std::size_t num_action_scale,
    double action_unit, double cover_penalty, double pen_rate,
    const double* action_scales);
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
extern "C" int smart_manifold_state_greedy_axis_refine_segment(
    void* handle, std::size_t num_action_scale, double action_unit,
    double cover_penalty, double pen_rate, std::size_t max_steps,
    const double* action_scales, std::intptr_t* out_actions,
    double* out_rewards, std::size_t* out_steps, double* out_last_score,
    std::size_t* out_exact_checks);
extern "C" int smart_manifold_state_greedy_axis_rollout_segment(
    void* handle, const std::uint8_t* bbox_mask,
    std::size_t num_action_scale, double action_unit, double cover_penalty,
    double pen_rate, std::size_t max_steps, const double* action_scales,
    std::intptr_t* out_actions, double* out_best_rewards,
    double* out_applied_rewards, std::uint8_t* out_next_mask,
    std::size_t* out_steps, double* out_last_score);

namespace smart_native {
namespace {

struct GmshMesh {
  std::vector<double> vertices;
  std::vector<std::size_t> faces;
  std::vector<std::size_t> voxels;
};

struct BBoxParams {
  std::vector<double> bounds;
  std::vector<double> rotations;
};

struct BoxFit {
  std::vector<double> bounds;
  std::vector<double> rotation;
  double volume = 0.0;
  bool valid = false;
};

struct MergeDeltaKey {
  double delta = 0.0;
  std::size_t left = 0;
  std::size_t right = 0;

  static MergeDeltaKey target(double value) {
    return MergeDeltaKey{value, 0, 0};
  }

  bool operator<(const MergeDeltaKey& other) const {
    if (delta != other.delta) return delta < other.delta;
    if (left != other.left) return left < other.left;
    return right < other.right;
  }
};

struct PartitionCandidate {
  double reward = -std::numeric_limits<double>::infinity();
  double delta = 0.0;
  std::size_t left = 0;
  std::size_t right = 0;
  std::uint64_t left_version = 0;
  std::uint64_t right_version = 0;
  BoxFit fit;
  std::vector<std::array<double, 3>> points;
  std::vector<std::size_t> partition;
};

struct RecenterCandidate {
  bool valid = false;
  std::size_t bbox_idx = 0;
  std::vector<double> bounds;
  std::vector<double> rotation;
};

struct StateDelete {
  void operator()(void* handle) const {
    if (handle != nullptr) {
      smart_manifold_state_delete(handle);
    }
  }
};

using NativeStatePtr = std::unique_ptr<void, StateDelete>;

std::string read_text(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open file: " + path);
  }
  std::ostringstream stream;
  stream << input.rdbuf();
  return stream.str();
}

int volume_method_id(const std::string& method) {
  if (method == "mesh") return 0;
  if (method == "properties" || method == "get_properties") return 1;
  throw std::runtime_error("unsupported volume method: " + method);
}

void mkdir_if_missing(const std::string& path) {
  if (path.empty()) return;
  if (::mkdir(path.c_str(), 0777) == 0 || errno == EEXIST) return;
  throw std::runtime_error("failed to create directory: " + path);
}

void ensure_directories(const std::string& directory) {
  if (directory.empty()) return;
  std::string current;
  current.reserve(directory.size());
  for (std::size_t idx = 0; idx < directory.size(); ++idx) {
    const char ch = directory[idx];
    current.push_back(ch);
    if ((ch == '/' || ch == '\\') && current.size() > 1) {
      mkdir_if_missing(current);
    }
  }
  mkdir_if_missing(current);
}

std::string path_join(const std::string& directory, const std::string& file) {
  if (directory.empty()) return file;
  const char last = directory[directory.size() - 1];
  if (last == '/' || last == '\\') return directory + file;
  return directory + "/" + file;
}

bool file_exists(const std::string& path) {
  struct stat info {};
  return ::stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

std::vector<double> parse_number_array_at(const std::string& text,
                                          std::size_t open_bracket) {
  if (open_bracket == std::string::npos || text[open_bracket] != '[') {
    throw std::runtime_error("expected JSON number array");
  }
  const std::size_t close = text.find(']', open_bracket + 1);
  if (close == std::string::npos) {
    throw std::runtime_error("unterminated JSON number array");
  }
  std::vector<double> values;
  const char* cursor = text.c_str() + open_bracket + 1;
  const char* end = text.c_str() + close;
  while (cursor < end) {
    while (cursor < end &&
           (*cursor == ' ' || *cursor == '\n' || *cursor == '\t' ||
            *cursor == '\r' || *cursor == ',')) {
      ++cursor;
    }
    if (cursor >= end) break;
    char* parsed_end = nullptr;
    const double value = std::strtod(cursor, &parsed_end);
    if (parsed_end == cursor) {
      throw std::runtime_error("failed to parse JSON number array");
    }
    values.push_back(value);
    cursor = parsed_end;
  }
  return values;
}

std::vector<double> load_number_array_file(const std::string& path) {
  if (path.empty()) return {};
  const std::string text = read_text(path);
  const std::size_t open = text.find('[');
  if (open == std::string::npos) {
    throw std::runtime_error("expected JSON number array in: " + path);
  }
  return parse_number_array_at(text, open);
}

BBoxParams load_bbox_params_file(const std::string& path) {
  const std::string text = read_text(path);
  BBoxParams params;
  std::size_t search = 0;
  const std::vector<double> identity = {1.0, 0.0, 0.0, 0.0, 1.0,
                                        0.0, 0.0, 0.0, 1.0};
  while (true) {
    const std::size_t bounds_key = text.find("\"bounds\"", search);
    if (bounds_key == std::string::npos) break;
    const std::size_t bounds_open = text.find('[', bounds_key);
    std::vector<double> bounds = parse_number_array_at(text, bounds_open);
    if (bounds.size() != 6) {
      throw std::runtime_error("bbox bounds must contain 6 numbers");
    }
    const std::size_t next_bounds = text.find("\"bounds\"", bounds_open + 1);
    const std::size_t rotation_key = text.find("\"rotation\"", bounds_open + 1);
    std::vector<double> rotation = identity;
    if (rotation_key != std::string::npos &&
        (next_bounds == std::string::npos || rotation_key < next_bounds)) {
      rotation = parse_number_array_at(text, text.find('[', rotation_key));
      if (rotation.size() != 9) rotation = identity;
    }
    params.bounds.insert(params.bounds.end(), bounds.begin(), bounds.end());
    params.rotations.insert(params.rotations.end(), rotation.begin(), rotation.end());
    search = bounds_open + 1;
  }
  if (params.bounds.empty()) {
    throw std::runtime_error("no bbox params found in: " + path);
  }
  return params;
}

std::vector<std::vector<std::size_t>> load_partitions_file(
    const std::string& path) {
  const std::string text = read_text(path);
  std::size_t start = std::string::npos;
  const std::size_t key = text.find("\"partitions\"");
  if (key != std::string::npos) {
    start = text.find('[', key);
  } else {
    start = text.find('[');
  }
  if (start == std::string::npos) {
    throw std::runtime_error("no partitions array found in: " + path);
  }
  std::vector<std::vector<std::size_t>> partitions;
  std::vector<std::size_t> current;
  int depth = 0;
  bool in_number = false;
  const char* number_start = nullptr;
  const char* base = text.c_str();
  for (std::size_t idx = start; idx < text.size(); ++idx) {
    const char ch = text[idx];
    if (ch == '[') {
      ++depth;
      if (depth == 2) current.clear();
      continue;
    }
    if (ch == ']') {
      if (in_number && depth == 2) {
        char* parsed_end = nullptr;
        const long value = std::strtol(number_start, &parsed_end, 10);
        if (parsed_end != number_start && value >= 0) {
          current.push_back(static_cast<std::size_t>(value));
        }
        in_number = false;
      }
      if (depth == 2 && !current.empty()) {
        std::sort(current.begin(), current.end());
        current.erase(std::unique(current.begin(), current.end()), current.end());
        partitions.push_back(current);
        current.clear();
      }
      --depth;
      if (depth <= 0) break;
      continue;
    }
    if (depth == 2 && (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-')) {
      if (!in_number) {
        in_number = true;
        number_start = base + idx;
      }
      continue;
    }
    if (in_number && depth == 2) {
      char* parsed_end = nullptr;
      const long value = std::strtol(number_start, &parsed_end, 10);
      if (parsed_end != number_start && value >= 0) {
        current.push_back(static_cast<std::size_t>(value));
      }
      in_number = false;
    }
  }
  if (partitions.empty()) {
    throw std::runtime_error("no valid partitions found in: " + path);
  }
  return partitions;
}

GmshMesh load_gmsh_mesh(const std::string& path) {
  std::size_t n_vertices = 0;
  std::size_t n_faces = 0;
  std::size_t n_voxels = 0;
  if (!smart_native_load_gmsh_counts(
          path.c_str(), &n_vertices, &n_faces, &n_voxels)) {
    throw std::runtime_error("failed to read Gmsh counts: " + path);
  }
  GmshMesh mesh;
  mesh.vertices.resize(n_vertices * 3);
  mesh.faces.resize(n_faces * 3);
  mesh.voxels.resize(n_voxels * 4);
  std::size_t out_vertices = 0;
  std::size_t out_faces = 0;
  std::size_t out_voxels = 0;
  if (!smart_native_load_gmsh(
          path.c_str(), mesh.vertices.data(), mesh.faces.data(),
          mesh.voxels.data(), mesh.vertices.size(), mesh.faces.size(),
          mesh.voxels.size(), &out_vertices, &out_faces, &out_voxels)) {
    throw std::runtime_error("failed to load Gmsh file: " + path);
  }
  mesh.vertices.resize(out_vertices * 3);
  mesh.faces.resize(out_faces * 3);
  mesh.voxels.resize(out_voxels * 4);
  return mesh;
}

int parse_obj_vertex_index(const std::string& token, std::size_t n_vertices) {
  const std::size_t slash = token.find('/');
  const std::string raw_token =
      slash == std::string::npos ? token : token.substr(0, slash);
  if (raw_token.empty()) {
    throw std::runtime_error("empty OBJ face vertex index");
  }
  char* end = nullptr;
  const long raw = std::strtol(raw_token.c_str(), &end, 10);
  if (end == raw_token.c_str() || *end != '\0') {
    throw std::runtime_error("invalid OBJ face vertex index: " + token);
  }
  if (raw > 0) {
    return static_cast<int>(raw - 1);
  }
  if (raw < 0) {
    return static_cast<int>(n_vertices) + static_cast<int>(raw);
  }
  throw std::runtime_error("OBJ face index cannot be 0");
}

GmshMesh load_obj_surface_as_gmsh(const std::string& path,
                                  const GmshMesh& tetra_mesh) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open OBJ surface: " + path);
  }
  GmshMesh surface;
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
        throw std::runtime_error("malformed OBJ vertex line in: " + path);
      }
      surface.vertices.push_back(x);
      surface.vertices.push_back(y);
      surface.vertices.push_back(z);
    } else if (line.rfind("f ", 0) == 0) {
      std::istringstream stream(line);
      std::string tag;
      stream >> tag;
      std::vector<std::size_t> indices;
      std::string token;
      const std::size_t n_vertices = surface.vertices.size() / 3;
      while (stream >> token) {
        const int parsed = parse_obj_vertex_index(token, n_vertices);
        if (parsed < 0 || static_cast<std::size_t>(parsed) >= n_vertices) {
          throw std::runtime_error("OBJ face index out of range in: " + path);
        }
        indices.push_back(static_cast<std::size_t>(parsed));
      }
      for (std::size_t tri = 1; tri + 1 < indices.size(); ++tri) {
        surface.faces.push_back(indices[0]);
        surface.faces.push_back(indices[tri]);
        surface.faces.push_back(indices[tri + 1]);
      }
    }
  }
  if (surface.vertices.empty() || surface.faces.empty()) {
    throw std::runtime_error("OBJ surface has no usable triangles: " + path);
  }
  surface.voxels = tetra_mesh.voxels;
  return surface;
}

GmshMesh scoring_surface_mesh(const std::string& msh_path,
                              const GmshMesh& tetra_mesh) {
  const std::string surface_path = msh_path + "__sf.obj";
  if (file_exists(surface_path)) {
    return load_obj_surface_as_gmsh(surface_path, tetra_mesh);
  }
  return tetra_mesh;
}

std::vector<float> vertices_to_float(const std::vector<double>& vertices) {
  std::vector<float> out;
  out.reserve(vertices.size());
  for (double value : vertices) {
    out.push_back(static_cast<float>(value));
  }
  return out;
}

std::vector<std::uint32_t> faces_to_uint32(const std::vector<std::size_t>& faces) {
  std::vector<std::uint32_t> out;
  out.reserve(faces.size());
  for (std::size_t value : faces) {
    out.push_back(static_cast<std::uint32_t>(value));
  }
  return out;
}

double tetra_volume_sum(const GmshMesh& mesh) {
  if (mesh.voxels.empty()) return 1.0;
  std::vector<double> volumes(mesh.voxels.size() / 4, 0.0);
  if (!smart_native_tetra_volumes(
          mesh.vertices.data(), mesh.vertices.size() / 3, mesh.voxels.data(),
          mesh.voxels.size() / 4, volumes.data())) {
    throw std::runtime_error("failed to compute tetra volumes");
  }
  double total = 0.0;
  for (double value : volumes) total += value;
  return total > 0.0 ? total : 1.0;
}

std::vector<double> tetra_centroids(const GmshMesh& mesh) {
  std::vector<double> centroids((mesh.voxels.size() / 4) * 3, 0.0);
  if (!mesh.voxels.empty() &&
      !smart_native_tetra_centroids(mesh.vertices.data(),
                                    mesh.vertices.size() / 3,
                                    mesh.voxels.data(),
                                    mesh.voxels.size() / 4,
                                    centroids.data())) {
    throw std::runtime_error("failed to compute tetra centroids");
  }
  return centroids;
}

double bbox_volume_row(const std::vector<double>& bounds) {
  if (bounds.size() != 6 || bounds[0] >= bounds[3] ||
      bounds[1] >= bounds[4] || bounds[2] >= bounds[5]) {
    return 0.0;
  }
  return (bounds[3] - bounds[0]) * (bounds[4] - bounds[1]) *
         (bounds[5] - bounds[2]);
}

double reward_for_merge_delta(double prev_bvs, double delta) {
  return -std::abs(prev_bvs + delta - 1.0) + std::abs(prev_bvs - 1.0);
}

std::array<double, 3> vertex_at(const GmshMesh& mesh, std::size_t index) {
  const std::size_t base = index * 3;
  if (base + 2 >= mesh.vertices.size()) return {0.0, 0.0, 0.0};
  return {mesh.vertices[base], mesh.vertices[base + 1], mesh.vertices[base + 2]};
}

std::vector<std::array<double, 3>> points_for_partition(
    const GmshMesh& mesh, const std::vector<std::size_t>& partition) {
  std::set<std::array<double, 3>> unique;
  const std::size_t n_voxels = mesh.voxels.size() / 4;
  for (std::size_t voxel_idx : partition) {
    if (voxel_idx >= n_voxels) continue;
    for (std::size_t local = 0; local < 4; ++local) {
      const std::size_t vertex_idx = mesh.voxels[voxel_idx * 4 + local];
      const std::size_t base = vertex_idx * 3;
      if (base + 2 >= mesh.vertices.size()) continue;
      const std::array<double, 3> point = {
          mesh.vertices[base], mesh.vertices[base + 1], mesh.vertices[base + 2]};
      if (std::isfinite(point[0]) && std::isfinite(point[1]) &&
          std::isfinite(point[2])) {
        unique.insert(point);
      }
    }
  }
  return {unique.begin(), unique.end()};
}

std::vector<std::array<double, 3>> unique_point_union(
    const std::vector<std::array<double, 3>>& left,
    const std::vector<std::array<double, 3>>& right) {
  std::set<std::array<double, 3>> unique;
  unique.insert(left.begin(), left.end());
  unique.insert(right.begin(), right.end());
  return {unique.begin(), unique.end()};
}

double dot3(const std::array<double, 3>& left,
            const std::array<double, 3>& right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

std::array<double, 3> cross3(const std::array<double, 3>& left,
                             const std::array<double, 3>& right) {
  return {left[1] * right[2] - left[2] * right[1],
          left[2] * right[0] - left[0] * right[2],
          left[0] * right[1] - left[1] * right[0]};
}

bool normalize3(std::array<double, 3>& value) {
  const double norm = std::sqrt(dot3(value, value));
  if (!std::isfinite(norm) || norm <= 1e-12) return false;
  value[0] /= norm;
  value[1] /= norm;
  value[2] /= norm;
  return true;
}

void canonicalize_axis_sign(std::array<double, 3>& axis) {
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

bool points_have_area(const std::vector<std::array<double, 3>>& points) {
  if (points.size() < 4) return false;
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
    if (mx[axis] - mn[axis] > 1e-9) ++varying_axes;
  }
  return varying_axes >= 2;
}

std::array<std::array<double, 3>, 3> pca_rotation_rows(
    const std::vector<std::array<double, 3>>& points) {
  std::array<std::array<double, 3>, 3> rows = {{
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
  }};
  if (!points_have_area(points)) return rows;
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
    if (max_offdiag < 1e-12) break;
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
    if (!normalize3(rows[row])) {
      return {{
          {1.0, 0.0, 0.0},
          {0.0, 1.0, 0.0},
          {0.0, 0.0, 1.0},
      }};
    }
    canonicalize_axis_sign(rows[row]);
  }
  rows[2] = cross3(rows[0], rows[1]);
  if (!normalize3(rows[2])) {
    return {{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
  }
  rows[1] = cross3(rows[2], rows[0]);
  normalize3(rows[1]);
  canonicalize_axis_sign(rows[0]);
  canonicalize_axis_sign(rows[1]);
  canonicalize_axis_sign(rows[2]);
  return rows;
}

std::vector<double> flatten_rotation_rows(
    const std::array<std::array<double, 3>, 3>& rows) {
  return {rows[0][0], rows[0][1], rows[0][2],
          rows[1][0], rows[1][1], rows[1][2],
          rows[2][0], rows[2][1], rows[2][2]};
}

BoxFit axis_fit_from_points(const std::vector<std::array<double, 3>>& points) {
  BoxFit fit;
  if (points.empty()) return fit;
  std::array<double, 3> mn = points.front();
  std::array<double, 3> mx = points.front();
  for (const auto& point : points) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      mn[axis] = std::min(mn[axis], point[axis]);
      mx[axis] = std::max(mx[axis], point[axis]);
    }
  }
  fit.bounds = {mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]};
  fit.rotation = {1.0, 0.0, 0.0, 0.0, 1.0,
                  0.0, 0.0, 0.0, 1.0};
  fit.volume = bbox_volume_row(fit.bounds);
  fit.valid = fit.volume > 0.0;
  return fit;
}

BoxFit box_fit_from_points(const std::vector<std::array<double, 3>>& points,
                           bool tilted) {
  if (!tilted || !points_have_area(points)) return axis_fit_from_points(points);
  const auto rows = pca_rotation_rows(points);
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
        dot3(point, rows[0]), dot3(point, rows[1]), dot3(point, rows[2])};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      mn[axis] = std::min(mn[axis], local[axis]);
      mx[axis] = std::max(mx[axis], local[axis]);
    }
  }
  BoxFit fit;
  fit.bounds = {mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]};
  fit.rotation = flatten_rotation_rows(rows);
  fit.volume = bbox_volume_row(fit.bounds);
  fit.valid = fit.volume > 0.0;
  if (!fit.valid) return axis_fit_from_points(points);
  return fit;
}

bool point_in_oriented_bounds(const double* point,
                              const std::vector<double>& bounds,
                              const std::vector<double>& rotation) {
  if (bounds.size() != 6 || rotation.size() != 9 || point == nullptr) {
    return false;
  }
  const double x = point[0] * rotation[0] + point[1] * rotation[1] +
                   point[2] * rotation[2];
  const double y = point[0] * rotation[3] + point[1] * rotation[4] +
                   point[2] * rotation[5];
  const double z = point[0] * rotation[6] + point[1] * rotation[7] +
                   point[2] * rotation[8];
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z) &&
         bounds[0] <= x && x <= bounds[3] &&
         bounds[1] <= y && y <= bounds[4] &&
         bounds[2] <= z && z <= bounds[5];
}

std::vector<std::array<double, 3>> recenter_points_from_tets(
    const GmshMesh& mesh,
    const std::vector<double>& centroids,
    const std::vector<double>& bounds,
    const std::vector<double>& rotation) {
  std::vector<std::array<double, 3>> selected;
  const std::size_t n_voxels = mesh.voxels.size() / 4;
  if (centroids.size() != n_voxels * 3) return selected;
  selected.reserve(n_voxels * 4);
  for (std::size_t voxel_idx = 0; voxel_idx < n_voxels; ++voxel_idx) {
    const double* centroid = centroids.data() + voxel_idx * 3;
    if (!point_in_oriented_bounds(centroid, bounds, rotation)) continue;
    for (std::size_t corner = 0; corner < 4; ++corner) {
      const std::size_t vertex_idx = mesh.voxels[voxel_idx * 4 + corner];
      const std::size_t base = vertex_idx * 3;
      if (base + 2 >= mesh.vertices.size()) continue;
      const std::array<double, 3> point = {
          mesh.vertices[base], mesh.vertices[base + 1], mesh.vertices[base + 2]};
      if (std::isfinite(point[0]) && std::isfinite(point[1]) &&
          std::isfinite(point[2])) {
        selected.push_back(point);
      }
    }
  }
  return selected;
}

RecenterCandidate recenter_candidate_from_points(
    std::size_t bbox_idx,
    const std::vector<double>& bounds,
    const std::vector<double>& rotation,
    const std::vector<std::array<double, 3>>& points) {
  RecenterCandidate candidate;
  candidate.bbox_idx = bbox_idx;
  if (bounds.size() != 6 || rotation.size() != 9 ||
      !points_have_area(points)) {
    return candidate;
  }
  const auto rows = pca_rotation_rows(points);
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
        dot3(point, rows[0]), dot3(point, rows[1]), dot3(point, rows[2])};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      mn[axis] = std::min(mn[axis], local[axis]);
      mx[axis] = std::max(mx[axis], local[axis]);
    }
  }
  std::vector<double> out_bounds = bounds;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const double center = 0.5 * (mn[axis] + mx[axis]);
    const double current_center = 0.5 * (bounds[axis] + bounds[axis + 3]);
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
  candidate.rotation = flatten_rotation_rows(rows);
  return candidate;
}

std::vector<std::vector<std::size_t>> build_partition_adjacency_pairs(
    const GmshMesh& mesh,
    const std::vector<std::vector<std::size_t>>& partitions,
    bool only_nearby) {
  const std::size_t n = partitions.size();
  std::set<std::pair<std::size_t, std::size_t>> pairs;
  if (n < 2) return {};
  if (!only_nearby) {
    for (std::size_t left = 0; left < n; ++left) {
      if (partitions[left].empty()) continue;
      for (std::size_t right = left + 1; right < n; ++right) {
        if (!partitions[right].empty()) pairs.emplace(left, right);
      }
    }
  } else {
    const std::size_t invalid = std::numeric_limits<std::size_t>::max();
    const std::size_t n_voxels = mesh.voxels.size() / 4;
    std::vector<std::size_t> voxel_partition(n_voxels, invalid);
    for (std::size_t part_idx = 0; part_idx < n; ++part_idx) {
      for (std::size_t voxel_idx : partitions[part_idx]) {
        if (voxel_idx < voxel_partition.size()) {
          voxel_partition[voxel_idx] = part_idx;
        }
      }
    }
    std::map<std::array<std::size_t, 3>, std::size_t> owner_by_face;
    static const std::size_t local_faces[4][3] = {
        {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
    for (std::size_t voxel_idx = 0; voxel_idx < n_voxels; ++voxel_idx) {
      const std::size_t part_idx = voxel_partition[voxel_idx];
      if (part_idx == invalid) continue;
      for (const auto& face : local_faces) {
        std::array<std::size_t, 3> key = {
            mesh.voxels[voxel_idx * 4 + face[0]],
            mesh.voxels[voxel_idx * 4 + face[1]],
            mesh.voxels[voxel_idx * 4 + face[2]]};
        std::sort(key.begin(), key.end());
        auto inserted = owner_by_face.emplace(key, part_idx);
        if (!inserted.second && inserted.first->second != part_idx) {
          std::size_t left = inserted.first->second;
          std::size_t right = part_idx;
          if (left > right) std::swap(left, right);
          pairs.emplace(left, right);
        }
      }
    }
  }
  std::vector<std::vector<std::size_t>> out;
  out.reserve(pairs.size());
  for (const auto& pair : pairs) out.push_back({pair.first, pair.second});
  return out;
}

std::vector<double> action_scales(std::size_t num_action_scale) {
  std::vector<double> scales(num_action_scale, 0.0);
  if (!smart_native_action_scales(num_action_scale, scales.data())) {
    throw std::runtime_error("failed to build native action scales");
  }
  return scales;
}

NativeStatePtr make_state(const GmshMesh& mesh,
                          const BBoxParams& params,
                          double volume_sum,
                          double last_bbox_score,
                          const NativeSearchConfig& config) {
  std::vector<float> vertices = vertices_to_float(mesh.vertices);
  std::vector<std::uint32_t> faces = faces_to_uint32(mesh.faces);
  const std::size_t n_boxes = params.bounds.size() / 6;
  NativeStatePtr state(smart_manifold_state_new(
      vertices.data(), vertices.size() / 3, faces.data(), faces.size() / 3,
      params.bounds.data(), params.rotations.data(), n_boxes, volume_sum,
      last_bbox_score, config.stateful_union_cache ? 1 : 0,
      config.cache_capacity, volume_method_id(config.volume_method)));
  if (!state) {
    throw std::runtime_error("failed to create native Manifold state");
  }
  return state;
}

void reset_state(void* state, const BBoxParams& params, double score) {
  if (!smart_manifold_state_reset(state, params.bounds.data(),
                                  params.rotations.data(),
                                  params.bounds.size() / 6, score)) {
    throw std::runtime_error("failed to reset native Manifold state");
  }
}

BBoxParams copy_state(void* state) {
  const std::size_t n_boxes = smart_manifold_state_num_boxes(state);
  BBoxParams params;
  params.bounds.resize(n_boxes * 6);
  params.rotations.resize(n_boxes * 9);
  if (!smart_manifold_state_copy(state, params.bounds.data(),
                                 params.rotations.data())) {
    throw std::runtime_error("failed to copy native Manifold state");
  }
  return params;
}

std::vector<double> oriented_box_vertices(const BBoxParams& params) {
  const std::size_t n_boxes = params.bounds.size() / 6;
  std::vector<double> out;
  out.reserve(n_boxes * 8 * 3);
  for (std::size_t box_idx = 0; box_idx < n_boxes; ++box_idx) {
    const double* row = params.bounds.data() + box_idx * 6;
    const double* rot = params.rotations.data() + box_idx * 9;
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
          out.push_back(base[0] + rot[0] * i * lengths[0] +
                        rot[3] * j * lengths[1] + rot[6] * k * lengths[2]);
          out.push_back(base[1] + rot[1] * i * lengths[0] +
                        rot[4] * j * lengths[1] + rot[7] * k * lengths[2]);
          out.push_back(base[2] + rot[2] * i * lengths[0] +
                        rot[5] * j * lengths[1] + rot[8] * k * lengths[2]);
        }
      }
    }
  }
  return out;
}

void write_json_number_array(std::ostream& output,
                             const double* values,
                             std::size_t n_values) {
  output << "[";
  for (std::size_t idx = 0; idx < n_values; ++idx) {
    if (idx > 0) output << ",";
    output << std::setprecision(17) << values[idx];
  }
  output << "]";
}

void write_bbox_params_json(const std::string& directory,
                            const BBoxParams& params) {
  std::ofstream output(path_join(directory, "bbox_params.json"));
  if (!output) {
    throw std::runtime_error("failed to write bbox_params.json");
  }
  const std::size_t n_boxes = params.bounds.size() / 6;
  output << "{\n";
  output << "  \"schema_version\": 1,\n";
  output << "  \"source\": \"smart_native_engine\",\n";
  output << "  \"boxes\": [\n";
  for (std::size_t idx = 0; idx < n_boxes; ++idx) {
    output << "    {\"index\": " << idx << ", \"bounds\": ";
    write_json_number_array(output, params.bounds.data() + idx * 6, 6);
    output << ", \"rotation\": ";
    write_json_number_array(output, params.rotations.data() + idx * 9, 9);
    output << "}";
    if (idx + 1 < n_boxes) output << ",";
    output << "\n";
  }
  output << "  ]\n";
  output << "}\n";
}

void write_bbox_dir(const std::string& directory, const BBoxParams& params) {
  ensure_directories(directory);
  const std::vector<double> flat = oriented_box_vertices(params);
  static const int faces[12][3] = {
      {0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6},
      {0, 1, 5}, {0, 5, 4}, {2, 6, 7}, {2, 7, 3},
      {0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5},
  };
  const std::size_t n_boxes = params.bounds.size() / 6;
  for (std::size_t box_idx = 0; box_idx < n_boxes; ++box_idx) {
    std::ofstream output(path_join(directory, "bbox" + std::to_string(box_idx) + ".obj"));
    if (!output) {
      throw std::runtime_error("failed to write bbox OBJ");
    }
    output << "o bbox_" << box_idx << "\n";
    output << std::setprecision(17);
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
  write_bbox_params_json(directory, params);
}

void write_stats_file(const std::string& path,
                      const NativeSearchResult& result) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to write native stats JSON");
  }
  output << "{\n"
         << "  \"backend\": \"smart-cpp-native\",\n"
         << "  \"core\": \"smart_native_engine\",\n"
         << "  \"command\": \"" << result.command << "\",\n"
         << "  \"output_dir\": \"" << result.output_dir << "\",\n"
         << "  \"output_path\": \"" << result.output_path << "\",\n"
         << "  \"axis_only\": " << (result.axis_only ? "true" : "false") << ",\n"
         << "  \"elapsed_sec\": " << std::setprecision(17) << result.elapsed_sec << ",\n"
         << "  \"initial_bbox_score\": " << result.initial_bbox_score << ",\n"
         << "  \"last_bbox_score\": " << result.last_bbox_score << ",\n"
         << "  \"best_reward\": " << result.best_reward << ",\n"
         << "  \"steps\": " << result.steps << ",\n"
         << "  \"iterations_run\": " << result.iterations_run << ",\n"
         << "  \"node_count\": " << result.node_count << ",\n"
         << "  \"exported_boxes\": " << result.exported_boxes << ",\n"
         << "  \"action_prior_logits\": " << result.action_prior_logits << ",\n"
         << "  \"action_value_logits\": " << result.action_value_logits << ",\n"
         << "  \"action_prior_top_k\": " << result.action_prior_top_k << ",\n"
         << "  \"transposition_table_size\": " << result.transposition_table_size << ",\n"
         << "  \"transposition_hits\": " << result.transposition_hits << ",\n"
         << "  \"transposition_stores\": " << result.transposition_stores << ",\n"
         << "  \"recenter_applies\": " << result.recenter_applies << ",\n"
         << "  \"recenter_invalid\": " << result.recenter_invalid << ",\n"
         << "  \"exact_checks\": " << result.exact_checks << "\n"
         << "}\n";
}

void write_merge_stats_file(const std::string& path,
                            const NativeSearchResult& result,
                            bool tilted,
                            bool only_nearby) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to write native merge stats JSON");
  }
  output << "{\n"
         << "  \"backend\": \"smart-cpp-native\",\n"
         << "  \"core\": \"smart_native_engine\",\n"
         << "  \"command\": \"merge\",\n"
         << "  \"elapsed_sec\": " << std::setprecision(17) << result.elapsed_sec << ",\n"
         << "  \"initial_partition_count\": " << result.initial_partition_count << ",\n"
         << "  \"active_partition_count\": " << result.active_partition_count << ",\n"
         << "  \"merge_steps\": " << result.steps << ",\n"
         << "  \"adjacency_pair_count\": " << result.adjacency_pair_count << ",\n"
         << "  \"adjacency_only_nearby\": " << (only_nearby ? "true" : "false") << ",\n"
         << "  \"tilted\": " << (tilted ? "true" : "false") << ",\n"
         << "  \"ordered_delta_queue\": true,\n"
         << "  \"candidate_inserts\": " << result.candidate_inserts << ",\n"
         << "  \"candidate_erases\": " << result.candidate_erases << ",\n"
         << "  \"candidate_queries\": " << result.candidate_queries << "\n"
         << "}\n";
}

void write_refine_mcts_stats_file(const std::string& path,
                                  const NativeRefineMctsResult& result) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to write native refine-mcts stats JSON");
  }
  output << "{\n"
         << "  \"backend\": \"smart-cpp-native\",\n"
         << "  \"core\": \"smart_native_engine\",\n"
         << "  \"command\": \"refine-mcts\",\n"
         << "  \"single_mesh_load\": true,\n"
         << "  \"single_state_bridge\": true,\n"
         << "  \"elapsed_sec\": " << std::setprecision(17)
         << result.elapsed_sec << ",\n"
         << "  \"refine_output_path\": \"" << result.refine.output_path << "\",\n"
         << "  \"mcts_output_path\": \"" << result.mcts.output_path << "\",\n"
         << "  \"refine_steps\": " << result.refine.steps << ",\n"
         << "  \"refine_exact_checks\": " << result.refine.exact_checks << ",\n"
         << "  \"mcts_iterations_run\": " << result.mcts.iterations_run << ",\n"
         << "  \"mcts_node_count\": " << result.mcts.node_count << ",\n"
         << "  \"mcts_best_reward\": " << result.mcts.best_reward << ",\n"
         << "  \"initial_bbox_score\": " << result.refine.initial_bbox_score << ",\n"
         << "  \"refine_last_bbox_score\": " << result.refine.last_bbox_score << ",\n"
         << "  \"mcts_last_bbox_score\": " << result.mcts.last_bbox_score << ",\n"
         << "  \"exported_boxes\": " << result.mcts.exported_boxes << "\n"
         << "}\n";
}

void write_greedy_segment_file(
    const std::string& path,
    const std::vector<std::vector<std::size_t>>& partitions) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to write greedy segment file: " + path);
  }
  output << partitions.size() << "\n";
  for (std::size_t idx = 0; idx < partitions.size(); ++idx) {
    for (std::size_t j = 0; j < partitions[idx].size(); ++j) {
      if (j > 0) output << " ";
      output << partitions[idx][j];
    }
    if (idx + 1 < partitions.size()) output << "\n";
  }
}

void write_merge_bbox_params_json(
    const std::string& path,
    const std::vector<std::vector<std::size_t>>& partitions,
    const std::vector<std::vector<double>>& bounds,
    const std::vector<std::vector<double>>& rotations) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to write merge bbox params JSON: " + path);
  }
  output << "{\n";
  output << "  \"schema_version\": 1,\n";
  output << "  \"source\": \"smart_native_engine.run_merge_files\",\n";
  output << "  \"boxes\": [\n";
  for (std::size_t idx = 0; idx < bounds.size(); ++idx) {
    output << "    {\"index\": " << idx << ", \"bounds\": ";
    write_json_number_array(output, bounds[idx].data(), bounds[idx].size());
    output << ", \"rotation\": ";
    write_json_number_array(output, rotations[idx].data(), rotations[idx].size());
    if (idx < partitions.size()) {
      output << ", \"partition\": [";
      for (std::size_t j = 0; j < partitions[idx].size(); ++j) {
        if (j > 0) output << ",";
        output << partitions[idx][j];
      }
      output << "]";
    }
    output << "}";
    if (idx + 1 < bounds.size()) output << ",";
    output << "\n";
  }
  output << "  ]\n";
  output << "}\n";
}

double discounted_return(const std::vector<double>& rewards, double gamma) {
  double total = 0.0;
  double discount = 1.0;
  for (double reward : rewards) {
    total += discount * reward;
    discount *= gamma;
  }
  return total;
}

std::vector<std::size_t> axis_actions(std::size_t n_boxes,
                                      std::size_t num_action_scale) {
  std::vector<std::size_t> actions;
  actions.reserve(n_boxes * 6 * num_action_scale);
  const std::size_t actions_per_bbox = 6 * num_action_scale + 1;
  for (std::size_t box_idx = 0; box_idx < n_boxes; ++box_idx) {
    const std::size_t base = box_idx * actions_per_bbox;
    for (std::size_t local = 0; local < 6 * num_action_scale; ++local) {
      actions.push_back(base + local);
    }
  }
  return actions;
}

std::vector<std::size_t> all_search_actions(std::size_t n_boxes,
                                            const NativeSearchConfig& config) {
  std::vector<std::size_t> actions;
  const std::size_t actions_per_bbox = 6 * config.num_action_scale + 1;
  actions.reserve(n_boxes * actions_per_bbox);
  for (std::size_t box_idx = 0; box_idx < n_boxes; ++box_idx) {
    const std::size_t base = box_idx * actions_per_bbox;
    for (std::size_t local = 0; local < 6 * config.num_action_scale; ++local) {
      actions.push_back(base + local);
    }
    if (config.native_recenter) {
      actions.push_back(base + actions_per_bbox - 1);
    }
  }
  return actions;
}

bool is_recenter_action(std::size_t action, const NativeSearchConfig& config) {
  if (!config.native_recenter) return false;
  const std::size_t actions_per_bbox = 6 * config.num_action_scale + 1;
  return actions_per_bbox > 0 &&
         action % actions_per_bbox == actions_per_bbox - 1;
}

double native_action_static_score(std::size_t action,
                                  const NativeSearchConfig& config) {
  double score = 0.0;
  if (action < config.action_prior_logits.size()) {
    score += (config.action_prior_weight + config.puct_prior_weight) *
             config.action_prior_logits[action];
  }
  if (action < config.action_value_logits.size()) {
    score += config.action_value_weight * config.action_value_logits[action];
  }
  return score;
}

std::vector<std::size_t> ordered_axis_actions(
    std::size_t n_boxes,
    const NativeSearchConfig& config) {
  std::vector<std::size_t> actions = all_search_actions(n_boxes, config);
  const bool use_scores = config.action_prior_weight != 0.0 ||
                          config.puct_prior_weight != 0.0 ||
                          config.action_value_weight != 0.0;
  if (use_scores) {
    std::stable_sort(actions.begin(), actions.end(), [&](std::size_t left,
                                                         std::size_t right) {
      const double left_score = native_action_static_score(left, config);
      const double right_score = native_action_static_score(right, config);
      if (left_score != right_score) return left_score > right_score;
      return left < right;
    });
  }
  if (config.action_prior_top_k > 0 &&
      actions.size() > config.action_prior_top_k && use_scores) {
    actions.resize(config.action_prior_top_k);
  }
  return actions;
}

RecenterCandidate recenter_candidate_for_bbox(
    void* state,
    const GmshMesh& mesh,
    const std::vector<double>& centroids,
    std::size_t bbox_idx) {
  BBoxParams params = copy_state(state);
  const std::size_t n_boxes = params.bounds.size() / 6;
  RecenterCandidate candidate;
  candidate.bbox_idx = bbox_idx;
  if (bbox_idx >= n_boxes) return candidate;
  std::vector<double> bounds(params.bounds.begin() + bbox_idx * 6,
                             params.bounds.begin() + bbox_idx * 6 + 6);
  std::vector<double> rotation(params.rotations.begin() + bbox_idx * 9,
                               params.rotations.begin() + bbox_idx * 9 + 9);
  std::vector<std::array<double, 3>> points =
      recenter_points_from_tets(mesh, centroids, bounds, rotation);
  return recenter_candidate_from_points(bbox_idx, bounds, rotation, points);
}

double score_recenter_action(void* state,
                             const GmshMesh& mesh,
                             const std::vector<double>& centroids,
                             std::size_t action,
                             const NativeSearchConfig& config,
                             double current_score,
                             std::size_t* invalid_count) {
  const std::size_t actions_per_bbox = 6 * config.num_action_scale + 1;
  const std::size_t bbox_idx = action / actions_per_bbox;
  RecenterCandidate candidate =
      recenter_candidate_for_bbox(state, mesh, centroids, bbox_idx);
  if (!candidate.valid) {
    if (invalid_count != nullptr) ++(*invalid_count);
    return -std::numeric_limits<double>::infinity();
  }
  const double score = smart_manifold_state_score_replacement(
      state, bbox_idx, candidate.bounds.data(), candidate.rotation.data(),
      config.cover_penalty, config.pen_rate);
  if (!std::isfinite(score)) {
    if (invalid_count != nullptr) ++(*invalid_count);
    return -std::numeric_limits<double>::infinity();
  }
  return score - current_score;
}

double apply_recenter_action(void* state,
                             const GmshMesh& mesh,
                             const std::vector<double>& centroids,
                             std::size_t action,
                             const NativeSearchConfig& config,
                             std::size_t* applied_count,
                             std::size_t* invalid_count) {
  const std::size_t actions_per_bbox = 6 * config.num_action_scale + 1;
  const std::size_t bbox_idx = action / actions_per_bbox;
  RecenterCandidate candidate =
      recenter_candidate_for_bbox(state, mesh, centroids, bbox_idx);
  if (!candidate.valid) {
    if (invalid_count != nullptr) ++(*invalid_count);
    return -std::numeric_limits<double>::infinity();
  }
  const double reward = smart_manifold_state_apply_replacement(
      state, bbox_idx, candidate.bounds.data(), candidate.rotation.data(),
      config.cover_penalty, config.pen_rate);
  if (!std::isfinite(reward)) {
    if (invalid_count != nullptr) ++(*invalid_count);
    return -std::numeric_limits<double>::infinity();
  }
  if (applied_count != nullptr) ++(*applied_count);
  return reward;
}

double apply_search_action(void* state,
                           const GmshMesh& mesh,
                           const std::vector<double>& centroids,
                           std::size_t action,
                           const NativeSearchConfig& config,
                           const std::vector<double>& scales,
                           std::size_t* recenter_applies,
                           std::size_t* recenter_invalid) {
  if (is_recenter_action(action, config)) {
    return apply_recenter_action(state, mesh, centroids, action, config,
                                 recenter_applies, recenter_invalid);
  }
  return smart_manifold_state_apply_axis_action(
      state, static_cast<std::intptr_t>(action), config.num_action_scale,
      config.action_unit, config.cover_penalty, config.pen_rate, scales.data());
}

std::string state_key_from_params(const BBoxParams& params) {
  std::ostringstream output;
  output << params.bounds.size() / 6 << ":";
  for (double value : params.bounds) {
    output << static_cast<long long>(std::llround(value * 1000000.0)) << ",";
  }
  output << "|";
  for (double value : params.rotations) {
    output << static_cast<long long>(std::llround(value * 1000000.0)) << ",";
  }
  return output.str();
}

}  // namespace

NativeSearchResult run_merge_files(const std::string& msh_path,
                                   const std::string& partitions_path,
                                   const std::string& output_segment_path,
                                   const NativeMergeConfig& config) {
  const auto started = std::chrono::steady_clock::now();
  GmshMesh mesh = load_gmsh_mesh(msh_path);
  std::vector<std::vector<std::size_t>> partitions =
      load_partitions_file(partitions_path);
  const double shape_volume = tetra_volume_sum(mesh);
  if (shape_volume <= 0.0) {
    throw std::runtime_error("native merge requires positive tet volume sum");
  }

  const std::size_t n = partitions.size();
  std::vector<std::vector<std::array<double, 3>>> points(n);
  std::vector<std::vector<std::size_t>> partition_state = partitions;
  std::vector<std::vector<double>> bounds(n);
  std::vector<std::vector<double>> rotations(n);
  std::vector<double> volumes(n, 0.0);
  for (std::size_t idx = 0; idx < n; ++idx) {
    points[idx] = points_for_partition(mesh, partitions[idx]);
    BoxFit fit = box_fit_from_points(points[idx], config.tilted);
    if (!fit.valid) {
      throw std::runtime_error("partition has no valid bbox fit");
    }
    bounds[idx] = fit.bounds;
    rotations[idx] = fit.rotation;
    volumes[idx] = fit.volume;
  }

  const auto adjacency_pairs =
      build_partition_adjacency_pairs(mesh, partitions, config.only_nearby);
  std::vector<std::set<std::size_t>> neighbors(n);
  for (const auto& pair : adjacency_pairs) {
    if (pair.size() != 2) continue;
    const std::size_t a = pair[0];
    const std::size_t b = pair[1];
    if (a < n && b < n && a != b) {
      neighbors[a].insert(b);
      neighbors[b].insert(a);
    }
  }

  std::vector<std::uint8_t> active(n, 1);
  std::vector<std::uint64_t> versions(n, 0);
  std::size_t active_count = n;
  double active_volume_total = 0.0;
  for (double value : volumes) active_volume_total += value;
  std::set<MergeDeltaKey> ordered_candidates;
  std::map<std::pair<std::size_t, std::size_t>, PartitionCandidate> candidates;
  std::size_t candidate_inserts = 0;
  std::size_t candidate_erases = 0;
  std::size_t candidate_queries = 0;
  std::size_t merge_steps = 0;

  auto pair_key = [](std::size_t a, std::size_t b) {
    if (a > b) std::swap(a, b);
    return std::make_pair(a, b);
  };
  auto key_from_candidate = [](const PartitionCandidate& candidate) {
    return MergeDeltaKey{candidate.delta, candidate.left, candidate.right};
  };
  auto erase_candidate = [&](std::size_t a, std::size_t b) {
    const auto key = pair_key(a, b);
    const auto found = candidates.find(key);
    if (found == candidates.end()) return;
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
    candidate.partition.erase(
        std::unique(candidate.partition.begin(), candidate.partition.end()),
        candidate.partition.end());
    candidate.fit = box_fit_from_points(candidate.points, config.tilted);
    candidate.delta =
        (candidate.fit.volume - volumes[left] - volumes[right]) / shape_volume;
    return candidate;
  };
  auto insert_candidate = [&](std::size_t a, std::size_t b) {
    if (a == b || a >= n || b >= n || !active[a] || !active[b]) return;
    const auto key = pair_key(a, b);
    erase_candidate(key.first, key.second);
    PartitionCandidate candidate = score_candidate(key.first, key.second);
    if (!candidate.fit.valid) return;
    candidates.emplace(key, candidate);
    ordered_candidates.insert(key_from_candidate(candidate));
    ++candidate_inserts;
  };
  auto best_candidate = [&]() -> std::optional<PartitionCandidate> {
    if (ordered_candidates.empty()) return std::nullopt;
    ++candidate_queries;
    const double prev_bvs = active_volume_total / shape_volume;
    const double target_delta = 1.0 - prev_bvs;
    std::vector<MergeDeltaKey> probes;
    auto upper = ordered_candidates.lower_bound(MergeDeltaKey::target(target_delta));
    if (upper != ordered_candidates.end()) probes.push_back(*upper);
    if (upper != ordered_candidates.begin()) probes.push_back(*std::prev(upper));
    PartitionCandidate best;
    double best_reward = -std::numeric_limits<double>::infinity();
    bool have_best = false;
    for (const MergeDeltaKey& probe : probes) {
      const auto found = candidates.find(pair_key(probe.left, probe.right));
      if (found == candidates.end()) continue;
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
    if (!have_best) return std::nullopt;
    return best;
  };

  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b : neighbors[a]) {
      if (a < b) insert_candidate(a, b);
    }
  }
  const double reward_threshold = -std::abs(config.merge_eps);
  while (active_count > 1) {
    if (config.final_k > 0 && active_count <= config.final_k) break;
    std::optional<PartitionCandidate> selected = best_candidate();
    if (!selected.has_value()) break;
    PartitionCandidate top = selected.value();
    if (config.final_k == 0 && !(top.reward > reward_threshold)) break;
    const std::size_t keep = top.left;
    const std::size_t drop = top.right;
    if (!active[keep] || !active[drop] ||
        versions[keep] != top.left_version ||
        versions[drop] != top.right_version) {
      erase_candidate(keep, drop);
      continue;
    }
    for (std::size_t nb : neighbors[keep]) erase_candidate(keep, nb);
    for (std::size_t nb : neighbors[drop]) erase_candidate(drop, nb);
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
      if (active[nb]) neighbors[nb].insert(keep);
    }
    neighbors[drop].clear();
    partition_state[drop].clear();
    points[drop].clear();
    ++merge_steps;
    for (std::size_t nb : neighbors[keep]) {
      if (active[nb]) insert_candidate(keep, nb);
    }
  }

  std::vector<std::vector<std::size_t>> active_partitions;
  std::vector<std::vector<double>> active_bounds;
  std::vector<std::vector<double>> active_rotations;
  for (std::size_t idx = 0; idx < n; ++idx) {
    if (!active[idx]) continue;
    active_partitions.push_back(partition_state[idx]);
    active_bounds.push_back(bounds[idx]);
    active_rotations.push_back(rotations[idx]);
  }

  const std::size_t slash = output_segment_path.find_last_of("/\\");
  if (slash != std::string::npos) {
    ensure_directories(output_segment_path.substr(0, slash));
  }
  write_greedy_segment_file(output_segment_path, active_partitions);
  write_merge_bbox_params_json(output_segment_path + ".bbox_params.json",
                               active_partitions, active_bounds,
                               active_rotations);

  const auto ended = std::chrono::steady_clock::now();
  NativeSearchResult result;
  result.command = "merge";
  result.output_path = output_segment_path;
  result.axis_only = !config.tilted;
  result.steps = merge_steps;
  result.initial_partition_count = n;
  result.active_partition_count = active_partitions.size();
  result.adjacency_pair_count = adjacency_pairs.size();
  result.candidate_inserts = candidate_inserts;
  result.candidate_erases = candidate_erases;
  result.candidate_queries = candidate_queries;
  result.exported_boxes = active_partitions.size();
  result.elapsed_sec = std::chrono::duration<double>(ended - started).count();
  write_merge_stats_file(output_segment_path + ".native_stats.json", result,
                         config.tilted, config.only_nearby);
  return result;
}

NativeSearchResult run_refine_files(const std::string& msh_path,
                                    const std::string& bbox_params_path,
                                    const std::string& output_dir,
                                    const NativeSearchConfig& config) {
  const auto started = std::chrono::steady_clock::now();
  GmshMesh mesh = load_gmsh_mesh(msh_path);
  GmshMesh surface_mesh = scoring_surface_mesh(msh_path, mesh);
  BBoxParams initial = load_bbox_params_file(bbox_params_path);
  const double volume_sum = tetra_volume_sum(mesh);
  const std::vector<double> centroids =
      config.native_recenter ? tetra_centroids(mesh) : std::vector<double>{};
  NativeStatePtr state = make_state(surface_mesh, initial, volume_sum, 0.0, config);
  const double initial_score =
      smart_manifold_state_score(state.get(), config.cover_penalty, config.pen_rate);
  reset_state(state.get(), initial, initial_score);

  const std::vector<double> scales = action_scales(config.num_action_scale);
  std::vector<std::intptr_t> actions(config.max_step, -1);
  std::vector<double> rewards(config.max_step, 0.0);
  std::size_t steps = 0;
  double last_score = initial_score;
  std::size_t recenter_applies = 0;
  std::size_t recenter_invalid = 0;
  std::size_t exact_checks = 0;
  if (config.native_recenter) {
    const std::vector<std::size_t> search_actions =
        all_search_actions(initial.bounds.size() / 6, config);
    for (std::size_t step = 0; step < config.max_step; ++step) {
      double best_reward = -std::numeric_limits<double>::infinity();
      std::size_t best_action = 0;
      const double current_score = smart_manifold_state_last_bbox_score(state.get());
      for (std::size_t action : search_actions) {
        double reward = -std::numeric_limits<double>::infinity();
        if (is_recenter_action(action, config)) {
          ++exact_checks;
          reward = score_recenter_action(state.get(), mesh, centroids, action,
                                         config, current_score,
                                         &recenter_invalid);
        } else {
          ++exact_checks;
          const double score = smart_manifold_state_score_axis_action(
              state.get(), static_cast<std::intptr_t>(action),
              config.num_action_scale, config.action_unit,
              config.cover_penalty, config.pen_rate, scales.data());
          reward = score - current_score;
        }
        if (reward > best_reward) {
          best_reward = reward;
          best_action = action;
        }
      }
      if (!std::isfinite(best_reward) || best_reward <= 0.0) break;
      const double applied = apply_search_action(
          state.get(), mesh, centroids, best_action, config, scales,
          &recenter_applies, &recenter_invalid);
      if (!std::isfinite(applied) || applied <= 0.0) break;
      actions[steps] = static_cast<std::intptr_t>(best_action);
      rewards[steps] = applied;
      ++steps;
      last_score = smart_manifold_state_last_bbox_score(state.get());
    }
  } else {
    std::size_t segment_exact_checks = 0;
    if (!smart_manifold_state_greedy_axis_refine_segment(
            state.get(), config.num_action_scale, config.action_unit,
            config.cover_penalty, config.pen_rate, config.max_step,
            scales.data(), actions.data(), rewards.data(), &steps,
            &last_score, &segment_exact_checks)) {
      throw std::runtime_error("native refine segment failed");
    }
    exact_checks += segment_exact_checks;
  }

  BBoxParams final_params = copy_state(state.get());
  write_bbox_dir(output_dir, final_params);
  const auto ended = std::chrono::steady_clock::now();

  NativeSearchResult result;
  result.command = "refine";
  result.output_dir = output_dir;
  result.output_path = output_dir;
  result.steps = steps;
  result.exported_boxes = final_params.bounds.size() / 6;
  result.initial_bbox_score = initial_score;
  result.last_bbox_score = last_score;
  result.axis_only = !config.native_recenter;
  result.recenter_applies = recenter_applies;
  result.recenter_invalid = recenter_invalid;
  result.exact_checks = exact_checks;
  result.elapsed_sec = std::chrono::duration<double>(ended - started).count();
  write_stats_file(path_join(output_dir, "native_stats.json"), result);
  return result;
}

NativeSearchResult run_mcts_files(const std::string& msh_path,
                                  const std::string& bbox_params_path,
                                  const std::string& output_dir,
                                  const NativeSearchConfig& config) {
  struct Node {
    std::vector<std::size_t> untried;
    std::vector<std::size_t> child_actions;
    std::vector<std::size_t> child_ids;
    std::size_t visits = 0;
    double q = 0.0;
  };

  const auto started = std::chrono::steady_clock::now();
  GmshMesh mesh = load_gmsh_mesh(msh_path);
  GmshMesh surface_mesh = scoring_surface_mesh(msh_path, mesh);
  BBoxParams initial = load_bbox_params_file(bbox_params_path);
  const std::size_t n_boxes = initial.bounds.size() / 6;
  const double volume_sum = tetra_volume_sum(mesh);
  const std::vector<double> centroids =
      config.native_recenter ? tetra_centroids(mesh) : std::vector<double>{};
  NativeStatePtr state = make_state(surface_mesh, initial, volume_sum, 0.0, config);
  const double initial_score =
      smart_manifold_state_score(state.get(), config.cover_penalty, config.pen_rate);
  reset_state(state.get(), initial, initial_score);

  const std::vector<double> scales = action_scales(config.num_action_scale);
  const std::vector<std::size_t> all_axis_actions =
      ordered_axis_actions(n_boxes, config);
  std::mt19937_64 rng(config.seed);
  std::vector<Node> nodes;
  nodes.push_back(Node{all_axis_actions});
  std::unordered_map<std::string, double> transpositions;
  std::size_t transposition_hits = 0;
  std::size_t transposition_stores = 0;
  std::size_t recenter_applies = 0;
  std::size_t recenter_invalid = 0;

  BBoxParams best_params = initial;
  double best_score = initial_score;
  double best_return = 0.0;
  std::size_t iterations_run = 0;
  for (std::size_t iter = 0; iter < config.mcts_iter; ++iter) {
    reset_state(state.get(), initial, initial_score);
    std::vector<std::size_t> path;
    std::vector<double> rewards;
    std::size_t node_id = 0;
    std::size_t steps = 0;
    while (steps < config.max_step) {
      path.push_back(node_id);
      Node& node = nodes[node_id];
      if (!node.untried.empty()) {
        std::size_t pos = 0;
        const bool use_scores = config.action_prior_weight != 0.0 ||
                                config.puct_prior_weight != 0.0 ||
                                config.action_value_weight != 0.0;
        if (!use_scores) {
          std::uniform_int_distribution<std::size_t> pick(0, node.untried.size() - 1);
          pos = pick(rng);
        } else {
          double best_static = -std::numeric_limits<double>::infinity();
          std::vector<std::size_t> best_positions;
          for (std::size_t idx = 0; idx < node.untried.size(); ++idx) {
            const double score = native_action_static_score(node.untried[idx], config);
            if (score > best_static) {
              best_static = score;
              best_positions.clear();
              best_positions.push_back(idx);
            } else if (score == best_static) {
              best_positions.push_back(idx);
            }
          }
          std::uniform_int_distribution<std::size_t> pick(0, best_positions.size() - 1);
          pos = best_positions[pick(rng)];
        }
        const std::size_t action = node.untried[pos];
        node.untried[pos] = node.untried.back();
        node.untried.pop_back();
        const double reward = apply_search_action(
            state.get(), mesh, centroids, action, config, scales,
            &recenter_applies, &recenter_invalid);
        rewards.push_back(reward);
        ++steps;
        const std::size_t child_id = nodes.size();
        node.child_actions.push_back(action);
        node.child_ids.push_back(child_id);
        nodes.push_back(Node{all_axis_actions});
        path.push_back(child_id);
        if (!std::isfinite(reward) || reward <= 0.0 || steps >= config.max_step) {
          break;
        }
        const std::size_t remaining = config.max_step - steps;
        if (remaining > 0) {
          std::vector<std::uint8_t> mask(n_boxes, 1);
          std::vector<std::uint8_t> next_mask(n_boxes, 1);
          std::vector<std::intptr_t> rollout_actions(remaining, -1);
          std::vector<double> rollout_best(remaining, 0.0);
          std::vector<double> rollout_rewards(remaining, 0.0);
          std::size_t rollout_steps = 0;
          double rollout_last = smart_manifold_state_last_bbox_score(state.get());
          if (smart_manifold_state_greedy_axis_rollout_segment(
                  state.get(), mask.data(), config.num_action_scale,
                  config.action_unit, config.cover_penalty, config.pen_rate,
                  remaining, scales.data(), rollout_actions.data(),
                  rollout_best.data(), rollout_rewards.data(), next_mask.data(),
                  &rollout_steps, &rollout_last)) {
            for (std::size_t idx = 0; idx < rollout_steps; ++idx) {
              rewards.push_back(rollout_rewards[idx]);
              ++steps;
              if (!std::isfinite(rollout_rewards[idx]) ||
                  rollout_rewards[idx] <= 0.0 || steps >= config.max_step) {
                break;
              }
            }
          }
        }
        break;
      }
      if (node.child_ids.empty()) break;
      double best_ucb = -std::numeric_limits<double>::infinity();
      std::size_t best_pos = 0;
      for (std::size_t pos = 0; pos < node.child_ids.size(); ++pos) {
        const Node& child = nodes[node.child_ids[pos]];
        const double explore = config.exp_weight *
            std::sqrt(std::log(static_cast<double>(node.visits + 1) + 1.0) /
                      static_cast<double>(child.visits + 1));
        const std::size_t action = node.child_actions[pos];
        double prior_bonus = 0.0;
        if (config.puct_prior_weight != 0.0 &&
            action < config.action_prior_logits.size()) {
          prior_bonus += config.puct_prior_weight *
              config.action_prior_logits[action] /
              static_cast<double>(child.visits + 1);
        }
        if (config.action_value_weight != 0.0 &&
            action < config.action_value_logits.size()) {
          prior_bonus += config.action_value_weight * config.action_value_logits[action];
        }
        const double ucb = child.q + explore + prior_bonus;
        if (ucb > best_ucb) {
          best_ucb = ucb;
          best_pos = pos;
        }
      }
      const std::size_t action = node.child_actions[best_pos];
      const double reward = apply_search_action(
          state.get(), mesh, centroids, action, config, scales,
          &recenter_applies, &recenter_invalid);
      rewards.push_back(reward);
      ++steps;
      node_id = node.child_ids[best_pos];
      if (!std::isfinite(reward) || reward <= 0.0) break;
    }
    double ret = discounted_return(rewards, config.gamma);
    if (config.transposition_table) {
      const std::string key = state_key_from_params(copy_state(state.get()));
      const auto found = transpositions.find(key);
      if (found != transpositions.end()) {
        ret = std::max(ret, found->second);
        ++transposition_hits;
      } else if (config.transposition_table_size > 0) {
        if (transpositions.size() >= config.transposition_table_size) {
          transpositions.erase(transpositions.begin());
        }
        transpositions.emplace(key, ret);
        ++transposition_stores;
      }
    }
    for (std::size_t id : path) {
      Node& node = nodes[id];
      node.visits += 1;
      node.q += (ret - node.q) / static_cast<double>(node.visits);
    }
    const double current_score = smart_manifold_state_last_bbox_score(state.get());
    if (ret > best_return && current_score >= best_score) {
      best_return = ret;
      best_score = current_score;
      best_params = copy_state(state.get());
    }
    iterations_run = iter + 1;
  }

  reset_state(state.get(), best_params, best_score);
  write_bbox_dir(output_dir, best_params);
  const auto ended = std::chrono::steady_clock::now();

  NativeSearchResult result;
  result.command = "mcts";
  result.output_dir = output_dir;
  result.output_path = output_dir;
  result.iterations_run = iterations_run;
  result.node_count = nodes.size();
  result.exported_boxes = best_params.bounds.size() / 6;
  result.best_reward = best_return;
  result.initial_bbox_score = initial_score;
  result.last_bbox_score = best_score;
  result.action_prior_logits = config.action_prior_logits.size();
  result.action_value_logits = config.action_value_logits.size();
  result.action_prior_top_k = config.action_prior_top_k;
  result.transposition_table_size = transpositions.size();
  result.transposition_hits = transposition_hits;
  result.transposition_stores = transposition_stores;
  result.axis_only = !config.native_recenter;
  result.recenter_applies = recenter_applies;
  result.recenter_invalid = recenter_invalid;
  result.elapsed_sec = std::chrono::duration<double>(ended - started).count();
  write_stats_file(path_join(output_dir, "native_stats.json"), result);
  return result;
}

NativeRefineMctsResult run_refine_mcts_files(
    const std::string& msh_path,
    const std::string& bbox_params_path,
    const std::string& refine_output_dir,
    const std::string& mcts_output_dir,
    const NativeSearchConfig& refine_config,
    const NativeSearchConfig& mcts_config) {
  struct Node {
    std::vector<std::size_t> untried;
    std::vector<std::size_t> child_actions;
    std::vector<std::size_t> child_ids;
    std::size_t visits = 0;
    double q = 0.0;
  };

  const auto combined_started = std::chrono::steady_clock::now();
  GmshMesh mesh = load_gmsh_mesh(msh_path);
  GmshMesh surface_mesh = scoring_surface_mesh(msh_path, mesh);
  BBoxParams initial = load_bbox_params_file(bbox_params_path);
  const double volume_sum = tetra_volume_sum(mesh);
  const bool need_centroids =
      refine_config.native_recenter || mcts_config.native_recenter;
  const std::vector<double> centroids =
      need_centroids ? tetra_centroids(mesh) : std::vector<double>{};

  if (refine_config.volume_method != mcts_config.volume_method) {
    throw std::runtime_error(
        "combined refine-mcts requires matching refine/MCTS volume_method");
  }
  NativeSearchConfig state_config = refine_config;
  state_config.stateful_union_cache =
      refine_config.stateful_union_cache || mcts_config.stateful_union_cache;
  state_config.cache_capacity =
      std::max(refine_config.cache_capacity, mcts_config.cache_capacity);

  NativeStatePtr state =
      make_state(surface_mesh, initial, volume_sum, 0.0, state_config);
  const double refine_initial_score = smart_manifold_state_score(
      state.get(), refine_config.cover_penalty, refine_config.pen_rate);
  reset_state(state.get(), initial, refine_initial_score);

  const auto refine_started = std::chrono::steady_clock::now();
  const std::vector<double> refine_scales =
      action_scales(refine_config.num_action_scale);
  std::vector<std::intptr_t> refine_actions(refine_config.max_step, -1);
  std::vector<double> refine_rewards(refine_config.max_step, 0.0);
  std::size_t refine_steps = 0;
  double refine_last_score = refine_initial_score;
  std::size_t refine_recenter_applies = 0;
  std::size_t refine_recenter_invalid = 0;
  std::size_t refine_exact_checks = 0;
  if (refine_config.native_recenter) {
    const std::vector<std::size_t> search_actions =
        all_search_actions(initial.bounds.size() / 6, refine_config);
    for (std::size_t step = 0; step < refine_config.max_step; ++step) {
      double best_reward = -std::numeric_limits<double>::infinity();
      std::size_t best_action = 0;
      const double current_score =
          smart_manifold_state_last_bbox_score(state.get());
      for (std::size_t action : search_actions) {
        double reward = -std::numeric_limits<double>::infinity();
        if (is_recenter_action(action, refine_config)) {
          ++refine_exact_checks;
          reward = score_recenter_action(
              state.get(), mesh, centroids, action, refine_config,
              current_score, &refine_recenter_invalid);
        } else {
          ++refine_exact_checks;
          const double score = smart_manifold_state_score_axis_action(
              state.get(), static_cast<std::intptr_t>(action),
              refine_config.num_action_scale, refine_config.action_unit,
              refine_config.cover_penalty, refine_config.pen_rate,
              refine_scales.data());
          reward = score - current_score;
        }
        if (reward > best_reward) {
          best_reward = reward;
          best_action = action;
        }
      }
      if (!std::isfinite(best_reward) || best_reward <= 0.0) break;
      const double applied = apply_search_action(
          state.get(), mesh, centroids, best_action, refine_config,
          refine_scales, &refine_recenter_applies, &refine_recenter_invalid);
      if (!std::isfinite(applied) || applied <= 0.0) break;
      refine_actions[refine_steps] = static_cast<std::intptr_t>(best_action);
      refine_rewards[refine_steps] = applied;
      ++refine_steps;
      refine_last_score = smart_manifold_state_last_bbox_score(state.get());
    }
  } else {
    std::size_t segment_exact_checks = 0;
    if (!smart_manifold_state_greedy_axis_refine_segment(
            state.get(), refine_config.num_action_scale,
            refine_config.action_unit, refine_config.cover_penalty,
            refine_config.pen_rate, refine_config.max_step,
            refine_scales.data(), refine_actions.data(),
            refine_rewards.data(), &refine_steps, &refine_last_score,
            &segment_exact_checks)) {
      throw std::runtime_error("native combined refine segment failed");
    }
    refine_exact_checks += segment_exact_checks;
  }
  BBoxParams refined_params = copy_state(state.get());
  write_bbox_dir(refine_output_dir, refined_params);
  const auto refine_ended = std::chrono::steady_clock::now();

  NativeSearchResult refine_result;
  refine_result.command = "refine";
  refine_result.output_dir = refine_output_dir;
  refine_result.output_path = refine_output_dir;
  refine_result.steps = refine_steps;
  refine_result.exported_boxes = refined_params.bounds.size() / 6;
  refine_result.initial_bbox_score = refine_initial_score;
  refine_result.last_bbox_score = refine_last_score;
  refine_result.axis_only = !refine_config.native_recenter;
  refine_result.recenter_applies = refine_recenter_applies;
  refine_result.recenter_invalid = refine_recenter_invalid;
  refine_result.exact_checks = refine_exact_checks;
  refine_result.elapsed_sec =
      std::chrono::duration<double>(refine_ended - refine_started).count();
  write_stats_file(path_join(refine_output_dir, "native_stats.json"),
                   refine_result);

  const auto mcts_started = std::chrono::steady_clock::now();
  double mcts_initial_score = refine_last_score;
  reset_state(state.get(), refined_params, mcts_initial_score);
  if (mcts_config.cover_penalty != refine_config.cover_penalty ||
      mcts_config.pen_rate != refine_config.pen_rate) {
    mcts_initial_score = smart_manifold_state_score(
        state.get(), mcts_config.cover_penalty, mcts_config.pen_rate);
    reset_state(state.get(), refined_params, mcts_initial_score);
  }
  const std::size_t n_boxes = refined_params.bounds.size() / 6;
  const std::vector<double> mcts_scales =
      action_scales(mcts_config.num_action_scale);
  const std::vector<std::size_t> all_axis_actions =
      ordered_axis_actions(n_boxes, mcts_config);
  std::mt19937_64 rng(mcts_config.seed);
  std::vector<Node> nodes;
  nodes.push_back(Node{all_axis_actions});
  std::unordered_map<std::string, double> transpositions;
  std::size_t transposition_hits = 0;
  std::size_t transposition_stores = 0;
  std::size_t mcts_recenter_applies = 0;
  std::size_t mcts_recenter_invalid = 0;

  BBoxParams best_params = refined_params;
  double best_score = mcts_initial_score;
  double best_return = 0.0;
  std::size_t iterations_run = 0;
  for (std::size_t iter = 0; iter < mcts_config.mcts_iter; ++iter) {
    reset_state(state.get(), refined_params, mcts_initial_score);
    std::vector<std::size_t> path;
    std::vector<double> rewards;
    std::size_t node_id = 0;
    std::size_t steps = 0;
    while (steps < mcts_config.max_step) {
      path.push_back(node_id);
      Node& node = nodes[node_id];
      if (!node.untried.empty()) {
        std::size_t pos = 0;
        const bool use_scores = mcts_config.action_prior_weight != 0.0 ||
                                mcts_config.puct_prior_weight != 0.0 ||
                                mcts_config.action_value_weight != 0.0;
        if (!use_scores) {
          std::uniform_int_distribution<std::size_t> pick(
              0, node.untried.size() - 1);
          pos = pick(rng);
        } else {
          double best_static = -std::numeric_limits<double>::infinity();
          std::vector<std::size_t> best_positions;
          for (std::size_t idx = 0; idx < node.untried.size(); ++idx) {
            const double score =
                native_action_static_score(node.untried[idx], mcts_config);
            if (score > best_static) {
              best_static = score;
              best_positions.clear();
              best_positions.push_back(idx);
            } else if (score == best_static) {
              best_positions.push_back(idx);
            }
          }
          std::uniform_int_distribution<std::size_t> pick(
              0, best_positions.size() - 1);
          pos = best_positions[pick(rng)];
        }
        const std::size_t action = node.untried[pos];
        node.untried[pos] = node.untried.back();
        node.untried.pop_back();
        const double reward = apply_search_action(
            state.get(), mesh, centroids, action, mcts_config, mcts_scales,
            &mcts_recenter_applies, &mcts_recenter_invalid);
        rewards.push_back(reward);
        ++steps;
        const std::size_t child_id = nodes.size();
        node.child_actions.push_back(action);
        node.child_ids.push_back(child_id);
        nodes.push_back(Node{all_axis_actions});
        path.push_back(child_id);
        if (!std::isfinite(reward) || reward <= 0.0 ||
            steps >= mcts_config.max_step) {
          break;
        }
        const std::size_t remaining = mcts_config.max_step - steps;
        if (remaining > 0) {
          std::vector<std::uint8_t> mask(n_boxes, 1);
          std::vector<std::uint8_t> next_mask(n_boxes, 1);
          std::vector<std::intptr_t> rollout_actions(remaining, -1);
          std::vector<double> rollout_best(remaining, 0.0);
          std::vector<double> rollout_rewards(remaining, 0.0);
          std::size_t rollout_steps = 0;
          double rollout_last =
              smart_manifold_state_last_bbox_score(state.get());
          if (smart_manifold_state_greedy_axis_rollout_segment(
                  state.get(), mask.data(), mcts_config.num_action_scale,
                  mcts_config.action_unit, mcts_config.cover_penalty,
                  mcts_config.pen_rate, remaining, mcts_scales.data(),
                  rollout_actions.data(), rollout_best.data(),
                  rollout_rewards.data(), next_mask.data(), &rollout_steps,
                  &rollout_last)) {
            for (std::size_t idx = 0; idx < rollout_steps; ++idx) {
              rewards.push_back(rollout_rewards[idx]);
              ++steps;
              if (!std::isfinite(rollout_rewards[idx]) ||
                  rollout_rewards[idx] <= 0.0 ||
                  steps >= mcts_config.max_step) {
                break;
              }
            }
          }
        }
        break;
      }
      if (node.child_ids.empty()) break;
      double best_ucb = -std::numeric_limits<double>::infinity();
      std::size_t best_pos = 0;
      for (std::size_t pos = 0; pos < node.child_ids.size(); ++pos) {
        const Node& child = nodes[node.child_ids[pos]];
        const double explore = mcts_config.exp_weight *
            std::sqrt(std::log(static_cast<double>(node.visits + 1) + 1.0) /
                      static_cast<double>(child.visits + 1));
        const std::size_t action = node.child_actions[pos];
        double prior_bonus = 0.0;
        if (mcts_config.puct_prior_weight != 0.0 &&
            action < mcts_config.action_prior_logits.size()) {
          prior_bonus += mcts_config.puct_prior_weight *
              mcts_config.action_prior_logits[action] /
              static_cast<double>(child.visits + 1);
        }
        if (mcts_config.action_value_weight != 0.0 &&
            action < mcts_config.action_value_logits.size()) {
          prior_bonus +=
              mcts_config.action_value_weight *
              mcts_config.action_value_logits[action];
        }
        const double ucb = child.q + explore + prior_bonus;
        if (ucb > best_ucb) {
          best_ucb = ucb;
          best_pos = pos;
        }
      }
      const std::size_t action = node.child_actions[best_pos];
      const double reward = apply_search_action(
          state.get(), mesh, centroids, action, mcts_config, mcts_scales,
          &mcts_recenter_applies, &mcts_recenter_invalid);
      rewards.push_back(reward);
      ++steps;
      node_id = node.child_ids[best_pos];
      if (!std::isfinite(reward) || reward <= 0.0) break;
    }
    double ret = discounted_return(rewards, mcts_config.gamma);
    if (mcts_config.transposition_table) {
      const std::string key = state_key_from_params(copy_state(state.get()));
      const auto found = transpositions.find(key);
      if (found != transpositions.end()) {
        ret = std::max(ret, found->second);
        ++transposition_hits;
      } else if (mcts_config.transposition_table_size > 0) {
        if (transpositions.size() >= mcts_config.transposition_table_size) {
          transpositions.erase(transpositions.begin());
        }
        transpositions.emplace(key, ret);
        ++transposition_stores;
      }
    }
    for (std::size_t id : path) {
      Node& node = nodes[id];
      node.visits += 1;
      node.q += (ret - node.q) / static_cast<double>(node.visits);
    }
    const double current_score =
        smart_manifold_state_last_bbox_score(state.get());
    if (ret > best_return && current_score >= best_score) {
      best_return = ret;
      best_score = current_score;
      best_params = copy_state(state.get());
    }
    iterations_run = iter + 1;
  }

  reset_state(state.get(), best_params, best_score);
  write_bbox_dir(mcts_output_dir, best_params);
  const auto mcts_ended = std::chrono::steady_clock::now();

  NativeSearchResult mcts_result;
  mcts_result.command = "mcts";
  mcts_result.output_dir = mcts_output_dir;
  mcts_result.output_path = mcts_output_dir;
  mcts_result.iterations_run = iterations_run;
  mcts_result.node_count = nodes.size();
  mcts_result.exported_boxes = best_params.bounds.size() / 6;
  mcts_result.best_reward = best_return;
  mcts_result.initial_bbox_score = mcts_initial_score;
  mcts_result.last_bbox_score = best_score;
  mcts_result.action_prior_logits = mcts_config.action_prior_logits.size();
  mcts_result.action_value_logits = mcts_config.action_value_logits.size();
  mcts_result.action_prior_top_k = mcts_config.action_prior_top_k;
  mcts_result.transposition_table_size = transpositions.size();
  mcts_result.transposition_hits = transposition_hits;
  mcts_result.transposition_stores = transposition_stores;
  mcts_result.axis_only = !mcts_config.native_recenter;
  mcts_result.recenter_applies = mcts_recenter_applies;
  mcts_result.recenter_invalid = mcts_recenter_invalid;
  mcts_result.elapsed_sec =
      std::chrono::duration<double>(mcts_ended - mcts_started).count();
  write_stats_file(path_join(mcts_output_dir, "native_stats.json"),
                   mcts_result);

  const auto combined_ended = std::chrono::steady_clock::now();
  NativeRefineMctsResult result;
  result.refine = refine_result;
  result.mcts = mcts_result;
  result.elapsed_sec =
      std::chrono::duration<double>(combined_ended - combined_started).count();
  write_refine_mcts_stats_file(
      path_join(mcts_output_dir, "refine_mcts_native_stats.json"), result);
  return result;
}

std::string result_json(const NativeSearchResult& result) {
  std::ostringstream output;
  output << "{"
         << "\"status\":\"" << result.status << "\","
         << "\"backend\":\"" << result.backend << "\","
         << "\"core\":\"smart_native_engine\","
         << "\"command\":\"" << result.command << "\","
         << "\"axis_only\":" << (result.axis_only ? "true" : "false") << ",";
  if (result.command == "refine") {
    output << "\"steps\":" << result.steps << ","
           << "\"exact_checks\":" << result.exact_checks << ","
           << "\"recenter_applies\":" << result.recenter_applies << ","
           << "\"recenter_invalid\":" << result.recenter_invalid << ",";
  } else if (result.command == "merge") {
    output << "\"merge_steps\":" << result.steps << ","
           << "\"initial_partition_count\":" << result.initial_partition_count << ","
           << "\"active_partition_count\":" << result.active_partition_count << ","
           << "\"adjacency_pair_count\":" << result.adjacency_pair_count << ","
           << "\"candidate_inserts\":" << result.candidate_inserts << ","
           << "\"candidate_erases\":" << result.candidate_erases << ","
           << "\"candidate_queries\":" << result.candidate_queries << ",";
  } else {
    output << "\"iterations_run\":" << result.iterations_run << ","
           << "\"node_count\":" << result.node_count << ","
           << "\"best_reward\":" << std::setprecision(17) << result.best_reward << ","
           << "\"action_prior_logits\":" << result.action_prior_logits << ","
           << "\"action_value_logits\":" << result.action_value_logits << ","
           << "\"action_prior_top_k\":" << result.action_prior_top_k << ","
           << "\"transposition_table_size\":" << result.transposition_table_size << ","
           << "\"transposition_hits\":" << result.transposition_hits << ","
           << "\"recenter_applies\":" << result.recenter_applies << ","
           << "\"recenter_invalid\":" << result.recenter_invalid << ",";
  }
  output << "\"initial_bbox_score\":" << std::setprecision(17)
         << result.initial_bbox_score << ","
         << "\"last_bbox_score\":" << result.last_bbox_score << ","
         << "\"output_dir\":\"" << result.output_dir << "\","
         << "\"output_path\":\"" << result.output_path << "\""
         << "}";
  return output.str();
}

std::string result_json(const NativeRefineMctsResult& result) {
  std::ostringstream output;
  output << "{"
         << "\"status\":\"" << result.status << "\","
         << "\"backend\":\"" << result.backend << "\","
         << "\"core\":\"smart_native_engine\","
         << "\"command\":\"refine-mcts\","
         << "\"elapsed_sec\":" << std::setprecision(17) << result.elapsed_sec
         << ",\"single_mesh_load\":true,"
         << "\"single_state_bridge\":true,"
         << "\"refine\":" << result_json(result.refine) << ","
         << "\"mcts\":" << result_json(result.mcts)
         << "}";
  return output.str();
}

}  // namespace smart_native
