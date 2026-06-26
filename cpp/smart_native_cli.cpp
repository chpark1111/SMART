#include "smart_native_core.hpp"
#include "smart_native_engine.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

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

namespace {

struct ObjMesh {
  std::vector<double> vertices;
  std::vector<std::vector<std::string>> face_tokens;
};

struct GmshMesh {
  std::vector<double> vertices;
  std::vector<std::size_t> faces;
  std::vector<std::size_t> voxels;
};

struct TriangleMesh {
  std::vector<double> vertices;
  std::vector<std::size_t> faces;
  double bounds[6] = {
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
  };
};

struct BBoxParams {
  std::vector<double> bounds;
  std::vector<double> rotations;
};

struct StateDelete {
  void operator()(void* handle) const {
    if (handle != nullptr) {
      smart_manifold_state_delete(handle);
    }
  }
};

using NativeStatePtr = std::unique_ptr<void, StateDelete>;

void usage() {
  std::cout
      << "smart-cpp-native commands:\n"
      << "  normalize --input model.obj --output model.obj "
         "[--mode bbox_diagonal|unit_bbox|unit_sphere] "
         "[--center bbox|mean] [--target 1.0]\n"
      << "  obj-info --input model.obj\n"
      << "  gmsh-info --msh tetra.msh\n"
      << "  split-obj-parts --input coacd.obj --output_dir coacd "
         "[--prefix part] [--split_on_usemtl]\n"
      << "  partition-coacd --msh tetra.msh --coacd_dir coacd "
         "--output coacd_partitions.json [--mesh_id id] [--partition_threads N|auto]\n"
      << "  partition-bsp --msh tetra.msh --bsp_obj bsp_seg.obj "
         "--output bsp_partitions.json [--parts_dir bsp_parts] [--mesh_id id] "
         "[--partition_threads N|auto]\n"
      << "  merge --msh tetra.msh --partitions coacd_partitions.json "
         "--output_segment greedy_segment0_coacd_mgeps0.02_fm.txt "
         "[--tilted|--no_tilted] [--only_nearby|--all_pairs]\n"
      << "  refine --msh tetra.msh --bbox_params bbox_params.json "
         "--output_dir bboxs_steps0 [--max_step 2000] [--native_recenter]\n"
      << "  mcts --msh tetra.msh --bbox_params bbox_params.json "
         "--output_dir bboxs_steps0 [--mcts_iter 3000] [--max_step 150] "
         "[--prior_logits_file logits.json] [--transposition_table] "
         "[--native_recenter]\n"
      << "  refine-mcts --msh tetra.msh --bbox_params bbox_params.json "
         "--refine_output_dir refine_bboxs_steps0 --mcts_output_dir mcts_bboxs_steps0 "
         "[--refine_max_step 2000] [--mcts_iter 3000] [--mcts_max_step 150] "
         "[--native_recenter]\n"
      << "  run-pipeline --input model.obj --work_dir work "
         "--manifoldplus_bin manifold --ftetwild_bin FloatTetwild_bin "
         "[--init_type coacd|bsp] [--coacd_bin coacd] [--bsp_obj bsp_seg.obj] "
         "[--epsilon 0.002] [--edge_length 0.1] "
         "[--partition_threads N|auto] [--reuse_existing] "
         "[--manifold_depth N] [--skip_manifoldplus] "
         "[--refine_max_step 2000] [--mcts_iter 3000] "
         "[--ftetwild_threads_supported true|false|auto]\n"
      << "  discover-meshes --data_root data --output meshes.tsv "
         "[--categories shapenet_airplane,shapenet_chair] "
         "[--limit_per_category 50] [--model_name model.obj] [--bsp_name bsp_seg.obj]\n"
      << "  run-batch --mesh_list meshes.tsv --output_root runs/native "
         "--manifoldplus_bin manifold --ftetwild_bin FloatTetwild_bin "
         "[--init_type coacd|bsp] [--coacd_bin coacd] [--jobs N|auto] "
         "[--resume_success]\n"
      << "  run-batch --data_root data --output_root runs/native "
         "--manifoldplus_bin manifold --ftetwild_bin FloatTetwild_bin "
         "[--categories shapenet_airplane,shapenet_chair] [--limit_per_category 50] "
         "[--no_category_tetra_defaults] [--jobs N|auto] [--resume_success]\n"
      << "  batch-summary --manifest runs/native/native_pipeline.jsonl\n";
}

std::string arg_value(int argc, char** argv, const std::string& name,
                      const std::string& fallback = "") {
  for (int idx = 2; idx + 1 < argc; ++idx) {
    if (argv[idx] == name) {
      return argv[idx + 1];
    }
  }
  return fallback;
}

bool has_flag(int argc, char** argv, const std::string& name) {
  for (int idx = 2; idx < argc; ++idx) {
    if (argv[idx] == name) {
      return true;
    }
  }
  return false;
}

std::string path_join_cli(const std::string& root, const std::string& child) {
  if (root.empty()) return child;
  if (root.back() == '/' || root.back() == '\\') return root + child;
  return root + "/" + child;
}

std::string read_text(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open file: " + path);
  }
  std::ostringstream stream;
  stream << input.rdbuf();
  return stream.str();
}

int mode_id(const std::string& mode) {
  if (mode == "bbox_diagonal") return 0;
  if (mode == "unit_bbox") return 1;
  if (mode == "unit_sphere") return 2;
  throw std::runtime_error("unsupported normalization mode: " + mode);
}

int center_id(const std::string& center) {
  if (center == "bbox") return 0;
  if (center == "mean") return 1;
  throw std::runtime_error("unsupported normalization center: " + center);
}

int volume_method_id(const std::string& method) {
  if (method == "mesh") return 0;
  if (method == "properties" || method == "get_properties") return 1;
  throw std::runtime_error("unsupported volume method: " + method);
}

std::size_t resolve_thread_count(const std::string& value,
                                 std::size_t work_items,
                                 std::size_t fallback = 1) {
  std::size_t requested = fallback;
  if (!value.empty()) {
    const std::string lower = [&]() {
      std::string out = value;
      std::transform(out.begin(), out.end(), out.begin(),
                     [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
      return out;
    }();
    if (lower == "auto" || lower == "0") {
      requested = static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
    } else {
      requested = static_cast<std::size_t>(std::stoull(value));
    }
  }
  if (requested == 0) {
    requested = 1;
  }
  if (work_items > 0) {
    requested = std::min(requested, work_items);
  }
  return std::max<std::size_t>(1, requested);
}

std::string resolved_thread_count_string(const std::string& value,
                                         std::size_t work_items,
                                         std::size_t fallback = 1) {
  return std::to_string(resolve_thread_count(value, work_items, fallback));
}

void mkdir_if_missing(const std::string& path) {
  if (path.empty()) {
    return;
  }
  if (::mkdir(path.c_str(), 0777) == 0 || errno == EEXIST) {
    return;
  }
  throw std::runtime_error("failed to create directory: " + path);
}

void ensure_directories(const std::string& directory) {
  if (directory.empty()) {
    return;
  }
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

std::string parent_path(const std::string& path) {
  const std::size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return "";
  }
  if (pos == 0) {
    return path.substr(0, 1);
  }
  return path.substr(0, pos);
}

std::string basename_path(const std::string& path) {
  const std::size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

bool file_exists(const std::string& path) {
  struct stat st;
  return !path.empty() && ::stat(path.c_str(), &st) == 0;
}

void copy_file_binary(const std::string& src, const std::string& dst) {
  const std::string dst_parent = parent_path(dst);
  if (!dst_parent.empty()) {
    ensure_directories(dst_parent);
  }
  std::ifstream input(src, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open source file for copy: " + src);
  }
  std::ofstream output(dst, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open destination file for copy: " + dst);
  }
  output << input.rdbuf();
  if (!output) {
    throw std::runtime_error("failed while copying file: " + src + " -> " + dst);
  }
}

bool directory_exists(const std::string& path) {
  struct stat st;
  return !path.empty() && ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string json_escape(const std::string& value) {
  std::ostringstream output;
  for (const char ch : value) {
    switch (ch) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default: output << ch; break;
    }
  }
  return output.str();
}

std::string command_for_json(const std::vector<std::string>& args) {
  std::ostringstream output;
  for (std::size_t idx = 0; idx < args.size(); ++idx) {
    if (idx > 0) output << ' ';
    output << args[idx];
  }
  return output.str();
}

std::string format_double(double value) {
  std::ostringstream output;
  output << std::setprecision(17) << value;
  return output.str();
}

struct ProcessResult {
  int return_code = -1;
  double elapsed_sec = 0.0;
};

ProcessResult run_process(const std::vector<std::string>& args,
                          const std::string& log_path,
                          double timeout_sec) {
  if (args.empty()) {
    throw std::runtime_error("cannot run empty command");
  }
  const std::string log_parent = parent_path(log_path);
  if (!log_parent.empty()) {
    ensure_directories(log_parent);
  }
  int log_fd = -1;
  if (!log_path.empty()) {
    log_fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (log_fd < 0) {
      throw std::runtime_error("failed to open process log: " + log_path);
    }
  }

  const auto started = std::chrono::steady_clock::now();
  const pid_t pid = ::fork();
  if (pid < 0) {
    if (log_fd >= 0) ::close(log_fd);
    throw std::runtime_error("fork failed for command: " + command_for_json(args));
  }
  if (pid == 0) {
    const char* suppress = ::getenv("SMART_SUPPRESS_MACOS_CRASH_DIALOG");
    if (suppress == nullptr || ::strcmp(suppress, "0") != 0) {
      ::setenv("CRASHREPORTER_DISABLE", "1", 0);
    }
    if (log_fd >= 0) {
      ::dup2(log_fd, STDOUT_FILENO);
      ::dup2(log_fd, STDERR_FILENO);
      ::close(log_fd);
    }
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& arg : args) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);
    ::execvp(argv[0], argv.data());
    _exit(127);
  }
  if (log_fd >= 0) {
    ::close(log_fd);
  }

  int status = 0;
  int sleep_ms = 1;
  while (true) {
    const pid_t done = ::waitpid(pid, &status, WNOHANG);
    if (done == pid) {
      break;
    }
    if (done < 0) {
      throw std::runtime_error("waitpid failed for command: " + command_for_json(args));
    }
    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(now - started).count();
    if (timeout_sec > 0.0 && elapsed > timeout_sec) {
      ::kill(pid, SIGTERM);
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &status, 0);
      throw std::runtime_error("process timed out after " +
                               std::to_string(timeout_sec) + "s: " +
                               command_for_json(args));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    sleep_ms = std::min(50, sleep_ms * 2);
  }

  const auto finished = std::chrono::steady_clock::now();
  ProcessResult result;
  result.elapsed_sec =
      std::chrono::duration<double>(finished - started).count();
  if (WIFEXITED(status)) {
    result.return_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.return_code = 128 + WTERMSIG(status);
  }
  return result;
}

void run_checked_process(const std::vector<std::string>& args,
                         const std::string& log_path,
                         double timeout_sec,
                         const std::string& stage_name) {
  const ProcessResult result = run_process(args, log_path, timeout_sec);
  if (result.return_code != 0) {
    throw std::runtime_error(stage_name + " failed with exit code " +
                             std::to_string(result.return_code) +
                             "; log=" + log_path);
  }
}

bool executable_supports_option(const std::string& executable,
                                const std::string& option,
                                const std::string& log_path) {
  static std::mutex cache_mutex;
  static std::map<std::string, bool> cache;
  const std::string cache_key = executable + "\n" + option;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto cached = cache.find(cache_key);
    if (cached != cache.end()) {
      return cached->second;
    }
  }
  bool supported = false;
  try {
    const ProcessResult result = run_process({executable, "--help"}, log_path, 10.0);
    (void)result;
    if (!file_exists(log_path)) {
      supported = false;
    } else {
      supported = read_text(log_path).find(option) != std::string::npos;
    }
  } catch (...) {
    supported = false;
  }
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache[cache_key] = supported;
  }
  return supported;
}

template <typename Fn>
void run_cpp_stage_to_log(const std::string& log_path,
                          const std::string& stage_name,
                          Fn&& fn) {
  const std::string log_parent = parent_path(log_path);
  if (!log_parent.empty()) {
    ensure_directories(log_parent);
  }
  std::ofstream log(log_path);
  if (!log) {
    throw std::runtime_error("failed to open log: " + log_path);
  }
  log << "smart-cpp-native internal stage: " << stage_name << "\n";
  auto* old_buffer = std::cout.rdbuf(log.rdbuf());
  try {
    fn();
    std::cout.rdbuf(old_buffer);
  } catch (...) {
    std::cout.rdbuf(old_buffer);
    throw;
  }
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
    if (cursor >= end) {
      break;
    }
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

BBoxParams load_bbox_params_file(const std::string& path) {
  const std::string text = read_text(path);
  BBoxParams params;
  std::size_t search = 0;
  const std::vector<double> identity = {1.0, 0.0, 0.0, 0.0, 1.0,
                                        0.0, 0.0, 0.0, 1.0};
  while (true) {
    const std::size_t bounds_key = text.find("\"bounds\"", search);
    if (bounds_key == std::string::npos) {
      break;
    }
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
      if (rotation.size() != 9) {
        rotation = identity;
      }
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

std::vector<double> load_number_array_file(const std::string& path) {
  if (path.empty()) return {};
  const std::string text = read_text(path);
  const std::size_t open = text.find('[');
  if (open == std::string::npos) {
    throw std::runtime_error("expected JSON number array in: " + path);
  }
  return parse_number_array_at(text, open);
}

ObjMesh load_obj_preserve_faces(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open OBJ input: " + path);
  }
  ObjMesh mesh;
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
      mesh.vertices.push_back(x);
      mesh.vertices.push_back(y);
      mesh.vertices.push_back(z);
    } else if (line.rfind("f ", 0) == 0) {
      std::istringstream stream(line);
      std::string tag;
      stream >> tag;
      std::vector<std::string> tokens;
      std::string token;
      while (stream >> token) {
        tokens.push_back(token);
      }
      if (!tokens.empty()) {
        mesh.face_tokens.push_back(std::move(tokens));
      }
    }
  }
  if (mesh.vertices.empty()) {
    throw std::runtime_error("no vertices found in OBJ: " + path);
  }
  return mesh;
}

int parse_obj_vertex_index(const std::string& token, std::size_t n_vertices) {
  std::string head = token;
  const std::size_t slash = head.find('/');
  if (slash != std::string::npos) {
    head = head.substr(0, slash);
  }
  if (head.empty()) {
    throw std::runtime_error("malformed OBJ face token: " + token);
  }
  const int raw = std::stoi(head);
  if (raw > 0) {
    return raw - 1;
  }
  if (raw < 0) {
    return static_cast<int>(n_vertices) + raw;
  }
  throw std::runtime_error("OBJ face index cannot be 0");
}

TriangleMesh load_obj_triangles(const std::string& path) {
  ObjMesh source = load_obj_preserve_faces(path);
  TriangleMesh mesh;
  mesh.vertices = std::move(source.vertices);
  const std::size_t n_vertices = mesh.vertices.size() / 3;
  for (std::size_t idx = 0; idx < n_vertices; ++idx) {
    const double* point = mesh.vertices.data() + idx * 3;
    for (int axis = 0; axis < 3; ++axis) {
      mesh.bounds[axis] = std::min(mesh.bounds[axis], point[axis]);
      mesh.bounds[axis + 3] = std::max(mesh.bounds[axis + 3], point[axis]);
    }
  }
  for (const auto& face : source.face_tokens) {
    if (face.size() < 3) {
      continue;
    }
    std::vector<std::size_t> indices;
    indices.reserve(face.size());
    for (const auto& token : face) {
      const int parsed = parse_obj_vertex_index(token, n_vertices);
      if (parsed < 0 || static_cast<std::size_t>(parsed) >= n_vertices) {
        throw std::runtime_error("OBJ face index out of range in: " + path);
      }
      indices.push_back(static_cast<std::size_t>(parsed));
    }
    for (std::size_t tri = 1; tri + 1 < indices.size(); ++tri) {
      mesh.faces.push_back(indices[0]);
      mesh.faces.push_back(indices[tri]);
      mesh.faces.push_back(indices[tri + 1]);
    }
  }
  if (mesh.faces.empty()) {
    throw std::runtime_error("no triangular faces found in OBJ: " + path);
  }
  return mesh;
}

struct SplitPartObj {
  std::vector<double> vertices;
  std::vector<std::size_t> faces;
  std::map<int, std::size_t> global_to_local;
};

std::size_t split_part_local_vertex(SplitPartObj& part,
                                    const std::vector<double>& global_vertices,
                                    int global_index) {
  auto found = part.global_to_local.find(global_index);
  if (found != part.global_to_local.end()) {
    return found->second;
  }
  if (global_index < 0 ||
      static_cast<std::size_t>(global_index) * 3 + 2 >= global_vertices.size()) {
    throw std::runtime_error("split-obj-parts face index out of range");
  }
  const std::size_t local = part.vertices.size() / 3;
  const std::size_t source = static_cast<std::size_t>(global_index) * 3;
  part.vertices.push_back(global_vertices[source + 0]);
  part.vertices.push_back(global_vertices[source + 1]);
  part.vertices.push_back(global_vertices[source + 2]);
  part.global_to_local[global_index] = local;
  return local;
}

void write_split_part_obj(const std::string& path, const SplitPartObj& part) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to write OBJ part: " + path);
  }
  for (std::size_t idx = 0; idx < part.vertices.size() / 3; ++idx) {
    output << "v " << std::setprecision(17) << part.vertices[idx * 3 + 0] << " "
           << part.vertices[idx * 3 + 1] << " " << part.vertices[idx * 3 + 2]
           << "\n";
  }
  for (std::size_t idx = 0; idx < part.faces.size(); idx += 3) {
    output << "f " << (part.faces[idx + 0] + 1) << " "
           << (part.faces[idx + 1] + 1) << " " << (part.faces[idx + 2] + 1)
           << "\n";
  }
}

std::size_t split_obj_parts_file(const std::string& input_path,
                                 const std::string& output_dir,
                                 const std::string& prefix,
                                 bool split_on_usemtl) {
  if (input_path.empty() || output_dir.empty()) {
    throw std::runtime_error("split-obj-parts requires --input and --output_dir");
  }

  std::ifstream input(input_path);
  if (!input) {
    throw std::runtime_error("failed to open OBJ for splitting: " + input_path);
  }
  ensure_directories(output_dir);
  std::vector<double> global_vertices;
  std::vector<SplitPartObj> parts;
  int current = -1;
  auto ensure_part = [&]() -> SplitPartObj& {
    if (current < 0) {
      parts.emplace_back();
      current = static_cast<int>(parts.size()) - 1;
    }
    return parts[static_cast<std::size_t>(current)];
  };

  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream stream(line);
    std::string tag;
    stream >> tag;
    if (tag == "o" || tag == "g") {
      parts.emplace_back();
      current = static_cast<int>(parts.size()) - 1;
    } else if (tag == "v") {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      stream >> x >> y >> z;
      global_vertices.push_back(x);
      global_vertices.push_back(y);
      global_vertices.push_back(z);
    } else if (tag == "usemtl" && split_on_usemtl) {
      parts.emplace_back();
      current = static_cast<int>(parts.size()) - 1;
    } else if (tag == "f") {
      std::vector<int> global_face;
      std::string token;
      const std::size_t n_vertices = global_vertices.size() / 3;
      while (stream >> token) {
        global_face.push_back(parse_obj_vertex_index(token, n_vertices));
      }
      if (global_face.size() < 3) {
        continue;
      }
      SplitPartObj& part = ensure_part();
      std::vector<std::size_t> local_face;
      local_face.reserve(global_face.size());
      for (const int global_index : global_face) {
        local_face.push_back(
            split_part_local_vertex(part, global_vertices, global_index));
      }
      for (std::size_t tri = 1; tri + 1 < local_face.size(); ++tri) {
        part.faces.push_back(local_face[0]);
        part.faces.push_back(local_face[tri]);
        part.faces.push_back(local_face[tri + 1]);
      }
    }
  }

  std::size_t written = 0;
  for (const auto& part : parts) {
    if (part.faces.empty()) {
      continue;
    }
    std::ostringstream name;
    name << prefix << "_" << std::setw(4) << std::setfill('0') << written
         << ".obj";
    write_split_part_obj(path_join(output_dir, name.str()), part);
    ++written;
  }
  if (written == 0) {
    throw std::runtime_error("split-obj-parts found no OBJ objects with faces");
  }
  return written;
}

void run_split_obj_parts_command(int argc, char** argv) {
  const std::string input_path = arg_value(argc, argv, "--input");
  const std::string output_dir = arg_value(argc, argv, "--output_dir");
  const std::string prefix = arg_value(argc, argv, "--prefix", "part");
  const bool split_on_usemtl = has_flag(argc, argv, "--split_on_usemtl");
  const std::size_t written =
      split_obj_parts_file(input_path, output_dir, prefix, split_on_usemtl);
  std::cout << "{"
            << "\"command\":\"split-obj-parts\","
            << "\"input\":\"" << input_path << "\","
            << "\"output_dir\":\"" << output_dir << "\","
            << "\"split_on_usemtl\":" << (split_on_usemtl ? "true" : "false") << ","
            << "\"parts\":" << written << "}" << "\n";
}

bool string_ends_with(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> list_obj_files(const std::string& directory) {
  DIR* dir = opendir(directory.c_str());
  if (dir == nullptr) {
    throw std::runtime_error("failed to open OBJ directory: " + directory);
  }
  std::vector<std::string> files;
  while (dirent* entry = readdir(dir)) {
    const std::string name(entry->d_name);
    if (name.empty() || name[0] == '.') {
      continue;
    }
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (string_ends_with(lower, ".obj")) {
      files.push_back(path_join(directory, name));
    }
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

std::vector<std::string> prefer_split_part_objs(const std::vector<std::string>& files) {
  std::vector<std::string> part_files;
  for (const auto& file : files) {
    const std::string name = basename_path(file);
    if (name.rfind("part_", 0) == 0) {
      part_files.push_back(file);
    }
  }
  return part_files.empty() ? files : part_files;
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
  if (mesh.voxels.empty()) {
    return 1.0;
  }
  std::vector<double> volumes(mesh.voxels.size() / 4, 0.0);
  if (!smart_native_tetra_volumes(
          mesh.vertices.data(), mesh.vertices.size() / 3, mesh.voxels.data(),
          mesh.voxels.size() / 4, volumes.data())) {
    throw std::runtime_error("failed to compute tetra volumes");
  }
  double total = 0.0;
  for (double value : volumes) {
    total += value;
  }
  return total > 0.0 ? total : 1.0;
}

std::vector<double> tetra_centroids(const GmshMesh& mesh) {
  std::vector<double> centroids((mesh.voxels.size() / 4) * 3, 0.0);
  if (!smart_native_tetra_centroids(
          mesh.vertices.data(), mesh.vertices.size() / 3, mesh.voxels.data(),
          mesh.voxels.size() / 4, centroids.data())) {
    throw std::runtime_error("failed to compute tetra centroids");
  }
  return centroids;
}

std::vector<std::vector<std::size_t>> tetra_adjacency(const GmshMesh& mesh) {
  const std::size_t n_voxels = mesh.voxels.size() / 4;
  std::vector<std::size_t> offsets(n_voxels + 1, 0);
  std::size_t n_values = 0;
  if (!smart_native_tetra_adjacency(mesh.voxels.data(), n_voxels, offsets.data(),
                                   nullptr, 0, &n_values)) {
    throw std::runtime_error("failed to count tetra adjacency");
  }
  std::vector<std::size_t> values(n_values, 0);
  if (!smart_native_tetra_adjacency(mesh.voxels.data(), n_voxels, offsets.data(),
                                   values.data(), values.size(), &n_values)) {
    throw std::runtime_error("failed to compute tetra adjacency");
  }
  values.resize(n_values);
  std::vector<std::vector<std::size_t>> adjacency(n_voxels);
  for (std::size_t idx = 0; idx < n_voxels; ++idx) {
    for (std::size_t pos = offsets[idx]; pos < offsets[idx + 1]; ++pos) {
      adjacency[idx].push_back(values[pos]);
    }
  }
  return adjacency;
}

double dot3_cli(const double* left, const double* right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

void cross3_cli(const double* left, const double* right, double* out) {
  out[0] = left[1] * right[2] - left[2] * right[1];
  out[1] = left[2] * right[0] - left[0] * right[2];
  out[2] = left[0] * right[1] - left[1] * right[0];
}

bool ray_intersects_triangle(const double* origin,
                             const double* direction,
                             const double* v0,
                             const double* v1,
                             const double* v2) {
  constexpr double eps = 1.0e-10;
  const double edge1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  const double edge2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
  double h[3];
  cross3_cli(direction, edge2, h);
  const double a = dot3_cli(edge1, h);
  if (std::abs(a) < eps) {
    return false;
  }
  const double f = 1.0 / a;
  const double s[3] = {origin[0] - v0[0], origin[1] - v0[1], origin[2] - v0[2]};
  const double u = f * dot3_cli(s, h);
  if (u < -eps || u > 1.0 + eps) {
    return false;
  }
  double q[3];
  cross3_cli(s, edge1, q);
  const double v = f * dot3_cli(direction, q);
  if (v < -eps || u + v > 1.0 + eps) {
    return false;
  }
  const double t = f * dot3_cli(edge2, q);
  return t > eps;
}

bool point_inside_mesh(const TriangleMesh& mesh, const double* point) {
  constexpr double eps = 1.0e-9;
  for (int axis = 0; axis < 3; ++axis) {
    if (point[axis] < mesh.bounds[axis] - eps ||
        point[axis] > mesh.bounds[axis + 3] + eps) {
      return false;
    }
  }
  const double direction[3] = {1.0, 0.372137, 0.157913};
  std::size_t hits = 0;
  for (std::size_t tri = 0; tri < mesh.faces.size() / 3; ++tri) {
    const double* v0 = mesh.vertices.data() + mesh.faces[tri * 3] * 3;
    const double* v1 = mesh.vertices.data() + mesh.faces[tri * 3 + 1] * 3;
    const double* v2 = mesh.vertices.data() + mesh.faces[tri * 3 + 2] * 3;
    if (ray_intersects_triangle(point, direction, v0, v1, v2)) {
      ++hits;
    }
  }
  return (hits % 2) == 1;
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
                          bool stateful_union_cache,
                          std::size_t cache_capacity,
                          int volume_method) {
  std::vector<float> vertices = vertices_to_float(mesh.vertices);
  std::vector<std::uint32_t> faces = faces_to_uint32(mesh.faces);
  const std::size_t n_boxes = params.bounds.size() / 6;
  NativeStatePtr state(smart_manifold_state_new(
      vertices.data(), vertices.size() / 3, faces.data(), faces.size() / 3,
      params.bounds.data(), params.rotations.data(), n_boxes, volume_sum,
      last_bbox_score, stateful_union_cache ? 1 : 0, cache_capacity,
      volume_method));
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

std::vector<float> oriented_box_vertices(const BBoxParams& params) {
  const std::size_t n_boxes = params.bounds.size() / 6;
  std::vector<float> out;
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
          out.push_back(static_cast<float>(
              base[0] + rot[0] * i * lengths[0] +
              rot[3] * j * lengths[1] + rot[6] * k * lengths[2]));
          out.push_back(static_cast<float>(
              base[1] + rot[1] * i * lengths[0] +
              rot[4] * j * lengths[1] + rot[7] * k * lengths[2]));
          out.push_back(static_cast<float>(
              base[2] + rot[2] * i * lengths[0] +
              rot[5] * j * lengths[1] + rot[8] * k * lengths[2]));
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
  output << "  \"source\": \"smart-cpp-native\",\n";
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
  const std::vector<float> flat = oriented_box_vertices(params);
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
                      const std::string& command,
                      double elapsed_sec,
                      double initial_score,
                      double final_score,
                      std::size_t steps,
                      std::size_t exported_boxes,
                      bool axis_only) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to write native stats JSON");
  }
  output << "{\n"
         << "  \"backend\": \"smart-cpp-native\",\n"
         << "  \"command\": \"" << command << "\",\n"
         << "  \"axis_only\": " << (axis_only ? "true" : "false") << ",\n"
         << "  \"elapsed_sec\": " << std::setprecision(17) << elapsed_sec << ",\n"
         << "  \"initial_bbox_score\": " << initial_score << ",\n"
         << "  \"last_bbox_score\": " << final_score << ",\n"
         << "  \"steps\": " << steps << ",\n"
         << "  \"exported_boxes\": " << exported_boxes << "\n"
         << "}\n";
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

void normalize_obj(const std::string& input_path, const std::string& output_path,
                   const std::string& mode, const std::string& center,
                   double target) {
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
      throw std::runtime_error("malformed OBJ vertex line: " + line);
    }
    vertices.push_back(x);
    vertices.push_back(y);
    vertices.push_back(z);
  }
  if (vertices.empty()) {
    throw std::runtime_error("no vertices found in OBJ: " + input_path);
  }

  std::vector<double> normalized(vertices.size(), 0.0);
  std::vector<double> stats(34, 0.0);
  if (!smart_native_normalize_vertices(
          vertices.data(), vertices.size() / 3, mode_id(mode), center_id(center),
          target, normalized.data(), stats.data())) {
    throw std::runtime_error("native normalization failed");
  }

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
  auto stats_object = [&](std::size_t offset) {
    std::cout << "{"
              << "\"vertex_count\":" << static_cast<std::size_t>(std::llround(stats[offset])) << ","
              << "\"bbox_min\":[" << stats[offset + 1] << ',' << stats[offset + 2] << ',' << stats[offset + 3] << "],"
              << "\"bbox_max\":[" << stats[offset + 4] << ',' << stats[offset + 5] << ',' << stats[offset + 6] << "],"
              << "\"bbox_extent\":[" << stats[offset + 7] << ',' << stats[offset + 8] << ',' << stats[offset + 9] << "],"
              << "\"bbox_diagonal\":" << stats[offset + 10] << ","
              << "\"bbox_center\":[" << stats[offset + 11] << ',' << stats[offset + 12] << ',' << stats[offset + 13] << "],"
              << "\"sphere_radius\":" << stats[offset + 14]
              << "}";
  };
  std::cout << "{"
            << "\"status\":\"success\","
            << "\"backend\":\"smart-cpp-native\","
            << "\"before\":";
  stats_object(0);
  std::cout << ",\"center\":[" << stats[15] << ',' << stats[16] << ',' << stats[17] << "],"
            << "\"scale\":" << stats[18] << ","
            << "\"after\":";
  stats_object(19);
  std::cout << "}\n";
}

void obj_info(const std::string& input_path) {
  ObjMesh mesh = load_obj_preserve_faces(input_path);
  std::cout << "{"
            << "\"vertices\":" << (mesh.vertices.size() / 3) << ","
            << "\"faces\":" << mesh.face_tokens.size()
            << "}\n";
}

void gmsh_info(const std::string& msh_path) {
  std::size_t vertices = 0;
  std::size_t faces = 0;
  std::size_t voxels = 0;
  if (!smart_native_load_gmsh_counts(
          msh_path.c_str(), &vertices, &faces, &voxels)) {
    throw std::runtime_error("failed to read Gmsh counts: " + msh_path);
  }
  std::cout << "{"
            << "\"vertices\":" << vertices << ","
            << "\"faces\":" << faces << ","
            << "\"voxels\":" << voxels
            << "}\n";
}

void write_partitions_json(const std::string& output_path,
                           const std::vector<std::vector<std::size_t>>& partitions,
                           const std::string& source,
                           const std::string& init_type,
                           const std::string& mesh_id,
                           std::size_t part_obj_count,
                           std::size_t tet_count,
                           std::size_t empty_component_count) {
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("failed to write partition JSON: " + output_path);
  }
  output << "{\n";
  output << "  \"schema_version\": 1,\n";
  output << "  \"source\": \"" << json_escape(source) << "\",\n";
  output << "  \"init_type\": \"" << json_escape(init_type) << "\",\n";
  output << "  \"mesh_id\": \"" << mesh_id << "\",\n";
  output << "  \"part_obj_count\": " << part_obj_count << ",\n";
  output << "  \"tet_count\": " << tet_count << ",\n";
  output << "  \"empty_component_count\": " << empty_component_count << ",\n";
  output << "  \"partitions\": [\n";
  for (std::size_t part_idx = 0; part_idx < partitions.size(); ++part_idx) {
    output << "    [";
    for (std::size_t item_idx = 0; item_idx < partitions[part_idx].size(); ++item_idx) {
      if (item_idx > 0) output << ", ";
      output << partitions[part_idx][item_idx];
    }
    output << "]";
    if (part_idx + 1 < partitions.size()) output << ",";
    output << "\n";
  }
  output << "  ]\n";
  output << "}\n";
}

std::string label_key(const std::vector<int>& labels) {
  if (labels.empty()) {
    return "";
  }
  std::ostringstream stream;
  for (std::size_t idx = 0; idx < labels.size(); ++idx) {
    if (idx > 0) stream << ' ';
    stream << labels[idx];
  }
  return stream.str();
}

void recompute_triangle_bounds(TriangleMesh& mesh) {
  for (int axis = 0; axis < 3; ++axis) {
    mesh.bounds[axis] = std::numeric_limits<double>::infinity();
    mesh.bounds[axis + 3] = -std::numeric_limits<double>::infinity();
  }
  for (std::size_t idx = 0; idx < mesh.vertices.size() / 3; ++idx) {
    const double* point = mesh.vertices.data() + idx * 3;
    for (int axis = 0; axis < 3; ++axis) {
      mesh.bounds[axis] = std::min(mesh.bounds[axis], point[axis]);
      mesh.bounds[axis + 3] = std::max(mesh.bounds[axis + 3], point[axis]);
    }
  }
}

void rotate_mesh_y_minus_90(TriangleMesh& mesh) {
  for (std::size_t idx = 0; idx < mesh.vertices.size() / 3; ++idx) {
    double* point = mesh.vertices.data() + idx * 3;
    const double x = point[0];
    const double z = point[2];
    point[0] = -z;
    point[2] = x;
  }
  recompute_triangle_bounds(mesh);
}

void partition_parts_to_json(const std::string& msh,
                             const std::vector<std::string>& part_files,
                             const std::string& output_path,
                             const std::string& mesh_id,
                             const std::string& init_type,
                             const std::string& source,
                             bool rotate_y_minus_90,
                             const std::string& command_name,
                             const std::string& partition_threads = "1") {
  const GmshMesh mesh = load_gmsh_mesh(msh);
  const std::size_t n_voxels = mesh.voxels.size() / 4;
  if (n_voxels == 0) {
    throw std::runtime_error(command_name + " requires tetra voxels");
  }
  if (part_files.empty()) {
    throw std::runtime_error(command_name + " found no OBJ parts");
  }
  std::vector<TriangleMesh> part_meshes;
  part_meshes.reserve(part_files.size());
  for (const auto& part_file : part_files) {
    TriangleMesh part = load_obj_triangles(part_file);
    if (rotate_y_minus_90) {
      rotate_mesh_y_minus_90(part);
    }
    part_meshes.push_back(std::move(part));
  }

  const std::vector<double> centroids = tetra_centroids(mesh);
  std::map<std::string, std::vector<std::size_t>> grouped;
  std::vector<std::uint8_t> assigned(n_voxels, 0);
  const std::size_t worker_count =
      resolve_thread_count(partition_threads, n_voxels, 1);
  if (worker_count == 1) {
    for (std::size_t tet_idx = 0; tet_idx < n_voxels; ++tet_idx) {
      const double* centroid = centroids.data() + tet_idx * 3;
      std::vector<int> labels;
      for (std::size_t part_idx = 0; part_idx < part_meshes.size(); ++part_idx) {
        if (point_inside_mesh(part_meshes[part_idx], centroid)) {
          labels.push_back(static_cast<int>(part_idx));
        }
      }
      if (!labels.empty()) {
        assigned[tet_idx] = 1;
        grouped[label_key(labels)].push_back(tet_idx);
      }
    }
  } else {
    std::vector<std::map<std::string, std::vector<std::size_t>>> local_grouped(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
      const std::size_t begin = (n_voxels * worker_idx) / worker_count;
      const std::size_t end = (n_voxels * (worker_idx + 1)) / worker_count;
      workers.emplace_back([&, worker_idx, begin, end]() {
        auto& local = local_grouped[worker_idx];
        for (std::size_t tet_idx = begin; tet_idx < end; ++tet_idx) {
          const double* centroid = centroids.data() + tet_idx * 3;
          std::vector<int> labels;
          for (std::size_t part_idx = 0; part_idx < part_meshes.size(); ++part_idx) {
            if (point_inside_mesh(part_meshes[part_idx], centroid)) {
              labels.push_back(static_cast<int>(part_idx));
            }
          }
          if (!labels.empty()) {
            assigned[tet_idx] = 1;
            local[label_key(labels)].push_back(tet_idx);
          }
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
    for (auto& local : local_grouped) {
      for (auto& entry : local) {
        auto& dest = grouped[entry.first];
        dest.insert(dest.end(), entry.second.begin(), entry.second.end());
      }
    }
  }

  std::vector<std::vector<std::size_t>> partitions;
  partitions.reserve(grouped.size());
  for (auto& entry : grouped) {
    std::sort(entry.second.begin(), entry.second.end());
    partitions.push_back(std::move(entry.second));
  }

  std::size_t empty_component_count = 0;
  const std::vector<std::vector<std::size_t>> adjacency = tetra_adjacency(mesh);
  std::vector<std::uint8_t> visited(n_voxels, 0);
  for (std::size_t tet_idx = 0; tet_idx < n_voxels; ++tet_idx) {
    if (assigned[tet_idx] || visited[tet_idx]) {
      continue;
    }
    std::vector<std::size_t> stack = {tet_idx};
    std::vector<std::size_t> component;
    visited[tet_idx] = 1;
    while (!stack.empty()) {
      const std::size_t current = stack.back();
      stack.pop_back();
      component.push_back(current);
      for (const std::size_t next : adjacency[current]) {
        if (!assigned[next] && !visited[next]) {
          visited[next] = 1;
          stack.push_back(next);
        }
      }
    }
    std::sort(component.begin(), component.end());
    partitions.push_back(std::move(component));
    ++empty_component_count;
  }
  std::sort(partitions.begin(), partitions.end(),
            [](const auto& left, const auto& right) {
              if (left.empty() || right.empty()) {
                return left.size() < right.size();
              }
              return left.front() < right.front();
            });

  std::size_t assigned_count = 0;
  for (const auto& partition : partitions) {
    assigned_count += partition.size();
  }
  if (assigned_count != n_voxels) {
    throw std::runtime_error(command_name + " did not assign every tetrahedron");
  }
  write_partitions_json(output_path, partitions, source, init_type, mesh_id,
                        part_files.size(), n_voxels, empty_component_count);
  std::cout << "{"
            << "\"status\":\"success\","
            << "\"backend\":\"smart-cpp-native\","
            << "\"command\":\"" << command_name << "\","
            << "\"init_type\":\"" << init_type << "\","
            << "\"rotate_y_minus_90\":" << (rotate_y_minus_90 ? "true" : "false") << ","
            << "\"partition_threads\":" << worker_count << ","
            << "\"part_obj_count\":" << part_files.size() << ","
            << "\"tet_count\":" << n_voxels << ","
            << "\"partition_count\":" << partitions.size() << ","
            << "\"empty_component_count\":" << empty_component_count
            << "}\n";
}

void run_partition_coacd_command(int argc, char** argv) {
  const std::string msh = arg_value(argc, argv, "--msh");
  std::string coacd_dir = arg_value(argc, argv, "--coacd_dir");
  if (coacd_dir.empty()) {
    coacd_dir = arg_value(argc, argv, "--coacd");
  }
  const std::string output_path = arg_value(argc, argv, "--output");
  const std::string mesh_id = arg_value(argc, argv, "--mesh_id");
  const std::string partition_threads =
      arg_value(argc, argv, "--partition_threads", "1");
  if (msh.empty() || coacd_dir.empty() || output_path.empty()) {
    throw std::runtime_error(
        "partition-coacd requires --msh, --coacd_dir, and --output");
  }
  const std::vector<std::string> part_files =
      prefer_split_part_objs(list_obj_files(coacd_dir));
  if (part_files.empty()) {
    throw std::runtime_error("partition-coacd found no OBJ parts in: " + coacd_dir);
  }
  partition_parts_to_json(
      msh, part_files, output_path, mesh_id, "coacd",
      "smart-cpp-native partition-coacd", false, "partition-coacd",
      partition_threads);
}

void run_partition_bsp_command(int argc, char** argv) {
  const std::string msh = arg_value(argc, argv, "--msh");
  const std::string bsp_obj = arg_value(argc, argv, "--bsp_obj");
  const std::string output_path = arg_value(argc, argv, "--output");
  const std::string mesh_id = arg_value(argc, argv, "--mesh_id");
  const std::string partition_threads =
      arg_value(argc, argv, "--partition_threads", "1");
  std::string parts_dir = arg_value(argc, argv, "--parts_dir");
  if (msh.empty() || bsp_obj.empty() || output_path.empty()) {
    throw std::runtime_error("partition-bsp requires --msh, --bsp_obj, and --output");
  }
  if (parts_dir.empty()) {
    const std::string parent = parent_path(output_path);
    parts_dir = path_join(parent.empty() ? "." : parent, "bsp_parts");
  }
  split_obj_parts_file(bsp_obj, parts_dir, "part", true);
  const std::vector<std::string> part_files =
      prefer_split_part_objs(list_obj_files(parts_dir));
  partition_parts_to_json(
      msh, part_files, output_path, mesh_id, "bsp",
      "smart-cpp-native partition-bsp", true, "partition-bsp",
      partition_threads);
}

void run_refine_command(int argc, char** argv) {
  const std::string msh = arg_value(argc, argv, "--msh");
  const std::string bbox_params = arg_value(argc, argv, "--bbox_params");
  const std::string output_dir = arg_value(argc, argv, "--output_dir");
  if (msh.empty() || bbox_params.empty() || output_dir.empty()) {
    throw std::runtime_error("refine requires --msh, --bbox_params, and --output_dir");
  }
  const std::size_t max_step =
      static_cast<std::size_t>(std::stoull(arg_value(argc, argv, "--max_step", "2000")));
  const std::size_t num_action_scale =
      static_cast<std::size_t>(std::stoull(arg_value(argc, argv, "--num_action_scale", "2")));
  const double action_unit = std::stod(arg_value(argc, argv, "--action_unit", "0.01"));
  const double cover_penalty = std::stod(arg_value(argc, argv, "--cover_penalty", "100"));
  const double pen_rate = std::stod(arg_value(argc, argv, "--pen_rate", "1.0"));
  const bool stateful_cache = !has_flag(argc, argv, "--no_stateful_union_cache");
  const std::size_t cache_capacity =
      static_cast<std::size_t>(std::stoull(arg_value(argc, argv, "--cache_capacity", "65536")));
  smart_native::NativeSearchConfig config;
  config.max_step = max_step;
  config.num_action_scale = num_action_scale;
  config.action_unit = action_unit;
  config.cover_penalty = cover_penalty;
  config.pen_rate = pen_rate;
  config.stateful_union_cache = stateful_cache;
  config.cache_capacity = cache_capacity;
  config.volume_method = arg_value(argc, argv, "--volume_method", "mesh");
  config.native_recenter = has_flag(argc, argv, "--native_recenter");
  const smart_native::NativeSearchResult result =
      smart_native::run_refine_files(msh, bbox_params, output_dir, config);
  std::cout << smart_native::result_json(result) << "\n";
}

void run_merge_command(int argc, char** argv) {
  const std::string msh = arg_value(argc, argv, "--msh");
  const std::string partitions = arg_value(argc, argv, "--partitions");
  std::string output_segment = arg_value(argc, argv, "--output_segment");
  if (output_segment.empty()) {
    output_segment = arg_value(argc, argv, "--output");
  }
  if (msh.empty() || partitions.empty() || output_segment.empty()) {
    throw std::runtime_error(
        "merge requires --msh, --partitions, and --output_segment");
  }
  smart_native::NativeMergeConfig config;
  config.merge_eps = std::stod(arg_value(argc, argv, "--merge_eps", "0.02"));
  config.final_k = static_cast<std::size_t>(
      std::stoull(arg_value(argc, argv, "--final_k", "0")));
  config.tilted = has_flag(argc, argv, "--no_tilted") ? false : true;
  if (has_flag(argc, argv, "--tilted")) {
    config.tilted = true;
  }
  config.only_nearby = has_flag(argc, argv, "--all_pairs") ? false : true;
  if (has_flag(argc, argv, "--only_nearby")) {
    config.only_nearby = true;
  }
  smart_native::NativeSearchResult result =
      smart_native::run_merge_files(msh, partitions, output_segment, config);
  std::cout << smart_native::result_json(result) << "\n";
}

void run_mcts_command(int argc, char** argv) {
  const std::string msh = arg_value(argc, argv, "--msh");
  const std::string bbox_params = arg_value(argc, argv, "--bbox_params");
  const std::string output_dir = arg_value(argc, argv, "--output_dir");
  if (msh.empty() || bbox_params.empty() || output_dir.empty()) {
    throw std::runtime_error("mcts requires --msh, --bbox_params, and --output_dir");
  }
  const std::size_t mcts_iter =
      static_cast<std::size_t>(std::stoull(arg_value(argc, argv, "--mcts_iter", "3000")));
  const std::size_t max_step =
      static_cast<std::size_t>(std::stoull(arg_value(argc, argv, "--max_step", "150")));
  const std::size_t num_action_scale =
      static_cast<std::size_t>(std::stoull(arg_value(argc, argv, "--num_action_scale", "2")));
  const double action_unit = std::stod(arg_value(argc, argv, "--action_unit", "0.02"));
  const double cover_penalty = std::stod(arg_value(argc, argv, "--cover_penalty", "100"));
  const double pen_rate = std::stod(arg_value(argc, argv, "--pen_rate", "1.0"));
  const double exp_weight = std::stod(arg_value(argc, argv, "--exp_w", "0.001"));
  const double gamma = std::stod(arg_value(argc, argv, "--gamma", "1.0"));
  const std::uint64_t seed =
      static_cast<std::uint64_t>(std::stoull(arg_value(argc, argv, "--seed", "0")));
  const bool stateful_cache = !has_flag(argc, argv, "--no_stateful_union_cache");
  const std::size_t cache_capacity =
      static_cast<std::size_t>(std::stoull(arg_value(argc, argv, "--cache_capacity", "65536")));
  const double action_prior_weight =
      std::stod(arg_value(argc, argv, "--action_prior_weight", "0.0"));
  const double puct_prior_weight =
      std::stod(arg_value(argc, argv, "--puct_prior_weight", "0.0"));
  const double action_value_weight =
      std::stod(arg_value(argc, argv, "--action_value_weight", "0.0"));
  const std::size_t action_prior_top_k = static_cast<std::size_t>(
      std::stoull(arg_value(argc, argv, "--action_prior_top_k", "0")));
  const bool transposition_table = has_flag(argc, argv, "--transposition_table");
  const std::size_t transposition_table_size = static_cast<std::size_t>(
      std::stoull(arg_value(argc, argv, "--transposition_table_size", "8192")));
  smart_native::NativeSearchConfig config;
  config.mcts_iter = mcts_iter;
  config.max_step = max_step;
  config.num_action_scale = num_action_scale;
  config.action_unit = action_unit;
  config.cover_penalty = cover_penalty;
  config.pen_rate = pen_rate;
  config.exp_weight = exp_weight;
  config.gamma = gamma;
  config.seed = seed;
  config.stateful_union_cache = stateful_cache;
  config.cache_capacity = cache_capacity;
  config.action_prior_weight = action_prior_weight;
  config.puct_prior_weight = puct_prior_weight;
  config.action_value_weight = action_value_weight;
  config.action_prior_top_k = action_prior_top_k;
  config.transposition_table = transposition_table;
  config.transposition_table_size = transposition_table_size;
  config.native_recenter = has_flag(argc, argv, "--native_recenter");
  config.volume_method = arg_value(argc, argv, "--volume_method", "mesh");
  config.action_prior_logits =
      load_number_array_file(arg_value(argc, argv, "--prior_logits_file"));
  config.action_value_logits =
      load_number_array_file(arg_value(argc, argv, "--value_logits_file"));
  const smart_native::NativeSearchResult result =
      smart_native::run_mcts_files(msh, bbox_params, output_dir, config);
  std::cout << smart_native::result_json(result) << "\n";
}

void run_refine_mcts_command(int argc, char** argv) {
  const std::string msh = arg_value(argc, argv, "--msh");
  const std::string bbox_params = arg_value(argc, argv, "--bbox_params");
  const std::string refine_output_dir = arg_value(argc, argv, "--refine_output_dir");
  const std::string mcts_output_dir = arg_value(argc, argv, "--mcts_output_dir");
  if (msh.empty() || bbox_params.empty() || refine_output_dir.empty() ||
      mcts_output_dir.empty()) {
    throw std::runtime_error(
        "refine-mcts requires --msh, --bbox_params, --refine_output_dir, and --mcts_output_dir");
  }

  const std::size_t num_action_scale =
      static_cast<std::size_t>(std::stoull(arg_value(argc, argv, "--num_action_scale", "2")));
  const double cover_penalty = std::stod(arg_value(argc, argv, "--cover_penalty", "100"));
  const double pen_rate = std::stod(arg_value(argc, argv, "--pen_rate", "1.0"));
  const bool stateful_cache = !has_flag(argc, argv, "--no_stateful_union_cache");
  const std::size_t cache_capacity =
      static_cast<std::size_t>(std::stoull(arg_value(argc, argv, "--cache_capacity", "65536")));
  const bool native_recenter = has_flag(argc, argv, "--native_recenter");
  const std::string volume_method = arg_value(argc, argv, "--volume_method", "mesh");

  smart_native::NativeSearchConfig refine_config;
  refine_config.max_step = static_cast<std::size_t>(
      std::stoull(arg_value(argc, argv, "--refine_max_step",
                            arg_value(argc, argv, "--max_step", "2000"))));
  refine_config.num_action_scale = num_action_scale;
  refine_config.action_unit = std::stod(arg_value(argc, argv, "--refine_action_unit",
                                                  arg_value(argc, argv, "--action_unit", "0.01")));
  refine_config.cover_penalty = cover_penalty;
  refine_config.pen_rate = pen_rate;
  refine_config.stateful_union_cache = stateful_cache;
  refine_config.cache_capacity = cache_capacity;
  refine_config.volume_method = volume_method;
  refine_config.native_recenter = native_recenter;

  smart_native::NativeSearchConfig mcts_config;
  mcts_config.mcts_iter = static_cast<std::size_t>(
      std::stoull(arg_value(argc, argv, "--mcts_iter", "3000")));
  mcts_config.max_step = static_cast<std::size_t>(
      std::stoull(arg_value(argc, argv, "--mcts_max_step", "150")));
  mcts_config.num_action_scale = num_action_scale;
  mcts_config.action_unit = std::stod(arg_value(argc, argv, "--mcts_action_unit",
                                                arg_value(argc, argv, "--action_unit", "0.02")));
  mcts_config.cover_penalty = cover_penalty;
  mcts_config.pen_rate = pen_rate;
  mcts_config.exp_weight = std::stod(arg_value(argc, argv, "--exp_w", "0.001"));
  mcts_config.gamma = std::stod(arg_value(argc, argv, "--gamma", "1.0"));
  mcts_config.seed =
      static_cast<std::uint64_t>(std::stoull(arg_value(argc, argv, "--seed", "0")));
  mcts_config.stateful_union_cache = stateful_cache;
  mcts_config.cache_capacity = cache_capacity;
  mcts_config.action_prior_weight =
      std::stod(arg_value(argc, argv, "--action_prior_weight", "0.0"));
  mcts_config.puct_prior_weight =
      std::stod(arg_value(argc, argv, "--puct_prior_weight", "0.0"));
  mcts_config.action_value_weight =
      std::stod(arg_value(argc, argv, "--action_value_weight", "0.0"));
  mcts_config.action_prior_top_k = static_cast<std::size_t>(
      std::stoull(arg_value(argc, argv, "--action_prior_top_k", "0")));
  mcts_config.transposition_table = has_flag(argc, argv, "--transposition_table");
  mcts_config.transposition_table_size = static_cast<std::size_t>(
      std::stoull(arg_value(argc, argv, "--transposition_table_size", "8192")));
  mcts_config.native_recenter = native_recenter;
  mcts_config.volume_method = volume_method;
  mcts_config.action_prior_logits =
      load_number_array_file(arg_value(argc, argv, "--prior_logits_file"));
  mcts_config.action_value_logits =
      load_number_array_file(arg_value(argc, argv, "--value_logits_file"));

  const smart_native::NativeRefineMctsResult result =
      smart_native::run_refine_mcts_files(
          msh, bbox_params, refine_output_dir, mcts_output_dir,
          refine_config, mcts_config);
  std::cout << smart_native::result_json(result) << "\n";
}

std::string lowercase_copy(std::string value);

void run_pipeline_command(int argc, char** argv, const std::string& self_bin) {
  const std::string input = arg_value(argc, argv, "--input");
  const std::string work_dir = arg_value(argc, argv, "--work_dir");
  const std::string manifoldplus_bin = arg_value(argc, argv, "--manifoldplus_bin");
  const std::string ftetwild_bin = arg_value(argc, argv, "--ftetwild_bin");
  const std::string coacd_bin = arg_value(argc, argv, "--coacd_bin");
  const std::string init_type = arg_value(argc, argv, "--init_type", "coacd");
  const bool skip_manifoldplus = has_flag(argc, argv, "--skip_manifoldplus");
  if (init_type != "coacd" && init_type != "bsp") {
    throw std::runtime_error("run-pipeline --init_type must be coacd or bsp");
  }
  if (input.empty() || work_dir.empty() ||
      (!skip_manifoldplus && manifoldplus_bin.empty()) ||
      ftetwild_bin.empty() || (init_type == "coacd" && coacd_bin.empty())) {
    throw std::runtime_error(
        "run-pipeline requires --input, --work_dir, --ftetwild_bin, and "
        "--coacd_bin when --init_type coacd; --manifoldplus_bin is required "
        "unless --skip_manifoldplus is set");
  }

  const auto started = std::chrono::steady_clock::now();
  ensure_directories(work_dir);
  const std::string logs_dir = path_join(work_dir, "logs");
  const std::string norm_dir = path_join(work_dir, "normalized");
  const std::string tetra_dir = path_join(work_dir, "tetra");
  const std::string coacd_dir = path_join(work_dir, "coacd");
  const std::string bsp_parts_dir = path_join(work_dir, "bsp_parts");
  const std::string merge_dir = path_join(work_dir, "merge");
  const std::string refine_dir = path_join(work_dir, "refine_bboxs_steps0");
  const std::string mcts_dir = path_join(work_dir, "mcts_bboxs_steps0");
  const std::string pipeline_stats_path = path_join(work_dir, "native_pipeline_stats.json");
  ensure_directories(logs_dir);
  ensure_directories(norm_dir);
  ensure_directories(tetra_dir);
  ensure_directories(init_type == "coacd" ? coacd_dir : bsp_parts_dir);
  ensure_directories(merge_dir);

  const std::string normalized_obj = path_join(norm_dir, "model.obj");
  const std::string manifold_obj = path_join(tetra_dir, "model_manifold.obj");
  const std::string tetra_msh = path_join(tetra_dir, "tetra.msh");
  const std::string tetra_surface = tetra_msh + "__sf.obj";
  const std::string coacd_combined = path_join(coacd_dir, "coacd_parts.obj");
  const std::string partitions_json =
      path_join(tetra_dir, init_type == "coacd" ? "coacd_partitions.json"
                                                : "bsp_partitions.json");
  const std::string init_suffix = init_type == "bsp" ? "" : "_" + init_type;
  const std::string segment_path =
      path_join(merge_dir, "greedy_segment0" + init_suffix + "_mgeps" +
                            arg_value(argc, argv, "--merge_eps", "0.02") +
                            "_fm.txt");
  const std::string bbox_params = segment_path + ".bbox_params.json";

  const std::string norm_mode = arg_value(argc, argv, "--normalize_mode", "bbox_diagonal");
  const std::string norm_center = arg_value(argc, argv, "--normalize_center", "bbox");
  const std::string norm_target = arg_value(argc, argv, "--normalize_target", "1.0");
  const std::string epsilon = arg_value(argc, argv, "--epsilon", "0.002");
  const std::string edge_length = arg_value(argc, argv, "--edge_length", "0.1");
  const std::string ftetwild_level = arg_value(argc, argv, "--ftetwild_level", "2");
  const std::string manifold_timeout =
      arg_value(argc, argv, "--manifold_timeout_sec", "600");
  const std::string manifold_depth = arg_value(argc, argv, "--manifold_depth", "0");
  const std::string ftetwild_timeout =
      arg_value(argc, argv, "--ftetwild_timeout_sec", "1200");
  const std::string coacd_timeout =
      arg_value(argc, argv, "--coacd_timeout_sec", "1200");
  const std::string partition_threads =
      arg_value(argc, argv, "--partition_threads", "1");
  const std::string ftetwild_threads_arg = arg_value(argc, argv, "--ftetwild_threads");
  const std::string ftetwild_threads = ftetwild_threads_arg.empty()
                                           ? ""
                                           : resolved_thread_count_string(
                                                 ftetwild_threads_arg, 0,
                                                 std::max(1u, std::thread::hardware_concurrency()));
  const std::string ftetwild_threads_supported_arg =
      lowercase_copy(arg_value(argc, argv, "--ftetwild_threads_supported", "auto"));
  bool ftetwild_supports_threads = false;
  if (!ftetwild_threads.empty()) {
    if (ftetwild_threads_supported_arg == "true" ||
        ftetwild_threads_supported_arg == "1" ||
        ftetwild_threads_supported_arg == "yes") {
      ftetwild_supports_threads = true;
    } else if (ftetwild_threads_supported_arg == "false" ||
               ftetwild_threads_supported_arg == "0" ||
               ftetwild_threads_supported_arg == "no") {
      ftetwild_supports_threads = false;
    } else {
      ftetwild_supports_threads =
          executable_supports_option(ftetwild_bin, "--max-threads",
                                     path_join(logs_dir, "ftetwild_help.log"));
    }
  }
  const bool reuse_existing = has_flag(argc, argv, "--reuse_existing");
  const bool reuse_preprocessing =
      reuse_existing || has_flag(argc, argv, "--reuse_preprocessing");
  std::vector<std::pair<std::string, double>> stage_timings;
  auto time_stage = [&](const std::string& name, auto&& fn) {
    const auto stage_started = std::chrono::steady_clock::now();
    fn();
    const auto stage_finished = std::chrono::steady_clock::now();
    stage_timings.push_back({
        name, std::chrono::duration<double>(stage_finished - stage_started).count()});
  };

  (void)self_bin;
  if (reuse_preprocessing && file_exists(normalized_obj)) {
    stage_timings.push_back({"normalize_reuse", 0.0});
  } else {
    time_stage("normalize", [&]() {
      run_cpp_stage_to_log(path_join(logs_dir, "normalize.log"), "normalize", [&]() {
      normalize_obj(input, normalized_obj, norm_mode, norm_center,
                    std::stod(norm_target));
      });
    });
  }
  if (!file_exists(normalized_obj)) {
    throw std::runtime_error("normalization did not create: " + normalized_obj);
  }

  if (reuse_preprocessing && file_exists(manifold_obj)) {
    stage_timings.push_back({"manifoldplus_reuse", 0.0});
  } else if (skip_manifoldplus) {
    time_stage("manifoldplus_skip", [&]() {
      copy_file_binary(normalized_obj, manifold_obj);
    });
  } else {
    time_stage("manifoldplus", [&]() {
      std::vector<std::string> manifold_args = {
          manifoldplus_bin, "--input", normalized_obj, "--output", manifold_obj};
      if (std::stoi(manifold_depth) > 0) {
        manifold_args.push_back("--depth");
        manifold_args.push_back(manifold_depth);
      }
      run_checked_process(
          manifold_args,
          path_join(logs_dir, "manifoldplus.log"), std::stod(manifold_timeout),
          "ManifoldPlus");
    });
  }
  if (!file_exists(manifold_obj)) {
    throw std::runtime_error("ManifoldPlus did not create: " + manifold_obj);
  }

  const auto build_ftetwild_args =
      [&](const std::string& length, const std::string& eps,
          bool coarsen) -> std::vector<std::string> {
    std::vector<std::string> args = {
        ftetwild_bin, "--input", manifold_obj, "--output", tetra_msh,
        "-q", "-l", length, "-e", eps, "--log",
        path_join(tetra_dir, "log.txt"), "--level", ftetwild_level, "--no-binary"};
    if (!has_flag(argc, argv, "--no_use_floodfill")) {
      args.push_back("--use-floodfill");
    }
    if (has_flag(argc, argv, "--use_general_wn")) {
      args.push_back("--use-general-wn");
    }
    if (has_flag(argc, argv, "--use_input_for_wn")) {
      args.push_back("--use-input-for-wn");
    }
    if (!has_flag(argc, argv, "--no_manifold_surface")) {
      args.push_back("--manifold-surface");
    }
    if (has_flag(argc, argv, "--skip_simplify")) {
      args.push_back("--skip-simplify");
    }
    if (ftetwild_supports_threads) {
      args.push_back("--max-threads");
      args.push_back(ftetwild_threads);
    }
    if (coarsen) {
      args.push_back("--coarsen");
    }
    return args;
  };
  const double ft_timeout = std::stod(ftetwild_timeout);
  bool ftetwild_retried = false;
  if (reuse_preprocessing && file_exists(tetra_msh) && file_exists(tetra_surface)) {
    stage_timings.push_back({"ftetwild_reuse", 0.0});
  } else {
    time_stage("ftetwild", [&]() {
      ProcessResult ftetwild_result =
          run_process(build_ftetwild_args(edge_length, epsilon,
                                          has_flag(argc, argv, "--coarsen")),
                      path_join(logs_dir, "ftetwild.log"), ft_timeout);
      if (ftetwild_result.return_code != 0) {
        ftetwild_retried = true;
        ::unlink(tetra_msh.c_str());
        ::unlink(tetra_surface.c_str());
        const double retry_eps =
            std::stod(epsilon) * std::stod(arg_value(argc, argv, "--retry_epsilon_scale", "2.0"));
        const double retry_length =
            std::stod(edge_length) *
            std::stod(arg_value(argc, argv, "--retry_edge_length_scale", "2.0"));
        ftetwild_result = run_process(
            build_ftetwild_args(format_double(retry_length), format_double(retry_eps), true),
            path_join(logs_dir, "ftetwild_retry.log"), ft_timeout);
        if (ftetwild_result.return_code != 0) {
          throw std::runtime_error(
              "fTetWild failed primary and retry attempts; logs=" +
              path_join(logs_dir, "ftetwild.log") + "," +
              path_join(logs_dir, "ftetwild_retry.log"));
        }
      }
    });
  }
  if (!file_exists(tetra_msh) || !file_exists(tetra_surface)) {
    throw std::runtime_error("fTetWild did not create tetra.msh and tetra.msh__sf.obj");
  }

  if (reuse_preprocessing && file_exists(partitions_json)) {
    stage_timings.push_back({"preseg_reuse", 0.0});
  } else if (init_type == "coacd") {
    std::vector<std::string> coacd_args;
    if (::access(coacd_bin.c_str(), X_OK) == 0) {
      coacd_args.push_back(coacd_bin);
    } else {
      coacd_args.push_back(arg_value(argc, argv, "--python", "python3"));
      coacd_args.push_back(coacd_bin);
    }
    coacd_args.insert(coacd_args.end(),
                      {"-i", tetra_surface, "-o", coacd_combined, "-t",
                       arg_value(argc, argv, "--coacd_threshold", "0.05"),
        "-c", arg_value(argc, argv, "--coacd_max_convex_hull", "64"),
        "-pm", arg_value(argc, argv, "--coacd_preprocess_mode", "auto"),
        "-pr", arg_value(argc, argv, "--coacd_preprocess_resolution", "50"),
        "-r", arg_value(argc, argv, "--coacd_resolution", "2000"),
        "-mn", arg_value(argc, argv, "--coacd_mcts_nodes", "20"),
        "-mi", arg_value(argc, argv, "--coacd_mcts_iterations", "150"),
        "-md", arg_value(argc, argv, "--coacd_mcts_max_depth", "3"),
                       "--seed", arg_value(argc, argv, "--coacd_seed", "7777")});
    if (has_flag(argc, argv, "--coacd_pca")) {
      coacd_args.push_back("--pca");
    }
    if (has_flag(argc, argv, "--coacd_no_merge")) {
      coacd_args.push_back("-nm");
    }
    if (!has_flag(argc, argv, "--coacd_no_decimate")) {
      coacd_args.push_back("-d");
    }
    time_stage("coacd", [&]() {
      run_checked_process(coacd_args, path_join(logs_dir, "coacd.log"),
                          std::stod(coacd_timeout), "CoACD");
    });
    if (!file_exists(coacd_combined)) {
      throw std::runtime_error("CoACD did not create: " + coacd_combined);
    }

    time_stage("split_obj_parts", [&]() {
      run_cpp_stage_to_log(path_join(logs_dir, "split_obj_parts.log"),
                           "split-obj-parts", [&]() {
      const std::size_t written =
          split_obj_parts_file(coacd_combined, coacd_dir, "part", false);
      std::cout << "{"
                << "\"command\":\"split-obj-parts\","
                << "\"input\":\"" << json_escape(coacd_combined) << "\","
                << "\"output_dir\":\"" << json_escape(coacd_dir) << "\","
                << "\"split_on_usemtl\":false,"
                << "\"parts\":" << written << "}\n";
      });
    });

    time_stage("partition_coacd", [&]() {
      run_cpp_stage_to_log(path_join(logs_dir, "partition_coacd.log"),
                           "partition-coacd", [&]() {
      const std::vector<std::string> part_files =
          prefer_split_part_objs(list_obj_files(coacd_dir));
      if (part_files.empty()) {
        throw std::runtime_error("partition-coacd found no OBJ parts in: " + coacd_dir);
      }
      partition_parts_to_json(
          tetra_msh, part_files, partitions_json,
          arg_value(argc, argv, "--mesh_id", ""), "coacd",
          "smart-cpp-native partition-coacd", false, "partition-coacd",
          partition_threads);
      });
    });
  } else {
    std::string bsp_obj = arg_value(argc, argv, "--bsp_obj");
    if (bsp_obj.empty()) {
      bsp_obj = path_join(parent_path(input), "bsp_seg.obj");
    }
    if (!file_exists(bsp_obj)) {
      throw std::runtime_error("run-pipeline --init_type bsp requires bsp_seg.obj: " +
                               bsp_obj);
    }
    time_stage("partition_bsp", [&]() {
      run_cpp_stage_to_log(path_join(logs_dir, "partition_bsp.log"),
                           "partition-bsp", [&]() {
      split_obj_parts_file(bsp_obj, bsp_parts_dir, "part", true);
      const std::vector<std::string> part_files =
          prefer_split_part_objs(list_obj_files(bsp_parts_dir));
      partition_parts_to_json(
          tetra_msh, part_files, partitions_json,
          arg_value(argc, argv, "--mesh_id", ""), "bsp",
          "smart-cpp-native partition-bsp", true, "partition-bsp",
          partition_threads);
      });
    });
  }
  if (!file_exists(partitions_json)) {
    throw std::runtime_error("partition metadata did not create: " + partitions_json);
  }

  if (reuse_existing && file_exists(bbox_params)) {
    stage_timings.push_back({"merge_reuse", 0.0});
  } else {
    time_stage("merge", [&]() {
      run_cpp_stage_to_log(path_join(logs_dir, "merge.log"), "merge", [&]() {
    smart_native::NativeMergeConfig merge_config;
    merge_config.merge_eps = std::stod(arg_value(argc, argv, "--merge_eps", "0.02"));
    merge_config.final_k = static_cast<std::size_t>(
        std::stoull(arg_value(argc, argv, "--final_k", "0")));
    merge_config.tilted = has_flag(argc, argv, "--no_tilted") ? false : true;
    if (has_flag(argc, argv, "--tilted")) {
      merge_config.tilted = true;
    }
    merge_config.only_nearby = has_flag(argc, argv, "--all_pairs") ? false : true;
    if (has_flag(argc, argv, "--only_nearby")) {
      merge_config.only_nearby = true;
    }
    const smart_native::NativeSearchResult merge_result =
        smart_native::run_merge_files(tetra_msh, partitions_json, segment_path,
                                      merge_config);
    std::cout << smart_native::result_json(merge_result) << "\n";
      });
    });
  }
  if (!file_exists(bbox_params)) {
    throw std::runtime_error("merge did not create bbox params: " + bbox_params);
  }

  if (reuse_existing && file_exists(path_join(mcts_dir, "bbox0.obj"))) {
    stage_timings.push_back({"refine_mcts_reuse", 0.0});
  } else {
    time_stage("refine_mcts", [&]() {
      run_cpp_stage_to_log(path_join(logs_dir, "refine_mcts.log"),
                           "refine-mcts", [&]() {
    const std::size_t num_action_scale = static_cast<std::size_t>(
        std::stoull(arg_value(argc, argv, "--num_action_scale", "2")));
    const double cover_penalty = std::stod(arg_value(argc, argv, "--cover_penalty", "100"));
    const double pen_rate = std::stod(arg_value(argc, argv, "--pen_rate", "1.0"));
    const bool stateful_cache = !has_flag(argc, argv, "--no_stateful_union_cache");
    const std::size_t cache_capacity = static_cast<std::size_t>(
        std::stoull(arg_value(argc, argv, "--cache_capacity", "65536")));
    const bool native_recenter = has_flag(argc, argv, "--native_recenter");
    const std::string volume_method = arg_value(argc, argv, "--volume_method", "mesh");

    smart_native::NativeSearchConfig refine_config;
    refine_config.max_step = static_cast<std::size_t>(
        std::stoull(arg_value(argc, argv, "--refine_max_step", "2000")));
    refine_config.num_action_scale = num_action_scale;
    refine_config.action_unit =
        std::stod(arg_value(argc, argv, "--refine_action_unit", "0.01"));
    refine_config.cover_penalty = cover_penalty;
    refine_config.pen_rate = pen_rate;
    refine_config.stateful_union_cache = stateful_cache;
    refine_config.cache_capacity = cache_capacity;
    refine_config.volume_method = volume_method;
    refine_config.native_recenter = native_recenter;

    smart_native::NativeSearchConfig mcts_config;
    mcts_config.mcts_iter = static_cast<std::size_t>(
        std::stoull(arg_value(argc, argv, "--mcts_iter", "3000")));
    mcts_config.max_step = static_cast<std::size_t>(
        std::stoull(arg_value(argc, argv, "--mcts_max_step", "150")));
    mcts_config.num_action_scale = num_action_scale;
    mcts_config.action_unit =
        std::stod(arg_value(argc, argv, "--mcts_action_unit", "0.02"));
    mcts_config.cover_penalty = cover_penalty;
    mcts_config.pen_rate = pen_rate;
    mcts_config.exp_weight = std::stod(arg_value(argc, argv, "--exp_w", "0.001"));
    mcts_config.gamma = std::stod(arg_value(argc, argv, "--gamma", "1.0"));
    mcts_config.seed =
        static_cast<std::uint64_t>(std::stoull(arg_value(argc, argv, "--seed", "0")));
    mcts_config.stateful_union_cache = stateful_cache;
    mcts_config.cache_capacity = cache_capacity;
    mcts_config.action_prior_weight =
        std::stod(arg_value(argc, argv, "--action_prior_weight", "0.0"));
    mcts_config.puct_prior_weight =
        std::stod(arg_value(argc, argv, "--puct_prior_weight", "0.0"));
    mcts_config.action_value_weight =
        std::stod(arg_value(argc, argv, "--action_value_weight", "0.0"));
    mcts_config.action_prior_top_k = static_cast<std::size_t>(
        std::stoull(arg_value(argc, argv, "--action_prior_top_k", "0")));
    mcts_config.transposition_table = has_flag(argc, argv, "--transposition_table");
    mcts_config.transposition_table_size = static_cast<std::size_t>(
        std::stoull(arg_value(argc, argv, "--transposition_table_size", "8192")));
    mcts_config.native_recenter = native_recenter;
    mcts_config.volume_method = volume_method;
    mcts_config.action_prior_logits =
        load_number_array_file(arg_value(argc, argv, "--prior_logits_file"));
    mcts_config.action_value_logits =
        load_number_array_file(arg_value(argc, argv, "--value_logits_file"));

    const smart_native::NativeRefineMctsResult search_result =
        smart_native::run_refine_mcts_files(
            tetra_msh, bbox_params, refine_dir, mcts_dir,
            refine_config, mcts_config);
    std::cout << smart_native::result_json(search_result) << "\n";
      });
    });
  }

  const auto finished = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(finished - started).count();
  {
    std::ofstream stats(pipeline_stats_path);
    if (!stats) {
      throw std::runtime_error("failed to write pipeline stats: " + pipeline_stats_path);
    }
    stats << "{\n";
    stats << "  \"status\": \"success\",\n";
    stats << "  \"backend\": \"smart-cpp-native\",\n";
    stats << "  \"command\": \"run-pipeline\",\n";
    stats << "  \"init_type\": \"" << json_escape(init_type) << "\",\n";
    stats << "  \"elapsed_sec\": " << std::setprecision(17) << elapsed << ",\n";
    stats << "  \"reuse_existing\": " << (reuse_existing ? "true" : "false") << ",\n";
    stats << "  \"reuse_preprocessing\": " << (reuse_preprocessing ? "true" : "false") << ",\n";
    stats << "  \"skip_manifoldplus\": " << (skip_manifoldplus ? "true" : "false") << ",\n";
    stats << "  \"manifold_depth\": " << std::stoi(manifold_depth) << ",\n";
    stats << "  \"ftetwild_retried\": " << (ftetwild_retried ? "true" : "false") << ",\n";
    stats << "  \"partition_threads\": \"" << json_escape(partition_threads) << "\",\n";
    stats << "  \"stages\": [\n";
    for (std::size_t idx = 0; idx < stage_timings.size(); ++idx) {
      stats << "    {\"name\": \"" << json_escape(stage_timings[idx].first)
            << "\", \"elapsed_sec\": " << std::setprecision(17)
            << stage_timings[idx].second << "}";
      if (idx + 1 < stage_timings.size()) {
        stats << ",";
      }
      stats << "\n";
    }
    stats << "  ]\n";
    stats << "}\n";
  }
  std::cout << "{"
            << "\"status\":\"success\","
            << "\"backend\":\"smart-cpp-native\","
            << "\"command\":\"run-pipeline\","
            << "\"core\":\"smart_native_pipeline\","
            << "\"init_type\":\"" << json_escape(init_type) << "\","
            << "\"elapsed_sec\":" << std::setprecision(17) << elapsed << ","
            << "\"skip_manifoldplus\":" << (skip_manifoldplus ? "true" : "false") << ","
            << "\"manifold_depth\":" << std::stoi(manifold_depth) << ","
            << "\"stats_path\":\"" << json_escape(pipeline_stats_path) << "\","
            << "\"input\":\"" << json_escape(input) << "\","
            << "\"work_dir\":\"" << json_escape(work_dir) << "\","
            << "\"normalized_obj\":\"" << json_escape(normalized_obj) << "\","
            << "\"tetra_msh\":\"" << json_escape(tetra_msh) << "\","
            << "\"tetra_surface\":\"" << json_escape(tetra_surface) << "\","
            << "\"preseg_dir\":\""
            << json_escape(init_type == "coacd" ? coacd_dir : bsp_parts_dir) << "\","
            << "\"coacd_dir\":\"" << json_escape(coacd_dir) << "\","
            << "\"bsp_parts_dir\":\"" << json_escape(bsp_parts_dir) << "\","
            << "\"partitions\":\"" << json_escape(partitions_json) << "\","
            << "\"merge_segment\":\"" << json_escape(segment_path) << "\","
            << "\"bbox_params\":\"" << json_escape(bbox_params) << "\","
            << "\"refine_output_dir\":\"" << json_escape(refine_dir) << "\","
            << "\"mcts_output_dir\":\"" << json_escape(mcts_dir) << "\""
            << "}\n";
}

struct BatchMeshRow {
  std::string mesh_id;
  std::string input;
  std::string bsp_obj;
};

struct TetraDefaults {
  std::string epsilon;
  std::string edge_length;
  std::string source;
};

struct BatchRunRecord {
  BatchMeshRow row;
  std::string work_dir;
  std::string log_path;
  std::string stats_path;
  std::string pipeline_execution;
  std::string tetra_default_source;
  std::string tetra_epsilon;
  std::string tetra_edge_length;
  std::string status = "success";
  std::string error;
  int return_code = 0;
  double elapsed_sec = 0.0;
};

struct DiscoveryResult {
  std::vector<BatchMeshRow> rows;
  std::map<std::string, std::size_t> counts;
  std::size_t skipped_missing_bsp = 0;
};

std::string trim_copy(const std::string& value) {
  std::size_t first = 0;
  while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) {
    ++first;
  }
  std::size_t last = value.size();
  while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) {
    --last;
  }
  return value.substr(first, last - first);
}

std::vector<std::string> split_csv(const std::string& value) {
  std::vector<std::string> items;
  std::string item;
  std::istringstream stream(value);
  while (std::getline(stream, item, ',')) {
    item = trim_copy(item);
    if (!item.empty()) {
      items.push_back(item);
    }
  }
  return items;
}

bool contains_string(const std::vector<std::string>& values, const std::string& value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

std::string lowercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

std::string category_from_batch_row(const BatchMeshRow& row) {
  const std::size_t separator = row.mesh_id.find("__");
  if (separator != std::string::npos && separator > 0) {
    return row.mesh_id.substr(0, separator);
  }
  const std::string mesh_dir = parent_path(row.input);
  const std::string category_dir = parent_path(mesh_dir);
  return basename_path(category_dir);
}

TetraDefaults smart_category_tetra_defaults(const BatchMeshRow& row) {
  const std::string category = lowercase_copy(category_from_batch_row(row));
  TetraDefaults defaults;
  if (category == "airplane" || category == "shapenet_airplane" ||
      category == "02691156") {
    defaults.epsilon = "0.002";
    defaults.edge_length = "0.1";
    defaults.source = "smart_airplane";
  } else if (category == "chair" || category == "table" ||
             category == "shapenet_chair" || category == "shapenet_table" ||
             category == "03001627" || category == "04379243") {
    defaults.epsilon = "0.004";
    defaults.edge_length = "0.2";
    defaults.source = "smart_chair_table";
  }
  return defaults;
}

std::vector<std::string> list_subdirectories(const std::string& directory) {
  DIR* dir = opendir(directory.c_str());
  if (dir == nullptr) {
    throw std::runtime_error("failed to open directory: " + directory);
  }
  std::vector<std::string> paths;
  while (dirent* entry = readdir(dir)) {
    const std::string name(entry->d_name);
    if (name.empty() || name[0] == '.') {
      continue;
    }
    const std::string path = path_join(directory, name);
    if (directory_exists(path)) {
      paths.push_back(path);
    }
  }
  closedir(dir);
  std::sort(paths.begin(), paths.end());
  return paths;
}

DiscoveryResult discover_mesh_rows_from_data_root(
    const std::string& data_root,
    const std::string& model_name,
    const std::string& bsp_name,
    const std::vector<std::string>& category_filter,
    std::size_t limit_per_category,
    bool require_bsp) {
  if (!directory_exists(data_root)) {
    throw std::runtime_error("discover-meshes data root does not exist: " + data_root);
  }

  DiscoveryResult result;
  for (const auto& category_dir : list_subdirectories(data_root)) {
    const std::string category = basename_path(category_dir);
    if (!category_filter.empty() && !contains_string(category_filter, category)) {
      continue;
    }
    for (const auto& mesh_dir : list_subdirectories(category_dir)) {
      const auto count_it = result.counts.find(category);
      if (limit_per_category > 0 && count_it != result.counts.end() &&
          count_it->second >= limit_per_category) {
        break;
      }
      const std::string model_path = path_join(mesh_dir, model_name);
      if (!file_exists(model_path)) {
        continue;
      }
      const std::string bsp_path = path_join(mesh_dir, bsp_name);
      if (require_bsp && !file_exists(bsp_path)) {
        ++result.skipped_missing_bsp;
        continue;
      }
      BatchMeshRow row;
      row.mesh_id = category + "__" + basename_path(mesh_dir);
      row.input = model_path;
      if (file_exists(bsp_path)) {
        row.bsp_obj = bsp_path;
      }
      result.rows.push_back(std::move(row));
      ++result.counts[category];
    }
  }
  return result;
}

void write_mesh_rows_tsv(const std::string& output, const std::vector<BatchMeshRow>& rows) {
  const std::string output_parent = parent_path(output);
  if (!output_parent.empty()) {
    ensure_directories(output_parent);
  }
  std::ofstream mesh_list(output);
  if (!mesh_list) {
    throw std::runtime_error("failed to open mesh list output: " + output);
  }
  for (const auto& row : rows) {
    mesh_list << row.mesh_id << '\t' << row.input;
    if (!row.bsp_obj.empty()) {
      mesh_list << '\t' << row.bsp_obj;
    }
    mesh_list << '\n';
  }
  mesh_list.close();
}

void run_discover_meshes_command(int argc, char** argv) {
  const std::string data_root = arg_value(argc, argv, "--data_root");
  const std::string output = arg_value(argc, argv, "--output");
  const std::string model_name = arg_value(argc, argv, "--model_name", "model.obj");
  const std::string bsp_name = arg_value(argc, argv, "--bsp_name", "bsp_seg.obj");
  const std::vector<std::string> category_filter = split_csv(arg_value(argc, argv, "--categories"));
  const std::size_t limit_per_category =
      static_cast<std::size_t>(std::stoull(arg_value(argc, argv, "--limit_per_category", "0")));
  const bool require_bsp = has_flag(argc, argv, "--require_bsp");
  if (data_root.empty() || output.empty()) {
    throw std::runtime_error("discover-meshes requires --data_root and --output");
  }
  const DiscoveryResult discovery = discover_mesh_rows_from_data_root(
      data_root, model_name, bsp_name, category_filter, limit_per_category, require_bsp);
  write_mesh_rows_tsv(output, discovery.rows);
  std::cout << "{"
            << "\"status\":\"success\","
            << "\"command\":\"discover-meshes\","
            << "\"data_root\":\"" << json_escape(data_root) << "\","
            << "\"output\":\"" << json_escape(output) << "\","
            << "\"mesh_count\":" << discovery.rows.size() << ","
            << "\"skipped_missing_bsp\":" << discovery.skipped_missing_bsp << ","
            << "\"categories\":{";
  bool first = true;
  for (const auto& entry : discovery.counts) {
    if (!first) {
      std::cout << ',';
    }
    first = false;
    std::cout << "\"" << json_escape(entry.first) << "\":" << entry.second;
  }
  std::cout << "}}\n";
}

std::vector<std::string> split_mesh_list_line(const std::string& line) {
  const char delimiter = line.find('\t') != std::string::npos ? '\t' : ',';
  std::vector<std::string> columns;
  std::string item;
  std::istringstream stream(line);
  while (std::getline(stream, item, delimiter)) {
    std::size_t first = 0;
    while (first < item.size() && std::isspace(static_cast<unsigned char>(item[first]))) {
      ++first;
    }
    std::size_t last = item.size();
    while (last > first && std::isspace(static_cast<unsigned char>(item[last - 1]))) {
      --last;
    }
    columns.push_back(item.substr(first, last - first));
  }
  return columns;
}

std::vector<BatchMeshRow> load_mesh_list(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open mesh list: " + path);
  }
  std::vector<BatchMeshRow> rows;
  std::string line;
  std::size_t line_no = 0;
  while (std::getline(input, line)) {
    ++line_no;
    std::size_t first = 0;
    while (first < line.size() && std::isspace(static_cast<unsigned char>(line[first]))) {
      ++first;
    }
    if (first == line.size() || line[first] == '#') {
      continue;
    }
    const std::vector<std::string> columns = split_mesh_list_line(line);
    if (columns.size() < 2 || columns[0].empty() || columns[1].empty()) {
      throw std::runtime_error(
          "mesh list rows must be mesh_id<TAB>input[<TAB>bsp_obj] at line " +
          std::to_string(line_no));
    }
    BatchMeshRow row;
    row.mesh_id = columns[0];
    row.input = columns[1];
    if (columns.size() >= 3) {
      row.bsp_obj = columns[2];
    }
    rows.push_back(std::move(row));
  }
  if (rows.empty()) {
    throw std::runtime_error("mesh list is empty: " + path);
  }
  return rows;
}

std::string json_string_value(const std::string& text,
                              const std::string& key,
                              const std::string& fallback = "") {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = text.find(needle);
  if (key_pos == std::string::npos) {
    return fallback;
  }
  const std::size_t colon = text.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return fallback;
  }
  std::size_t quote = text.find('"', colon + 1);
  if (quote == std::string::npos) {
    return fallback;
  }
  std::string value;
  bool escaping = false;
  for (std::size_t idx = quote + 1; idx < text.size(); ++idx) {
    const char ch = text[idx];
    if (escaping) {
      switch (ch) {
        case 'n': value.push_back('\n'); break;
        case 'r': value.push_back('\r'); break;
        case 't': value.push_back('\t'); break;
        default: value.push_back(ch); break;
      }
      escaping = false;
      continue;
    }
    if (ch == '\\') {
      escaping = true;
      continue;
    }
    if (ch == '"') {
      return value;
    }
    value.push_back(ch);
  }
  return fallback;
}

double json_number_value_after(const std::string& text,
                               const std::string& key,
                               std::size_t start,
                               double fallback = 0.0) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = text.find(needle, start);
  if (key_pos == std::string::npos) {
    return fallback;
  }
  const std::size_t colon = text.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return fallback;
  }
  const char* begin = text.c_str() + colon + 1;
  char* end = nullptr;
  const double value = std::strtod(begin, &end);
  if (end == begin) {
    return fallback;
  }
  return value;
}

double json_number_value(const std::string& text,
                         const std::string& key,
                         double fallback = 0.0) {
  return json_number_value_after(text, key, 0, fallback);
}

bool json_bool_value(const std::string& text,
                     const std::string& key,
                     bool fallback = false) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = text.find(needle);
  if (key_pos == std::string::npos) {
    return fallback;
  }
  const std::size_t colon = text.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return fallback;
  }
  std::size_t cursor = colon + 1;
  while (cursor < text.size() &&
         std::isspace(static_cast<unsigned char>(text[cursor]))) {
    ++cursor;
  }
  if (text.compare(cursor, 4, "true") == 0) {
    return true;
  }
  if (text.compare(cursor, 5, "false") == 0) {
    return false;
  }
  return fallback;
}

struct StageSummary {
  std::size_t count = 0;
  double total_sec = 0.0;
  double max_sec = 0.0;
};

void collect_stage_timings(const std::string& stats_text,
                           std::map<std::string, StageSummary>& stages,
                           std::size_t& reuse_stage_count) {
  const std::size_t stages_pos = stats_text.find("\"stages\"");
  if (stages_pos == std::string::npos) {
    return;
  }
  std::size_t cursor = stages_pos;
  while (true) {
    const std::size_t name_pos = stats_text.find("\"name\"", cursor);
    if (name_pos == std::string::npos) {
      break;
    }
    const std::size_t elapsed_pos = stats_text.find("\"elapsed_sec\"", name_pos);
    if (elapsed_pos == std::string::npos) {
      break;
    }
    const std::string name = json_string_value(stats_text.substr(name_pos), "name");
    const double elapsed = json_number_value_after(stats_text, "elapsed_sec", name_pos, 0.0);
    if (!name.empty()) {
      StageSummary& summary = stages[name];
      summary.count += 1;
      summary.total_sec += elapsed;
      summary.max_sec = std::max(summary.max_sec, elapsed);
      if (name.size() >= 6 && name.substr(name.size() - 6) == "_reuse") {
        ++reuse_stage_count;
      }
    }
    cursor = elapsed_pos + 1;
  }
}

void append_option_if_present(std::vector<std::string>& args,
                              int argc,
                              char** argv,
                              const std::string& option) {
  const std::string value = arg_value(argc, argv, option);
  if (!value.empty()) {
    args.push_back(option);
    args.push_back(value);
  }
}

void append_flag_if_present(std::vector<std::string>& args,
                            int argc,
                            char** argv,
                            const std::string& flag) {
  if (has_flag(argc, argv, flag)) {
    args.push_back(flag);
  }
}

void append_run_pipeline_options(std::vector<std::string>& args,
                                 int argc,
                                 char** argv) {
  const std::vector<std::string> value_options = {
      "--epsilon",
      "--edge_length",
      "--merge_eps",
      "--refine_max_step",
      "--mcts_iter",
      "--mcts_max_step",
      "--normalize_mode",
      "--normalize_target",
      "--normalize_center",
      "--cover_penalty",
      "--pen_rate",
      "--refine_action_unit",
      "--mcts_action_unit",
      "--num_action_scale",
      "--manifold_timeout_sec",
      "--ftetwild_timeout_sec",
      "--ftetwild_level",
      "--retry_epsilon_scale",
      "--retry_edge_length_scale",
      "--coacd_timeout_sec",
      "--coacd_threshold",
      "--coacd_max_convex_hull",
      "--coacd_preprocess_mode",
      "--coacd_preprocess_resolution",
      "--coacd_resolution",
      "--coacd_mcts_nodes",
      "--coacd_mcts_iterations",
      "--coacd_mcts_max_depth",
      "--coacd_seed",
      "--partition_threads",
      "--exp_w",
      "--gamma",
      "--cache_capacity",
      "--volume_method",
      "--seed",
      "--final_k",
      "--ftetwild_threads",
      "--python",
      "--prior_logits_file",
      "--value_logits_file",
      "--ftetwild_threads_supported",
  };
  for (const auto& option : value_options) {
    append_option_if_present(args, argc, argv, option);
  }
  const std::vector<std::string> flags = {
      "--coarsen",
      "--no_use_floodfill",
      "--use_general_wn",
      "--use_input_for_wn",
      "--no_manifold_surface",
      "--skip_simplify",
      "--coacd_pca",
      "--coacd_no_merge",
      "--coacd_no_decimate",
      "--no_tilted",
      "--all_pairs",
      "--native_recenter",
      "--reuse_existing",
      "--no_stateful_union_cache",
      "--transposition_table",
  };
  for (const auto& flag : flags) {
    append_flag_if_present(args, argc, argv, flag);
  }
}

void run_batch_command(int argc, char** argv, const std::string& self_bin) {
  const std::string mesh_list = arg_value(argc, argv, "--mesh_list");
  const std::string data_root = arg_value(argc, argv, "--data_root");
  const std::string output_root = arg_value(argc, argv, "--output_root");
  const std::string manifoldplus_bin = arg_value(argc, argv, "--manifoldplus_bin");
  const std::string ftetwild_bin = arg_value(argc, argv, "--ftetwild_bin");
  const std::string coacd_bin = arg_value(argc, argv, "--coacd_bin");
  const std::string init_type = arg_value(argc, argv, "--init_type", "coacd");
  if ((mesh_list.empty() && data_root.empty()) || output_root.empty() || manifoldplus_bin.empty() ||
      ftetwild_bin.empty() || (init_type == "coacd" && coacd_bin.empty())) {
    throw std::runtime_error(
        "run-batch requires --mesh_list or --data_root, --output_root, --manifoldplus_bin, "
        "--ftetwild_bin, and --coacd_bin when --init_type coacd");
  }
  if (init_type != "coacd" && init_type != "bsp") {
    throw std::runtime_error("run-batch --init_type must be coacd or bsp");
  }
  std::vector<BatchMeshRow> rows;
  DiscoveryResult discovery;
  if (!mesh_list.empty()) {
    rows = load_mesh_list(mesh_list);
  } else {
    const std::string model_name = arg_value(argc, argv, "--model_name", "model.obj");
    const std::string bsp_name = arg_value(argc, argv, "--bsp_name", "bsp_seg.obj");
    const std::vector<std::string> category_filter =
        split_csv(arg_value(argc, argv, "--categories"));
    const std::size_t limit_per_category =
        static_cast<std::size_t>(std::stoull(arg_value(argc, argv, "--limit_per_category", "0")));
    const bool require_bsp = has_flag(argc, argv, "--require_bsp") || init_type == "bsp";
    discovery = discover_mesh_rows_from_data_root(
        data_root, model_name, bsp_name, category_filter, limit_per_category, require_bsp);
    rows = discovery.rows;
    if (rows.empty()) {
      throw std::runtime_error("run-batch discovered no meshes under: " + data_root);
    }
    const std::string discovered_mesh_list =
        arg_value(argc, argv, "--discovered_mesh_list",
                  path_join(output_root, "native_meshes.tsv"));
    write_mesh_rows_tsv(discovered_mesh_list, rows);
  }
  ensure_directories(output_root);
  const std::string manifest =
      arg_value(argc, argv, "--manifest", path_join(output_root, "native_pipeline.jsonl"));
  const std::string manifest_parent = parent_path(manifest);
  if (!manifest_parent.empty()) {
    ensure_directories(manifest_parent);
  }
  const std::ios_base::openmode manifest_mode =
      std::ios::out |
      (has_flag(argc, argv, "--append_manifest") ? std::ios::app : std::ios::trunc);
  std::ofstream manifest_out(manifest, manifest_mode);
  if (!manifest_out) {
    throw std::runtime_error("failed to open batch manifest: " + manifest);
  }

  const bool fail_fast = has_flag(argc, argv, "--fail_fast");
  const bool batch_subprocess = has_flag(argc, argv, "--batch_subprocess");
  const bool resume_success = has_flag(argc, argv, "--resume_success");
  const bool use_category_tetra_defaults =
      !has_flag(argc, argv, "--no_category_tetra_defaults");
  const bool has_global_epsilon = !arg_value(argc, argv, "--epsilon").empty();
  const bool has_global_edge_length = !arg_value(argc, argv, "--edge_length").empty();
  std::string ftetwild_threads_supported_override;
  if (!arg_value(argc, argv, "--ftetwild_threads").empty() &&
      arg_value(argc, argv, "--ftetwild_threads_supported").empty()) {
    ensure_directories(path_join(output_root, "logs"));
    const bool supported = executable_supports_option(
        ftetwild_bin, "--max-threads",
        path_join(path_join(output_root, "logs"), "ftetwild_help.log"));
    ftetwild_threads_supported_override = supported ? "true" : "false";
  }
  std::size_t requested_jobs =
      resolve_thread_count(arg_value(argc, argv, "--jobs", "1"), rows.size(), 1);
  const bool parallel = requested_jobs > 1 && rows.size() > 1 && !fail_fast;
  const bool execute_subprocess = batch_subprocess || parallel;
  const std::string pipeline_execution =
      parallel ? "subprocess_parallel" : (execute_subprocess ? "subprocess" : "in_process");
  const auto batch_started = std::chrono::steady_clock::now();

  auto run_one_row = [&](const BatchMeshRow& row) -> BatchRunRecord {
    BatchRunRecord record;
    record.row = row;
    record.work_dir = path_join(output_root, row.mesh_id);
    record.log_path = path_join(path_join(record.work_dir, "logs"), "run_pipeline.log");
    record.stats_path = path_join(record.work_dir, "native_pipeline_stats.json");
    const std::string work_dir = path_join(output_root, row.mesh_id);
    std::vector<std::string> args = {
        self_bin,
        "run-pipeline",
        "--input",
        row.input,
        "--work_dir",
        record.work_dir,
        "--manifoldplus_bin",
        manifoldplus_bin,
        "--ftetwild_bin",
        ftetwild_bin,
        "--init_type",
        init_type,
        "--mesh_id",
        row.mesh_id,
    };
    if (init_type == "coacd") {
      args.push_back("--coacd_bin");
      args.push_back(coacd_bin);
    } else {
      const std::string bsp_obj = !row.bsp_obj.empty()
                                      ? row.bsp_obj
                                      : arg_value(argc, argv, "--bsp_obj");
      if (!bsp_obj.empty()) {
        args.push_back("--bsp_obj");
        args.push_back(bsp_obj);
      }
    }
    append_run_pipeline_options(args, argc, argv);
    if (!ftetwild_threads_supported_override.empty()) {
      args.push_back("--ftetwild_threads_supported");
      args.push_back(ftetwild_threads_supported_override);
    }
    TetraDefaults tetra_defaults;
    if (use_category_tetra_defaults && (!has_global_epsilon || !has_global_edge_length)) {
      tetra_defaults = smart_category_tetra_defaults(row);
      if (!tetra_defaults.epsilon.empty() && !has_global_epsilon) {
        args.push_back("--epsilon");
        args.push_back(tetra_defaults.epsilon);
      }
      if (!tetra_defaults.edge_length.empty() && !has_global_edge_length) {
        args.push_back("--edge_length");
        args.push_back(tetra_defaults.edge_length);
      }
    }
    record.tetra_default_source = tetra_defaults.source;
    record.tetra_epsilon =
        has_global_epsilon ? arg_value(argc, argv, "--epsilon") : tetra_defaults.epsilon;
    record.tetra_edge_length = has_global_edge_length
                                   ? arg_value(argc, argv, "--edge_length")
                                   : tetra_defaults.edge_length;
    record.pipeline_execution = pipeline_execution;

    if (resume_success && file_exists(record.stats_path)) {
      const std::string stats_text = read_text(record.stats_path);
      if (json_string_value(stats_text, "status") == "success") {
        record.pipeline_execution = "resume_success";
        record.elapsed_sec = 0.0;
        record.return_code = 0;
        record.status = "success";
        return record;
      }
    }

    const auto started = std::chrono::steady_clock::now();
    try {
      if (execute_subprocess) {
        const ProcessResult result = run_process(args, record.log_path, 0.0);
        record.return_code = result.return_code;
        record.elapsed_sec = result.elapsed_sec;
        if (record.return_code != 0) {
          record.status = "failed";
          record.error =
              "run-pipeline exited with code " + std::to_string(record.return_code);
        }
      } else {
        run_cpp_stage_to_log(record.log_path, "run-pipeline", [&]() {
          std::vector<char*> pipeline_argv;
          pipeline_argv.reserve(args.size() + 1);
          for (auto& arg : args) {
            pipeline_argv.push_back(const_cast<char*>(arg.c_str()));
          }
          pipeline_argv.push_back(nullptr);
          run_pipeline_command(static_cast<int>(args.size()), pipeline_argv.data(), self_bin);
        });
        const auto finished = std::chrono::steady_clock::now();
        record.elapsed_sec = std::chrono::duration<double>(finished - started).count();
      }
    } catch (const std::exception& exc) {
      record.status = "failed";
      record.error = exc.what();
      const auto finished = std::chrono::steady_clock::now();
      record.elapsed_sec = std::chrono::duration<double>(finished - started).count();
      record.return_code = 2;
    }
    return record;
  };

  std::vector<BatchRunRecord> records;
  if (parallel) {
    records.resize(rows.size());
    std::atomic<std::size_t> next_index{0};
    const std::size_t worker_count = std::min<std::size_t>(requested_jobs, rows.size());
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
      workers.emplace_back([&]() {
        while (true) {
          const std::size_t row_idx = next_index.fetch_add(1);
          if (row_idx >= rows.size()) {
            break;
          }
          records[row_idx] = run_one_row(rows[row_idx]);
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
  } else {
    for (const auto& row : rows) {
      records.push_back(run_one_row(row));
      if (fail_fast && records.back().status != "success") {
        break;
      }
    }
  }

  std::size_t success = 0;
  std::size_t failed = 0;
  for (const auto& record : records) {
    if (record.status == "success") {
      ++success;
    } else {
      ++failed;
    }
    manifest_out << "{"
                 << "\"stage\":\"native_pipeline\","
                 << "\"backend\":\"smart-cpp-native\","
                 << "\"command\":\"run-batch\","
                 << "\"mesh_id\":\"" << json_escape(record.row.mesh_id) << "\","
                 << "\"input\":\"" << json_escape(record.row.input) << "\","
                 << "\"init_type\":\"" << json_escape(init_type) << "\","
                 << "\"pipeline_execution\":\""
                 << json_escape(record.pipeline_execution) << "\","
                 << "\"tetra_default_source\":\""
                 << json_escape(record.tetra_default_source) << "\","
                 << "\"tetra_epsilon\":\""
                 << json_escape(record.tetra_epsilon) << "\","
                 << "\"tetra_edge_length\":\""
                 << json_escape(record.tetra_edge_length) << "\","
                 << "\"work_dir\":\"" << json_escape(record.work_dir) << "\","
                 << "\"log_path\":\"" << json_escape(record.log_path) << "\","
                 << "\"stats_path\":\"" << json_escape(record.stats_path) << "\","
                 << "\"status\":\"" << record.status << "\","
                 << "\"return_code\":" << record.return_code << ","
                 << "\"elapsed_sec\":" << std::setprecision(17)
                 << record.elapsed_sec;
    if (!record.error.empty()) {
      manifest_out << ",\"error\":\"" << json_escape(record.error) << "\"";
    }
    manifest_out << "}\n";
    manifest_out.flush();
  }
  std::cout << "{"
            << "\"status\":\"success\","
            << "\"backend\":\"smart-cpp-native\","
            << "\"command\":\"run-batch\","
            << "\"pipeline_execution\":\""
            << pipeline_execution << "\","
            << "\"jobs\":" << (parallel ? requested_jobs : 1) << ","
            << "\"mesh_count\":" << rows.size() << ","
            << "\"attempted\":" << records.size() << ","
            << "\"success\":" << success << ","
            << "\"failed\":" << failed << ","
            << "\"elapsed_sec\":" << std::setprecision(17)
            << std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - batch_started).count() << ","
            << "\"discovered_missing_bsp\":" << discovery.skipped_missing_bsp << ","
            << "\"manifest\":\"" << json_escape(manifest) << "\""
            << "}\n";
}

void run_batch_summary_command(int argc, char** argv) {
  const std::string manifest = arg_value(argc, argv, "--manifest");
  if (manifest.empty()) {
    throw std::runtime_error("batch-summary requires --manifest");
  }
  std::ifstream input(manifest);
  if (!input) {
    throw std::runtime_error("failed to open batch manifest: " + manifest);
  }

  std::size_t records = 0;
  std::size_t success = 0;
  std::size_t failed = 0;
  std::size_t missing_stats = 0;
  std::size_t reuse_stage_count = 0;
  double total_elapsed = 0.0;
  double max_elapsed = 0.0;
  std::string slowest_mesh;
  std::map<std::string, std::size_t> status_counts;
  std::map<std::string, std::size_t> execution_counts;
  std::map<std::string, std::size_t> category_counts;
  std::map<std::string, StageSummary> stages;

  std::string line;
  while (std::getline(input, line)) {
    line = trim_copy(line);
    if (line.empty()) {
      continue;
    }
    ++records;
    const std::string status = json_string_value(line, "status", "unknown");
    const std::string mesh_id = json_string_value(line, "mesh_id", "");
    const std::string execution = json_string_value(line, "pipeline_execution", "unknown");
    std::string stats_path = json_string_value(line, "stats_path", "");
    if (stats_path.empty()) {
      const std::string output_path = json_string_value(line, "output_path", "");
      if (!output_path.empty()) {
        stats_path = path_join(output_path, "native_pipeline_stats.json");
      }
    }
    if (stats_path.empty()) {
      const std::string work_dir = json_string_value(line, "work_dir", "");
      if (!work_dir.empty()) {
        stats_path = path_join(work_dir, "native_pipeline_stats.json");
      }
    }
    const double elapsed = json_number_value(line, "elapsed_sec", 0.0);
    ++status_counts[status];
    ++execution_counts[execution];
    if (status == "success") {
      ++success;
    } else {
      ++failed;
    }
    total_elapsed += elapsed;
    if (elapsed > max_elapsed) {
      max_elapsed = elapsed;
      slowest_mesh = mesh_id;
    }
    std::string category = json_string_value(line, "category", "");
    if (category.empty()) {
      const std::size_t separator = mesh_id.find("__");
      if (separator != std::string::npos) {
        category = mesh_id.substr(0, separator);
      }
    }
    if (!category.empty()) {
      ++category_counts[category];
    }
    if (!stats_path.empty() && file_exists(stats_path)) {
      collect_stage_timings(read_text(stats_path), stages, reuse_stage_count);
    } else {
      ++missing_stats;
    }
  }

  std::string slowest_stage_total;
  std::string slowest_stage_max;
  double slowest_stage_total_sec = 0.0;
  double slowest_stage_max_sec = 0.0;
  for (const auto& entry : stages) {
    if (entry.second.total_sec > slowest_stage_total_sec) {
      slowest_stage_total_sec = entry.second.total_sec;
      slowest_stage_total = entry.first;
    }
    if (entry.second.max_sec > slowest_stage_max_sec) {
      slowest_stage_max_sec = entry.second.max_sec;
      slowest_stage_max = entry.first;
    }
  }

  std::cout << "{"
            << "\"status\":\"success\","
            << "\"backend\":\"smart-cpp-native\","
            << "\"command\":\"batch-summary\","
            << "\"manifest\":\"" << json_escape(manifest) << "\","
            << "\"records\":" << records << ","
            << "\"success\":" << success << ","
            << "\"failed\":" << failed << ","
            << "\"missing_stats\":" << missing_stats << ","
            << "\"reuse_stage_count\":" << reuse_stage_count << ","
            << "\"total_elapsed_sec\":" << std::setprecision(17) << total_elapsed << ","
            << "\"average_elapsed_sec\":"
            << std::setprecision(17)
            << (records == 0 ? 0.0 : total_elapsed / static_cast<double>(records)) << ","
            << "\"max_elapsed_sec\":" << std::setprecision(17) << max_elapsed << ","
            << "\"slowest_mesh\":\"" << json_escape(slowest_mesh) << "\","
            << "\"slowest_stage_by_total\":{"
            << "\"name\":\"" << json_escape(slowest_stage_total) << "\","
            << "\"elapsed_sec\":" << std::setprecision(17)
            << slowest_stage_total_sec << "},"
            << "\"slowest_stage_by_single\":{"
            << "\"name\":\"" << json_escape(slowest_stage_max) << "\","
            << "\"elapsed_sec\":" << std::setprecision(17)
            << slowest_stage_max_sec << "},";

  auto write_counts = [](const std::map<std::string, std::size_t>& counts) {
    std::cout << "{";
    bool first = true;
    for (const auto& entry : counts) {
      if (!first) {
        std::cout << ",";
      }
      first = false;
      std::cout << "\"" << json_escape(entry.first) << "\":" << entry.second;
    }
    std::cout << "}";
  };

  std::cout << "\"status_counts\":";
  write_counts(status_counts);
  std::cout << ",\"pipeline_execution_counts\":";
  write_counts(execution_counts);
  std::cout << ",\"category_counts\":";
  write_counts(category_counts);
  std::cout << ",\"stages\":{";
  bool first = true;
  for (const auto& entry : stages) {
    if (!first) {
      std::cout << ",";
    }
    first = false;
    const StageSummary& stage = entry.second;
    std::cout << "\"" << json_escape(entry.first) << "\":{"
              << "\"count\":" << stage.count << ","
              << "\"total_sec\":" << std::setprecision(17)
              << stage.total_sec << ","
              << "\"average_sec\":" << std::setprecision(17)
              << (stage.count == 0 ? 0.0
                  : stage.total_sec / static_cast<double>(stage.count)) << ","
              << "\"max_sec\":" << std::setprecision(17)
              << stage.max_sec << "}";
  }
  std::cout << "}}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "--help" ||
        std::string(argv[1]) == "-h") {
      usage();
      return argc < 2 ? 1 : 0;
    }
    const std::string command = argv[1];
    if (command == "normalize") {
      const std::string input = arg_value(argc, argv, "--input");
      const std::string output = arg_value(argc, argv, "--output");
      if (input.empty() || output.empty()) {
        throw std::runtime_error("normalize requires --input and --output");
      }
      normalize_obj(input, output,
                    arg_value(argc, argv, "--mode", "bbox_diagonal"),
                    arg_value(argc, argv, "--center", "bbox"),
                    std::stod(arg_value(argc, argv, "--target", "1.0")));
      return 0;
    }
    if (command == "obj-info") {
      const std::string input = arg_value(argc, argv, "--input");
      if (input.empty()) {
        throw std::runtime_error("obj-info requires --input");
      }
      obj_info(input);
      return 0;
    }
    if (command == "gmsh-info") {
      const std::string msh = arg_value(argc, argv, "--msh");
      if (msh.empty()) {
        throw std::runtime_error("gmsh-info requires --msh");
      }
      gmsh_info(msh);
      return 0;
    }
    if (command == "split-obj-parts") {
      run_split_obj_parts_command(argc, argv);
      return 0;
    }
    if (command == "partition-coacd") {
      run_partition_coacd_command(argc, argv);
      return 0;
    }
    if (command == "partition-bsp") {
      run_partition_bsp_command(argc, argv);
      return 0;
    }
    if (command == "merge") {
      run_merge_command(argc, argv);
      return 0;
    }
    if (command == "refine") {
      run_refine_command(argc, argv);
      return 0;
    }
    if (command == "mcts") {
      run_mcts_command(argc, argv);
      return 0;
    }
    if (command == "refine-mcts") {
      run_refine_mcts_command(argc, argv);
      return 0;
    }
    if (command == "run-pipeline") {
      run_pipeline_command(argc, argv, argv[0]);
      return 0;
    }
    if (command == "discover-meshes") {
      run_discover_meshes_command(argc, argv);
      return 0;
    }
    if (command == "run-batch") {
      run_batch_command(argc, argv, argv[0]);
      return 0;
    }
    if (command == "batch-summary") {
      run_batch_summary_command(argc, argv);
      return 0;
    }
    throw std::runtime_error("unknown smart-cpp-native command: " + command);
  } catch (const std::exception& exc) {
    std::cerr << "smart-cpp-native: " << exc.what() << '\n';
    return 2;
  }
}
