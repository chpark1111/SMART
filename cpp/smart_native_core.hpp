#pragma once

#include <cstddef>
#include <cstdint>

extern "C" std::size_t smart_native_action_count(std::size_t num_bbox,
                                                  std::size_t num_action_scale);

extern "C" int smart_native_action_scales(std::size_t num_action_scale,
                                           double* out_scales);

extern "C" int smart_native_action_indices(std::size_t num_bbox,
                                            std::size_t num_action_scale,
                                            std::size_t* out_triplets);

extern "C" int smart_native_opposite_actions(std::size_t num_bbox,
                                              std::size_t num_action_scale,
                                              std::size_t* out_actions);

extern "C" int smart_native_child_action_mask(std::size_t num_actions,
                                               std::size_t action,
                                               std::size_t num_action_scale,
                                               const std::uint8_t* parent_mask,
                                               std::uint8_t* out_mask);

extern "C" double smart_native_discounted_reward(const double* rewards,
                                                  std::size_t n_rewards,
                                                  double gamma);

extern "C" int smart_native_best_ucb_child(std::size_t parent_visits,
                                            const double* child_qs,
                                            const std::size_t* child_visits,
                                            std::size_t n_children,
                                            double exp_weight,
                                            std::size_t tie_pick,
                                            std::size_t* out_position);

extern "C" int smart_native_ucb_best_count(std::size_t parent_visits,
                                            const double* child_qs,
                                            const std::size_t* child_visits,
                                            std::size_t n_children,
                                            double exp_weight,
                                            std::size_t* out_count);

extern "C" int smart_native_prob_skip_exploration(double parent_reward,
                                                   const double* child_rewards,
                                                   const double* child_qs,
                                                   std::size_t n_children,
                                                   double best_reward,
                                                   double skip_rate,
                                                   double* out_probability);

extern "C" int smart_native_bbox_volumes(const double* bounds,
                                          std::size_t n_boxes,
                                          double* out_volumes);

extern "C" int smart_native_bbox_valid_mask(const double* bounds,
                                             std::size_t n_boxes,
                                             std::uint8_t* out_mask);

extern "C" int smart_native_total_bbox_volume(const double* bounds,
                                               std::size_t n_boxes,
                                               double* out_total_volume);

extern "C" int smart_native_bbox_union_bounds(const double* bounds,
                                               std::size_t n_boxes,
                                               double* out_bounds);

extern "C" int smart_native_bbox_union_volume(const double* bounds,
                                               std::size_t n_boxes,
                                               double* out_volume);

extern "C" int smart_native_coverage_mask(const double* points,
                                           std::size_t n_points,
                                           const double* bounds,
                                           std::uint8_t* out_mask);

extern "C" int smart_native_apply_axis_action(const double* bounds,
                                               std::size_t n_boxes,
                                               std::size_t action,
                                               std::size_t num_action_scale,
                                               double action_unit,
                                               double* out_bounds);

extern "C" int smart_native_normalize_vertices(const double* vertices,
                                                std::size_t n_vertices,
                                                int mode,
                                                int center_mode,
                                                double target,
                                                double* out_vertices,
                                                double* out_stats);

extern "C" int smart_native_tetra_volumes(const double* vertices,
                                           std::size_t n_vertices,
                                           const std::size_t* voxels,
                                           std::size_t n_voxels,
                                           double* out_volumes);

extern "C" int smart_native_tetra_centroids(const double* vertices,
                                             std::size_t n_vertices,
                                             const std::size_t* voxels,
                                             std::size_t n_voxels,
                                             double* out_centroids);

extern "C" int smart_native_tetra_surface_faces(const std::size_t* voxels,
                                                 std::size_t n_voxels,
                                                 std::size_t* out_faces,
                                                 std::size_t* out_n_faces);

extern "C" int smart_native_tetra_adjacency(const std::size_t* voxels,
                                             std::size_t n_voxels,
                                             std::size_t* out_offsets,
                                             std::size_t* out_values,
                                             std::size_t values_capacity,
                                             std::size_t* out_n_values);

extern "C" int smart_native_load_gmsh_counts(const char* path,
                                              std::size_t* out_n_vertices,
                                              std::size_t* out_n_faces,
                                              std::size_t* out_n_voxels);

extern "C" int smart_native_load_gmsh(const char* path,
                                       double* out_vertices,
                                       std::size_t* out_faces,
                                       std::size_t* out_voxels,
                                       std::size_t vertex_capacity,
                                       std::size_t face_capacity,
                                       std::size_t voxel_capacity,
                                       std::size_t* out_n_vertices,
                                       std::size_t* out_n_faces,
                                       std::size_t* out_n_voxels);

extern "C" int smart_native_save_gmsh(const char* path,
                                       const double* vertices,
                                       std::size_t n_vertices,
                                       const std::size_t* faces,
                                       std::size_t n_faces,
                                       const std::size_t* voxels,
                                       std::size_t n_voxels);

extern "C" int smart_native_symmetric_chamfer(const double* left,
                                               std::size_t n_left,
                                               const double* right,
                                               std::size_t n_right,
                                               double* out_distance);

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
    std::size_t* out_n_rewards);

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
    std::size_t* out_n_points);

extern "C" int smart_native_action_upper_rewards(const double* bounds,
                                                  std::size_t n_boxes,
                                                  std::size_t num_action_scale,
                                                  double action_unit,
                                                  double volume_sum,
                                                  double last_bbox_score,
                                                  double* out_rewards);

extern "C" int smart_native_bbox_action_upper_rewards(const double* bounds,
                                                       std::size_t n_boxes,
                                                       std::size_t bbox_idx,
                                                       std::size_t num_action_scale,
                                                       double action_unit,
                                                       double volume_sum,
                                                       double last_bbox_score,
                                                       double* out_rewards);

extern "C" int smart_native_bavf_scores(const double* part_volumes,
                                         const double* bbox_volumes,
                                         std::size_t n_items,
                                         double alpha,
                                         double* out_scores);

extern "C" int smart_native_merge_bavf_reward(double prev_bvs,
                                               double left_bbox_volume,
                                               double right_bbox_volume,
                                               double merged_bbox_volume,
                                               double shape_volume,
                                               double* out_reward);

extern "C" int smart_native_softmax_scaled(const double* values,
                                            std::size_t n_values,
                                            double scale,
                                            double* out_probs);

extern "C" double smart_native_incremental_average(double previous,
                                                    std::size_t count,
                                                    double value);
