use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fs;

const RUST_ACTION_SELECTION_MIN_ACTIONS: usize = 32;

#[cfg(not(smart_no_manifold_bridge))]
extern "C" {
    fn smart_manifold_cube_volume(x: f32, y: f32, z: f32) -> f32;
    fn smart_manifold_mesh_new(
        vertices: *const f32,
        n_vertices: usize,
        faces: *const u32,
        n_faces: usize,
    ) -> *mut std::ffi::c_void;
    fn smart_manifold_delete(handle: *mut std::ffi::c_void);
    fn smart_manifold_handle_volume(handle: *mut std::ffi::c_void) -> f64;
    fn smart_manifold_handle_volume_properties(handle: *mut std::ffi::c_void) -> f64;
    fn smart_manifold_residual_volume_for_boxes(
        handle: *mut std::ffi::c_void,
        box_vertices: *const f32,
        n_boxes: usize,
    ) -> f64;
    fn smart_manifold_residual_volume_for_boxes_properties(
        handle: *mut std::ffi::c_void,
        box_vertices: *const f32,
        n_boxes: usize,
    ) -> f64;
    fn smart_manifold_residual_volume_for_boxes_pair(
        handle: *mut std::ffi::c_void,
        box_vertices: *const f32,
        n_boxes: usize,
        out_mesh_volume: *mut f64,
        out_properties_volume: *mut f64,
    ) -> i32;
    fn smart_manifold_best_axis_actions_for_mask(
        handle: *mut std::ffi::c_void,
        bounds: *const f64,
        rotations: *const f64,
        bbox_mask: *const u8,
        n_boxes: usize,
        num_action_scale: usize,
        action_unit: f64,
        volume_sum: f64,
        last_bbox_score: f64,
        cover_penalty: f64,
        pen_rate: f64,
        initial_best: f64,
        action_scales: *const f64,
        out_actions: *mut isize,
        out_rewards: *mut f64,
        volume_method: i32,
    ) -> i32;
    fn smart_manifold_state_new(
        vertices: *const f32,
        n_vertices: usize,
        faces: *const u32,
        n_faces: usize,
        bounds: *const f64,
        rotations: *const f64,
        n_boxes: usize,
        volume_sum: f64,
        last_bbox_score: f64,
        stateful_union_cache: i32,
        cache_capacity: usize,
        volume_method: i32,
    ) -> *mut std::ffi::c_void;
    fn smart_manifold_state_delete(handle: *mut std::ffi::c_void);
    fn smart_manifold_state_reset(
        handle: *mut std::ffi::c_void,
        bounds: *const f64,
        rotations: *const f64,
        n_boxes: usize,
        last_bbox_score: f64,
    ) -> i32;
    fn smart_manifold_state_num_boxes(handle: *mut std::ffi::c_void) -> usize;
    fn smart_manifold_state_copy(
        handle: *mut std::ffi::c_void,
        out_bounds: *mut f64,
        out_rotations: *mut f64,
    ) -> i32;
    fn smart_manifold_state_copy_bbox(
        handle: *mut std::ffi::c_void,
        bbox_idx: usize,
        out_bounds: *mut f64,
        out_rotation: *mut f64,
    ) -> i32;
    fn smart_manifold_state_last_bbox_score(handle: *mut std::ffi::c_void) -> f64;
    fn smart_manifold_state_total_bbox_volume(handle: *mut std::ffi::c_void) -> f64;
    fn smart_manifold_state_valid_count(handle: *mut std::ffi::c_void) -> usize;
    fn smart_manifold_state_cache_stats(handle: *mut std::ffi::c_void, out_values: *mut u64)
        -> i32;
    fn smart_manifold_state_covered(handle: *mut std::ffi::c_void) -> f64;
    fn smart_manifold_state_score(
        handle: *mut std::ffi::c_void,
        cover_penalty: f64,
        pen_rate: f64,
    ) -> f64;
    fn smart_manifold_state_score_axis_action(
        handle: *mut std::ffi::c_void,
        action: isize,
        num_action_scale: usize,
        action_unit: f64,
        cover_penalty: f64,
        pen_rate: f64,
        action_scales: *const f64,
    ) -> f64;
    fn smart_manifold_state_score_replacement(
        handle: *mut std::ffi::c_void,
        bbox_idx: usize,
        candidate_bounds: *const f64,
        candidate_rotation: *const f64,
        cover_penalty: f64,
        pen_rate: f64,
    ) -> f64;
    fn smart_manifold_state_best_axis_actions_for_mask(
        handle: *mut std::ffi::c_void,
        bbox_mask: *const u8,
        num_action_scale: usize,
        action_unit: f64,
        cover_penalty: f64,
        pen_rate: f64,
        initial_best: f64,
        action_scales: *const f64,
        out_actions: *mut isize,
        out_rewards: *mut f64,
    ) -> i32;
    fn smart_manifold_state_apply_axis_action(
        handle: *mut std::ffi::c_void,
        action: isize,
        num_action_scale: usize,
        action_unit: f64,
        cover_penalty: f64,
        pen_rate: f64,
        action_scales: *const f64,
    ) -> f64;
    fn smart_manifold_state_rollback(handle: *mut std::ffi::c_void) -> i32;
    fn smart_manifold_mesh_volume(
        vertices: *const f32,
        n_vertices: usize,
        faces: *const u32,
        n_faces: usize,
    ) -> f32;
    fn smart_manifold_axis_box_intersection_volume(
        vertices: *const f32,
        n_vertices: usize,
        faces: *const u32,
        n_faces: usize,
        lx: f32,
        ly: f32,
        lz: f32,
        rx: f32,
        ry: f32,
        rz: f32,
    ) -> f32;
}

#[pyfunction]
fn manifold_bridge_available() -> bool {
    cfg!(not(smart_no_manifold_bridge))
}

#[pyfunction]
fn manifold_cube_volume(x: f64, y: f64, z: f64) -> PyResult<f64> {
    if x < 0.0 || y < 0.0 || z < 0.0 {
        return Err(PyValueError::new_err(
            "cube dimensions must be non-negative",
        ));
    }
    #[cfg(not(smart_no_manifold_bridge))]
    {
        Ok(unsafe { smart_manifold_cube_volume(x as f32, y as f32, z as f32) } as f64)
    }
    #[cfg(smart_no_manifold_bridge)]
    {
        let _ = (x, y, z);
        Err(PyValueError::new_err(
            "Manifold C++ bridge is unavailable in this build",
        ))
    }
}

#[pyfunction]
fn manifold_mesh_volume(vertices: Vec<Vec<f64>>, faces: Vec<Vec<usize>>) -> PyResult<f64> {
    let (flat_vertices, flat_faces) = flatten_bridge_mesh(&vertices, &faces)?;
    #[cfg(not(smart_no_manifold_bridge))]
    {
        let volume = unsafe {
            smart_manifold_mesh_volume(
                flat_vertices.as_ptr(),
                vertices.len(),
                flat_faces.as_ptr(),
                faces.len(),
            )
        };
        if volume.is_finite() {
            Ok(volume as f64)
        } else {
            Err(PyValueError::new_err("Manifold bridge mesh volume failed"))
        }
    }
    #[cfg(smart_no_manifold_bridge)]
    {
        let _ = (flat_vertices, flat_faces);
        Err(PyValueError::new_err(
            "Manifold C++ bridge is unavailable in this build",
        ))
    }
}

#[pyfunction]
fn manifold_axis_box_intersection_volume(
    vertices: Vec<Vec<f64>>,
    faces: Vec<Vec<usize>>,
    bounds: Vec<f64>,
) -> PyResult<f64> {
    check_bounds(&bounds)?;
    let (flat_vertices, flat_faces) = flatten_bridge_mesh(&vertices, &faces)?;
    #[cfg(not(smart_no_manifold_bridge))]
    {
        let volume = unsafe {
            smart_manifold_axis_box_intersection_volume(
                flat_vertices.as_ptr(),
                vertices.len(),
                flat_faces.as_ptr(),
                faces.len(),
                bounds[0] as f32,
                bounds[1] as f32,
                bounds[2] as f32,
                bounds[3] as f32,
                bounds[4] as f32,
                bounds[5] as f32,
            )
        };
        if volume.is_finite() {
            Ok(volume as f64)
        } else {
            Err(PyValueError::new_err(
                "Manifold bridge axis-box intersection failed",
            ))
        }
    }
    #[cfg(smart_no_manifold_bridge)]
    {
        let _ = (flat_vertices, flat_faces);
        Err(PyValueError::new_err(
            "Manifold C++ bridge is unavailable in this build",
        ))
    }
}

#[pyclass(unsendable)]
struct ManifoldBridgeMesh {
    ptr: *mut std::ffi::c_void,
}

impl Drop for ManifoldBridgeMesh {
    fn drop(&mut self) {
        #[cfg(not(smart_no_manifold_bridge))]
        unsafe {
            if !self.ptr.is_null() {
                smart_manifold_delete(self.ptr);
                self.ptr = std::ptr::null_mut();
            }
        }
    }
}

#[pymethods]
impl ManifoldBridgeMesh {
    #[new]
    fn new(vertices: Vec<Vec<f64>>, faces: Vec<Vec<usize>>) -> PyResult<Self> {
        let (flat_vertices, flat_faces) = flatten_bridge_mesh(&vertices, &faces)?;
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let ptr = unsafe {
                smart_manifold_mesh_new(
                    flat_vertices.as_ptr(),
                    vertices.len(),
                    flat_faces.as_ptr(),
                    faces.len(),
                )
            };
            if ptr.is_null() {
                Err(PyValueError::new_err(
                    "Manifold bridge mesh construction failed",
                ))
            } else {
                Ok(Self { ptr })
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = (flat_vertices, flat_faces);
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn volume(&self) -> PyResult<f64> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let volume = unsafe { smart_manifold_handle_volume(self.ptr) };
            if volume.is_finite() {
                Ok(volume)
            } else {
                Err(PyValueError::new_err("Manifold bridge mesh volume failed"))
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn volume_properties(&self) -> PyResult<f64> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let volume = unsafe { smart_manifold_handle_volume_properties(self.ptr) };
            if volume.is_finite() {
                Ok(volume)
            } else {
                Err(PyValueError::new_err(
                    "Manifold bridge properties volume failed",
                ))
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn residual_volume_for_boxes(&self, box_vertices: Vec<Vec<Vec<f64>>>) -> PyResult<f64> {
        let flat_boxes = flatten_bridge_box_vertices(&box_vertices)?;
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let residual = unsafe {
                smart_manifold_residual_volume_for_boxes(
                    self.ptr,
                    flat_boxes.as_ptr(),
                    box_vertices.len(),
                )
            };
            if residual.is_finite() {
                Ok(residual)
            } else {
                Err(PyValueError::new_err(
                    "Manifold bridge residual volume evaluation failed",
                ))
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = flat_boxes;
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn residual_volume_for_boxes_properties(
        &self,
        box_vertices: Vec<Vec<Vec<f64>>>,
    ) -> PyResult<f64> {
        let flat_boxes = flatten_bridge_box_vertices(&box_vertices)?;
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let residual = unsafe {
                smart_manifold_residual_volume_for_boxes_properties(
                    self.ptr,
                    flat_boxes.as_ptr(),
                    box_vertices.len(),
                )
            };
            if residual.is_finite() {
                Ok(residual)
            } else {
                Err(PyValueError::new_err(
                    "Manifold bridge properties residual volume evaluation failed",
                ))
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = flat_boxes;
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn residual_volume_for_boxes_pair(
        &self,
        box_vertices: Vec<Vec<Vec<f64>>>,
    ) -> PyResult<(f64, f64)> {
        let flat_boxes = flatten_bridge_box_vertices(&box_vertices)?;
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let mut mesh_volume = 0.0_f64;
            let mut properties_volume = 0.0_f64;
            let ok = unsafe {
                smart_manifold_residual_volume_for_boxes_pair(
                    self.ptr,
                    flat_boxes.as_ptr(),
                    box_vertices.len(),
                    &mut mesh_volume,
                    &mut properties_volume,
                )
            };
            if ok == 1 && mesh_volume.is_finite() && properties_volume.is_finite() {
                Ok((mesh_volume, properties_volume))
            } else {
                Err(PyValueError::new_err(
                    "Manifold bridge residual volume pair evaluation failed",
                ))
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = flat_boxes;
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn residual_volume_for_box_params(
        &self,
        bounds: Vec<Vec<f64>>,
        rotations: Vec<Vec<f64>>,
    ) -> PyResult<f64> {
        let flat_boxes = flatten_bridge_oriented_box_vertices(&bounds, &rotations)?;
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let residual = unsafe {
                smart_manifold_residual_volume_for_boxes(
                    self.ptr,
                    flat_boxes.as_ptr(),
                    bounds.len(),
                )
            };
            if residual.is_finite() {
                Ok(residual)
            } else {
                Err(PyValueError::new_err(
                    "Manifold bridge residual volume evaluation failed",
                ))
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = flat_boxes;
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn residual_volume_for_box_params_properties(
        &self,
        bounds: Vec<Vec<f64>>,
        rotations: Vec<Vec<f64>>,
    ) -> PyResult<f64> {
        let flat_boxes = flatten_bridge_oriented_box_vertices(&bounds, &rotations)?;
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let residual = unsafe {
                smart_manifold_residual_volume_for_boxes_properties(
                    self.ptr,
                    flat_boxes.as_ptr(),
                    bounds.len(),
                )
            };
            if residual.is_finite() {
                Ok(residual)
            } else {
                Err(PyValueError::new_err(
                    "Manifold bridge properties residual volume evaluation failed",
                ))
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = flat_boxes;
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn residual_volume_for_box_params_pair(
        &self,
        bounds: Vec<Vec<f64>>,
        rotations: Vec<Vec<f64>>,
    ) -> PyResult<(f64, f64)> {
        let flat_boxes = flatten_bridge_oriented_box_vertices(&bounds, &rotations)?;
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let mut mesh_volume = 0.0_f64;
            let mut properties_volume = 0.0_f64;
            let ok = unsafe {
                smart_manifold_residual_volume_for_boxes_pair(
                    self.ptr,
                    flat_boxes.as_ptr(),
                    bounds.len(),
                    &mut mesh_volume,
                    &mut properties_volume,
                )
            };
            if ok == 1 && mesh_volume.is_finite() && properties_volume.is_finite() {
                Ok((mesh_volume, properties_volume))
            } else {
                Err(PyValueError::new_err(
                    "Manifold bridge residual volume pair evaluation failed",
                ))
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = flat_boxes;
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    #[pyo3(signature = (bounds, rotations, bbox_idx, num_action_scale, action_unit, volume_sum, last_bbox_score, cover_penalty, pen_rate, initial_best, volume_method="mesh"))]
    #[allow(clippy::too_many_arguments)]
    fn best_axis_action(
        &self,
        bounds: Vec<Vec<f64>>,
        rotations: Vec<Vec<f64>>,
        bbox_idx: isize,
        num_action_scale: usize,
        action_unit: f64,
        volume_sum: f64,
        last_bbox_score: f64,
        cover_penalty: f64,
        pen_rate: f64,
        initial_best: f64,
        volume_method: &str,
    ) -> PyResult<(isize, f64)> {
        if volume_sum <= 0.0 {
            return Err(PyValueError::new_err("volume_sum must be positive"));
        }
        check_action_scale(num_action_scale)?;
        check_bridge_bbox_params(&bounds, &rotations)?;
        let action_scales = build_action_scales(num_action_scale)?;
        let volume_method = parse_volume_method(volume_method)?;
        self.best_axis_action_ref(
            &bounds,
            &rotations,
            bbox_idx,
            num_action_scale,
            action_unit,
            volume_sum,
            last_bbox_score,
            cover_penalty,
            pen_rate,
            initial_best,
            &action_scales,
            volume_method,
        )
    }

    #[pyo3(signature = (bounds, rotations, volume_sum, volume_method="mesh"))]
    fn covered_for_bounds(
        &self,
        bounds: Vec<Vec<f64>>,
        rotations: Vec<Vec<f64>>,
        volume_sum: f64,
        volume_method: &str,
    ) -> PyResult<f64> {
        if volume_sum <= 0.0 {
            return Err(PyValueError::new_err("volume_sum must be positive"));
        }
        let volume_method = parse_volume_method(volume_method)?;
        self.covered_for_bounds_ref(&bounds, &rotations, volume_sum, volume_method)
    }

    #[pyo3(signature = (bounds, rotations, bbox_mask, num_action_scale, action_unit, volume_sum, last_bbox_score, cover_penalty, pen_rate, initial_best, volume_method="mesh"))]
    #[allow(clippy::too_many_arguments)]
    fn best_axis_actions_for_mask(
        &self,
        bounds: Vec<Vec<f64>>,
        rotations: Vec<Vec<f64>>,
        bbox_mask: Vec<bool>,
        num_action_scale: usize,
        action_unit: f64,
        volume_sum: f64,
        last_bbox_score: f64,
        cover_penalty: f64,
        pen_rate: f64,
        initial_best: f64,
        volume_method: &str,
    ) -> PyResult<(Vec<isize>, Vec<f64>)> {
        if volume_sum <= 0.0 {
            return Err(PyValueError::new_err("volume_sum must be positive"));
        }
        check_action_scale(num_action_scale)?;
        check_bridge_bbox_params(&bounds, &rotations)?;
        if bbox_mask.len() != bounds.len() {
            return Err(PyValueError::new_err(
                "bbox_mask length must match bbox count",
            ));
        }
        let action_scales = build_action_scales(num_action_scale)?;
        let volume_method = parse_volume_method(volume_method)?;
        self.best_axis_actions_for_mask_ref(
            &bounds,
            &rotations,
            &bbox_mask,
            num_action_scale,
            action_unit,
            volume_sum,
            last_bbox_score,
            cover_penalty,
            pen_rate,
            initial_best,
            &action_scales,
            volume_method,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn greedy_axis_refine_segment(
        &self,
        bounds: Vec<Vec<f64>>,
        rotations: Vec<Vec<f64>>,
        num_action_scale: usize,
        action_unit: f64,
        volume_sum: f64,
        last_bbox_score: f64,
        cover_penalty: f64,
        pen_rate: f64,
        max_steps: usize,
    ) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<usize>, f64)> {
        if volume_sum <= 0.0 {
            return Err(PyValueError::new_err("volume_sum must be positive"));
        }
        check_action_scale(num_action_scale)?;
        check_bridge_bbox_params(&bounds, &rotations)?;
        let mut current_bounds = bounds;
        let current_rotations = rotations;
        let mut current_score = last_bbox_score;
        let mut rewards = Vec::new();
        let mut actions = Vec::new();
        let action_scales = build_action_scales(num_action_scale)?;
        let volume_method = 0;

        for _ in 0..max_steps {
            let bvs_reward = -((bbox_total_volume_raw(&current_bounds) / volume_sum - 1.0).abs())
                - current_score;
            let (action, reward) = self.best_axis_action_ref(
                &current_bounds,
                &current_rotations,
                -1,
                num_action_scale,
                action_unit,
                volume_sum,
                current_score,
                cover_penalty,
                pen_rate,
                bvs_reward,
                &action_scales,
                volume_method,
            )?;
            if action < 0 || reward <= 0.0 {
                break;
            }
            let action_usize = usize::try_from(action)
                .map_err(|_| PyValueError::new_err("selected action is out of range"))?;
            apply_axis_action_to_bounds(
                &mut current_bounds,
                action_usize,
                num_action_scale,
                &action_scales,
                action_unit,
            )?;
            current_score += reward;
            actions.push(action_usize);
            rewards.push(reward);
        }

        Ok((
            current_bounds,
            current_rotations,
            rewards,
            actions,
            current_score,
        ))
    }
}

impl ManifoldBridgeMesh {
    #[allow(clippy::too_many_arguments)]
    fn best_axis_actions_for_mask_cpp(
        &self,
        bounds: &[Vec<f64>],
        rotations: &[Vec<f64>],
        bbox_mask: &[bool],
        num_action_scale: usize,
        action_unit: f64,
        volume_sum: f64,
        last_bbox_score: f64,
        cover_penalty: f64,
        pen_rate: f64,
        initial_best: f64,
        action_scales: &[f64],
        volume_method: i32,
    ) -> PyResult<Option<(Vec<isize>, Vec<f64>)>> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            if volume_sum <= 0.0 {
                return Err(PyValueError::new_err("volume_sum must be positive"));
            }
            if bounds.len() != rotations.len() || bounds.len() != bbox_mask.len() {
                return Err(PyValueError::new_err(
                    "bounds, rotations, and bbox_mask must have the same length",
                ));
            }
            let n_boxes = bounds.len();
            let mut flat_bounds = Vec::with_capacity(n_boxes * 6);
            for row in bounds {
                check_bounds(row)?;
                flat_bounds.extend_from_slice(row);
            }
            let mut flat_rotations = Vec::with_capacity(n_boxes * 9);
            for rotation in rotations {
                if rotation.len() != 9 {
                    return Err(PyValueError::new_err(
                        "rotations must be flattened 3x3 row-major matrices",
                    ));
                }
                flat_rotations.extend_from_slice(rotation);
            }
            let flat_mask: Vec<u8> = bbox_mask
                .iter()
                .map(|enabled| if *enabled { 1_u8 } else { 0_u8 })
                .collect();
            let mut actions = vec![-1_isize; n_boxes];
            let mut rewards = vec![initial_best; n_boxes];
            let ok = unsafe {
                smart_manifold_best_axis_actions_for_mask(
                    self.ptr,
                    flat_bounds.as_ptr(),
                    flat_rotations.as_ptr(),
                    flat_mask.as_ptr(),
                    n_boxes,
                    num_action_scale,
                    action_unit,
                    volume_sum,
                    last_bbox_score,
                    cover_penalty,
                    pen_rate,
                    initial_best,
                    action_scales.as_ptr(),
                    actions.as_mut_ptr(),
                    rewards.as_mut_ptr(),
                    volume_method,
                )
            };
            if ok == 1 {
                Ok(Some((actions, rewards)))
            } else {
                Ok(None)
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = (
                bounds,
                rotations,
                bbox_mask,
                num_action_scale,
                action_unit,
                volume_sum,
                last_bbox_score,
                cover_penalty,
                pen_rate,
                initial_best,
                action_scales,
                volume_method,
            );
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn best_axis_actions_for_mask_ref(
        &self,
        bounds: &[Vec<f64>],
        rotations: &[Vec<f64>],
        bbox_mask: &[bool],
        num_action_scale: usize,
        action_unit: f64,
        volume_sum: f64,
        last_bbox_score: f64,
        cover_penalty: f64,
        pen_rate: f64,
        initial_best: f64,
        action_scales: &[f64],
        volume_method: i32,
    ) -> PyResult<(Vec<isize>, Vec<f64>)> {
        if let Some(result) = self.best_axis_actions_for_mask_cpp(
            bounds,
            rotations,
            bbox_mask,
            num_action_scale,
            action_unit,
            volume_sum,
            last_bbox_score,
            cover_penalty,
            pen_rate,
            initial_best,
            action_scales,
            volume_method,
        )? {
            return Ok(result);
        }

        let actions_per_bbox = 6 * num_action_scale + 1;
        let mut old_volumes = Vec::with_capacity(bounds.len());
        let mut total_volume = 0.0;
        for row in bounds {
            let volume = if bbox_is_valid_raw(row) {
                bbox_volume(row)?
            } else {
                0.0
            };
            old_volumes.push(volume);
            total_volume += volume;
        }

        let mut actions = vec![-1_isize; bounds.len()];
        let mut rewards = vec![initial_best; bounds.len()];
        let mut candidate = bounds.to_vec();
        for (idx, enabled) in bbox_mask.iter().enumerate() {
            if !enabled {
                continue;
            }
            let mut best_action = -1_isize;
            let mut best_reward = initial_best;
            for coord_idx in 0..6 {
                for (scale_idx, scale) in action_scales.iter().enumerate() {
                    let action = (idx * actions_per_bbox + coord_idx * num_action_scale + scale_idx)
                        as isize;
                    let original_value = candidate[idx][coord_idx];
                    candidate[idx][coord_idx] += scale * action_unit;
                    let candidate_volume = if bbox_is_valid_raw(&candidate[idx]) {
                        bbox_volume(&candidate[idx])?
                    } else {
                        0.0
                    };
                    let bvs = (total_volume - old_volumes[idx] + candidate_volume) / volume_sum;
                    let upper_reward = -((bvs - 1.0).abs()) - last_bbox_score;
                    if upper_reward <= best_reward {
                        candidate[idx][coord_idx] = original_value;
                        continue;
                    }
                    let covered =
                        self.covered_for_bounds_ref(&candidate, rotations, volume_sum, volume_method)?;
                    let score = -((bvs - 1.0).abs()) - (1.0 - covered) * pen_rate * cover_penalty;
                    let reward = score - last_bbox_score;
                    if best_reward < reward {
                        best_reward = reward;
                        best_action = action;
                    }
                    candidate[idx][coord_idx] = original_value;
                }
            }
            actions[idx] = best_action;
            rewards[idx] = best_reward;
        }

        Ok((actions, rewards))
    }

    #[allow(clippy::too_many_arguments)]
    fn best_axis_action_ref(
        &self,
        bounds: &[Vec<f64>],
        rotations: &[Vec<f64>],
        bbox_idx: isize,
        num_action_scale: usize,
        action_unit: f64,
        volume_sum: f64,
        last_bbox_score: f64,
        cover_penalty: f64,
        pen_rate: f64,
        initial_best: f64,
        action_scales: &[f64],
        volume_method: i32,
    ) -> PyResult<(isize, f64)> {
        let bbox_mask = if bbox_idx < 0 {
            vec![true; bounds.len()]
        } else {
            let idx = usize::try_from(bbox_idx)
                .map_err(|_| PyValueError::new_err("bbox_idx is out of range"))?;
            if idx >= bounds.len() {
                return Err(PyValueError::new_err("bbox_idx is out of range"));
            }
            let mut mask = vec![false; bounds.len()];
            mask[idx] = true;
            mask
        };
        if let Some((actions, rewards)) = self.best_axis_actions_for_mask_cpp(
            bounds,
            rotations,
            &bbox_mask,
            num_action_scale,
            action_unit,
            volume_sum,
            last_bbox_score,
            cover_penalty,
            pen_rate,
            initial_best,
            action_scales,
            volume_method,
        )? {
            let mut best_action = -1_isize;
            let mut best_reward = initial_best;
            for (action, reward) in actions.into_iter().zip(rewards.into_iter()) {
                if best_reward < reward {
                    best_reward = reward;
                    best_action = action;
                }
            }
            return Ok((best_action, best_reward));
        }

        let actions_per_bbox = 6 * num_action_scale + 1;
        let bbox_range: Vec<usize> = if bbox_idx < 0 {
            (0..bounds.len()).collect()
        } else {
            let idx = usize::try_from(bbox_idx)
                .map_err(|_| PyValueError::new_err("bbox_idx is out of range"))?;
            if idx >= bounds.len() {
                return Err(PyValueError::new_err("bbox_idx is out of range"));
            }
            vec![idx]
        };

        let mut old_volumes = Vec::with_capacity(bounds.len());
        let mut total_volume = 0.0;
        for row in bounds {
            let volume = if bbox_is_valid_raw(row) {
                bbox_volume(row)?
            } else {
                0.0
            };
            old_volumes.push(volume);
            total_volume += volume;
        }

        let mut best_action: isize = -1;
        let mut best_reward = initial_best;
        let mut candidate = bounds.to_vec();
        for idx in bbox_range {
            for coord_idx in 0..6 {
                for (scale_idx, scale) in action_scales.iter().enumerate() {
                    let action = (idx * actions_per_bbox + coord_idx * num_action_scale + scale_idx)
                        as isize;
                    let original_value = candidate[idx][coord_idx];
                    candidate[idx][coord_idx] += scale * action_unit;
                    let candidate_volume = if bbox_is_valid_raw(&candidate[idx]) {
                        bbox_volume(&candidate[idx])?
                    } else {
                        0.0
                    };
                    let bvs = (total_volume - old_volumes[idx] + candidate_volume) / volume_sum;
                    let upper_reward = -((bvs - 1.0).abs()) - last_bbox_score;
                    if upper_reward <= best_reward {
                        candidate[idx][coord_idx] = original_value;
                        continue;
                    }
                    let covered =
                        self.covered_for_bounds_ref(&candidate, rotations, volume_sum, volume_method)?;
                    let score = -((bvs - 1.0).abs()) - (1.0 - covered) * pen_rate * cover_penalty;
                    let reward = score - last_bbox_score;
                    if best_reward < reward {
                        best_reward = reward;
                        best_action = action;
                    }
                    candidate[idx][coord_idx] = original_value;
                }
            }
        }
        Ok((best_action, best_reward))
    }

    fn covered_for_bounds_ref(
        &self,
        bounds: &[Vec<f64>],
        rotations: &[Vec<f64>],
        volume_sum: f64,
        volume_method: i32,
    ) -> PyResult<f64> {
        if volume_sum <= 0.0 {
            return Err(PyValueError::new_err("volume_sum must be positive"));
        }
        let mut valid_bounds = Vec::new();
        let mut valid_rotations = Vec::new();
        for (row, rotation) in bounds.iter().zip(rotations.iter()) {
            if bbox_is_valid_raw(&row) {
                valid_bounds.push(row.clone());
                valid_rotations.push(rotation.clone());
            }
        }
        if valid_bounds.is_empty() {
            return Ok(0.0);
        }
        let flat_boxes = flatten_bridge_oriented_box_vertices(&valid_bounds, &valid_rotations)?;
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let residual = unsafe {
                if volume_method == 0 {
                    smart_manifold_residual_volume_for_boxes(
                        self.ptr,
                        flat_boxes.as_ptr(),
                        valid_bounds.len(),
                    )
                } else {
                    smart_manifold_residual_volume_for_boxes_properties(
                        self.ptr,
                        flat_boxes.as_ptr(),
                        valid_bounds.len(),
                    )
                }
            };
            if residual.is_finite() {
                Ok(1.0 - residual / volume_sum)
            } else {
                Err(PyValueError::new_err(
                    "Manifold bridge coverage evaluation failed",
                ))
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = flat_boxes;
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }
}

#[pyclass(unsendable)]
struct ManifoldState {
    ptr: *mut std::ffi::c_void,
    num_action_scale: usize,
    action_unit: f64,
    volume_sum: f64,
    action_scales: Vec<f64>,
}

impl Drop for ManifoldState {
    fn drop(&mut self) {
        #[cfg(not(smart_no_manifold_bridge))]
        unsafe {
            if !self.ptr.is_null() {
                smart_manifold_state_delete(self.ptr);
                self.ptr = std::ptr::null_mut();
            }
        }
    }
}

#[pymethods]
impl ManifoldState {
    #[new]
    #[pyo3(signature = (vertices, faces, bounds, rotations, num_action_scale, action_unit, volume_sum, last_bbox_score, stateful_union_cache=true, cache_capacity=65536, volume_method="mesh"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vertices: Vec<Vec<f64>>,
        faces: Vec<Vec<usize>>,
        bounds: Vec<Vec<f64>>,
        rotations: Vec<Vec<f64>>,
        num_action_scale: usize,
        action_unit: f64,
        volume_sum: f64,
        last_bbox_score: f64,
        stateful_union_cache: bool,
        cache_capacity: usize,
        volume_method: &str,
    ) -> PyResult<Self> {
        if volume_sum <= 0.0 {
            return Err(PyValueError::new_err("volume_sum must be positive"));
        }
        check_action_scale(num_action_scale)?;
        check_bridge_bbox_params(&bounds, &rotations)?;
        let (flat_vertices, flat_faces) = flatten_bridge_mesh(&vertices, &faces)?;
        let flat_bounds = flatten_f64_rows(&bounds, 6, "bounds")?;
        let flat_rotations = flatten_f64_rows(&rotations, 9, "rotations")?;
        let action_scales = build_action_scales(num_action_scale)?;
        let volume_method = parse_volume_method(volume_method)?;

        #[cfg(not(smart_no_manifold_bridge))]
        {
            let ptr = unsafe {
                smart_manifold_state_new(
                    flat_vertices.as_ptr(),
                    vertices.len(),
                    flat_faces.as_ptr(),
                    faces.len(),
                    flat_bounds.as_ptr(),
                    flat_rotations.as_ptr(),
                    bounds.len(),
                    volume_sum,
                    last_bbox_score,
                    i32::from(stateful_union_cache),
                    cache_capacity.max(1),
                    volume_method,
                )
            };
            if ptr.is_null() {
                Err(PyValueError::new_err("Manifold state construction failed"))
            } else {
                Ok(Self {
                    ptr,
                    num_action_scale,
                    action_unit,
                    volume_sum,
                    action_scales,
                })
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = (
                flat_vertices,
                flat_faces,
                flat_bounds,
                flat_rotations,
                stateful_union_cache,
                cache_capacity,
                volume_method,
            );
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn reset_to_state(
        &mut self,
        bounds: Vec<Vec<f64>>,
        rotations: Vec<Vec<f64>>,
        last_bbox_score: f64,
    ) -> PyResult<()> {
        check_bridge_bbox_params(&bounds, &rotations)?;
        let flat_bounds = flatten_f64_rows(&bounds, 6, "bounds")?;
        let flat_rotations = flatten_f64_rows(&rotations, 9, "rotations")?;
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let ok = unsafe {
                smart_manifold_state_reset(
                    self.ptr,
                    flat_bounds.as_ptr(),
                    flat_rotations.as_ptr(),
                    bounds.len(),
                    last_bbox_score,
                )
            };
            if ok == 1 {
                Ok(())
            } else {
                Err(PyValueError::new_err("Manifold state reset failed"))
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = (flat_bounds, flat_rotations);
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn bounds(&self) -> PyResult<Vec<Vec<f64>>> {
        let (bounds, _) = self.copy_bounds_rotations()?;
        Ok(bounds)
    }

    fn rotations(&self) -> PyResult<Vec<Vec<f64>>> {
        let (_, rotations) = self.copy_bounds_rotations()?;
        Ok(rotations)
    }

    fn state(&self) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, f64)> {
        let (bounds, rotations) = self.copy_bounds_rotations()?;
        Ok((bounds, rotations, self.last_bbox_score()?))
    }

    fn bbox_params(&self, bbox_idx: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
        self.copy_bbox_params(bbox_idx)
    }

    fn last_bbox_score(&self) -> PyResult<f64> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let value = unsafe { smart_manifold_state_last_bbox_score(self.ptr) };
            finite_or_error(value, "Manifold state last score failed")
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn total_volume(&self) -> PyResult<f64> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let value = unsafe { smart_manifold_state_total_bbox_volume(self.ptr) };
            finite_or_error(value, "Manifold state total bbox volume failed")
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn bvs(&self) -> PyResult<f64> {
        Ok(self.total_volume()? / self.volume_sum)
    }

    fn valid_count(&self) -> PyResult<usize> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            Ok(unsafe { smart_manifold_state_valid_count(self.ptr) })
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn cache_stats(&self) -> PyResult<HashMap<String, u64>> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let mut values = [0_u64; 7];
            let ok = unsafe { smart_manifold_state_cache_stats(self.ptr, values.as_mut_ptr()) };
            if ok != 1 {
                return Err(PyValueError::new_err("Manifold state cache stats failed"));
            }
            let mut out = HashMap::new();
            out.insert("reward_cache_size".to_string(), values[0]);
            out.insert("reward_cache_hits".to_string(), values[1]);
            out.insert("reward_cache_misses".to_string(), values[2]);
            out.insert("version".to_string(), values[3]);
            out.insert("state_hash".to_string(), values[4]);
            out.insert("except_union_builds".to_string(), values[5]);
            out.insert("except_union_cache_hits".to_string(), values[6]);
            Ok(out)
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn covered(&self) -> PyResult<f64> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let value = unsafe { smart_manifold_state_covered(self.ptr) };
            finite_or_error(value, "Manifold state coverage failed")
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn score(&self, cover_penalty: f64, pen_rate: f64) -> PyResult<f64> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let value = unsafe { smart_manifold_state_score(self.ptr, cover_penalty, pen_rate) };
            finite_or_error(value, "Manifold state score failed")
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = (cover_penalty, pen_rate);
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn score_axis_action(&self, action: isize, cover_penalty: f64, pen_rate: f64) -> PyResult<f64> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let value = unsafe {
                smart_manifold_state_score_axis_action(
                    self.ptr,
                    action,
                    self.num_action_scale,
                    self.action_unit,
                    cover_penalty,
                    pen_rate,
                    self.action_scales.as_ptr(),
                )
            };
            finite_or_error(value, "Manifold state action score failed")
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = (action, cover_penalty, pen_rate);
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn score_axis_action_reward(
        &self,
        action: isize,
        cover_penalty: f64,
        pen_rate: f64,
    ) -> PyResult<f64> {
        Ok(self.score_axis_action(action, cover_penalty, pen_rate)? - self.last_bbox_score()?)
    }

    fn score_replacement(
        &self,
        bbox_idx: usize,
        candidate_bounds: Vec<f64>,
        candidate_rotation: Vec<f64>,
        cover_penalty: f64,
        pen_rate: f64,
    ) -> PyResult<f64> {
        if candidate_bounds.len() != 6 {
            return Err(PyValueError::new_err("candidate_bounds must have length 6"));
        }
        if candidate_rotation.len() != 9 {
            return Err(PyValueError::new_err(
                "candidate_rotation must have length 9",
            ));
        }
        if bbox_idx >= self.num_boxes()? {
            return Err(PyValueError::new_err("bbox_idx is out of range"));
        }
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let value = unsafe {
                smart_manifold_state_score_replacement(
                    self.ptr,
                    bbox_idx,
                    candidate_bounds.as_ptr(),
                    candidate_rotation.as_ptr(),
                    cover_penalty,
                    pen_rate,
                )
            };
            finite_or_error(value, "Manifold state replacement score failed")
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = (
                bbox_idx,
                candidate_bounds,
                candidate_rotation,
                cover_penalty,
                pen_rate,
            );
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn score_action_batch(
        &self,
        bbox_mask: Vec<bool>,
        cover_penalty: f64,
        pen_rate: f64,
        initial_best: f64,
    ) -> PyResult<(Vec<isize>, Vec<f64>)> {
        let n_boxes = self.num_boxes()?;
        if bbox_mask.len() != n_boxes {
            return Err(PyValueError::new_err(
                "bbox_mask length must match bbox count",
            ));
        }
        let flat_mask: Vec<u8> = bbox_mask
            .iter()
            .map(|enabled| if *enabled { 1_u8 } else { 0_u8 })
            .collect();
        let mut actions = vec![-1_isize; n_boxes];
        let mut rewards = vec![initial_best; n_boxes];
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let ok = unsafe {
                smart_manifold_state_best_axis_actions_for_mask(
                    self.ptr,
                    flat_mask.as_ptr(),
                    self.num_action_scale,
                    self.action_unit,
                    cover_penalty,
                    pen_rate,
                    initial_best,
                    self.action_scales.as_ptr(),
                    actions.as_mut_ptr(),
                    rewards.as_mut_ptr(),
                )
            };
            if ok == 1 {
                Ok((actions, rewards))
            } else {
                Err(PyValueError::new_err(
                    "Manifold state batch action scoring failed",
                ))
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = flat_mask;
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn apply_axis_action(
        &mut self,
        action: isize,
        cover_penalty: f64,
        pen_rate: f64,
    ) -> PyResult<f64> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let reward = unsafe {
                smart_manifold_state_apply_axis_action(
                    self.ptr,
                    action,
                    self.num_action_scale,
                    self.action_unit,
                    cover_penalty,
                    pen_rate,
                    self.action_scales.as_ptr(),
                )
            };
            finite_or_error(reward, "Manifold state apply failed")
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            let _ = (action, cover_penalty, pen_rate);
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn apply_axis_action_delta(
        &mut self,
        action: isize,
        cover_penalty: f64,
        pen_rate: f64,
    ) -> PyResult<(f64, usize, Vec<f64>, Vec<f64>, f64)> {
        let n_boxes = self.num_boxes()?;
        let actions_per_bbox = 6 * self.num_action_scale + 1;
        if action < 0
            || usize::try_from(action).map_or(true, |idx| idx >= n_boxes * actions_per_bbox)
        {
            return Err(PyValueError::new_err("action is out of range"));
        }
        let action_idx =
            usize::try_from(action).map_err(|_| PyValueError::new_err("action is out of range"))?;
        if action_idx % actions_per_bbox == actions_per_bbox - 1 {
            return Err(PyValueError::new_err(
                "apply_axis_action_delta only supports axis actions",
            ));
        }
        let bbox_idx = action_idx / actions_per_bbox;
        let reward = self.apply_axis_action(action, cover_penalty, pen_rate)?;
        let (bounds, rotation) = self.copy_bbox_params(bbox_idx)?;
        Ok((reward, bbox_idx, bounds, rotation, self.last_bbox_score()?))
    }

    fn rollback(&mut self) -> PyResult<()> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let ok = unsafe { smart_manifold_state_rollback(self.ptr) };
            if ok == 1 {
                Ok(())
            } else {
                Err(PyValueError::new_err("Manifold state rollback failed"))
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn greedy_axis_refine_segment(
        &mut self,
        cover_penalty: f64,
        pen_rate: f64,
        max_steps: usize,
    ) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<usize>, f64)> {
        let mut rewards = Vec::new();
        let mut actions = Vec::new();
        for _ in 0..max_steps {
            let n_boxes = self.num_boxes()?;
            let mask = vec![true; n_boxes];
            let bvs_reward = -(self.bvs()? - 1.0).abs() - self.last_bbox_score()?;
            let (batch_actions, batch_rewards) =
                self.score_action_batch(mask, cover_penalty, pen_rate, bvs_reward)?;
            let mut best_action = -1_isize;
            let mut best_reward = bvs_reward;
            for (action, reward) in batch_actions.into_iter().zip(batch_rewards.into_iter()) {
                if best_reward < reward {
                    best_reward = reward;
                    best_action = action;
                }
            }
            if best_action < 0 || best_reward <= 0.0 {
                break;
            }
            let reward = self.apply_axis_action(best_action, cover_penalty, pen_rate)?;
            if !reward.is_finite() || reward <= 0.0 {
                break;
            }
            actions.push(usize::try_from(best_action).map_err(|_| {
                PyValueError::new_err("selected Manifold state action is out of range")
            })?);
            rewards.push(reward);
        }
        let (bounds, rotations) = self.copy_bounds_rotations()?;
        Ok((bounds, rotations, rewards, actions, self.last_bbox_score()?))
    }
}

impl ManifoldState {
    fn num_boxes(&self) -> PyResult<usize> {
        #[cfg(not(smart_no_manifold_bridge))]
        {
            Ok(unsafe { smart_manifold_state_num_boxes(self.ptr) })
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ))
        }
    }

    fn copy_bounds_rotations(&self) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let n_boxes = self.num_boxes()?;
        let mut flat_bounds = vec![0.0_f64; n_boxes * 6];
        let mut flat_rotations = vec![0.0_f64; n_boxes * 9];
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let ok = unsafe {
                smart_manifold_state_copy(
                    self.ptr,
                    flat_bounds.as_mut_ptr(),
                    flat_rotations.as_mut_ptr(),
                )
            };
            if ok != 1 {
                return Err(PyValueError::new_err("Manifold state copy failed"));
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            return Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ));
        }
        Ok((
            unflatten_f64_rows(&flat_bounds, 6),
            unflatten_f64_rows(&flat_rotations, 9),
        ))
    }

    fn copy_bbox_params(&self, bbox_idx: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if bbox_idx >= self.num_boxes()? {
            return Err(PyValueError::new_err("bbox_idx is out of range"));
        }
        let mut bounds = vec![0.0_f64; 6];
        let mut rotation = vec![0.0_f64; 9];
        #[cfg(not(smart_no_manifold_bridge))]
        {
            let ok = unsafe {
                smart_manifold_state_copy_bbox(
                    self.ptr,
                    bbox_idx,
                    bounds.as_mut_ptr(),
                    rotation.as_mut_ptr(),
                )
            };
            if ok != 1 {
                return Err(PyValueError::new_err("Manifold state bbox copy failed"));
            }
        }
        #[cfg(smart_no_manifold_bridge)]
        {
            return Err(PyValueError::new_err(
                "Manifold C++ bridge is unavailable in this build",
            ));
        }
        Ok((bounds, rotation))
    }
}

#[pyfunction]
fn bbox_volumes(bounds: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    bounds.iter().map(|row| bbox_volume(row)).collect()
}

#[pyfunction]
fn coverage_mask(points: Vec<Vec<f64>>, bounds: Vec<f64>) -> PyResult<Vec<bool>> {
    check_bounds(&bounds)?;
    let mut out = Vec::with_capacity(points.len());
    for point in points {
        if point.len() != 3 {
            return Err(PyValueError::new_err("points must be [x, y, z] rows"));
        }
        out.push(
            bounds[0] <= point[0]
                && point[0] <= bounds[3]
                && bounds[1] <= point[1]
                && point[1] <= bounds[4]
                && bounds[2] <= point[2]
                && point[2] <= bounds[5],
        );
    }
    Ok(out)
}

#[pyfunction]
fn centroid_proxy_axis_rewards(
    centroids: Vec<Vec<f64>>,
    volumes: Vec<f64>,
    bounds: Vec<Vec<f64>>,
    rotations: Vec<Vec<f64>>,
    num_action_scale: usize,
    action_unit: f64,
    volume_sum: f64,
    last_bbox_score: f64,
    cover_penalty: f64,
    pen_rate: f64,
) -> PyResult<Vec<(usize, f64)>> {
    if centroids.len() != volumes.len() {
        return Err(PyValueError::new_err(
            "centroids and volumes must have the same length",
        ));
    }
    if volume_sum <= 0.0 {
        return Err(PyValueError::new_err("volume_sum must be positive"));
    }
    let mut cached_centroids = Vec::with_capacity(centroids.len());
    for point in &centroids {
        if point.len() != 3 {
            return Err(PyValueError::new_err("centroids must be [x, y, z] rows"));
        }
        cached_centroids.push([point[0], point[1], point[2]]);
    }
    check_bridge_bbox_params(&bounds, &rotations)?;
    check_action_scale(num_action_scale)?;

    centroid_proxy_axis_rewards_cached(
        &cached_centroids,
        &volumes,
        &bounds,
        &rotations,
        num_action_scale,
        action_unit,
        volume_sum,
        last_bbox_score,
        cover_penalty,
        pen_rate,
    )
}

#[pyclass]
struct CandidateBitsetState {
    centroids: Vec<[f64; 3]>,
    volumes: Vec<f64>,
    volume_sum: f64,
}

#[pymethods]
impl CandidateBitsetState {
    #[new]
    fn new(centroids: Vec<Vec<f64>>, volumes: Vec<f64>, volume_sum: f64) -> PyResult<Self> {
        if centroids.len() != volumes.len() {
            return Err(PyValueError::new_err(
                "centroids and volumes must have the same length",
            ));
        }
        if volume_sum <= 0.0 {
            return Err(PyValueError::new_err("volume_sum must be positive"));
        }
        let mut cached_centroids = Vec::with_capacity(centroids.len());
        for point in centroids {
            if point.len() != 3 {
                return Err(PyValueError::new_err("centroids must be [x, y, z] rows"));
            }
            cached_centroids.push([point[0], point[1], point[2]]);
        }
        Ok(Self {
            centroids: cached_centroids,
            volumes,
            volume_sum,
        })
    }

    fn num_centroids(&self) -> usize {
        self.centroids.len()
    }

    fn volume_sum(&self) -> f64 {
        self.volume_sum
    }

    fn axis_rewards(
        &self,
        bounds: Vec<Vec<f64>>,
        rotations: Vec<Vec<f64>>,
        num_action_scale: usize,
        action_unit: f64,
        last_bbox_score: f64,
        cover_penalty: f64,
        pen_rate: f64,
    ) -> PyResult<Vec<(usize, f64)>> {
        check_bridge_bbox_params(&bounds, &rotations)?;
        check_action_scale(num_action_scale)?;
        centroid_proxy_axis_rewards_cached(
            &self.centroids,
            &self.volumes,
            &bounds,
            &rotations,
            num_action_scale,
            action_unit,
            self.volume_sum,
            last_bbox_score,
            cover_penalty,
            pen_rate,
        )
    }

    fn topk_axis_actions(
        &self,
        bounds: Vec<Vec<f64>>,
        rotations: Vec<Vec<f64>>,
        num_action_scale: usize,
        action_unit: f64,
        last_bbox_score: f64,
        cover_penalty: f64,
        pen_rate: f64,
        bbox_idx: isize,
        top_k: usize,
    ) -> PyResult<Vec<(usize, f64)>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        check_bridge_bbox_params(&bounds, &rotations)?;
        check_action_scale(num_action_scale)?;
        let mut records = centroid_proxy_axis_rewards_cached(
            &self.centroids,
            &self.volumes,
            &bounds,
            &rotations,
            num_action_scale,
            action_unit,
            self.volume_sum,
            last_bbox_score,
            cover_penalty,
            pen_rate,
        )?;
        if bbox_idx >= 0 {
            let bbox_idx = usize::try_from(bbox_idx)
                .map_err(|_| PyValueError::new_err("bbox_idx is out of range"))?;
            if bbox_idx >= bounds.len() {
                return Err(PyValueError::new_err("bbox_idx is out of range"));
            }
            let actions_per_bbox = 6 * num_action_scale + 1;
            let start = bbox_idx * actions_per_bbox;
            let end = start + 6 * num_action_scale;
            records.retain(|(action, _)| *action >= start && *action < end);
        }
        records.retain(|(_, reward)| reward.is_finite());
        records.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        records.truncate(top_k);
        Ok(records)
    }
}

fn centroid_proxy_axis_rewards_cached(
    centroids: &[[f64; 3]],
    volumes: &[f64],
    bounds: &[Vec<f64>],
    rotations: &[Vec<f64>],
    num_action_scale: usize,
    action_unit: f64,
    volume_sum: f64,
    last_bbox_score: f64,
    cover_penalty: f64,
    pen_rate: f64,
) -> PyResult<Vec<(usize, f64)>> {
    let action_scales = build_action_scales(num_action_scale)?;
    let mut base_masks = Vec::with_capacity(bounds.len());
    let mut bbox_volumes = Vec::with_capacity(bounds.len());
    let mut total_bbox_volume = 0.0;
    for (bbox_idx, row) in bounds.iter().enumerate() {
        let volume = if bbox_is_valid(row)? {
            bbox_volume(row)?
        } else {
            0.0
        };
        bbox_volumes.push(volume);
        total_bbox_volume += volume;
        if volume > 0.0 {
            base_masks.push(centroid_mask_bits_cached(
                centroids,
                row,
                &rotations[bbox_idx],
            )?);
        } else {
            base_masks.push(empty_bitset(centroids.len()));
        }
    }

    let actions_per_bbox = 6 * num_action_scale + 1;
    let mut out = Vec::with_capacity(bounds.len() * 6 * num_action_scale);
    for bbox_idx in 0..bounds.len() {
        for coord_idx in 0..6 {
            for (scale_idx, scale) in action_scales.iter().enumerate() {
                let action = bbox_idx * actions_per_bbox + coord_idx * num_action_scale + scale_idx;
                let mut candidate = bounds[bbox_idx].clone();
                candidate[coord_idx] += scale * action_unit;
                if !bbox_is_valid(&candidate)? {
                    out.push((action, f64::NEG_INFINITY));
                    continue;
                }

                let mut union =
                    centroid_mask_bits_cached(centroids, &candidate, &rotations[bbox_idx])?;
                for (other_idx, mask) in base_masks.iter().enumerate() {
                    if other_idx != bbox_idx {
                        or_bitset_in_place(&mut union, mask);
                    }
                }

                let covered = masked_volume_sum(&union, &volumes) / volume_sum;
                let candidate_volume = bbox_volume(&candidate)?;
                let new_total = total_bbox_volume - bbox_volumes[bbox_idx] + candidate_volume;
                let bvs = new_total / volume_sum;
                let proxy_score = -(bvs - 1.0).abs() - (1.0 - covered) * pen_rate * cover_penalty;
                out.push((action, proxy_score - last_bbox_score));
            }
        }
    }
    Ok(out)
}

#[pyfunction]
fn action_count(num_bbox: usize, num_action_scale: usize) -> usize {
    num_bbox * (6 * num_action_scale + 1)
}

#[pyfunction]
fn action_scales(num_action_scale: usize) -> PyResult<Vec<f64>> {
    build_action_scales(num_action_scale)
}

#[pyfunction]
fn action_indices(num_bbox: usize, num_action_scale: usize) -> PyResult<Vec<Vec<usize>>> {
    check_action_scale(num_action_scale)?;
    let mut out = Vec::with_capacity(action_count(num_bbox, num_action_scale));
    for bbox_idx in 0..num_bbox {
        for coord_idx in 0..6 {
            for scale_idx in 0..num_action_scale {
                out.push(vec![bbox_idx, coord_idx, scale_idx]);
            }
        }
        out.push(vec![bbox_idx, 6, 0]);
    }
    Ok(out)
}

#[pyfunction]
fn opposite_action(action: usize, num_action_scale: usize) -> PyResult<usize> {
    check_action_scale(num_action_scale)?;
    let (bbox_idx, coord_idx, scale_idx) = decode_action(action, num_action_scale);
    if coord_idx == 6 {
        Ok(action)
    } else {
        Ok(encode_action(
            bbox_idx,
            coord_idx,
            num_action_scale - 1 - scale_idx,
            num_action_scale,
        ))
    }
}

#[pyfunction]
fn opposite_actions(num_bbox: usize, num_action_scale: usize) -> PyResult<Vec<usize>> {
    check_action_scale(num_action_scale)?;
    let total = action_count(num_bbox, num_action_scale);
    let mut out = Vec::with_capacity(total);
    for action in 0..total {
        out.push(opposite_action(action, num_action_scale)?);
    }
    Ok(out)
}

#[pyfunction]
fn opposite_action_mask(
    action: usize,
    num_bbox: usize,
    num_action_scale: usize,
) -> PyResult<Vec<bool>> {
    check_action_scale(num_action_scale)?;
    let total = action_count(num_bbox, num_action_scale);
    if action >= total {
        return Err(PyValueError::new_err("action is out of range"));
    }
    let mut out = vec![false; total];
    out[opposite_action(action, num_action_scale)?] = true;
    Ok(out)
}

#[pyfunction]
fn untried_actions(action_mask: Vec<bool>) -> Vec<usize> {
    action_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, masked)| if !masked { Some(idx) } else { None })
        .collect()
}

#[pyfunction]
fn single_untried_action_mask(total_actions: usize, action: usize) -> PyResult<Vec<bool>> {
    if action >= total_actions {
        return Err(PyValueError::new_err("action is out of range"));
    }
    let mut out = vec![true; total_actions];
    out[action] = false;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (total_actions, action, num_action_scale, parent_mask=None))]
fn mcts_child_action_mask(
    total_actions: usize,
    action: usize,
    num_action_scale: usize,
    parent_mask: Option<Vec<bool>>,
) -> PyResult<Vec<bool>> {
    check_action_scale(num_action_scale)?;
    let per_bbox = 6 * num_action_scale + 1;
    if total_actions == 0 || total_actions % per_bbox != 0 {
        return Err(PyValueError::new_err(
            "total_actions must match the legacy bbox action space",
        ));
    }
    if action >= total_actions {
        return Err(PyValueError::new_err("action is out of range"));
    }

    let mut out = match parent_mask {
        Some(mask) => {
            if mask.len() != total_actions {
                return Err(PyValueError::new_err(
                    "parent_mask length must match total_actions",
                ));
            }
            mask
        }
        None => vec![false; total_actions],
    };
    out[opposite_action(action, num_action_scale)?] = true;
    Ok(out)
}

#[pyfunction]
fn apply_axis_action(
    bounds: Vec<Vec<f64>>,
    action: usize,
    num_action_scale: usize,
    action_unit: f64,
) -> PyResult<Vec<Vec<f64>>> {
    check_action_scale(num_action_scale)?;
    let scales = build_action_scales(num_action_scale)?;

    let mut out = bounds;
    apply_axis_action_to_bounds(&mut out, action, num_action_scale, &scales, action_unit)?;

    Ok(out)
}

#[pyfunction]
fn action_upper_rewards(
    bounds: Vec<Vec<f64>>,
    num_action_scale: usize,
    action_unit: f64,
    volume_sum: f64,
    last_bbox_score: f64,
) -> PyResult<Vec<f64>> {
    check_action_scale(num_action_scale)?;
    if volume_sum <= 0.0 {
        return Err(PyValueError::new_err("volume_sum must be positive"));
    }

    let scales = build_action_scales(num_action_scale)?;
    let (old_volumes, total_volume) = summarize_bbox_volumes(&bounds)?;
    bbox_state_action_upper_rewards(
        &bounds,
        &old_volumes,
        total_volume,
        &scales,
        action_unit,
        volume_sum,
        last_bbox_score,
    )
}

#[pyfunction]
fn bbox_action_upper_rewards(
    bounds: Vec<Vec<f64>>,
    bbox_idx: usize,
    num_action_scale: usize,
    action_unit: f64,
    volume_sum: f64,
    last_bbox_score: f64,
) -> PyResult<Vec<f64>> {
    check_action_scale(num_action_scale)?;
    if volume_sum <= 0.0 {
        return Err(PyValueError::new_err("volume_sum must be positive"));
    }
    if bbox_idx >= bounds.len() {
        return Err(PyValueError::new_err("bbox_idx is out of range"));
    }

    let scales = build_action_scales(num_action_scale)?;
    let (old_volumes, total_volume) = summarize_bbox_volumes(&bounds)?;
    bbox_state_single_bbox_upper_rewards(
        &bounds,
        &old_volumes,
        total_volume,
        bbox_idx,
        &scales,
        action_unit,
        volume_sum,
        last_bbox_score,
    )
}

#[pyfunction]
fn bbox_valid_mask(bounds: Vec<Vec<f64>>) -> PyResult<Vec<bool>> {
    bounds.iter().map(|row| bbox_is_valid(row)).collect()
}

#[pyfunction]
fn total_bbox_volume(bounds: Vec<Vec<f64>>) -> PyResult<f64> {
    let mut total = 0.0;
    for row in bounds {
        total += bbox_volume(&row)?;
    }
    Ok(total)
}

#[pyfunction]
fn bbox_union_bounds(bounds: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    if bounds.is_empty() {
        return Err(PyValueError::new_err("bounds must not be empty"));
    }
    let mut out = bounds[0].clone();
    check_bounds(&out)?;
    for row in bounds.iter().skip(1) {
        check_bounds(row)?;
        out[0] = out[0].min(row[0]);
        out[1] = out[1].min(row[1]);
        out[2] = out[2].min(row[2]);
        out[3] = out[3].max(row[3]);
        out[4] = out[4].max(row[4]);
        out[5] = out[5].max(row[5]);
    }
    Ok(out)
}

#[pyfunction]
fn bbox_union_volume(bounds: Vec<Vec<f64>>) -> PyResult<f64> {
    let union = bbox_union_bounds(bounds)?;
    bbox_volume(&union)
}

#[pyfunction]
fn bbox_rot_state_key(bounds: Vec<Vec<f64>>, rotations: Vec<Vec<f64>>) -> PyResult<String> {
    if bounds.len() != rotations.len() {
        return Err(PyValueError::new_err(
            "bounds and rotations must have the same length",
        ));
    }
    let mut out = String::new();
    for (idx, (row, rot)) in bounds.iter().zip(rotations.iter()).enumerate() {
        check_bounds(row)?;
        if rot.len() != 9 {
            return Err(PyValueError::new_err("rotations must contain 3x3 rows"));
        }
        if idx > 0 {
            out.push('|');
        }
        out.push('b');
        append_float_bits(&mut out, row);
        out.push('r');
        append_float_bits(&mut out, rot);
    }
    Ok(out)
}

#[pyfunction]
fn bavf_scores(part_volumes: Vec<f64>, bbox_volumes: Vec<f64>, alpha: f64) -> PyResult<Vec<f64>> {
    if part_volumes.len() != bbox_volumes.len() {
        return Err(PyValueError::new_err(
            "part_volumes and bbox_volumes must have the same length",
        ));
    }
    Ok(part_volumes
        .iter()
        .zip(bbox_volumes.iter())
        .map(|(part, bbox)| {
            if *bbox > 0.0 {
                alpha * part / bbox
            } else {
                0.0
            }
        })
        .collect())
}

#[pyfunction]
fn merge_bavf_reward(
    prev_bvs: f64,
    left_bbox_volume: f64,
    right_bbox_volume: f64,
    merged_bbox_volume: f64,
    shape_volume: f64,
) -> PyResult<f64> {
    if shape_volume <= 0.0 {
        return Err(PyValueError::new_err("shape_volume must be positive"));
    }
    let new_bvs = (prev_bvs * shape_volume - left_bbox_volume - right_bbox_volume
        + merged_bbox_volume)
        / shape_volume;
    Ok(-(new_bvs - 1.0).abs() + (prev_bvs - 1.0).abs())
}

#[pyfunction]
fn softmax_scaled(values: Vec<f64>, scale: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let scaled: Vec<f64> = values.iter().map(|value| value * scale).collect();
    let max_value = scaled
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
    let exps: Vec<f64> = scaled
        .iter()
        .map(|value| (value - max_value).exp())
        .collect();
    let total: f64 = exps.iter().sum();
    if total == 0.0 {
        vec![1.0 / values.len() as f64; values.len()]
    } else {
        exps.iter().map(|value| value / total).collect()
    }
}

#[pyfunction]
fn ucb_scores(
    parent_visits: usize,
    child_qs: Vec<f64>,
    child_visits: Vec<usize>,
    exp_weight: f64,
) -> PyResult<Vec<f64>> {
    if child_qs.len() != child_visits.len() {
        return Err(PyValueError::new_err(
            "child_qs and child_visits must have the same length",
        ));
    }
    if parent_visits == 0 {
        return Ok(vec![f64::INFINITY; child_qs.len()]);
    }
    let log_parent = (parent_visits as f64).ln();
    Ok(child_qs
        .iter()
        .zip(child_visits.iter())
        .map(|(q_value, visit_count)| {
            if *visit_count == 0 {
                f64::INFINITY
            } else {
                q_value + exp_weight * (2.0 * log_parent / *visit_count as f64).sqrt()
            }
        })
        .collect())
}

#[pyfunction]
fn ucb_best_indices(
    parent_visits: usize,
    child_qs: Vec<f64>,
    child_visits: Vec<usize>,
    exp_weight: f64,
) -> PyResult<Vec<usize>> {
    let scores = ucb_scores(parent_visits, child_qs, child_visits, exp_weight)?;
    if scores.is_empty() {
        return Ok(Vec::new());
    }
    let mut max_score = scores[0];
    for score in scores.iter().skip(1) {
        if *score > max_score {
            max_score = *score;
        }
    }
    Ok(scores
        .iter()
        .enumerate()
        .filter_map(
            |(idx, score)| {
                if *score == max_score {
                    Some(idx)
                } else {
                    None
                }
            },
        )
        .collect())
}

#[pyfunction]
fn incremental_average(previous: f64, count: usize, value: f64) -> f64 {
    previous / (count + 1) as f64 * count as f64 + value / (count + 1) as f64
}

#[pyfunction]
fn discounted_reward(rewards: Vec<f64>, gamma: f64) -> f64 {
    let mut out = 0.0;
    for reward in rewards.iter().rev() {
        out = out * gamma + reward;
    }
    out
}

#[pyfunction]
fn symmetric_chamfer(left: Vec<Vec<f64>>, right: Vec<Vec<f64>>) -> PyResult<f64> {
    if left.is_empty() || right.is_empty() {
        return Err(PyValueError::new_err("point sets must not be empty"));
    }
    check_points(&left)?;
    check_points(&right)?;

    let right_to_left = mean_nearest_squared_distance(&right, &left);
    let left_to_right = mean_nearest_squared_distance(&left, &right);
    Ok(right_to_left + left_to_right)
}

#[pyfunction]
fn tetra_volumes(vertices: Vec<Vec<f64>>, voxels: Vec<Vec<usize>>) -> PyResult<Vec<f64>> {
    check_vertices(&vertices)?;
    check_voxels(&voxels, vertices.len())?;
    let mut out = Vec::with_capacity(voxels.len());
    for voxel in &voxels {
        let p0 = &vertices[voxel[0]];
        let p1 = &vertices[voxel[1]];
        let p2 = &vertices[voxel[2]];
        let p3 = &vertices[voxel[3]];
        let a = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let b = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let c = [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]];
        out.push(dot3(&a, &cross3(&b, &c)).abs() / 6.0);
    }
    Ok(out)
}

#[pyfunction]
fn tetra_centroids(vertices: Vec<Vec<f64>>, voxels: Vec<Vec<usize>>) -> PyResult<Vec<f64>> {
    check_vertices(&vertices)?;
    check_voxels(&voxels, vertices.len())?;
    let mut out = Vec::with_capacity(voxels.len() * 3);
    for voxel in &voxels {
        for axis in 0..3 {
            out.push(
                (vertices[voxel[0]][axis]
                    + vertices[voxel[1]][axis]
                    + vertices[voxel[2]][axis]
                    + vertices[voxel[3]][axis])
                    / 4.0,
            );
        }
    }
    Ok(out)
}

#[pyfunction]
fn tetra_surface_faces(voxels: Vec<Vec<usize>>) -> PyResult<Vec<Vec<usize>>> {
    tetra_surface_faces_from_voxels(&voxels)
}

#[pyfunction]
fn tetra_adjacency(voxels: Vec<Vec<usize>>) -> PyResult<Vec<Vec<usize>>> {
    check_voxels_shape(&voxels)?;
    let mut face_to_voxels: HashMap<[usize; 3], Vec<usize>> = HashMap::new();
    let mut ordered_faces: Vec<[usize; 3]> = Vec::new();
    for (voxel_idx, voxel) in voxels.iter().enumerate() {
        for face in tet_faces(voxel) {
            let key = sorted_face(face);
            if !face_to_voxels.contains_key(&key) {
                ordered_faces.push(key);
            }
            face_to_voxels.entry(key).or_default().push(voxel_idx);
        }
    }

    let mut adjacency: Vec<BTreeSet<usize>> = (0..voxels.len()).map(|_| BTreeSet::new()).collect();
    for key in ordered_faces {
        let owners = &face_to_voxels[&key];
        if owners.len() < 2 {
            continue;
        }
        for owner in owners {
            for other in owners {
                if other != owner {
                    adjacency[*owner].insert(*other);
                }
            }
        }
    }
    Ok(adjacency
        .into_iter()
        .map(|values| values.into_iter().collect())
        .collect())
}

#[pyfunction]
fn load_gmsh(path: String) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let bytes = fs::read(&path)
        .map_err(|err| PyValueError::new_err(format!("Failed to read Gmsh file {path}: {err}")))?;
    let text = String::from_utf8_lossy(&bytes);
    parse_gmsh(&text, &path)
}

#[pyfunction]
fn save_gmsh(
    path: String,
    vertices: Vec<Vec<f64>>,
    faces: Vec<Vec<usize>>,
    voxels: Vec<Vec<usize>>,
) -> PyResult<()> {
    check_vertices(&vertices)?;
    check_faces(&faces, vertices.len())?;
    check_voxels(&voxels, vertices.len())?;

    let path_obj = std::path::Path::new(&path);
    if let Some(parent) = path_obj.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| {
                PyValueError::new_err(format!(
                    "Failed to create parent directory for {path}: {err}"
                ))
            })?;
        }
    }

    let element_count = faces.len() + voxels.len();
    let mut text = String::new();
    text.push_str("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n");
    text.push_str("$Nodes\n");
    text.push_str(&format!("{}\n", vertices.len()));
    for (index, vertex) in vertices.iter().enumerate() {
        text.push_str(&format!(
            "{} {:.17} {:.17} {:.17}\n",
            index + 1,
            vertex[0],
            vertex[1],
            vertex[2]
        ));
    }
    text.push_str("$EndNodes\n");
    text.push_str("$Elements\n");
    text.push_str(&format!("{element_count}\n"));

    let mut element_id = 1usize;
    for face in &faces {
        text.push_str(&format!(
            "{} 2 0 {} {} {}\n",
            element_id,
            face[0] + 1,
            face[1] + 1,
            face[2] + 1
        ));
        element_id += 1;
    }
    for voxel in &voxels {
        text.push_str(&format!(
            "{} 4 0 {} {} {} {}\n",
            element_id,
            voxel[0] + 1,
            voxel[1] + 1,
            voxel[2] + 1,
            voxel[3] + 1
        ));
        element_id += 1;
    }
    text.push_str("$EndElements\n");

    fs::write(&path, text)
        .map_err(|err| PyValueError::new_err(format!("Failed to write Gmsh file {path}: {err}")))
}

type PartitionSummaries = (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>);

#[pyfunction(signature = (vertices, voxels, volumes, partitions, unique_points=false))]
fn partition_summaries(
    vertices: Vec<Vec<f64>>,
    voxels: Vec<Vec<usize>>,
    volumes: Vec<f64>,
    partitions: Vec<Vec<usize>>,
    unique_points: bool,
) -> PyResult<PartitionSummaries> {
    check_vertices(&vertices)?;
    check_voxels(&voxels, vertices.len())?;
    if volumes.len() != voxels.len() {
        return Err(PyValueError::new_err(
            "volumes must have the same length as voxels",
        ));
    }

    let mut part_volumes = Vec::with_capacity(partitions.len());
    let mut part_bounds = Vec::with_capacity(partitions.len());
    let mut part_points = Vec::with_capacity(partitions.len());

    for partition in &partitions {
        if partition.is_empty() {
            return Err(PyValueError::new_err("partition must not be empty"));
        }
        for tet_idx in partition {
            if *tet_idx >= voxels.len() {
                return Err(PyValueError::new_err(
                    "partition voxel index is out of range",
                ));
            }
        }

        let first_vertex = &vertices[voxels[partition[0]][0]];
        let mut min_x = first_vertex[0];
        let mut min_y = first_vertex[1];
        let mut min_z = first_vertex[2];
        let mut max_x = first_vertex[0];
        let mut max_y = first_vertex[1];
        let mut max_z = first_vertex[2];
        let mut volume = 0.0;
        let mut points = Vec::with_capacity(partition.len() * 4);

        for tet_idx in partition {
            volume += volumes[*tet_idx];
            for vertex_idx in &voxels[*tet_idx] {
                let vertex = &vertices[*vertex_idx];
                min_x = min_x.min(vertex[0]);
                min_y = min_y.min(vertex[1]);
                min_z = min_z.min(vertex[2]);
                max_x = max_x.max(vertex[0]);
                max_y = max_y.max(vertex[1]);
                max_z = max_z.max(vertex[2]);
                points.push([vertex[0], vertex[1], vertex[2]]);
            }
        }
        if unique_points {
            dedupe_points_exact(&mut points);
        }

        part_volumes.push(volume);
        part_bounds.push(vec![min_x, min_y, min_z, max_x, max_y, max_z]);
        part_points.push(points.iter().flat_map(|point| point.to_vec()).collect());
    }

    Ok((part_volumes, part_bounds, part_points))
}

#[pyfunction(signature = (vertices, voxels, box_vertices, surface_volume, max_boxes, box_volumes=None))]
fn tet_clipping_metrics(
    vertices: Vec<Vec<f64>>,
    voxels: Vec<Vec<usize>>,
    box_vertices: Vec<Vec<Vec<f64>>>,
    surface_volume: f64,
    max_boxes: usize,
    box_volumes: Option<Vec<f64>>,
) -> PyResult<HashMap<String, f64>> {
    check_vertices(&vertices)?;
    check_voxels(&voxels, vertices.len())?;
    if surface_volume <= 0.0 {
        return Err(PyValueError::new_err("surface_volume must be positive"));
    }
    if box_vertices.is_empty() {
        return Err(PyValueError::new_err("box_vertices must not be empty"));
    }
    if box_vertices.len() > max_boxes {
        return Err(PyValueError::new_err(format!(
            "box count {} exceeds max_boxes {max_boxes}",
            box_vertices.len()
        )));
    }
    let explicit_box_volumes = box_volumes;
    if let Some(volumes) = &explicit_box_volumes {
        if volumes.len() != box_vertices.len() {
            return Err(PyValueError::new_err(format!(
                "box_volumes length {} does not match box count {}",
                volumes.len(),
                box_vertices.len()
            )));
        }
    }

    let verts3 = vertices_to_arrays(&vertices)?;
    let vox4 = voxels_to_arrays(&voxels, verts3.len())?;
    let box_arrays = box_vertices_to_arrays(&box_vertices)?;
    tet_clipping_metrics_internal(
        &verts3,
        &vox4,
        &box_arrays,
        surface_volume,
        max_boxes,
        explicit_box_volumes.as_deref(),
    )
}

#[pyclass]
struct TetClippingState {
    vertices: Vec<[f64; 3]>,
    voxels: Vec<[usize; 4]>,
    surface_volume: f64,
}

#[pymethods]
impl TetClippingState {
    #[new]
    fn new(
        vertices: Vec<Vec<f64>>,
        voxels: Vec<Vec<usize>>,
        surface_volume: f64,
    ) -> PyResult<Self> {
        check_vertices(&vertices)?;
        check_voxels(&voxels, vertices.len())?;
        if surface_volume <= 0.0 {
            return Err(PyValueError::new_err("surface_volume must be positive"));
        }
        let verts3 = vertices_to_arrays(&vertices)?;
        let vox4 = voxels_to_arrays(&voxels, verts3.len())?;
        Ok(Self {
            vertices: verts3,
            voxels: vox4,
            surface_volume,
        })
    }

    #[pyo3(signature = (box_vertices, max_boxes, box_volumes=None))]
    fn metrics(
        &self,
        box_vertices: Vec<Vec<Vec<f64>>>,
        max_boxes: usize,
        box_volumes: Option<Vec<f64>>,
    ) -> PyResult<HashMap<String, f64>> {
        let box_arrays = box_vertices_to_arrays(&box_vertices)?;
        tet_clipping_metrics_internal(
            &self.vertices,
            &self.voxels,
            &box_arrays,
            self.surface_volume,
            max_boxes,
            box_volumes.as_deref(),
        )
    }

    #[pyo3(signature = (bounds, rotations, max_boxes))]
    fn metrics_for_boxes(
        &self,
        bounds: Vec<Vec<f64>>,
        rotations: Vec<Vec<f64>>,
        max_boxes: usize,
    ) -> PyResult<HashMap<String, f64>> {
        if bounds.is_empty() {
            return Err(PyValueError::new_err("bounds must not be empty"));
        }
        if bounds.len() > max_boxes {
            return Err(PyValueError::new_err(format!(
                "box count {} exceeds max_boxes {max_boxes}",
                bounds.len()
            )));
        }
        if rotations.len() != bounds.len() {
            return Err(PyValueError::new_err(
                "rotations length must match bounds length",
            ));
        }

        let mut box_infos = Vec::with_capacity(bounds.len());
        let mut box_volumes = Vec::with_capacity(bounds.len());
        for (row, rotation) in bounds.iter().zip(rotations.iter()) {
            check_bounds(row)?;
            if rotation.len() != 9 {
                return Err(PyValueError::new_err(
                    "rotations must be flattened 3x3 row-major matrices",
                ));
            }
            box_volumes.push(bbox_volume(row)?);
            box_infos.push(convex_info_from_oriented_box(row, rotation)?);
        }

        tet_clipping_metrics_from_infos(
            &self.vertices,
            &self.voxels,
            &box_infos,
            &box_volumes,
            self.surface_volume,
        )
    }
}

fn tet_clipping_metrics_internal(
    verts3: &[[f64; 3]],
    voxels: &[[usize; 4]],
    box_vertices: &[Vec<[f64; 3]>],
    surface_volume: f64,
    max_boxes: usize,
    explicit_box_volumes: Option<&[f64]>,
) -> PyResult<HashMap<String, f64>> {
    if surface_volume <= 0.0 {
        return Err(PyValueError::new_err("surface_volume must be positive"));
    }
    if box_vertices.is_empty() {
        return Err(PyValueError::new_err("box_vertices must not be empty"));
    }
    if box_vertices.len() > max_boxes {
        return Err(PyValueError::new_err(format!(
            "box count {} exceeds max_boxes {max_boxes}",
            box_vertices.len()
        )));
    }
    if let Some(volumes) = explicit_box_volumes {
        if volumes.len() != box_vertices.len() {
            return Err(PyValueError::new_err(format!(
                "box_volumes length {} does not match box count {}",
                volumes.len(),
                box_vertices.len()
            )));
        }
    }

    let mut box_infos = Vec::with_capacity(box_vertices.len());
    let mut box_volumes = Vec::with_capacity(box_vertices.len());
    for (idx, points) in box_vertices.iter().enumerate() {
        let volume = match explicit_box_volumes {
            Some(volumes) => volumes[idx],
            None => convex_hull_volume(points)?,
        };
        box_volumes.push(volume);
        box_infos.push(convex_info_from_points(points)?);
    }

    tet_clipping_metrics_from_infos(verts3, voxels, &box_infos, &box_volumes, surface_volume)
}

fn tet_clipping_metrics_from_infos(
    verts3: &[[f64; 3]],
    voxels: &[[usize; 4]],
    box_infos: &[ConvexInfo],
    box_volumes: &[f64],
    surface_volume: f64,
) -> PyResult<HashMap<String, f64>> {
    let box_indices: Vec<usize> = (0..box_infos.len()).collect();
    let box_union_volume = union_volume_indices(&box_infos, &box_indices, None, None)?;
    let mut per_box_intersections = vec![0.0; box_infos.len()];
    let mut shape_box_union_intersection = 0.0;
    let mut tet_volume_sum = 0.0;

    for voxel in voxels {
        let tet_points = [
            verts3[voxel[0]],
            verts3[voxel[1]],
            verts3[voxel[2]],
            verts3[voxel[3]],
        ];
        let tet_info = convex_info_from_points(&tet_points)?;
        tet_volume_sum += tet_volume(&tet_points);

        let mut overlapping = Vec::new();
        for (idx, box_info) in box_infos.iter().enumerate() {
            if aabb_overlap(&tet_info, box_info) {
                overlapping.push(idx);
            }
        }
        if overlapping.is_empty() {
            continue;
        }

        let mut single_cache = HashMap::new();
        for idx in &overlapping {
            let volume = intersection_volume_infos(&[&tet_info, &box_infos[*idx]])?;
            single_cache.insert(*idx, volume);
            per_box_intersections[*idx] += volume;
        }

        let tet_union = union_volume_indices(
            &box_infos,
            &overlapping,
            Some(&tet_info),
            Some(&single_cache),
        )?;
        shape_box_union_intersection += tet_union.min(tet_volume(&tet_points));
    }

    let mut mov = 0.0;
    for (box_volume, intersection) in box_volumes.iter().zip(per_box_intersections.iter()) {
        if *intersection > 1e-10 {
            let part_ov = (box_volume - intersection).max(0.0) / intersection;
            if part_ov > mov {
                mov = part_ov;
            }
        }
    }
    let box_volume_sum: f64 = box_volumes.iter().sum();
    let covered = shape_box_union_intersection / surface_volume;
    let outside_box_volume = (box_union_volume - shape_box_union_intersection).max(0.0);
    let tov = if covered >= 0.99 {
        (box_union_volume - surface_volume) / surface_volume
    } else {
        outside_box_volume / surface_volume
    };
    let union_volume = surface_volume + box_union_volume - shape_box_union_intersection;
    let viou = if union_volume <= 0.0 {
        0.0
    } else {
        shape_box_union_intersection / union_volume
    };

    let mut out = HashMap::new();
    out.insert("num_box".to_string(), box_infos.len() as f64);
    out.insert("BVS".to_string(), box_volume_sum / surface_volume);
    out.insert("MOV".to_string(), mov);
    out.insert("Covered".to_string(), covered);
    out.insert("TOV".to_string(), tov);
    out.insert("vIoU".to_string(), viou);
    out.insert("tet_volume_sum".to_string(), tet_volume_sum);
    out.insert("surface_volume".to_string(), surface_volume);
    out.insert("box_union_volume".to_string(), box_union_volume);
    out.insert(
        "shape_box_union_intersection".to_string(),
        shape_box_union_intersection,
    );
    Ok(out)
}

#[derive(Clone)]
struct MctsCallbackNode {
    q: f64,
    reward: f64,
    num_vis: usize,
    state_key: Option<String>,
    child_ids: Vec<usize>,
    child_actions: Vec<usize>,
    action_mask: Vec<bool>,
    untried_actions: Vec<usize>,
}

impl MctsCallbackNode {
    fn new(action_mask: Vec<bool>, state_key: Option<String>) -> Self {
        let untried_actions = action_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, masked)| if !masked { Some(idx) } else { None })
            .collect();
        Self {
            q: -f64::MAX,
            reward: -f64::MAX,
            num_vis: 0,
            state_key,
            child_ids: Vec::new(),
            child_actions: Vec::new(),
            action_mask,
            untried_actions,
        }
    }

    fn add_child(&mut self, action: usize, child_id: usize) {
        if let Some(pos) = self
            .untried_actions
            .iter()
            .position(|candidate| *candidate == action)
        {
            self.untried_actions.remove(pos);
        }
        self.child_ids.push(child_id);
        self.child_actions.push(action);
    }
}

#[derive(Clone, Copy)]
struct MctsTranspositionEntry {
    q: f64,
    reward: f64,
    num_vis: usize,
}

struct MctsCallbackRunner<'py> {
    py: Python<'py>,
    env: &'py Bound<'py, PyAny>,
    np_random: Bound<'py, PyAny>,
    nodes: Vec<MctsCallbackNode>,
    num_bbox: usize,
    num_actions: usize,
    actions_per_bbox: usize,
    exp_weight: f64,
    skip_rate: f64,
    pns: bool,
    grdexp: bool,
    mask_prun: bool,
    gamma: f64,
    exp_action_reward: Vec<f64>,
    exp_action_cnt: Vec<usize>,
    action_prior_weight: f64,
    action_prior_logits: Vec<f64>,
    exp_action: usize,
    opposite_actions: Vec<usize>,
    best_reward: f64,
    not_updated: usize,
    iterations_run: usize,
    fused_rollout_steps: usize,
    use_rust_action_selection: bool,
    skip_summary_metrics: bool,
    stateful_unscored_apply: bool,
    use_fused_rollout_step: bool,
    use_transposition_table: bool,
    transposition_table_size: usize,
    transposition_hits: usize,
    transposition_table: HashMap<String, MctsTranspositionEntry>,
    transposition_order: VecDeque<String>,
}

impl<'py> MctsCallbackRunner<'py> {
    fn new(
        py: Python<'py>,
        args: &'py Bound<'py, PyAny>,
        env: &'py Bound<'py, PyAny>,
    ) -> PyResult<Self> {
        let num_bbox = env.getattr("num_bbox")?.extract::<usize>()?;
        let num_action_scale = env.getattr("num_action_scale")?.extract::<usize>()?;
        let num_actions = action_count(num_bbox, num_action_scale);
        let actions_per_bbox = if num_bbox == 0 {
            0
        } else {
            num_actions / num_bbox
        };
        let use_rust_action_selection = num_actions >= RUST_ACTION_SELECTION_MIN_ACTIONS;
        let np_random = PyModule::import_bound(py, "numpy")?.getattr("random")?;
        let root_mask = vec![false; num_actions];
        let nodes = vec![MctsCallbackNode::new(root_mask, None)];
        let opposite_actions = build_opposite_actions_vec(num_bbox, num_action_scale)?;
        let skip_summary_metrics = get_attr_bool(args, "skip_summary_metrics", false)?;
        let action_prior_weight = get_attr_f64(args, "action_prior_weight", 0.0)?;
        let action_prior_logits =
            load_action_prior_logits(py, args, num_actions, num_action_scale, action_prior_weight)?;
        Ok(Self {
            py,
            env,
            np_random,
            nodes,
            num_bbox,
            num_actions,
            actions_per_bbox,
            exp_weight: get_attr_f64(args, "exp_w", 1.0)?,
            skip_rate: get_attr_f64(args, "skip_rate", 0.7)?,
            pns: get_attr_bool(args, "pns", false)?,
            grdexp: get_attr_bool(args, "grdexp", false)?,
            mask_prun: get_attr_bool(args, "mask_prun", false)?,
            gamma: get_attr_f64(args, "gamma", 1.0)?,
            exp_action_reward: vec![0.0; num_actions],
            exp_action_cnt: vec![0; num_actions],
            action_prior_weight,
            action_prior_logits,
            exp_action: 0,
            opposite_actions,
            best_reward: 0.0,
            not_updated: 0,
            iterations_run: 0,
            fused_rollout_steps: 0,
            use_rust_action_selection,
            skip_summary_metrics,
            stateful_unscored_apply: get_attr_bool(args, "stateful_unscored_apply", false)?,
            use_fused_rollout_step: get_attr_bool(args, "mcts_fused_rollout_step", false)?,
            use_transposition_table: get_attr_bool(args, "transposition_table", false)?,
            transposition_table_size: get_attr_usize(args, "transposition_table_size", 8192)?,
            transposition_hits: 0,
            transposition_table: HashMap::new(),
            transposition_order: VecDeque::new(),
        })
    }

    fn run(&mut self, num_iter: usize) -> PyResult<()> {
        self.env.call_method0("reset")?;
        if self.use_transposition_table {
            self.nodes[0].state_key = self.env_state_key()?;
        }
        for ith in 0..num_iter {
            let (mut path, mut rewards) = self.select(0)?;
            let (path_out, rewards_out, grd_cnt) = self.simulate(&mut path, &mut rewards)?;
            self.backpropagate(&path_out, &rewards_out);
            self.select_best(&path_out, &rewards_out, ith + 1, grd_cnt)?;

            self.env.call_method0("reset")?;
            self.iterations_run = ith + 1;
            if ith > 100 && self.best_reward < 1e-2 {
                break;
            }
            if self.not_updated > 400 {
                break;
            }
        }
        Ok(())
    }

    fn select(&mut self, root_id: usize) -> PyResult<(Vec<usize>, Vec<f64>)> {
        let mut node_id = root_id;
        let mut path = Vec::new();
        let mut rewards = Vec::new();
        loop {
            path.push(node_id);
            if self.env_done()? {
                return Ok((path, rewards));
            }

            if self.nodes[node_id].untried_actions.is_empty() {
                let (child_id, action) = self.ucb_select(node_id)?;
                let reward = self.env_step_reward(action)?;
                rewards.push(reward);
                node_id = child_id;
            } else if self.pns && self.rand_f64()? < self.prob_skip_exploration(node_id) {
                let (child_id, action) = self.ucb_select(node_id)?;
                let reward = self.env_step_reward(action)?;
                rewards.push(reward);
                node_id = child_id;
            } else {
                let action = if self.pns {
                    let actions = self.nodes[node_id].untried_actions.clone();
                    let probs = self.exp_prob(&actions, 100.0);
                    self.random_choice(&actions, &probs)?
                } else {
                    let sample_id =
                        self.random_randint(self.nodes[node_id].untried_actions.len())?;
                    self.nodes[node_id].untried_actions[sample_id]
                };

                let reward = self.env_step_reward(action)?;
                rewards.push(reward);

                let parent_mask = if self.mask_prun {
                    Some(self.nodes[node_id].action_mask.clone())
                } else {
                    None
                };
                let child_mask = self.child_action_mask(action, parent_mask.as_deref());
                let child_id = self.nodes.len();
                let mut child_node = MctsCallbackNode::new(child_mask, self.env_state_key()?);
                self.seed_from_transposition(&mut child_node);
                self.nodes.push(child_node);
                self.nodes[node_id].add_child(action, child_id);
                path.push(child_id);
                self.exp_action = action;
                return Ok((path, rewards));
            }
        }
    }

    fn simulate(
        &mut self,
        path: &mut Vec<usize>,
        rewards: &mut Vec<f64>,
    ) -> PyResult<(Vec<usize>, Vec<f64>, usize)> {
        let mut grd_cnt = 0;
        let mut mask_bbox = vec![true; self.num_bbox];
        let mut node_id = *path
            .last()
            .ok_or_else(|| PyValueError::new_err("MCTS path is empty"))?;

        while !self.env_done()? {
            if self.use_rust_action_selection && self.use_fused_rollout_step {
                let fused = self
                    .env
                    .call_method1("_bridge_mcts_greedy_rollout_step", (mask_bbox.clone(),))?;
                if !fused.is_none() {
                    let action_value = fused.get_item(0)?;
                    let expected_reward = fused.get_item(1)?.extract::<f64>()?;
                    let reward = fused.get_item(2)?.extract::<f64>()?;
                    let done = fused.get_item(3)?.extract::<i64>()? != 0;
                    mask_bbox = fused.get_item(4)?.extract::<Vec<bool>>()?;
                    if action_value.is_none() || expected_reward <= 0.0 {
                        break;
                    }
                    let action = action_value.extract::<usize>()?;
                    if !reward.is_finite() || reward <= 0.0 {
                        break;
                    }
                    self.fused_rollout_steps += 1;
                    grd_cnt += 1;
                    rewards.push(reward);

                    if self.grdexp {
                        self.nodes[node_id].untried_actions = vec![action];
                        self.nodes[node_id].action_mask = self.single_untried_action_mask(action);
                        let child_mask = self.child_action_mask(action, None);
                        let child_id = self.nodes.len();
                        let mut child_node =
                            MctsCallbackNode::new(child_mask, self.env_state_key()?);
                        self.seed_from_transposition(&mut child_node);
                        self.nodes.push(child_node);
                        self.nodes[node_id].add_child(action, child_id);
                        path.push(child_id);
                        node_id = child_id;
                    }
                    if done {
                        break;
                    }
                    continue;
                }
            }

            let mut mx_action: Option<usize> = None;
            let mut mx_reward = -f64::MAX;
            let batch_actions_rewards = if self.use_rust_action_selection {
                let batch = self
                    .env
                    .call_method1("_bridge_greedy_samples_for_mask", (mask_bbox.clone(),))?;
                if batch.is_none() {
                    None
                } else {
                    Some((
                        batch.get_item(0)?.extract::<Vec<isize>>()?,
                        batch.get_item(1)?.extract::<Vec<f64>>()?,
                    ))
                }
            } else {
                None
            };
            for idx in 0..self.num_bbox {
                if !mask_bbox[idx] {
                    continue;
                }
                let (action, reward) = if let Some((actions, rewards)) = &batch_actions_rewards {
                    if actions.len() != self.num_bbox || rewards.len() != self.num_bbox {
                        return Err(PyValueError::new_err(
                            "bridge batch greedy result length does not match bbox count",
                        ));
                    }
                    let action = if actions[idx] >= 0 {
                        let action = usize::try_from(actions[idx])
                            .map_err(|_| PyValueError::new_err("bridge action is invalid"))?;
                        if action >= self.num_actions {
                            return Err(PyValueError::new_err(
                                "bridge batch action is out of range",
                            ));
                        }
                        Some(action)
                    } else {
                        None
                    };
                    (action, rewards[idx])
                } else if self.use_rust_action_selection {
                    let (action, reward) = select_greedy_action_from_env(
                        self.env,
                        Some(idx),
                        self.num_actions,
                        self.actions_per_bbox,
                    )?;
                    (Some(action), reward)
                } else {
                    let result = self.env.call_method1("ith_bbox_greedy_sample", (idx,))?;
                    (
                        Some(result.get_item(0)?.extract::<usize>()?),
                        result.get_item(1)?.extract::<f64>()?,
                    )
                };
                if mx_reward < reward {
                    mx_reward = reward;
                    mx_action = action;
                }
                if reward < 0.0 {
                    mask_bbox[idx] = false;
                }
            }

            if mx_reward <= 0.0 {
                break;
            }
            let action = mx_action
                .ok_or_else(|| PyValueError::new_err("MCTS greedy rollout selected no action"))?;
            let reward = self.env_step_scored_reward(action, mx_reward)?;
            if !reward.is_finite() || reward <= 0.0 {
                break;
            }
            let reward_to_store = if rust_isclose(reward, mx_reward, 1e-9, 1e-12) {
                mx_reward
            } else {
                reward
            };
            grd_cnt += 1;
            rewards.push(reward_to_store);

            if self.grdexp {
                self.nodes[node_id].untried_actions = vec![action];
                self.nodes[node_id].action_mask = self.single_untried_action_mask(action);
                let child_mask = self.child_action_mask(action, None);
                let child_id = self.nodes.len();
                let mut child_node = MctsCallbackNode::new(child_mask, self.env_state_key()?);
                self.seed_from_transposition(&mut child_node);
                self.nodes.push(child_node);
                self.nodes[node_id].add_child(action, child_id);
                path.push(child_id);
                node_id = child_id;
            }
        }

        Ok((path.clone(), rewards.clone(), grd_cnt))
    }

    fn backpropagate(&mut self, path: &[usize], rewards: &[f64]) {
        let reward_sum = discounted_reward_slice(rewards, self.gamma);
        if self.pns {
            let cnt = self.exp_action_cnt[self.exp_action];
            let previous = self.exp_action_reward[self.exp_action];
            self.exp_action_reward[self.exp_action] = previous / (cnt + 1) as f64 * cnt as f64
                + (reward_sum - self.best_reward) / (cnt + 1) as f64;
            self.exp_action_cnt[self.exp_action] += 1;
        }
        for node_id in path.iter().rev() {
            let node = &mut self.nodes[*node_id];
            if node.num_vis == 0 {
                node.reward = reward_sum;
            }
            node.num_vis += 1;
            if reward_sum > node.q {
                node.q = reward_sum;
            }
            self.store_transposition(*node_id);
        }
    }

    fn select_best(
        &mut self,
        _path: &[usize],
        rewards: &[f64],
        num_iter: usize,
        _grd_cnt: usize,
    ) -> PyResult<()> {
        let reward_sum = discounted_reward_slice(rewards, self.gamma);
        self.not_updated += 1;
        if reward_sum > self.best_reward {
            self.best_reward = reward_sum;
            self.not_updated = 0;
            if !self.skip_summary_metrics {
                self.env.call_method0("current_state_summary")?;
            }
            self.env.call_method1("render", (num_iter,))?;
        }
        Ok(())
    }

    fn ucb_select(&mut self, node_id: usize) -> PyResult<(usize, usize)> {
        let parent_visits = self.nodes[node_id].num_vis;
        let child_ids = self.nodes[node_id].child_ids.clone();
        let child_actions = self.nodes[node_id].child_actions.clone();
        let mut scores = Vec::with_capacity(child_ids.len());
        if parent_visits == 0 {
            scores.resize(child_ids.len(), f64::INFINITY);
        } else {
            let log_parent = (parent_visits as f64).ln();
            for child_id in &child_ids {
                let child = &self.nodes[*child_id];
                if child.num_vis == 0 {
                    scores.push(f64::INFINITY);
                } else {
                    scores.push(
                        child.q
                            + self.exp_weight * (2.0 * log_parent / child.num_vis as f64).sqrt(),
                    );
                }
            }
        }
        let max_score = scores
            .iter()
            .fold(f64::NEG_INFINITY, |acc, score| acc.max(*score));
        let mut best_positions = Vec::new();
        for (idx, score) in scores.iter().enumerate() {
            if *score == max_score {
                best_positions.push(idx);
            }
        }
        let next_idx = self.random_randint(best_positions.len())?;
        let pos = best_positions[next_idx];
        Ok((child_ids[pos], child_actions[pos]))
    }

    fn prob_skip_exploration(&self, node_id: usize) -> f64 {
        let mut mx_q = 0.0;
        let node = &self.nodes[node_id];
        for child_id in &node.child_ids {
            let child = &self.nodes[*child_id];
            if child.reward > node.reward && child.q > mx_q {
                mx_q = child.q;
            }
        }
        (mx_q / (self.best_reward + 1e-9)).clamp(0.0, self.skip_rate)
    }

    fn exp_prob(&self, actions: &[usize], scale: f64) -> Vec<f64> {
        if actions.is_empty() {
            return Vec::new();
        }
        let mut values = Vec::with_capacity(actions.len());
        for action in actions {
            values.push(
                self.exp_action_reward[*action] * scale
                    + self.action_prior_weight * self.action_prior_logits[*action],
            );
        }
        let max_value = values
            .iter()
            .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
        let exps: Vec<f64> = values
            .iter()
            .map(|value| (value - max_value).exp())
            .collect();
        let total: f64 = exps.iter().sum();
        if total == 0.0 {
            vec![1.0 / actions.len() as f64; actions.len()]
        } else {
            exps.iter().map(|value| value / total).collect()
        }
    }

    fn child_action_mask(&self, action: usize, parent_mask: Option<&[bool]>) -> Vec<bool> {
        let mut out = match parent_mask {
            Some(mask) => mask.to_vec(),
            None => vec![false; self.num_actions],
        };
        out[self.opposite_actions[action]] = true;
        out
    }

    fn single_untried_action_mask(&self, action: usize) -> Vec<bool> {
        let mut out = vec![true; self.num_actions];
        out[action] = false;
        out
    }

    fn env_done(&self) -> PyResult<bool> {
        Ok(self.env.getattr("done")?.extract::<i64>()? != 0)
    }

    fn env_step_reward(&self, action: usize) -> PyResult<f64> {
        let cached = self
            .env
            .call_method1("_bridge_apply_cached_action", (action,))?;
        if !cached.is_none() {
            return cached.get_item(0)?.extract::<f64>();
        }
        if self.stateful_unscored_apply {
            let applied = self
                .env
                .call_method1("_bridge_apply_unscored_action", (action,))?;
            if !applied.is_none() {
                return applied.get_item(0)?.extract::<f64>();
            }
        }
        let result = self.env.call_method1("step", (action,))?;
        result.get_item(0)?.extract::<f64>()
    }

    fn env_step_scored_reward(&self, action: usize, expected_reward: f64) -> PyResult<f64> {
        let cached = self
            .env
            .call_method1("_bridge_apply_cached_action", (action,))?;
        if !cached.is_none() {
            return cached.get_item(0)?.extract::<f64>();
        }
        let scored = self
            .env
            .call_method1("_bridge_apply_scored_action", (action, expected_reward))?;
        if !scored.is_none() {
            return scored.get_item(0)?.extract::<f64>();
        }
        self.env_step_reward(action)
    }

    fn env_state_key(&self) -> PyResult<Option<String>> {
        if !self.use_transposition_table {
            return Ok(None);
        }
        match self.env.call_method0("_state_cache_key") {
            Ok(value) => {
                if value.is_none() {
                    Ok(None)
                } else {
                    Ok(Some(value.str()?.to_str()?.to_string()))
                }
            }
            Err(_) => Ok(None),
        }
    }

    fn seed_from_transposition(&mut self, node: &mut MctsCallbackNode) {
        if !self.use_transposition_table {
            return;
        }
        let Some(key) = node.state_key.as_ref() else {
            return;
        };
        let Some(entry) = self.transposition_table.get(key).copied() else {
            return;
        };
        node.q = entry.q;
        node.reward = entry.reward;
        node.num_vis = entry.num_vis;
        self.transposition_hits += 1;
    }

    fn store_transposition(&mut self, node_id: usize) {
        if !self.use_transposition_table || self.transposition_table_size == 0 {
            return;
        }
        let node = &self.nodes[node_id];
        if node.num_vis == 0 {
            return;
        }
        let Some(key) = node.state_key.as_ref() else {
            return;
        };
        self.transposition_table.insert(
            key.clone(),
            MctsTranspositionEntry {
                q: node.q,
                reward: node.reward,
                num_vis: node.num_vis,
            },
        );
        self.transposition_order.push_back(key.clone());
        while self.transposition_table.len() > self.transposition_table_size {
            let Some(old_key) = self.transposition_order.pop_front() else {
                break;
            };
            if self
                .transposition_order
                .iter()
                .all(|queued_key| queued_key != &old_key)
            {
                self.transposition_table.remove(&old_key);
            }
        }
    }

    fn rand_f64(&self) -> PyResult<f64> {
        self.np_random.call_method0("rand")?.extract::<f64>()
    }

    fn random_randint(&self, upper: usize) -> PyResult<usize> {
        if upper == 0 {
            return Err(PyValueError::new_err(
                "randint upper bound must be positive",
            ));
        }
        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("size", 1)?;
        let value = self
            .np_random
            .call_method("randint", (upper,), Some(&kwargs))?;
        value.get_item(0)?.extract::<usize>()
    }

    fn random_choice(&self, actions: &[usize], probs: &[f64]) -> PyResult<usize> {
        if actions.is_empty() {
            return Err(PyValueError::new_err("choice actions must not be empty"));
        }
        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("size", 1)?;
        kwargs.set_item("p", PyList::new_bound(self.py, probs))?;
        let value = self.np_random.call_method(
            "choice",
            (PyList::new_bound(self.py, actions),),
            Some(&kwargs),
        )?;
        value.get_item(0)?.extract::<usize>()
    }
}

#[pyfunction]
fn run_mcts_callbacks(
    py: Python<'_>,
    args: &Bound<'_, PyAny>,
    env: &Bound<'_, PyAny>,
    num_iter: usize,
) -> PyResult<HashMap<String, f64>> {
    let mut runner = MctsCallbackRunner::new(py, args, env)?;
    runner.run(num_iter)?;
    let mut out = HashMap::new();
    out.insert("best_reward".to_string(), runner.best_reward);
    out.insert("iterations_run".to_string(), runner.iterations_run as f64);
    out.insert("node_count".to_string(), runner.nodes.len() as f64);
    out.insert(
        "transposition_hits".to_string(),
        runner.transposition_hits as f64,
    );
    out.insert(
        "transposition_table_size".to_string(),
        runner.transposition_table.len() as f64,
    );
    out.insert(
        "fused_rollout_steps".to_string(),
        runner.fused_rollout_steps as f64,
    );
    if let Ok(value) = env.getattr("_initial_bbox_cache_hits") {
        if let Ok(count) = value.extract::<usize>() {
            out.insert("initial_bbox_cache_hits".to_string(), count as f64);
        }
    }
    if let Ok(value) = env.getattr("_initial_bbox_cache_misses") {
        if let Ok(count) = value.extract::<usize>() {
            out.insert("initial_bbox_cache_misses".to_string(), count as f64);
        }
    }
    Ok(out)
}

#[pyfunction]
fn run_greedy_refine_callbacks(
    args: &Bound<'_, PyAny>,
    env: &Bound<'_, PyAny>,
) -> PyResult<(Vec<f64>, usize)> {
    let print_off = get_attr_bool(args, "print_off", false)?;
    env.call_method0("reset")?;
    let num_bbox = env.getattr("num_bbox")?.extract::<usize>()?;
    let num_action_scale = env.getattr("num_action_scale")?.extract::<usize>()?;
    let num_actions = action_count(num_bbox, num_action_scale);
    let actions_per_bbox = if num_bbox == 0 {
        0
    } else {
        num_actions / num_bbox
    };
    let use_rust_action_selection = num_actions >= RUST_ACTION_SELECTION_MIN_ACTIONS;
    let mut done = false;
    let mut rewards = Vec::new();
    let mut count = 0usize;

    while !done {
        let remaining_steps = remaining_env_steps(env)?;
        if remaining_steps > 0 {
            let segment = env.call_method1("_bridge_axis_refine_segment", (remaining_steps,))?;
            if !segment.is_none() {
                let segment_rewards = segment.get_item(0)?.extract::<Vec<f64>>()?;
                let segment_actions = segment.get_item(1)?.extract::<Vec<usize>>()?;
                done = segment.get_item(2)?.extract::<i64>()? != 0;
                for (action, reward) in segment_actions.iter().zip(segment_rewards.iter()) {
                    if !print_off {
                        println!("{action} {reward}");
                    }
                    rewards.push(*reward);
                    count += 1;
                }
                if done {
                    break;
                }
                if !segment_rewards.is_empty() {
                    continue;
                }
            }
        }

        let (action, candidate_reward) = if use_rust_action_selection {
            select_greedy_action_from_env(env, None, num_actions, actions_per_bbox)?
        } else {
            let greedy = env.call_method1("greedy_sample", (true,))?;
            (
                greedy.get_item(0)?.extract::<usize>()?,
                greedy.get_item(1)?.extract::<f64>()?,
            )
        };
        if candidate_reward <= 0.0 {
            break;
        }

        let scored_step =
            env.call_method1("_bridge_apply_scored_action", (action, candidate_reward))?;
        let reward = if scored_step.is_none() {
            let cached_step = env.call_method1("_bridge_apply_cached_action", (action,))?;
            if !cached_step.is_none() {
                done = cached_step.get_item(1)?.extract::<i64>()? != 0;
                cached_step.get_item(0)?.extract::<f64>()?
            } else {
                let step = env.call_method1("step", (action,))?;
                done = step.get_item(2)?.extract::<i64>()? != 0;
                step.get_item(0)?.extract::<f64>()?
            }
        } else {
            done = scored_step.get_item(1)?.extract::<i64>()? != 0;
            scored_step.get_item(0)?.extract::<f64>()?
        };
        if !print_off {
            println!("{action} {reward}");
        }
        rewards.push(reward);
        count += 1;
    }

    Ok((rewards, count))
}

fn remaining_env_steps(env: &Bound<'_, PyAny>) -> PyResult<usize> {
    let max_step = env.getattr("max_step")?.extract::<isize>()?;
    let step_cnt = env.getattr("step_cnt")?.extract::<isize>()?;
    let remaining = max_step - 1 - step_cnt;
    if remaining <= 0 {
        Ok(0)
    } else {
        Ok(remaining as usize)
    }
}

fn select_greedy_action_from_env(
    env: &Bound<'_, PyAny>,
    bbox_idx: Option<usize>,
    num_actions: usize,
    actions_per_bbox: usize,
) -> PyResult<(usize, f64)> {
    let bbox_arg = bbox_idx.map_or(-1_i64, |value| value as i64);
    let bridge_result = env.call_method1("_bridge_greedy_sample", (bbox_arg,))?;
    if !bridge_result.is_none() {
        let action = bridge_result.get_item(0)?.extract::<isize>()?;
        let reward = bridge_result.get_item(1)?.extract::<f64>()?;
        if action >= 0 {
            let action = usize::try_from(action)
                .map_err(|_| PyValueError::new_err("bridge action is out of range"))?;
            if action >= num_actions {
                return Err(PyValueError::new_err("bridge action is out of range"));
            }
            return Ok((action, reward));
        }
    }

    let upper_rewards = match bbox_idx {
        Some(idx) => env.call_method1("_action_upper_rewards", (idx,))?,
        None => env.call_method0("_action_upper_rewards")?,
    };
    let upper_rewards = upper_rewards.extract::<Vec<f64>>()?;

    let mut candidates = Vec::with_capacity(upper_rewards.len());
    for (local_idx, upper_reward) in upper_rewards.into_iter().enumerate() {
        let action = match bbox_idx {
            Some(idx) => idx * actions_per_bbox + local_idx,
            None => local_idx,
        };
        if action >= num_actions {
            return Err(PyValueError::new_err(
                "greedy action index exceeded environment action count",
            ));
        }
        candidates.push((action, upper_reward));
    }

    let mut best_action: Option<usize> = None;
    let mut best_reward = -f64::MAX;
    for (action, upper_reward) in candidates {
        if upper_reward <= best_reward {
            continue;
        }
        let reward = env_step_trial_reward(env, action)?;
        if best_reward < reward {
            best_reward = reward;
            best_action = Some(action);
        }
    }

    best_action
        .map(|action| (action, best_reward))
        .ok_or_else(|| PyValueError::new_err("greedy action selection found no action"))
}

fn env_step_trial_reward(env: &Bound<'_, PyAny>, action: usize) -> PyResult<f64> {
    env.call_method1("step", (action, 0))?.extract::<f64>()
}

#[pyclass]
#[derive(Clone)]
struct BBoxState {
    bounds: Vec<Vec<f64>>,
    num_action_scale: usize,
    action_unit: f64,
    volume_sum: f64,
    last_bbox_score: f64,
    action_scales: Vec<f64>,
    bbox_volumes: Vec<f64>,
    total_bbox_volume: f64,
}

#[pymethods]
impl BBoxState {
    #[new]
    fn new(
        bounds: Vec<Vec<f64>>,
        num_action_scale: usize,
        action_unit: f64,
        volume_sum: f64,
        last_bbox_score: f64,
    ) -> PyResult<Self> {
        if volume_sum <= 0.0 {
            return Err(PyValueError::new_err("volume_sum must be positive"));
        }
        let action_scales = build_action_scales(num_action_scale)?;
        let (bbox_volumes, total_bbox_volume) = summarize_bbox_volumes(&bounds)?;
        Ok(Self {
            bounds,
            num_action_scale,
            action_unit,
            volume_sum,
            last_bbox_score,
            action_scales,
            bbox_volumes,
            total_bbox_volume,
        })
    }

    fn num_bbox(&self) -> usize {
        self.bounds.len()
    }

    fn num_actions(&self) -> usize {
        action_count(self.bounds.len(), self.num_action_scale)
    }

    fn bounds(&self) -> Vec<Vec<f64>> {
        self.bounds.clone()
    }

    fn volumes(&self) -> Vec<f64> {
        self.bbox_volumes.clone()
    }

    fn total_volume(&self) -> f64 {
        self.total_bbox_volume
    }

    fn bvs(&self) -> f64 {
        self.total_bbox_volume / self.volume_sum
    }

    fn valid_mask(&self) -> PyResult<Vec<bool>> {
        self.bounds.iter().map(|row| bbox_is_valid(row)).collect()
    }

    fn valid_count(&self) -> PyResult<usize> {
        let mut count = 0;
        for row in &self.bounds {
            if bbox_is_valid(row)? {
                count += 1;
            }
        }
        Ok(count)
    }

    fn last_bbox_score(&self) -> f64 {
        self.last_bbox_score
    }

    fn set_last_bbox_score(&mut self, last_bbox_score: f64) {
        self.last_bbox_score = last_bbox_score;
    }

    fn with_last_bbox_score(&self, last_bbox_score: f64) -> Self {
        let mut out = self.clone();
        out.last_bbox_score = last_bbox_score;
        out
    }

    fn state_key(&self) -> String {
        bbox_state_key(&self.bounds)
    }

    fn action_upper_rewards(&self) -> PyResult<Vec<f64>> {
        bbox_state_action_upper_rewards(
            &self.bounds,
            &self.bbox_volumes,
            self.total_bbox_volume,
            &self.action_scales,
            self.action_unit,
            self.volume_sum,
            self.last_bbox_score,
        )
    }

    fn bbox_action_upper_rewards(&self, bbox_idx: usize) -> PyResult<Vec<f64>> {
        bbox_state_single_bbox_upper_rewards(
            &self.bounds,
            &self.bbox_volumes,
            self.total_bbox_volume,
            bbox_idx,
            &self.action_scales,
            self.action_unit,
            self.volume_sum,
            self.last_bbox_score,
        )
    }

    fn apply_axis_action(&self, action: usize) -> PyResult<Vec<Vec<f64>>> {
        let mut out = self.bounds.clone();
        apply_axis_action_to_bounds(
            &mut out,
            action,
            self.num_action_scale,
            &self.action_scales,
            self.action_unit,
        )?;
        Ok(out)
    }

    fn after_axis_action(&self, action: usize) -> PyResult<Self> {
        let bounds = self.apply_axis_action(action)?;
        Self::new(
            bounds,
            self.num_action_scale,
            self.action_unit,
            self.volume_sum,
            self.last_bbox_score,
        )
    }

    fn apply_axis_action_in_place(&mut self, action: usize) -> PyResult<()> {
        apply_axis_action_to_bounds(
            &mut self.bounds,
            action,
            self.num_action_scale,
            &self.action_scales,
            self.action_unit,
        )?;
        let (bbox_volumes, total_bbox_volume) = summarize_bbox_volumes(&self.bounds)?;
        self.bbox_volumes = bbox_volumes;
        self.total_bbox_volume = total_bbox_volume;
        Ok(())
    }
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BBoxState>()?;
    module.add_class::<TetClippingState>()?;
    module.add_class::<ManifoldBridgeMesh>()?;
    module.add_class::<ManifoldState>()?;
    module.add_class::<CandidateBitsetState>()?;
    module.add_function(wrap_pyfunction!(manifold_bridge_available, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_cube_volume, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_mesh_volume, module)?)?;
    module.add_function(wrap_pyfunction!(
        manifold_axis_box_intersection_volume,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(bbox_volumes, module)?)?;
    module.add_function(wrap_pyfunction!(coverage_mask, module)?)?;
    module.add_function(wrap_pyfunction!(centroid_proxy_axis_rewards, module)?)?;
    module.add_function(wrap_pyfunction!(action_count, module)?)?;
    module.add_function(wrap_pyfunction!(action_scales, module)?)?;
    module.add_function(wrap_pyfunction!(action_indices, module)?)?;
    module.add_function(wrap_pyfunction!(opposite_action, module)?)?;
    module.add_function(wrap_pyfunction!(opposite_actions, module)?)?;
    module.add_function(wrap_pyfunction!(opposite_action_mask, module)?)?;
    module.add_function(wrap_pyfunction!(untried_actions, module)?)?;
    module.add_function(wrap_pyfunction!(single_untried_action_mask, module)?)?;
    module.add_function(wrap_pyfunction!(mcts_child_action_mask, module)?)?;
    module.add_function(wrap_pyfunction!(apply_axis_action, module)?)?;
    module.add_function(wrap_pyfunction!(action_upper_rewards, module)?)?;
    module.add_function(wrap_pyfunction!(bbox_action_upper_rewards, module)?)?;
    module.add_function(wrap_pyfunction!(bbox_valid_mask, module)?)?;
    module.add_function(wrap_pyfunction!(total_bbox_volume, module)?)?;
    module.add_function(wrap_pyfunction!(bbox_union_bounds, module)?)?;
    module.add_function(wrap_pyfunction!(bbox_union_volume, module)?)?;
    module.add_function(wrap_pyfunction!(bbox_rot_state_key, module)?)?;
    module.add_function(wrap_pyfunction!(bavf_scores, module)?)?;
    module.add_function(wrap_pyfunction!(merge_bavf_reward, module)?)?;
    module.add_function(wrap_pyfunction!(softmax_scaled, module)?)?;
    module.add_function(wrap_pyfunction!(ucb_scores, module)?)?;
    module.add_function(wrap_pyfunction!(ucb_best_indices, module)?)?;
    module.add_function(wrap_pyfunction!(incremental_average, module)?)?;
    module.add_function(wrap_pyfunction!(discounted_reward, module)?)?;
    module.add_function(wrap_pyfunction!(symmetric_chamfer, module)?)?;
    module.add_function(wrap_pyfunction!(tetra_volumes, module)?)?;
    module.add_function(wrap_pyfunction!(tetra_centroids, module)?)?;
    module.add_function(wrap_pyfunction!(tetra_surface_faces, module)?)?;
    module.add_function(wrap_pyfunction!(tetra_adjacency, module)?)?;
    module.add_function(wrap_pyfunction!(load_gmsh, module)?)?;
    module.add_function(wrap_pyfunction!(save_gmsh, module)?)?;
    module.add_function(wrap_pyfunction!(partition_summaries, module)?)?;
    module.add_function(wrap_pyfunction!(tet_clipping_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(run_mcts_callbacks, module)?)?;
    module.add_function(wrap_pyfunction!(run_greedy_refine_callbacks, module)?)?;
    Ok(())
}

fn bbox_volume(row: &[f64]) -> PyResult<f64> {
    check_bounds(row)?;
    Ok((row[3] - row[0]).max(0.0) * (row[4] - row[1]).max(0.0) * (row[5] - row[2]).max(0.0))
}

fn finite_or_error(value: f64, message: &str) -> PyResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PyValueError::new_err(message.to_string()))
    }
}

fn parse_volume_method(volume_method: &str) -> PyResult<i32> {
    match volume_method {
        "mesh" => Ok(0),
        "properties" => Ok(1),
        other => Err(PyValueError::new_err(format!(
            "unsupported Manifold volume method: {other}"
        ))),
    }
}

fn flatten_f64_rows(rows: &[Vec<f64>], width: usize, label: &str) -> PyResult<Vec<f64>> {
    let mut flat = Vec::with_capacity(rows.len() * width);
    for row in rows {
        if row.len() != width {
            return Err(PyValueError::new_err(format!(
                "{label} must contain {width}-value rows",
            )));
        }
        flat.extend_from_slice(row);
    }
    Ok(flat)
}

fn unflatten_f64_rows(flat: &[f64], width: usize) -> Vec<Vec<f64>> {
    flat.chunks(width).map(|row| row.to_vec()).collect()
}

fn flatten_bridge_mesh(
    vertices: &[Vec<f64>],
    faces: &[Vec<usize>],
) -> PyResult<(Vec<f32>, Vec<u32>)> {
    check_vertices(vertices)?;
    check_faces(faces, vertices.len())?;
    let mut flat_vertices = Vec::with_capacity(vertices.len() * 3);
    for vertex in vertices {
        flat_vertices.push(vertex[0] as f32);
        flat_vertices.push(vertex[1] as f32);
        flat_vertices.push(vertex[2] as f32);
    }
    let mut flat_faces = Vec::with_capacity(faces.len() * 3);
    for face in faces {
        flat_faces.push(
            u32::try_from(face[0])
                .map_err(|_| PyValueError::new_err("face index does not fit into uint32"))?,
        );
        flat_faces.push(
            u32::try_from(face[1])
                .map_err(|_| PyValueError::new_err("face index does not fit into uint32"))?,
        );
        flat_faces.push(
            u32::try_from(face[2])
                .map_err(|_| PyValueError::new_err("face index does not fit into uint32"))?,
        );
    }
    Ok((flat_vertices, flat_faces))
}

fn flatten_bridge_box_vertices(box_vertices: &[Vec<Vec<f64>>]) -> PyResult<Vec<f32>> {
    let mut flat = Vec::with_capacity(box_vertices.len() * 8 * 3);
    for box_points in box_vertices {
        if box_points.len() != 8 {
            return Err(PyValueError::new_err(
                "each bridge box must contain exactly 8 vertices",
            ));
        }
        for point in box_points {
            if point.len() != 3 {
                return Err(PyValueError::new_err(
                    "bridge box vertices must be [x, y, z] rows",
                ));
            }
            flat.push(point[0] as f32);
            flat.push(point[1] as f32);
            flat.push(point[2] as f32);
        }
    }
    Ok(flat)
}

fn flatten_bridge_oriented_box_vertices(
    bounds: &[Vec<f64>],
    rotations: &[Vec<f64>],
) -> PyResult<Vec<f32>> {
    if bounds.len() != rotations.len() {
        return Err(PyValueError::new_err(
            "bounds and rotations must have the same length",
        ));
    }
    let mut flat = Vec::with_capacity(bounds.len() * 8 * 3);
    for (row, rot) in bounds.iter().zip(rotations.iter()) {
        check_bounds(row)?;
        if rot.len() != 9 {
            return Err(PyValueError::new_err(
                "rotations must be flattened 3x3 row-major matrices",
            ));
        }
        let lengths = [row[3] - row[0], row[4] - row[1], row[5] - row[2]];
        let base = [
            row[0] * rot[0] + row[1] * rot[3] + row[2] * rot[6],
            row[0] * rot[1] + row[1] * rot[4] + row[2] * rot[7],
            row[0] * rot[2] + row[1] * rot[5] + row[2] * rot[8],
        ];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let point = [
                        base[0]
                            + rot[0] * i as f64 * lengths[0]
                            + rot[3] * j as f64 * lengths[1]
                            + rot[6] * k as f64 * lengths[2],
                        base[1]
                            + rot[1] * i as f64 * lengths[0]
                            + rot[4] * j as f64 * lengths[1]
                            + rot[7] * k as f64 * lengths[2],
                        base[2]
                            + rot[2] * i as f64 * lengths[0]
                            + rot[5] * j as f64 * lengths[1]
                            + rot[8] * k as f64 * lengths[2],
                    ];
                    flat.push(point[0] as f32);
                    flat.push(point[1] as f32);
                    flat.push(point[2] as f32);
                }
            }
        }
    }
    Ok(flat)
}

fn summarize_bbox_volumes(bounds: &[Vec<f64>]) -> PyResult<(Vec<f64>, f64)> {
    let mut volumes = Vec::with_capacity(bounds.len());
    let mut total = 0.0;
    for row in bounds {
        let volume = bbox_volume(row)?;
        volumes.push(volume);
        total += volume;
    }
    Ok((volumes, total))
}

fn empty_bitset(num_items: usize) -> Vec<u64> {
    vec![0; (num_items + 63) / 64]
}

fn centroid_mask_bits_cached(
    centroids: &[[f64; 3]],
    bounds: &[f64],
    rotation: &[f64],
) -> PyResult<Vec<u64>> {
    check_bounds(bounds)?;
    if rotation.len() != 9 {
        return Err(PyValueError::new_err(
            "rotation must be a flattened 3x3 row-major matrix",
        ));
    }
    let mut bits = empty_bitset(centroids.len());
    for (idx, point) in centroids.iter().enumerate() {
        let x = point[0] * rotation[0] + point[1] * rotation[1] + point[2] * rotation[2];
        let y = point[0] * rotation[3] + point[1] * rotation[4] + point[2] * rotation[5];
        let z = point[0] * rotation[6] + point[1] * rotation[7] + point[2] * rotation[8];
        if bounds[0] <= x
            && x <= bounds[3]
            && bounds[1] <= y
            && y <= bounds[4]
            && bounds[2] <= z
            && z <= bounds[5]
        {
            bits[idx / 64] |= 1_u64 << (idx % 64);
        }
    }
    Ok(bits)
}

fn or_bitset_in_place(left: &mut [u64], right: &[u64]) {
    for (left_word, right_word) in left.iter_mut().zip(right.iter()) {
        *left_word |= *right_word;
    }
}

fn masked_volume_sum(mask: &[u64], volumes: &[f64]) -> f64 {
    let mut total = 0.0;
    for (word_idx, word) in mask.iter().enumerate() {
        let mut remaining = *word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let volume_idx = word_idx * 64 + bit;
            if volume_idx < volumes.len() {
                total += volumes[volume_idx];
            }
            remaining &= remaining - 1;
        }
    }
    total
}

fn apply_axis_action_to_bounds(
    bounds: &mut [Vec<f64>],
    action: usize,
    num_action_scale: usize,
    action_scales: &[f64],
    action_unit: f64,
) -> PyResult<()> {
    check_action_scale(num_action_scale)?;
    let (bbox_idx, coord_idx, scale_idx) = decode_action(action, num_action_scale);

    if bbox_idx >= bounds.len() {
        return Err(PyValueError::new_err("action bbox index is out of range"));
    }
    for row in bounds.iter() {
        check_bounds(row)?;
    }

    if coord_idx < 6 {
        bounds[bbox_idx][coord_idx] += action_scales[scale_idx] * action_unit;
    }
    Ok(())
}

fn bbox_state_action_upper_rewards(
    bounds: &[Vec<f64>],
    old_volumes: &[f64],
    total_volume: f64,
    action_scales: &[f64],
    action_unit: f64,
    volume_sum: f64,
    last_bbox_score: f64,
) -> PyResult<Vec<f64>> {
    let num_action_scale = action_scales.len();
    let mut out = Vec::with_capacity(action_count(bounds.len(), num_action_scale));
    for (bbox_idx, row) in bounds.iter().enumerate() {
        append_bbox_action_upper_rewards(
            &mut out,
            row,
            old_volumes[bbox_idx],
            total_volume,
            action_scales,
            action_unit,
            volume_sum,
            last_bbox_score,
        )?;
    }
    Ok(out)
}

fn bbox_state_single_bbox_upper_rewards(
    bounds: &[Vec<f64>],
    old_volumes: &[f64],
    total_volume: f64,
    bbox_idx: usize,
    action_scales: &[f64],
    action_unit: f64,
    volume_sum: f64,
    last_bbox_score: f64,
) -> PyResult<Vec<f64>> {
    if bbox_idx >= bounds.len() {
        return Err(PyValueError::new_err("bbox_idx is out of range"));
    }
    let mut out = Vec::with_capacity(6 * action_scales.len() + 1);
    append_bbox_action_upper_rewards(
        &mut out,
        &bounds[bbox_idx],
        old_volumes[bbox_idx],
        total_volume,
        action_scales,
        action_unit,
        volume_sum,
        last_bbox_score,
    )?;
    Ok(out)
}

fn append_bbox_action_upper_rewards(
    out: &mut Vec<f64>,
    row: &[f64],
    old_volume: f64,
    total_volume: f64,
    action_scales: &[f64],
    action_unit: f64,
    volume_sum: f64,
    last_bbox_score: f64,
) -> PyResult<()> {
    for coord_idx in 0..6 {
        for scale in action_scales {
            let mut candidate = row.to_vec();
            candidate[coord_idx] += scale * action_unit;
            let new_volume = if bbox_is_valid(&candidate)? {
                bbox_volume(&candidate)?
            } else {
                0.0
            };
            let new_total = total_volume - old_volume + new_volume;
            let bvs = new_total / volume_sum;
            out.push(-(bvs - 1.0).abs() - last_bbox_score);
        }
    }
    let bvs = total_volume / volume_sum;
    out.push(-(bvs - 1.0).abs() - last_bbox_score);
    Ok(())
}

fn bbox_state_key(bounds: &[Vec<f64>]) -> String {
    let mut out = String::new();
    for (idx, row) in bounds.iter().enumerate() {
        if idx > 0 {
            out.push('|');
        }
        append_float_bits(&mut out, row);
    }
    out
}

fn append_float_bits(out: &mut String, values: &[f64]) {
    for (value_idx, value) in values.iter().enumerate() {
        if value_idx > 0 {
            out.push(',');
        }
        out.push_str(&format!("{:016x}", value.to_bits()));
    }
}

fn check_bounds(row: &[f64]) -> PyResult<()> {
    if row.len() != 6 {
        return Err(PyValueError::new_err(
            "bounds must contain six values: min_x min_y min_z max_x max_y max_z",
        ));
    }
    Ok(())
}

fn check_points(points: &[Vec<f64>]) -> PyResult<()> {
    for point in points {
        if point.len() != 3 {
            return Err(PyValueError::new_err("points must be [x, y, z] rows"));
        }
    }
    Ok(())
}

fn mean_nearest_squared_distance(source: &[Vec<f64>], target: &[Vec<f64>]) -> f64 {
    let mut total = 0.0;
    for point in source {
        let mut best = f64::INFINITY;
        for candidate in target {
            let dx = point[0] - candidate[0];
            let dy = point[1] - candidate[1];
            let dz = point[2] - candidate[2];
            let dist = dx * dx + dy * dy + dz * dz;
            if dist < best {
                best = dist;
            }
        }
        total += best;
    }
    total / source.len() as f64
}

#[derive(Clone)]
struct ConvexInfo {
    planes: Vec<[f64; 4]>,
    aabb_min: [f64; 3],
    aabb_max: [f64; 3],
}

const PLANE_DEDUPE_TOL: f64 = 1e-6;

fn vertices_to_arrays(vertices: &[Vec<f64>]) -> PyResult<Vec<[f64; 3]>> {
    check_vertices(vertices)?;
    Ok(vertices
        .iter()
        .map(|vertex| [vertex[0], vertex[1], vertex[2]])
        .collect())
}

fn voxels_to_arrays(voxels: &[Vec<usize>], vertex_count: usize) -> PyResult<Vec<[usize; 4]>> {
    check_voxels(voxels, vertex_count)?;
    Ok(voxels
        .iter()
        .map(|voxel| [voxel[0], voxel[1], voxel[2], voxel[3]])
        .collect())
}

fn box_vertices_to_arrays(box_vertices: &[Vec<Vec<f64>>]) -> PyResult<Vec<Vec<[f64; 3]>>> {
    let mut out = Vec::with_capacity(box_vertices.len());
    for points in box_vertices {
        out.push(vertices_to_arrays(points)?);
    }
    Ok(out)
}

fn convex_info_from_points(points: &[[f64; 3]]) -> PyResult<ConvexInfo> {
    if points.len() < 4 {
        return Err(PyValueError::new_err(
            "convex polyhedron needs at least 4 points",
        ));
    }
    let planes = convex_hull_planes(points)?;
    if planes.is_empty() {
        return Err(PyValueError::new_err(
            "convex polyhedron has no hull planes",
        ));
    }
    let mut aabb_min = points[0];
    let mut aabb_max = points[0];
    for point in points.iter().skip(1) {
        for axis in 0..3 {
            aabb_min[axis] = aabb_min[axis].min(point[axis]);
            aabb_max[axis] = aabb_max[axis].max(point[axis]);
        }
    }
    Ok(ConvexInfo {
        planes,
        aabb_min,
        aabb_max,
    })
}

fn convex_info_from_oriented_box(bounds: &[f64], rotation: &[f64]) -> PyResult<ConvexInfo> {
    check_bounds(bounds)?;
    if rotation.len() != 9 {
        return Err(PyValueError::new_err(
            "rotation must be a flattened 3x3 row-major matrix",
        ));
    }
    let rot = [
        [rotation[0], rotation[1], rotation[2]],
        [rotation[3], rotation[4], rotation[5]],
        [rotation[6], rotation[7], rotation[8]],
    ];

    let vertices = oriented_box_vertices(bounds, &rot);
    let mut aabb_min = vertices[0];
    let mut aabb_max = vertices[0];
    for point in vertices.iter().skip(1) {
        for axis in 0..3 {
            aabb_min[axis] = aabb_min[axis].min(point[axis]);
            aabb_max[axis] = aabb_max[axis].max(point[axis]);
        }
    }

    let mut planes = Vec::with_capacity(6);
    for axis in 0..3 {
        let normal_max = normalize3(rot[axis])?;
        let normal_min = scale3(normal_max, -1.0);
        push_unique_plane(
            &mut planes,
            [
                normal_max[0],
                normal_max[1],
                normal_max[2],
                -bounds[axis + 3],
            ],
        );
        push_unique_plane(
            &mut planes,
            [normal_min[0], normal_min[1], normal_min[2], bounds[axis]],
        );
    }

    Ok(ConvexInfo {
        planes,
        aabb_min,
        aabb_max,
    })
}

fn oriented_box_vertices(bounds: &[f64], rot: &[[f64; 3]; 3]) -> Vec<[f64; 3]> {
    let xs = [bounds[0], bounds[3]];
    let ys = [bounds[1], bounds[4]];
    let zs = [bounds[2], bounds[5]];
    let mut out = Vec::with_capacity(8);
    for x in xs {
        for y in ys {
            for z in zs {
                out.push(transform_local_point([x, y, z], rot));
            }
        }
    }
    out
}

fn transform_local_point(point: [f64; 3], rot: &[[f64; 3]; 3]) -> [f64; 3] {
    [
        point[0] * rot[0][0] + point[1] * rot[1][0] + point[2] * rot[2][0],
        point[0] * rot[0][1] + point[1] * rot[1][1] + point[2] * rot[2][1],
        point[0] * rot[0][2] + point[1] * rot[1][2] + point[2] * rot[2][2],
    ]
}

fn convex_hull_planes(points: &[[f64; 3]]) -> PyResult<Vec<[f64; 4]>> {
    let mut planes = Vec::new();
    let eps = 1e-9;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            for k in (j + 1)..points.len() {
                let ab = sub3(points[j], points[i]);
                let ac = sub3(points[k], points[i]);
                let mut normal = cross_arr3(ab, ac);
                let norm = norm3(normal);
                if norm <= 1e-12 {
                    continue;
                }
                normal = scale3(normal, 1.0 / norm);
                let mut d = -dot_arr3(normal, points[i]);
                let mut max_dist = f64::NEG_INFINITY;
                let mut min_dist = f64::INFINITY;
                for point in points {
                    let dist = dot_arr3(normal, *point) + d;
                    max_dist = max_dist.max(dist);
                    min_dist = min_dist.min(dist);
                }
                if max_dist <= eps {
                    // normal already points outward; inside is <= 0
                } else if min_dist >= -eps {
                    normal = scale3(normal, -1.0);
                    d = -d;
                } else {
                    continue;
                }
                let plane = [normal[0], normal[1], normal[2], d];
                push_unique_plane(&mut planes, plane);
            }
        }
    }
    Ok(planes)
}

fn push_unique_plane(planes: &mut Vec<[f64; 4]>, candidate: [f64; 4]) {
    if planes.iter().any(|plane| planes_close(*plane, candidate)) {
        return;
    }
    planes.push(candidate);
}

fn planes_close(left: [f64; 4], right: [f64; 4]) -> bool {
    (left[0] - right[0]).abs() <= PLANE_DEDUPE_TOL
        && (left[1] - right[1]).abs() <= PLANE_DEDUPE_TOL
        && (left[2] - right[2]).abs() <= PLANE_DEDUPE_TOL
        && (left[3] - right[3]).abs() <= PLANE_DEDUPE_TOL
}

fn convex_hull_volume(points: &[[f64; 3]]) -> PyResult<f64> {
    if points.len() < 4 {
        return Ok(0.0);
    }
    let planes = convex_hull_planes(points)?;
    convex_volume_from_planes_and_points(&planes, points)
}

fn convex_volume_from_planes(planes: &[[f64; 4]]) -> PyResult<f64> {
    let points = halfspace_vertices(planes, 1e-9)?;
    if points.len() < 4 {
        return Ok(0.0);
    }
    convex_volume_from_planes_and_points(planes, &points)
}

fn convex_volume_from_planes_and_points(planes: &[[f64; 4]], points: &[[f64; 3]]) -> PyResult<f64> {
    let mut total = 0.0;
    for plane in planes {
        let normal = [plane[0], plane[1], plane[2]];
        let mut face_points = Vec::new();
        for point in points {
            let dist = dot_arr3(normal, *point) + plane[3];
            if dist.abs() <= 1e-9 {
                face_points.push(*point);
            }
        }
        dedupe_points(&mut face_points);
        if face_points.len() < 3 {
            continue;
        }
        let center = centroid3(&face_points);
        let mut u = [0.0, 0.0, 0.0];
        for point in &face_points {
            let candidate = sub3(*point, center);
            let length = norm3(candidate);
            if length > 1e-12 {
                u = scale3(candidate, 1.0 / length);
                break;
            }
        }
        if norm3(u) <= 1e-12 {
            continue;
        }
        let v = cross_arr3(normal, u);
        face_points.sort_by(|left, right| {
            let left_vec = sub3(*left, center);
            let right_vec = sub3(*right, center);
            let left_angle = dot_arr3(left_vec, v).atan2(dot_arr3(left_vec, u));
            let right_angle = dot_arr3(right_vec, v).atan2(dot_arr3(right_vec, u));
            left_angle
                .partial_cmp(&right_angle)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for idx in 1..(face_points.len() - 1) {
            let p0 = face_points[0];
            let mut p1 = face_points[idx];
            let mut p2 = face_points[idx + 1];
            let tri_normal = cross_arr3(sub3(p1, p0), sub3(p2, p0));
            if dot_arr3(tri_normal, normal) < 0.0 {
                std::mem::swap(&mut p1, &mut p2);
            }
            total += dot_arr3(p0, cross_arr3(p1, p2)) / 6.0;
        }
    }
    Ok(total.abs())
}

fn halfspace_vertices(planes: &[[f64; 4]], tol: f64) -> PyResult<Vec<[f64; 3]>> {
    let mut points = Vec::new();
    for i in 0..planes.len() {
        for j in (i + 1)..planes.len() {
            for k in (j + 1)..planes.len() {
                let matrix = [
                    [planes[i][0], planes[i][1], planes[i][2]],
                    [planes[j][0], planes[j][1], planes[j][2]],
                    [planes[k][0], planes[k][1], planes[k][2]],
                ];
                let rhs = [-planes[i][3], -planes[j][3], -planes[k][3]];
                let Some(point) = solve_3x3(matrix, rhs) else {
                    continue;
                };
                if planes
                    .iter()
                    .all(|plane| dot_arr3([plane[0], plane[1], plane[2]], point) + plane[3] <= tol)
                {
                    points.push(point);
                }
            }
        }
    }
    dedupe_points(&mut points);
    Ok(points)
}

fn intersection_volume_infos(infos: &[&ConvexInfo]) -> PyResult<f64> {
    if infos.is_empty() || !all_aabb_overlap(infos) {
        return Ok(0.0);
    }
    let mut planes = Vec::new();
    for info in infos {
        for plane in &info.planes {
            push_unique_plane(&mut planes, *plane);
        }
    }
    convex_volume_from_planes(&planes)
}

fn union_volume_indices(
    box_infos: &[ConvexInfo],
    indices: &[usize],
    base_info: Option<&ConvexInfo>,
    single_cache: Option<&HashMap<usize, f64>>,
) -> PyResult<f64> {
    let mut total = 0.0;
    let subset_count = 1usize
        .checked_shl(indices.len() as u32)
        .ok_or_else(|| PyValueError::new_err("too many boxes for inclusion-exclusion"))?;
    for mask in 1..subset_count {
        let mut refs = Vec::new();
        if let Some(base) = base_info {
            refs.push(base);
        }
        let mut bits = 0usize;
        let mut single_idx = 0usize;
        for (pos, idx) in indices.iter().enumerate() {
            if (mask & (1usize << pos)) != 0 {
                bits += 1;
                single_idx = *idx;
                refs.push(&box_infos[*idx]);
            }
        }
        let volume = if bits == 1 {
            if let Some(cache) = single_cache {
                *cache
                    .get(&single_idx)
                    .unwrap_or(&intersection_volume_infos(&refs)?)
            } else {
                intersection_volume_infos(&refs)?
            }
        } else {
            intersection_volume_infos(&refs)?
        };
        if bits % 2 == 1 {
            total += volume;
        } else {
            total -= volume;
        }
    }
    Ok(total.max(0.0))
}

fn aabb_overlap(left: &ConvexInfo, right: &ConvexInfo) -> bool {
    for axis in 0..3 {
        if left.aabb_min[axis] > right.aabb_max[axis] + 1e-12
            || right.aabb_min[axis] > left.aabb_max[axis] + 1e-12
        {
            return false;
        }
    }
    true
}

fn all_aabb_overlap(infos: &[&ConvexInfo]) -> bool {
    if infos.is_empty() {
        return false;
    }
    let mut mins = infos[0].aabb_min;
    let mut maxs = infos[0].aabb_max;
    for info in infos.iter().skip(1) {
        for axis in 0..3 {
            mins[axis] = mins[axis].max(info.aabb_min[axis]);
            maxs[axis] = maxs[axis].min(info.aabb_max[axis]);
        }
    }
    mins[0] <= maxs[0] + 1e-12 && mins[1] <= maxs[1] + 1e-12 && mins[2] <= maxs[2] + 1e-12
}

fn dedupe_points(points: &mut Vec<[f64; 3]>) {
    let mut seen = BTreeSet::new();
    points.retain(|point| {
        seen.insert(format!(
            "{:.10},{:.10},{:.10}",
            clean_key_float(point[0]),
            clean_key_float(point[1]),
            clean_key_float(point[2])
        ))
    });
}

fn dedupe_points_exact(points: &mut Vec<[f64; 3]>) {
    let mut seen = BTreeSet::new();
    points
        .retain(|point| seen.insert((point[0].to_bits(), point[1].to_bits(), point[2].to_bits())));
}

fn clean_key_float(value: f64) -> f64 {
    if value.abs() < 5e-9 {
        0.0
    } else {
        value
    }
}

fn centroid3(points: &[[f64; 3]]) -> [f64; 3] {
    let mut out = [0.0, 0.0, 0.0];
    for point in points {
        out[0] += point[0];
        out[1] += point[1];
        out[2] += point[2];
    }
    let scale = 1.0 / points.len() as f64;
    [out[0] * scale, out[1] * scale, out[2] * scale]
}

fn solve_3x3(matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<[f64; 3]> {
    let det = determinant3(matrix);
    if det.abs() <= 1e-12 {
        return None;
    }
    let mut mx = matrix;
    mx[0][0] = rhs[0];
    mx[1][0] = rhs[1];
    mx[2][0] = rhs[2];
    let mut my = matrix;
    my[0][1] = rhs[0];
    my[1][1] = rhs[1];
    my[2][1] = rhs[2];
    let mut mz = matrix;
    mz[0][2] = rhs[0];
    mz[1][2] = rhs[1];
    mz[2][2] = rhs[2];
    Some([
        determinant3(mx) / det,
        determinant3(my) / det,
        determinant3(mz) / det,
    ])
}

fn determinant3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn sub3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale3(value: [f64; 3], scale: f64) -> [f64; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

fn dot_arr3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross_arr3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm3(value: [f64; 3]) -> f64 {
    dot_arr3(value, value).sqrt()
}

fn normalize3(value: [f64; 3]) -> PyResult<[f64; 3]> {
    let norm = norm3(value);
    if norm <= 1e-12 {
        return Err(PyValueError::new_err("rotation axis has near-zero length"));
    }
    Ok(scale3(value, 1.0 / norm))
}

fn tet_volume(points: &[[f64; 3]; 4]) -> f64 {
    let a = sub3(points[1], points[0]);
    let b = sub3(points[2], points[0]);
    let c = sub3(points[3], points[0]);
    dot_arr3(a, cross_arr3(b, c)).abs() / 6.0
}

fn check_vertices(vertices: &[Vec<f64>]) -> PyResult<()> {
    for vertex in vertices {
        if vertex.len() != 3 {
            return Err(PyValueError::new_err("vertices must be [x, y, z] rows"));
        }
    }
    Ok(())
}

fn check_faces(faces: &[Vec<usize>], vertex_count: usize) -> PyResult<()> {
    for face in faces {
        if face.len() != 3 {
            return Err(PyValueError::new_err("faces must be triangle index rows"));
        }
        for index in face {
            if *index >= vertex_count {
                return Err(PyValueError::new_err("face index is out of range"));
            }
        }
    }
    Ok(())
}

fn check_voxels_shape(voxels: &[Vec<usize>]) -> PyResult<()> {
    for voxel in voxels {
        if voxel.len() != 4 {
            return Err(PyValueError::new_err(
                "voxels must be tetrahedron index rows",
            ));
        }
    }
    Ok(())
}

fn check_voxels(voxels: &[Vec<usize>], vertex_count: usize) -> PyResult<()> {
    check_voxels_shape(voxels)?;
    for voxel in voxels {
        for index in voxel {
            if *index >= vertex_count {
                return Err(PyValueError::new_err("voxel index is out of range"));
            }
        }
    }
    Ok(())
}

fn parse_gmsh(
    text: &str,
    path: &str,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let lines: Vec<&str> = text.lines().collect();
    let nodes_marker = lines.iter().position(|line| *line == "$Nodes");
    let elements_marker = lines.iter().position(|line| *line == "$Elements");
    if nodes_marker.is_none() || elements_marker.is_none() {
        return Err(PyValueError::new_err(format!(
            "Unsupported or invalid Gmsh file: {path}"
        )));
    }

    let nodes_start = nodes_marker.unwrap() + 1;
    let node_count = parse_first_usize(lines.get(nodes_start).copied(), "node count")?;
    let mut node_id_to_index: HashMap<i64, usize> = HashMap::with_capacity(node_count);
    let mut vertices = Vec::with_capacity(node_count);
    for offset in 0..node_count {
        let line = lines.get(nodes_start + 1 + offset).ok_or_else(|| {
            PyValueError::new_err(format!("Unexpected end of nodes in Gmsh file: {path}"))
        })?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(PyValueError::new_err(format!(
                "Invalid node row in Gmsh file: {path}"
            )));
        }
        let node_id = parse_i64(parts[0], "node id")?;
        node_id_to_index.insert(node_id, offset);
        vertices.push(vec![
            parse_f64(parts[1], "node x")?,
            parse_f64(parts[2], "node y")?,
            parse_f64(parts[3], "node z")?,
        ]);
    }

    let elements_start = elements_marker.unwrap() + 1;
    let element_count = parse_first_usize(lines.get(elements_start).copied(), "element count")?;
    let mut faces = Vec::new();
    let mut voxels = Vec::new();
    for offset in 0..element_count {
        let line = lines.get(elements_start + 1 + offset).ok_or_else(|| {
            PyValueError::new_err(format!("Unexpected end of elements in Gmsh file: {path}"))
        })?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(PyValueError::new_err(format!(
                "Invalid element row in Gmsh file: {path}"
            )));
        }
        let element_type = parse_i64(parts[1], "element type")?;
        let tag_count = parse_first_usize(parts.get(2).copied(), "element tag count")?;
        let ids_start = 3 + tag_count;
        if ids_start > parts.len() {
            return Err(PyValueError::new_err(format!(
                "Invalid element tags in Gmsh file: {path}"
            )));
        }
        let mut ids = Vec::with_capacity(parts.len().saturating_sub(ids_start));
        for raw_id in &parts[ids_start..] {
            let node_id = parse_i64(raw_id, "element node id")?;
            let index = node_id_to_index.get(&node_id).ok_or_else(|| {
                PyValueError::new_err(format!("Unknown node id {node_id} in Gmsh file: {path}"))
            })?;
            ids.push(*index);
        }
        if element_type == 2 && ids.len() >= 3 {
            faces.push(ids[..3].to_vec());
        } else if element_type == 4 && ids.len() >= 4 {
            voxels.push(ids[..4].to_vec());
        }
    }
    if faces.is_empty() && !voxels.is_empty() {
        faces = tetra_surface_faces_from_voxels(&voxels)?;
    }
    Ok((vertices, faces, voxels))
}

fn parse_first_usize(value: Option<&str>, label: &str) -> PyResult<usize> {
    let token = value
        .and_then(|line| line.split_whitespace().next())
        .ok_or_else(|| PyValueError::new_err(format!("Missing {label}")))?;
    token
        .parse::<usize>()
        .map_err(|err| PyValueError::new_err(format!("Invalid {label}: {err}")))
}

fn parse_i64(value: &str, label: &str) -> PyResult<i64> {
    value
        .parse::<i64>()
        .map_err(|err| PyValueError::new_err(format!("Invalid {label}: {err}")))
}

fn parse_f64(value: &str, label: &str) -> PyResult<f64> {
    value
        .parse::<f64>()
        .map_err(|err| PyValueError::new_err(format!("Invalid {label}: {err}")))
}

fn dot3(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross3(left: &[f64; 3], right: &[f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn tet_faces(voxel: &[usize]) -> [[usize; 3]; 4] {
    [
        [voxel[0], voxel[1], voxel[2]],
        [voxel[0], voxel[1], voxel[3]],
        [voxel[0], voxel[2], voxel[3]],
        [voxel[1], voxel[2], voxel[3]],
    ]
}

fn sorted_face(face: [usize; 3]) -> [usize; 3] {
    let mut out = face;
    out.sort_unstable();
    out
}

fn tetra_surface_faces_from_voxels(voxels: &[Vec<usize>]) -> PyResult<Vec<Vec<usize>>> {
    check_voxels_shape(voxels)?;
    let mut key_to_index: HashMap<[usize; 3], usize> = HashMap::new();
    let mut ordered_faces: Vec<([usize; 3], Option<[usize; 3]>)> = Vec::new();
    for voxel in voxels {
        for face in tet_faces(voxel) {
            let key = sorted_face(face);
            if let Some(index) = key_to_index.get(&key) {
                ordered_faces[*index].1 = None;
            } else {
                key_to_index.insert(key, ordered_faces.len());
                ordered_faces.push((key, Some(face)));
            }
        }
    }
    Ok(ordered_faces
        .into_iter()
        .filter_map(|(_, face)| face.map(|value| value.to_vec()))
        .collect())
}

fn bbox_is_valid(row: &[f64]) -> PyResult<bool> {
    check_bounds(row)?;
    Ok(bbox_is_valid_raw(row))
}

fn bbox_is_valid_raw(row: &[f64]) -> bool {
    row.len() == 6 && row[0] < row[3] && row[1] < row[4] && row[2] < row[5]
}

fn bbox_total_volume_raw(bounds: &[Vec<f64>]) -> f64 {
    let mut total = 0.0;
    for row in bounds {
        if bbox_is_valid_raw(row) {
            total += (row[3] - row[0]) * (row[4] - row[1]) * (row[5] - row[2]);
        }
    }
    total
}

fn check_bridge_bbox_params(bounds: &[Vec<f64>], rotations: &[Vec<f64>]) -> PyResult<()> {
    if bounds.len() != rotations.len() {
        return Err(PyValueError::new_err(
            "bounds and rotations must have the same length",
        ));
    }
    for row in bounds {
        if row.len() != 6 {
            return Err(PyValueError::new_err("bounds must be 6-value rows"));
        }
    }
    for rotation in rotations {
        if rotation.len() != 9 {
            return Err(PyValueError::new_err(
                "rotations must be flattened 3x3 row-major matrices",
            ));
        }
    }
    Ok(())
}

fn check_action_scale(num_action_scale: usize) -> PyResult<()> {
    if num_action_scale == 0 {
        return Err(PyValueError::new_err("num_action_scale must be positive"));
    }
    Ok(())
}

fn build_action_scales(num_action_scale: usize) -> PyResult<Vec<f64>> {
    check_action_scale(num_action_scale)?;
    if num_action_scale % 2 != 0 {
        return Err(PyValueError::new_err(
            "num_action_scale must be the expanded even legacy value",
        ));
    }
    let half = num_action_scale / 2;
    let mut out = Vec::with_capacity(num_action_scale);
    for exp in (0..half).rev() {
        out.push(-2.0_f64.powi(exp as i32));
    }
    for exp in 0..half {
        out.push(2.0_f64.powi(exp as i32));
    }
    Ok(out)
}

fn build_opposite_actions_vec(num_bbox: usize, num_action_scale: usize) -> PyResult<Vec<usize>> {
    check_action_scale(num_action_scale)?;
    let total = action_count(num_bbox, num_action_scale);
    let mut out = Vec::with_capacity(total);
    for action in 0..total {
        let (bbox_idx, coord_idx, scale_idx) = decode_action(action, num_action_scale);
        if coord_idx == 6 {
            out.push(action);
        } else {
            out.push(encode_action(
                bbox_idx,
                coord_idx,
                num_action_scale - 1 - scale_idx,
                num_action_scale,
            ));
        }
    }
    Ok(out)
}

fn discounted_reward_slice(rewards: &[f64], gamma: f64) -> f64 {
    let mut out = 0.0;
    for reward in rewards.iter().rev() {
        out = out * gamma + reward;
    }
    out
}

fn rust_isclose(left: f64, right: f64, rel_tol: f64, abs_tol: f64) -> bool {
    (left - right).abs() <= abs_tol.max(rel_tol * left.abs().max(right.abs()))
}

fn get_attr_f64(obj: &Bound<'_, PyAny>, name: &str, default: f64) -> PyResult<f64> {
    match obj.getattr(name) {
        Ok(value) => value.extract::<f64>(),
        Err(_) => Ok(default),
    }
}

fn get_attr_bool(obj: &Bound<'_, PyAny>, name: &str, default: bool) -> PyResult<bool> {
    match obj.getattr(name) {
        Ok(value) => value.extract::<bool>(),
        Err(_) => Ok(default),
    }
}

fn get_attr_usize(obj: &Bound<'_, PyAny>, name: &str, default: usize) -> PyResult<usize> {
    match obj.getattr(name) {
        Ok(value) => value.extract::<usize>(),
        Err(_) => Ok(default),
    }
}

fn get_attr_string(obj: &Bound<'_, PyAny>, name: &str, default: &str) -> PyResult<String> {
    match obj.getattr(name) {
        Ok(value) => value.extract::<String>(),
        Err(_) => Ok(default.to_string()),
    }
}

fn load_action_prior_logits(
    py: Python<'_>,
    args: &Bound<'_, PyAny>,
    num_actions: usize,
    num_action_scale: usize,
    action_prior_weight: f64,
) -> PyResult<Vec<f64>> {
    let mut logits = vec![0.0; num_actions];
    if action_prior_weight == 0.0 {
        return Ok(logits);
    }

    let path = get_attr_string(args, "action_prior_path", "")?;
    if path.trim().is_empty() {
        return Ok(logits);
    }
    let payload_text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(_) => return Ok(logits),
    };
    let json = PyModule::import_bound(py, "json")?;
    let payload = json.call_method1("loads", (payload_text,))?;

    let default_logit = payload
        .call_method1("get", ("default_logit", 0.0))?
        .extract::<f64>()
        .unwrap_or(0.0);
    let coord_scale_logits = {
        let value = payload.call_method1("get", ("coord_scale_logits",))?;
        if value.is_none() {
            let fallback = payload.call_method1("get", ("priors",))?;
            if fallback.is_none() {
                None
            } else {
                Some(fallback)
            }
        } else {
            Some(value)
        }
    };
    let action_logits = {
        let value = payload.call_method1("get", ("action_logits",))?;
        if value.is_none() {
            None
        } else {
            Some(value)
        }
    };

    let per_bbox = 6 * num_action_scale + 1;
    for (action, logit) in logits.iter_mut().enumerate() {
        if let Some(action_map) = &action_logits {
            let value = action_map.call_method1("get", (action.to_string(),))?;
            if !value.is_none() {
                *logit = value.extract::<f64>().unwrap_or(default_logit);
                continue;
            }
        }
        let local = action % per_bbox;
        let key = if local == per_bbox - 1 {
            "6:0".to_string()
        } else {
            format!("{}:{}", local / num_action_scale, local % num_action_scale)
        };
        if let Some(coord_map) = &coord_scale_logits {
            let value = coord_map.call_method1("get", (key,))?;
            if !value.is_none() {
                *logit = value.extract::<f64>().unwrap_or(default_logit);
                continue;
            }
        }
        *logit = default_logit;
    }

    Ok(logits)
}

fn decode_action(action: usize, num_action_scale: usize) -> (usize, usize, usize) {
    let per_bbox = 6 * num_action_scale + 1;
    let bbox_idx = action / per_bbox;
    let local_idx = action % per_bbox;
    if local_idx == 6 * num_action_scale {
        (bbox_idx, 6, 0)
    } else {
        (
            bbox_idx,
            local_idx / num_action_scale,
            local_idx % num_action_scale,
        )
    }
}

fn encode_action(
    bbox_idx: usize,
    coord_idx: usize,
    scale_idx: usize,
    num_action_scale: usize,
) -> usize {
    bbox_idx * (6 * num_action_scale + 1) + coord_idx * num_action_scale + scale_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_scales_match_legacy_order() {
        assert_eq!(build_action_scales(2).unwrap(), vec![-1.0, 1.0]);
        assert_eq!(build_action_scales(4).unwrap(), vec![-2.0, -1.0, 1.0, 2.0]);
        assert!(build_action_scales(0).is_err());
        assert!(build_action_scales(3).is_err());
    }

    #[test]
    fn action_encoding_round_trips() {
        let num_action_scale = 4;
        assert_eq!(action_count(3, num_action_scale), 75);
        for bbox_idx in 0..3 {
            for coord_idx in 0..6 {
                for scale_idx in 0..num_action_scale {
                    let action = encode_action(bbox_idx, coord_idx, scale_idx, num_action_scale);
                    assert_eq!(
                        decode_action(action, num_action_scale),
                        (bbox_idx, coord_idx, scale_idx)
                    );
                }
            }
            let recenter = bbox_idx * (6 * num_action_scale + 1) + 6 * num_action_scale;
            assert_eq!(decode_action(recenter, num_action_scale), (bbox_idx, 6, 0));
        }
    }

    #[test]
    fn tetra_volume_unit_simplex() {
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert!((tet_volume(&points) - (1.0 / 6.0)).abs() < 1.0e-12);
    }

    #[test]
    fn tetra_surface_faces_remove_shared_faces() {
        let faces = tetra_surface_faces_from_voxels(&vec![vec![0, 1, 2, 3], vec![0, 1, 2, 4]])
            .unwrap();
        let face_keys: BTreeSet<Vec<usize>> = faces
            .iter()
            .map(|face| {
                let mut key = face.clone();
                key.sort_unstable();
                key
            })
            .collect();
        assert_eq!(faces.len(), 6);
        assert!(!face_keys.contains(&vec![0, 1, 2]));
    }
}
