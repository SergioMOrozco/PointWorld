# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one live PointWorld inference sample from a buffered window (real RGBD observation at
t=0 + real, already-FK'd robot gripper point clouds for t=0..10, from teleop.py's
TeleopPointCloudSystem) -- see online_eval/rolling_buffer.py for how that window is assembled
from the live camera/leader-arm stream.

Reuses dataset_components/pipeline.py::apply_release_pipeline_to_sample -- an existing,
already-correct single-sample (non-WebDataset) transform pipeline -- for everything downstream
of assembling the raw numpy sample dict. See CLAUDE.md and the online_eval plan notes for the
full reasoning; key points inlined below where they affect this file specifically.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
import torch

from dataset_components.pipeline import apply_release_pipeline_to_sample
from dataset_components.transforms import center_shift, enforce_max_num_points, grid_sample_transform
from online_eval.so101_robot import SO101_GRIPPER_LINKS, RobotLinkNormalEstimator
from visualization.viser_tools.visualization_utils import CameraObservation, project_depth_to_world

CAMERA_HW = (180, 320)  # (H, W), hard-required by assert_camera_payload_resolution
PRESAMPLE_SEED = 0
ROBOT_COLOR_RGB = (255, 0, 255)  # matches robot_sampler.py's own placeholder magenta


@dataclass
class RawCameraFrame:
    rgb: np.ndarray  # (H_raw, W_raw, 3) uint8
    depth: np.ndarray  # (H_raw, W_raw) float32, meters
    intrinsic: np.ndarray  # (3, 3) float32, for the raw (un-resized) image
    extrinsic_world_to_cam: np.ndarray  # (4, 4) float32


def raw_camera_frame_from_datapoint(datapoint: dict) -> RawCameraFrame:
    """Adapt one of teleop.py's TeleopPointCloudSystem.step() ``datapoints`` entries into a
    RawCameraFrame.

    ``datapoint`` keys (see lerobot_playground/point_clouds/camera_stream.py::get_datapoints):
    ``color`` (H,W,3 uint8, BGR -- RealSense stream is configured bgr8), ``depth`` (H,W, raw
    sensor units), ``depth_scale`` (meters per raw unit), ``max_depth`` (meters, truncation),
    ``color_intrinsics`` (a pyrealsense2.intrinsics object on real hardware; a plain (3,3) array
    in --dry_run synthetic data), ``X_WC`` (4x4 world-from-camera, i.e. cam-to-world -- the
    inverse of what RawCameraFrame stores), ``obj_mask`` (optional, zero = excluded).
    """
    rgb = cv2.cvtColor(datapoint["color"], cv2.COLOR_BGR2RGB)

    depth = datapoint["depth"].astype(np.float32) * float(datapoint["depth_scale"])
    obj_mask = datapoint.get("obj_mask")
    if obj_mask is not None:
        depth = np.where(obj_mask != 0, depth, 0.0).astype(np.float32)
    max_depth = datapoint.get("max_depth")
    if max_depth is not None:
        depth = np.where(depth <= max_depth, depth, 0.0).astype(np.float32)

    intr = datapoint["color_intrinsics"]
    if hasattr(intr, "fx"):  # pyrealsense2.intrinsics object
        intrinsic = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=np.float32
        )
    else:  # already a plain (3,3) array (e.g. --dry_run synthetic data)
        intrinsic = np.asarray(intr, dtype=np.float32)

    extrinsic_world_to_cam = np.linalg.inv(np.asarray(datapoint["X_WC"], dtype=np.float64)).astype(
        np.float32
    )

    return RawCameraFrame(
        rgb=rgb, depth=depth, intrinsic=intrinsic, extrinsic_world_to_cam=extrinsic_world_to_cam
    )


def _resize_camera_frame(frame: RawCameraFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize RGB/depth to CAMERA_HW and scale intrinsics to match."""
    h_raw, w_raw = frame.depth.shape[:2]
    h_new, w_new = CAMERA_HW
    sx, sy = w_new / w_raw, h_new / h_raw

    rgb = cv2.resize(frame.rgb, (w_new, h_new), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(frame.depth, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

    intrinsic = frame.intrinsic.copy().astype(np.float32)
    intrinsic[0, 0] *= sx  # fx
    intrinsic[1, 1] *= sy  # fy
    intrinsic[0, 2] *= sx  # cx
    intrinsic[1, 2] *= sy  # cy

    return rgb.astype(np.uint8), depth.astype(np.float32), intrinsic


def backproject_scene_points(frame: RawCameraFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RGBD -> world-frame point cloud + colors (cheap; no normal estimation).

    Returns (points (Np,3) float32, colors (Np,3) uint8, rgb, depth, intrinsic) -- the latter
    three are the resized (CAMERA_HW) camera arrays, reused by build_live_sample for the
    cam0_* fields.
    """
    rgb, depth, intrinsic = _resize_camera_frame(frame)
    camera = CameraObservation(
        name="cam0",
        intrinsic=intrinsic,
        extrinsic_world_to_cam=frame.extrinsic_world_to_cam.astype(np.float32),
        rgb=rgb,
        depth=depth,
    )
    points, colors = project_depth_to_world(
        camera, bounds_min=np.zeros(3), bounds_max=np.zeros(3), filter_bounds=False
    )
    return points, colors, rgb, depth, intrinsic


def estimate_normals(points: np.ndarray, camera_position: np.ndarray) -> np.ndarray:
    """Open3D normal estimation -- deliberately called AFTER voxel-downsampling in
    build_live_sample (not on the raw, un-downsampled backprojected cloud), since this is the
    dominant per-update cost: estimating normals on the full ~180x320 backprojected cloud
    (tens of thousands of points) before the pipeline's own voxel-downsample (to
    args.max_scene_points, e.g. 12000, at a coarser 1.5cm grid_size) ever runs was making each
    live update take multiple seconds, which is what caused the viewer to visibly "reload"
    continuously even with an inference/visualization-rate throttle (see live_loop.py's
    viz_hz) -- the throttle can't help if a single update already takes longer than the
    throttle period.
    """
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_position.astype(np.float64))
    return np.asarray(pcd.normals, dtype=np.float32)


def build_robot_geometry_from_link_pcds(
    link_pcds_trajectory: list[dict[str, np.ndarray]],
    robot_normal_estimator: RobotLinkNormalEstimator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the robot's gripper geometry across all T=11 steps from teleop's already-FK'd
    per-link point clouds (see online_eval/so101_robot.py for why normals need special handling).

    Args:
        link_pcds_trajectory: length-T list of {link_name: (Ni,3) world-frame points} dicts,
            t=0 = real current state, t=1..10 = real subsequent teleop ticks.
        robot_normal_estimator: shared across calls so its reference frame stays fixed.

    Returns (robot_flows (T,Nr,3), robot_colors (T,Nr,3) uint8, robot_normals (T,Nr,3)).
    """
    T = len(link_pcds_trajectory)
    per_tick_points = []
    per_tick_normals = []
    for link_pcds in link_pcds_trajectory:
        points = np.concatenate(
            [np.asarray(link_pcds[name], dtype=np.float32) for name in SO101_GRIPPER_LINKS], axis=0
        )
        per_tick_points.append(points)

        normals_by_link = robot_normal_estimator.normals_for_tick(link_pcds)
        normals = np.concatenate([normals_by_link[name] for name in SO101_GRIPPER_LINKS], axis=0)
        per_tick_normals.append(normals)

    robot_flows = np.stack(per_tick_points, axis=0)
    robot_normals = np.stack(per_tick_normals, axis=0)
    Nr = robot_flows.shape[1]
    robot_colors = np.tile(np.array(ROBOT_COLOR_RGB, dtype=np.uint8), (T, Nr, 1))
    return robot_flows, robot_colors, robot_normals


def build_live_sample(
    *,
    now_frame: RawCameraFrame,
    link_pcds_trajectory: list[dict[str, np.ndarray]],
    gripper_trajectory: np.ndarray,
    robot_normal_estimator: RobotLinkNormalEstimator,
    args,
    device: str,
    sample_key: str,
) -> dict:
    """Assemble one live inference sample, ready to feed to BaseModel.forward.

    Args:
        now_frame: the buffered t=0 camera frame (from N steps ago -- see rolling_buffer.py).
        link_pcds_trajectory: length-11 list of teleop's per-link world-frame point clouds,
            t=0..10 (t=0 = the delayed "now", t=1..10 = real subsequent teleop ticks).
        gripper_trajectory: (11,) or (11, 1) real gripper trajectory, t=0..10.
        robot_normal_estimator: see online_eval/so101_robot.py::RobotLinkNormalEstimator.
        args: the args returned by online_eval.model_loading.load_pointworld_model (carries
            grid_size / max_scene_points / max_robot_points / robot_features / scene_features).
        sample_key: arbitrary string, used only for deterministic-seeding inside
            enforce_max_num_points (mirrors sample['__key__'] in the WDS pipeline).

    Returns a dict of batched (leading dim 1) torch tensors on `device`, plus a plain
    list[str] '__domain__', ready to pass directly to BaseModel.forward.
    """
    scene_points, scene_colors, rgb, depth, intrinsic = backproject_scene_points(now_frame)
    T = len(link_pcds_trajectory)

    robot_flows, robot_colors, robot_normals = build_robot_geometry_from_link_pcds(
        link_pcds_trajectory, robot_normal_estimator
    )

    gripper_trajectory = np.asarray(gripper_trajectory, dtype=np.float32).reshape(T, 1)

    # scene_normals is deliberately NOT included yet -- computed below, AFTER downsampling (see
    # estimate_normals()'s docstring for why). grid_sample_transform/enforce_max_num_points only
    # downsample keys that already exist and match shape, so omitting scene_normals here is safe.
    sample = {
        "scene_flows": np.tile(scene_points[None], (T, 1, 1)),
        "scene_colors": np.tile(scene_colors[None], (T, 1, 1)),
        "robot_flows": robot_flows,
        "robot_colors": robot_colors,
        "robot_normals": robot_normals,
        "right_gripper_open": gripper_trajectory,
        "cam0_initial_rgb": rgb,
        "cam0_initial_depth": depth,
        "cam0_intrinsic": intrinsic,
        "cam0_extrinsic": now_frame.extrinsic_world_to_cam.astype(np.float32),
        "__key__": sample_key,
        "__domain__": "so101",
    }

    # Downsample before calling the release pipeline: skip_scene_sampling=True (below) means it
    # will NOT re-run grid_sample_transform/enforce_max_num_points itself (see
    # dataset_components/pipeline.py:238-243,307-317, "Teleop uses pre-filtered scenes").
    sample = grid_sample_transform(sample, grid_size=args.grid_size, mode="test")
    sample = enforce_max_num_points(
        sample, max_scene_points=args.max_scene_points, deterministic=True, seed=PRESAMPLE_SEED
    )
    sample = center_shift(sample)

    # Now estimate normals on the (much smaller) downsampled+centered scene cloud. Normals are
    # direction vectors, so translation (center_shift) doesn't affect them; use the
    # already-shifted cam0_extrinsic (center_shift updates it in-place) for a consistent
    # camera position.
    cam_to_world = np.linalg.inv(sample["cam0_extrinsic"].astype(np.float64))
    camera_position = cam_to_world[:3, 3]
    downsampled_scene_points = sample["scene_flows"][0]
    scene_normals = estimate_normals(downsampled_scene_points, camera_position)
    sample["scene_normals"] = np.tile(scene_normals[None], (T, 1, 1))

    sample = apply_release_pipeline_to_sample(
        sample,
        domain="so101",
        mode="test",
        args=args,
        has_bimanual_robot=False,
        rank=0,
        include_scene_data=True,
        skip_scene_sampling=True,
    )

    batched = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batched[key] = value.unsqueeze(0).to(device)
        else:
            batched[key] = value
    batched["__domain__"] = ["so101"]

    T_final = batched["scene_flows"].shape[1]
    Ns_final = batched["scene_flows"].shape[2]
    Nr_final = batched["robot_flows"].shape[2]
    batched["scene_exists"] = torch.ones(1, T_final, Ns_final, dtype=torch.bool, device=device)
    batched["robot_exists"] = torch.ones(1, T_final, Nr_final, dtype=torch.bool, device=device)

    return batched
