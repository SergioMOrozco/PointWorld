# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one live PointWorld inference sample from a buffered window (real RGBD observation at
t=0 + real robot joint trajectory for t=0..10) -- see online_eval/rolling_buffer.py for how that
window is assembled from the live camera/leader-arm stream.

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
from robot_sampler import RobotSampler
from visualization.viser_tools.visualization_utils import CameraObservation, project_depth_to_world

CAMERA_HW = (180, 320)  # (H, W), hard-required by assert_camera_payload_resolution
PRESAMPLE_SEED = 0


@dataclass
class RawCameraFrame:
    rgb: np.ndarray  # (H_raw, W_raw, 3) uint8
    depth: np.ndarray  # (H_raw, W_raw) float32, meters
    intrinsic: np.ndarray  # (3, 3) float32, for the raw (un-resized) image
    extrinsic_world_to_cam: np.ndarray  # (4, 4) float32


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


def backproject_scene(frame: RawCameraFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGBD -> world-frame point cloud + colors + estimated normals.

    Returns (points (Np,3) float32, colors (Np,3) uint8, normals (Np,3) float32).
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

    if points.shape[0] == 0:
        normals = np.zeros((0, 3), dtype=np.float32)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
        )
        cam_to_world = np.linalg.inv(frame.extrinsic_world_to_cam.astype(np.float64))
        camera_position = cam_to_world[:3, 3]
        pcd.orient_normals_towards_camera_location(camera_position)
        normals = np.asarray(pcd.normals, dtype=np.float32)

    return points, colors, normals, rgb, depth, intrinsic


def build_robot_geometry(
    sampler: RobotSampler, joint_trajectory: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FK the robot's gripper geometry across all T=11 steps of a joint trajectory.

    Args:
        sampler: presampled RobotSampler (see online_eval/so101_robot.py).
        joint_trajectory: (T, n_joints) array, URDF joint order (sampler.joint_names),
            t=0 = real current state, t=1..10 = real subsequent leader states.

    Returns (robot_flows (T,Nr,3), robot_colors (T,Nr,3) uint8, robot_normals (T,Nr,3)).
    """
    T = joint_trajectory.shape[0]
    joint_values = {
        name: torch.as_tensor(joint_trajectory[:, i], dtype=torch.float32, device=sampler.device)
        for i, name in enumerate(sampler.joint_names)
    }
    points, colors, normals = sampler.compute_points(joint_values)
    assert points.shape[0] == T
    return (
        points.detach().cpu().numpy().astype(np.float32),
        colors.detach().cpu().numpy().astype(np.uint8),
        normals.detach().cpu().numpy().astype(np.float32),
    )


def build_live_sample(
    *,
    now_frame: RawCameraFrame,
    joint_trajectory: np.ndarray,
    gripper_trajectory: np.ndarray,
    sampler: RobotSampler,
    args,
    device: str,
    sample_key: str,
) -> dict:
    """Assemble one live inference sample, ready to feed to BaseModel.forward.

    Args:
        now_frame: the buffered t=0 camera frame (from N steps ago -- see rolling_buffer.py).
        joint_trajectory: (11, n_joints) real joint trajectory, t=0..10.
        gripper_trajectory: (11,) or (11, 1) real gripper joint trajectory, t=0..10.
        sampler: presampled SO101 RobotSampler.
        args: the args returned by online_eval.model_loading.load_pointworld_model (carries
            grid_size / max_scene_points / max_robot_points / robot_features / scene_features).
        sample_key: arbitrary string, used only for deterministic-seeding inside
            enforce_max_num_points (mirrors sample['__key__'] in the WDS pipeline).

    Returns a dict of batched (leading dim 1) torch tensors on `device`, plus a plain
    list[str] '__domain__', ready to pass directly to BaseModel.forward.
    """
    scene_points, scene_colors, scene_normals, rgb, depth, intrinsic = backproject_scene(now_frame)
    T = joint_trajectory.shape[0]
    Ns = scene_points.shape[0]

    robot_flows, robot_colors, robot_normals = build_robot_geometry(sampler, joint_trajectory)

    gripper_trajectory = np.asarray(gripper_trajectory, dtype=np.float32).reshape(T, 1)

    sample = {
        "scene_flows": np.tile(scene_points[None], (T, 1, 1)),
        "scene_colors": np.tile(scene_colors[None], (T, 1, 1)),
        "scene_normals": np.tile(scene_normals[None], (T, 1, 1)),
        "robot_flows": robot_flows,
        "robot_colors": robot_colors,
        "robot_normals": robot_normals,
        "right_gripper_open": gripper_trajectory,
        "joint_positions": joint_trajectory.astype(np.float32),
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

    # joint_names is only kept by the pipeline's gather() when has_bimanual_robot=True (not our
    # case); attach it directly for visualization's generic-URDF path (never read by
    # BaseModel.forward). joint_positions itself DOES survive gather() unconditionally.
    sample["joint_names"] = list(sampler.joint_names)

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
