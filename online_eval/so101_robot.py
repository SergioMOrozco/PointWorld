# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SO101 robot-geometry configuration for the live/online PointWorld evaluation.

See online_eval/tools/inspect_so101_urdf.py and check_mesh_link_matching.py for how
SO101_GRIPPER_LINKS was derived: SO101's gripper-region meshes ('wrist_roll_follower' on
gripper_link, 'moving_jaw' on moving_jaw_so101_v1_link) don't match RobotSampler's default
GRIPPER_KEYWORDS -- that only matters for the (now unused, dataset-side) RobotSampler/FK path;
it's kept here purely as documentation of which two links the SO101 checkpoint was trained on.

teleop.py's TeleopPointCloudSystem already performs forward kinematics (via
lerobot_playground.point_clouds.robot_state.RobotState) and hands back per-link world-frame
point clouds directly -- so the live loop no longer needs its own RobotSampler/URDF-FK path.
What it still needs is per-tick surface normals for those points, which teleop doesn't provide.
RobotLinkNormalEstimator below fills that gap: teleop's per-link points are a FIXED, cached set
of local mesh points (sampled once at init) rigidly transformed each tick, so point index i of a
given link is the same physical point at every tick. That means normals only need to be
estimated once (on whichever tick arrives first), and every subsequent tick's normals can be
obtained by rotating the cached reference normals with the closed-form rigid rotation (Kabsch/
SVD) between the reference tick's points and the current tick's points -- exact, since it's a
true rigid transform of the same point set, not an approximation.
"""

from pathlib import Path

import numpy as np
import open3d as o3d

SO101_URDF_PATH = str(
    Path(__file__).resolve().parent.parent / "assets" / "so101" / "so101_new_calib.urdf"
)
SO101_GRIPPER_LINKS = ["gripper_link", "moving_jaw_so101_v1_link"]


def _estimate_local_normals(points: np.ndarray) -> np.ndarray:
    """One-time Open3D normal estimation for a link's reference points, oriented away from
    that link's own centroid (a reasonable outward heuristic for a single small mesh part)."""
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=20))
    normals = np.asarray(pcd.normals, dtype=np.float64)
    centroid = points.astype(np.float64).mean(axis=0)
    outward = points.astype(np.float64) - centroid[None, :]
    flip = np.sum(normals * outward, axis=1) < 0.0
    normals[flip] *= -1.0
    return normals


def _kabsch_rotation(ref_points: np.ndarray, cur_points: np.ndarray) -> np.ndarray:
    """Closed-form rotation R (3,3) minimizing ||R @ ref_centered - cur_centered||, via SVD.

    Assumes ref_points[i] and cur_points[i] are the same physical point (index-aligned
    correspondence), which holds here because teleop rigidly transforms the same cached local
    points every tick -- so this recovers the exact per-tick FK rotation without teleop needing
    to expose it directly.
    """
    ref_centroid = ref_points.mean(axis=0)
    cur_centroid = cur_points.mean(axis=0)
    ref_centered = ref_points - ref_centroid[None, :]
    cur_centered = cur_points - cur_centroid[None, :]

    H = ref_centered.T @ cur_centered
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T
    return R


class RobotLinkNormalEstimator:
    """Lazily caches reference per-link points/normals on the first tick, then rotates them into
    every subsequent tick's frame via rigid registration (see module docstring)."""

    def __init__(self, link_names: list[str] = SO101_GRIPPER_LINKS):
        self.link_names = list(link_names)
        self._ref_points: dict[str, np.ndarray] = {}
        self._ref_normals: dict[str, np.ndarray] = {}

    def _ensure_reference(self, link_pcds: dict[str, np.ndarray]) -> None:
        if self._ref_points:
            return
        for name in self.link_names:
            pts = np.asarray(link_pcds[name], dtype=np.float64)
            self._ref_points[name] = pts
            self._ref_normals[name] = _estimate_local_normals(pts)

    def normals_for_tick(self, link_pcds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return per-link normals (float32) for one tick's link point clouds."""
        self._ensure_reference(link_pcds)
        result = {}
        for name in self.link_names:
            cur_points = np.asarray(link_pcds[name], dtype=np.float64)
            ref_points = self._ref_points[name]
            if cur_points.shape[0] != ref_points.shape[0]:
                # Point count changed (shouldn't happen given teleop's fixed presampling, but
                # fall back to a fresh local estimate rather than crashing).
                result[name] = _estimate_local_normals(cur_points).astype(np.float32)
                continue
            R = _kabsch_rotation(ref_points, cur_points)
            rotated = (R @ self._ref_normals[name].T).T
            result[name] = rotated.astype(np.float32)
        return result
